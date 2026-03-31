#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 5+: TensorFlow <-> MindSpore differential test result analysis + sample extraction.

Purpose:
- Read JSON test result files under tf_ms_log_1/.
- When multiple JSON files exist for the same operator, keep only the latest timestamp.
- Summarize consistency/inconsistency/error distribution per operator.
- Generate summary reports (TXT + CSV).
- Generate 5 categories of sample JSON (keep full iteration info):
    1) Execution succeeded and comparison consistent
    2) Execution succeeded but comparison inconsistent
    3) tf_error only
    4) ms_error only
    5) both_error

Notes:
- This script does not generate analysis_report_*.json summary files.

Usage:
        conda activate tf_env
        python tf_ms_test_1/analyze_results_with_samples.py \
                [--result-dir tf_ms_test_1/tf_ms_log_1] \
                [--output-dir tf_ms_test_1/analysis]
"""

import argparse
import copy
import csv
import io
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

# Force UTF-8 output on Windows.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

RESULT_FILE_PATTERN = re.compile(r"^(llm_enhanced_.+?)_(\d{8}_\d{6})\.json$")


def _select_latest_result_files(result_dir: str) -> List[str]:
    """Group by operator prefix and keep only the latest timestamped result file."""
    latest_by_operator: Dict[str, Tuple[datetime, str]] = {}

    for filename in sorted(os.listdir(result_dir)):
        if not (filename.startswith("llm_enhanced_") and filename.endswith(".json")):
            continue

        match = RESULT_FILE_PATTERN.match(filename)
        if not match:
            # If filename does not match the convention, use full filename as the key.
            operator_key = os.path.splitext(filename)[0]
            timestamp = datetime.min
        else:
            operator_key = match.group(1)
            timestamp = datetime.strptime(match.group(2), "%Y%m%d_%H%M%S")

        previous = latest_by_operator.get(operator_key)
        if previous is None or timestamp > previous[0] or (timestamp == previous[0] and filename > previous[1]):
            latest_by_operator[operator_key] = (timestamp, filename)

    selected_files = sorted(item[1] for item in latest_by_operator.values())
    return selected_files


def load_all_results(result_dir: str) -> List[Dict[str, Any]]:
    """Load latest JSON result files de-duplicated by operator."""
    results: List[Dict[str, Any]] = []
    if not os.path.exists(result_dir):
        print(f"❌ Result directory does not exist: {result_dir}")
        return results

    selected_files = _select_latest_result_files(result_dir)
    all_candidate_count = sum(
        1
        for filename in os.listdir(result_dir)
        if filename.startswith("llm_enhanced_") and filename.endswith(".json")
    )
    skipped_count = max(all_candidate_count - len(selected_files), 0)
    if skipped_count > 0:
        print(
            "ℹ️ Detected multiple versions for the same operator; kept only the latest by timestamp, "
            f"skipped {skipped_count} old files"
        )

    for filename in selected_files:
        filepath = os.path.join(result_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
            if isinstance(data, dict):
                data["_source_file"] = filename
                results.append(data)
            else:
                print(f"⚠️ Skipped non-object JSON: {filename}")
        except Exception as error:  # pylint: disable=broad-except
            print(f"⚠️ Failed to load {filename}: {error}")

    return results


def _normalize_status(execution_result: Dict[str, Any]) -> str:
    """Normalize status to the internal identifier used by this script."""
    status = str(execution_result.get("status", ""))
    if status == "mindspore_error":
        return "ms_error"
    return status


def analyze_single_operator(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze test results for a single operator."""
    tf_api = str(data.get("tf_api", "unknown"))
    ms_api = str(data.get("mindspore_api", data.get("ms_api", "")))
    iterations = data.get("results", [])
    if not isinstance(iterations, list):
        iterations = []

    analysis: Dict[str, Any] = {
        "tf_api": tf_api,
        "ms_api": ms_api,
        "total_iterations": len(iterations),
        "consistent_count": 0,
        "inconsistent_count": 0,
        "tf_error_count": 0,
        "ms_error_count": 0,
        "both_error_count": 0,
        "comparison_error_count": 0,
        "final_status": "unknown",
        "errors": [],
    }

    for item in iterations:
        if not isinstance(item, dict):
            continue

        exec_result = item.get("execution_result", {})
        if not isinstance(exec_result, dict):
            continue

        status = _normalize_status(exec_result)

        if status == "consistent":
            analysis["consistent_count"] += 1
        elif status == "inconsistent":
            analysis["inconsistent_count"] += 1
        elif status == "tf_error":
            analysis["tf_error_count"] += 1
            tf_error_message = exec_result.get("tf_error")
            if isinstance(tf_error_message, str) and tf_error_message:
                analysis["errors"].append(f"[TF] {tf_error_message[:100]}")
        elif status == "ms_error":
            analysis["ms_error_count"] += 1
            ms_error_message = exec_result.get("ms_error")
            if isinstance(ms_error_message, str) and ms_error_message:
                analysis["errors"].append(f"[MS] {ms_error_message[:100]}")
        elif status == "both_error":
            analysis["both_error_count"] += 1
            tf_error_message = exec_result.get("tf_error")
            ms_error_message = exec_result.get("ms_error")
            if isinstance(tf_error_message, str) and tf_error_message:
                analysis["errors"].append(f"[TF] {tf_error_message[:100]}")
            if isinstance(ms_error_message, str) and ms_error_message:
                analysis["errors"].append(f"[MS] {ms_error_message[:100]}")
        elif status == "comparison_error":
            analysis["comparison_error_count"] += 1

    if analysis["consistent_count"] > 0 and analysis["inconsistent_count"] == 0:
        analysis["final_status"] = "consistent"
    elif analysis["inconsistent_count"] > 0:
        analysis["final_status"] = "inconsistent"
    elif analysis["tf_error_count"] + analysis["ms_error_count"] + analysis["both_error_count"] > 0:
        analysis["final_status"] = "error"
    else:
        analysis["final_status"] = "unknown"

    analysis["errors"] = list(dict.fromkeys(analysis["errors"]))[:5]
    return analysis


def _classify_iteration(execution_result: Dict[str, Any]) -> str:
    """Classify an iteration into one of five categories; return empty string to skip."""
    status = _normalize_status(execution_result)
    tf_success = execution_result.get("tf_success")
    ms_success = execution_result.get("mindspore_success", execution_result.get("ms_success"))
    results_match = execution_result.get("results_match")

    if status == "consistent" and tf_success is True and ms_success is True and results_match is True:
        return "consistent_success"
    if status == "inconsistent" and tf_success is True and ms_success is True and results_match is False:
        return "inconsistent_success"
    if status == "tf_error":
        return "tf_error_only"
    if status == "ms_error":
        return "ms_error_only"
    if status == "both_error":
        return "both_error"

    # Compatibility for legacy data with unstable or missing status.
    if tf_success is True and ms_success is True and results_match is False:
        return "inconsistent_success"
    if tf_success is False and ms_success is True:
        return "tf_error_only"
    if tf_success is True and ms_success is False:
        return "ms_error_only"
    if tf_success is False and ms_success is False:
        return "both_error"

    return ""


def extract_samples(all_results: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Set[str]]]:
    """Extract five categories of samples while preserving full iteration info."""
    categorized_samples: Dict[str, List[Dict[str, Any]]] = {
        "consistent_success": [],
        "inconsistent_success": [],
        "tf_error_only": [],
        "ms_error_only": [],
        "both_error": [],
    }
    categorized_apis: Dict[str, Set[str]] = {
        "consistent_success": set(),
        "inconsistent_success": set(),
        "tf_error_only": set(),
        "ms_error_only": set(),
        "both_error": set(),
    }

    for data in all_results:
        tf_api = str(data.get("tf_api", "unknown"))
        iterations = data.get("results", [])
        if not isinstance(iterations, list):
            continue

        for item in iterations:
            if not isinstance(item, dict):
                continue
            execution_result = item.get("execution_result", {})
            if not isinstance(execution_result, dict):
                continue

            category = _classify_iteration(execution_result)
            if not category:
                continue

            categorized_samples[category].append(copy.deepcopy(item))
            categorized_apis[category].add(tf_api)

    return categorized_samples, categorized_apis


def generate_reports(all_analyses: List[Dict[str, Any]], output_dir: str, timestamp: str) -> Tuple[str, str]:
    """Generate TXT/CSV summary reports."""
    os.makedirs(output_dir, exist_ok=True)

    total_operators = len(all_analyses)
    consistent_ops = [item for item in all_analyses if item["final_status"] == "consistent"]
    inconsistent_ops = [item for item in all_analyses if item["final_status"] == "inconsistent"]
    error_ops = [item for item in all_analyses if item["final_status"] == "error"]
    unknown_ops = [item for item in all_analyses if item["final_status"] == "unknown"]

    total_iterations = sum(item["total_iterations"] for item in all_analyses)
    total_consistent = sum(item["consistent_count"] for item in all_analyses)
    total_inconsistent = sum(item["inconsistent_count"] for item in all_analyses)
    total_tf_errors = sum(item["tf_error_count"] for item in all_analyses)
    total_ms_errors = sum(item["ms_error_count"] for item in all_analyses)
    total_both_errors = sum(item["both_error_count"] for item in all_analyses)

    txt_file = os.path.join(output_dir, f"analysis_report_{timestamp}.txt")
    with open(txt_file, "w", encoding="utf-8") as file_handle:
        file_handle.write("=" * 80 + "\n")
        file_handle.write("TensorFlow <-> MindSpore Differential Test Analysis Report\n")
        file_handle.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_handle.write("=" * 80 + "\n\n")

        file_handle.write("=" * 50 + "\n")
        file_handle.write("📊 Overall Summary\n")
        file_handle.write("=" * 50 + "\n")
        file_handle.write(f"Total operators tested: {total_operators}\n")
        file_handle.write(
            f"  ✅ Consistent: {len(consistent_ops)} "
            f"({len(consistent_ops) / max(total_operators, 1) * 100:.1f}%)\n"
        )
        file_handle.write(
            f"  ❌ Inconsistent: {len(inconsistent_ops)} "
            f"({len(inconsistent_ops) / max(total_operators, 1) * 100:.1f}%)\n"
        )
        file_handle.write(
            f"  ⚠️ Error: {len(error_ops)} "
            f"({len(error_ops) / max(total_operators, 1) * 100:.1f}%)\n"
        )
        file_handle.write(f"  ❓ Unknown: {len(unknown_ops)}\n\n")

        file_handle.write("=" * 50 + "\n")
        file_handle.write("📦 Sample counts by category (by iteration)\n")
        file_handle.write("=" * 50 + "\n")
        file_handle.write(f"Total iterations: {total_iterations}\n")
        file_handle.write(f"  Consistent: {total_consistent}\n")
        file_handle.write(f"  Inconsistent: {total_inconsistent}\n")
        file_handle.write(f"  TF errors: {total_tf_errors}\n")
        file_handle.write(f"  MS errors: {total_ms_errors}\n")
        file_handle.write(f"  Both errors: {total_both_errors}\n\n")

        file_handle.write("=" * 50 + "\n")
        file_handle.write(f"✅ Consistent operators ({len(consistent_ops)})\n")
        file_handle.write("=" * 50 + "\n")
        for item in sorted(consistent_ops, key=lambda element: element["tf_api"]):
            file_handle.write(
                f"  {item['tf_api']} → {item['ms_api']} "
                f"({item['consistent_count']}/{item['total_iterations']} consistent)\n"
            )

        file_handle.write("\n" + "=" * 50 + "\n")
        file_handle.write(f"❌ Inconsistent operators ({len(inconsistent_ops)})\n")
        file_handle.write("=" * 50 + "\n")
        for item in sorted(inconsistent_ops, key=lambda element: element["tf_api"]):
            file_handle.write(f"  {item['tf_api']} → {item['ms_api']}\n")
            file_handle.write(
                f"    Consistent: {item['consistent_count']}, Inconsistent: {item['inconsistent_count']}\n"
            )
            for error_text in item["errors"][:3]:
                file_handle.write(f"    ! {error_text}\n")

        file_handle.write("\n" + "=" * 50 + "\n")
        file_handle.write(f"⚠️ Error operators ({len(error_ops)})\n")
        file_handle.write("=" * 50 + "\n")
        for item in sorted(error_ops, key=lambda element: element["tf_api"]):
            file_handle.write(f"  {item['tf_api']} → {item['ms_api']}\n")
            file_handle.write(
                f"    TF errors: {item['tf_error_count']}, MS errors: {item['ms_error_count']}, "
                f"Both errors: {item['both_error_count']}\n"
            )
            for error_text in item["errors"][:3]:
                file_handle.write(f"    ! {error_text}\n")

    print(f"📄 TXT report saved: {txt_file}")

    csv_file = os.path.join(output_dir, f"analysis_report_{timestamp}.csv")
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow([
            "tf_api",
            "ms_api",
            "final_status",
            "total_iterations",
            "consistent_count",
            "inconsistent_count",
            "tf_error_count",
            "ms_error_count",
            "both_error_count",
            "error_summary",
        ])
        for item in sorted(all_analyses, key=lambda element: element["tf_api"]):
            writer.writerow([
                item["tf_api"],
                item["ms_api"],
                item["final_status"],
                item["total_iterations"],
                item["consistent_count"],
                item["inconsistent_count"],
                item["tf_error_count"],
                item["ms_error_count"],
                item["both_error_count"],
                "; ".join(item["errors"][:3]) if item["errors"] else "",
            ])

    print(f"📄 CSV report saved: {csv_file}")
    return txt_file, csv_file


def generate_sample_files(
    categorized_samples: Dict[str, List[Dict[str, Any]]],
    categorized_apis: Dict[str, Set[str]],
    sample_dir: str,
    timestamp: str,
) -> List[str]:
    """Output five sample files, one JSON per category."""
    os.makedirs(sample_dir, exist_ok=True)

    category_meta = {
        "consistent_success": "consistent_success_samples",
        "inconsistent_success": "inconsistent_success_samples",
        "tf_error_only": "tf_error_only_samples",
        "ms_error_only": "ms_error_only_samples",
        "both_error": "both_error_samples",
    }

    output_files: List[str] = []
    generated_at = datetime.now().isoformat()

    for category, base_name in category_meta.items():
        file_path = os.path.join(sample_dir, f"{base_name}_{timestamp}.json")
        payload = {
            "generated_at": generated_at,
            "category": category,
            "api_count": len(categorized_apis.get(category, set())),
            "sample_count": len(categorized_samples.get(category, [])),
            "samples": categorized_samples.get(category, []),
        }
        with open(file_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2)

        output_files.append(file_path)
        print(f"📦 Sample file saved: {file_path}")

    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorFlow <-> MindSpore differential analysis + sample extraction")
    parser.add_argument(
        "--result-dir",
        "-r",
        default=os.path.join(ROOT_DIR, "tf_ms_test_1", "tf_ms_log_1"),
        help="Test results directory path",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=os.path.join(ROOT_DIR, "tf_ms_test_1", "analysis"),
        help="Summary report output directory (TXT/CSV)",
    )
    parser.add_argument(
        "--sample-dir",
        "-s",
        default=os.path.join(ROOT_DIR, "tf_ms_test_1", "analysis"),
        help="Sample JSON output directory",
    )

    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("TensorFlow <-> MindSpore differential analysis + sample extraction")
    print("=" * 80)
    print(f"📁 Results dir: {args.result_dir}")
    print(f"📁 Report dir: {args.output_dir}")
    print(f"📁 Sample dir: {args.sample_dir}")

    all_results = load_all_results(args.result_dir)
    if not all_results:
        print("⚠️ No test result files found")
        return

    print(f"\n📋 Loaded results for {len(all_results)} operators")

    all_analyses: List[Dict[str, Any]] = []
    for data in all_results:
        all_analyses.append(analyze_single_operator(data))

    generate_reports(all_analyses, args.output_dir, timestamp)

    categorized_samples, categorized_apis = extract_samples(all_results)
    generate_sample_files(categorized_samples, categorized_apis, args.sample_dir, timestamp)

    consistent = sum(1 for item in all_analyses if item["final_status"] == "consistent")
    inconsistent = sum(1 for item in all_analyses if item["final_status"] == "inconsistent")
    error = sum(1 for item in all_analyses if item["final_status"] == "error")

    print("\n" + "=" * 50)
    print("📊 Quick summary")
    print("=" * 50)
    print(f"✅ Consistent: {consistent}/{len(all_analyses)}")
    print(f"❌ Inconsistent: {inconsistent}/{len(all_analyses)}")
    print(f"⚠️ Error: {error}/{len(all_analyses)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
