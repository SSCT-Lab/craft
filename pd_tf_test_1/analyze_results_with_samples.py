#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 5+: Paddle ↔ TensorFlow differential test result analysis + sample extraction script.

Functions:
- Read JSON test result files under pd_tf_log_1/.
- Compute per-operator consistency/inconsistency/error distributions.
- Generate summary reports (TXT + CSV).
- Generate 5 categories of sample JSON (preserving full iteration info):
    1) Execution succeeded and comparison consistent
    2) Execution succeeded but comparison inconsistent
    3) pd_error only
    4) tensorflow_error only
    5) both_error

Notes:
- This script does not generate analysis_report_*.json summary files.

Usage:
        conda activate tf_env
        python pd_tf_test_1/analyze_results_with_samples.py \
                [--result-dir pd_tf_test_1/pd_tf_log_1] \
                [--output-dir pd_tf_test_1/analysis]
"""

import argparse
import copy
import csv
import io
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def load_all_results(result_dir: str) -> List[Dict[str, Any]]:
    """Load all JSON result files under the result directory."""
    results: List[Dict[str, Any]] = []
    if not os.path.exists(result_dir):
        print(f"❌ Result directory not found: {result_dir}")
        return results

    for filename in sorted(os.listdir(result_dir)):
        if not (filename.startswith("llm_enhanced_") and filename.endswith(".json")):
            continue

        filepath = os.path.join(result_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
            if isinstance(data, dict):
                data["_source_file"] = filename
                results.append(data)
            else:
                print(f"⚠️ Skipping non-object JSON: {filename}")
        except Exception as error:  # pylint: disable=broad-except
            print(f"⚠️ Failed to load {filename}: {error}")

    return results


def _get_target_api(data: Dict[str, Any]) -> str:
    """Compatible with legacy field names."""
    return str(data.get("tensorflow_api", ""))


def _normalize_status(execution_result: Dict[str, Any]) -> str:
    """Normalize status to the internal labels used by this script."""
    status = str(execution_result.get("status", ""))
    if status == "paddle_error":
        return "pd_error"
    if status == "tensorflow_error":
        return "target_error"
    return status


def analyze_single_operator(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze test results for a single operator."""
    pd_api = str(data.get("pd_api", "unknown"))
    target_api = _get_target_api(data)
    iterations = data.get("results", [])
    if not isinstance(iterations, list):
        iterations = []

    analysis: Dict[str, Any] = {
        "pd_api": pd_api,
        "target_api": target_api,
        "total_iterations": len(iterations),
        "consistent_count": 0,
        "inconsistent_count": 0,
        "pd_error_count": 0,
        "target_error_count": 0,
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
        elif status == "pd_error":
            analysis["pd_error_count"] += 1
            pd_error_message = exec_result.get("pd_error")
            if isinstance(pd_error_message, str) and pd_error_message:
                analysis["errors"].append(f"[PD] {pd_error_message[:100]}")
        elif status == "target_error":
            analysis["target_error_count"] += 1
            target_error_message = exec_result.get("tensorflow_error")
            if isinstance(target_error_message, str) and target_error_message:
                analysis["errors"].append(f"[TF] {target_error_message[:100]}")
        elif status == "both_error":
            analysis["both_error_count"] += 1
            pd_error_message = exec_result.get("pd_error")
            target_error_message = exec_result.get("tensorflow_error")
            if isinstance(pd_error_message, str) and pd_error_message:
                analysis["errors"].append(f"[PD] {pd_error_message[:100]}")
            if isinstance(target_error_message, str) and target_error_message:
                analysis["errors"].append(f"[TF] {target_error_message[:100]}")
        elif status == "comparison_error":
            analysis["comparison_error_count"] += 1

    if analysis["consistent_count"] > 0 and analysis["inconsistent_count"] == 0:
        analysis["final_status"] = "consistent"
    elif analysis["inconsistent_count"] > 0:
        analysis["final_status"] = "inconsistent"
    elif analysis["pd_error_count"] + analysis["target_error_count"] + analysis["both_error_count"] > 0:
        analysis["final_status"] = "error"
    else:
        analysis["final_status"] = "unknown"

    analysis["errors"] = list(dict.fromkeys(analysis["errors"]))[:5]
    return analysis


def _classify_iteration(execution_result: Dict[str, Any]) -> str:
    """Classify an iteration into one of five categories, or return empty string to skip sampling."""
    status = _normalize_status(execution_result)
    pd_success = execution_result.get("pd_success")
    target_success = execution_result.get("tensorflow_success")
    results_match = execution_result.get("results_match")

    if status == "consistent" and pd_success is True and target_success is True and results_match is True:
        return "consistent_success"
    if status == "inconsistent" and pd_success is True and target_success is True and results_match is False:
        return "inconsistent_success"
    if status == "pd_error":
        return "pd_error_only"
    if status == "target_error":
        return "target_error_only"
    if status == "both_error":
        return "both_error"

    # Compatible with legacy data where status is unstable or missing
    if pd_success is True and target_success is True and results_match is False:
        return "inconsistent_success"
    if pd_success is False and target_success is True:
        return "pd_error_only"
    if pd_success is True and target_success is False:
        return "target_error_only"
    if pd_success is False and target_success is False:
        return "both_error"

    return ""


def extract_samples(all_results: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Set[str]]]:
    """Extract five categories of samples, preserving each iteration entry."""
    categorized_samples: Dict[str, List[Dict[str, Any]]] = {
        "consistent_success": [],
        "inconsistent_success": [],
        "pd_error_only": [],
        "target_error_only": [],
        "both_error": [],
    }
    categorized_apis: Dict[str, Set[str]] = {
        "consistent_success": set(),
        "inconsistent_success": set(),
        "pd_error_only": set(),
        "target_error_only": set(),
        "both_error": set(),
    }

    for data in all_results:
        pd_api = str(data.get("pd_api", "unknown"))
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
            categorized_apis[category].add(pd_api)

    return categorized_samples, categorized_apis


def generate_reports(all_analyses: List[Dict[str, Any]], output_dir: str, timestamp: str) -> Tuple[str, str]:
    """Generate TXT / CSV summary reports."""
    os.makedirs(output_dir, exist_ok=True)

    total_operators = len(all_analyses)
    consistent_ops = [item for item in all_analyses if item["final_status"] == "consistent"]
    inconsistent_ops = [item for item in all_analyses if item["final_status"] == "inconsistent"]
    error_ops = [item for item in all_analyses if item["final_status"] == "error"]
    unknown_ops = [item for item in all_analyses if item["final_status"] == "unknown"]

    total_iterations = sum(item["total_iterations"] for item in all_analyses)
    total_consistent = sum(item["consistent_count"] for item in all_analyses)
    total_inconsistent = sum(item["inconsistent_count"] for item in all_analyses)
    total_pd_errors = sum(item["pd_error_count"] for item in all_analyses)
    total_target_errors = sum(item["target_error_count"] for item in all_analyses)
    total_both_errors = sum(item["both_error_count"] for item in all_analyses)

    txt_file = os.path.join(output_dir, f"analysis_report_{timestamp}.txt")
    with open(txt_file, "w", encoding="utf-8") as file_handle:
        file_handle.write("=" * 80 + "\n")
        file_handle.write("Paddle ↔ TensorFlow Differential Test Analysis Report\n")
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
        file_handle.write("📦 Sample counts for five categories (by iteration entries)\n")
        file_handle.write("=" * 50 + "\n")
        file_handle.write(f"Total iterations: {total_iterations}\n")
        file_handle.write(f"  Consistent count: {total_consistent}\n")
        file_handle.write(f"  Inconsistent count: {total_inconsistent}\n")
        file_handle.write(f"  PD error count: {total_pd_errors}\n")
        file_handle.write(f"  TF error count: {total_target_errors}\n")
        file_handle.write(f"  Both error count: {total_both_errors}\n\n")

        file_handle.write("=" * 50 + "\n")
        file_handle.write(f"✅ Consistent operators ({len(consistent_ops)})\n")
        file_handle.write("=" * 50 + "\n")
        for item in sorted(consistent_ops, key=lambda element: element["pd_api"]):
            file_handle.write(
                f"  {item['pd_api']} → {item['target_api']} "
                f"({item['consistent_count']}/{item['total_iterations']} consistent)\n"
            )

        file_handle.write("\n" + "=" * 50 + "\n")
        file_handle.write(f"❌ Inconsistent operators ({len(inconsistent_ops)})\n")
        file_handle.write("=" * 50 + "\n")
        for item in sorted(inconsistent_ops, key=lambda element: element["pd_api"]):
            file_handle.write(f"  {item['pd_api']} → {item['target_api']}\n")
            file_handle.write(
                f"    Consistent: {item['consistent_count']}, Inconsistent: {item['inconsistent_count']}\n"
            )
            for error_text in item["errors"][:3]:
                file_handle.write(f"    ! {error_text}\n")

        file_handle.write("\n" + "=" * 50 + "\n")
        file_handle.write(f"⚠️ Error operators ({len(error_ops)})\n")
        file_handle.write("=" * 50 + "\n")
        for item in sorted(error_ops, key=lambda element: element["pd_api"]):
            file_handle.write(f"  {item['pd_api']} → {item['target_api']}\n")
            file_handle.write(
                f"    PD errors: {item['pd_error_count']}, TF errors: {item['target_error_count']}, "
                f"Both errors: {item['both_error_count']}\n"
            )
            for error_text in item["errors"][:3]:
                file_handle.write(f"    ! {error_text}\n")

    print(f"📄 TXT report saved: {txt_file}")

    csv_file = os.path.join(output_dir, f"analysis_report_{timestamp}.csv")
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow([
            "pd_api",
            "tensorflow_api",
            "final_status",
            "total_iterations",
            "consistent_count",
            "inconsistent_count",
            "pd_error_count",
            "tensorflow_error_count",
            "both_error_count",
            "error_summary",
        ])

        for item in sorted(all_analyses, key=lambda element: element["pd_api"]):
            writer.writerow([
                item["pd_api"],
                item["target_api"],
                item["final_status"],
                item["total_iterations"],
                item["consistent_count"],
                item["inconsistent_count"],
                item["pd_error_count"],
                item["target_error_count"],
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
    """Write five sample files, one JSON per category."""
    os.makedirs(sample_dir, exist_ok=True)

    category_meta = {
        "consistent_success": "consistent_success_samples",
        "inconsistent_success": "inconsistent_success_samples",
        "pd_error_only": "pd_error_only_samples",
        "target_error_only": "tensorflow_error_only_samples",
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
    parser = argparse.ArgumentParser(description="Paddle ↔ TensorFlow differential test analysis + sample extraction")
    parser.add_argument(
        "--result-dir",
        "-r",
        default=os.path.join(ROOT_DIR, "pd_tf_test_1", "pd_tf_log_1"),
        help="Test result directory path",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=os.path.join(ROOT_DIR, "pd_tf_test_1", "analysis"),
        help="Summary report output directory (TXT/CSV)",
    )
    parser.add_argument(
        "--sample-dir",
        "-s",
        default=os.path.join(ROOT_DIR, "pd_tf_test_1", "analysis"),
        help="Sample JSON output directory",
    )

    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("Paddle ↔ TensorFlow differential test analysis + sample extraction")
    print("=" * 80)
    print(f"📁 Result dir: {args.result_dir}")
    print(f"📁 Report dir: {args.output_dir}")
    print(f"📁 Sample dir: {args.sample_dir}")

    all_results = load_all_results(args.result_dir)
    if not all_results:
        print("⚠️ No test result files found")
        return

    print(f"\n📋 Loaded test results for {len(all_results)} operators")

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
    print("📊 Quick Summary")
    print("=" * 50)
    print(f"✅ Consistent: {consistent}/{len(all_analyses)}")
    print(f"❌ Inconsistent: {inconsistent}/{len(all_analyses)}")
    print(f"⚠️ Error: {error}/{len(all_analyses)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
