#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 5+: TensorFlow ↔ PyTorch differential test analysis + sample extraction

Features:
- Read JSON test result files under tf_pt_log_1/
- Compute per-operator consistent/inconsistent/error distribution
- Generate reports (TXT + CSV)
- Generate 5 sample JSON categories (keep full iteration info):
    1) success and consistent
    2) success but inconsistent
    3) tf_error only
    4) pytorch_error only
    5) both_error

Notes:
- This script does NOT generate analysis_report_*.json summary files.

Usage:
        conda activate tf_env
        python tf_pt_test/analyze_results_with_samples.py \
                [--result-dir tf_pt_test/tf_pt_log_1] \
                [--output-dir tf_pt_test/analysis]
"""

import os
import sys
import io
import json
import csv
import argparse
import copy
from typing import Any, Dict, List, Set, Tuple
from datetime import datetime

# Force UTF-8 output on Windows.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def load_all_results(result_dir: str) -> List[Dict[str, Any]]:
    """Load all JSON result files from the result directory."""
    results: List[Dict[str, Any]] = []
    if not os.path.exists(result_dir):
        print(f"❌ Result directory does not exist: {result_dir}")
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
                print(f"⚠️ Skipped non-object JSON: {filename}")
        except Exception as error:
            print(f"⚠️ Failed to load {filename}: {error}")
    return results


def analyze_single_operator(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze test results for a single operator."""
    tf_api = data.get("tf_api", "unknown")
    pytorch_api = data.get("pytorch_api", "")
    iterations = data.get("results", [])
    if not isinstance(iterations, list):
        iterations = []

    analysis: Dict[str, Any] = {
        "tf_api": tf_api,
        "pytorch_api": pytorch_api,
        "total_iterations": len(iterations),
        "consistent_count": 0,
        "inconsistent_count": 0,
        "tf_error_count": 0,
        "pytorch_error_count": 0,
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

        status = exec_result.get("status", "unknown")

        if status == "consistent":
            analysis["consistent_count"] += 1
        elif status == "inconsistent":
            analysis["inconsistent_count"] += 1
        elif status == "tf_error":
            analysis["tf_error_count"] += 1
            tf_error_message = exec_result.get("tf_error")
            if isinstance(tf_error_message, str) and tf_error_message:
                analysis["errors"].append(f"[TF] {tf_error_message[:100]}")
        elif status == "pytorch_error":
            analysis["pytorch_error_count"] += 1
            pytorch_error_message = exec_result.get("pytorch_error")
            if isinstance(pytorch_error_message, str) and pytorch_error_message:
                analysis["errors"].append(f"[PT] {pytorch_error_message[:100]}")
        elif status == "both_error":
            analysis["both_error_count"] += 1
            tf_error_message = exec_result.get("tf_error")
            pt_error_message = exec_result.get("pytorch_error")
            if isinstance(tf_error_message, str) and tf_error_message:
                analysis["errors"].append(f"[TF] {tf_error_message[:100]}")
            if isinstance(pt_error_message, str) and pt_error_message:
                analysis["errors"].append(f"[PT] {pt_error_message[:100]}")
        elif status == "comparison_error":
            analysis["comparison_error_count"] += 1

    if analysis["consistent_count"] > 0 and analysis["inconsistent_count"] == 0:
        analysis["final_status"] = "consistent"
    elif analysis["inconsistent_count"] > 0:
        analysis["final_status"] = "inconsistent"
    elif analysis["tf_error_count"] + analysis["pytorch_error_count"] + analysis["both_error_count"] > 0:
        analysis["final_status"] = "error"
    else:
        analysis["final_status"] = "unknown"

    analysis["errors"] = list(dict.fromkeys(analysis["errors"]))[:5]
    return analysis


def _classify_iteration(execution_result: Dict[str, Any]) -> str:
    """Classify an iteration into one of five categories, or return empty string to skip."""
    status = execution_result.get("status")
    tf_success = execution_result.get("tf_success")
    pytorch_success = execution_result.get("pytorch_success")
    results_match = execution_result.get("results_match")

    if status == "consistent" and tf_success is True and pytorch_success is True and results_match is True:
        return "consistent_success"
    if status == "inconsistent" and tf_success is True and pytorch_success is True and results_match is False:
        return "inconsistent_success"
    if status == "tf_error":
        return "tf_error_only"
    if status == "pytorch_error":
        return "pytorch_error_only"
    if status == "both_error":
        return "both_error"

    # Handle historical data with unstable/missing status.
    if tf_success is True and pytorch_success is True and results_match is False:
        return "inconsistent_success"
    if tf_success is False and pytorch_success is True:
        return "tf_error_only"
    if tf_success is True and pytorch_success is False:
        return "pytorch_error_only"
    if tf_success is False and pytorch_success is False:
        return "both_error"

    return ""


def extract_samples(all_results: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Set[str]]]:
    """Extract five sample categories while keeping full iteration info."""
    categorized_samples: Dict[str, List[Dict[str, Any]]] = {
        "consistent_success": [],
        "inconsistent_success": [],
        "tf_error_only": [],
        "pytorch_error_only": [],
        "both_error": [],
    }
    categorized_apis: Dict[str, Set[str]] = {
        "consistent_success": set(),
        "inconsistent_success": set(),
        "tf_error_only": set(),
        "pytorch_error_only": set(),
        "both_error": set(),
    }

    for data in all_results:
        tf_api = data.get("tf_api", "unknown")
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
    total_pt_errors = sum(item["pytorch_error_count"] for item in all_analyses)

    txt_file = os.path.join(output_dir, f"analysis_report_{timestamp}.txt")
    with open(txt_file, "w", encoding="utf-8") as file_handle:
        file_handle.write("=" * 80 + "\n")
        file_handle.write("TensorFlow ↔ PyTorch differential test analysis report\n")
        file_handle.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_handle.write("=" * 80 + "\n\n")

        file_handle.write("=" * 50 + "\n")
        file_handle.write("📊 Overall summary\n")
        file_handle.write("=" * 50 + "\n")
        file_handle.write(f"Total operators: {total_operators}\n")
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

        file_handle.write(f"Total iterations: {total_iterations}\n")
        file_handle.write(f"  Consistent count: {total_consistent}\n")
        file_handle.write(f"  Inconsistent count: {total_inconsistent}\n")
        file_handle.write(f"  TF error count: {total_tf_errors}\n")
        file_handle.write(f"  PT error count: {total_pt_errors}\n\n")

        file_handle.write("=" * 50 + "\n")
        file_handle.write(f"✅ Consistent operators ({len(consistent_ops)})\n")
        file_handle.write("=" * 50 + "\n")
        for item in sorted(consistent_ops, key=lambda element: element["tf_api"]):
            file_handle.write(
                f"  {item['tf_api']} → {item['pytorch_api']} "
                f"({item['consistent_count']}/{item['total_iterations']} consistent)\n"
            )

        file_handle.write("\n" + "=" * 50 + "\n")
        file_handle.write(f"❌ Inconsistent operators ({len(inconsistent_ops)})\n")
        file_handle.write("=" * 50 + "\n")
        for item in sorted(inconsistent_ops, key=lambda element: element["tf_api"]):
            file_handle.write(f"  {item['tf_api']} → {item['pytorch_api']}\n")
            file_handle.write(
                f"    Consistent: {item['consistent_count']}, Inconsistent: {item['inconsistent_count']}\n"
            )
            for error_text in item["errors"][:3]:
                file_handle.write(f"    ! {error_text}\n")

        file_handle.write("\n" + "=" * 50 + "\n")
        file_handle.write(f"⚠️ Error operators ({len(error_ops)})\n")
        file_handle.write("=" * 50 + "\n")
        for item in sorted(error_ops, key=lambda element: element["tf_api"]):
            file_handle.write(f"  {item['tf_api']} → {item['pytorch_api']}\n")
            file_handle.write(
                f"    TF errors: {item['tf_error_count']}, PT errors: {item['pytorch_error_count']}, "
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
            "pytorch_api",
            "final_status",
            "total_iterations",
            "consistent_count",
            "inconsistent_count",
            "tf_error_count",
            "pytorch_error_count",
            "both_error_count",
            "error_summary",
        ])
        for item in sorted(all_analyses, key=lambda element: element["tf_api"]):
            writer.writerow([
                item["tf_api"],
                item["pytorch_api"],
                item["final_status"],
                item["total_iterations"],
                item["consistent_count"],
                item["inconsistent_count"],
                item["tf_error_count"],
                item["pytorch_error_count"],
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
        "tf_error_only": "tf_error_only_samples",
        "pytorch_error_only": "pytorch_error_only_samples",
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
    parser = argparse.ArgumentParser(description="TensorFlow ↔ PyTorch differential test analysis + sample extraction")
    parser.add_argument(
        "--result-dir",
        "-r",
        default=os.path.join(ROOT_DIR, "tf_pt_test", "tf_pt_log_1"),
        help="Test result directory path",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=os.path.join(ROOT_DIR, "tf_pt_test", "analysis"),
        help="Report output directory (TXT/CSV)",
    )
    parser.add_argument(
        "--sample-dir",
        "-s",
        default=os.path.join(ROOT_DIR, "tf_pt_test", "analysis"),
        help="Sample JSON output directory",
    )

    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("TensorFlow ↔ PyTorch differential test analysis + sample extraction")
    print("=" * 80)
    print(f"📁 Result dir: {args.result_dir}")
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
