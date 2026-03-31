#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 5+: Paddle <-> PyTorch differential testing result analysis + sample extraction

Function:
- Read JSON result files from pd_pt_log_1/
- Count per-operator consistent/inconsistent/error distribution
- Generate summary reports (TXT + CSV)
- Generate 5 categories of sample JSON (preserve full iteration info):
    1) Execute successful and results consistent
    2) Execute successful but results inconsistent
    3) pd_error only
    4) pytorch_error only
    5) both_error

Note:
- This script does not generate analysis_report_*.json summary files.

Usage:
        conda activate tf_env
        python pd_pt_test/analyze_results_with_samples.py \
                [--result-dir pd_pt_test/pd_pt_log_1] \
                [--output-dir pd_pt_test/analysis]
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

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def load_all_results(result_dir: str) -> List[Dict[str, Any]]:
    """Load all JSON result files in the result directory."""
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
            print(f"⚠️ Load {filename} failed: {error}")
    return results


def analyze_single_operator(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the test results of a single operator."""
    pd_api = data.get("pd_api", "unknown")
    pytorch_api = data.get("pytorch_api", "")
    iterations = data.get("results", [])
    if not isinstance(iterations, list):
        iterations = []

    analysis: Dict[str, Any] = {
        "pd_api": pd_api,
        "pytorch_api": pytorch_api,
        "total_iterations": len(iterations),
        "consistent_count": 0,
        "inconsistent_count": 0,
        "pd_error_count": 0,
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
        elif status == "pd_error":
            analysis["pd_error_count"] += 1
            pd_error_message = exec_result.get("pd_error")
            if isinstance(pd_error_message, str) and pd_error_message:
                analysis["errors"].append(f"[PD] {pd_error_message[:100]}")
        elif status == "pytorch_error":
            analysis["pytorch_error_count"] += 1
            pytorch_error_message = exec_result.get("pytorch_error")
            if isinstance(pytorch_error_message, str) and pytorch_error_message:
                analysis["errors"].append(f"[PT] {pytorch_error_message[:100]}")
        elif status == "both_error":
            analysis["both_error_count"] += 1
            pd_error_message = exec_result.get("pd_error")
            pt_error_message = exec_result.get("pytorch_error")
            if isinstance(pd_error_message, str) and pd_error_message:
                analysis["errors"].append(f"[PD] {pd_error_message[:100]}")
            if isinstance(pt_error_message, str) and pt_error_message:
                analysis["errors"].append(f"[PT] {pt_error_message[:100]}")
        elif status == "comparison_error":
            analysis["comparison_error_count"] += 1

    if analysis["consistent_count"] > 0 and analysis["inconsistent_count"] == 0:
        analysis["final_status"] = "consistent"
    elif analysis["inconsistent_count"] > 0:
        analysis["final_status"] = "inconsistent"
    elif analysis["pd_error_count"] + analysis["pytorch_error_count"] + analysis["both_error_count"] > 0:
        analysis["final_status"] = "error"
    else:
        analysis["final_status"] = "unknown"

    analysis["errors"] = list(dict.fromkeys(analysis["errors"]))[:5]
    return analysis


def _classify_iteration(execution_result: Dict[str, Any]) -> str:
    """Classify an iteration into one of five categories, or return empty to skip."""
    status = execution_result.get("status")
    pd_success = execution_result.get("pd_success")
    pytorch_success = execution_result.get("pytorch_success")
    results_match = execution_result.get("results_match")

    if status == "consistent" and pd_success is True and pytorch_success is True and results_match is True:
        return "consistent_success"
    if status == "inconsistent" and pd_success is True and pytorch_success is True and results_match is False:
        return "inconsistent_success"
    if status == "pd_error":
        return "pd_error_only"
    if status == "pytorch_error":
        return "pytorch_error_only"
    if status == "both_error":
        return "both_error"

    # Compatibility for legacy data with unstable/missing status
    if pd_success is True and pytorch_success is True and results_match is False:
        return "inconsistent_success"
    if pd_success is False and pytorch_success is True:
        return "pd_error_only"
    if pd_success is True and pytorch_success is False:
        return "pytorch_error_only"
    if pd_success is False and pytorch_success is False:
        return "both_error"

    return ""


def extract_samples(all_results: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Set[str]]]:
    """Extract five categories of samples, preserving full iteration info."""
    categorized_samples: Dict[str, List[Dict[str, Any]]] = {
        "consistent_success": [],
        "inconsistent_success": [],
        "pd_error_only": [],
        "pytorch_error_only": [],
        "both_error": [],
    }
    categorized_apis: Dict[str, Set[str]] = {
        "consistent_success": set(),
        "inconsistent_success": set(),
        "pd_error_only": set(),
        "pytorch_error_only": set(),
        "both_error": set(),
    }

    for data in all_results:
        pd_api = data.get("pd_api", "unknown")
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
    total_pt_errors = sum(item["pytorch_error_count"] for item in all_analyses)

    txt_file = os.path.join(output_dir, f"analysis_report_{timestamp}.txt")
    with open(txt_file, "w", encoding="utf-8") as file_handle:
        file_handle.write("=" * 80 + "\n")
        file_handle.write("Paddle ↔ PyTorch differential testing analysis report\n")
        file_handle.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_handle.write("=" * 80 + "\n\n")

        file_handle.write("=" * 50 + "\n")
        file_handle.write("📊 Overall statistics\n")
        file_handle.write("=" * 50 + "\n")
        file_handle.write(f"Total operators tested: {total_operators}\n")
        file_handle.write(
            f"  ✅ consistent (consistent): {len(consistent_ops)} "
            f"({len(consistent_ops) / max(total_operators, 1) * 100:.1f}%)\n"
        )
        file_handle.write(
            f"  ❌ inconsistent: {len(inconsistent_ops)} "
            f"({len(inconsistent_ops) / max(total_operators, 1) * 100:.1f}%)\n"
        )
        file_handle.write(
            f"  ⚠️ error (error): {len(error_ops)} "
            f"({len(error_ops) / max(total_operators, 1) * 100:.1f}%)\n"
        )
        file_handle.write(f"  ❓ unknown: {len(unknown_ops)}\n\n")

        file_handle.write(f"Total iterations: {total_iterations}\n")
        file_handle.write(f"  consistent count: {total_consistent}\n")
        file_handle.write(f"  inconsistent count: {total_inconsistent}\n")
        file_handle.write(f"  PD error count: {total_pd_errors}\n")
        file_handle.write(f"  PT error count: {total_pt_errors}\n\n")

        file_handle.write("=" * 50 + "\n")
        file_handle.write(f"✅ consistentoperator ({len(consistent_ops)} items)\n")
        file_handle.write("=" * 50 + "\n")
        for item in sorted(consistent_ops, key=lambda element: element["pd_api"]):
            file_handle.write(
                f"  {item['pd_api']} → {item['pytorch_api']} "
                f"({item['consistent_count']}/{item['total_iterations']} consistent)\n"
            )

        file_handle.write("\n" + "=" * 50 + "\n")
        file_handle.write(f"❌ inconsistent operators ({len(inconsistent_ops)} items)\n")
        file_handle.write("=" * 50 + "\n")
        for item in sorted(inconsistent_ops, key=lambda element: element["pd_api"]):
            file_handle.write(f"  {item['pd_api']} → {item['pytorch_api']}\n")
            file_handle.write(
                f"    consistent: {item['consistent_count']}, inconsistent: {item['inconsistent_count']}\n"
            )
            for error_text in item["errors"][:3]:
                file_handle.write(f"    ! {error_text}\n")

        file_handle.write("\n" + "=" * 50 + "\n")
        file_handle.write(f"⚠️ erroroperator ({len(error_ops)} items)\n")
        file_handle.write("=" * 50 + "\n")
        for item in sorted(error_ops, key=lambda element: element["pd_api"]):
            file_handle.write(f"  {item['pd_api']} → {item['pytorch_api']}\n")
            file_handle.write(
                f"    PDerror: {item['pd_error_count']}, PTerror: {item['pytorch_error_count']}, "
                f"both errors: {item['both_error_count']}\n"
            )
            for error_text in item["errors"][:3]:
                file_handle.write(f"    ! {error_text}\n")

    print(f"📄 TXT report saved: {txt_file}")

    csv_file = os.path.join(output_dir, f"analysis_report_{timestamp}.csv")
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow([
            "pd_api",
            "pytorch_api",
            "final_status",
            "total_iterations",
            "consistent_count",
            "inconsistent_count",
            "pd_error_count",
            "pytorch_error_count",
            "both_error_count",
            "error_summary",
        ])
        for item in sorted(all_analyses, key=lambda element: element["pd_api"]):
            writer.writerow([
                item["pd_api"],
                item["pytorch_api"],
                item["final_status"],
                item["total_iterations"],
                item["consistent_count"],
                item["inconsistent_count"],
                item["pd_error_count"],
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
    """Output five sample files, one JSON per category."""
    os.makedirs(sample_dir, exist_ok=True)

    category_meta = {
        "consistent_success": "consistent_success_samples",
        "inconsistent_success": "inconsistent_success_samples",
        "pd_error_only": "pd_error_only_samples",
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
    parser = argparse.ArgumentParser(description="Paddle ↔ PyTorch differential testing analysis + sample extraction")
    parser.add_argument(
        "--result-dir",
        "-r",
        default=os.path.join(ROOT_DIR, "pd_pt_test", "pd_pt_log_1"),
        help="Test results directory path",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=os.path.join(ROOT_DIR, "pd_pt_test", "analysis"),
        help="Summary report output directory (TXT/CSV)",
    )
    parser.add_argument(
        "--sample-dir",
        "-s",
        default=os.path.join(ROOT_DIR, "pd_pt_test", "analysis"),
        help="Sample JSON output directory",
    )

    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("Paddle ↔ PyTorch differential testing analysis + sample extraction")
    print("=" * 80)
    print(f"📁 Results directory: {args.result_dir}")
    print(f"📁 Report directory: {args.output_dir}")
    print(f"📁 Sample directory: {args.sample_dir}")

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
    print("📊 Quick summary")
    print("=" * 50)
    print(f"✅ consistent: {consistent}/{len(all_analyses)}")
    print(f"❌ inconsistent: {inconsistent}/{len(all_analyses)}")
    print(f"⚠️ error: {error}/{len(all_analyses)}")
    print("=" * 50)


if __name__ == "__main__":
    main()

