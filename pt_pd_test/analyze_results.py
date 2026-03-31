#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 5: Differential test result analysis script (PyTorch -> Paddle)

Functions:
- Read JSON test result files in pt_pd_test/pt_pd_log_1
- If multiple JSON files exist for the same operator, keep only the newest timestamp
- Count consistency/inconsistency/error distribution for each operator
- Generate summary reports (TXT + CSV + JSON)

Usage:
    conda activate tf_env
    python pt_pd_test/analyze_results.py [--result-dir pt_pd_test/pt_pd_log_1]
"""

import os
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import re
import json
import csv
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def parse_datetime_from_filename(filename: str) -> Optional[datetime]:
    match = re.search(r"_(\d{8}_\d{6})\.json$", filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def parse_record_timestamp(data: Dict[str, Any], filename: str) -> datetime:
    timestamp_text = data.get("timestamp")
    if isinstance(timestamp_text, str):
        try:
            return datetime.fromisoformat(timestamp_text)
        except ValueError:
            pass

    filename_dt = parse_datetime_from_filename(filename)
    if filename_dt is not None:
        return filename_dt

    return datetime.min


def normalize_operator_name(data: Dict[str, Any], filename: str) -> str:
    operator = data.get("operator")
    if isinstance(operator, str) and operator.strip():
        return operator.strip()

    base = os.path.basename(filename)
    match = re.match(r"^llm_enhanced_(.+)_\d{8}_\d{6}\.json$", base)
    if match:
        return match.group(1)
    return base


def load_latest_results_by_operator(result_dir: str) -> Tuple[List[Dict[str, Any]], int]:
    latest_by_operator: Dict[str, Dict[str, Any]] = {}
    total_json_files = 0

    if not os.path.exists(result_dir):
        print(f"❌ Result directory does not exist: {result_dir}")
        return [], total_json_files

    for filename in sorted(os.listdir(result_dir)):
        if not (filename.startswith("llm_enhanced_") and filename.endswith(".json")):
            continue

        filepath = os.path.join(result_dir, filename)
        total_json_files += 1

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load {filename}: {e}")
            continue

        operator_name = normalize_operator_name(data, filename)
        record_dt = parse_record_timestamp(data, filename)
        prev = latest_by_operator.get(operator_name)

        if prev is None or record_dt > prev["dt"] or (record_dt == prev["dt"] and filename > prev["filename"]):
            latest_by_operator[operator_name] = {
                "data": data,
                "dt": record_dt,
                "filename": filename,
            }

    return [entry["data"] for entry in latest_by_operator.values()], total_json_files


def analyze_single_operator(data: Dict[str, Any]) -> Dict[str, Any]:
    operator = data.get("operator", "unknown")
    iterations = data.get("results", [])

    analysis = {
        "torch_api": operator,
        "paddle_api": "",
        "total_iterations": len(iterations),
        "consistent_count": 0,
        "inconsistent_count": 0,
        "torch_error_count": 0,
        "paddle_error_count": 0,
        "both_error_count": 0,
        "comparison_error_count": 0,
        "final_status": "unknown",
        "errors": [],
    }

    for item in iterations:
        exec_result = item.get("execution_result", {})

        if not analysis["paddle_api"]:
            analysis["paddle_api"] = exec_result.get("paddle_api", "")

        status = exec_result.get("status", "unknown")
        results_match = exec_result.get("results_match")

        if status == "compared":
            if results_match is True:
                analysis["consistent_count"] += 1
            elif results_match is False:
                analysis["inconsistent_count"] += 1
                cmp_error = exec_result.get("comparison_error")
                if cmp_error:
                    analysis["errors"].append(f"[CMP] {str(cmp_error)[:160]}")
            else:
                analysis["comparison_error_count"] += 1
        elif status == "torch_failed":
            analysis["torch_error_count"] += 1
            err = exec_result.get("torch_error")
            if err:
                analysis["errors"].append(f"[PT] {str(err)[:160]}")
        elif status == "paddle_failed":
            analysis["paddle_error_count"] += 1
            err = exec_result.get("paddle_error")
            if err:
                analysis["errors"].append(f"[PD] {str(err)[:160]}")
        elif status == "both_failed":
            analysis["both_error_count"] += 1
            torch_err = exec_result.get("torch_error")
            paddle_err = exec_result.get("paddle_error")
            if torch_err:
                analysis["errors"].append(f"[PT] {str(torch_err)[:160]}")
            if paddle_err:
                analysis["errors"].append(f"[PD] {str(paddle_err)[:160]}")
        else:
            torch_success = exec_result.get("torch_success")
            paddle_success = exec_result.get("paddle_success")
            if torch_success and paddle_success:
                if results_match is True:
                    analysis["consistent_count"] += 1
                elif results_match is False:
                    analysis["inconsistent_count"] += 1
                else:
                    analysis["comparison_error_count"] += 1
            elif torch_success and not paddle_success:
                analysis["paddle_error_count"] += 1
            elif not torch_success and paddle_success:
                analysis["torch_error_count"] += 1
            elif torch_success is False and paddle_success is False:
                analysis["both_error_count"] += 1
            else:
                analysis["comparison_error_count"] += 1

    if analysis["consistent_count"] > 0 and analysis["inconsistent_count"] == 0:
        analysis["final_status"] = "consistent"
    elif analysis["inconsistent_count"] > 0:
        analysis["final_status"] = "inconsistent"
    elif (
        analysis["torch_error_count"]
        + analysis["paddle_error_count"]
        + analysis["both_error_count"]
        + analysis["comparison_error_count"]
        > 0
    ):
        analysis["final_status"] = "error"
    else:
        analysis["final_status"] = "unknown"

    analysis["errors"] = list(dict.fromkeys(analysis["errors"]))[:5]
    return analysis


def generate_reports(all_analyses: List[Dict[str, Any]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    total_operators = len(all_analyses)
    consistent_ops = [a for a in all_analyses if a["final_status"] == "consistent"]
    inconsistent_ops = [a for a in all_analyses if a["final_status"] == "inconsistent"]
    error_ops = [a for a in all_analyses if a["final_status"] == "error"]
    unknown_ops = [a for a in all_analyses if a["final_status"] == "unknown"]

    total_iterations = sum(a["total_iterations"] for a in all_analyses)
    total_consistent = sum(a["consistent_count"] for a in all_analyses)
    total_inconsistent = sum(a["inconsistent_count"] for a in all_analyses)
    total_torch_errors = sum(a["torch_error_count"] for a in all_analyses)
    total_paddle_errors = sum(a["paddle_error_count"] for a in all_analyses)
    total_both_errors = sum(a["both_error_count"] for a in all_analyses)

    txt_file = os.path.join(output_dir, f"analysis_report_{timestamp}.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch ↔ Paddle Differential Test Analysis Report\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total test operators: {total_operators}\n")
        f.write(f"consistent: {len(consistent_ops)}\n")
        f.write(f"inconsistent: {len(inconsistent_ops)}\n")
        f.write(f"error: {len(error_ops)}\n")
        f.write(f"unknown: {len(unknown_ops)}\n\n")
        f.write(f"Total iterations: {total_iterations}\n")
        f.write(f"Consistent count: {total_consistent}\n")
        f.write(f"Inconsistent count: {total_inconsistent}\n")
        f.write(f"PT error count: {total_torch_errors}\n")
        f.write(f"PD error count: {total_paddle_errors}\n")
        f.write(f"Both-sides error count: {total_both_errors}\n")

    print(f"📄 TXT report saved: {txt_file}")

    csv_file = os.path.join(output_dir, f"analysis_report_{timestamp}.csv")
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "torch_api",
            "paddle_api",
            "final_status",
            "total_iterations",
            "consistent_count",
            "inconsistent_count",
            "torch_error_count",
            "paddle_error_count",
            "both_error_count",
            "comparison_error_count",
            "error_summary",
        ])
        for a in sorted(all_analyses, key=lambda x: x["torch_api"]):
            writer.writerow([
                a["torch_api"],
                a["paddle_api"],
                a["final_status"],
                a["total_iterations"],
                a["consistent_count"],
                a["inconsistent_count"],
                a["torch_error_count"],
                a["paddle_error_count"],
                a["both_error_count"],
                a["comparison_error_count"],
                "; ".join(a["errors"][:3]) if a["errors"] else "",
            ])
    print(f"📄 CSV report saved: {csv_file}")

    json_file = os.path.join(output_dir, f"analysis_report_{timestamp}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "summary": {
                    "total_operators": total_operators,
                    "consistent": len(consistent_ops),
                    "inconsistent": len(inconsistent_ops),
                    "error": len(error_ops),
                    "unknown": len(unknown_ops),
                    "total_iterations": total_iterations,
                    "total_consistent_iterations": total_consistent,
                    "total_inconsistent_iterations": total_inconsistent,
                    "total_torch_errors": total_torch_errors,
                    "total_paddle_errors": total_paddle_errors,
                    "total_both_errors": total_both_errors,
                },
                "operators": all_analyses,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"📄 JSON report saved: {json_file}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch ↔ Paddle differential test analysis")
    parser.add_argument(
        "--result-dir",
        "-r",
        default=os.path.join(ROOT_DIR, "pt_pd_test", "pt_pd_log_1"),
        help="Test result directory path",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=os.path.join(ROOT_DIR, "pt_pd_test", "analysis"),
        help="Analysis report output directory",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("PyTorch ↔ Paddle differential test analysis")
    print("=" * 80)
    print(f"📁 Result directory: {args.result_dir}")
    print(f"📁 Output directory: {args.output_dir}")

    all_results, total_json_files = load_latest_results_by_operator(args.result_dir)
    if not all_results:
        print("⚠️ No available test result files found")
        return

    print(f"\n📋 JSON files scanned: {total_json_files}")
    print(f"📌 Operators after de-dup (latest timestamp per operator): {len(all_results)}")

    all_analyses = [analyze_single_operator(data) for data in all_results]
    generate_reports(all_analyses, args.output_dir)

    consistent = sum(1 for a in all_analyses if a["final_status"] == "consistent")
    inconsistent = sum(1 for a in all_analyses if a["final_status"] == "inconsistent")
    error = sum(1 for a in all_analyses if a["final_status"] == "error")
    unknown = sum(1 for a in all_analyses if a["final_status"] == "unknown")

    print("\n" + "=" * 50)
    print("📊 Quick summary")
    print("=" * 50)
    print(f"✅ Consistent: {consistent}/{len(all_analyses)}")
    print(f"❌ Inconsistent: {inconsistent}/{len(all_analyses)}")
    print(f"⚠️ Error: {error}/{len(all_analyses)}")
    print(f"❓ Unknown: {unknown}/{len(all_analyses)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
