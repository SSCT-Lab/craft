#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 5: Differential test result analysis script

Purpose:
- Read JSON test result files under ms_pt_log_1/
- Summarize per-operator consistent/inconsistent/error distribution
- Generate reports (TXT + CSV + JSON)

Usage:
    conda activate tf_env
    python ms_pt_test/analyze_results.py [--result-dir ms_pt_test/ms_pt_log_1]
"""

import os
import sys
import io

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Any
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def load_all_results(result_dir: str) -> List[Dict[str, Any]]:
    """Load all JSON result files under the results directory."""
    results = []
    if not os.path.exists(result_dir):
        print(f"❌ Result directory does not exist: {result_dir}")
        return results

    for filename in sorted(os.listdir(result_dir)):
        if filename.startswith("llm_enhanced_") and filename.endswith(".json"):
            filepath = os.path.join(result_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                results.append(data)
            except Exception as e:
                print(f"⚠️ Failed to load {filename}: {e}")
    return results


def analyze_single_operator(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze test results for a single operator."""
    ms_api = data.get("ms_api", "unknown")
    pytorch_api = data.get("pytorch_api", "")
    iterations = data.get("results", [])

    analysis = {
        "ms_api": ms_api,
        "pytorch_api": pytorch_api,
        "total_iterations": len(iterations),
        "consistent_count": 0,
        "inconsistent_count": 0,
        "ms_error_count": 0,
        "pytorch_error_count": 0,
        "both_error_count": 0,
        "comparison_error_count": 0,
        "final_status": "unknown",
        "errors": [],
    }

    for item in iterations:
        exec_result = item.get("execution_result", {})
        status = exec_result.get("status", "unknown")

        if status == "consistent":
            analysis["consistent_count"] += 1
        elif status == "inconsistent":
            analysis["inconsistent_count"] += 1
        elif status == "ms_error":
            analysis["ms_error_count"] += 1
            if exec_result.get("ms_error"):
                analysis["errors"].append(f"[MS] {exec_result['ms_error'][:100]}")
        elif status == "pytorch_error":
            analysis["pytorch_error_count"] += 1
            if exec_result.get("pytorch_error"):
                analysis["errors"].append(f"[PT] {exec_result['pytorch_error'][:100]}")
        elif status == "both_error":
            analysis["both_error_count"] += 1
        elif status == "comparison_error":
            analysis["comparison_error_count"] += 1

    # Determine final status
    if analysis["consistent_count"] > 0 and analysis["inconsistent_count"] == 0:
        analysis["final_status"] = "consistent"
    elif analysis["inconsistent_count"] > 0:
        analysis["final_status"] = "inconsistent"
    elif analysis["ms_error_count"] + analysis["pytorch_error_count"] + analysis["both_error_count"] > 0:
        analysis["final_status"] = "error"
    else:
        analysis["final_status"] = "unknown"

    # Deduplicate error messages
    analysis["errors"] = list(set(analysis["errors"]))[:5]

    return analysis


def generate_reports(all_analyses: List[Dict[str, Any]], output_dir: str):
    """Generate summary reports."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Overall summary ----
    total_operators = len(all_analyses)
    consistent_ops = [a for a in all_analyses if a["final_status"] == "consistent"]
    inconsistent_ops = [a for a in all_analyses if a["final_status"] == "inconsistent"]
    error_ops = [a for a in all_analyses if a["final_status"] == "error"]
    unknown_ops = [a for a in all_analyses if a["final_status"] == "unknown"]

    total_iterations = sum(a["total_iterations"] for a in all_analyses)
    total_consistent = sum(a["consistent_count"] for a in all_analyses)
    total_inconsistent = sum(a["inconsistent_count"] for a in all_analyses)
    total_ms_errors = sum(a["ms_error_count"] for a in all_analyses)
    total_pt_errors = sum(a["pytorch_error_count"] for a in all_analyses)

    # ---- 1. TXT report ----
    txt_file = os.path.join(output_dir, f"analysis_report_{timestamp}.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MindSpore ↔ PyTorch Differential Test Report\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("=" * 50 + "\n")
        f.write("📊 Overall Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total operators: {total_operators}\n")
        f.write(f"  ✅ Consistent: {len(consistent_ops)} "
                f"({len(consistent_ops)/max(total_operators,1)*100:.1f}%)\n")
        f.write(f"  ❌ Inconsistent: {len(inconsistent_ops)} "
                f"({len(inconsistent_ops)/max(total_operators,1)*100:.1f}%)\n")
        f.write(f"  ⚠️ Error: {len(error_ops)} "
                f"({len(error_ops)/max(total_operators,1)*100:.1f}%)\n")
        f.write(f"  ❓ Unknown: {len(unknown_ops)}\n\n")

        f.write(f"Total iterations: {total_iterations}\n")
        f.write(f"  Consistent count: {total_consistent}\n")
        f.write(f"  Inconsistent count: {total_inconsistent}\n")
        f.write(f"  MS error count: {total_ms_errors}\n")
        f.write(f"  PT error count: {total_pt_errors}\n\n")

        # Consistent operators
        f.write("=" * 50 + "\n")
        f.write(f"✅ Consistent operators ({len(consistent_ops)})\n")
        f.write("=" * 50 + "\n")
        for a in sorted(consistent_ops, key=lambda x: x["ms_api"]):
            f.write(f"  {a['ms_api']} → {a['pytorch_api']} "
                    f"({a['consistent_count']}/{a['total_iterations']} consistent)\n")

        # Inconsistent operators
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"❌ Inconsistent operators ({len(inconsistent_ops)})\n")
        f.write("=" * 50 + "\n")
        for a in sorted(inconsistent_ops, key=lambda x: x["ms_api"]):
            f.write(f"  {a['ms_api']} → {a['pytorch_api']}\n")
            f.write(f"    Consistent: {a['consistent_count']}, Inconsistent: {a['inconsistent_count']}\n")
            for err in a["errors"][:3]:
                f.write(f"    ! {err}\n")

        # Error operators
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"⚠️ Error operators ({len(error_ops)})\n")
        f.write("=" * 50 + "\n")
        for a in sorted(error_ops, key=lambda x: x["ms_api"]):
            f.write(f"  {a['ms_api']} → {a['pytorch_api']}\n")
            f.write(f"    MS errors: {a['ms_error_count']}, PT errors: {a['pytorch_error_count']}, "
                f"both errors: {a['both_error_count']}\n")
            for err in a["errors"][:3]:
                f.write(f"    ! {err}\n")

    print(f"📄 TXT report saved: {txt_file}")

    # ---- 2. CSV report ----
    csv_file = os.path.join(output_dir, f"analysis_report_{timestamp}.csv")
    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "ms_api", "pytorch_api", "final_status",
            "total_iterations", "consistent_count", "inconsistent_count",
            "ms_error_count", "pytorch_error_count", "both_error_count",
            "error_summary",
        ])
        for a in sorted(all_analyses, key=lambda x: x["ms_api"]):
            writer.writerow([
                a["ms_api"], a["pytorch_api"], a["final_status"],
                a["total_iterations"], a["consistent_count"], a["inconsistent_count"],
                a["ms_error_count"], a["pytorch_error_count"], a["both_error_count"],
                "; ".join(a["errors"][:3]) if a["errors"] else "",
            ])
    print(f"📄 CSV report saved: {csv_file}")

    # ---- 3. JSON report ----
    json_file = os.path.join(output_dir, f"analysis_report_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
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
            },
            "operators": all_analyses,
        }, f, indent=2, ensure_ascii=False)
    print(f"📄 JSON report saved: {json_file}")


def main():
    parser = argparse.ArgumentParser(description="MindSpore ↔ PyTorch differential test analysis")
    parser.add_argument(
        "--result-dir", "-r",
        default=os.path.join(ROOT_DIR, "ms_pt_test", "ms_pt_log_1"),
        help="Result directory path",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=os.path.join(ROOT_DIR, "ms_pt_test", "analysis"),
        help="Report output directory",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MindSpore ↔ PyTorch differential test analysis")
    print("=" * 80)
    print(f"📁 Result directory: {args.result_dir}")
    print(f"📁 Output directory: {args.output_dir}")

    # Load results
    all_results = load_all_results(args.result_dir)
    if not all_results:
        print("⚠️ No test result files found")
        return

    print(f"\n📋 Loaded results for {len(all_results)} operators")

    # Analyze each operator
    all_analyses = []
    for data in all_results:
        analysis = analyze_single_operator(data)
        all_analyses.append(analysis)

    # Generate reports
    generate_reports(all_analyses, args.output_dir)

    # Print console summary
    consistent = sum(1 for a in all_analyses if a["final_status"] == "consistent")
    inconsistent = sum(1 for a in all_analyses if a["final_status"] == "inconsistent")
    error = sum(1 for a in all_analyses if a["final_status"] == "error")

    print("\n" + "=" * 50)
    print("📊 Quick summary")
    print("=" * 50)
    print(f"✅ Consistent: {consistent}/{len(all_analyses)}")
    print(f"❌ Inconsistent: {inconsistent}/{len(all_analyses)}")
    print(f"⚠️ Error: {error}/{len(all_analyses)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
