#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 5: Analyze PD<->PT differential testing results

Function:
- Read all JSON result files under pd_pt_log_1/
- Generate summary report: consistent/inconsistent/error operator distribution
- Analyze the effect of LLM repair/mutation
- Output detailed analysis results

Usage:
    conda activate tf_env
    python pd_pt_test/analyze_results.py [--result-dir pd_pt_test/pd_pt_log_1] [--output pd_pt_test/analysis]
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_result_files(result_dir: str) -> List[Dict[str, Any]]:
    """Load all JSON result files."""
    results = []
    if not os.path.isdir(result_dir):
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
                print(f"  ⚠️ Readfailed: {filename}: {e}")

    return results


def analyze_operator_results(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the test results of a single operator."""
    pd_api = result_data.get("pd_api", "unknown")
    pytorch_api = result_data.get("pytorch_api", "unknown")
    iterations = result_data.get("results", [])

    analysis = {
        "pd_api": pd_api,
        "pytorch_api": pytorch_api,
        "total_iterations": len(iterations),
        "consistent_count": 0,
        "inconsistent_count": 0,
        "pd_error_count": 0,
        "pytorch_error_count": 0,
        "both_error_count": 0,
        "llm_mutations": 0,
        "llm_repairs": 0,
        "llm_skips": 0,
        "llm_generated_count": result_data.get("llm_generated_test_cases", 0),
        "successful_count": result_data.get("successful_test_cases", 0),
        "final_status": "unknown",
        "inconsistency_details": [],
        "error_details": [],
    }

    for iter_data in iterations:
        exec_result = iter_data.get("execution_result", {})
        status = exec_result.get("status", "unknown")

        if status == "consistent":
            analysis["consistent_count"] += 1
        elif status == "inconsistent":
            analysis["inconsistent_count"] += 1
            detail = exec_result.get("comparison_error", "")
            if detail:
                analysis["inconsistency_details"].append(detail)
        elif status == "paddle_error":
            analysis["pd_error_count"] += 1
            analysis["error_details"].append(f"PD: {exec_result.get('pd_error', '')[:100]}")
        elif status == "pytorch_error":
            analysis["pytorch_error_count"] += 1
            analysis["error_details"].append(f"PT: {exec_result.get('pytorch_error', '')[:100]}")
        elif status == "both_error":
            analysis["both_error_count"] += 1

        llm_op = iter_data.get("llm_operation", {})
        if isinstance(llm_op, dict):
            op_type = llm_op.get("operation", "")
            if op_type == "mutation":
                analysis["llm_mutations"] += 1
            elif op_type == "repair":
                analysis["llm_repairs"] += 1
            elif op_type == "skip":
                analysis["llm_skips"] += 1

    # Determine final status
    if analysis["consistent_count"] > 0 and analysis["inconsistent_count"] == 0:
        analysis["final_status"] = "all_consistent"
    elif analysis["inconsistent_count"] > 0:
        analysis["final_status"] = "has_inconsistency"
    elif analysis["pd_error_count"] + analysis["pytorch_error_count"] + analysis["both_error_count"] == analysis["total_iterations"]:
        analysis["final_status"] = "all_error"
    else:
        analysis["final_status"] = "mixed"

    return analysis


def generate_summary_report(all_analyses: List[Dict[str, Any]], output_dir: str):
    """Generate summary report."""
    os.makedirs(output_dir, exist_ok=True)

    # ---- Summary stats ----
    total_ops = len(all_analyses)
    status_counts = Counter(a["final_status"] for a in all_analyses)
    total_iterations = sum(a["total_iterations"] for a in all_analyses)
    total_consistent = sum(a["consistent_count"] for a in all_analyses)
    total_inconsistent = sum(a["inconsistent_count"] for a in all_analyses)
    total_llm_generated = sum(a["llm_generated_count"] for a in all_analyses)
    total_successful = sum(a["successful_count"] for a in all_analyses)
    total_mutations = sum(a["llm_mutations"] for a in all_analyses)
    total_repairs = sum(a["llm_repairs"] for a in all_analyses)
    total_skips = sum(a["llm_skips"] for a in all_analyses)

    # ---- Console output ----
    print("\n" + "=" * 80)
    print("📊 PD<->PT differential testing analysis report")
    print("=" * 80)

    print("\n1. Operator-level stats")
    print(f"   Total operators: {total_ops}")
    print(f"   All consistent: {status_counts.get('all_consistent', 0)}")
    print(f"   Has inconsistency: {status_counts.get('has_inconsistency', 0)}")
    print(f"   All errors: {status_counts.get('all_error', 0)}")
    print(f"   Mixed status: {status_counts.get('mixed', 0)}")

    print("\n2. Iteration-level stats")
    print(f"   Total iterations: {total_iterations}")
    print(f"   Consistent results: {total_consistent}")
    print(f"   Inconsistent results: {total_inconsistent}")

    print("\n3. LLM operation stats")
    print(f"   LLM generated case count: {total_llm_generated}")
    print(f"   Successful execute case count: {total_successful}")
    if total_llm_generated > 0:
        print(f"   Success rate: {total_successful / total_llm_generated * 100:.2f}%")
    print(f"   Mutation count: {total_mutations}")
    print(f"   Repair count: {total_repairs}")
    print(f"   Skip count: {total_skips}")

    # ---- Inconsistent operator list ----
    inconsistent_ops = [a for a in all_analyses if a["final_status"] == "has_inconsistency"]
    if inconsistent_ops:
        print(f"\n4. Inconsistent operators found ({len(inconsistent_ops)} items)")
        print(f"   {'PD API':<40} {'PT API':<40} {'consistent/inconsistent'}")
        print(f"   {'-' * 40} {'-' * 40} {'-' * 12}")
        for a in inconsistent_ops:
            print(f"   {a['pd_api']:<40} {a['pytorch_api']:<40} {a['consistent_count']}/{a['inconsistent_count']}")

    # ---- Save detailed report ----
    report_file = os.path.join(output_dir, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PD<->PT differential testing detailed analysis report\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. Overall statistics\n")
        f.write(f"   Total operators: {total_ops}\n")
        f.write(f"   All consistent: {status_counts.get('all_consistent', 0)}\n")
        f.write(f"   Has inconsistency: {status_counts.get('has_inconsistency', 0)}\n")
        f.write(f"   All errors: {status_counts.get('all_error', 0)}\n")
        f.write(f"   Mixed status: {status_counts.get('mixed', 0)}\n\n")

        f.write("2. LLM operation stats\n")
        f.write(f"   Generated case count: {total_llm_generated}\n")
        f.write(f"   Successful execute count: {total_successful}\n")
        f.write(f"   mutation: {total_mutations}, repair: {total_repairs}, skip: {total_skips}\n\n")

        f.write("3. Per-operator details\n")
        f.write("-" * 80 + "\n")
        for a in sorted(all_analyses, key=lambda x: x["pd_api"]):
            f.write(f"\n  {a['pd_api']} → {a['pytorch_api']}\n")
            f.write(f"    status: {a['final_status']}\n")
            f.write(f"    Iterations: {a['total_iterations']}, consistent: {a['consistent_count']}, inconsistent: {a['inconsistent_count']}\n")
            f.write(f"    LLM: generated={a['llm_generated_count']}, successful={a['successful_count']}\n")
            if a["inconsistency_details"]:
                f.write("    Inconsistency details:\n")
                for detail in a["inconsistency_details"][:3]:
                    f.write(f"      - {detail}\n")
            if a["error_details"]:
                f.write("    Error details:\n")
                for detail in a["error_details"][:3]:
                    f.write(f"      - {detail}\n")

    print(f"\n💾 Detailed report saved to: {report_file}")

    # ---- Save CSV summary ----
    csv_file = os.path.join(output_dir, f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pd_api", "pytorch_api", "final_status",
            "total_iterations", "consistent_count", "inconsistent_count",
            "llm_generated_count", "successful_count",
            "llm_mutations", "llm_repairs", "llm_skips",
        ])
        writer.writeheader()
        for a in sorted(all_analyses, key=lambda x: x["pd_api"]):
            writer.writerow({k: a[k] for k in writer.fieldnames})

    print(f"💾 CSV summary saved to: {csv_file}")

    # ---- Save JSON summary ----
    json_file = os.path.join(output_dir, f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_operators": total_ops,
                "status_distribution": dict(status_counts),
                "total_iterations": total_iterations,
                "total_consistent": total_consistent,
                "total_inconsistent": total_inconsistent,
                "total_llm_generated": total_llm_generated,
                "total_successful": total_successful,
                "llm_mutations": total_mutations,
                "llm_repairs": total_repairs,
                "llm_skips": total_skips,
            },
            "operators": all_analyses,
        }, f, indent=2, ensure_ascii=False)

    print(f"💾 JSON data saved to: {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Analyze PD↔PT differential testing results"
    )
    parser.add_argument(
        "--result-dir", "-r",
        default=os.path.join(ROOT_DIR, "pd_pt_test", "pd_pt_log_1"),
        help="Result directory path"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "pd_pt_test", "analysis"),
        help="Analysis output directory"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Step 5: PD↔PT differential testing analysis")
    print("=" * 80)
    print(f"📁 Results directory: {args.result_dir}")

    # Load results
    result_files = load_result_files(args.result_dir)
    print(f"📋 Loaded {len(result_files)} result files")

    if not result_files:
        print("⚠️ No result files found; run Step 4 first")
        return

    # Analyze each operator
    all_analyses = []
    for result_data in result_files:
        analysis = analyze_operator_results(result_data)
        all_analyses.append(analysis)

    # Generate reports
    generate_summary_report(all_analyses, args.output)

    print("\n✅ Analysis completed")


if __name__ == "__main__":
    main()

