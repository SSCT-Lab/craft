"""
PyTorch-MindSpore fuzzing differential testing result analysis tool

Features:
    1. Read fuzzing test results
    2. Count bug candidates and error category distribution
    3. Generate analysis reports
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict


# Results directory
RESULT_DIR = Path(__file__).parent / "result"


def load_fuzzing_results(result_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all fuzzing result files.
    """
    results = []
    result_files = sorted(result_dir.glob("*_fuzzing_result_*.json"))
    
    for file in result_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_source_file"] = file.name
                results.append(data)
        except Exception as e:
            print(f"[WARN] Failed to load {file.name}: {e}")
    
    return results


def categorize_error(error_msg: str) -> str:
    """
    Categorize error message.
    """
    if error_msg is None:
        return "No error"
    
    error_lower = error_msg.lower()
    
    # Shape-related
    if "shape" in error_lower or "dimension" in error_lower or "size" in error_lower:
        return "Shape mismatch"
    
    # Dtype-related
    if "dtype" in error_lower or "type" in error_lower:
        return "Dtype error"
    
    # Numeric-related
    if "nan" in error_lower or "inf" in error_lower or "overflow" in error_lower:
        return "Numeric anomaly"
    
    # Memory-related
    if "memory" in error_lower or "oom" in error_lower or "cuda" in error_lower:
        return "Memory/device error"
    
    # MindSpore-specific
    if "mindspore" in error_lower:
        if "not support" in error_lower or "unsupported" in error_lower:
            return "MindSpore unsupported"
        if "graph" in error_lower or "compile" in error_lower:
            return "MindSpore compile error"
    
    # Argument-related
    if "argument" in error_lower or "param" in error_lower or "invalid" in error_lower:
        return "Invalid argument"
    
    # Attribute-related
    if "attribute" in error_lower or "has no" in error_lower:
        return "Attribute/method missing"
    
    # Value mismatch
    if "mismatch" in error_msg or "mismatch" in error_lower:
        return "Result mismatch"
    
    return "Other error"


def analyze_single_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze fuzzing results for a single operator.
    """
    analysis = {
        "operator": result.get("operator", "unknown"),
        "torch_api": result.get("torch_api", ""),
        "mindspore_api": result.get("mindspore_api", ""),
        "total_cases": result.get("total_cases", 0),
        "total_fuzzing_rounds": result.get("total_fuzzing_rounds", 0),
        "bug_candidates": result.get("bug_candidates", 0),
        "error_categories": defaultdict(int),
        "bug_details": [],
        "execution_stats": {
            "torch_success": 0,
            "torch_fail": 0,
            "mindspore_success": 0,
            "mindspore_fail": 0,
            "both_success_match": 0,
            "both_success_mismatch": 0
        }
    }
    
    for case_result in result.get("results", []):
        for fuzz_result in case_result.get("fuzzing_results", []):
            if not fuzz_result.get("success", False):
                analysis["error_categories"]["Fuzzing failed"] += 1
                continue
            
            exec_result = fuzz_result.get("execution_result", {})
            
            # Execution stats
            torch_ok = exec_result.get("torch_success", False)
            ms_ok = exec_result.get("mindspore_success", False)
            
            if torch_ok:
                analysis["execution_stats"]["torch_success"] += 1
            else:
                analysis["execution_stats"]["torch_fail"] += 1
            
            if ms_ok:
                analysis["execution_stats"]["mindspore_success"] += 1
            else:
                analysis["execution_stats"]["mindspore_fail"] += 1
            
            if torch_ok and ms_ok:
                if exec_result.get("results_match", False):
                    analysis["execution_stats"]["both_success_match"] += 1
                else:
                    analysis["execution_stats"]["both_success_mismatch"] += 1
            
            # Analyze potential bugs
            if fuzz_result.get("is_bug_candidate", False):
                error_msg = exec_result.get("comparison_error") or \
                           exec_result.get("torch_error") or \
                           exec_result.get("mindspore_error")
                category = categorize_error(error_msg)
                analysis["error_categories"][category] += 1
                
                analysis["bug_details"].append({
                    "round": fuzz_result.get("round"),
                    "mutation_strategy": fuzz_result.get("mutation_strategy", ""),
                    "error_category": category,
                    "error_detail": error_msg,
                    "torch_success": torch_ok,
                    "mindspore_success": ms_ok,
                    "torch_shape": exec_result.get("torch_shape"),
                    "mindspore_shape": exec_result.get("mindspore_shape"),
                    "original_case_info": case_result.get("original_case_info", {})
                })
    
    # Convert defaultdict to dict
    analysis["error_categories"] = dict(analysis["error_categories"])
    
    return analysis


def generate_summary_report(all_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary report.
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_operators": len(all_analyses),
        "total_cases": 0,
        "total_fuzzing_rounds": 0,
        "total_bug_candidates": 0,
        "operators_with_bugs": 0,
        "global_error_categories": defaultdict(int),
        "global_execution_stats": {
            "torch_success": 0,
            "torch_fail": 0,
            "mindspore_success": 0,
            "mindspore_fail": 0,
            "both_success_match": 0,
            "both_success_mismatch": 0
        },
        "operator_summary": [],
        "top_bug_operators": []
    }
    
    for analysis in all_analyses:
        summary["total_cases"] += analysis["total_cases"]
        summary["total_fuzzing_rounds"] += analysis["total_fuzzing_rounds"]
        summary["total_bug_candidates"] += analysis["bug_candidates"]
        
        if analysis["bug_candidates"] > 0:
            summary["operators_with_bugs"] += 1
        
        # Aggregate error categories
        for category, count in analysis["error_categories"].items():
            summary["global_error_categories"][category] += count
        
        # Aggregate execution stats
        for key in summary["global_execution_stats"]:
            summary["global_execution_stats"][key] += analysis["execution_stats"].get(key, 0)
        
        # Operator summary
        summary["operator_summary"].append({
            "operator": analysis["operator"],
            "torch_api": analysis["torch_api"],
            "mindspore_api": analysis["mindspore_api"],
            "cases": analysis["total_cases"],
            "fuzzing_rounds": analysis["total_fuzzing_rounds"],
            "bug_candidates": analysis["bug_candidates"],
            "bug_rate": f"{analysis['bug_candidates'] / max(1, analysis['total_fuzzing_rounds']) * 100:.1f}%"
        })
    
    # Convert defaultdict
    summary["global_error_categories"] = dict(summary["global_error_categories"])
    
    # Sort by bug count
    summary["top_bug_operators"] = sorted(
        summary["operator_summary"],
        key=lambda x: x["bug_candidates"],
        reverse=True
    )[:20]
    
    # Compute overall rate
    total_rounds = max(1, summary["total_fuzzing_rounds"])
    summary["overall_bug_rate"] = f"{summary['total_bug_candidates'] / total_rounds * 100:.2f}%"
    
    return summary


def print_report(summary: Dict[str, Any]) -> None:
    """
    Print analysis report.
    """
    print("\n" + "=" * 80)
    print("PyTorch-MindSpore Fuzzing Differential Test Analysis Report")
    print("=" * 80)
    print(f"Generated at: {summary['timestamp']}")
    print()
    
    # Overview
    print("[Overall Summary]")
    print(f"  Operators analyzed: {summary['total_operators']}")
    print(f"  Total test cases: {summary['total_cases']}")
    print(f"  Total fuzzing rounds: {summary['total_fuzzing_rounds']}")
    print(f"  Potential issues found: {summary['total_bug_candidates']}")
    print(f"  Operators with issues: {summary['operators_with_bugs']}")
    print(f"  Overall issue discovery rate: {summary['overall_bug_rate']}")
    print()
    
    # Execution stats
    print("[Execution Stats]")
    stats = summary["global_execution_stats"]
    print(f"  PyTorch success: {stats['torch_success']}")
    print(f"  PyTorch failure: {stats['torch_fail']}")
    print(f"  MindSpore success: {stats['mindspore_success']}")
    print(f"  MindSpore failure: {stats['mindspore_fail']}")
    print(f"  Both succeeded and matched: {stats['both_success_match']}")
    print(f"  Both succeeded but mismatched: {stats['both_success_mismatch']}")
    print()
    
    # Error categories
    print("[Issue Category Stats]")
    categories = summary["global_error_categories"]
    if categories:
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories:
            print(f"  {category}: {count}")
    else:
        print("  No issues recorded")
    print()
    
    # Top bug operators
    print("[Top 10 Operators with Most Issues]")
    for i, op in enumerate(summary["top_bug_operators"][:10], 1):
        if op["bug_candidates"] > 0:
            print(f"  {i}. {op['operator']}: {op['bug_candidates']} issues "
                f"(issue rate: {op['bug_rate']}, PyTorch: {op['torch_api']}, MindSpore: {op['mindspore_api']})")
    
    print("\n" + "=" * 80)


def export_detailed_bugs(all_analyses: List[Dict[str, Any]], output_file: Path) -> None:
    """
    Export detailed bug list.
    """
    all_bugs = []
    
    for analysis in all_analyses:
        for bug in analysis["bug_details"]:
            all_bugs.append({
                "operator": analysis["operator"],
                "torch_api": analysis["torch_api"],
                "mindspore_api": analysis["mindspore_api"],
                **bug
            })
    
    # Group by error category
    bugs_by_category = defaultdict(list)
    for bug in all_bugs:
        bugs_by_category[bug["error_category"]].append(bug)
    
    output = {
        "total_bugs": len(all_bugs),
        "bugs_by_category": {k: len(v) for k, v in bugs_by_category.items()},
        "detailed_bugs": dict(bugs_by_category)
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\nDetailed bug list exported: {output_file}")


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description="PyTorch-MindSpore Fuzzing Differential Test Analysis"
    )
    parser.add_argument(
        "--result-dir", "-r",
        type=Path,
        default=RESULT_DIR,
        help=f"Result file directory (default {RESULT_DIR})"
    )
    parser.add_argument(
        "--export", "-e",
        type=Path,
        default=None,
        help="File path to export detailed bug list"
    )
    parser.add_argument(
        "--summary-output", "-s",
        type=Path,
        default=None,
        help="JSON file path to save summary report"
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading fuzzing results from {args.result_dir}...")
    results = load_fuzzing_results(args.result_dir)
    
    if not results:
        print("[WARN] No fuzzing result files found")
        return
    
    print(f"Loaded {len(results)} result files")
    
    # Analyze each result
    all_analyses = []
    for result in results:
        analysis = analyze_single_result(result)
        all_analyses.append(analysis)
    
    # Generate summary report
    summary = generate_summary_report(all_analyses)
    
    # Print report
    print_report(summary)
    
    # Save summary report
    if args.summary_output:
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Summary report saved: {args.summary_output}")
    
    # Export detailed bug list
    if args.export:
        export_detailed_bugs(all_analyses, args.export)
    
    # Default export
    default_export = args.result_dir / f"bug_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    export_detailed_bugs(all_analyses, default_export)


if __name__ == "__main__":
    main()
