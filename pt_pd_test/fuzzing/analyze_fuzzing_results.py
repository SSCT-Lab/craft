"""  
PyTorch-PaddlePaddle fuzzing result analysis tool

Features:
    1. Analyze all JSON files in the fuzzing result directory
    2. Count discovered potential issues
    3. Generate detailed analysis reports
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def analyze_fuzzing_results(result_dir: str) -> Dict[str, Any]:
    """
    Analyze fuzzing results.
    """
    result_path = Path(result_dir)
    
    # Support filenames with or without timestamps
    json_files = sorted(result_path.glob("*_fuzzing_result*.json"))
    
    print(f"Found {len(json_files)} result files")
    
    # Summary stats
    stats = {
        "total_operators": 0,
        "total_cases": 0,
        "total_fuzzing_rounds": 0,
        "total_bug_candidates": 0,
        "operators_with_bugs": [],
        "bug_details": [],
        "success_rounds": 0,
        "failed_rounds": 0,
        "error_categories": {},  # Error category stats
    }
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            stats["total_operators"] += 1
            stats["total_cases"] += data.get("total_cases", 0)
            stats["total_fuzzing_rounds"] += data.get("total_fuzzing_rounds", 0)
            
            bug_count = data.get("bug_candidates", 0)
            stats["total_bug_candidates"] += bug_count
            
            if bug_count > 0:
                stats["operators_with_bugs"].append({
                    "operator": data.get("operator"),
                    "torch_api": data.get("torch_api"),
                    "paddle_api": data.get("paddle_api"),
                    "bug_count": bug_count
                })
            
            # Analyze fuzzing results for each case
            for result in data.get("results", []):
                for fr in result.get("fuzzing_results", []):
                    if fr.get("success"):
                        stats["success_rounds"] += 1
                        
                        if fr.get("is_bug_candidate"):
                            exec_result = fr.get("execution_result", {})
                            
                            # Error categorization
                            error_type = categorize_error(exec_result)
                            stats["error_categories"][error_type] = stats["error_categories"].get(error_type, 0) + 1
                            
                            bug_detail = {
                                "operator": data.get("operator"),
                                "torch_api": data.get("torch_api"),
                                "paddle_api": data.get("paddle_api"),
                                "round": fr.get("round"),
                                "mutation_strategy": fr.get("mutation_strategy"),
                                "error_type": error_type,
                                "torch_success": exec_result.get("torch_success"),
                                "paddle_success": exec_result.get("paddle_success"),
                                "torch_error": exec_result.get("torch_error"),
                                "paddle_error": exec_result.get("paddle_error"),
                                "comparison_error": exec_result.get("comparison_error"),
                                "torch_test_case": fr.get("torch_test_case"),
                                "paddle_test_case": fr.get("paddle_test_case"),
                            }
                            stats["bug_details"].append(bug_detail)
                    else:
                        stats["failed_rounds"] += 1
                        
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    return stats


def categorize_error(exec_result: Dict[str, Any]) -> str:
    """
    Categorize errors.
    """
    torch_success = exec_result.get("torch_success", False)
    paddle_success = exec_result.get("paddle_success", False)
    comparison_error = exec_result.get("comparison_error", "")
    
    # Execution status mismatch
    if torch_success != paddle_success:
        if torch_success:
            return "PaddlePaddle execution failed"
        else:
            return "PyTorch execution failed"
    
    # Both failed
    if not torch_success and not paddle_success:
        return "Both executions failed"
    
    # Result comparison error
    if comparison_error:
        if "shape mismatch" in comparison_error:
            return "Shape mismatch"
        elif "value mismatch" in comparison_error:
            return "Value mismatch"
        elif "NaN" in comparison_error:
            return "NaN position mismatch"
        elif "Inf" in comparison_error:
            return "Inf handling mismatch"
        else:
            return "Other comparison error"
    
    return "Unknown error"


def generate_analysis_report(stats: Dict[str, Any], output_file: str) -> None:
    """
    Generate analysis report.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch-PaddlePaddle Fuzzing Differential Test Analysis Report\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall summary
        f.write("[Overall Summary]\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total operators: {stats['total_operators']}\n")
        f.write(f"Total cases: {stats['total_cases']}\n")
        f.write(f"Total fuzzing rounds: {stats['total_fuzzing_rounds']}\n")
        f.write(f"Successful rounds: {stats['success_rounds']}\n")
        f.write(f"Failed rounds: {stats['failed_rounds']}\n")
        f.write(f"Potential issues found: {stats['total_bug_candidates']}\n")
        f.write(f"Operators with issues: {len(stats['operators_with_bugs'])}\n")
        f.write("\n")
        
        # Error category summary
        if stats['error_categories']:
            f.write("=" * 80 + "\n")
            f.write("[Error Category Summary]\n")
            f.write("-" * 40 + "\n\n")
            
            sorted_categories = sorted(
                stats['error_categories'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for error_type, count in sorted_categories:
                percentage = count / stats['total_bug_candidates'] * 100 if stats['total_bug_candidates'] > 0 else 0
                f.write(f"  {error_type}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
        
        # Operators with issues
        if stats['operators_with_bugs']:
            f.write("=" * 80 + "\n")
            f.write("[Operators with Potential Issues]\n")
            f.write("-" * 40 + "\n\n")
            
            # Sort by issue count
            sorted_operators = sorted(
                stats['operators_with_bugs'],
                key=lambda x: x['bug_count'],
                reverse=True
            )
            
            for idx, op in enumerate(sorted_operators, 1):
                f.write(f"{idx}. {op['operator']}\n")
                f.write(f"   PyTorch API: {op['torch_api']}\n")
                f.write(f"   PaddlePaddle API: {op['paddle_api']}\n")
                f.write(f"   Issues found: {op['bug_count']}\n")
                f.write("\n")
        
        # Issue details (limit count to avoid overly long report)
        if stats['bug_details']:
            f.write("=" * 80 + "\n")
            f.write("[Issue Details (first 50)]\n")
            f.write("-" * 40 + "\n\n")
            
            for idx, bug in enumerate(stats['bug_details'][:50], 1):
                f.write(f"Issue {idx}:\n")
                f.write("-" * 60 + "\n")
                f.write(f"Operator: {bug['operator']}\n")
                f.write(f"PyTorch API: {bug['torch_api']}\n")
                f.write(f"PaddlePaddle API: {bug['paddle_api']}\n")
                f.write(f"Error type: {bug['error_type']}\n")
                f.write(f"Mutation strategy: {bug['mutation_strategy']}\n")
                f.write(f"PyTorch status: {'Success' if bug['torch_success'] else 'Failed'}\n")
                f.write(f"PaddlePaddle status: {'Success' if bug['paddle_success'] else 'Failed'}\n")
                
                if bug['torch_error']:
                    f.write(f"PyTorch error: {bug['torch_error']}\n")
                if bug['paddle_error']:
                    f.write(f"PaddlePaddle error: {bug['paddle_error']}\n")
                if bug['comparison_error']:
                    f.write(f"Comparison error: {bug['comparison_error']}\n")
                
                f.write("\nPyTorch test case:\n")
                f.write(json.dumps(bug['torch_test_case'], ensure_ascii=False, indent=2))
                f.write("\n\nPaddlePaddle test case:\n")
                f.write(json.dumps(bug['paddle_test_case'], ensure_ascii=False, indent=2))
                f.write("\n\n")
            
            if len(stats['bug_details']) > 50:
                f.write(f"\n... {len(stats['bug_details']) - 50} more issues not listed\n\n")
        else:
            f.write("=" * 80 + "\n")
            f.write("[Issue Details]\n")
            f.write("-" * 40 + "\n")
            f.write("No potential issues found.\n\n")
        
        # Suggestions
        f.write("=" * 80 + "\n")
        f.write("[Analysis Suggestions]\n")
        f.write("-" * 40 + "\n")
        f.write("1. Value mismatch: check for floating-point precision errors (< 1e-5 is often acceptable)\n")
        f.write("2. Shape mismatch: check differences in how frameworks handle input dimensions\n")
        f.write("3. Execution failure: check API parameter compatibility and dtype support\n")
        f.write("4. NaN/Inf: check boundary value handling differences\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Report generation completed\n")
        f.write("=" * 80 + "\n")


def generate_summary_json(stats: Dict[str, Any], output_file: str) -> None:
    """
    Generate JSON summary statistics.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_operators": stats["total_operators"],
        "total_cases": stats["total_cases"],
        "total_fuzzing_rounds": stats["total_fuzzing_rounds"],
        "success_rounds": stats["success_rounds"],
        "failed_rounds": stats["failed_rounds"],
        "total_bug_candidates": stats["total_bug_candidates"],
        "operators_with_bugs_count": len(stats["operators_with_bugs"]),
        "error_categories": stats["error_categories"],
        "operators_with_bugs": stats["operators_with_bugs"],
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"Summary JSON saved to: {output_file}")


def main():
    """
    Program entry point.
    """
    result_dir = Path(__file__).parent / "result"
    report_file = Path(__file__).parent / "fuzzing_analysis_report.txt"
    summary_file = Path(__file__).parent / "fuzzing_analysis_summary.json"
    
    print("=" * 60)
    print("PyTorch-PaddlePaddle fuzzing result analysis")
    print("=" * 60)
    print(f"Result directory: {result_dir}")
    
    # Check if result directory exists
    if not result_dir.exists():
        print(f"\n[WARN] Result directory does not exist: {result_dir}")
        print("Run llm_fuzzing_diff_test.py first to generate results")
        return
    
    print("\nAnalyzing fuzzing results...")
    
    stats = analyze_fuzzing_results(str(result_dir))
    
    # Print summary
    print(f"\nSummary:")
    print(f"  - Total operators: {stats['total_operators']}")
    print(f"  - Total cases: {stats['total_cases']}")
    print(f"  - Total fuzzing rounds: {stats['total_fuzzing_rounds']}")
    print(f"  - Successful rounds: {stats['success_rounds']}")
    print(f"  - Failed rounds: {stats['failed_rounds']}")
    print(f"  - Potential issues found: {stats['total_bug_candidates']}")
    print(f"  - Operators with issues: {len(stats['operators_with_bugs'])}")
    
    if stats['error_categories']:
        print(f"\nError categories:")
        for error_type, count in sorted(stats['error_categories'].items(), key=lambda x: x[1], reverse=True):
            print(f"    - {error_type}: {count}")
    
    # Generate report
    generate_analysis_report(stats, str(report_file))
    print(f"\nAnalysis report generated: {report_file}")
    
    # Generate JSON summary
    generate_summary_json(stats, str(summary_file))
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
