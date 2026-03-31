"""  
PyTorch-TensorFlow Fuzzing Results analysis tools  Function description:
    1. Analyze all JSON files in the fuzzing result directory
    2. Potential problems found in statistics
    3. Generate detailed analysis reports
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def analyze_fuzzing_results(result_dir: str) -> Dict[str, Any]:
    """
    Analyze fuzzing results
    """
    result_path = Path(result_dir)
    json_files = sorted(result_path.glob("*_fuzzing_result.json"))
    
    print(f"turn up {len(json_files)} result files")
    
    # Statistics
    stats = {
        "total_operators": 0,
        "total_cases": 0,
        "total_fuzzing_rounds": 0,
        "total_bug_candidates": 0,
        "operators_with_bugs": [],
        "bug_details": [],
        "success_rounds": 0,
        "failed_rounds": 0,
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
                    "tensorflow_api": data.get("tensorflow_api"),
                    "bug_count": bug_count
                })
            
            # Analyze fuzzing results for each use case
            for result in data.get("results", []):
                for fr in result.get("fuzzing_results", []):
                    if fr.get("success"):
                        stats["success_rounds"] += 1
                        
                        if fr.get("is_bug_candidate"):
                            exec_result = fr.get("execution_result", {})
                            bug_detail = {
                                "operator": data.get("operator"),
                                "torch_api": data.get("torch_api"),
                                "tensorflow_api": data.get("tensorflow_api"),
                                "round": fr.get("round"),
                                "mutation_strategy": fr.get("mutation_strategy"),
                                "torch_success": exec_result.get("torch_success"),
                                "tensorflow_success": exec_result.get("tensorflow_success"),
                                "torch_error": exec_result.get("torch_error"),
                                "tensorflow_error": exec_result.get("tensorflow_error"),
                                "comparison_error": exec_result.get("comparison_error"),
                                "torch_test_case": fr.get("torch_test_case"),
                                "tensorflow_test_case": fr.get("tensorflow_test_case"),
                            }
                            stats["bug_details"].append(bug_detail)
                    else:
                        stats["failed_rounds"] += 1
                        
        except Exception as e:
            print(f"deal with {json_file.name} error: {e}")
    
    return stats


def generate_analysis_report(stats: Dict[str, Any], output_file: str) -> None:
    """
    Generate analysis report
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch-TensorFlow Fuzzing Differential test result analysis report\n")
        f.write(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("【Overall statistics】\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of test operators: {stats['total_operators']}\n")
        f.write(f"Number of test cases: {stats['total_cases']}\n")
        f.write(f"Total fuzzing rounds: {stats['total_fuzzing_rounds']}\n")
        f.write(f"successfully executed round: {stats['success_rounds']}\n")
        f.write(f"Execution failed round: {stats['failed_rounds']}\n")
        f.write(f"Number of potential problems found: {stats['total_bug_candidates']}\n")
        f.write(f"Problematic number of operators: {len(stats['operators_with_bugs'])}\n")
        f.write("\n")
        
        # List of problematic operators
        if stats['operators_with_bugs']:
            f.write("=" * 80 + "\n")
            f.write("【Operators with potential problems】\n")
            f.write("-" * 40 + "\n\n")
            
            for idx, op in enumerate(stats['operators_with_bugs'], 1):
                f.write(f"{idx}. {op['operator']}\n")
                f.write(f"   PyTorch API: {op['torch_api']}\n")
                f.write(f"   TensorFlow API: {op['tensorflow_api']}\n")
                f.write(f"   Number of problems found: {op['bug_count']}\n")
                f.write("\n")
        
        # Problem details
        if stats['bug_details']:
            f.write("=" * 80 + "\n")
            f.write("【Problem details】\n")
            f.write("-" * 40 + "\n\n")
            
            for idx, bug in enumerate(stats['bug_details'], 1):
                f.write(f"question {idx}:\n")
                f.write("-" * 60 + "\n")
                f.write(f"operator: {bug['operator']}\n")
                f.write(f"PyTorch API: {bug['torch_api']}\n")
                f.write(f"TensorFlow API: {bug['tensorflow_api']}\n")
                f.write(f"mutation strategy: {bug['mutation_strategy']}\n")
                f.write(f"PyTorch Execution status: {'success' if bug['torch_success'] else 'fail'}\n")
                f.write(f"TensorFlow Execution status: {'success' if bug['tensorflow_success'] else 'fail'}\n")
                
                if bug['torch_error']:
                    f.write(f"PyTorch mistake: {bug['torch_error']}\n")
                if bug['tensorflow_error']:
                    f.write(f"TensorFlow mistake: {bug['tensorflow_error']}\n")
                if bug['comparison_error']:
                    f.write(f"comparison error: {bug['comparison_error']}\n")
                
                f.write("\nPyTorch test case:\n")
                f.write(json.dumps(bug['torch_test_case'], ensure_ascii=False, indent=2))
                f.write("\n\nTensorFlow test case:\n")
                f.write(json.dumps(bug['tensorflow_test_case'], ensure_ascii=False, indent=2))
                f.write("\n\n")
        else:
            f.write("=" * 80 + "\n")
            f.write("【Problem details】\n")
            f.write("-" * 40 + "\n")
            f.write("No potential issues found。\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Report generation completed\n")
        f.write("=" * 80 + "\n")


def main():
    """
    Main program entrance
    """
    result_dir = Path(__file__).parent / "result"
    report_file = Path(__file__).parent / "fuzzing_analysis_report.txt"
    
    print("=" * 60)
    print("Analyze fuzzing results...")
    print("=" * 60)
    
    stats = analyze_fuzzing_results(str(result_dir))
    
    # Print statistical summary
    print(f"\nStatistical summary:")
    print(f"  - Number of test operators: {stats['total_operators']}")
    print(f"  - Number of test cases: {stats['total_cases']}")
    print(f"  - Total fuzzing rounds: {stats['total_fuzzing_rounds']}")
    print(f"  - Number of potential problems found: {stats['total_bug_candidates']}")
    
    # Generate report
    generate_analysis_report(stats, str(report_file))
    print(f"\nAnalysis report has been generated: {report_file}")


if __name__ == "__main__":
    main()
