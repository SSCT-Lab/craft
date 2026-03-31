"""  
PyTorch-PaddlePaddle successful test case extractor (grouped by operator).

Features:
    1. Scan all JSON test result files under the specified directory
    2. Extract test cases with no errors
    3. Group by operator name and generate a JSON file per operator
    4. Each sample includes full torch_test_case, paddle_test_case, and execution_result
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


def extract_operator_name(filename: str) -> str:
    """
    Extract operator name from filename.
    
    Example: llm_enhanced_torch_abs_20260125_022445.json -> torch_abs
    
    Args:
        filename (str): JSON filename
    
    Returns:
        str: Operator name
    """
    # Match between llm_enhanced_ and _date
    # Date format: 8 digits (YYYYMMDD)
    pattern = r'llm_enhanced_(.+?)_(\d{8})_'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    
    # Fallback: strip prefix and suffix
    name = filename.replace('llm_enhanced_', '').replace('.json', '')
    # Try removing datetime suffix
    parts = name.rsplit('_', 2)
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        return '_'.join(parts[:-2])
    return name


def analyze_and_extract_success_cases(log_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze all JSON test result files under a directory and extract success cases by operator.
    
    Args:
        log_dir (str): Log directory path
    
    Returns:
        Dict[str, List[Dict]]: operator name -> list of success cases
    """
    # Success cases by operator
    operator_success_cases: Dict[str, List[Dict[str, Any]]] = {}
    
    # Stats
    total_files = 0
    total_cases = 0
    total_success = 0
    
    # Resolve log path and find matching JSON files
    log_path = Path(log_dir)
    json_files = sorted(log_path.glob("llm_enhanced_torch*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    # Analyze all JSON files
    for json_file in json_files:
        total_files += 1
        try:
            # Read JSON file
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract operator name
            operator_name = extract_operator_name(json_file.name)
            
            # Initialize list if operator not seen
            if operator_name not in operator_success_cases:
                operator_success_cases[operator_name] = []
            
            # Check results field
            if "results" not in data:
                continue
                
            # Traverse each test result
            for result in data["results"]:
                total_cases += 1
                
                # Check execution_result field
                if "execution_result" not in result:
                    continue
                    
                exec_result = result["execution_result"]
                
                # Check full success (no errors and results_match)
                torch_error = exec_result.get("torch_error")
                paddle_error = exec_result.get("paddle_error")
                comparison_error = exec_result.get("comparison_error")
                results_match = exec_result.get("results_match", False)
                
                # Success condition: no errors and match
                if (torch_error is None and 
                    paddle_error is None and 
                    comparison_error is None and
                    results_match):
                    
                    total_success += 1
                    
                    # Build full success case info
                    success_case = {
                        "source_file": json_file.name,
                        "operator": data.get("operator", operator_name),
                        "iteration": result.get("iteration", "N/A"),
                        "case_number": result.get("case_number", "N/A"),
                        "is_llm_generated": result.get("is_llm_generated", False),
                        "torch_test_case": result.get("torch_test_case", {}),
                        "paddle_test_case": result.get("paddle_test_case", {}),
                        "execution_result": exec_result
                    }
                    
                    # Keep llm_operation if present
                    if "llm_operation" in result:
                        success_case["llm_operation"] = result["llm_operation"]
                    
                    operator_success_cases[operator_name].append(success_case)
                    
        except Exception as e:
            print(f"Error processing file {json_file.name}: {e}")
    
    # Print stats
    print(f"\nStatistics:")
    print(f"  - Files processed: {total_files}")
    print(f"  - Total cases: {total_cases}")
    print(f"  - Successful cases: {total_success}")
    print(f"  - Success rate: {total_success / total_cases * 100:.2f}%" if total_cases > 0 else "  - Success rate: N/A")
    print(f"  - Operators involved: {len(operator_success_cases)}")
    
    return operator_success_cases


def save_operator_json_files(
    operator_cases: Dict[str, List[Dict[str, Any]]], 
    output_dir: str
) -> None:
    """
    Save success cases grouped by operator as separate JSON files.
    
    Args:
        operator_cases (Dict): operator name -> list of success cases
        output_dir (str): Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for operator_name, cases in operator_cases.items():
        if not cases:
            continue
            
        # Build output file path
        output_file = output_path / f"{operator_name}_success_cases.json"
        
        # Build output data
        output_data = {
            "operator": operator_name,
            "export_time": datetime.now().isoformat(),
            "total_success_cases": len(cases),
            "success_cases": cases
        }
        
        # Write JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        saved_count += 1
    
    print(f"\nSaved success case files for {saved_count} operators to: {output_dir}")


def generate_summary_report(
    operator_cases: Dict[str, List[Dict[str, Any]]], 
    output_file: str
) -> None:
    """
    Generate summary report.
    
    Args:
        operator_cases (Dict): operator name -> list of success cases
        output_file (str): Output report file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by success case count
    sorted_operators = sorted(
        operator_cases.items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    )
    
    total_success = sum(len(cases) for cases in operator_cases.values())
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch-PaddlePaddle Success Case Extraction Report\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"[Summary]\n")
        f.write(f"  - Operators involved: {len(operator_cases)}\n")
        f.write(f"  - Total success cases: {total_success}\n")
        f.write(f"  - Avg cases per operator: {total_success / len(operator_cases):.2f}\n" if operator_cases else "")
        f.write("\n")
        
        f.write("[Success cases by operator]\n")
        f.write("-" * 60 + "\n")
        for idx, (op_name, cases) in enumerate(sorted_operators, 1):
            f.write(f"{idx:3d}. {op_name}: {len(cases)} cases\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Report generation completed\n")
        f.write("=" * 80 + "\n")
    
    print(f"Report saved to: {output_file}")


def save_all_success_cases(
    operator_cases: Dict[str, List[Dict[str, Any]]], 
    output_file: str
) -> None:
    """
    Save all success cases into a single JSON file.
    
    Args:
        operator_cases (Dict): operator name -> list of success cases
        output_file (str): Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_success = sum(len(cases) for cases in operator_cases.values())
    
    output_data = {
        "export_time": datetime.now().isoformat(),
        "total_operators": len(operator_cases),
        "total_success_cases": total_success,
        "operators": operator_cases
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"All success cases saved to: {output_file}")


def generate_summary_txt(
    operator_cases: Dict[str, List[Dict[str, Any]]], 
    output_file: str,
    total_cases: int = 0,
    total_success: int = 0
) -> None:
    """
    Generate a concise summary.
    
    Args:
        operator_cases (Dict): operator name -> list of success cases
        output_file (str): Output file path
        total_cases (int): Total cases
        total_success (int): Successful cases
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_success_count = sum(len(cases) for cases in operator_cases.values())
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("PyTorch-PaddlePaddle Success Case Summary\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Operators involved: {len(operator_cases)}\n")
        f.write(f"Successful cases: {total_success_count}\n")
        if total_cases > 0:
            f.write(f"Success rate: {total_success_count / total_cases * 100:.2f}%\n")
    
    print(f"Summary saved to: {output_file}")


def main():
    """
    Program entry point.
    """
    # Set paths
    script_dir = Path(__file__).parent
    log_dir = script_dir.parent / "pt_pd_log_1"
    success_cases_dir = script_dir / "success_cases"
    
    print("=" * 60)
    print("PyTorch-PaddlePaddle success case extractor")
    print("=" * 60)
    print(f"Log directory: {log_dir}")
    print(f"Output directory: {success_cases_dir}")
    print()
    
    # Check log directory
    if not log_dir.exists():
        print(f"[ERROR] Log directory does not exist: {log_dir}")
        return
    
    # Analyze and extract success cases
    operator_cases = analyze_and_extract_success_cases(str(log_dir))
    
    if not operator_cases:
        print("\n[WARN] No success cases extracted")
        return
    
    # Save per-operator JSON files
    save_operator_json_files(operator_cases, str(success_cases_dir))
    
    # Save all success cases to a single file
    save_all_success_cases(
        operator_cases, 
        str(script_dir / "success_cases_data.json")
    )
    
    # Generate report
    generate_summary_report(
        operator_cases, 
        str(script_dir / "success_cases_report.txt")
    )
    
    # Generate concise summary
    total_success = sum(len(cases) for cases in operator_cases.values())
    generate_summary_txt(
        operator_cases,
        str(script_dir / "success_cases_summary.txt"),
        total_cases=0,  # Computed in analyze_and_extract_success_cases
        total_success=total_success
    )
    
    print("\n" + "=" * 60)
    print("Extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
