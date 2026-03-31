"""  
PyTorch-TensorFlow Successful test case extraction tool (version classified by operator)  Function description:
    1. Scan all test result JSON files in the specified directory
    2. Extract all test cases without any errors
    3. Classified by operator name, each operator generates an independent JSON file
    4. Each example contains the complete torch_test_case, tensorflow_test_case and execution_result
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


def extract_operator_name(filename: str) -> str:
    """
    Extract operator name from file name          Example: llm_enhanced_torch_abs_20260123_191052.json -> torch_abs
    
    parameter:
        filename (str): JSON file name          Return:
        str: Operator name
    """
    # Matches the part after llm_enhanced_ to the part before _date
    # Date format is 8 digits（YYYYMMDD）
    pattern = r'llm_enhanced_(.+?)_(\d{8})_'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    
    # Alternative solution: remove prefixes and suffixes
    name = filename.replace('llm_enhanced_', '').replace('.json', '')
    # Try removing the date and time suffix
    parts = name.rsplit('_', 2)
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        return '_'.join(parts[:-2])
    return name


def analyze_and_extract_success_cases(log_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze all JSON test result files in the specified directory and extract successful use cases according to operators          parameters:
        log_dir (str): The path to the directory where the log file is located          Return:
        Dict[str, List[Dict]]: Operator name -> Mapping of successful use case lists
    """
    # Dictionary of successful use cases by operator
    operator_success_cases: Dict[str, List[Dict[str, Any]]] = {}
    
    # Statistics
    total_files = 0
    total_cases = 0
    total_success = 0
    
    # Get the log directory path and find all matching JSON files
    log_path = Path(log_dir)
    json_files = sorted(log_path.glob("llm_enhanced_torch*.json"))
    print(f"turn up {len(json_files)} JSON files")
    
    # Iterate through all JSON files for analysis
    for json_file in json_files:
        total_files += 1
        try:
            # Read JSON file content
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract operator name
            operator_name = extract_operator_name(json_file.name)
            
            # If the operator has not been recorded yet, initialize the list
            if operator_name not in operator_success_cases:
                operator_success_cases[operator_name] = []
            
            # Check if JSON data contains results field
            if "results" not in data:
                continue
                
            # Iterate through each test result
            for result in data["results"]:
                total_cases += 1
                
                # Check whether the execution result field is included
                if "execution_result" not in result:
                    continue
                    
                exec_result = result["execution_result"]
                
                # Check for complete success (all three errors are null and the result matches）
                torch_error = exec_result.get("torch_error")
                tensorflow_error = exec_result.get("tensorflow_error")
                comparison_error = exec_result.get("comparison_error")
                results_match = exec_result.get("results_match", False)
                
                # Conditions for complete success: no errors and matching results
                if (torch_error is None and 
                    tensorflow_error is None and 
                    comparison_error is None and
                    results_match):
                    
                    total_success += 1
                    
                    # Build complete information about successful use cases
                    success_case = {
                        "source_file": json_file.name,
                        "operator": data.get("operator", operator_name),
                        "iteration": result.get("iteration", "N/A"),
                        "case_number": result.get("case_number", "N/A"),
                        "is_llm_generated": result.get("is_llm_generated", False),
                        "torch_test_case": result.get("torch_test_case", {}),
                        "tensorflow_test_case": result.get("tensorflow_test_case", {}),
                        "execution_result": exec_result
                    }
                    
                    # If there is llm_operation information, it is also retained
                    if "llm_operation" in result:
                        success_case["llm_operation"] = result["llm_operation"]
                    
                    operator_success_cases[operator_name].append(success_case)
                    
        except Exception as e:
            print(f"Process files {json_file.name} error: {e}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  - Number of files processed: {total_files}")
    print(f"  - Total number of use cases: {total_cases}")
    print(f"  - Number of successful use cases: {total_success}")
    print(f"  - success rate: {total_success / total_cases * 100:.2f}%" if total_cases > 0 else "  - success rate: N/A")
    print(f"  - Number of operators involved: {len(operator_success_cases)}")
    
    return operator_success_cases


def save_operator_json_files(
    operator_cases: Dict[str, List[Dict[str, Any]]], 
    output_dir: str
) -> None:
    """
    Save successful use cases classified by operator as separate JSON files          parameters:
        operator_cases (Dict): Operator name -> Mapping of successful use case lists
        output_dir (str): output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for operator_name, cases in operator_cases.items():
        if not cases:
            continue
            
        # Build output file path
        output_file = output_path / f"{operator_name}_success_cases.json"
        
        # Build output data structure
        output_data = {
            "operator": operator_name,
            "export_time": datetime.now().isoformat(),
            "total_success_cases": len(cases),
            "success_cases": cases
        }
        
        # Write to JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        saved_count += 1
    
    print(f"\nsaved {saved_count} Successful use case files of operators arrive: {output_dir}")


def generate_summary_report(
    operator_cases: Dict[str, List[Dict[str, Any]]], 
    output_file: str
) -> None:
    """
    Generate summary report          parameters:
        operator_cases (Dict): Operator name -> Mapping of successful use case lists
        output_file (str): Output report file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by number of successful use cases
    sorted_operators = sorted(
        operator_cases.items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    )
    
    total_cases = sum(len(cases) for cases in operator_cases.values())
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch-TensorFlow Summary report of successful test cases (classified by operator）\n")
        f.write(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【Overall statistics】\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of operators involved: {len(operator_cases)}\n")
        f.write(f"Total number of successful use cases: {total_cases}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        f.write("【Statistics of successful use cases of each operator】\n")
        f.write("-" * 40 + "\n\n")
        
        for idx, (operator_name, cases) in enumerate(sorted_operators, 1):
            if not cases:
                continue
            
            # Count the number of use cases generated by LLM
            llm_generated = sum(1 for c in cases if c.get("is_llm_generated", False))
            original = len(cases) - llm_generated
            
            f.write(f"{idx}. {operator_name}\n")
            f.write(f"   Number of successful use cases: {len(cases)} (original: {original}, LLMgenerate: {llm_generated})\n")
            
            # Count different shapes and dtype
            shapes = set()
            dtypes = set()
            for case in cases:
                input_info = case.get("torch_test_case", {}).get("input", {})
                if isinstance(input_info, dict):
                    shape = input_info.get("shape", [])
                    dtype = input_info.get("dtype", "")
                    if shape:
                        shapes.add(str(shape))
                    if dtype:
                        dtypes.add(dtype)
            
            if shapes:
                f.write(f"   test shapes: {', '.join(list(shapes)[:5])}" + 
                       (" ..." if len(shapes) > 5 else "") + "\n")
            if dtypes:
                f.write(f"   test dtypes: {', '.join(dtypes)}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Report generation completed\n")
        f.write("=" * 80 + "\n")


def main():
    """
    Main program entrance
    """
    # Configuration path
    log_dir = r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1"
    output_dir = r"d:\graduate\DFrameworkTest\pt_tf_test\fuzzing\success_cases"
    report_file = r"d:\graduate\DFrameworkTest\pt_tf_test\fuzzing\success_cases_summary.txt"
    
    print("=" * 60)
    print("Start extracting successful test cases and classify them by operators...")
    print("=" * 60)
    
    # Analyze and extract successful use cases
    operator_cases = analyze_and_extract_success_cases(log_dir)
    
    # Save the JSON file of each operator
    save_operator_json_files(operator_cases, output_dir)
    
    # Generate summary report
    generate_summary_report(operator_cases, report_file)
    print(f"Summary report generated: {report_file}")
    
    print("\n" + "=" * 60)
    print("Extraction completed！")
    print("=" * 60)


if __name__ == "__main__":
    main()
