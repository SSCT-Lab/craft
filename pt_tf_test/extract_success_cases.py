"""  
PyTorch-TensorFlow Successful test case extraction tool  Function description:
    1. Scan all test result JSON files in the specified directory
    2. Extract all test cases without any errors (torch_error, tensorflow_error, comparison_error are null）
    3. Summarize the number of successful use cases and corresponding number of iterations by file
    4. Generate detailed successful use case analysis reports
"""

import json
from pathlib import Path
from datetime import datetime


def analyze_success_cases(log_dir: str) -> tuple:
    """
    Analyze all JSON test result files in the specified directory and extract successful use case statistics          parameters:
        log_dir (str): The path to the directory where the log file is located          Return:
        tuple: A tuple containing the following elements
            - total_success_cases (int): Total number of fully successful test cases
            - total_cases (int): The total number of all test cases
            - files_all_success (list): List of files for which all use cases were successful
            - files_with_success (list): List of file details containing successful use cases
            - all_success_details (list): Details of all successful use cases          Exception handling:
        - When a file parsing error is encountered, an error message will be printed but the overall analysis will not be interrupted.
    """
    # Initialize global counter
    total_success_cases = 0      # Total number of fully successful use cases
    total_cases = 0              # Total number of all use cases
    files_all_success = []       # Documentation where all use cases were successful
    files_with_success = []      # Contains file details for successful use cases
    all_success_details = []     # Details of all successful use cases
    
    # Get the log directory path and find all matching JSON files
    log_path = Path(log_dir)
    json_files = sorted(log_path.glob("llm_enhanced_torch*.json"))  # Sort by file name
    print(f"turn up {len(json_files)} JSON files")
    
    # Iterate through all JSON files for analysis
    for json_file in json_files:
        try:
            # Read JSON file content
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Initialize the counter for the current file
            file_success_cases = 0       # Number of successful use cases for the current file
            file_total_cases = 0         # The total number of use cases in the current file
            success_iterations = []      # List of iterations for successful use cases
            success_case_details = []    # Details of successful use cases
            
            # Get operator name
            operator_name = data.get("operator", "unknown")
            
            # Check if JSON data contains results field
            if "results" in data:
                # Iterate through each test result
                for result in data["results"]:
                    file_total_cases += 1
                    total_cases += 1
                    
                    # Check whether the execution result field is included
                    if "execution_result" in result:
                        exec_result = result["execution_result"]
                        iteration = result.get("iteration", "N/A")
                        case_number = result.get("case_number", "N/A")
                        is_llm_generated = result.get("is_llm_generated", False)
                        
                        # The check is completely successful (all three errors are null）
                        torch_error = exec_result.get("torch_error")
                        tensorflow_error = exec_result.get("tensorflow_error")
                        comparison_error = exec_result.get("comparison_error")
                        results_match = exec_result.get("results_match", False)
                        
                        # Conditions for complete success: no errors and matching results
                        if (torch_error is None and 
                            tensorflow_error is None and 
                            comparison_error is None and
                            results_match):
                            
                            file_success_cases += 1
                            total_success_cases += 1
                            success_iterations.append(iteration)
                            
                            # Record details of successful use cases
                            case_detail = {
                                "filename": json_file.name,
                                "operator": operator_name,
                                "iteration": iteration,
                                "case_number": case_number,
                                "is_llm_generated": is_llm_generated,
                                "torch_shape": exec_result.get("torch_shape", []),
                                "torch_dtype": exec_result.get("torch_dtype", ""),
                                "tensorflow_shape": exec_result.get("tensorflow_shape", []),
                                "tensorflow_dtype": exec_result.get("tensorflow_dtype", ""),
                                "input_info": result.get("torch_test_case", {}).get("input", {})
                            }
                            success_case_details.append(case_detail)
                            all_success_details.append(case_detail)
            
            # Record statistics of the current file
            if file_success_cases > 0:
                file_info = {
                    "filename": json_file.name,
                    "operator": operator_name,
                    "success_cases": file_success_cases,
                    "total_cases": file_total_cases,
                    "success_rate": f"{file_success_cases / file_total_cases * 100:.1f}%" if file_total_cases > 0 else "0%",
                    "success_iterations": success_iterations,
                    "success_details": success_case_details
                }
                files_with_success.append(file_info)
                
                # If all use cases are successful, add the full success file list
                if file_success_cases == file_total_cases:
                    files_all_success.append(file_info)
                    
        except Exception as e:
            # Exception handling: print errors without interrupting the overall analysis process
            print(f"Process files {json_file.name} error: {e}")
    
    return total_success_cases, total_cases, files_all_success, files_with_success, all_success_details


def generate_report(output_file: str, 
                    total_success: int, 
                    total_cases: int,
                    files_all_success: list,
                    files_with_success: list,
                    all_success_details: list) -> None:
    """
    Generate a formatted successful use case analysis report and write it to a text file          parameters:
        output_file (str): Path to output report file
        total_success (int): Total number of successful use cases
        total_cases (int): Total number of use cases
        files_all_success (list): List of files for which all use cases were successful
        files_with_success (list): List of file details containing successful use cases
        all_success_details (list): Details of all successful use cases
    """
    # Make sure the output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        # Write report header
        f.write("=" * 80 + "\n")
        f.write("PyTorch-TensorFlow Successful test case analysis report\n")
        f.write(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write overall statistics
        f.write("【Overall statistics】\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total number of successful use cases: {total_success}\n")
        f.write(f"Total number of use cases: {total_cases}\n")
        success_rate = total_success / total_cases * 100 if total_cases > 0 else 0
        f.write(f"overall success rate: {success_rate:.2f}%\n")
        f.write(f"Number of files containing successful use cases: {len(files_with_success)}\n")
        f.write(f"Number of files for which all use cases were successful: {len(files_all_success)}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Writing to a list of files where all use cases were successful
        f.write("【Documentation where all use cases were successful】\n")
        f.write("-" * 40 + "\n")
        if files_all_success:
            for idx, file_info in enumerate(files_all_success, 1):
                f.write(f"{idx}. {file_info['filename']}\n")
                f.write(f"   operator: {file_info['operator']}\n")
                f.write(f"   Number of successful use cases: {file_info['success_cases']}/{file_info['total_cases']}\n")
                f.write(f"   Number of iterations: {', '.join(map(str, file_info['success_iterations']))}\n")
                f.write("\n")
        else:
            f.write("No file found with all use cases successful。\n\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Write details to file containing success case
        f.write("【Contains file details for successful use cases】\n")
        f.write("-" * 40 + "\n\n")
        if files_with_success:
            for idx, file_info in enumerate(files_with_success, 1):
                f.write(f"{idx}. file name: {file_info['filename']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"   operator: {file_info['operator']}\n")
                f.write(f"   Number of successful use cases: {file_info['success_cases']}/{file_info['total_cases']} ({file_info['success_rate']})\n")
                f.write(f"   iteration value for successful use cases: {', '.join(map(str, file_info['success_iterations']))}\n")
                
                # Output brief information for each successful use case
                f.write("   Successful use case details:\n")
                for detail in file_info['success_details']:
                    input_info = detail.get('input_info', {})
                    # Handle cases where input_info may be a list or dictionary
                    if isinstance(input_info, dict):
                        shape = input_info.get('shape', [])
                        dtype = input_info.get('dtype', 'unknown')
                    elif isinstance(input_info, list) and len(input_info) > 0:
                        # If it is a list, take the first element
                        first_input = input_info[0] if isinstance(input_info[0], dict) else {}
                        shape = first_input.get('shape', [])
                        dtype = first_input.get('dtype', 'unknown')
                    else:
                        shape = []
                        dtype = 'unknown'
                    llm_tag = "[LLMgenerate]" if detail.get('is_llm_generated') else "[Original use case]"
                    f.write(f"      - iteration {detail['iteration']}, case {detail['case_number']}: "
                            f"shape={shape}, dtype={dtype} {llm_tag}\n")
                f.write("\n")
        else:
            f.write("No file containing successful use case found。\n\n")
        
        # Write to the end of the report
        f.write("=" * 80 + "\n")
        f.write("Report generation completed\n")
        f.write("=" * 80 + "\n")


def export_success_cases_json(output_file: str, all_success_details: list) -> None:
    """
    Export details of all successful use cases to a JSON file for easy subsequent processing          parameters:
        output_file (str): Path to output JSON file
        all_success_details (list): Details of all successful use cases
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_data = {
        "export_time": datetime.now().isoformat(),
        "total_success_cases": len(all_success_details),
        "success_cases": all_success_details
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)


def main():
    """
    Main program entry: perform successful use case analysis and generate reports          Execution process:
        1. Specify log directory and output file path
        2. Call analyze_success_cases() to analyze all JSON files
        3. Print statistics summary to the console
        4. Call generate_report() to generate a detailed report file
        5. Call export_success_cases_json() to export success cases JSON
    """
    # Configure input and output paths
    log_dir = r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1"
    output_report = r"d:\graduate\DFrameworkTest\pt_tf_test\fuzzing\success_cases_report.txt"
    output_json = r"d:\graduate\DFrameworkTest\pt_tf_test\fuzzing\success_cases_data.json"
    
    # Start analysis
    print("Start analyzing the JSON file and extract successful use cases...")
    (total_success, total_cases, files_all_success, 
     files_with_success, all_success_details) = analyze_success_cases(log_dir)
    
    # Print statistics summary to the console
    print("\n" + "=" * 50)
    print("Analysis completed！")
    print("=" * 50)
    print(f"Successful use case statistics:")
    print(f"  - Total number of successful use cases: {total_success}")
    print(f"  - Total number of use cases: {total_cases}")
    success_rate = total_success / total_cases * 100 if total_cases > 0 else 0
    print(f"  - overall success rate: {success_rate:.2f}%")
    print(f"  - Number of files containing successful use cases: {len(files_with_success)}")
    print(f"  - Number of files for which all use cases were successful: {len(files_all_success)}")
    
    # Generate detailed report files
    generate_report(output_report, total_success, total_cases, 
                    files_all_success, files_with_success, all_success_details)
    print(f"\nText report generated: {output_report}")
    
    # Export successful use case JSON
    export_success_cases_json(output_json, all_success_details)
    print(f"JSON Data has been exported: {output_json}")


if __name__ == "__main__":
    main()
