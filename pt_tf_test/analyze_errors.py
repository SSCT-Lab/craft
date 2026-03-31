"""  
PyTorch-TensorFlow Test error analysis tools  Function description:
    1. Scan all test result JSON files in the specified directory
    2. Statistics type three error：torch_error、tensorflow_error、comparison_error
    3. Summarize the number of errors and corresponding number of iterations by file
    4. Generate detailed error analysis reports
"""

import json
from pathlib import Path

def analyze_json_files(log_dir):
    """
    Analyze all JSON test result files in the specified directory and extract error statistics          parameters:
        log_dir (str): The path to the directory where the log file is located          Return:
        tuple: A tuple containing the following four elements
            - total_torch_errors (int): torch_error The total number of non-null values
            - total_tensorflow_errors (int): tensorflow_error The total number of non-null values
            - total_comparison_errors (int): comparison_error The total number of non-null values
            - files_with_errors (list): List of file details containing errors          Exception handling:
        - When a file parsing error is encountered, an error message will be printed but the overall analysis will not be interrupted.
    """
    """
    Analyze all JSON test result files in the specified directory and extract error statistics          parameters:
        log_dir (str): The path to the directory where the log file is located          Return:
        tuple: A tuple containing the following five elements
            - total_torch_errors (int): Total number of examples reporting errors in PyTorch only
            - total_tensorflow_errors (int): Total number of examples with TensorFlow errors only
            - total_both_errors (int): The total number of examples in which both frameworks reported errors
            - total_comparison_errors (int): comparison_error The total number of non-null values
            - files_with_errors (list): List of file details containing errors          Exception handling:
        - When a file parsing error is encountered, an error message will be printed but the overall analysis will not be interrupted.
    """
    # Initialize global error counter
    total_torch_errors = 0           # Total number of PyTorch execution errors only
    total_tensorflow_errors = 0      # Total number of TensorFlow execution errors only
    total_both_errors = 0            # The total number of errors reported by both frameworks
    total_comparison_errors = 0      # Total number of errors compared to results
    files_with_errors = []           # Store file details with errors
    
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
            
            # Initialize the error counter for the current file
            file_torch_errors = 0            # PyTorch error count only
            file_tensorflow_errors = 0       # TensorFlow error count only
            file_both_errors = 0             # Both frames have wrong numbers
            file_comparison_errors = 0       # Number of comparison errors
            
            # Record the number of iterations corresponding to each error type (used to track which test case has an error）
            error_iterations = {
                "torch_error": [],           # PyTorch error iteration only
                "tensorflow_error": [],      # TensorFlow error only iteration
                "both_error": [],            # Wrong iteration in both frameworks
                "comparison_error": []       # Compare the wrong iterations
            }
            
            # Check whether the results field is included in the JSON data (test results array）
            if "results" in data:
                # Iterate through each test result
                for result in data["results"]:
                    # Check whether the execution result field is included
                    if "execution_result" in result:
                        exec_result = result["execution_result"]
                        iteration = result.get("iteration", "N/A")  # Get the number of iterations, the default value is "N/A"
                        
                        # Check execution error type
                        has_torch_error = exec_result.get("torch_error") is not None
                        has_tensorflow_error = exec_result.get("tensorflow_error") is not None
                        
                        # Distinguish between three cases: PyTorch only error, TensorFlow only error, both
                        if has_torch_error and has_tensorflow_error:
                            # Both frameworks reported errors
                            file_both_errors += 1
                            total_both_errors += 1
                            error_iterations["both_error"].append(iteration)
                        elif has_torch_error:
                            # Only PyTorch reports an error
                            file_torch_errors += 1
                            total_torch_errors += 1
                            error_iterations["torch_error"].append(iteration)
                        elif has_tensorflow_error:
                            # Only TensorFlow reports an error
                            file_tensorflow_errors += 1
                            total_tensorflow_errors += 1
                            error_iterations["tensorflow_error"].append(iteration)
                        
                        # Check result comparison error (output inconsistent）
                        if exec_result.get("comparison_error") is not None:
                            file_comparison_errors += 1
                            total_comparison_errors += 1
                            error_iterations["comparison_error"].append(iteration)
            
            # If the current file contains errors of any kind, log details
            if file_torch_errors > 0 or file_tensorflow_errors > 0 or file_both_errors > 0 or file_comparison_errors > 0:
                files_with_errors.append(
                    {
                        "filename": json_file.name,                          # file name
                        "torch_errors": file_torch_errors,                   # PyTorch error count only
                        "tensorflow_errors": file_tensorflow_errors,         # TensorFlow error count only
                        "both_errors": file_both_errors,                     # Both frames have wrong numbers
                        "comparison_errors": file_comparison_errors,         # Number of comparison errors
                        "torch_error_iterations": error_iterations["torch_error"],           # PyTorch only wrong iteration count list
                        "tensorflow_error_iterations": error_iterations["tensorflow_error"],  # TensorFlow only error list of iterations
                        "both_error_iterations": error_iterations["both_error"],             # Wrong list of iterations for both frameworks
                        "comparison_error_iterations": error_iterations["comparison_error"],  # List of iterations comparing errors
                    }
                )
        except Exception as e:
            # Exception handling: print errors without interrupting the overall analysis process
            print(f"Process files {json_file.name} error: {e}")
    
    # Return statistical results
    return total_torch_errors, total_tensorflow_errors, total_both_errors, total_comparison_errors, files_with_errors

def generate_report(output_file, total_torch, total_tensorflow, total_both, total_comparison, files_with_errors):
    """
    Generate a formatted error analysis report and write it to a text file          parameters:
        output_file (str): Path to output report file
        total_torch (int): Total number of PyTorch errors only
        total_tensorflow (int): Total number of TensorFlow errors only
        total_both (int): The total number of errors reported by both frameworks
        total_comparison (int): comparison_error total
        files_with_errors (list): List of file details containing errors          Output format:
        - Overall error statistics (total number of four types of errors)         - Detailed error file information (number of errors per file and corresponding number of iterations）
    """
    # Open the output file for writing in UTF-8 encoding
    with open(output_file, "w", encoding="utf-8") as f:
        # Write report header
        f.write("=" * 80 + "\n")
        f.write("PyTorch-TensorFlow Test error analysis report\n")
        f.write("=" * 80 + "\n\n")
        
        # Write overall error statistics
        f.write("【Overall error statistics】\n")
        f.write(f"Number of examples of errors reported by PyTorch only: {total_torch}\n")
        f.write(f"Only the number of examples of TensorFlow error reports: {total_tensorflow}\n")
        f.write(f"Number of examples where both frameworks reported errors: {total_both}\n")
        f.write(f"comparison_error Total number of non-null values: {total_comparison}\n")
        f.write(f"Total number of files with errors: {len(files_with_errors)}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Write detailed error file information
        if files_with_errors:
            f.write("【Detailed error file information】\n\n")
            # Iterate through all files containing errors
            for idx, file_info in enumerate(files_with_errors, 1):
                f.write(f"{idx}. file name: {file_info['filename']}\n")
                f.write("-" * 80 + "\n")
                
                # If there is a PyTorch-only error, write details
                if file_info["torch_errors"] > 0:
                    f.write(f"   Number of examples of errors reported by PyTorch only: {file_info['torch_errors']}\n")
                    f.write(f"   Iteration value for the application case: {', '.join(map(str, file_info['torch_error_iterations']))}\n")
                
                # If there is a TensorFlow-only error, write details
                if file_info["tensorflow_errors"] > 0:
                    f.write(f"   Only the number of examples of TensorFlow error reports: {file_info['tensorflow_errors']}\n")
                    f.write(f"   Iteration value for the application case: {', '.join(map(str, file_info['tensorflow_error_iterations']))}\n")
                
                # If there are errors in both frames, write details
                if file_info["both_errors"] > 0:
                    f.write(f"   Number of examples where both frameworks reported errors: {file_info['both_errors']}\n")
                    f.write(f"   Iteration value for the application case: {', '.join(map(str, file_info['both_error_iterations']))}\n")
                
                # If there is a comparison error, write details
                if file_info["comparison_errors"] > 0:
                    f.write(f"   comparison_error Number of non-null values: {file_info['comparison_errors']}\n")
                    f.write(f"   Iteration value for the application case: {', '.join(map(str, file_info['comparison_error_iterations']))}\n")
                
                f.write("\n")
        else:
            # No errors found
            f.write("【Detailed error file information】\n\n")
            f.write("No files found with errors。\n\n")
        
        # Write to the end of the report
        f.write("=" * 80 + "\n")
        f.write("Report generation completed\n")
        f.write("=" * 80 + "\n")

def main():
    """
    Main program entry: perform error analysis and generate reports          Execution process:
        1. Specify log directory and output file path
        2. Call analyze_json_files() to analyze all JSON files
        3. Print statistics summary to the console
        4. Call generate_report() to generate a detailed report file
    """
    """
    Main program entry: perform error analysis and generate reports          Execution process:
        1. Specify log directory and output file path
        2. Call analyze_json_files() to analyze all JSON files
        3. Print statistics summary to the console
        4. Call generate_report() to generate a detailed report file
    """
    # Configure input and output paths
    # log_dir = r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log"          # JSON Log file directory
    log_dir = r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1" 
    output_file = r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1\error_analysis_report_new.txt"  # Output report file
    
    # Start analysis
    print("Start parsing the JSON file...")
    total_torch, total_tensorflow, total_both, total_comparison, files_with_errors = analyze_json_files(log_dir)
    
    # Print statistics summary to the console
    print("\nAnalysis completed！")
    print("Total error statistics:")
    print(f"  - Only PyTorch reports an error: {total_torch}")
    print(f"  - Only TensorFlow reports an error: {total_tensorflow}")
    print(f"  - Both frameworks reported errors: {total_both}")
    print(f"  - comparison_error: {total_comparison}")
    print(f"  - Number of files containing errors: {len(files_with_errors)}")
    
    # Generate detailed report files
    generate_report(output_file, total_torch, total_tensorflow, total_both, total_comparison, files_with_errors)
    print(f"\nReport generated: {output_file}")

if __name__ == "__main__":
    main()
