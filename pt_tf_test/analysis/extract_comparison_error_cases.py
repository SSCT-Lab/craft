"""
comparison_error Sample extraction tool  Function description:
    1. Identify files containing comparison_error from the error analysis report
    2. Extract all test cases with non-null comparison_error from these JSON files
    3. Formatted output sample details (including test code, error information, etc.)）
    4. Generate independent sample report files to facilitate in-depth analysis of the reasons for inconsistent results  Usage scenarios:
    When testing shows that both PyTorch and TensorFlow can execute, but the results are inconsistent,     Use this tool to extract specific test case code for human review and debugging。

"""

import re
import json
from pathlib import Path

def parse_report(report_path):
    """
    Parse the error analysis report and extract the JSON file name list containing comparison_error          parameters:
        report_path (str): The path to the error analysis report file (given by analyze_errors.py generated)          Return:
        list: List of JSON filenames containing comparison_error          parsing logic:
        - Find contains "file name: xxx.json" of rows         - Check if the file part contains "comparison_error Number of non-null values" Field         - The file is considered to contain only if both conditions are met at the same time comparison_error
    """
    """
    Parse the error analysis report and extract the JSON file name list containing comparison_error          parameters:
        report_path (str): The path to the error analysis report file (given by analyze_errors.py generated)          Return:
        list: List of JSON filenames containing comparison_error          parsing logic:
        - Find contains "file name: xxx.json" of rows         - Check if the file part contains "comparison_error Number of non-null values" Field         - The file is considered to contain only if both conditions are met at the same time comparison_error
    """
    files = []               # Store the file name containing comparison_error
    current_file = None      # The name of the file currently being processed
    has_field = False        # Mark whether the current file contains the comparison_error field
    
    # Read report file line by line
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            # Use regular expressions to match filename lines (format："file name: xxx.json"）
            m = re.search(r"file name:\s+([^\s]+\.json)", line)
            if m:
                # If a new file name is found, first check whether the previous file meets the conditions
                if current_file and has_field:
                    files.append(current_file)  # The previous file meets the conditions and is added to the result list
                
                # Update to new filename, reset tags
                current_file = m.group(1).strip()
                has_field = False
                continue
            
            # Check if the current row contains the comparison_error field
            if "comparison_error Number of non-null values" in line:
                has_field = True  # Mark the current file contains comparison_error
        
        # Process the last file (needs to be checked separately after loop ends）
        if current_file and has_field:
            files.append(current_file)
    
    return files

def collect_samples(json_path):
    """
    Collect all examples with non-null comparison_error from the JSON test results file          parameters:
        json_path (str): JSON Path to test results file          Return:
        list: A list of sample information dictionaries containing comparison_error, each dictionary containing：
            - iteration: Number of iterations
            - comparison_error: error message
            - torch_test_case/tensorflow_test_case: Separate test code (if present）
            - test_case: Unified test code (if present）
            - raw_item: Raw data (if none of the above exists)          Data structure adaptation:
        This function can handle multiple JSON data formats, with the following priorities:：
        1. separated torch_test_case and tensorflow_test_case
        2. unified test_case
        3. Complete raw_item (recovery plan）
    """
    results = []  # Store all examples containing comparison_error
    
    # Read JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Iterate through all test results
    for item in data.get("results", []):
        # Get the execution result field (use or {} Avoid None values）
        exec_res = item.get("execution_result") or {}
        # Extract comparison_error field
        val = exec_res.get("comparison_error")
        
        # Only handle examples where comparison_error is not null
        if val is not None:
            # Build basic information
            entry = {
                "iteration": item.get("iteration"),     # Number of iterations (for traceback）
                "comparison_error": val,                # error message
            }
            
            # Extract test case codes based on data format (decreasing priority）
            if "torch_test_case" in item or "tensorflow_test_case" in item:
                # Format 1: Separated PyTorch and TensorFlow test code
                entry["torch_test_case"] = item.get("torch_test_case")
                entry["tensorflow_test_case"] = item.get("tensorflow_test_case")
            elif "test_case" in item:
                # Format 2: Unified test code
                entry["test_case"] = item.get("test_case")
            else:
                # Format 3: Save the complete original data (a cover-up plan）
                entry["raw_item"] = item
            
            results.append(entry)
    
    return results

def format_section(filename, samples):
    """
    Formatted output of all comparison_error examples in a single file          parameters:
        filename (str): JSON file name
        samples (list): collect_samples() Returned sample list          Return:
        str: Formatted text, including file name, sample number, error message and test code          Output format:
        ================================================================================
        document: xxx.json
        --------------------------------------------------------------------------------
        Sample 1:
        comparison_error: <error message>
        torch_test_case: <PyTorch test code JSON>
        tensorflow_test_case: <TensorFlow test code JSON>
        ...
    """
    lines = []  # Store formatted lines of text
    
    # Add file header (using = as delimiter）
    lines.append("=" * 80)
    lines.append(f"document: {filename}")
    lines.append("-" * 80)
    
    # Iterate through all examples in the file
    for idx, s in enumerate(samples, 1):
        lines.append(f"Sample {idx}:")
        lines.append(f"comparison_error: {s.get('comparison_error')}")
        
        # Output the test case code according to the data format (logic corresponding to collect_samples）
        if "torch_test_case" in s or "tensorflow_test_case" in s:
            # Format 1: separated test code
            if s.get("torch_test_case") is not None:
                lines.append("torch_test_case:")
                # use ensure_ascii=False Keep Chinese，indent=2 Enhance readability
                lines.append(json.dumps(s["torch_test_case"], ensure_ascii=False, indent=2))
            if s.get("tensorflow_test_case") is not None:
                lines.append("tensorflow_test_case:")
                lines.append(json.dumps(s["tensorflow_test_case"], ensure_ascii=False, indent=2))
        elif "test_case" in s:
            # Format 2: Unified test code
            lines.append("test_case:")
            lines.append(json.dumps(s["test_case"], ensure_ascii=False, indent=2))
        else:
            # Format 3: Output complete raw data
            lines.append("raw_item:")
            lines.append(json.dumps(s.get("raw_item"), ensure_ascii=False, indent=2))
        
        lines.append("")  # Add blank lines between examples to enhance readability
    
    return "\n".join(lines)

def main():
    """
    Main program entry: extract and format all comparison_error samples          Execution process:
        1. Configure input and output paths
        2. Parse the error analysis report and obtain the file list containing comparison_error
        3. Read these files one by one and extract error samples
        4. Format and output all samples to report files
        5. Print output file path          output file:
        analysis/comparison_error_samples_report.txt
        Contains details of all comparison_error examples for human review
    """
    # Configuration path
    base_dir = Path(r"d:\graduate\DFrameworkTest\pt_tf_test")         # Project root directory
    # report_path = base_dir / "error_analysis_report.txt"              # Input: error analysis report
    report_path = base_dir / "pt_tf_log_1" / "error_analysis_report_new.txt"  # Fix: Splice paths step by step
    # log_dir = base_dir / "pt_tf_log"                                  # JSON Log file directory
    log_dir = base_dir / "pt_tf_log_1" 
    output_dir = base_dir / "analysis"                                # Output directory
    output_dir.mkdir(parents=True, exist_ok=True)                     # Make sure the output directory exists
    # output_path = output_dir / "comparison_error_samples_report.txt"  # Output file path
    output_path = output_dir / "new_comparison_error_samples_report.txt"
    
    # Step 1: Parse the report and get a list of file names containing comparison_error
    target_files = parse_report(report_path)
    
    # Step 2: Process these files one by one, collect samples and format them
    sections = []  # Store formatted output for each file
    for fname in target_files:
        jpath = log_dir / fname
        
        # Check if the file exists (fault tolerance）
        if not jpath.exists():
            continue
        
        # Extract comparison_error sample from JSON file
        samples = collect_samples(jpath)
        
        # If the file does contain samples, format them and add them to the results
        if samples:
            sections.append(format_section(fname, samples))
    
    # Step 3: Generate final report content
    content = "\n".join(sections) if sections else "No comparison_error non-null example"
    
    # Step 4: Write output file
    with open(output_path, "w", encoding="utf-8") as wf:
        wf.write(content)
    
    # Print output file path (to facilitate users to quickly locate）
    print(str(output_path))

if __name__ == "__main__":
    main()
