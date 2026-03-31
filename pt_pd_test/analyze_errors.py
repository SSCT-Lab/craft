import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_json_files(log_dir):
    """Analyze JSON files starting with llm_enhanced_torch in the pt_pd_log folder."""
    
    # Total error counts
    total_torch_errors = 0
    total_paddle_errors = 0
    total_comparison_errors = 0
    
    # Store info for files with errors
    files_with_errors = []
    
    # Get all matching JSON files
    log_path = Path(log_dir)
    json_files = sorted(log_path.glob("llm_enhanced_torch*.json"))
    
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Count errors for current file
            file_torch_errors = 0
            file_paddle_errors = 0
            file_comparison_errors = 0
            error_iterations = {
                'torch_error': [],
                'paddle_error': [],
                'comparison_error': []
            }
            
            # Iterate all test results
            if 'results' in data:
                for result in data['results']:
                    if 'execution_result' in result:
                        exec_result = result['execution_result']
                        iteration = result.get('iteration', 'N/A')
                        
                        # Check torch_error
                        if exec_result.get('torch_error') is not None:
                            file_torch_errors += 1
                            total_torch_errors += 1
                            error_iterations['torch_error'].append(iteration)
                        
                        # Check paddle_error
                        if exec_result.get('paddle_error') is not None:
                            file_paddle_errors += 1
                            total_paddle_errors += 1
                            error_iterations['paddle_error'].append(iteration)
                        
                        # Check comparison_error
                        if exec_result.get('comparison_error') is not None:
                            file_comparison_errors += 1
                            total_comparison_errors += 1
                            error_iterations['comparison_error'].append(iteration)
            
            # Record the file if any errors exist
            if file_torch_errors > 0 or file_paddle_errors > 0 or file_comparison_errors > 0:
                files_with_errors.append({
                    'filename': json_file.name,
                    'torch_errors': file_torch_errors,
                    'paddle_errors': file_paddle_errors,
                    'comparison_errors': file_comparison_errors,
                    'torch_error_iterations': error_iterations['torch_error'],
                    'paddle_error_iterations': error_iterations['paddle_error'],
                    'comparison_error_iterations': error_iterations['comparison_error']
                })
        
        except Exception as e:
            print(f"Error processing file {json_file.name}: {e}")
    
    return total_torch_errors, total_paddle_errors, total_comparison_errors, files_with_errors

def generate_report(output_file, total_torch, total_paddle, total_comparison, files_with_errors):
    """Generate a text error report."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch-PaddlePaddle Test Error Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall summary
        f.write("[Overall Error Summary]\n")
        f.write(f"torch_error non-null total: {total_torch}\n")
        f.write(f"paddle_error non-null total: {total_paddle}\n")
        f.write(f"comparison_error non-null total: {total_comparison}\n")
        f.write(f"Total files with errors: {len(files_with_errors)}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Detailed file error info
        if files_with_errors:
            f.write("[Detailed Error File Info]\n\n")
            
            for idx, file_info in enumerate(files_with_errors, 1):
                f.write(f"{idx}. Filename: {file_info['filename']}\n")
                f.write("-" * 80 + "\n")
                
                # torch_error info
                if file_info['torch_errors'] > 0:
                    f.write(f"   torch_error non-null count: {file_info['torch_errors']}\n")
                    f.write(f"   Iteration values for cases: {', '.join(map(str, file_info['torch_error_iterations']))}\n")
                
                # paddle_error info
                if file_info['paddle_errors'] > 0:
                    f.write(f"   paddle_error non-null count: {file_info['paddle_errors']}\n")
                    f.write(f"   Iteration values for cases: {', '.join(map(str, file_info['paddle_error_iterations']))}\n")
                
                # comparison_error info
                if file_info['comparison_errors'] > 0:
                    f.write(f"   comparison_error non-null count: {file_info['comparison_errors']}\n")
                    f.write(f"   Iteration values for cases: {', '.join(map(str, file_info['comparison_error_iterations']))}\n")
                
                f.write("\n")
        else:
            f.write("[Detailed Error File Info]\n\n")
            f.write("No files with errors were found.\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Report generation completed\n")
        f.write("=" * 80 + "\n")

def main():
    # Set paths
    log_dir = r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log"
    output_file = r"d:\graduate\DFrameworkTest\pt_pd_test\error_analysis_report.txt"
    
    print("Starting JSON file analysis...")
    
    # Analyze files
    total_torch, total_paddle, total_comparison, files_with_errors = analyze_json_files(log_dir)
    
    # Generate report
    print(f"\nAnalysis complete!")
    print("Total error summary:")
    print(f"  - torch_error: {total_torch}")
    print(f"  - paddle_error: {total_paddle}")
    print(f"  - comparison_error: {total_comparison}")
    print(f"  - Files with errors: {len(files_with_errors)}")
    
    generate_report(output_file, total_torch, total_paddle, total_comparison, files_with_errors)
    
    print(f"\nReport generated: {output_file}")

if __name__ == "__main__":
    main()
