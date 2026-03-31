"""
Analyze files in the ms_a directory, extract API names,
and check whether those APIs appear in torch_error_samples_report.txt.

Functions:
1. Traverse all files under pt_ms_test/analysis/ms_a
2. Extract the torch_xxx part from filenames (after llm_enhanced_, before _date)
3. Check whether these API names appear in torch_error_samples_report.txt
4. Print all API names that do not appear
"""

import os
import re
from typing import Set


def extract_api_name_from_filename(filename: str) -> str:
    """
    Extract the API name from a filename.

    Filename format: llm_enhanced_torch_xxx_date.json_sampleN.txt
    Extract: torch_xxx part

    Args:
        filename: File name

    Returns:
        Extracted API name (underscore-separated)
    """
    # Pattern: llm_enhanced_(torch_xxx)_date
    # Date format: 8 digits (e.g., 20251202)
    pattern = r'llm_enhanced_(torch_[\w_]+?)_(\d{8}_\d{6})'
    match = re.search(pattern, filename)
    
    if match:
        return match.group(1)
    
    return ""


def extract_apis_from_report(report_path: str) -> Set[str]:
    """
    Extract all API names that appear in torch_error_samples_report.txt.

    Report format: File: llm_enhanced_torch_xxx_date.json

    Args:
        report_path: Report file path

    Returns:
        Set of API names appearing in the report
    """
    apis = set()
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Match filename lines in the report
    pattern = r'File: llm_enhanced_(torch_[\w_]+?)_(\d{8}_\d{6})\.json'
    matches = re.findall(pattern, content)
    
    for match in matches:
        api_name = match[0]
        apis.add(api_name)
    
    return apis


def main():
    """Main entry point."""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ms_a directory path
    ms_a_dir = os.path.join(script_dir, 'analysis', 'ms_a')
    
    # torch_error_samples_report.txt path
    report_path = os.path.join(script_dir, 'analysis', 'torch_error_samples_report.txt')
    
    print("=" * 60)
    print("📊 ms_a Directory API Analysis Tool")
    print("=" * 60)
    
    # 1. Extract all API names from ms_a directory
    ms_a_apis = set()
    
    if not os.path.exists(ms_a_dir):
        print(f"❌ Directory does not exist: {ms_a_dir}")
        return
    
    files = os.listdir(ms_a_dir)
    print(f"\n📁 Number of files in ms_a directory: {len(files)}")
    
    for filename in files:
        api_name = extract_api_name_from_filename(filename)
        if api_name:
            ms_a_apis.add(api_name)
    
    print(f"✅ Extracted {len(ms_a_apis)} unique API names")
    
    # 2. Extract API names from torch_error_samples_report.txt
    if not os.path.exists(report_path):
        print(f"❌ Report file does not exist: {report_path}")
        return
    
    report_apis = extract_apis_from_report(report_path)
    print(f"✅ torch_error_report contains {len(report_apis)} unique API names")
    
    # 3. Find APIs not appearing in the report
    apis_not_in_report = ms_a_apis - report_apis
    
    print("\n" + "=" * 60)
    print(f"📋 APIs not appearing in torch_error_report ({len(apis_not_in_report)}):")
    print("=" * 60)
    
    # Print in alphabetical order
    for api_name in sorted(apis_not_in_report):
        print(f"  - {api_name}")
    
    # Extra info: print APIs that appear in the report (for verification)
    apis_in_report = ms_a_apis & report_apis
    print("\n" + "-" * 60)
    print(f"📋 APIs appearing in torch_error_report ({len(apis_in_report)}):")
    print("-" * 60)
    for api_name in sorted(apis_in_report):
        print(f"  - {api_name}")
    
    print("\n" + "=" * 60)
    print("🎉 Analysis completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
