#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze pt_tf_test/fuzzing/result fuzzing result files in the directory Count the number of bug candidates and error type distribution  Analyze content：
1. is_bug_candidate: true total number
2. The number of simultaneous torch_error and tensorflow_error
3. Only the number of torch_error
4. Only the number of tensorflow_error
5. Only the number of comparison_error (both frameworks succeed but the results are inconsistent)  Note: - The same operator may have multiple result files (with different timestamps), the last file shall prevail.
"""

import os
import json
import re
from collections import defaultdict
from datetime import datetime


def parse_timestamp_from_filename(filename: str) -> datetime:
    """
    Extract timestamp from filename     file name format: torch_xxx_fuzzing_result_YYYYMMDD_HHMMSS.json
    """
    match = re.search(r'_(\d{8}_\d{6})\.json$', filename)
    if match:
        timestamp_str = match.group(1)
        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    return datetime.min


def get_operator_name_from_filename(filename: str) -> str:
    """
    Extract operator name from file name     file name format: torch_xxx_fuzzing_result_YYYYMMDD_HHMMSS.json
    """
    match = re.match(r'(.+)_fuzzing_result_\d{8}_\d{6}\.json$', filename)
    if match:
        return match.group(1)
    return filename


def analyze_fuzzing_results(result_dir: str):
    """Analyze fuzzing results"""
    
    # Collect all result files and group them by operator name
    operator_files = defaultdict(list)
    
    for filename in os.listdir(result_dir):
        if filename.endswith('.json') and 'fuzzing_result' in filename:
            operator_name = get_operator_name_from_filename(filename)
            timestamp = parse_timestamp_from_filename(filename)
            filepath = os.path.join(result_dir, filename)
            operator_files[operator_name].append((timestamp, filepath, filename))
    
    # For each operator, select the latest file
    latest_files = {}
    for operator_name, files in operator_files.items():
        # Sort by timestamp, take the latest
        files.sort(key=lambda x: x[0], reverse=True)
        latest_files[operator_name] = files[0]  # (timestamp, filepath, filename)
    
    print(f"📁 Results directory: {result_dir}")
    print(f"📊 Found in total {len(operator_files)} operator result file")
    print(f"   （Among them are {sum(1 for f in operator_files.values() if len(f) > 1)} operator has multiple result files, the latest one has been selected）")
    print("=" * 80)
    
    # Statistics counter
    total_bug_candidates = 0
    both_errors = 0          # There are both torch_error and tensorflow_error
    only_torch_error = 0     # only torch_error
    only_tensorflow_error = 0 # only tensorflow_error
    only_comparison_error = 0 # only comparison_error
    
    # Detailed records
    bug_details = {
        'both_errors': [],
        'only_torch_error': [],
        'only_tensorflow_error': [],
        'only_comparison_error': []
    }
    
    # Traverse the latest files of each operator
    for operator_name, (timestamp, filepath, filename) in sorted(latest_files.items()):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Iterate through all test results
            for result_item in data.get('results', []):
                for fuzzing_result in result_item.get('fuzzing_results', []):
                    if fuzzing_result.get('is_bug_candidate', False):
                        total_bug_candidates += 1
                        
                        exec_result = fuzzing_result.get('execution_result', {})
                        torch_error = exec_result.get('torch_error')
                        tensorflow_error = exec_result.get('tensorflow_error')
                        comparison_error = exec_result.get('comparison_error')
                        
                        # Determine error type
                        has_torch_error = torch_error is not None and torch_error != ""
                        has_tensorflow_error = tensorflow_error is not None and tensorflow_error != ""
                        has_comparison_error = comparison_error is not None and comparison_error != ""
                        
                        bug_info = {
                            'operator': operator_name,
                            'file': filename,
                            'round': fuzzing_result.get('round'),
                            'mutation_strategy': fuzzing_result.get('mutation_strategy', '')[:50],
                            'torch_error': str(torch_error)[:100] if torch_error else None,
                            'tensorflow_error': str(tensorflow_error)[:100] if tensorflow_error else None,
                            'comparison_error': str(comparison_error)[:100] if comparison_error else None
                        }
                        
                        if has_torch_error and has_tensorflow_error:
                            both_errors += 1
                            bug_details['both_errors'].append(bug_info)
                        elif has_torch_error and not has_tensorflow_error:
                            only_torch_error += 1
                            bug_details['only_torch_error'].append(bug_info)
                        elif has_tensorflow_error and not has_torch_error:
                            only_tensorflow_error += 1
                            bug_details['only_tensorflow_error'].append(bug_info)
                        elif has_comparison_error:
                            only_comparison_error += 1
                            bug_details['only_comparison_error'].append(bug_info)
        
        except Exception as e:
            print(f"⚠️ parse file {filename} fail: {e}")
    
    # Output statistical results
    print("\n" + "=" * 80)
    print("📈 Bug Candidate statistical results (PyTorch vs TensorFlow)")
    print("=" * 80)
    print(f"🔴 Bug Total number of candidates (is_bug_candidate=true): {total_bug_candidates}")
    print(f"   ├── There are both torch_error and tensorflow_error: {both_errors}")
    print(f"   ├── only torch_error:                       {only_torch_error}")
    print(f"   ├── only tensorflow_error:                  {only_tensorflow_error}")
    print(f"   └── Only comparison_error (results are inconsistent):     {only_comparison_error}")
    print("=" * 80)
    
    # Output details
    if bug_details['both_errors']:
        print(f"\n📋 There are two bug candidates for framework errors at the same time ({len(bug_details['both_errors'])} indivual):")
        for i, bug in enumerate(bug_details['both_errors'][:10], 1):
            print(f"  {i}. {bug['operator']} (round {bug['round']})")
            print(f"     PyTorchmistake: {bug['torch_error']}")
            print(f"     TensorFlowmistake: {bug['tensorflow_error']}")
        if len(bug_details['both_errors']) > 10:
            print(f"  ... besides {len(bug_details['both_errors']) - 10} indivual")
    
    if bug_details['only_torch_error']:
        print(f"\n📋 Only bug candidates for PyTorch errors ({len(bug_details['only_torch_error'])} indivual):")
        for i, bug in enumerate(bug_details['only_torch_error'][:10], 1):
            print(f"  {i}. {bug['operator']} (round {bug['round']})")
            print(f"     mistake: {bug['torch_error']}")
        if len(bug_details['only_torch_error']) > 10:
            print(f"  ... besides {len(bug_details['only_torch_error']) - 10} indivual")
    
    if bug_details['only_tensorflow_error']:
        print(f"\n📋 Only bug candidates for TensorFlow errors ({len(bug_details['only_tensorflow_error'])} indivual):")
        for i, bug in enumerate(bug_details['only_tensorflow_error'][:10], 1):
            print(f"  {i}. {bug['operator']} (round {bug['round']})")
            print(f"     mistake: {bug['tensorflow_error']}")
        if len(bug_details['only_tensorflow_error']) > 10:
            print(f"  ... besides {len(bug_details['only_tensorflow_error']) - 10} indivual")
    
    if bug_details['only_comparison_error']:
        print(f"\n📋 Only bug candidates with inconsistent results ({len(bug_details['only_comparison_error'])} indivual):")
        for i, bug in enumerate(bug_details['only_comparison_error'][:10], 1):
            print(f"  {i}. {bug['operator']} (round {bug['round']})")
            print(f"     difference: {bug['comparison_error']}")
        if len(bug_details['only_comparison_error']) > 10:
            print(f"  ... besides {len(bug_details['only_comparison_error']) - 10} indivual")
    
    # Save detailed report
    report_path = os.path.join(os.path.dirname(result_dir), 'bug_candidates_report.json')
    report_data = {
        'summary': {
            'total_operators': len(latest_files),
            'total_bug_candidates': total_bug_candidates,
            'both_errors': both_errors,
            'only_torch_error': only_torch_error,
            'only_tensorflow_error': only_tensorflow_error,
            'only_comparison_error': only_comparison_error
        },
        'details': bug_details
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed report saved to: {report_path}")
    
    return report_data


if __name__ == '__main__':
    # Get the directory where the current script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, 'result')
    
    if os.path.exists(result_dir):
        analyze_fuzzing_results(result_dir)
    else:
        print(f"❌ The result directory does not exist: {result_dir}")
