#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze fuzzing result files under pt_pd_test/fuzzing/result.
Count bug candidates and error type distribution.

Analysis items:
1. Total is_bug_candidate: true
2. Both torch_error and paddle_error
3. Only torch_error
4. Only paddle_error
5. Only comparison_error (both frameworks succeeded but results differ)

Note:
- One operator may have multiple result files (different timestamps); use the latest file
"""

import os
import json
import re
from collections import defaultdict
from datetime import datetime


def parse_timestamp_from_filename(filename: str) -> datetime:
    """
    Extract timestamp from filename.
    Filename format: torch_xxx_fuzzing_result_YYYYMMDD_HHMMSS.json
    """
    match = re.search(r'_(\d{8}_\d{6})\.json$', filename)
    if match:
        timestamp_str = match.group(1)
        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    return datetime.min


def get_operator_name_from_filename(filename: str) -> str:
    """
    Extract operator name from filename.
    Filename format: torch_xxx_fuzzing_result_YYYYMMDD_HHMMSS.json
    """
    match = re.match(r'(.+)_fuzzing_result_\d{8}_\d{6}\.json$', filename)
    if match:
        return match.group(1)
    return filename


def analyze_fuzzing_results(result_dir: str):
    """Analyze fuzzing results."""
    
    # Collect all result files grouped by operator
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
        # Sort by timestamp, take latest
        files.sort(key=lambda x: x[0], reverse=True)
        latest_files[operator_name] = files[0]  # (timestamp, filepath, filename)
    
    print(f"📁 Result directory: {result_dir}")
    print(f"📊 Found result files for {len(operator_files)} operators")
    print(f"   ({sum(1 for f in operator_files.values() if len(f) > 1)} operators have multiple result files; latest selected)")
    print("=" * 80)
    
    # Counters
    total_bug_candidates = 0
    both_errors = 0          # Both torch_error and paddle_error
    only_torch_error = 0     # Only torch_error
    only_paddle_error = 0    # Only paddle_error
    only_comparison_error = 0 # Only comparison_error
    
    # Detailed records
    bug_details = {
        'both_errors': [],
        'only_torch_error': [],
        'only_paddle_error': [],
        'only_comparison_error': []
    }
    
    # Traverse latest file for each operator
    for operator_name, (timestamp, filepath, filename) in sorted(latest_files.items()):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Traverse all test results
            for result_item in data.get('results', []):
                for fuzzing_result in result_item.get('fuzzing_results', []):
                    if fuzzing_result.get('is_bug_candidate', False):
                        total_bug_candidates += 1
                        
                        exec_result = fuzzing_result.get('execution_result', {})
                        torch_error = exec_result.get('torch_error')
                        paddle_error = exec_result.get('paddle_error')
                        comparison_error = exec_result.get('comparison_error')
                        
                        # Determine error type
                        has_torch_error = torch_error is not None and torch_error != ""
                        has_paddle_error = paddle_error is not None and paddle_error != ""
                        has_comparison_error = comparison_error is not None and comparison_error != ""
                        
                        bug_info = {
                            'operator': operator_name,
                            'file': filename,
                            'round': fuzzing_result.get('round'),
                            'mutation_strategy': fuzzing_result.get('mutation_strategy', '')[:50],
                            'torch_error': str(torch_error)[:100] if torch_error else None,
                            'paddle_error': str(paddle_error)[:100] if paddle_error else None,
                            'comparison_error': str(comparison_error)[:100] if comparison_error else None
                        }
                        
                        if has_torch_error and has_paddle_error:
                            both_errors += 1
                            bug_details['both_errors'].append(bug_info)
                        elif has_torch_error and not has_paddle_error:
                            only_torch_error += 1
                            bug_details['only_torch_error'].append(bug_info)
                        elif has_paddle_error and not has_torch_error:
                            only_paddle_error += 1
                            bug_details['only_paddle_error'].append(bug_info)
                        elif has_comparison_error:
                            only_comparison_error += 1
                            bug_details['only_comparison_error'].append(bug_info)
        
        except Exception as e:
            print(f"⚠️ Failed to parse file {filename}: {e}")
    
    # Output summary
    print("\n" + "=" * 80)
    print("📈 Bug candidate summary (PyTorch vs PaddlePaddle)")
    print("=" * 80)
    print(f"🔴 Total bug candidates (is_bug_candidate=true): {total_bug_candidates}")
    print(f"   ├── Both torch_error and paddle_error:  {both_errors}")
    print(f"   ├── Only torch_error:                   {only_torch_error}")
    print(f"   ├── Only paddle_error:                  {only_paddle_error}")
    print(f"   └── Only comparison_error (mismatch):  {only_comparison_error}")
    print("=" * 80)
    
    # Output details
    if bug_details['both_errors']:
        print(f"\n📋 Bug candidates with both framework errors ({len(bug_details['both_errors'])}):")
        for i, bug in enumerate(bug_details['both_errors'][:10], 1):
            print(f"  {i}. {bug['operator']} (round {bug['round']})")
            print(f"     PyTorch error: {bug['torch_error']}")
            print(f"     PaddlePaddle error: {bug['paddle_error']}")
        if len(bug_details['both_errors']) > 10:
            print(f"  ... {len(bug_details['both_errors']) - 10} more")
    
    if bug_details['only_torch_error']:
        print(f"\n📋 Bug candidates with only PyTorch errors ({len(bug_details['only_torch_error'])}):")
        for i, bug in enumerate(bug_details['only_torch_error'][:10], 1):
            print(f"  {i}. {bug['operator']} (round {bug['round']})")
            print(f"     Error: {bug['torch_error']}")
        if len(bug_details['only_torch_error']) > 10:
            print(f"  ... {len(bug_details['only_torch_error']) - 10} more")
    
    if bug_details['only_paddle_error']:
        print(f"\n📋 Bug candidates with only PaddlePaddle errors ({len(bug_details['only_paddle_error'])}):")
        for i, bug in enumerate(bug_details['only_paddle_error'][:10], 1):
            print(f"  {i}. {bug['operator']} (round {bug['round']})")
            print(f"     Error: {bug['paddle_error']}")
        if len(bug_details['only_paddle_error']) > 10:
            print(f"  ... {len(bug_details['only_paddle_error']) - 10} more")
    
    if bug_details['only_comparison_error']:
        print(f"\n📋 Bug candidates with only mismatches ({len(bug_details['only_comparison_error'])}):")
        for i, bug in enumerate(bug_details['only_comparison_error'][:10], 1):
            print(f"  {i}. {bug['operator']} (round {bug['round']})")
            print(f"     Difference: {bug['comparison_error']}")
        if len(bug_details['only_comparison_error']) > 10:
            print(f"  ... {len(bug_details['only_comparison_error']) - 10} more")
    
    # Save detailed report
    report_path = os.path.join(os.path.dirname(result_dir), 'bug_candidates_report.json')
    report_data = {
        'summary': {
            'total_operators': len(latest_files),
            'total_bug_candidates': total_bug_candidates,
            'both_errors': both_errors,
            'only_torch_error': only_torch_error,
            'only_paddle_error': only_paddle_error,
            'only_comparison_error': only_comparison_error
        },
        'details': bug_details
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed report saved to: {report_path}")
    
    return report_data


if __name__ == '__main__':
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, 'result')
    
    if os.path.exists(result_dir):
        analyze_fuzzing_results(result_dir)
    else:
        print(f"❌ Result directory does not exist: {result_dir}")
