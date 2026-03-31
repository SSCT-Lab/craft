#!/usr/bin/env python3
"""
Analyze pt-to-ms test logs and count LLM-generated cases and successful executions.
Aligned with the pt-to-pd analysis convention, and supports two log sources:
1) Batch text logs: batch_test_log_*.txt (if present)
2) LLM-enhanced JSON logs: llm_enhanced_torch*.json (auto statistics)
"""
import re
import json
from pathlib import Path

def analyze_log_file(log_file_path):
    """
    Analyze batch text logs and extract statistics.
    Returns the same structure as pt_pd_test/analyze_log.py.
    """
    total_llm_cases = 0
    total_successful_cases = 0
    operator_count = 0
    operator_details = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'\[(\d+)/(\d+)\]\s+(\S+)\s+Status:\s+✅\s+Completed\s+Total iterations:\s+(\d+)\s+LLM generated cases:\s+(\d+)\s+Successful cases:\s+(\d+)(?:\s+Success rate:\s+([\d.]+)%)?'
    matches = re.findall(pattern, content)
    
    for match in matches:
        seq_num, total_ops, operator_name, iterations, llm_cases, successful_cases, *success_rate = match
        llm_cases = int(llm_cases)
        successful_cases = int(successful_cases)
        total_llm_cases += llm_cases
        total_successful_cases += successful_cases
        operator_count += 1
        rate = (successful_cases / llm_cases * 100) if llm_cases > 0 else 0.0
        operator_details.append({
            'seq': int(seq_num),
            'name': operator_name,
            'llm_cases': llm_cases,
            'successful_cases': successful_cases,
            'success_rate': rate
        })
    
    return {
        'operator_count': operator_count,
        'total_llm_cases': total_llm_cases,
        'total_successful_cases': total_successful_cases,
        'overall_success_rate': (total_successful_cases / total_llm_cases * 100) if total_llm_cases > 0 else 0,
        'operator_details': operator_details
    }

def analyze_json_file(json_file_path):
        """
        Analyze LLM-enhanced JSON logs and extract statistics.
        Expected keys:
            - operator
            - llm_generated_test_cases
            - successful_test_cases
        """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    operator_name = data.get('operator', 'N/A')
    llm_cases = int(data.get('llm_generated_test_cases', 0))
    successful_cases = int(data.get('successful_test_cases', 0))
    rate = (successful_cases / llm_cases * 100) if llm_cases > 0 else 0.0
    return {
        'operator_count': 1,
        'total_llm_cases': llm_cases,
        'total_successful_cases': successful_cases,
        'overall_success_rate': rate,
        'operator_details': [{
            'seq': 1,
            'name': operator_name,
            'llm_cases': llm_cases,
            'successful_cases': successful_cases,
            'success_rate': rate
        }]
    }

def print_statistics(stats):
    print("="*80)
    print("📊 Batch Test Log Statistical Analysis (PyTorch → MindSpore)")
    print("="*80)
    print(f"\n✅ Total operators tested: {stats['operator_count']}")
    print(f"📝 Total LLM-generated cases: {stats['total_llm_cases']}")
    print(f"✅ Total successful cases: {stats['total_successful_cases']}")
    print(f"📈 Overall success rate: {stats['overall_success_rate']:.2f}%")
    
    success_rate_distribution = {
        '0%': 0,
        '1-25%': 0,
        '26-50%': 0,
        '51-75%': 0,
        '76-99%': 0,
        '100%': 0
    }
    for op in stats['operator_details']:
        rate = op['success_rate']
        if rate == 0:
            success_rate_distribution['0%'] += 1
        elif rate <= 25:
            success_rate_distribution['1-25%'] += 1
        elif rate <= 50:
            success_rate_distribution['26-50%'] += 1
        elif rate <= 75:
            success_rate_distribution['51-75%'] += 1
        elif rate < 100:
            success_rate_distribution['76-99%'] += 1
        else:
            success_rate_distribution['100%'] += 1
    print(f"\n📊 Success rate distribution:")
    for range_name, count in success_rate_distribution.items():
        percentage = (count / stats['operator_count'] * 100) if stats['operator_count'] > 0 else 0
        print(f"  {range_name:10s}: {count:3d} operators ({percentage:5.2f}%)")
    
    print(f"\n🏆 Top 10 operators by success rate:")
    sorted_ops = sorted(stats['operator_details'], key=lambda x: x['success_rate'], reverse=True)
    for i, op in enumerate(sorted_ops[:10], 1):
        print(f"  {i:2d}. {op['name']:40s} - {op['successful_cases']}/{op['llm_cases']} ({op['success_rate']:.2f}%)")
    
    zero_success_ops = [op for op in stats['operator_details'] if op['success_rate'] == 0 and op['llm_cases'] > 0]
    if zero_success_ops:
        print(f"\n⚠️ Operators with 0% success rate (total {len(zero_success_ops)}):")
        for op in zero_success_ops[:20]:
            print(f"  - {op['name']:40s} (LLM generated {op['llm_cases']} cases)")
        if len(zero_success_ops) > 20:
            print(f"  ... {len(zero_success_ops) - 20} more operators")

def merge_statistics(all_stats_list):
    merged_stats = {
        'operator_count': 0,
        'total_llm_cases': 0,
        'total_successful_cases': 0,
        'overall_success_rate': 0,
        'operator_details': []
    }
    operator_dict = {}
    for stats in all_stats_list:
        merged_stats['operator_count'] += stats['operator_count']
        merged_stats['total_llm_cases'] += stats['total_llm_cases']
        merged_stats['total_successful_cases'] += stats['total_successful_cases']
        for op in stats['operator_details']:
            name = op['name']
            if name in operator_dict:
                operator_dict[name]['llm_cases'] += op['llm_cases']
                operator_dict[name]['successful_cases'] += op['successful_cases']
                lc = operator_dict[name]['llm_cases']
                operator_dict[name]['success_rate'] = (operator_dict[name]['successful_cases'] / lc * 100) if lc > 0 else 0.0
            else:
                operator_dict[name] = {
                    'seq': op['seq'],
                    'name': name,
                    'llm_cases': op['llm_cases'],
                    'successful_cases': op['successful_cases'],
                    'success_rate': op['success_rate']
                }
    merged_stats['operator_details'] = list(operator_dict.values())
    if merged_stats['total_llm_cases'] > 0:
        merged_stats['overall_success_rate'] = merged_stats['total_successful_cases'] / merged_stats['total_llm_cases'] * 100
    merged_stats['operator_count'] = len(merged_stats['operator_details'])
    return merged_stats

if __name__ == "__main__":
    log_files = [
        # r"d:\graduate\DFrameworkTest\pt_ms_test\pt_ms_log\batch_test_log_20251214_170607.txt",
        # r"d:\graduate\DFrameworkTest\pt_ms_test\pt_ms_log\batch_test_log_20251215_145128.txt",
        # r"d:\graduate\DFrameworkTest\pt_ms_test\pt_ms_log\batch_test_log_20251215_183338.txt",
        # r"d:\graduate\DFrameworkTest\pt_ms_test\pt_ms_log\batch_test_log_20251215_190033.txt",
        # r"d:\graduate\DFrameworkTest\pt_ms_test\pt_ms_log\batch_test_log_20251215_191126.txt",
        # r"d:\graduate\DFrameworkTest\pt_ms_test\pt_ms_log\batch_test_log_20251216_004326.txt",
        r"d:\graduate\DFrameworkTest\pt_ms_test\pt_ms_log_1\batch_test_log_20260125_214238.txt",
        r"d:\graduate\DFrameworkTest\pt_ms_test\pt_ms_log_1\batch_test_log_20260126_002545.txt",
        r"d:\graduate\DFrameworkTest\pt_ms_test\pt_ms_log_1\batch_test_log_20260126_111523.txt",
    ]
    
    print("="*80)
    print("📊 Batch Log File Statistical Analysis (PyTorch → MindSpore)")
    print("="*80)
    print(f"\n📂 Analyzing {len(log_files)} log files:\n")
    for i, log_file in enumerate(log_files, 1):
        print(f"  {i}. {Path(log_file).name}")
    
    all_stats_list = []
    for log_file in log_files:
        try:
            stats = analyze_log_file(log_file)
            all_stats_list.append(stats)
            print(f"     ✅ Success - {stats['operator_count']} operators, {stats['total_llm_cases']} LLM cases, {stats['total_successful_cases']} successes")
        except FileNotFoundError:
            print(f"     ❌ File not found, skipping")
        except Exception as e:
            print(f"     ❌ Analysis failed: {e}")
    
    if not all_stats_list:
        print("\n❌ No log files were successfully analyzed")
        raise SystemExit(1)
    
    print(f"\n{'='*80}")
    print("📊 Merged Statistics")
    print("="*80)
    merged_stats = merge_statistics(all_stats_list)
    print_statistics(merged_stats)
    print("\n" + "="*80)
    print("✅ Analysis completed")
    print("="*80)
