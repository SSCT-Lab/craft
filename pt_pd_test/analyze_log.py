#!/usr/bin/env python3
"""
Analyze batch test log files and count LLM-generated cases and successful cases.
"""
import re

def analyze_log_file(log_file_path):
    """
    Analyze a log file and extract statistics.
    
    Args:
        log_file_path: Log file path
    
    Returns:
        Statistics result dictionary
    """
    total_llm_cases = 0
    total_successful_cases = 0
    operator_count = 0
    operator_details = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Use regex to match per-operator statistics
    # Pattern: [index/total] operator_name
    #          Status: ✅ Done
    #          Total iterations: X
    #          LLM generated cases: X
    #          Successful cases: X
    #          Success rate: X%
    
    pattern = r'\[(\d+)/(\d+)\]\s+(\S+)\s+Status:\s+✅\s+Done\s+Total iterations:\s+(\d+)\s+LLM generated cases:\s+(\d+)\s+Successful cases:\s+(\d+)(?:\s+Success rate:\s+([\d.]+)%)?'
    
    matches = re.findall(pattern, content)
    
    for match in matches:
        seq_num, total_ops, operator_name, iterations, llm_cases, successful_cases, *success_rate = match
        
        llm_cases = int(llm_cases)
        successful_cases = int(successful_cases)
        
        total_llm_cases += llm_cases
        total_successful_cases += successful_cases
        operator_count += 1
        
        # Compute success rate
        if llm_cases > 0:
            rate = (successful_cases / llm_cases) * 100
        else:
            rate = 0.0
        
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


def print_statistics(stats):
    """Print statistics results."""
    print("="*80)
    print("📊 Batch test log statistics analysis")
    print("="*80)
    print(f"\n✅ Total test operators: {stats['operator_count']}")
    print(f"📝 Total LLM-generated cases: {stats['total_llm_cases']}")
    print(f"✅ Total successful cases: {stats['total_successful_cases']}")
    print(f"📈 Overall success rate: {stats['overall_success_rate']:.2f}%")
    
    # Success rate distribution
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
    
    # Top 10 operators by success rate
    print(f"\n🏆 Top 10 operators by success rate:")
    sorted_ops = sorted(stats['operator_details'], key=lambda x: x['success_rate'], reverse=True)
    for i, op in enumerate(sorted_ops[:10], 1):
        print(f"  {i:2d}. {op['name']:40s} - {op['successful_cases']}/{op['llm_cases']} ({op['success_rate']:.2f}%)")
    
    # Operators with 0% success rate
    zero_success_ops = [op for op in stats['operator_details'] if op['success_rate'] == 0 and op['llm_cases'] > 0]
    if zero_success_ops:
        print(f"\n⚠️ Operators with 0% success rate (total {len(zero_success_ops)}):")
        for op in zero_success_ops[:20]:  # Show only the first 20
            print(f"  - {op['name']:40s} (LLM generated {op['llm_cases']} cases)")
        if len(zero_success_ops) > 20:
            print(f"  ... {len(zero_success_ops) - 20} more operators")


def merge_statistics(all_stats_list):
    """
    Merge statistics results from multiple log files.
    
    Args:
        all_stats_list: List of statistics results
    
    Returns:
        Merged statistics results
    """
    merged_stats = {
        'operator_count': 0,
        'total_llm_cases': 0,
        'total_successful_cases': 0,
        'overall_success_rate': 0,
        'operator_details': []
    }
    
    # Operator set for de-duplication (by operator name)
    operator_dict = {}
    
    for stats in all_stats_list:
        merged_stats['operator_count'] += stats['operator_count']
        merged_stats['total_llm_cases'] += stats['total_llm_cases']
        merged_stats['total_successful_cases'] += stats['total_successful_cases']
        
        # Merge operator details (accumulate if the operator appears in multiple files)
        for op in stats['operator_details']:
            op_name = op['name']
            if op_name in operator_dict:
                # Accumulate data
                operator_dict[op_name]['llm_cases'] += op['llm_cases']
                operator_dict[op_name]['successful_cases'] += op['successful_cases']
                # Recalculate success rate
                if operator_dict[op_name]['llm_cases'] > 0:
                    operator_dict[op_name]['success_rate'] = (
                        operator_dict[op_name]['successful_cases'] / 
                        operator_dict[op_name]['llm_cases'] * 100
                    )
            else:
                # New operator
                operator_dict[op_name] = {
                    'seq': op['seq'],
                    'name': op_name,
                    'llm_cases': op['llm_cases'],
                    'successful_cases': op['successful_cases'],
                    'success_rate': op['success_rate']
                }
    
    # Convert to list
    merged_stats['operator_details'] = list(operator_dict.values())
    
    # Recalculate overall success rate
    if merged_stats['total_llm_cases'] > 0:
        merged_stats['overall_success_rate'] = (
            merged_stats['total_successful_cases'] / 
            merged_stats['total_llm_cases'] * 100
        )
    
    # Update operator count (after de-duplication)
    merged_stats['operator_count'] = len(merged_stats['operator_details'])
    
    return merged_stats


if __name__ == "__main__":
    # Multiple log file paths
    log_files = [
        r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log\batch_test_log_20251124_192159.txt",
        r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log\batch_test_log_20251125_001535.txt",
        # r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log\batch_test_log_20251125_140543.txt",
        r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log\batch_test_log_20251125_140826.txt",
        r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log\batch_test_log_20251125_141815.txt",
        r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log\batch_test_log_20251125_151044.txt",
        r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log\batch_test_log_20251125_194530.txt",
        r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log\batch_test_log_20251201_230510.txt",
        r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log\batch_test_log_20251201_235243.txt",
        r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log\batch_test_log_20251202_003954.txt",
        r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log\batch_test_log_20251202_005325.txt",
        r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log\batch_test_log_20251202_123851.txt",
        r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log\batch_test_log_20251202_124451.txt",
        r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log\batch_test_log_20251202_131840.txt",
        r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log\batch_test_log_20251202_133443.txt",
        # r"D:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log_1\batch_test_log_20260124_175428.txt",
        # r"D:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log_1\batch_test_log_20260124_222701.txt",
        # r"D:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log_1\batch_test_log_20260124_223118.txt",
        # r"D:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log_1\batch_test_log_20260124_224931.txt",
        # r"D:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log_1\batch_test_log_20260125_000122.txt",
        # r"D:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log_1\batch_test_log_20260125_004650.txt",
        # r"D:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log_1\batch_test_log_20260125_084213.txt",
    ]
    
    print("="*80)
    print("📊 Batch log file statistics analysis")
    print("="*80)
    print(f"\n📂 Total log files analyzed: {len(log_files)}\n")
    
    all_stats_list = []
    
    # Analyze each log file
    for i, log_file in enumerate(log_files, 1):
        print(f"  {i}. {log_file.split('/')[-1]}")
        try:
            stats = analyze_log_file(log_file)
            all_stats_list.append(stats)
            print(f"     ✅ Success - {stats['operator_count']} operators, {stats['total_llm_cases']} LLM cases, {stats['total_successful_cases']} successful")
        except FileNotFoundError:
            print(f"     ❌ File not found, skipping")
        except Exception as e:
            print(f"     ❌ Analysis failed: {e}")
    
    if not all_stats_list:
        print("\n❌ No log files were analyzed successfully")
        exit(1)
    
    print(f"\n{'='*80}")
    print("📊 Merged statistics results")
    print("="*80)
    
    # Merge statistics results
    merged_stats = merge_statistics(all_stats_list)
    
    # Print merged statistics results
    print_statistics(merged_stats)
    
    print("\n" + "="*80)
    print("✅ Analysis complete")
    print("="*80)
