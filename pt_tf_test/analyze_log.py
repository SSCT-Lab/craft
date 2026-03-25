#!/usr/bin/env python3
"""
分析批量测试日志文件，统计LLM生成用例数和成功执行用例数（PyTorch → TensorFlow）
"""
import re

def analyze_log_file(log_file_path):
    total_llm_cases = 0
    total_successful_cases = 0
    operator_count = 0
    operator_details = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'\[(\d+)/(\d+)\]\s+(\S+)\s+状态:\s+✅\s+完成\s+总迭代次数:\s+(\d+)\s+LLM生成用例数:\s+(\d+)\s+成功执行用例数:\s+(\d+)(?:\s+成功率:\s+([\d.]+)%)?'
    matches = re.findall(pattern, content)
    
    for match in matches:
        seq_num, total_ops, operator_name, iterations, llm_cases, successful_cases, *success_rate = match
        llm_cases = int(llm_cases)
        successful_cases = int(successful_cases)
        total_llm_cases += llm_cases
        total_successful_cases += successful_cases
        operator_count += 1
        rate = (successful_cases / llm_cases) * 100 if llm_cases > 0 else 0.0
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
    print("="*80)
    print("📊 批量测试日志统计分析（PyTorch → TensorFlow）")
    print("="*80)
    print(f"\n✅ 测试算子总数: {stats['operator_count']}")
    print(f"📝 LLM生成用例总数: {stats['total_llm_cases']}")
    print(f"✅ 成功执行用例总数: {stats['total_successful_cases']}")
    print(f"📈 总体成功率: {stats['overall_success_rate']:.2f}%")
    
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
    print(f"\n📊 成功率分布:")
    for range_name, count in success_rate_distribution.items():
        percentage = (count / stats['operator_count'] * 100) if stats['operator_count'] > 0 else 0
        print(f"  {range_name:10s}: {count:3d} 个算子 ({percentage:5.2f}%)")
    
    print(f"\n🏆 成功率最高的前10个算子:")
    sorted_ops = sorted(stats['operator_details'], key=lambda x: x['success_rate'], reverse=True)
    for i, op in enumerate(sorted_ops[:10], 1):
        print(f"  {i:2d}. {op['name']:40s} - {op['successful_cases']}/{op['llm_cases']} ({op['success_rate']:.2f}%)")
    
    zero_success_ops = [op for op in stats['operator_details'] if op['success_rate'] == 0 and op['llm_cases'] > 0]
    if zero_success_ops:
        print(f"\n⚠️ 成功率为0的算子（共{len(zero_success_ops)}个）:")
        for op in zero_success_ops[:20]:
            print(f"  - {op['name']:40s} (LLM生成了{op['llm_cases']}个用例)")
        if len(zero_success_ops) > 20:
            print(f"  ... 还有 {len(zero_success_ops) - 20} 个算子")

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
            op_name = op['name']
            if op_name in operator_dict:
                operator_dict[op_name]['llm_cases'] += op['llm_cases']
                operator_dict[op_name]['successful_cases'] += op['successful_cases']
                lc = operator_dict[op_name]['llm_cases']
                operator_dict[op_name]['success_rate'] = (operator_dict[op_name]['successful_cases'] / lc * 100) if lc > 0 else 0.0
            else:
                operator_dict[op_name] = {
                    'seq': op['seq'],
                    'name': op_name,
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
        r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log\batch_test_log_20251215_162151.txt",
        r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log\batch_test_log_20251215_163032.txt",
        r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log\batch_test_log_20251215_171217.txt",
        r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log\batch_test_log_20251215_200801.txt",
        r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log\batch_test_log_20251215_225440.txt",
        r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log\batch_test_log_20251215_234447.txt",
        r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log\batch_test_log_20251216_002607.txt",

        # r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1\batch_test_log_20260123_130821.txt",
        # r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1\batch_test_log_20260123_151945.txt",
        # r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1\batch_test_log_20260123_165129.txt",
        # r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1\batch_test_log_20260123_170300.txt",
        # r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1\batch_test_log_20260123_182808.txt",
        # r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1\batch_test_log_20260123_183252.txt",
        # r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1\batch_test_log_20260123_201401.txt",
    ]
    
    print("="*80)
    print("📊 批量日志文件统计分析（PyTorch → TensorFlow）")
    print("="*80)
    print(f"\n📂 共分析 {len(log_files)} 个日志文件:\n")
    
    all_stats_list = []
    for i, log_file in enumerate(log_files, 1):
        print(f"  {i}. {log_file.split('/')[-1]}")
        try:
            stats = analyze_log_file(log_file)
            all_stats_list.append(stats)
            print(f"     ✅ 成功 - {stats['operator_count']}个算子, {stats['total_llm_cases']}个LLM用例, {stats['total_successful_cases']}个成功")
        except FileNotFoundError:
            print(f"     ❌ 文件不存在，跳过")
        except Exception as e:
            print(f"     ❌ 分析失败: {e}")
    
    if not all_stats_list:
        print("\n❌ 没有成功分析任何日志文件")
        raise SystemExit(1)
    
    print(f"\n{'='*80}")
    print("📊 合并统计结果")
    print("="*80)
    merged_stats = merge_statistics(all_stats_list)
    print_statistics(merged_stats)
    print("\n" + "="*80)
    print("✅ 分析完成")
    print("="*80)
