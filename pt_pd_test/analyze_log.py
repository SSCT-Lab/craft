#!/usr/bin/env python3
"""
分析批量测试日志文件，统计LLM生成用例数和成功执行用例数
"""
import re

def analyze_log_file(log_file_path):
    """
    分析日志文件，提取统计信息
    
    Args:
        log_file_path: 日志文件路径
    
    Returns:
        统计结果字典
    """
    total_llm_cases = 0
    total_successful_cases = 0
    operator_count = 0
    operator_details = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式匹配每个算子的统计信息
    # 匹配模式：[序号/总数] 算子名
    #           状态: ✅ 完成
    #           总迭代次数: X
    #           LLM生成用例数: X
    #           成功执行用例数: X
    #           成功率: X%
    
    pattern = r'\[(\d+)/(\d+)\]\s+(\S+)\s+状态:\s+✅\s+完成\s+总迭代次数:\s+(\d+)\s+LLM生成用例数:\s+(\d+)\s+成功执行用例数:\s+(\d+)(?:\s+成功率:\s+([\d.]+)%)?'
    
    matches = re.findall(pattern, content)
    
    for match in matches:
        seq_num, total_ops, operator_name, iterations, llm_cases, successful_cases, *success_rate = match
        
        llm_cases = int(llm_cases)
        successful_cases = int(successful_cases)
        
        total_llm_cases += llm_cases
        total_successful_cases += successful_cases
        operator_count += 1
        
        # 计算成功率
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
    """打印统计结果"""
    print("="*80)
    print("📊 批量测试日志统计分析")
    print("="*80)
    print(f"\n✅ 测试算子总数: {stats['operator_count']}")
    print(f"📝 LLM生成用例总数: {stats['total_llm_cases']}")
    print(f"✅ 成功执行用例总数: {stats['total_successful_cases']}")
    print(f"📈 总体成功率: {stats['overall_success_rate']:.2f}%")
    
    # 统计成功率分布
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
    
    # 显示成功率最高的前10个算子
    print(f"\n🏆 成功率最高的前10个算子:")
    sorted_ops = sorted(stats['operator_details'], key=lambda x: x['success_rate'], reverse=True)
    for i, op in enumerate(sorted_ops[:10], 1):
        print(f"  {i:2d}. {op['name']:40s} - {op['successful_cases']}/{op['llm_cases']} ({op['success_rate']:.2f}%)")
    
    # 显示成功率为0的算子
    zero_success_ops = [op for op in stats['operator_details'] if op['success_rate'] == 0 and op['llm_cases'] > 0]
    if zero_success_ops:
        print(f"\n⚠️ 成功率为0的算子（共{len(zero_success_ops)}个）:")
        for op in zero_success_ops[:20]:  # 只显示前20个
            print(f"  - {op['name']:40s} (LLM生成了{op['llm_cases']}个用例)")
        if len(zero_success_ops) > 20:
            print(f"  ... 还有 {len(zero_success_ops) - 20} 个算子")


def merge_statistics(all_stats_list):
    """
    合并多个日志文件的统计结果
    
    Args:
        all_stats_list: 多个统计结果的列表
    
    Returns:
        合并后的统计结果
    """
    merged_stats = {
        'operator_count': 0,
        'total_llm_cases': 0,
        'total_successful_cases': 0,
        'overall_success_rate': 0,
        'operator_details': []
    }
    
    # 用于去重的算子集合（按算子名）
    operator_dict = {}
    
    for stats in all_stats_list:
        merged_stats['operator_count'] += stats['operator_count']
        merged_stats['total_llm_cases'] += stats['total_llm_cases']
        merged_stats['total_successful_cases'] += stats['total_successful_cases']
        
        # 合并算子详情（如果同一个算子在多个文件中出现，累加其数据）
        for op in stats['operator_details']:
            op_name = op['name']
            if op_name in operator_dict:
                # 累加数据
                operator_dict[op_name]['llm_cases'] += op['llm_cases']
                operator_dict[op_name]['successful_cases'] += op['successful_cases']
                # 重新计算成功率
                if operator_dict[op_name]['llm_cases'] > 0:
                    operator_dict[op_name]['success_rate'] = (
                        operator_dict[op_name]['successful_cases'] / 
                        operator_dict[op_name]['llm_cases'] * 100
                    )
            else:
                # 新算子
                operator_dict[op_name] = {
                    'seq': op['seq'],
                    'name': op_name,
                    'llm_cases': op['llm_cases'],
                    'successful_cases': op['successful_cases'],
                    'success_rate': op['success_rate']
                }
    
    # 转换为列表
    merged_stats['operator_details'] = list(operator_dict.values())
    
    # 重新计算总体成功率
    if merged_stats['total_llm_cases'] > 0:
        merged_stats['overall_success_rate'] = (
            merged_stats['total_successful_cases'] / 
            merged_stats['total_llm_cases'] * 100
        )
    
    # 更新算子数量（去重后的）
    merged_stats['operator_count'] = len(merged_stats['operator_details'])
    
    return merged_stats


if __name__ == "__main__":
    # 多个日志文件路径
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
    print("📊 批量日志文件统计分析")
    print("="*80)
    print(f"\n📂 共分析 {len(log_files)} 个日志文件:\n")
    
    all_stats_list = []
    
    # 分析每个日志文件
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
        exit(1)
    
    print(f"\n{'='*80}")
    print("📊 合并统计结果")
    print("="*80)
    
    # 合并统计结果
    merged_stats = merge_statistics(all_stats_list)
    
    # 打印合并后的统计结果
    print_statistics(merged_stats)
    
    print("\n" + "="*80)
    print("✅ 分析完成")
    print("="*80)
