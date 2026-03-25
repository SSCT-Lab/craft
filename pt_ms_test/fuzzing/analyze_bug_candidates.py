#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析 pt_ms_test/fuzzing/result 目录下的 fuzzing 结果文件
统计 bug 候选的数量和错误类型分布

分析内容：
1. is_bug_candidate: true 的总数
2. 同时有 torch_error 和 mindspore_error 的数量
3. 只有 torch_error 的数量
4. 只有 mindspore_error 的数量
5. 只有 comparison_error 的数量（两框架都成功但结果不一致）

注意：
- 同一个算子可能有多个结果文件（时间戳不同），以最后一个文件为准
"""

import os
import json
import re
from collections import defaultdict
from datetime import datetime


def parse_timestamp_from_filename(filename: str) -> datetime:
    """
    从文件名中提取时间戳
    文件名格式: torch_xxx_fuzzing_result_YYYYMMDD_HHMMSS.json
    """
    match = re.search(r'_(\d{8}_\d{6})\.json$', filename)
    if match:
        timestamp_str = match.group(1)
        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    return datetime.min


def get_operator_name_from_filename(filename: str) -> str:
    """
    从文件名中提取算子名称
    文件名格式: torch_xxx_fuzzing_result_YYYYMMDD_HHMMSS.json
    """
    match = re.match(r'(.+)_fuzzing_result_\d{8}_\d{6}\.json$', filename)
    if match:
        return match.group(1)
    return filename


def analyze_fuzzing_results(result_dir: str):
    """分析 fuzzing 结果"""
    
    # 收集所有结果文件，按算子名称分组
    operator_files = defaultdict(list)
    
    for filename in os.listdir(result_dir):
        if filename.endswith('.json') and 'fuzzing_result' in filename:
            operator_name = get_operator_name_from_filename(filename)
            timestamp = parse_timestamp_from_filename(filename)
            filepath = os.path.join(result_dir, filename)
            operator_files[operator_name].append((timestamp, filepath, filename))
    
    # 对每个算子，选择最新的文件
    latest_files = {}
    for operator_name, files in operator_files.items():
        # 按时间戳排序，取最新的
        files.sort(key=lambda x: x[0], reverse=True)
        latest_files[operator_name] = files[0]  # (timestamp, filepath, filename)
    
    print(f"📁 结果目录: {result_dir}")
    print(f"📊 共发现 {len(operator_files)} 个算子的结果文件")
    print(f"   （其中有 {sum(1 for f in operator_files.values() if len(f) > 1)} 个算子有多个结果文件，已选取最新的）")
    print("=" * 80)
    
    # 统计计数器
    total_bug_candidates = 0
    both_errors = 0          # 同时有 torch_error 和 mindspore_error
    only_torch_error = 0     # 只有 torch_error
    only_mindspore_error = 0 # 只有 mindspore_error
    only_comparison_error = 0 # 只有 comparison_error
    
    # 详细记录
    bug_details = {
        'both_errors': [],
        'only_torch_error': [],
        'only_mindspore_error': [],
        'only_comparison_error': []
    }
    
    # 遍历每个算子的最新文件
    for operator_name, (timestamp, filepath, filename) in sorted(latest_files.items()):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 遍历所有测试结果
            for result_item in data.get('results', []):
                for fuzzing_result in result_item.get('fuzzing_results', []):
                    if fuzzing_result.get('is_bug_candidate', False):
                        total_bug_candidates += 1
                        
                        exec_result = fuzzing_result.get('execution_result', {})
                        torch_error = exec_result.get('torch_error')
                        mindspore_error = exec_result.get('mindspore_error')
                        comparison_error = exec_result.get('comparison_error')
                        
                        # 判断错误类型
                        has_torch_error = torch_error is not None and torch_error != ""
                        has_mindspore_error = mindspore_error is not None and mindspore_error != ""
                        has_comparison_error = comparison_error is not None and comparison_error != ""
                        
                        bug_info = {
                            'operator': operator_name,
                            'file': filename,
                            'round': fuzzing_result.get('round'),
                            'mutation_strategy': fuzzing_result.get('mutation_strategy', '')[:50],
                            'torch_error': str(torch_error)[:100] if torch_error else None,
                            'mindspore_error': str(mindspore_error)[:100] if mindspore_error else None,
                            'comparison_error': str(comparison_error)[:100] if comparison_error else None
                        }
                        
                        if has_torch_error and has_mindspore_error:
                            both_errors += 1
                            bug_details['both_errors'].append(bug_info)
                        elif has_torch_error and not has_mindspore_error:
                            only_torch_error += 1
                            bug_details['only_torch_error'].append(bug_info)
                        elif has_mindspore_error and not has_torch_error:
                            only_mindspore_error += 1
                            bug_details['only_mindspore_error'].append(bug_info)
                        elif has_comparison_error:
                            only_comparison_error += 1
                            bug_details['only_comparison_error'].append(bug_info)
        
        except Exception as e:
            print(f"⚠️ 解析文件 {filename} 失败: {e}")
    
    # 输出统计结果
    print("\n" + "=" * 80)
    print("📈 Bug 候选统计结果 (PyTorch vs MindSpore)")
    print("=" * 80)
    print(f"🔴 Bug 候选总数 (is_bug_candidate=true): {total_bug_candidates}")
    print(f"   ├── 同时有 torch_error 和 mindspore_error: {both_errors}")
    print(f"   ├── 只有 torch_error:                     {only_torch_error}")
    print(f"   ├── 只有 mindspore_error:                 {only_mindspore_error}")
    print(f"   └── 只有 comparison_error (结果不一致):   {only_comparison_error}")
    print("=" * 80)
    
    # 输出详细信息
    if bug_details['both_errors']:
        print(f"\n📋 同时有两种框架错误的 Bug 候选 ({len(bug_details['both_errors'])} 个):")
        for i, bug in enumerate(bug_details['both_errors'][:10], 1):
            print(f"  {i}. {bug['operator']} (round {bug['round']})")
            print(f"     PyTorch错误: {bug['torch_error']}")
            print(f"     MindSpore错误: {bug['mindspore_error']}")
        if len(bug_details['both_errors']) > 10:
            print(f"  ... 还有 {len(bug_details['both_errors']) - 10} 个")
    
    if bug_details['only_torch_error']:
        print(f"\n📋 只有 PyTorch 错误的 Bug 候选 ({len(bug_details['only_torch_error'])} 个):")
        for i, bug in enumerate(bug_details['only_torch_error'][:10], 1):
            print(f"  {i}. {bug['operator']} (round {bug['round']})")
            print(f"     错误: {bug['torch_error']}")
        if len(bug_details['only_torch_error']) > 10:
            print(f"  ... 还有 {len(bug_details['only_torch_error']) - 10} 个")
    
    if bug_details['only_mindspore_error']:
        print(f"\n📋 只有 MindSpore 错误的 Bug 候选 ({len(bug_details['only_mindspore_error'])} 个):")
        for i, bug in enumerate(bug_details['only_mindspore_error'][:10], 1):
            print(f"  {i}. {bug['operator']} (round {bug['round']})")
            print(f"     错误: {bug['mindspore_error']}")
        if len(bug_details['only_mindspore_error']) > 10:
            print(f"  ... 还有 {len(bug_details['only_mindspore_error']) - 10} 个")
    
    if bug_details['only_comparison_error']:
        print(f"\n📋 只有结果不一致的 Bug 候选 ({len(bug_details['only_comparison_error'])} 个):")
        for i, bug in enumerate(bug_details['only_comparison_error'][:10], 1):
            print(f"  {i}. {bug['operator']} (round {bug['round']})")
            print(f"     差异: {bug['comparison_error']}")
        if len(bug_details['only_comparison_error']) > 10:
            print(f"  ... 还有 {len(bug_details['only_comparison_error']) - 10} 个")
    
    # 保存详细报告
    report_path = os.path.join(os.path.dirname(result_dir), 'bug_candidates_report.json')
    report_data = {
        'summary': {
            'total_operators': len(latest_files),
            'total_bug_candidates': total_bug_candidates,
            'both_errors': both_errors,
            'only_torch_error': only_torch_error,
            'only_mindspore_error': only_mindspore_error,
            'only_comparison_error': only_comparison_error
        },
        'details': bug_details
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细报告已保存到: {report_path}")
    
    return report_data


if __name__ == '__main__':
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, 'result')
    
    if os.path.exists(result_dir):
        analyze_fuzzing_results(result_dir)
    else:
        print(f"❌ 结果目录不存在: {result_dir}")
