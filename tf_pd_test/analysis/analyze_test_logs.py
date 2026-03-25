#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TensorFlow-PaddlePaddle 测试日志分析脚本

功能：
1. 分析 tf_pd_log_1 目录下的 llm_enhanced_ 开头的算子测试日志文件
2. 统计四种错误类型：
   - 两个框架都出错 (both_error)
   - 只有 TensorFlow 出错 (tensorflow_only_error)
   - 只有 PaddlePaddle 出错 (paddle_only_error)
   - 只有比较错误 (comparison_error_only)
3. 生成统计报告和分类JSON文件
"""

import os
import json
import re
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional


def get_latest_log_files(log_dir: str) -> Dict[str, str]:
    """
    获取每个算子最新的日志文件
    
    同一个算子可能有多个日志文件（时间戳不同），以最后一个文件为准
    
    Args:
        log_dir: 日志目录路径
    
    Returns:
        算子名 -> 最新日志文件路径的映射
    """
    operator_files: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    
    # 匹配文件名格式: llm_enhanced_{operator}_{YYYYMMDD}_{HHMMSS}.json
    pattern = re.compile(r'^llm_enhanced_(.+)_(\d{8})_(\d{6})\.json$')
    
    for filename in os.listdir(log_dir):
        match = pattern.match(filename)
        if match:
            operator_name = match.group(1)  # 例如: torch_abs, torch_nn_Conv2d
            timestamp = f"{match.group(2)}_{match.group(3)}"  # YYYYMMDD_HHMMSS
            filepath = os.path.join(log_dir, filename)
            operator_files[operator_name].append((timestamp, filepath))
    
    # 对每个算子，按时间戳排序，取最新的
    latest_files: Dict[str, str] = {}
    for operator, files in operator_files.items():
        # 按时间戳字符串排序（字典序即时间顺序）
        files.sort(key=lambda x: x[0])
        latest_files[operator] = files[-1][1]  # 取最后一个（最新的）
    
    return latest_files


def analyze_single_file(filepath: str) -> Dict[str, Any]:
    """
    分析单个日志文件
    
    Args:
        filepath: 日志文件路径
    
    Returns:
        分析结果字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    operator = data.get('operator', 'unknown')
    results = data.get('results', [])
    
    # 分类统计
    categories = {
        'both_error': [],           # 两个框架都出错
        'tensorflow_only_error': [],  # 只有 TensorFlow 出错
        'paddle_only_error': [],      # 只有 PaddlePaddle 出错
        'comparison_error_only': [],  # 只有比较错误（两框架都执行成功但结果不一致）
        'all_success': []             # 两框架都成功且结果一致
    }
    
    for result in results:
        exec_result = result.get('execution_result', {})
        
        tensorflow_error = exec_result.get('tensorflow_error')
        paddle_error = exec_result.get('paddle_error')
        comparison_error = exec_result.get('comparison_error')
        tensorflow_success = exec_result.get('tensorflow_success', False)
        paddle_success = exec_result.get('paddle_success', False)
        results_match = exec_result.get('results_match', False)
        
        # 构建完整的测试用例信息
        case_info = {
            'operator': operator,
            'iteration': result.get('iteration'),
            'case_number': result.get('case_number'),
            'is_llm_generated': result.get('is_llm_generated', False),
            'tensorflow_test_case': result.get('tensorflow_test_case'),
            'paddle_test_case': result.get('paddle_test_case'),
            'execution_result': exec_result,
            'llm_operation': result.get('llm_operation'),
            'source_file': os.path.basename(filepath)
        }
        
        # 判断错误类型
        has_tf_error = tensorflow_error is not None and tensorflow_error != ""
        has_paddle_error = paddle_error is not None and paddle_error != ""
        has_comparison_error = comparison_error is not None and comparison_error != ""
        
        if has_tf_error and has_paddle_error:
            # 两个框架都出错
            categories['both_error'].append(case_info)
        elif has_tf_error and not has_paddle_error:
            # 只有 TensorFlow 出错
            categories['tensorflow_only_error'].append(case_info)
        elif has_paddle_error and not has_tf_error:
            # 只有 PaddlePaddle 出错
            categories['paddle_only_error'].append(case_info)
        elif has_comparison_error:
            # 只有比较错误（两框架都执行成功但比较出错）
            categories['comparison_error_only'].append(case_info)
        elif tensorflow_success and paddle_success and results_match:
            # 完全成功
            categories['all_success'].append(case_info)
        # 其他情况（如两框架都执行成功但不一致且没有 comparison_error）
        elif tensorflow_success and paddle_success and not results_match:
            # 结果不匹配（当作 comparison_error）
            case_info['inferred_error'] = 'results_not_match'
            categories['comparison_error_only'].append(case_info)
    
    return {
        'operator': operator,
        'filepath': filepath,
        'total_cases': len(results),
        'categories': categories
    }


def run_analysis(log_dir: str, output_dir: str) -> None:
    """
    运行完整分析
    
    Args:
        log_dir: 日志文件目录
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取最新的日志文件
    latest_files = get_latest_log_files(log_dir)
    print(f"找到 {len(latest_files)} 个算子的日志文件")
    
    # 汇总统计
    total_stats = {
        'both_error': 0,
        'tensorflow_only_error': 0,
        'paddle_only_error': 0,
        'comparison_error_only': 0,
        'all_success': 0,
        'total_cases': 0
    }
    
    # 分类用例收集
    all_categories = {
        'both_error': [],
        'tensorflow_only_error': [],
        'paddle_only_error': [],
        'comparison_error_only': []
    }
    
    # 每个算子的统计
    operator_stats = []
    
    # 分析每个文件
    for operator, filepath in sorted(latest_files.items()):
        try:
            result = analyze_single_file(filepath)
            categories = result['categories']
            
            # 更新总统计
            for cat_name in total_stats.keys():
                if cat_name in categories:
                    count = len(categories[cat_name])
                    total_stats[cat_name] += count
                elif cat_name == 'total_cases':
                    total_stats['total_cases'] += result['total_cases']
            
            # 收集分类用例
            for cat_name in all_categories.keys():
                all_categories[cat_name].extend(categories.get(cat_name, []))
            
            # 记录算子统计
            op_stat = {
                'operator': result['operator'],
                'total_cases': result['total_cases'],
                'both_error': len(categories.get('both_error', [])),
                'tensorflow_only_error': len(categories.get('tensorflow_only_error', [])),
                'paddle_only_error': len(categories.get('paddle_only_error', [])),
                'comparison_error_only': len(categories.get('comparison_error_only', [])),
                'all_success': len(categories.get('all_success', []))
            }
            operator_stats.append(op_stat)
            
        except Exception as e:
            print(f"分析文件 {filepath} 时出错: {e}")
    
    # 计算总用例数
    total_stats['total_cases'] = sum([
        total_stats['both_error'],
        total_stats['tensorflow_only_error'],
        total_stats['paddle_only_error'],
        total_stats['comparison_error_only'],
        total_stats['all_success']
    ])
    
    # 生成统计报告
    report = generate_report(total_stats, operator_stats, len(latest_files))
    
    # 保存统计报告
    report_file = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n统计报告已保存到: {report_file}")
    
    # 保存JSON格式的统计数据
    summary_data = {
        'analysis_time': datetime.now().isoformat(),
        'log_directory': log_dir,
        'total_operators': len(latest_files),
        'total_stats': total_stats,
        'operator_stats': operator_stats
    }
    summary_file = os.path.join(output_dir, 'analysis_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"统计摘要已保存到: {summary_file}")
    
    # 保存各分类的测试用例到单独的JSON文件
    for cat_name, cases in all_categories.items():
        if cases:  # 只保存非空的类别
            cat_file = os.path.join(output_dir, f'{cat_name}_cases.json')
            cat_data = {
                'category': cat_name,
                'total_count': len(cases),
                'cases': cases
            }
            with open(cat_file, 'w', encoding='utf-8') as f:
                json.dump(cat_data, f, ensure_ascii=False, indent=2)
            print(f"分类 '{cat_name}' 的 {len(cases)} 个用例已保存到: {cat_file}")
    
    # 打印报告
    print("\n" + "="*80)
    print(report)


def generate_report(total_stats: Dict[str, int], 
                    operator_stats: List[Dict], 
                    total_operators: int) -> str:
    """
    生成统计报告
    
    Args:
        total_stats: 总统计数据
        operator_stats: 每个算子的统计数据
        total_operators: 算子总数
    
    Returns:
        报告文本
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("TensorFlow vs PaddlePaddle 测试日志分析报告")
    report_lines.append("=" * 80)
    report_lines.append(f"\n分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"分析的算子数量: {total_operators}")
    report_lines.append(f"分析的测试用例总数: {total_stats['total_cases']}")
    
    report_lines.append("\n" + "-" * 80)
    report_lines.append("错误分类统计汇总")
    report_lines.append("-" * 80)
    
    total = total_stats['total_cases']
    if total > 0:
        report_lines.append(f"\n1. 两个框架都出错 (both_error):          {total_stats['both_error']:>6} 个 ({total_stats['both_error']/total*100:>6.2f}%)")
        report_lines.append(f"2. 只有 TensorFlow 出错 (tf_only):        {total_stats['tensorflow_only_error']:>6} 个 ({total_stats['tensorflow_only_error']/total*100:>6.2f}%)")
        report_lines.append(f"3. 只有 PaddlePaddle 出错 (paddle_only):  {total_stats['paddle_only_error']:>6} 个 ({total_stats['paddle_only_error']/total*100:>6.2f}%)")
        report_lines.append(f"4. 只有比较错误 (comparison_error):       {total_stats['comparison_error_only']:>6} 个 ({total_stats['comparison_error_only']/total*100:>6.2f}%)")
        report_lines.append(f"5. 全部成功 (all_success):                {total_stats['all_success']:>6} 个 ({total_stats['all_success']/total*100:>6.2f}%)")
    
    # 按错误数量排序的算子列表
    report_lines.append("\n" + "-" * 80)
    report_lines.append("各算子错误统计详情")
    report_lines.append("-" * 80)
    
    # 表头
    header = f"{'算子名称':<45} {'总数':>6} {'双错':>6} {'TF错':>6} {'PD错':>6} {'比较错':>6} {'成功':>6}"
    report_lines.append("\n" + header)
    report_lines.append("-" * len(header))
    
    # 按有错误的用例数排序
    sorted_stats = sorted(operator_stats, 
                         key=lambda x: (x['both_error'] + x['tensorflow_only_error'] + 
                                       x['paddle_only_error'] + x['comparison_error_only']),
                         reverse=True)
    
    for stat in sorted_stats:
        line = (f"{stat['operator']:<45} "
                f"{stat['total_cases']:>6} "
                f"{stat['both_error']:>6} "
                f"{stat['tensorflow_only_error']:>6} "
                f"{stat['paddle_only_error']:>6} "
                f"{stat['comparison_error_only']:>6} "
                f"{stat['all_success']:>6}")
        report_lines.append(line)
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("分析说明")
    report_lines.append("=" * 80)
    report_lines.append("""
错误类型说明:
1. both_error: TensorFlow 和 PaddlePaddle 执行时都出现错误
2. tensorflow_only_error: 只有 TensorFlow 执行出错，PaddlePaddle 执行成功
3. paddle_only_error: 只有 PaddlePaddle 执行出错，TensorFlow 执行成功
4. comparison_error_only: 两个框架都执行成功，但结果比较时出现不一致
5. all_success: 两个框架都执行成功且结果一致

注意: 
- 同一算子可能有多个日志文件（时间戳不同），分析时以最新的文件为准
- 生成的各分类JSON文件包含完整的测试用例信息，可用于进一步分析
""")
    
    return "\n".join(report_lines)


def main():
    """主函数"""
    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # tf_pd_test 目录
    
    log_dir = os.path.join(base_dir, 'tf_pd_log_1')
    output_dir = script_dir  # 输出到当前 analysis 目录
    
    if not os.path.exists(log_dir):
        print(f"错误: 日志目录不存在: {log_dir}")
        return
    
    print(f"日志目录: {log_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    run_analysis(log_dir, output_dir)


if __name__ == '__main__':
    main()
