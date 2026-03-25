"""  
PyTorch-TensorFlow Fuzzing 结果分析工具

功能说明:
    1. 分析 fuzzing 结果目录下所有 JSON 文件
    2. 统计发现的潜在问题
    3. 生成详细的分析报告
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def analyze_fuzzing_results(result_dir: str) -> Dict[str, Any]:
    """
    分析 fuzzing 结果
    """
    result_path = Path(result_dir)
    json_files = sorted(result_path.glob("*_fuzzing_result.json"))
    
    print(f"找到 {len(json_files)} 个结果文件")
    
    # 统计数据
    stats = {
        "total_operators": 0,
        "total_cases": 0,
        "total_fuzzing_rounds": 0,
        "total_bug_candidates": 0,
        "operators_with_bugs": [],
        "bug_details": [],
        "success_rounds": 0,
        "failed_rounds": 0,
    }
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            stats["total_operators"] += 1
            stats["total_cases"] += data.get("total_cases", 0)
            stats["total_fuzzing_rounds"] += data.get("total_fuzzing_rounds", 0)
            
            bug_count = data.get("bug_candidates", 0)
            stats["total_bug_candidates"] += bug_count
            
            if bug_count > 0:
                stats["operators_with_bugs"].append({
                    "operator": data.get("operator"),
                    "torch_api": data.get("torch_api"),
                    "tensorflow_api": data.get("tensorflow_api"),
                    "bug_count": bug_count
                })
            
            # 分析每个用例的 fuzzing 结果
            for result in data.get("results", []):
                for fr in result.get("fuzzing_results", []):
                    if fr.get("success"):
                        stats["success_rounds"] += 1
                        
                        if fr.get("is_bug_candidate"):
                            exec_result = fr.get("execution_result", {})
                            bug_detail = {
                                "operator": data.get("operator"),
                                "torch_api": data.get("torch_api"),
                                "tensorflow_api": data.get("tensorflow_api"),
                                "round": fr.get("round"),
                                "mutation_strategy": fr.get("mutation_strategy"),
                                "torch_success": exec_result.get("torch_success"),
                                "tensorflow_success": exec_result.get("tensorflow_success"),
                                "torch_error": exec_result.get("torch_error"),
                                "tensorflow_error": exec_result.get("tensorflow_error"),
                                "comparison_error": exec_result.get("comparison_error"),
                                "torch_test_case": fr.get("torch_test_case"),
                                "tensorflow_test_case": fr.get("tensorflow_test_case"),
                            }
                            stats["bug_details"].append(bug_detail)
                    else:
                        stats["failed_rounds"] += 1
                        
        except Exception as e:
            print(f"处理 {json_file.name} 时出错: {e}")
    
    return stats


def generate_analysis_report(stats: Dict[str, Any], output_file: str) -> None:
    """
    生成分析报告
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch-TensorFlow Fuzzing 差分测试结果分析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # 总体统计
        f.write("【总体统计】\n")
        f.write("-" * 40 + "\n")
        f.write(f"测试算子数: {stats['total_operators']}\n")
        f.write(f"测试用例数: {stats['total_cases']}\n")
        f.write(f"总 Fuzzing 轮次: {stats['total_fuzzing_rounds']}\n")
        f.write(f"成功执行轮次: {stats['success_rounds']}\n")
        f.write(f"执行失败轮次: {stats['failed_rounds']}\n")
        f.write(f"发现潜在问题数: {stats['total_bug_candidates']}\n")
        f.write(f"有问题的算子数: {len(stats['operators_with_bugs'])}\n")
        f.write("\n")
        
        # 有问题的算子列表
        if stats['operators_with_bugs']:
            f.write("=" * 80 + "\n")
            f.write("【存在潜在问题的算子】\n")
            f.write("-" * 40 + "\n\n")
            
            for idx, op in enumerate(stats['operators_with_bugs'], 1):
                f.write(f"{idx}. {op['operator']}\n")
                f.write(f"   PyTorch API: {op['torch_api']}\n")
                f.write(f"   TensorFlow API: {op['tensorflow_api']}\n")
                f.write(f"   发现问题数: {op['bug_count']}\n")
                f.write("\n")
        
        # 问题详情
        if stats['bug_details']:
            f.write("=" * 80 + "\n")
            f.write("【问题详情】\n")
            f.write("-" * 40 + "\n\n")
            
            for idx, bug in enumerate(stats['bug_details'], 1):
                f.write(f"问题 {idx}:\n")
                f.write("-" * 60 + "\n")
                f.write(f"算子: {bug['operator']}\n")
                f.write(f"PyTorch API: {bug['torch_api']}\n")
                f.write(f"TensorFlow API: {bug['tensorflow_api']}\n")
                f.write(f"变异策略: {bug['mutation_strategy']}\n")
                f.write(f"PyTorch 执行状态: {'成功' if bug['torch_success'] else '失败'}\n")
                f.write(f"TensorFlow 执行状态: {'成功' if bug['tensorflow_success'] else '失败'}\n")
                
                if bug['torch_error']:
                    f.write(f"PyTorch 错误: {bug['torch_error']}\n")
                if bug['tensorflow_error']:
                    f.write(f"TensorFlow 错误: {bug['tensorflow_error']}\n")
                if bug['comparison_error']:
                    f.write(f"比较错误: {bug['comparison_error']}\n")
                
                f.write("\nPyTorch 测试用例:\n")
                f.write(json.dumps(bug['torch_test_case'], ensure_ascii=False, indent=2))
                f.write("\n\nTensorFlow 测试用例:\n")
                f.write(json.dumps(bug['tensorflow_test_case'], ensure_ascii=False, indent=2))
                f.write("\n\n")
        else:
            f.write("=" * 80 + "\n")
            f.write("【问题详情】\n")
            f.write("-" * 40 + "\n")
            f.write("未发现任何潜在问题。\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 80 + "\n")


def main():
    """
    主程序入口
    """
    result_dir = Path(__file__).parent / "result"
    report_file = Path(__file__).parent / "fuzzing_analysis_report.txt"
    
    print("=" * 60)
    print("分析 Fuzzing 结果...")
    print("=" * 60)
    
    stats = analyze_fuzzing_results(str(result_dir))
    
    # 打印统计摘要
    print(f"\n统计摘要:")
    print(f"  - 测试算子数: {stats['total_operators']}")
    print(f"  - 测试用例数: {stats['total_cases']}")
    print(f"  - 总 Fuzzing 轮次: {stats['total_fuzzing_rounds']}")
    print(f"  - 发现潜在问题数: {stats['total_bug_candidates']}")
    
    # 生成报告
    generate_analysis_report(stats, str(report_file))
    print(f"\n分析报告已生成: {report_file}")


if __name__ == "__main__":
    main()
