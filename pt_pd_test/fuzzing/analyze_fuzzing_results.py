"""  
PyTorch-PaddlePaddle Fuzzing 结果分析工具

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
    
    # 支持带时间戳和不带时间戳的文件名
    json_files = sorted(result_path.glob("*_fuzzing_result*.json"))
    
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
        "error_categories": {},  # 错误分类统计
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
                    "paddle_api": data.get("paddle_api"),
                    "bug_count": bug_count
                })
            
            # 分析每个用例的 fuzzing 结果
            for result in data.get("results", []):
                for fr in result.get("fuzzing_results", []):
                    if fr.get("success"):
                        stats["success_rounds"] += 1
                        
                        if fr.get("is_bug_candidate"):
                            exec_result = fr.get("execution_result", {})
                            
                            # 错误分类
                            error_type = categorize_error(exec_result)
                            stats["error_categories"][error_type] = stats["error_categories"].get(error_type, 0) + 1
                            
                            bug_detail = {
                                "operator": data.get("operator"),
                                "torch_api": data.get("torch_api"),
                                "paddle_api": data.get("paddle_api"),
                                "round": fr.get("round"),
                                "mutation_strategy": fr.get("mutation_strategy"),
                                "error_type": error_type,
                                "torch_success": exec_result.get("torch_success"),
                                "paddle_success": exec_result.get("paddle_success"),
                                "torch_error": exec_result.get("torch_error"),
                                "paddle_error": exec_result.get("paddle_error"),
                                "comparison_error": exec_result.get("comparison_error"),
                                "torch_test_case": fr.get("torch_test_case"),
                                "paddle_test_case": fr.get("paddle_test_case"),
                            }
                            stats["bug_details"].append(bug_detail)
                    else:
                        stats["failed_rounds"] += 1
                        
        except Exception as e:
            print(f"处理 {json_file.name} 时出错: {e}")
    
    return stats


def categorize_error(exec_result: Dict[str, Any]) -> str:
    """
    对错误进行分类
    """
    torch_success = exec_result.get("torch_success", False)
    paddle_success = exec_result.get("paddle_success", False)
    comparison_error = exec_result.get("comparison_error", "")
    
    # 执行状态不一致
    if torch_success != paddle_success:
        if torch_success:
            return "PaddlePaddle 执行失败"
        else:
            return "PyTorch 执行失败"
    
    # 两者都失败
    if not torch_success and not paddle_success:
        return "双方都执行失败"
    
    # 结果比较错误
    if comparison_error:
        if "形状不一致" in comparison_error:
            return "形状不一致"
        elif "数值不一致" in comparison_error:
            return "数值不一致"
        elif "NaN" in comparison_error:
            return "NaN 位置不一致"
        elif "Inf" in comparison_error:
            return "Inf 处理不一致"
        else:
            return "其他比较错误"
    
    return "未知错误"


def generate_analysis_report(stats: Dict[str, Any], output_file: str) -> None:
    """
    生成分析报告
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch-PaddlePaddle Fuzzing 差分测试结果分析报告\n")
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
        
        # 错误分类统计
        if stats['error_categories']:
            f.write("=" * 80 + "\n")
            f.write("【错误分类统计】\n")
            f.write("-" * 40 + "\n\n")
            
            sorted_categories = sorted(
                stats['error_categories'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for error_type, count in sorted_categories:
                percentage = count / stats['total_bug_candidates'] * 100 if stats['total_bug_candidates'] > 0 else 0
                f.write(f"  {error_type}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
        
        # 有问题的算子列表
        if stats['operators_with_bugs']:
            f.write("=" * 80 + "\n")
            f.write("【存在潜在问题的算子】\n")
            f.write("-" * 40 + "\n\n")
            
            # 按问题数量排序
            sorted_operators = sorted(
                stats['operators_with_bugs'],
                key=lambda x: x['bug_count'],
                reverse=True
            )
            
            for idx, op in enumerate(sorted_operators, 1):
                f.write(f"{idx}. {op['operator']}\n")
                f.write(f"   PyTorch API: {op['torch_api']}\n")
                f.write(f"   PaddlePaddle API: {op['paddle_api']}\n")
                f.write(f"   发现问题数: {op['bug_count']}\n")
                f.write("\n")
        
        # 问题详情（限制数量避免报告过长）
        if stats['bug_details']:
            f.write("=" * 80 + "\n")
            f.write("【问题详情（前 50 个）】\n")
            f.write("-" * 40 + "\n\n")
            
            for idx, bug in enumerate(stats['bug_details'][:50], 1):
                f.write(f"问题 {idx}:\n")
                f.write("-" * 60 + "\n")
                f.write(f"算子: {bug['operator']}\n")
                f.write(f"PyTorch API: {bug['torch_api']}\n")
                f.write(f"PaddlePaddle API: {bug['paddle_api']}\n")
                f.write(f"错误类型: {bug['error_type']}\n")
                f.write(f"变异策略: {bug['mutation_strategy']}\n")
                f.write(f"PyTorch 执行状态: {'成功' if bug['torch_success'] else '失败'}\n")
                f.write(f"PaddlePaddle 执行状态: {'成功' if bug['paddle_success'] else '失败'}\n")
                
                if bug['torch_error']:
                    f.write(f"PyTorch 错误: {bug['torch_error']}\n")
                if bug['paddle_error']:
                    f.write(f"PaddlePaddle 错误: {bug['paddle_error']}\n")
                if bug['comparison_error']:
                    f.write(f"比较错误: {bug['comparison_error']}\n")
                
                f.write("\nPyTorch 测试用例:\n")
                f.write(json.dumps(bug['torch_test_case'], ensure_ascii=False, indent=2))
                f.write("\n\nPaddlePaddle 测试用例:\n")
                f.write(json.dumps(bug['paddle_test_case'], ensure_ascii=False, indent=2))
                f.write("\n\n")
            
            if len(stats['bug_details']) > 50:
                f.write(f"\n... 还有 {len(stats['bug_details']) - 50} 个问题未列出\n\n")
        else:
            f.write("=" * 80 + "\n")
            f.write("【问题详情】\n")
            f.write("-" * 40 + "\n")
            f.write("未发现任何潜在问题。\n\n")
        
        # 建议
        f.write("=" * 80 + "\n")
        f.write("【分析建议】\n")
        f.write("-" * 40 + "\n")
        f.write("1. 数值不一致问题：检查是否为浮点精度误差（< 1e-5 通常可忽略）\n")
        f.write("2. 形状不一致问题：检查两框架对输入维度的处理是否有差异\n")
        f.write("3. 执行失败问题：检查 API 参数兼容性和数据类型支持\n")
        f.write("4. NaN/Inf 问题：检查边界值处理的差异\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 80 + "\n")


def generate_summary_json(stats: Dict[str, Any], output_file: str) -> None:
    """
    生成 JSON 格式的统计摘要
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_operators": stats["total_operators"],
        "total_cases": stats["total_cases"],
        "total_fuzzing_rounds": stats["total_fuzzing_rounds"],
        "success_rounds": stats["success_rounds"],
        "failed_rounds": stats["failed_rounds"],
        "total_bug_candidates": stats["total_bug_candidates"],
        "operators_with_bugs_count": len(stats["operators_with_bugs"]),
        "error_categories": stats["error_categories"],
        "operators_with_bugs": stats["operators_with_bugs"],
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"统计摘要 JSON 已保存到: {output_file}")


def main():
    """
    主程序入口
    """
    result_dir = Path(__file__).parent / "result"
    report_file = Path(__file__).parent / "fuzzing_analysis_report.txt"
    summary_file = Path(__file__).parent / "fuzzing_analysis_summary.json"
    
    print("=" * 60)
    print("PyTorch-PaddlePaddle Fuzzing 结果分析")
    print("=" * 60)
    print(f"结果目录: {result_dir}")
    
    # 检查结果目录是否存在
    if not result_dir.exists():
        print(f"\n[WARN] 结果目录不存在: {result_dir}")
        print("请先运行 llm_fuzzing_diff_test.py 生成测试结果")
        return
    
    print("\n分析 Fuzzing 结果...")
    
    stats = analyze_fuzzing_results(str(result_dir))
    
    # 打印统计摘要
    print(f"\n统计摘要:")
    print(f"  - 测试算子数: {stats['total_operators']}")
    print(f"  - 测试用例数: {stats['total_cases']}")
    print(f"  - 总 Fuzzing 轮次: {stats['total_fuzzing_rounds']}")
    print(f"  - 成功执行轮次: {stats['success_rounds']}")
    print(f"  - 执行失败轮次: {stats['failed_rounds']}")
    print(f"  - 发现潜在问题数: {stats['total_bug_candidates']}")
    print(f"  - 有问题的算子数: {len(stats['operators_with_bugs'])}")
    
    if stats['error_categories']:
        print(f"\n错误分类:")
        for error_type, count in sorted(stats['error_categories'].items(), key=lambda x: x[1], reverse=True):
            print(f"    - {error_type}: {count}")
    
    # 生成报告
    generate_analysis_report(stats, str(report_file))
    print(f"\n分析报告已生成: {report_file}")
    
    # 生成 JSON 摘要
    generate_summary_json(stats, str(summary_file))
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
