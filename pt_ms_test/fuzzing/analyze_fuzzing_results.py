"""
PyTorch-MindSpore Fuzzing 差分测试结果分析工具

功能说明:
    1. 读取 fuzzing 测试结果
    2. 统计 bug 候选、错误类型分布
    3. 生成分析报告
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict


# 结果目录
RESULT_DIR = Path(__file__).parent / "result"


def load_fuzzing_results(result_dir: Path) -> List[Dict[str, Any]]:
    """
    加载所有 fuzzing 结果文件
    """
    results = []
    result_files = sorted(result_dir.glob("*_fuzzing_result_*.json"))
    
    for file in result_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_source_file"] = file.name
                results.append(data)
        except Exception as e:
            print(f"[WARN] 加载 {file.name} 失败: {e}")
    
    return results


def categorize_error(error_msg: str) -> str:
    """
    对错误信息进行分类
    """
    if error_msg is None:
        return "无错误"
    
    error_lower = error_msg.lower()
    
    # 形状相关
    if "shape" in error_lower or "dimension" in error_lower or "size" in error_lower:
        return "形状不匹配"
    
    # 数据类型相关
    if "dtype" in error_lower or "type" in error_lower:
        return "数据类型错误"
    
    # 数值相关
    if "nan" in error_lower or "inf" in error_lower or "overflow" in error_lower:
        return "数值异常"
    
    # 内存相关
    if "memory" in error_lower or "oom" in error_lower or "cuda" in error_lower:
        return "内存/设备错误"
    
    # MindSpore 特有错误
    if "mindspore" in error_lower:
        if "not support" in error_lower or "unsupported" in error_lower:
            return "MindSpore不支持"
        if "graph" in error_lower or "compile" in error_lower:
            return "MindSpore编译错误"
    
    # 参数相关
    if "argument" in error_lower or "param" in error_lower or "invalid" in error_lower:
        return "参数错误"
    
    # 属性相关
    if "attribute" in error_lower or "has no" in error_lower:
        return "属性/方法不存在"
    
    # 值不一致
    if "不一致" in error_msg or "mismatch" in error_lower:
        return "结果不一致"
    
    return "其他错误"


def analyze_single_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析单个算子的 fuzzing 结果
    """
    analysis = {
        "operator": result.get("operator", "unknown"),
        "torch_api": result.get("torch_api", ""),
        "mindspore_api": result.get("mindspore_api", ""),
        "total_cases": result.get("total_cases", 0),
        "total_fuzzing_rounds": result.get("total_fuzzing_rounds", 0),
        "bug_candidates": result.get("bug_candidates", 0),
        "error_categories": defaultdict(int),
        "bug_details": [],
        "execution_stats": {
            "torch_success": 0,
            "torch_fail": 0,
            "mindspore_success": 0,
            "mindspore_fail": 0,
            "both_success_match": 0,
            "both_success_mismatch": 0
        }
    }
    
    for case_result in result.get("results", []):
        for fuzz_result in case_result.get("fuzzing_results", []):
            if not fuzz_result.get("success", False):
                analysis["error_categories"]["Fuzzing 失败"] += 1
                continue
            
            exec_result = fuzz_result.get("execution_result", {})
            
            # 统计执行情况
            torch_ok = exec_result.get("torch_success", False)
            ms_ok = exec_result.get("mindspore_success", False)
            
            if torch_ok:
                analysis["execution_stats"]["torch_success"] += 1
            else:
                analysis["execution_stats"]["torch_fail"] += 1
            
            if ms_ok:
                analysis["execution_stats"]["mindspore_success"] += 1
            else:
                analysis["execution_stats"]["mindspore_fail"] += 1
            
            if torch_ok and ms_ok:
                if exec_result.get("results_match", False):
                    analysis["execution_stats"]["both_success_match"] += 1
                else:
                    analysis["execution_stats"]["both_success_mismatch"] += 1
            
            # 分析潜在 bug
            if fuzz_result.get("is_bug_candidate", False):
                error_msg = exec_result.get("comparison_error") or \
                           exec_result.get("torch_error") or \
                           exec_result.get("mindspore_error")
                category = categorize_error(error_msg)
                analysis["error_categories"][category] += 1
                
                analysis["bug_details"].append({
                    "round": fuzz_result.get("round"),
                    "mutation_strategy": fuzz_result.get("mutation_strategy", ""),
                    "error_category": category,
                    "error_detail": error_msg,
                    "torch_success": torch_ok,
                    "mindspore_success": ms_ok,
                    "torch_shape": exec_result.get("torch_shape"),
                    "mindspore_shape": exec_result.get("mindspore_shape"),
                    "original_case_info": case_result.get("original_case_info", {})
                })
    
    # 转换 defaultdict 为普通 dict
    analysis["error_categories"] = dict(analysis["error_categories"])
    
    return analysis


def generate_summary_report(all_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    生成汇总报告
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_operators": len(all_analyses),
        "total_cases": 0,
        "total_fuzzing_rounds": 0,
        "total_bug_candidates": 0,
        "operators_with_bugs": 0,
        "global_error_categories": defaultdict(int),
        "global_execution_stats": {
            "torch_success": 0,
            "torch_fail": 0,
            "mindspore_success": 0,
            "mindspore_fail": 0,
            "both_success_match": 0,
            "both_success_mismatch": 0
        },
        "operator_summary": [],
        "top_bug_operators": []
    }
    
    for analysis in all_analyses:
        summary["total_cases"] += analysis["total_cases"]
        summary["total_fuzzing_rounds"] += analysis["total_fuzzing_rounds"]
        summary["total_bug_candidates"] += analysis["bug_candidates"]
        
        if analysis["bug_candidates"] > 0:
            summary["operators_with_bugs"] += 1
        
        # 汇总错误分类
        for category, count in analysis["error_categories"].items():
            summary["global_error_categories"][category] += count
        
        # 汇总执行统计
        for key in summary["global_execution_stats"]:
            summary["global_execution_stats"][key] += analysis["execution_stats"].get(key, 0)
        
        # 算子摘要
        summary["operator_summary"].append({
            "operator": analysis["operator"],
            "torch_api": analysis["torch_api"],
            "mindspore_api": analysis["mindspore_api"],
            "cases": analysis["total_cases"],
            "fuzzing_rounds": analysis["total_fuzzing_rounds"],
            "bug_candidates": analysis["bug_candidates"],
            "bug_rate": f"{analysis['bug_candidates'] / max(1, analysis['total_fuzzing_rounds']) * 100:.1f}%"
        })
    
    # 转换 defaultdict
    summary["global_error_categories"] = dict(summary["global_error_categories"])
    
    # 按 bug 数量排序
    summary["top_bug_operators"] = sorted(
        summary["operator_summary"],
        key=lambda x: x["bug_candidates"],
        reverse=True
    )[:20]
    
    # 计算整体比率
    total_rounds = max(1, summary["total_fuzzing_rounds"])
    summary["overall_bug_rate"] = f"{summary['total_bug_candidates'] / total_rounds * 100:.2f}%"
    
    return summary


def print_report(summary: Dict[str, Any]) -> None:
    """
    打印分析报告
    """
    print("\n" + "=" * 80)
    print("PyTorch-MindSpore Fuzzing 差分测试结果分析报告")
    print("=" * 80)
    print(f"生成时间: {summary['timestamp']}")
    print()
    
    # 概述
    print("【总体概述】")
    print(f"  分析算子数: {summary['total_operators']}")
    print(f"  总测试用例数: {summary['total_cases']}")
    print(f"  总 Fuzzing 轮数: {summary['total_fuzzing_rounds']}")
    print(f"  发现潜在问题数: {summary['total_bug_candidates']}")
    print(f"  有问题的算子数: {summary['operators_with_bugs']}")
    print(f"  整体问题发现率: {summary['overall_bug_rate']}")
    print()
    
    # 执行统计
    print("【执行统计】")
    stats = summary["global_execution_stats"]
    print(f"  PyTorch 执行成功: {stats['torch_success']}")
    print(f"  PyTorch 执行失败: {stats['torch_fail']}")
    print(f"  MindSpore 执行成功: {stats['mindspore_success']}")
    print(f"  MindSpore 执行失败: {stats['mindspore_fail']}")
    print(f"  双方成功且结果一致: {stats['both_success_match']}")
    print(f"  双方成功但结果不一致: {stats['both_success_mismatch']}")
    print()
    
    # 错误分类
    print("【问题分类统计】")
    categories = summary["global_error_categories"]
    if categories:
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories:
            print(f"  {category}: {count}")
    else:
        print("  无问题记录")
    print()
    
    # Top bug 算子
    print("【问题最多的算子 (Top 10)】")
    for i, op in enumerate(summary["top_bug_operators"][:10], 1):
        if op["bug_candidates"] > 0:
            print(f"  {i}. {op['operator']}: {op['bug_candidates']} 个问题 "
                  f"(问题率: {op['bug_rate']}, PyTorch: {op['torch_api']}, MindSpore: {op['mindspore_api']})")
    
    print("\n" + "=" * 80)


def export_detailed_bugs(all_analyses: List[Dict[str, Any]], output_file: Path) -> None:
    """
    导出详细的 bug 列表
    """
    all_bugs = []
    
    for analysis in all_analyses:
        for bug in analysis["bug_details"]:
            all_bugs.append({
                "operator": analysis["operator"],
                "torch_api": analysis["torch_api"],
                "mindspore_api": analysis["mindspore_api"],
                **bug
            })
    
    # 按错误类别分组
    bugs_by_category = defaultdict(list)
    for bug in all_bugs:
        bugs_by_category[bug["error_category"]].append(bug)
    
    output = {
        "total_bugs": len(all_bugs),
        "bugs_by_category": {k: len(v) for k, v in bugs_by_category.items()},
        "detailed_bugs": dict(bugs_by_category)
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n详细 bug 列表已导出: {output_file}")


def main():
    """
    主程序入口
    """
    parser = argparse.ArgumentParser(
        description="PyTorch-MindSpore Fuzzing 差分测试结果分析"
    )
    parser.add_argument(
        "--result-dir", "-r",
        type=Path,
        default=RESULT_DIR,
        help=f"结果文件目录（默认 {RESULT_DIR}）"
    )
    parser.add_argument(
        "--export", "-e",
        type=Path,
        default=None,
        help="导出详细 bug 列表的文件路径"
    )
    parser.add_argument(
        "--summary-output", "-s",
        type=Path,
        default=None,
        help="保存汇总报告的 JSON 文件路径"
    )
    
    args = parser.parse_args()
    
    # 加载结果
    print(f"从 {args.result_dir} 加载 fuzzing 结果...")
    results = load_fuzzing_results(args.result_dir)
    
    if not results:
        print("[WARN] 未找到任何 fuzzing 结果文件")
        return
    
    print(f"已加载 {len(results)} 个结果文件")
    
    # 分析每个结果
    all_analyses = []
    for result in results:
        analysis = analyze_single_result(result)
        all_analyses.append(analysis)
    
    # 生成汇总报告
    summary = generate_summary_report(all_analyses)
    
    # 打印报告
    print_report(summary)
    
    # 保存汇总报告
    if args.summary_output:
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"汇总报告已保存: {args.summary_output}")
    
    # 导出详细 bug 列表
    if args.export:
        export_detailed_bugs(all_analyses, args.export)
    
    # 默认导出
    default_export = args.result_dir / f"bug_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    export_detailed_bugs(all_analyses, default_export)


if __name__ == "__main__":
    main()
