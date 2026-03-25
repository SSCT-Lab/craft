#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 5: 分析 PD↔PT 差分测试结果

功能：
- 读取 pd_pt_log_1/ 目录下的所有 JSON 结果文件
- 生成统计报告：一致/不一致/出错的算子分布
- 分析 LLM 修复/变异的效果
- 输出详细的分析结果

用法：
    conda activate tf_env
    python pd_pt_test/analyze_results.py [--result-dir pd_pt_test/pd_pt_log_1] [--output pd_pt_test/analysis]
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_result_files(result_dir: str) -> List[Dict[str, Any]]:
    """加载所有结果 JSON 文件"""
    results = []
    if not os.path.isdir(result_dir):
        print(f"❌ 结果目录不存在: {result_dir}")
        return results

    for filename in sorted(os.listdir(result_dir)):
        if filename.startswith("llm_enhanced_") and filename.endswith(".json"):
            filepath = os.path.join(result_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                results.append(data)
            except Exception as e:
                print(f"  ⚠️ 读取失败: {filename}: {e}")

    return results


def analyze_operator_results(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """分析单个算子的测试结果"""
    pd_api = result_data.get("pd_api", "unknown")
    pytorch_api = result_data.get("pytorch_api", "unknown")
    iterations = result_data.get("results", [])

    analysis = {
        "pd_api": pd_api,
        "pytorch_api": pytorch_api,
        "total_iterations": len(iterations),
        "consistent_count": 0,
        "inconsistent_count": 0,
        "pd_error_count": 0,
        "pytorch_error_count": 0,
        "both_error_count": 0,
        "llm_mutations": 0,
        "llm_repairs": 0,
        "llm_skips": 0,
        "llm_generated_count": result_data.get("llm_generated_test_cases", 0),
        "successful_count": result_data.get("successful_test_cases", 0),
        "final_status": "unknown",
        "inconsistency_details": [],
        "error_details": [],
    }

    for iter_data in iterations:
        exec_result = iter_data.get("execution_result", {})
        status = exec_result.get("status", "unknown")

        if status == "consistent":
            analysis["consistent_count"] += 1
        elif status == "inconsistent":
            analysis["inconsistent_count"] += 1
            detail = exec_result.get("comparison_error", "")
            if detail:
                analysis["inconsistency_details"].append(detail)
        elif status == "paddle_error":
            analysis["pd_error_count"] += 1
            analysis["error_details"].append(f"PD: {exec_result.get('pd_error', '')[:100]}")
        elif status == "pytorch_error":
            analysis["pytorch_error_count"] += 1
            analysis["error_details"].append(f"PT: {exec_result.get('pytorch_error', '')[:100]}")
        elif status == "both_error":
            analysis["both_error_count"] += 1

        llm_op = iter_data.get("llm_operation", {})
        if isinstance(llm_op, dict):
            op_type = llm_op.get("operation", "")
            if op_type == "mutation":
                analysis["llm_mutations"] += 1
            elif op_type == "repair":
                analysis["llm_repairs"] += 1
            elif op_type == "skip":
                analysis["llm_skips"] += 1

    # 确定最终状态
    if analysis["consistent_count"] > 0 and analysis["inconsistent_count"] == 0:
        analysis["final_status"] = "all_consistent"
    elif analysis["inconsistent_count"] > 0:
        analysis["final_status"] = "has_inconsistency"
    elif analysis["pd_error_count"] + analysis["pytorch_error_count"] + analysis["both_error_count"] == analysis["total_iterations"]:
        analysis["final_status"] = "all_error"
    else:
        analysis["final_status"] = "mixed"

    return analysis


def generate_summary_report(all_analyses: List[Dict[str, Any]], output_dir: str):
    """生成汇总报告"""
    os.makedirs(output_dir, exist_ok=True)

    # ---- 统计汇总 ----
    total_ops = len(all_analyses)
    status_counts = Counter(a["final_status"] for a in all_analyses)
    total_iterations = sum(a["total_iterations"] for a in all_analyses)
    total_consistent = sum(a["consistent_count"] for a in all_analyses)
    total_inconsistent = sum(a["inconsistent_count"] for a in all_analyses)
    total_llm_generated = sum(a["llm_generated_count"] for a in all_analyses)
    total_successful = sum(a["successful_count"] for a in all_analyses)
    total_mutations = sum(a["llm_mutations"] for a in all_analyses)
    total_repairs = sum(a["llm_repairs"] for a in all_analyses)
    total_skips = sum(a["llm_skips"] for a in all_analyses)

    # ---- 控制台输出 ----
    print("\n" + "=" * 80)
    print("📊 PD↔PT 差分测试结果分析报告")
    print("=" * 80)

    print(f"\n1. 算子级别统计")
    print(f"   总算子数: {total_ops}")
    print(f"   全部一致: {status_counts.get('all_consistent', 0)}")
    print(f"   存在不一致: {status_counts.get('has_inconsistency', 0)}")
    print(f"   全部报错: {status_counts.get('all_error', 0)}")
    print(f"   混合状态: {status_counts.get('mixed', 0)}")

    print(f"\n2. 迭代级别统计")
    print(f"   总迭代次数: {total_iterations}")
    print(f"   结果一致: {total_consistent}")
    print(f"   结果不一致: {total_inconsistent}")

    print(f"\n3. LLM 操作统计")
    print(f"   LLM 生成用例数: {total_llm_generated}")
    print(f"   成功执行用例数: {total_successful}")
    if total_llm_generated > 0:
        print(f"   成功率: {total_successful / total_llm_generated * 100:.2f}%")
    print(f"   变异次数: {total_mutations}")
    print(f"   修复次数: {total_repairs}")
    print(f"   跳过次数: {total_skips}")

    # ---- 不一致的算子列表 ----
    inconsistent_ops = [a for a in all_analyses if a["final_status"] == "has_inconsistency"]
    if inconsistent_ops:
        print(f"\n4. 发现不一致的算子 ({len(inconsistent_ops)} 个)")
        print(f"   {'PD API':<40} {'PT API':<40} {'一致/不一致'}")
        print(f"   {'-' * 40} {'-' * 40} {'-' * 12}")
        for a in inconsistent_ops:
            print(f"   {a['pd_api']:<40} {a['pytorch_api']:<40} {a['consistent_count']}/{a['inconsistent_count']}")

    # ---- 保存详细报告 ----
    report_file = os.path.join(output_dir, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PD↔PT 差分测试详细分析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. 总体统计\n")
        f.write(f"   总算子数: {total_ops}\n")
        f.write(f"   全部一致: {status_counts.get('all_consistent', 0)}\n")
        f.write(f"   存在不一致: {status_counts.get('has_inconsistency', 0)}\n")
        f.write(f"   全部报错: {status_counts.get('all_error', 0)}\n")
        f.write(f"   混合状态: {status_counts.get('mixed', 0)}\n\n")

        f.write("2. LLM 操作统计\n")
        f.write(f"   生成用例数: {total_llm_generated}\n")
        f.write(f"   成功执行数: {total_successful}\n")
        f.write(f"   变异: {total_mutations}, 修复: {total_repairs}, 跳过: {total_skips}\n\n")

        f.write("3. 各算子详细结果\n")
        f.write("-" * 80 + "\n")
        for a in sorted(all_analyses, key=lambda x: x["pd_api"]):
            f.write(f"\n  {a['pd_api']} → {a['pytorch_api']}\n")
            f.write(f"    状态: {a['final_status']}\n")
            f.write(f"    迭代: {a['total_iterations']}, 一致: {a['consistent_count']}, 不一致: {a['inconsistent_count']}\n")
            f.write(f"    LLM: 生成={a['llm_generated_count']}, 成功={a['successful_count']}\n")
            if a["inconsistency_details"]:
                f.write(f"    不一致详情:\n")
                for detail in a["inconsistency_details"][:3]:
                    f.write(f"      - {detail}\n")
            if a["error_details"]:
                f.write(f"    错误详情:\n")
                for detail in a["error_details"][:3]:
                    f.write(f"      - {detail}\n")

    print(f"\n💾 详细报告已保存到: {report_file}")

    # ---- 保存 CSV 摘要 ----
    csv_file = os.path.join(output_dir, f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pd_api", "pytorch_api", "final_status",
            "total_iterations", "consistent_count", "inconsistent_count",
            "llm_generated_count", "successful_count",
            "llm_mutations", "llm_repairs", "llm_skips",
        ])
        writer.writeheader()
        for a in sorted(all_analyses, key=lambda x: x["pd_api"]):
            writer.writerow({k: a[k] for k in writer.fieldnames})

    print(f"💾 CSV摘要已保存到: {csv_file}")

    # ---- 保存 JSON 摘要 ----
    json_file = os.path.join(output_dir, f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_operators": total_ops,
                "status_distribution": dict(status_counts),
                "total_iterations": total_iterations,
                "total_consistent": total_consistent,
                "total_inconsistent": total_inconsistent,
                "total_llm_generated": total_llm_generated,
                "total_successful": total_successful,
                "llm_mutations": total_mutations,
                "llm_repairs": total_repairs,
                "llm_skips": total_skips,
            },
            "operators": all_analyses,
        }, f, indent=2, ensure_ascii=False)

    print(f"💾 JSON数据已保存到: {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: 分析 PD↔PT 差分测试结果"
    )
    parser.add_argument(
        "--result-dir", "-r",
        default=os.path.join(ROOT_DIR, "pd_pt_test", "pd_pt_log_1"),
        help="结果目录路径"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "pd_pt_test", "analysis"),
        help="分析输出目录"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Step 5: PD↔PT 差分测试结果分析")
    print("=" * 80)
    print(f"📁 结果目录: {args.result_dir}")

    # 加载结果
    result_files = load_result_files(args.result_dir)
    print(f"📋 加载了 {len(result_files)} 个结果文件")

    if not result_files:
        print("⚠️ 没有找到结果文件，请先运行 Step 4")
        return

    # 分析每个算子
    all_analyses = []
    for result_data in result_files:
        analysis = analyze_operator_results(result_data)
        all_analyses.append(analysis)

    # 生成报告
    generate_summary_report(all_analyses, args.output)

    print("\n✅ 分析完成")


if __name__ == "__main__":
    main()
