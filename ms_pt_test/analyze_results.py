#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 5: 差分测试结果分析脚本

功能：
- 读取 ms_pt_log_1/ 目录中的 JSON 测试结果文件
- 统计各算子的一致性/不一致性/错误分布
- 生成可视化统计报告（TXT + CSV + JSON）

用法：
    conda activate tf_env
    python ms_pt_test/analyze_results.py [--result-dir ms_pt_test/ms_pt_log_1]
"""

import os
import sys
import io

# Windows 环境下强制使用 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Any
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def load_all_results(result_dir: str) -> List[Dict[str, Any]]:
    """加载结果目录下的所有 JSON 结果文件"""
    results = []
    if not os.path.exists(result_dir):
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
                print(f"⚠️ 加载 {filename} 失败: {e}")
    return results


def analyze_single_operator(data: Dict[str, Any]) -> Dict[str, Any]:
    """分析单个算子的测试结果"""
    ms_api = data.get("ms_api", "unknown")
    pytorch_api = data.get("pytorch_api", "")
    iterations = data.get("results", [])

    analysis = {
        "ms_api": ms_api,
        "pytorch_api": pytorch_api,
        "total_iterations": len(iterations),
        "consistent_count": 0,
        "inconsistent_count": 0,
        "ms_error_count": 0,
        "pytorch_error_count": 0,
        "both_error_count": 0,
        "comparison_error_count": 0,
        "final_status": "unknown",
        "errors": [],
    }

    for item in iterations:
        exec_result = item.get("execution_result", {})
        status = exec_result.get("status", "unknown")

        if status == "consistent":
            analysis["consistent_count"] += 1
        elif status == "inconsistent":
            analysis["inconsistent_count"] += 1
        elif status == "ms_error":
            analysis["ms_error_count"] += 1
            if exec_result.get("ms_error"):
                analysis["errors"].append(f"[MS] {exec_result['ms_error'][:100]}")
        elif status == "pytorch_error":
            analysis["pytorch_error_count"] += 1
            if exec_result.get("pytorch_error"):
                analysis["errors"].append(f"[PT] {exec_result['pytorch_error'][:100]}")
        elif status == "both_error":
            analysis["both_error_count"] += 1
        elif status == "comparison_error":
            analysis["comparison_error_count"] += 1

    # 判定最终状态
    if analysis["consistent_count"] > 0 and analysis["inconsistent_count"] == 0:
        analysis["final_status"] = "consistent"
    elif analysis["inconsistent_count"] > 0:
        analysis["final_status"] = "inconsistent"
    elif analysis["ms_error_count"] + analysis["pytorch_error_count"] + analysis["both_error_count"] > 0:
        analysis["final_status"] = "error"
    else:
        analysis["final_status"] = "unknown"

    # 去重错误信息
    analysis["errors"] = list(set(analysis["errors"]))[:5]

    return analysis


def generate_reports(all_analyses: List[Dict[str, Any]], output_dir: str):
    """生成统计报告"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- 总体统计 ----
    total_operators = len(all_analyses)
    consistent_ops = [a for a in all_analyses if a["final_status"] == "consistent"]
    inconsistent_ops = [a for a in all_analyses if a["final_status"] == "inconsistent"]
    error_ops = [a for a in all_analyses if a["final_status"] == "error"]
    unknown_ops = [a for a in all_analyses if a["final_status"] == "unknown"]

    total_iterations = sum(a["total_iterations"] for a in all_analyses)
    total_consistent = sum(a["consistent_count"] for a in all_analyses)
    total_inconsistent = sum(a["inconsistent_count"] for a in all_analyses)
    total_ms_errors = sum(a["ms_error_count"] for a in all_analyses)
    total_pt_errors = sum(a["pytorch_error_count"] for a in all_analyses)

    # ---- 1. TXT 报告 ----
    txt_file = os.path.join(output_dir, f"analysis_report_{timestamp}.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MindSpore ↔ PyTorch 差分测试结果分析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("=" * 50 + "\n")
        f.write("📊 总体统计\n")
        f.write("=" * 50 + "\n")
        f.write(f"测试算子总数: {total_operators}\n")
        f.write(f"  ✅ 一致 (consistent): {len(consistent_ops)} "
                f"({len(consistent_ops)/max(total_operators,1)*100:.1f}%)\n")
        f.write(f"  ❌ 不一致 (inconsistent): {len(inconsistent_ops)} "
                f"({len(inconsistent_ops)/max(total_operators,1)*100:.1f}%)\n")
        f.write(f"  ⚠️ 错误 (error): {len(error_ops)} "
                f"({len(error_ops)/max(total_operators,1)*100:.1f}%)\n")
        f.write(f"  ❓ 未知 (unknown): {len(unknown_ops)}\n\n")

        f.write(f"总迭代次数: {total_iterations}\n")
        f.write(f"  一致次数: {total_consistent}\n")
        f.write(f"  不一致次数: {total_inconsistent}\n")
        f.write(f"  MS错误次数: {total_ms_errors}\n")
        f.write(f"  PT错误次数: {total_pt_errors}\n\n")

        # 一致算子列表
        f.write("=" * 50 + "\n")
        f.write(f"✅ 一致算子 ({len(consistent_ops)} 个)\n")
        f.write("=" * 50 + "\n")
        for a in sorted(consistent_ops, key=lambda x: x["ms_api"]):
            f.write(f"  {a['ms_api']} → {a['pytorch_api']} "
                    f"({a['consistent_count']}/{a['total_iterations']} 次一致)\n")

        # 不一致算子列表
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"❌ 不一致算子 ({len(inconsistent_ops)} 个)\n")
        f.write("=" * 50 + "\n")
        for a in sorted(inconsistent_ops, key=lambda x: x["ms_api"]):
            f.write(f"  {a['ms_api']} → {a['pytorch_api']}\n")
            f.write(f"    一致: {a['consistent_count']}, 不一致: {a['inconsistent_count']}\n")
            for err in a["errors"][:3]:
                f.write(f"    ! {err}\n")

        # 错误算子列表
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"⚠️ 错误算子 ({len(error_ops)} 个)\n")
        f.write("=" * 50 + "\n")
        for a in sorted(error_ops, key=lambda x: x["ms_api"]):
            f.write(f"  {a['ms_api']} → {a['pytorch_api']}\n")
            f.write(f"    MS错误: {a['ms_error_count']}, PT错误: {a['pytorch_error_count']}, "
                    f"双错误: {a['both_error_count']}\n")
            for err in a["errors"][:3]:
                f.write(f"    ! {err}\n")

    print(f"📄 TXT报告已保存: {txt_file}")

    # ---- 2. CSV 报告 ----
    csv_file = os.path.join(output_dir, f"analysis_report_{timestamp}.csv")
    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "ms_api", "pytorch_api", "final_status",
            "total_iterations", "consistent_count", "inconsistent_count",
            "ms_error_count", "pytorch_error_count", "both_error_count",
            "error_summary",
        ])
        for a in sorted(all_analyses, key=lambda x: x["ms_api"]):
            writer.writerow([
                a["ms_api"], a["pytorch_api"], a["final_status"],
                a["total_iterations"], a["consistent_count"], a["inconsistent_count"],
                a["ms_error_count"], a["pytorch_error_count"], a["both_error_count"],
                "; ".join(a["errors"][:3]) if a["errors"] else "",
            ])
    print(f"📄 CSV报告已保存: {csv_file}")

    # ---- 3. JSON 报告 ----
    json_file = os.path.join(output_dir, f"analysis_report_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_operators": total_operators,
                "consistent": len(consistent_ops),
                "inconsistent": len(inconsistent_ops),
                "error": len(error_ops),
                "unknown": len(unknown_ops),
                "total_iterations": total_iterations,
                "total_consistent_iterations": total_consistent,
                "total_inconsistent_iterations": total_inconsistent,
            },
            "operators": all_analyses,
        }, f, indent=2, ensure_ascii=False)
    print(f"📄 JSON报告已保存: {json_file}")


def main():
    parser = argparse.ArgumentParser(description="MindSpore ↔ PyTorch 差分测试结果分析")
    parser.add_argument(
        "--result-dir", "-r",
        default=os.path.join(ROOT_DIR, "ms_pt_test", "ms_pt_log_1"),
        help="测试结果目录路径",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=os.path.join(ROOT_DIR, "ms_pt_test", "analysis"),
        help="分析报告输出目录",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MindSpore ↔ PyTorch 差分测试结果分析")
    print("=" * 80)
    print(f"📁 结果目录: {args.result_dir}")
    print(f"📁 输出目录: {args.output_dir}")

    # 加载结果
    all_results = load_all_results(args.result_dir)
    if not all_results:
        print("⚠️ 未找到任何测试结果文件")
        return

    print(f"\n📋 加载了 {len(all_results)} 个算子的测试结果")

    # 分析每个算子
    all_analyses = []
    for data in all_results:
        analysis = analyze_single_operator(data)
        all_analyses.append(analysis)

    # 生成报告
    generate_reports(all_analyses, args.output_dir)

    # 打印控制台摘要
    consistent = sum(1 for a in all_analyses if a["final_status"] == "consistent")
    inconsistent = sum(1 for a in all_analyses if a["final_status"] == "inconsistent")
    error = sum(1 for a in all_analyses if a["final_status"] == "error")

    print("\n" + "=" * 50)
    print("📊 快速统计")
    print("=" * 50)
    print(f"✅ 一致: {consistent}/{len(all_analyses)}")
    print(f"❌ 不一致: {inconsistent}/{len(all_analyses)}")
    print(f"⚠️ 错误: {error}/{len(all_analyses)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
