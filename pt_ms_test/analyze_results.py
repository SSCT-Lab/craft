#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 5: 差分测试结果分析脚本（PyTorch -> MindSpore）

功能：
- 读取 pt_ms_test/pt_ms_log_1 目录中的 JSON 测试结果文件
- 若同一算子存在多个 JSON 文件，仅保留时间戳最新的文件
- 统计各算子的一致性/不一致性/错误分布
- 生成可视化统计报告（TXT + CSV + JSON）

用法：
    conda activate tf_env
    python pt_ms_test/analyze_results.py [--result-dir pt_ms_test/pt_ms_log_1]
"""

import os
import sys
import io

# Windows 环境下强制使用 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import re
import json
import csv
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def parse_datetime_from_filename(filename: str) -> Optional[datetime]:
    """从文件名中解析时间戳: *_YYYYmmdd_HHMMSS.json"""
    match = re.search(r"_(\d{8}_\d{6})\.json$", filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def parse_record_timestamp(data: Dict[str, Any], filename: str) -> datetime:
    """优先用 JSON 内 timestamp，其次回退到文件名时间戳，最后给最小值。"""
    timestamp_text = data.get("timestamp")
    if isinstance(timestamp_text, str):
        try:
            return datetime.fromisoformat(timestamp_text)
        except ValueError:
            pass

    filename_dt = parse_datetime_from_filename(filename)
    if filename_dt is not None:
        return filename_dt

    return datetime.min


def normalize_operator_name(data: Dict[str, Any], filename: str) -> str:
    """提取算子名，优先使用 JSON 内 operator。"""
    operator = data.get("operator")
    if isinstance(operator, str) and operator.strip():
        return operator.strip()

    # 兜底：从文件名中提取 llm_enhanced_ 与时间戳之间部分
    base = os.path.basename(filename)
    match = re.match(r"^llm_enhanced_(.+)_\d{8}_\d{6}\.json$", base)
    if match:
        return match.group(1)

    return base


def load_latest_results_by_operator(result_dir: str) -> Tuple[List[Dict[str, Any]], int]:
    """加载结果目录下所有 JSON，并为每个算子保留时间戳最新的一份。"""
    latest_by_operator: Dict[str, Dict[str, Any]] = {}
    total_json_files = 0

    if not os.path.exists(result_dir):
        print(f"❌ 结果目录不存在: {result_dir}")
        return [], total_json_files

    for filename in sorted(os.listdir(result_dir)):
        if not (filename.startswith("llm_enhanced_") and filename.endswith(".json")):
            continue

        filepath = os.path.join(result_dir, filename)
        total_json_files += 1

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ 加载 {filename} 失败: {e}")
            continue

        operator_name = normalize_operator_name(data, filename)
        record_dt = parse_record_timestamp(data, filename)

        prev = latest_by_operator.get(operator_name)
        if prev is None:
            latest_by_operator[operator_name] = {
                "data": data,
                "dt": record_dt,
                "filename": filename,
            }
            continue

        # 新时间更晚则替换；时间相同时按文件名字典序兜底，保证确定性
        if record_dt > prev["dt"] or (record_dt == prev["dt"] and filename > prev["filename"]):
            latest_by_operator[operator_name] = {
                "data": data,
                "dt": record_dt,
                "filename": filename,
            }

    selected_results = [entry["data"] for entry in latest_by_operator.values()]
    return selected_results, total_json_files


def analyze_single_operator(data: Dict[str, Any]) -> Dict[str, Any]:
    """分析单个算子的测试结果。"""
    operator = data.get("operator", "unknown")
    iterations = data.get("results", [])

    analysis = {
        "torch_api": operator,
        "mindspore_api": "",
        "total_iterations": len(iterations),
        "consistent_count": 0,
        "inconsistent_count": 0,
        "torch_error_count": 0,
        "mindspore_error_count": 0,
        "both_error_count": 0,
        "comparison_error_count": 0,
        "final_status": "unknown",
        "errors": [],
    }

    for item in iterations:
        exec_result = item.get("execution_result", {})

        if not analysis["mindspore_api"]:
            analysis["mindspore_api"] = exec_result.get("mindspore_api", "")

        status = exec_result.get("status", "unknown")
        results_match = exec_result.get("results_match")

        if status == "compared":
            if results_match is True:
                analysis["consistent_count"] += 1
            elif results_match is False:
                # compared 但结果不一致，归入不一致
                analysis["inconsistent_count"] += 1
                comparison_error = exec_result.get("comparison_error")
                if comparison_error:
                    analysis["errors"].append(f"[CMP] {str(comparison_error)[:160]}")
            else:
                analysis["comparison_error_count"] += 1
                comparison_error = exec_result.get("comparison_error")
                if comparison_error:
                    analysis["errors"].append(f"[CMP] {str(comparison_error)[:160]}")
        elif status == "torch_failed":
            analysis["torch_error_count"] += 1
            torch_error = exec_result.get("torch_error")
            if torch_error:
                analysis["errors"].append(f"[PT] {str(torch_error)[:160]}")
        elif status == "mindspore_failed":
            analysis["mindspore_error_count"] += 1
            mindspore_error = exec_result.get("mindspore_error")
            if mindspore_error:
                analysis["errors"].append(f"[MS] {str(mindspore_error)[:160]}")
        elif status == "both_failed":
            analysis["both_error_count"] += 1
            torch_error = exec_result.get("torch_error")
            mindspore_error = exec_result.get("mindspore_error")
            if torch_error:
                analysis["errors"].append(f"[PT] {str(torch_error)[:160]}")
            if mindspore_error:
                analysis["errors"].append(f"[MS] {str(mindspore_error)[:160]}")
        else:
            # 兼容未知状态：根据成功标志与结果一致性补判
            torch_success = exec_result.get("torch_success")
            mindspore_success = exec_result.get("mindspore_success")
            if torch_success and mindspore_success:
                if results_match is True:
                    analysis["consistent_count"] += 1
                elif results_match is False:
                    analysis["inconsistent_count"] += 1
                else:
                    analysis["comparison_error_count"] += 1
            elif torch_success and not mindspore_success:
                analysis["mindspore_error_count"] += 1
            elif not torch_success and mindspore_success:
                analysis["torch_error_count"] += 1
            elif torch_success is False and mindspore_success is False:
                analysis["both_error_count"] += 1
            else:
                analysis["comparison_error_count"] += 1

    # 判定最终状态
    if analysis["consistent_count"] > 0 and analysis["inconsistent_count"] == 0:
        analysis["final_status"] = "consistent"
    elif analysis["inconsistent_count"] > 0:
        analysis["final_status"] = "inconsistent"
    elif (
        analysis["torch_error_count"]
        + analysis["mindspore_error_count"]
        + analysis["both_error_count"]
        + analysis["comparison_error_count"]
        > 0
    ):
        analysis["final_status"] = "error"
    else:
        analysis["final_status"] = "unknown"

    # 去重并截断错误信息
    analysis["errors"] = list(dict.fromkeys(analysis["errors"]))[:5]

    return analysis


def generate_reports(all_analyses: List[Dict[str, Any]], output_dir: str):
    """生成统计报告。"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    total_operators = len(all_analyses)
    consistent_ops = [a for a in all_analyses if a["final_status"] == "consistent"]
    inconsistent_ops = [a for a in all_analyses if a["final_status"] == "inconsistent"]
    error_ops = [a for a in all_analyses if a["final_status"] == "error"]
    unknown_ops = [a for a in all_analyses if a["final_status"] == "unknown"]

    total_iterations = sum(a["total_iterations"] for a in all_analyses)
    total_consistent = sum(a["consistent_count"] for a in all_analyses)
    total_inconsistent = sum(a["inconsistent_count"] for a in all_analyses)
    total_torch_errors = sum(a["torch_error_count"] for a in all_analyses)
    total_ms_errors = sum(a["mindspore_error_count"] for a in all_analyses)
    total_both_errors = sum(a["both_error_count"] for a in all_analyses)

    # 1) TXT 报告
    txt_file = os.path.join(output_dir, f"analysis_report_{timestamp}.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch ↔ MindSpore 差分测试结果分析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("=" * 50 + "\n")
        f.write("📊 总体统计\n")
        f.write("=" * 50 + "\n")
        f.write(f"测试算子总数: {total_operators}\n")
        f.write(
            f"  ✅ 一致 (consistent): {len(consistent_ops)} "
            f"({len(consistent_ops)/max(total_operators,1)*100:.1f}%)\n"
        )
        f.write(
            f"  ❌ 不一致 (inconsistent): {len(inconsistent_ops)} "
            f"({len(inconsistent_ops)/max(total_operators,1)*100:.1f}%)\n"
        )
        f.write(
            f"  ⚠️ 错误 (error): {len(error_ops)} "
            f"({len(error_ops)/max(total_operators,1)*100:.1f}%)\n"
        )
        f.write(f"  ❓ 未知 (unknown): {len(unknown_ops)}\n\n")

        f.write(f"总迭代次数: {total_iterations}\n")
        f.write(f"  一致次数: {total_consistent}\n")
        f.write(f"  不一致次数: {total_inconsistent}\n")
        f.write(f"  PT错误次数: {total_torch_errors}\n")
        f.write(f"  MS错误次数: {total_ms_errors}\n")
        f.write(f"  双边错误次数: {total_both_errors}\n\n")

        f.write("=" * 50 + "\n")
        f.write(f"✅ 一致算子 ({len(consistent_ops)} 个)\n")
        f.write("=" * 50 + "\n")
        for a in sorted(consistent_ops, key=lambda x: x["torch_api"]):
            f.write(
                f"  {a['torch_api']} → {a['mindspore_api']} "
                f"({a['consistent_count']}/{a['total_iterations']} 次一致)\n"
            )

        f.write("\n" + "=" * 50 + "\n")
        f.write(f"❌ 不一致算子 ({len(inconsistent_ops)} 个)\n")
        f.write("=" * 50 + "\n")
        for a in sorted(inconsistent_ops, key=lambda x: x["torch_api"]):
            f.write(f"  {a['torch_api']} → {a['mindspore_api']}\n")
            f.write(f"    一致: {a['consistent_count']}, 不一致: {a['inconsistent_count']}\n")
            for err in a["errors"][:3]:
                f.write(f"    ! {err}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write(f"⚠️ 错误算子 ({len(error_ops)} 个)\n")
        f.write("=" * 50 + "\n")
        for a in sorted(error_ops, key=lambda x: x["torch_api"]):
            f.write(f"  {a['torch_api']} → {a['mindspore_api']}\n")
            f.write(
                f"    PT错误: {a['torch_error_count']}, MS错误: {a['mindspore_error_count']}, "
                f"双错误: {a['both_error_count']}, 比较错误: {a['comparison_error_count']}\n"
            )
            for err in a["errors"][:3]:
                f.write(f"    ! {err}\n")

    print(f"📄 TXT报告已保存: {txt_file}")

    # 2) CSV 报告
    csv_file = os.path.join(output_dir, f"analysis_report_{timestamp}.csv")
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "torch_api",
            "mindspore_api",
            "final_status",
            "total_iterations",
            "consistent_count",
            "inconsistent_count",
            "torch_error_count",
            "mindspore_error_count",
            "both_error_count",
            "comparison_error_count",
            "error_summary",
        ])
        for a in sorted(all_analyses, key=lambda x: x["torch_api"]):
            writer.writerow([
                a["torch_api"],
                a["mindspore_api"],
                a["final_status"],
                a["total_iterations"],
                a["consistent_count"],
                a["inconsistent_count"],
                a["torch_error_count"],
                a["mindspore_error_count"],
                a["both_error_count"],
                a["comparison_error_count"],
                "; ".join(a["errors"][:3]) if a["errors"] else "",
            ])
    print(f"📄 CSV报告已保存: {csv_file}")

    # 3) JSON 报告
    json_file = os.path.join(output_dir, f"analysis_report_{timestamp}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            {
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
                    "total_torch_errors": total_torch_errors,
                    "total_mindspore_errors": total_ms_errors,
                    "total_both_errors": total_both_errors,
                },
                "operators": all_analyses,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"📄 JSON报告已保存: {json_file}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch ↔ MindSpore 差分测试结果分析")
    parser.add_argument(
        "--result-dir",
        "-r",
        default=os.path.join(ROOT_DIR, "pt_ms_test", "pt_ms_log_1"),
        help="测试结果目录路径",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=os.path.join(ROOT_DIR, "pt_ms_test", "analysis"),
        help="分析报告输出目录",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("PyTorch ↔ MindSpore 差分测试结果分析")
    print("=" * 80)
    print(f"📁 结果目录: {args.result_dir}")
    print(f"📁 输出目录: {args.output_dir}")

    all_results, total_json_files = load_latest_results_by_operator(args.result_dir)
    if not all_results:
        print("⚠️ 未找到任何可用测试结果文件")
        return

    print(f"\n📋 扫描到 JSON 文件数: {total_json_files}")
    print(f"📌 去重后算子数（每个算子仅最新时间戳）: {len(all_results)}")

    all_analyses = []
    for data in all_results:
        analysis = analyze_single_operator(data)
        all_analyses.append(analysis)

    generate_reports(all_analyses, args.output_dir)

    consistent = sum(1 for a in all_analyses if a["final_status"] == "consistent")
    inconsistent = sum(1 for a in all_analyses if a["final_status"] == "inconsistent")
    error = sum(1 for a in all_analyses if a["final_status"] == "error")
    unknown = sum(1 for a in all_analyses if a["final_status"] == "unknown")

    print("\n" + "=" * 50)
    print("📊 快速统计")
    print("=" * 50)
    print(f"✅ 一致: {consistent}/{len(all_analyses)}")
    print(f"❌ 不一致: {inconsistent}/{len(all_analyses)}")
    print(f"⚠️ 错误: {error}/{len(all_analyses)}")
    print(f"❓ 未知: {unknown}/{len(all_analyses)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
