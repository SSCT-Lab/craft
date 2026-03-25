#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 5+: TensorFlow ↔ MindSpore 差分测试结果分析 + 样例提取脚本

功能：
- 读取 tf_ms_log_1/ 目录中的 JSON 测试结果文件
- 同一算子存在多个 JSON 文件时，仅保留时间戳最新的一个
- 统计各算子的一致性/不一致性/错误分布
- 生成统计报告（TXT + CSV）
- 生成 5 类样例 JSON（完整保留 iteration 信息）：
  1) 执行成功且比较一致
  2) 执行成功但比较不一致
  3) 仅 tf_error
  4) 仅 ms_error
  5) both_error

说明：
- 本脚本不生成 analysis_report_*.json 汇总文件。

用法：
    conda activate tf_env
    python tf_ms_test_1/analyze_results_with_samples.py \
        [--result-dir tf_ms_test_1/tf_ms_log_1] \
        [--output-dir tf_ms_test_1/analysis]
"""

import argparse
import copy
import csv
import io
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

# Windows 环境下强制使用 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

RESULT_FILE_PATTERN = re.compile(r"^(llm_enhanced_.+?)_(\d{8}_\d{6})\.json$")


def _select_latest_result_files(result_dir: str) -> List[str]:
    """按算子前缀分组，仅保留时间戳最新的结果文件。"""
    latest_by_operator: Dict[str, Tuple[datetime, str]] = {}

    for filename in sorted(os.listdir(result_dir)):
        if not (filename.startswith("llm_enhanced_") and filename.endswith(".json")):
            continue

        match = RESULT_FILE_PATTERN.match(filename)
        if not match:
            # 不符合命名规范的文件，按完整文件名作为唯一 key 参与分析
            operator_key = os.path.splitext(filename)[0]
            timestamp = datetime.min
        else:
            operator_key = match.group(1)
            timestamp = datetime.strptime(match.group(2), "%Y%m%d_%H%M%S")

        previous = latest_by_operator.get(operator_key)
        if previous is None or timestamp > previous[0] or (timestamp == previous[0] and filename > previous[1]):
            latest_by_operator[operator_key] = (timestamp, filename)

    selected_files = sorted(item[1] for item in latest_by_operator.values())
    return selected_files


def load_all_results(result_dir: str) -> List[Dict[str, Any]]:
    """加载结果目录下按算子去重后的最新 JSON 结果文件。"""
    results: List[Dict[str, Any]] = []
    if not os.path.exists(result_dir):
        print(f"❌ 结果目录不存在: {result_dir}")
        return results

    selected_files = _select_latest_result_files(result_dir)
    all_candidate_count = sum(
        1
        for filename in os.listdir(result_dir)
        if filename.startswith("llm_enhanced_") and filename.endswith(".json")
    )
    skipped_count = max(all_candidate_count - len(selected_files), 0)
    if skipped_count > 0:
        print(f"ℹ️ 检测到同算子多版本结果，已按时间戳仅保留最新文件，跳过 {skipped_count} 个旧文件")

    for filename in selected_files:
        filepath = os.path.join(result_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
            if isinstance(data, dict):
                data["_source_file"] = filename
                results.append(data)
            else:
                print(f"⚠️ 跳过非对象JSON: {filename}")
        except Exception as error:  # pylint: disable=broad-except
            print(f"⚠️ 加载 {filename} 失败: {error}")

    return results


def _normalize_status(execution_result: Dict[str, Any]) -> str:
    """将状态规范为脚本内部使用的统一标识。"""
    status = str(execution_result.get("status", ""))
    if status == "mindspore_error":
        return "ms_error"
    return status


def analyze_single_operator(data: Dict[str, Any]) -> Dict[str, Any]:
    """分析单个算子的测试结果。"""
    tf_api = str(data.get("tf_api", "unknown"))
    ms_api = str(data.get("mindspore_api", data.get("ms_api", "")))
    iterations = data.get("results", [])
    if not isinstance(iterations, list):
        iterations = []

    analysis: Dict[str, Any] = {
        "tf_api": tf_api,
        "ms_api": ms_api,
        "total_iterations": len(iterations),
        "consistent_count": 0,
        "inconsistent_count": 0,
        "tf_error_count": 0,
        "ms_error_count": 0,
        "both_error_count": 0,
        "comparison_error_count": 0,
        "final_status": "unknown",
        "errors": [],
    }

    for item in iterations:
        if not isinstance(item, dict):
            continue

        exec_result = item.get("execution_result", {})
        if not isinstance(exec_result, dict):
            continue

        status = _normalize_status(exec_result)

        if status == "consistent":
            analysis["consistent_count"] += 1
        elif status == "inconsistent":
            analysis["inconsistent_count"] += 1
        elif status == "tf_error":
            analysis["tf_error_count"] += 1
            tf_error_message = exec_result.get("tf_error")
            if isinstance(tf_error_message, str) and tf_error_message:
                analysis["errors"].append(f"[TF] {tf_error_message[:100]}")
        elif status == "ms_error":
            analysis["ms_error_count"] += 1
            ms_error_message = exec_result.get("ms_error")
            if isinstance(ms_error_message, str) and ms_error_message:
                analysis["errors"].append(f"[MS] {ms_error_message[:100]}")
        elif status == "both_error":
            analysis["both_error_count"] += 1
            tf_error_message = exec_result.get("tf_error")
            ms_error_message = exec_result.get("ms_error")
            if isinstance(tf_error_message, str) and tf_error_message:
                analysis["errors"].append(f"[TF] {tf_error_message[:100]}")
            if isinstance(ms_error_message, str) and ms_error_message:
                analysis["errors"].append(f"[MS] {ms_error_message[:100]}")
        elif status == "comparison_error":
            analysis["comparison_error_count"] += 1

    if analysis["consistent_count"] > 0 and analysis["inconsistent_count"] == 0:
        analysis["final_status"] = "consistent"
    elif analysis["inconsistent_count"] > 0:
        analysis["final_status"] = "inconsistent"
    elif analysis["tf_error_count"] + analysis["ms_error_count"] + analysis["both_error_count"] > 0:
        analysis["final_status"] = "error"
    else:
        analysis["final_status"] = "unknown"

    analysis["errors"] = list(dict.fromkeys(analysis["errors"]))[:5]
    return analysis


def _classify_iteration(execution_result: Dict[str, Any]) -> str:
    """将 iteration 分类为五类之一，或返回空字符串表示不纳入样例。"""
    status = _normalize_status(execution_result)
    tf_success = execution_result.get("tf_success")
    ms_success = execution_result.get("mindspore_success", execution_result.get("ms_success"))
    results_match = execution_result.get("results_match")

    if status == "consistent" and tf_success is True and ms_success is True and results_match is True:
        return "consistent_success"
    if status == "inconsistent" and tf_success is True and ms_success is True and results_match is False:
        return "inconsistent_success"
    if status == "tf_error":
        return "tf_error_only"
    if status == "ms_error":
        return "ms_error_only"
    if status == "both_error":
        return "both_error"

    # 兼容 status 不稳定或缺失的历史数据
    if tf_success is True and ms_success is True and results_match is False:
        return "inconsistent_success"
    if tf_success is False and ms_success is True:
        return "tf_error_only"
    if tf_success is True and ms_success is False:
        return "ms_error_only"
    if tf_success is False and ms_success is False:
        return "both_error"

    return ""


def extract_samples(all_results: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Set[str]]]:
    """提取五类样例，完整保留每条 iteration 信息。"""
    categorized_samples: Dict[str, List[Dict[str, Any]]] = {
        "consistent_success": [],
        "inconsistent_success": [],
        "tf_error_only": [],
        "ms_error_only": [],
        "both_error": [],
    }
    categorized_apis: Dict[str, Set[str]] = {
        "consistent_success": set(),
        "inconsistent_success": set(),
        "tf_error_only": set(),
        "ms_error_only": set(),
        "both_error": set(),
    }

    for data in all_results:
        tf_api = str(data.get("tf_api", "unknown"))
        iterations = data.get("results", [])
        if not isinstance(iterations, list):
            continue

        for item in iterations:
            if not isinstance(item, dict):
                continue
            execution_result = item.get("execution_result", {})
            if not isinstance(execution_result, dict):
                continue

            category = _classify_iteration(execution_result)
            if not category:
                continue

            categorized_samples[category].append(copy.deepcopy(item))
            categorized_apis[category].add(tf_api)

    return categorized_samples, categorized_apis


def generate_reports(all_analyses: List[Dict[str, Any]], output_dir: str, timestamp: str) -> Tuple[str, str]:
    """生成 TXT / CSV 统计报告。"""
    os.makedirs(output_dir, exist_ok=True)

    total_operators = len(all_analyses)
    consistent_ops = [item for item in all_analyses if item["final_status"] == "consistent"]
    inconsistent_ops = [item for item in all_analyses if item["final_status"] == "inconsistent"]
    error_ops = [item for item in all_analyses if item["final_status"] == "error"]
    unknown_ops = [item for item in all_analyses if item["final_status"] == "unknown"]

    total_iterations = sum(item["total_iterations"] for item in all_analyses)
    total_consistent = sum(item["consistent_count"] for item in all_analyses)
    total_inconsistent = sum(item["inconsistent_count"] for item in all_analyses)
    total_tf_errors = sum(item["tf_error_count"] for item in all_analyses)
    total_ms_errors = sum(item["ms_error_count"] for item in all_analyses)
    total_both_errors = sum(item["both_error_count"] for item in all_analyses)

    txt_file = os.path.join(output_dir, f"analysis_report_{timestamp}.txt")
    with open(txt_file, "w", encoding="utf-8") as file_handle:
        file_handle.write("=" * 80 + "\n")
        file_handle.write("TensorFlow ↔ MindSpore 差分测试结果分析报告\n")
        file_handle.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_handle.write("=" * 80 + "\n\n")

        file_handle.write("=" * 50 + "\n")
        file_handle.write("📊 总体统计\n")
        file_handle.write("=" * 50 + "\n")
        file_handle.write(f"测试算子总数: {total_operators}\n")
        file_handle.write(
            f"  ✅ 一致 (consistent): {len(consistent_ops)} "
            f"({len(consistent_ops) / max(total_operators, 1) * 100:.1f}%)\n"
        )
        file_handle.write(
            f"  ❌ 不一致 (inconsistent): {len(inconsistent_ops)} "
            f"({len(inconsistent_ops) / max(total_operators, 1) * 100:.1f}%)\n"
        )
        file_handle.write(
            f"  ⚠️ 错误 (error): {len(error_ops)} "
            f"({len(error_ops) / max(total_operators, 1) * 100:.1f}%)\n"
        )
        file_handle.write(f"  ❓ 未知 (unknown): {len(unknown_ops)}\n\n")

        file_handle.write("=" * 50 + "\n")
        file_handle.write("📦 五类样例计数（按迭代条目）\n")
        file_handle.write("=" * 50 + "\n")
        file_handle.write(f"总迭代次数: {total_iterations}\n")
        file_handle.write(f"  一致次数: {total_consistent}\n")
        file_handle.write(f"  不一致次数: {total_inconsistent}\n")
        file_handle.write(f"  TF错误次数: {total_tf_errors}\n")
        file_handle.write(f"  MS错误次数: {total_ms_errors}\n")
        file_handle.write(f"  双错误次数: {total_both_errors}\n\n")

        file_handle.write("=" * 50 + "\n")
        file_handle.write(f"✅ 一致算子 ({len(consistent_ops)} 个)\n")
        file_handle.write("=" * 50 + "\n")
        for item in sorted(consistent_ops, key=lambda element: element["tf_api"]):
            file_handle.write(
                f"  {item['tf_api']} → {item['ms_api']} "
                f"({item['consistent_count']}/{item['total_iterations']} 次一致)\n"
            )

        file_handle.write("\n" + "=" * 50 + "\n")
        file_handle.write(f"❌ 不一致算子 ({len(inconsistent_ops)} 个)\n")
        file_handle.write("=" * 50 + "\n")
        for item in sorted(inconsistent_ops, key=lambda element: element["tf_api"]):
            file_handle.write(f"  {item['tf_api']} → {item['ms_api']}\n")
            file_handle.write(
                f"    一致: {item['consistent_count']}, 不一致: {item['inconsistent_count']}\n"
            )
            for error_text in item["errors"][:3]:
                file_handle.write(f"    ! {error_text}\n")

        file_handle.write("\n" + "=" * 50 + "\n")
        file_handle.write(f"⚠️ 错误算子 ({len(error_ops)} 个)\n")
        file_handle.write("=" * 50 + "\n")
        for item in sorted(error_ops, key=lambda element: element["tf_api"]):
            file_handle.write(f"  {item['tf_api']} → {item['ms_api']}\n")
            file_handle.write(
                f"    TF错误: {item['tf_error_count']}, MS错误: {item['ms_error_count']}, "
                f"双错误: {item['both_error_count']}\n"
            )
            for error_text in item["errors"][:3]:
                file_handle.write(f"    ! {error_text}\n")

    print(f"📄 TXT报告已保存: {txt_file}")

    csv_file = os.path.join(output_dir, f"analysis_report_{timestamp}.csv")
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow([
            "tf_api",
            "ms_api",
            "final_status",
            "total_iterations",
            "consistent_count",
            "inconsistent_count",
            "tf_error_count",
            "ms_error_count",
            "both_error_count",
            "error_summary",
        ])
        for item in sorted(all_analyses, key=lambda element: element["tf_api"]):
            writer.writerow([
                item["tf_api"],
                item["ms_api"],
                item["final_status"],
                item["total_iterations"],
                item["consistent_count"],
                item["inconsistent_count"],
                item["tf_error_count"],
                item["ms_error_count"],
                item["both_error_count"],
                "; ".join(item["errors"][:3]) if item["errors"] else "",
            ])

    print(f"📄 CSV报告已保存: {csv_file}")
    return txt_file, csv_file


def generate_sample_files(
    categorized_samples: Dict[str, List[Dict[str, Any]]],
    categorized_apis: Dict[str, Set[str]],
    sample_dir: str,
    timestamp: str,
) -> List[str]:
    """输出五类样例文件，每类一个 JSON。"""
    os.makedirs(sample_dir, exist_ok=True)

    category_meta = {
        "consistent_success": "consistent_success_samples",
        "inconsistent_success": "inconsistent_success_samples",
        "tf_error_only": "tf_error_only_samples",
        "ms_error_only": "ms_error_only_samples",
        "both_error": "both_error_samples",
    }

    output_files: List[str] = []
    generated_at = datetime.now().isoformat()

    for category, base_name in category_meta.items():
        file_path = os.path.join(sample_dir, f"{base_name}_{timestamp}.json")
        payload = {
            "generated_at": generated_at,
            "category": category,
            "api_count": len(categorized_apis.get(category, set())),
            "sample_count": len(categorized_samples.get(category, [])),
            "samples": categorized_samples.get(category, []),
        }
        with open(file_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2)

        output_files.append(file_path)
        print(f"📦 样例文件已保存: {file_path}")

    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorFlow ↔ MindSpore 差分测试结果分析 + 样例提取")
    parser.add_argument(
        "--result-dir",
        "-r",
        default=os.path.join(ROOT_DIR, "tf_ms_test_1", "tf_ms_log_1"),
        help="测试结果目录路径",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=os.path.join(ROOT_DIR, "tf_ms_test_1", "analysis"),
        help="统计报告输出目录（TXT/CSV）",
    )
    parser.add_argument(
        "--sample-dir",
        "-s",
        default=os.path.join(ROOT_DIR, "tf_ms_test_1", "analysis"),
        help="样例JSON输出目录",
    )

    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("TensorFlow ↔ MindSpore 差分测试结果分析 + 样例提取")
    print("=" * 80)
    print(f"📁 结果目录: {args.result_dir}")
    print(f"📁 报告目录: {args.output_dir}")
    print(f"📁 样例目录: {args.sample_dir}")

    all_results = load_all_results(args.result_dir)
    if not all_results:
        print("⚠️ 未找到任何测试结果文件")
        return

    print(f"\n📋 加载了 {len(all_results)} 个算子的测试结果")

    all_analyses: List[Dict[str, Any]] = []
    for data in all_results:
        all_analyses.append(analyze_single_operator(data))

    generate_reports(all_analyses, args.output_dir, timestamp)

    categorized_samples, categorized_apis = extract_samples(all_results)
    generate_sample_files(categorized_samples, categorized_apis, args.sample_dir, timestamp)

    consistent = sum(1 for item in all_analyses if item["final_status"] == "consistent")
    inconsistent = sum(1 for item in all_analyses if item["final_status"] == "inconsistent")
    error = sum(1 for item in all_analyses if item["final_status"] == "error")

    print("\n" + "=" * 50)
    print("📊 快速统计")
    print("=" * 50)
    print(f"✅ 一致: {consistent}/{len(all_analyses)}")
    print(f"❌ 不一致: {inconsistent}/{len(all_analyses)}")
    print(f"⚠️ 错误: {error}/{len(all_analyses)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
