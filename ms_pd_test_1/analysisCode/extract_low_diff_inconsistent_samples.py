#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 inconsistent_success_samples JSON 中提取 comparison_error 为
"结果不一致，最大差异: xxx" 且 xxx < threshold 的完整样例。

默认输入:
    ms_pd_test_1/analysis/inconsistent_success_samples_*.json（自动取最新）
默认输出:
    ms_pd_test_1/analysis/inconsistent_success_samples_lt1_*.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


ERROR_PATTERN = re.compile(r"^结果不一致，最大差异:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")


def extract_max_diff(error_text: Any) -> Optional[float]:
    """从 comparison_error 中解析最大差异数值，解析失败返回 None。"""
    if not isinstance(error_text, str):
        return None

    match = ERROR_PATTERN.match(error_text.strip())
    if not match:
        return None

    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def build_default_output_path(input_path: Path) -> Path:
    """根据输入文件名自动构造输出文件名。"""
    stem = input_path.stem
    if stem.startswith("inconsistent_success_samples_"):
        suffix = stem.replace("inconsistent_success_samples_", "", 1)
        output_name = f"inconsistent_success_samples_lt1_{suffix}.json"
    else:
        output_name = f"{stem}_lt1.json"
    return input_path.parent / output_name


def find_latest_inconsistent_file(analysis_dir: Path) -> Path:
    """自动选择 analysis 目录下最新的 inconsistent_success_samples 文件。"""
    candidates = sorted(analysis_dir.glob("inconsistent_success_samples_*.json"))
    if not candidates:
        raise FileNotFoundError(f"未找到文件: {analysis_dir / 'inconsistent_success_samples_*.json'}")
    return candidates[-1]


def filter_samples(input_path: Path, output_path: Path, threshold: float = 1.0) -> Dict[str, Any]:
    """筛选满足 comparison_error 最大差异 < threshold 的完整样例并写入输出文件。"""
    with input_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    samples = raw_data.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("输入 JSON 的 samples 字段不是列表")

    selected_samples: List[Dict[str, Any]] = []

    for sample in samples:
        if not isinstance(sample, dict):
            continue

        execution_result = sample.get("execution_result", {})
        if not isinstance(execution_result, dict):
            continue

        comparison_error = execution_result.get("comparison_error")
        max_diff = extract_max_diff(comparison_error)
        if max_diff is None:
            continue

        if max_diff < threshold:
            selected_samples.append(sample)

    output_data = {
        "source_file": str(input_path),
        "filter_rule": "comparison_error 匹配 '结果不一致，最大差异: xxx' 且 xxx < threshold",
        "threshold": threshold,
        "total_samples": len(samples),
        "matched_samples": len(selected_samples),
        "samples": selected_samples,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, ensure_ascii=False, indent=2)

    return output_data


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[2]
    analysis_dir = root_dir / "ms_pd_test_1" / "analysis"
    default_input = find_latest_inconsistent_file(analysis_dir)
    default_output = build_default_output_path(default_input)

    parser = argparse.ArgumentParser(
        description="提取 comparison_error 最大差异 < threshold 的完整样例"
    )
    parser.add_argument("--input", type=Path, default=default_input, help=f"输入 JSON（默认: {default_input}）")
    parser.add_argument("--output", type=Path, default=default_output, help=f"输出 JSON（默认: {default_output}）")
    parser.add_argument("--threshold", type=float, default=1.0, help="最大差异阈值（默认: 1.0）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = filter_samples(input_path=input_path, output_path=output_path, threshold=args.threshold)

    print("=" * 80)
    print("筛选完成")
    print("=" * 80)
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"总样例数: {result['total_samples']}")
    print(f"命中样例数: {result['matched_samples']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
