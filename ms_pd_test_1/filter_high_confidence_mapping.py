#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3.5a: 规范化映射记录（MS→PD）

功能：
- 读取 MS→PD 映射 CSV
- 保留全部记录
- 对置信度为 medium/low 的记录，将 paddle-api 置为"无对应实现"
- 输出新的 CSV

用法：
    conda activate tf_env
    python ms_pd_test_1/filter_high_confidence_mapping.py \
        --input ms_pd_test_1/data/ms_pd_mapping.csv \
        --output ms_pd_test_1/data/ms_pd_mapping_high.csv
"""

import argparse
import csv
import os
import sys

from typing import Dict, List, Tuple

DEFAULT_INPUT = os.path.join("ms_pd_test_1", "data", "ms_pd_mapping.csv")
DEFAULT_OUTPUT = os.path.join("ms_pd_test_1", "data", "ms_pd_mapping_high.csv")


def load_csv_rows(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def save_csv_rows(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="规范化 MS→PD 映射记录（低置信度置为无对应实现）")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="输入映射 CSV 路径")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="输出筛选后 CSV 路径")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return

    rows, fieldnames = load_csv_rows(args.input)
    if not fieldnames:
        print("❌ CSV 解析失败：表头为空")
        return

    normalized_rows: List[Dict[str, str]] = []
    converted_count = 0
    for row in rows:
        confidence = row.get("confidence", "").strip().lower()
        new_row = dict(row)
        if confidence in {"medium", "low"}:
            new_row["paddle-api"] = "无对应实现"
            new_row["reason"] = "原LLM置信度不高"
            converted_count += 1
        normalized_rows.append(new_row)

    save_csv_rows(args.output, normalized_rows, fieldnames)

    print("=" * 80)
    print("筛选完成（MS→PD）")
    print("=" * 80)
    print(f"原始行数: {len(rows)}")
    print(f"保留行数: {len(normalized_rows)}")
    print(f"低置信度改写数: {converted_count}")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
