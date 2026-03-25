#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
筛选置信度为 high 的映射记录

功能：
- 读取 TF→PT 映射 CSV
- 仅保留 confidence == "high" 的记录
- 输出新的 CSV

用法：
    conda activate tf_env
    python tf_pt_test/filter_high_confidence_mapping.py `
        --input tf_pt_test/data/tf_pt_mapping.csv `
        --output tf_pt_test/data/tf_pt_mapping_high.csv
"""

import argparse
import csv
import os
from typing import Dict, List, Tuple

DEFAULT_INPUT = os.path.join("tf_pt_test", "data", "tf_pt_mapping.csv")
DEFAULT_OUTPUT = os.path.join("tf_pt_test", "data", "tf_pt_mapping_high.csv")


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
    parser = argparse.ArgumentParser(description="筛选置信度为 high 的映射记录")
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT,
        help="输入映射 CSV 路径",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help="输出筛选后 CSV 路径",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return

    rows, fieldnames = load_csv_rows(args.input)
    if not fieldnames:
        print("❌ CSV 解析失败：表头为空")
        return

    filtered = [row for row in rows if (row.get("confidence", "").strip().lower() == "high")]

    save_csv_rows(args.output, filtered, fieldnames)

    print("=" * 80)
    print("筛选完成")
    print("=" * 80)
    print(f"原始行数: {len(rows)}")
    print(f"保留行数: {len(filtered)}")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
