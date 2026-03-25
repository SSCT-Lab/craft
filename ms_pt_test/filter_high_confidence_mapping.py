#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
规范化映射记录：
- 保留全部记录
- 对置信度为 medium/low 的记录，将 pytorch-api 置为"无对应实现"
- 对置信度为 medium/low 的记录，将 reason 置为"原LLM置信度不高"

用法：
    conda activate tf_env
    python ms_pt_test/filter_high_confidence_mapping.py `
        --input ms_pt_test/data/ms_pt_mapping.csv `
        --output ms_pt_test/data/ms_pt_mapping_high.csv
"""

import argparse
import csv
import os
import sys
import io

# Windows 环境下强制使用 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from typing import Dict, List, Tuple

DEFAULT_INPUT = os.path.join("ms_pt_test", "data", "ms_pt_mapping.csv")
DEFAULT_OUTPUT = os.path.join("ms_pt_test", "data", "ms_pt_mapping_high.csv")


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
    parser = argparse.ArgumentParser(description="规范化映射记录（低置信度置为无对应实现）")
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
            new_row["pytorch-api"] = "无对应实现"
            new_row["reason"] = "原LLM置信度不高"
            converted_count += 1
        normalized_rows.append(new_row)

    save_csv_rows(args.output, normalized_rows, fieldnames)

    print("=" * 80)
    print("筛选完成")
    print("=" * 80)
    print(f"原始行数: {len(rows)}")
    print(f"保留行数: {len(normalized_rows)}")
    print(f"低置信度改写数: {converted_count}")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
