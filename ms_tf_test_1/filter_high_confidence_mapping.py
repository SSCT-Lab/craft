#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3.5a: MS → TF 映射后筛选

功能：
- 读取 Step 3 输出的 ms_tf_mapping.csv
- 保留所有行（与 ms_pt_test 一致）
- 对 medium/low confidence 的映射，将 tensorflow-api 改为 "无对应实现"
- 输出筛选后的 CSV

用法：
    conda activate tf_env
    python ms_tf_test_1/filter_high_confidence_mapping.py [--input] [--output]

输出：ms_tf_test_1/data/ms_tf_mapping_high_confidence.csv
"""

import os
import sys
import csv
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def filter_high_confidence(input_csv: str, output_csv: str) -> dict:
    """
    筛选映射表：保留所有行，对 medium/low 置为 '无对应实现'。

    Returns:
        统计信息字典
    """
    if not os.path.exists(input_csv):
        print(f"❌ 输入文件不存在: {input_csv}")
        return {"error": "file_not_found"}

    rows = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    stats = {
        "total": len(rows),
        "high": 0,
        "medium_to_none": 0,
        "low_to_none": 0,
        "already_none": 0,
        "unknown_to_none": 0,
    }

    for row in rows:
        confidence = row.get("confidence", "").strip().lower()
        tf_api = row.get("tensorflow-api", "").strip()

        if tf_api == "无对应实现" or tf_api == "":
            stats["already_none"] += 1
        elif confidence == "high":
            stats["high"] += 1
        elif confidence == "medium":
            row["tensorflow-api"] = "无对应实现"
            row["reason"] = f"[原medium] {row.get('reason', '')}"
            stats["medium_to_none"] += 1
        else:
            # low / unknown / 其他
            if confidence == "low":
                stats["low_to_none"] += 1
            else:
                stats["unknown_to_none"] += 1
            row["tensorflow-api"] = "无对应实现"
            row["reason"] = f"[原{confidence}] {row.get('reason', '')}"

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=["mindspore-api", "tensorflow-api", "confidence", "reason"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Step 3.5a: 筛选 MS→TF 映射表中的 high confidence"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(ROOT_DIR, "ms_tf_test_1", "data", "ms_tf_mapping.csv"),
        help="输入映射 CSV 文件路径",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "ms_tf_test_1", "data", "ms_tf_mapping_high_confidence.csv"),
        help="输出筛选后 CSV 文件路径",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Step 3.5a: MS → TF 映射后筛选")
    print("=" * 80)

    stats = filter_high_confidence(args.input, args.output)

    if "error" in stats:
        sys.exit(1)

    print(f"\n📊 筛选结果:")
    print(f"  总行数:            {stats['total']}")
    print(f"  ▸ high（保留）:    {stats['high']}")
    print(f"  ▸ medium→无对应:   {stats['medium_to_none']}")
    print(f"  ▸ low→无对应:      {stats['low_to_none']}")
    print(f"  ▸ 原无对应实现:    {stats['already_none']}")
    print(f"  ▸ unknown→无对应:  {stats['unknown_to_none']}")
    print(f"\n💾 已保存到: {args.output}")


if __name__ == "__main__":
    main()
