#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3.5a: Filter MS -> TF mapping

Purpose:
- Read ms_tf_mapping.csv from Step 3
- Keep all rows (consistent with ms_pt_test)
- For medium/low confidence mappings, set tensorflow-api to "no_matching_impl"
- Output the filtered CSV

Usage:
    conda activate tf_env
    python ms_tf_test_1/filter_high_confidence_mapping.py [--input] [--output]

Output: ms_tf_test_1/data/ms_tf_mapping_high_confidence.csv
"""

import os
import sys
import csv
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def filter_high_confidence(input_csv: str, output_csv: str) -> dict:
    """
    Filter mappings: keep all rows, set medium/low to 'no_matching_impl'.

    Returns:
        Stats dict
    """
    if not os.path.exists(input_csv):
        print(f"❌ Input file does not exist: {input_csv}")
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

        if tf_api == "no_matching_impl" or tf_api == "":
            stats["already_none"] += 1
        elif confidence == "high":
            stats["high"] += 1
        elif confidence == "medium":
            row["tensorflow-api"] = "no_matching_impl"
            row["reason"] = f"[orig medium] {row.get('reason', '')}"
            stats["medium_to_none"] += 1
        else:
            # low / unknown / other
            if confidence == "low":
                stats["low_to_none"] += 1
            else:
                stats["unknown_to_none"] += 1
            row["tensorflow-api"] = "no_matching_impl"
            row["reason"] = f"[orig {confidence}] {row.get('reason', '')}"

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
        description="Step 3.5a: Filter high-confidence mappings in MS -> TF mapping table"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(ROOT_DIR, "ms_tf_test_1", "data", "ms_tf_mapping.csv"),
        help="Input mapping CSV path",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "ms_tf_test_1", "data", "ms_tf_mapping_high_confidence.csv"),
        help="Output filtered CSV path",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Step 3.5a: Filter MS -> TF mapping")
    print("=" * 80)

    stats = filter_high_confidence(args.input, args.output)

    if "error" in stats:
        sys.exit(1)

    print(f"\n📊 Filter summary:")
    print(f"  Total rows:            {stats['total']}")
    print(f"  ▸ high (kept):         {stats['high']}")
    print(f"  ▸ medium -> none:      {stats['medium_to_none']}")
    print(f"  ▸ low -> none:         {stats['low_to_none']}")
    print(f"  ▸ already none:        {stats['already_none']}")
    print(f"  ▸ unknown -> none:     {stats['unknown_to_none']}")
    print(f"\n💾 Saved to: {args.output}")


if __name__ == "__main__":
    main()
