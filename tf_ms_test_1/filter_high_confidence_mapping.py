#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3.5a: Filter mapping records with confidence == high.

Purpose:
- Read the TF->MS mapping CSV.
- Keep only records where confidence == "high".
- Output a new CSV.

Usage:
    conda activate tf_env
    python tf_ms_test_1/filter_high_confidence_mapping.py \
        --input tf_ms_test_1/data/tf_ms_mapping.csv \
        --output tf_ms_test_1/data/tf_ms_mapping_high.csv
"""

import argparse
import csv
import os
from typing import Dict, List, Tuple

DEFAULT_INPUT = os.path.join("tf_ms_test_1", "data", "tf_ms_mapping.csv")
DEFAULT_OUTPUT = os.path.join("tf_ms_test_1", "data", "tf_ms_mapping_high.csv")


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
    parser = argparse.ArgumentParser(description="Filter mapping records with confidence == high")
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT,
        help="Input mapping CSV path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help="Output filtered CSV path",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return

    rows, fieldnames = load_csv_rows(args.input)
    if not fieldnames:
        print("❌ CSV parse failed: header is empty")
        return

    filtered = [row for row in rows if (row.get("confidence", "").strip().lower() == "high")]

    save_csv_rows(args.output, filtered, fieldnames)

    print("=" * 80)
    print("Filtering complete")
    print("=" * 80)
    print(f"Original rows: {len(rows)}")
    print(f"Kept rows: {len(filtered)}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
