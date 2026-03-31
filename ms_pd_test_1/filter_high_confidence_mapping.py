#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3.5a: Normalize mapping records (MS->PD)

Functionality:
- Read MS->PD mapping CSV
- Keep all records
- For medium/low confidence records, set paddle-api to "no_matching_impl"
- Output a new CSV

Usage:
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
    parser = argparse.ArgumentParser(description="Normalize MS->PD mapping records (set low confidence to no matching impl)")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input mapping CSV path")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output filtered CSV path")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return

    rows, fieldnames = load_csv_rows(args.input)
    if not fieldnames:
        print("❌ CSV parse failed: empty header")
        return

    normalized_rows: List[Dict[str, str]] = []
    converted_count = 0
    for row in rows:
        confidence = row.get("confidence", "").strip().lower()
        new_row = dict(row)
        if confidence in {"medium", "low"}:
            new_row["paddle-api"] = "no_matching_impl"
            new_row["reason"] = "original_llm_confidence_too_low"
            converted_count += 1
        normalized_rows.append(new_row)

    save_csv_rows(args.output, normalized_rows, fieldnames)

    print("=" * 80)
    print("Filtering complete (MS->PD)")
    print("=" * 80)
    print(f"Original rows: {len(rows)}")
    print(f"Kept rows: {len(normalized_rows)}")
    print(f"Low-confidence rewritten: {converted_count}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
