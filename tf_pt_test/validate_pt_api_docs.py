#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate whether PyTorch APIs exist (based on official docs)

Features:
- Read TF→PT mapping CSV
- Fetch official docs for each pytorch-api
- If the doc is missing or too short, set pytorch-api to "无对应实现"
- Output a new CSV

Usage:
    conda activate tf_env
    python tf_pt_test/validate_pt_api_docs.py \
        --input tf_pt_test/data/tf_pt_mapping_high.csv \
        --output tf_pt_test/data/tf_pt_mapping_validated.csv
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_crawler_pytorch import PyTorchDocCrawler

DEFAULT_INPUT = str(ROOT / "tf_pt_test" / "data" / "tf_pt_mapping_high.csv")
DEFAULT_OUTPUT = str(ROOT / "tf_pt_test" / "data" / "tf_pt_mapping_validated.csv")
DEFAULT_DELAY = 0.5
DEFAULT_MIN_HTML_CHARS = 2000
DEFAULT_MIN_DESC_CHARS = 30


def normalize_api_name(api_name: str) -> str:
    return (api_name or "").strip().lstrip(".")


def is_doc_valid(
    crawler: PyTorchDocCrawler,
    api_name: str,
    min_html_chars: int,
    min_desc_chars: int,
    delay: float,
) -> Tuple[bool, str]:
    """Check whether the PyTorch API doc looks valid."""
    normalized = normalize_api_name(api_name)
    if not normalized:
        return False, "empty_api"

    time.sleep(delay)
    doc = crawler.crawl(normalized)
    if not doc:
        return False, "doc_not_found"

    raw_html = doc.get("raw_html", "") or ""
    description = doc.get("description", "") or ""
    title = doc.get("title", "") or ""

    if len(raw_html) < min_html_chars and len(description) < min_desc_chars:
        return False, "doc_too_short"

    last_part = normalized.split(".")[-1]
    api_match = (
        normalized in title
        or normalized in raw_html
        or last_part in title
        or last_part in raw_html
    )
    if not api_match and len(description) < min_desc_chars:
        return False, "no_api_match"

    return True, "ok"


def load_csv_rows(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def save_csv_rows(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_reason(original_reason: str, new_reason: str) -> str:
    original_reason = (original_reason or "").strip()
    if not original_reason:
        return new_reason
    if new_reason in original_reason:
        return original_reason
    return f"{original_reason}; {new_reason}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PyTorch API docs and fix mappings")
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT,
        help="Input TF→PT mapping CSV path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help="Output corrected CSV path",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay seconds per request (default {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--min-html-chars",
        type=int,
        default=DEFAULT_MIN_HTML_CHARS,
        help=f"Min HTML length threshold (default {DEFAULT_MIN_HTML_CHARS})",
    )
    parser.add_argument(
        "--min-desc-chars",
        type=int,
        default=DEFAULT_MIN_DESC_CHARS,
        help=f"Min description length threshold (default {DEFAULT_MIN_DESC_CHARS})",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return

    rows, fieldnames = load_csv_rows(args.input)
    if not fieldnames:
        print("❌ CSV parse failed: header is empty")
        return

    crawler = PyTorchDocCrawler()

    total = len(rows)
    checked = 0
    invalid = 0

    for idx, row in enumerate(rows, start=1):
        pt_api = normalize_api_name(row.get("pytorch-api", ""))
        if not pt_api or pt_api == "无对应实现":
            continue

        ok, reason = is_doc_valid(
            crawler,
            pt_api,
            args.min_html_chars,
            args.min_desc_chars,
            args.delay,
        )
        checked += 1

        if ok:
            print(f"  ✅ [{idx}/{total}] {pt_api}")
            continue

        invalid += 1
        row["pytorch-api"] = "无对应实现"
        row["reason"] = build_reason(row.get("reason", ""), f"pytorch_doc_invalid:{reason}")
        print(f"  ❌ [{idx}/{total}] {pt_api} ({reason})")

    save_csv_rows(args.output, rows, fieldnames)

    print("=" * 80)
    print("Validation complete")
    print("=" * 80)
    print(f"Total rows: {total}")
    print(f"Checked rows: {checked}")
    print(f"Invalid rows: {invalid}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
