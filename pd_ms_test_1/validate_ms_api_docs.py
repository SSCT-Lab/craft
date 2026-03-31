#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3.5b: validate whether MindSpore APIs really exist (based on official docs)

Function:
- Read PD->MS mapping CSV
- Fetch official documentation for each mindspore-api
- If doc is missing or too short, set mindspore-api to "No corresponding implementation"
- Write a new CSV

MindSpore doc notes:
- URL format: https://www.mindspore.cn/docs/en/r2.8.0/api_python/{sub_module}/{api_name}.html
- There is no genindex page, but the API search page can be used as a fallback
- MindSpore doc structure differs from PyTorch and needs adaptation

Usage:
    conda activate tf_env
    python pd_ms_test_1/validate_ms_api_docs.py \
        --input pd_ms_test_1/data/pd_ms_mapping_high.csv \
        --output pd_ms_test_1/data/pd_ms_mapping_validated.csv
"""

import argparse
import csv
import os
import sys
import time
import re
import requests
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_crawler_mindspore import MindSporeDocCrawler

DEFAULT_INPUT = str(ROOT / "pd_ms_test_1" / "data" / "pd_ms_mapping_high.csv")
DEFAULT_OUTPUT = str(ROOT / "pd_ms_test_1" / "data" / "pd_ms_mapping_validated.csv")
DEFAULT_DELAY = 0.5
DEFAULT_MIN_HTML_CHARS = 1500
DEFAULT_MIN_DESC_CHARS = 20
# MindSpore API search page (fallback validation)
MS_SEARCH_URL = "https://www.mindspore.cn/docs/en/r2.8.0/api_python/mindspore/mindspore.ops.html"
REQUEST_TIMEOUT = 15


def normalize_api_name(api_name: str) -> str:
    return (api_name or "").strip().lstrip(".")


@lru_cache(maxsize=8)
def _load_ms_module_page(module_name: str) -> str:
    """
    Load the API list page for a MindSpore module as a fallback validation.

    MindSpore has no global genindex, but each submodule has its own doc page listing APIs.
    Examples:
    - mindspore.ops: https://www.mindspore.cn/docs/en/r2.8.0/api_python/mindspore.ops.html
    - mindspore.nn:  https://www.mindspore.cn/docs/en/r2.8.0/api_python/mindspore.nn.html
    """
    base_url = "https://www.mindspore.cn/docs/en/r2.8.0/api_python"
    url = f"{base_url}/{module_name}.html"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
    except Exception:
        return ""
    if response.status_code != 200:
        return ""
    return response.text


def _get_module_name(api_name: str) -> str:
        """
        Extract module name from API name.
        Examples: mindspore.ops.abs -> mindspore.ops
              mindspore.nn.Conv2d -> mindspore.nn
              mindspore.Tensor -> mindspore
        """
    parts = api_name.split(".")
    if len(parts) <= 2:
        return "mindspore"
    # Use the first two parts as module name.
    return ".".join(parts[:2])


def has_module_page_match(api_name: str) -> bool:
    """
    Use MindSpore module doc pages to exactly match symbols as a fallback when doc fetch fails.
    """
    normalized = normalize_api_name(api_name)
    if not normalized:
        return False

    module_name = _get_module_name(normalized)
    html = _load_ms_module_page(module_name)
    if not html:
        return False

    # Exact match API name
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_\.]){re.escape(normalized)}(?![A-Za-z0-9_])",
        flags=re.IGNORECASE,
    )
    return bool(pattern.search(html))


def is_doc_valid(
    crawler: MindSporeDocCrawler,
    api_name: str,
    min_html_chars: int,
    min_desc_chars: int,
    delay: float,
) -> Tuple[bool, str]:
    """Check whether the MindSpore API doc is trustworthy."""
    normalized = normalize_api_name(api_name)
    if not normalized:
        return False, "empty_api"

    time.sleep(delay)
    doc = crawler.crawl(normalized)
    if not doc:
        # Fetch failed; try module page fallback.
        if has_module_page_match(normalized):
            return True, "ok_module_page_fallback"
        return False, "doc_not_found"

    raw_html = doc.get("raw_html", "") or ""
    description = doc.get("description", "") or ""
    title = doc.get("title", "") or ""

    # MindSpore pages can be shorter than PyTorch; relax thresholds.
    if len(raw_html) < min_html_chars and len(description) < min_desc_chars:
        if has_module_page_match(normalized):
            return True, "ok_module_page_fallback"
        return False, "doc_too_short"

    # Check whether the page contains relevant API content.
    last_part = normalized.split(".")[-1]
    api_match = (
        normalized in title
        or normalized in raw_html
        or last_part in title
        or last_part in raw_html
    )
    if not api_match and len(description) < min_desc_chars:
        if has_module_page_match(normalized):
            return True, "ok_module_page_fallback"
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
    parser = argparse.ArgumentParser(description="Validate MindSpore API docs and fix mappings")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input PD->MS mapping CSV path")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output fixed CSV path")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"Delay seconds per request (default {DEFAULT_DELAY})")
    parser.add_argument("--min-html-chars", type=int, default=DEFAULT_MIN_HTML_CHARS, help=f"Minimum doc HTML chars (default {DEFAULT_MIN_HTML_CHARS})")
    parser.add_argument("--min-desc-chars", type=int, default=DEFAULT_MIN_DESC_CHARS, help=f"Minimum description chars (default {DEFAULT_MIN_DESC_CHARS})")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return

    rows, fieldnames = load_csv_rows(args.input)
    if not fieldnames:
        print("❌ CSV parse failed: header is empty")
        return

    crawler = MindSporeDocCrawler()

    total = len(rows)
    checked = 0
    invalid = 0

    for idx, row in enumerate(rows, start=1):
        ms_api = normalize_api_name(row.get("mindspore-api", ""))
        if not ms_api or ms_api == "No corresponding implementation":
            continue

        ok, reason = is_doc_valid(
            crawler, ms_api,
            args.min_html_chars, args.min_desc_chars, args.delay,
        )
        checked += 1

        if ok:
            print(f"  ✅ [{idx}/{total}] {ms_api}")
            continue

        invalid += 1
        row["mindspore-api"] = "No corresponding implementation"
        row["reason"] = build_reason(row.get("reason", ""), f"mindspore_doc_invalid:{reason}")
        print(f"  ❌ [{idx}/{total}] {ms_api} ({reason})")

    save_csv_rows(args.output, rows, fieldnames)

    print("=" * 80)
    print("Validation completed (PD->MS)")
    print("=" * 80)
    print(f"Total rows: {total}")
    print(f"Checked entries: {checked}")
    print(f"Invalid entries: {invalid}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()

