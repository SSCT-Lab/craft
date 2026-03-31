#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3.5b: Validate whether TensorFlow APIs exist (via official docs pages).

Functions:
- Read the PD→TF mapping CSV.
- Fetch official docs for each tensorflow-api.
- If docs are missing or too short, set tensorflow-api to "no_equivalent_impl".
- Write a new CSV.

TensorFlow doc characteristics:
- Doc URL format: https://www.tensorflow.org/api_docs/python/tf/xxx/yyy
- TF API paths use slashes (not dots), e.g., tf.keras.layers.Conv2D → tf/keras/layers/Conv2D
- TF has an official API search page for fallback validation

Usage:
    conda activate tf_env
    python pd_tf_test_1/validate_tf_api_docs.py \
        --input pd_tf_test_1/data/pd_tf_mapping_high.csv \
        --output pd_tf_test_1/data/pd_tf_mapping_validated.csv
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

from component.doc.doc_crawler_tensorflow import TensorFlowDocCrawler

DEFAULT_INPUT = str(ROOT / "pd_tf_test_1" / "data" / "pd_tf_mapping_high.csv")
DEFAULT_OUTPUT = str(ROOT / "pd_tf_test_1" / "data" / "pd_tf_mapping_validated.csv")
DEFAULT_DELAY = 0.5
DEFAULT_MIN_HTML_CHARS = 1500
DEFAULT_MIN_DESC_CHARS = 20
# Validate with direct TF doc URLs; _api/xxx endpoint can serve as a fallback.
TF_API_BASE = "https://www.tensorflow.org/api_docs/python/"
REQUEST_TIMEOUT = 15


def normalize_api_name(api_name: str) -> str:
    return (api_name or "").strip().lstrip(".")


def _build_tf_doc_url(api_name: str) -> str:
    """
    Build the official TF API doc URL.
    Example: tf.keras.layers.Conv2D → https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
    """
    normalized = normalize_api_name(api_name)
    # TF doc paths use slashes
    path = normalized.replace(".", "/")
    return f"{TF_API_BASE}{path}"


@lru_cache(maxsize=16)
def _check_tf_url_exists(url: str) -> Tuple[bool, str]:
    """
    Check whether a TF doc URL exists via HTTP GET.
    The TF docs server returns 404 or redirects to a search page for missing APIs.
    """
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if response.status_code == 200:
            # Check whether redirected to search or 404 page
            if "Page not found" in response.text or "404" in response.text[:500]:
                return False, "redirected_to_404"
            return True, "ok_direct"
        return False, f"http_{response.status_code}"
    except Exception as e:
        return False, f"request_error:{str(e)[:50]}"


def has_tf_direct_url_match(api_name: str) -> bool:
    """
    Validate API existence by requesting the TF doc URL.
    This is a TF-specific fallback (doc URL structure is clear and stable).
    """
    normalized = normalize_api_name(api_name)
    if not normalized:
        return False

    url = _build_tf_doc_url(normalized)
    ok, _ = _check_tf_url_exists(url)
    return ok


def is_doc_valid(
    crawler: TensorFlowDocCrawler,
    api_name: str,
    min_html_chars: int,
    min_desc_chars: int,
    delay: float,
) -> Tuple[bool, str]:
    """Check whether TensorFlow API docs are valid."""
    normalized = normalize_api_name(api_name)
    if not normalized:
        return False, "empty_api"

    time.sleep(delay)
    doc = crawler.crawl(normalized)
    if not doc:
        # Crawl failed; try direct URL as fallback
        if has_tf_direct_url_match(normalized):
            return True, "ok_direct_url_fallback"
        return False, "doc_not_found"

    raw_html = doc.get("raw_html", "") or ""
    description = doc.get("description", "") or ""
    title = doc.get("title", "") or ""

    if len(raw_html) < min_html_chars and len(description) < min_desc_chars:
        if has_tf_direct_url_match(normalized):
            return True, "ok_direct_url_fallback"
        return False, "doc_too_short"

    # Check whether the page truly contains the API content
    last_part = normalized.split(".")[-1]
    api_match = (
        normalized in title
        or normalized in raw_html
        or last_part in title
        or last_part in raw_html
    )
    if not api_match and len(description) < min_desc_chars:
        if has_tf_direct_url_match(normalized):
            return True, "ok_direct_url_fallback"
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
    parser = argparse.ArgumentParser(description="Validate TensorFlow API docs and fix mappings")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input PD→TF mapping CSV path")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output corrected CSV path")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"Delay seconds per request (default {DEFAULT_DELAY})")
    parser.add_argument("--min-html-chars", type=int, default=DEFAULT_MIN_HTML_CHARS, help=f"Minimum doc HTML chars (default {DEFAULT_MIN_HTML_CHARS})")
    parser.add_argument("--min-desc-chars", type=int, default=DEFAULT_MIN_DESC_CHARS, help=f"Minimum description chars (default {DEFAULT_MIN_DESC_CHARS})")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return

    rows, fieldnames = load_csv_rows(args.input)
    if not fieldnames:
        print("❌ CSV parse failed: empty header")
        return

    crawler = TensorFlowDocCrawler()

    total = len(rows)
    checked = 0
    invalid = 0

    for idx, row in enumerate(rows, start=1):
        tf_api = normalize_api_name(row.get("tensorflow-api", ""))
        if not tf_api or tf_api == "no_equivalent_impl":
            continue

        ok, reason = is_doc_valid(
            crawler, tf_api,
            args.min_html_chars, args.min_desc_chars, args.delay,
        )
        checked += 1

        if ok:
            print(f"  ✅ [{idx}/{total}] {tf_api}")
            continue

        invalid += 1
        row["tensorflow-api"] = "no_equivalent_impl"
        row["reason"] = build_reason(row.get("reason", ""), f"tensorflow_doc_invalid:{reason}")
        print(f"  ❌ [{idx}/{total}] {tf_api} ({reason})")

    save_csv_rows(args.output, rows, fieldnames)

    print("=" * 80)
    print("Validation complete (PD→TF)")
    print("=" * 80)
    print(f"Total rows: {total}")
    print(f"Checked items: {checked}")
    print(f"Invalid items: {invalid}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
