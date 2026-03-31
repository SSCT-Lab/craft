#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1.5: Filter out non-existent MindSpore APIs

Purpose:
- Read the API list from Step 1 (ms_apis.json)
- Visit MindSpore official docs to validate API existence
- Output only APIs that pass validation

Usage:
    conda activate tf_env
    python ms_pt_test/filter_existing_ms_apis.py `
        --input ms_pt_test/data/ms_apis.json `
        --output ms_pt_test/data/ms_apis_existing.json
"""

import os
import sys
import io
import json
import time
import argparse
import requests
from typing import Dict, List, Any, Tuple
from bs4 import BeautifulSoup

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from component.doc.doc_crawler_mindspore import MindSporeDocCrawler

DEFAULT_INPUT = os.path.join(ROOT_DIR, "ms_pt_test", "data", "ms_apis.json")
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "ms_pt_test", "data", "ms_apis_existing.json")
DEFAULT_DELAY = 0.5
DEFAULT_MIN_PAGE_CHARS = 1200
REQUEST_TIMEOUT = 10

BAD_TITLE_PATTERNS = [
    "404", "not found", "page not found", "error",
]


def normalize_api_name(api_name: str) -> str:
    """Normalize API name."""
    return (api_name or "").strip().lstrip(".")


def _extract_main_html(soup: BeautifulSoup) -> str:
    main_content = (
        soup.find("main")
        or soup.find("div", class_="section")
        or soup.find("div", class_="document")
        or soup.find("body")
    )
    return str(main_content) if main_content else ""


def _title_invalid(title: str) -> bool:
    lowered = (title or "").lower()
    for pattern in BAD_TITLE_PATTERNS:
        if pattern in lowered:
            return True
    return False


def _check_page_valid(
    raw_html: str, title: str, api_name: str, min_page_chars: int
) -> Tuple[bool, str]:
    """
    Determine whether a MindSpore doc page is a real API page.

    Returns (valid, reason)
    """
    if not title or _title_invalid(title):
        return False, "invalid_title"

    if len(raw_html) < min_page_chars:
        return False, "doc_too_short"

    last_part = api_name.split(".")[-1]
    api_match = (
        api_name in title
        or api_name in raw_html
        or last_part in title
        or last_part in raw_html
    )
    if not api_match:
        return False, "no_api_match"

    return True, "ok"


def validate_api_doc(
    crawler: MindSporeDocCrawler,
    api_name: str,
    min_page_chars: int,
    delay: float,
) -> Tuple[bool, str, str]:
    """
    Check whether MindSpore API docs exist and are valid.

    Returns (ok, reason, url)
    """
    normalized = normalize_api_name(api_name)
    if not normalized:
        return False, "empty_api", ""

    url = crawler.build_doc_url(normalized)

    time.sleep(delay)

    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        return False, f"request_error: {str(e)[:60]}", url

    if resp.status_code != 200:
        return False, f"http_{resp.status_code}", url

    soup = BeautifulSoup(resp.content, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.text if title_tag else ""
    raw_html = _extract_main_html(soup)

    valid, reason = _check_page_valid(raw_html, title, normalized, min_page_chars)
    return valid, reason, url


def _apply_limit(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit is None or limit <= 0:
        return items
    return items[:limit]


def main():
    parser = argparse.ArgumentParser(
        description="Step 1.5: Filter out non-existent MindSpore APIs"
    )
    parser.add_argument(
        "--input", "-i", default=DEFAULT_INPUT,
        help="Step 1 output file path"
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help="Filtered output file path"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Delay per request in seconds (default {DEFAULT_DELAY})"
    )
    parser.add_argument(
        "--min-page-chars", type=int, default=DEFAULT_MIN_PAGE_CHARS,
        help=f"Minimum page character threshold (default {DEFAULT_MIN_PAGE_CHARS})"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Validate only the first N APIs (for quick testing)"
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Clear cached MindSpore docs (recommended after logic changes)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Step 1.5: Filter out non-existent MindSpore APIs")
    print("=" * 80)

    if not os.path.exists(args.input):
        print(f"Input file does not exist: {args.input}")
        sys.exit(1)

    # Clear old cache
    if args.clear_cache:
        import glob
        cache_pattern = os.path.join("data", "docs_cache", "mindspore_*.json")
        cached_files = glob.glob(cache_pattern)
        if cached_files:
            for cf in cached_files:
                os.remove(cf)
            print(f"Cleared {len(cached_files)} MindSpore doc cache files")
        else:
            print("No MindSpore doc cache files to clear")

    with open(args.input, 'r', encoding='utf-8') as f:
        api_data = json.load(f)

    all_apis = api_data.get("apis", [])
    all_apis = _apply_limit(all_apis, args.limit)

    print(f"Loaded {len(all_apis)} MindSpore APIs")

    crawler = MindSporeDocCrawler()
    valid_apis = []
    invalid_apis = []

    for idx, api_info in enumerate(all_apis, 1):
        api_name = api_info.get("ms_api", "")

        ok, reason, url = validate_api_doc(
            crawler, api_name,
            args.min_page_chars, args.delay,
        )

        if ok:
            valid_apis.append(api_info)
            print(f"  OK  [{idx}/{len(all_apis)}] {api_name}")
        else:
            invalid_apis.append({"api": api_name, "reason": reason, "url": url})
            print(f"  BAD [{idx}/{len(all_apis)}] {api_name} ({reason})")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    output_data = {
        "total_apis": len(valid_apis),
        "filtered_out": len(invalid_apis),
        "source_file": args.input,
        "apis": valid_apis,
        "invalid_apis": invalid_apis,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Filter results")
    print("=" * 80)
    print(f"  Original APIs: {len(all_apis)}")
    print(f"  Valid APIs: {len(valid_apis)}")
    print(f"  Filtered out: {len(invalid_apis)}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
