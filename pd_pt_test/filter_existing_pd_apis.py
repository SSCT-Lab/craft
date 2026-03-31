#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1.5: Filter non-existing Paddle APIs

Function:
- Read the API list output from Step 1
- Visit PaddlePaddle official doc pages to validate API existence
- Output only APIs that pass validation

Usage：
    conda activate tf_env
    python pd_pt_test/filter_existing_pd_apis.py `
        --input pd_pt_test/data/pd_apis_new.json `
        --output pd_pt_test/data/pd_apis_existing.json
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Any, Tuple
from bs4 import BeautifulSoup

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from component.doc.doc_crawler_paddle import PaddleDocCrawler

DEFAULT_INPUT = os.path.join(ROOT_DIR, "pd_pt_test", "data", "pd_apis_new.json")
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "pd_pt_test", "data", "pd_apis_existing.json")
DEFAULT_DELAY = 0.5
DEFAULT_MIN_PAGE_CHARS = 2000
DEFAULT_MIN_DESC_CHARS = 50


def normalize_api_name(api_name: str) -> str:
    """Normalize API name"""
    return (api_name or "").strip().lstrip(".")


# Paddle docs return HTTP 200 for non-existing URLs (soft 404);
# the page title is a generic intro page and needs feature detection.
PADDLE_GENERIC_PAGE_TITLES = [
    "Guides-Document-PaddlePaddle Deep Learning Platform",
    "User Guide - Docs - PaddlePaddle Deep Learning Platform",
]

# Suffix pattern for real API page titles (e.g., "add-API Document-PaddlePaddle ...")
PADDLE_API_TITLE_SUFFIX = "API Document-PaddlePaddle Deep Learning Platform"
PADDLE_API_TITLE_SUFFIX_CN = "API Document - PaddlePaddle Deep Learning Platform"

# Base URLs for English and Chinese docs (some APIs only have Chinese docs)
PADDLE_DOC_BASES = [
    ("en", "https://www.paddlepaddle.org.cn/documentation/docs/en/api/", "_en.html"),
    ("zh", "https://www.paddlepaddle.org.cn/documentation/docs/zh/api/", "_cn.html"),
]

REQUEST_TIMEOUT = 10


def _build_doc_url(api_name: str, base: str, suffix: str) -> str:
    """Build Paddle Document URL"""
    path = api_name.lstrip(".").replace(".", "/")
    return f"{base}{path}{suffix}"


def _check_page_valid(
    raw_html: str, title: str, api_name: str, min_page_chars: int,
) -> Tuple[bool, str]:
    """
    Determine whether a Paddle doc page is a real API page.

    Returns (valid, reason)
    """
    title_stripped = title.strip()

    # ===== Check 1: detectsoft 404 genericpage =====
    if title_stripped in PADDLE_GENERIC_PAGE_TITLES:
        return False, "soft_404_generic_page"

    # ===== Check 2: title contains "API Document" or "APIDocument" =====
    if (PADDLE_API_TITLE_SUFFIX not in title
            and PADDLE_API_TITLE_SUFFIX_CN not in title):
        return False, "not_api_page"

    # ===== Check 3: content length =====
    if len(raw_html) < min_page_chars:
        return False, "doc_too_short"

    # ===== Check 4: API namematch =====
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
    crawler,
    api_name: str,
    min_page_chars: int,
    min_desc_chars: int,
    delay: float,
) -> Tuple[bool, str]:
    """
    Check whether Paddle API docs exist and are valid.

    Validation strategy:
    1. Try English then Chinese docs (some APIs only have Chinese docs)
    2. For each URL, detect soft 404 by generic page title
    3. Validate content length and API name match
    4. Do not use crawler.crawl() to avoid cached generic error pages
    """
    normalized = normalize_api_name(api_name)
    if not normalized:
        return False, "empty_api"

    last_reason = "doc_not_found"

    for lang, base_url, suffix in PADDLE_DOC_BASES:
        url = _build_doc_url(normalized, base_url, suffix)

        time.sleep(delay)

        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        except Exception as e:
            last_reason = f"request_error_{lang}: {str(e)[:60]}"
            continue

        if resp.status_code != 200:
            last_reason = f"http_{resp.status_code}_{lang}"
            continue

        soup = BeautifulSoup(resp.content, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.text if title_tag else ""

        main_content = (
            soup.find("main")
            or soup.find("div", class_="section")
            or soup.find("body")
        )
        raw_html = str(main_content) if main_content else ""

        valid, reason = _check_page_valid(raw_html, title, normalized, min_page_chars)
        if valid:
            return True, f"ok_{lang}"
        last_reason = f"{reason}_{lang}"

    return False, last_reason


def main():
    parser = argparse.ArgumentParser(
        description="Step 1.5: Filter non-existing Paddle APIs"
    )
    parser.add_argument(
        "--input", "-i", default=DEFAULT_INPUT,
        help="Step 1 Output filepath"
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help="Filtered output filepath"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Delay per request in seconds (default {DEFAULT_DELAY})"
    )
    parser.add_argument(
        "--min-page-chars", type=int, default=DEFAULT_MIN_PAGE_CHARS,
        help=f"Minimum page char threshold (default {DEFAULT_MIN_PAGE_CHARS})"
    )
    parser.add_argument(
        "--min-desc-chars", type=int, default=DEFAULT_MIN_DESC_CHARS,
        help=f"Minimum description char threshold (default {DEFAULT_MIN_DESC_CHARS})"
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Clear previous Paddle doc cache (recommended after modifying validation logic)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Step 1.5: Filter non-existing Paddle APIs")
    print("=" * 80)

    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        sys.exit(1)

    # Clear old cache to avoid cached generic 404 pages
    if args.clear_cache:
        import glob
        cache_pattern = os.path.join("data", "docs_cache", "paddle_*.json")
        cached_files = glob.glob(cache_pattern)
        if cached_files:
            for cf in cached_files:
                os.remove(cf)
            print(f"🗑️ Cleared {len(cached_files)} Paddle doc cache files")
        else:
            print("ℹ️ No Paddle doc cache to clear")

    with open(args.input, 'r', encoding='utf-8') as f:
        api_data = json.load(f)

    all_apis = api_data.get("apis", [])
    print(f"📋 Loaded {len(all_apis)} Paddle APIs")

    crawler = PaddleDocCrawler()
    valid_apis = []
    invalid_apis = []

    for idx, api_info in enumerate(all_apis, 1):
        api_name = api_info["pd_api"]

        ok, reason = validate_api_doc(
            crawler, api_name,
            args.min_page_chars, args.min_desc_chars, args.delay,
        )

        if ok:
            valid_apis.append(api_info)
            print(f"  ✅ [{idx}/{len(all_apis)}] {api_name}")
        else:
            invalid_apis.append({"api": api_name, "reason": reason})
            print(f"  ❌ [{idx}/{len(all_apis)}] {api_name} ({reason})")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    output_data = {
        "total_apis": len(valid_apis),
        "filtered_out": len(invalid_apis),
        "apis": valid_apis,
        "invalid_apis": invalid_apis,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print(f"📊 Filtering results")
    print(f"{'=' * 80}")
    print(f"  Original API count: {len(all_apis)}")
    print(f"  Valid API count: {len(valid_apis)}")
    print(f"  Filtered out: {len(invalid_apis)}")
    print(f"  💾 Saved to: {args.output}")


if __name__ == "__main__":
    main()

