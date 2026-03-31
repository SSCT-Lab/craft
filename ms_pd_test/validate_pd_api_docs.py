#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3.5b: Verify whether PaddlePaddle APIs actually exist (based on official doc pages)

Functionality:
- Read MS->PD mapping CSV
- Fetch official docs for each paddle-api
- If the doc is missing or too short, set paddle-api to "no_matching_impl"
- Output a new CSV

PaddlePaddle doc notes:
- Doc URL format: https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/xxx.html
- Submodule doc: paddle.nn.Conv2D -> api/paddle/nn/Conv2D_cn.html
- Validate by crawling official docs with PaddleDocCrawler

Usage:
    conda activate tf_env
    python ms_pd_test_1/validate_pd_api_docs.py \
        --input ms_pd_test_1/data/ms_pd_mapping_high.csv \
        --output ms_pd_test_1/data/ms_pd_mapping_validated.csv
"""

import argparse
import csv
import os
import sys

import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_crawler_paddle import PaddleDocCrawler

DEFAULT_INPUT = str(ROOT / "ms_pd_test_1" / "data" / "ms_pd_mapping_high.csv")
DEFAULT_OUTPUT = str(ROOT / "ms_pd_test_1" / "data" / "ms_pd_mapping_validated.csv")
DEFAULT_DELAY = 0.5
DEFAULT_MIN_PAGE_CHARS = 2000
DEFAULT_MIN_DESC_CHARS = 50
REQUEST_TIMEOUT = 10

# Paddle docs often return soft-404 pages (HTTP 200) when missing; check titles too.
PADDLE_GENERIC_PAGE_TITLES = [
    "Guides-Document-PaddlePaddle Deep Learning Platform",
    "User-Guide-Docs-PaddlePaddle Deep Learning Platform",
]

PADDLE_API_TITLE_SUFFIX = "API Document-PaddlePaddle Deep Learning Platform"
PADDLE_API_TITLE_SUFFIX_CN = "API-Docs-PaddlePaddle Deep Learning Platform"

PADDLE_DOC_BASES = [
    ("en", "https://www.paddlepaddle.org.cn/documentation/docs/en/api/", "_en.html"),
    ("zh", "https://www.paddlepaddle.org.cn/documentation/docs/zh/api/", "_cn.html"),
]


def normalize_api_name(api_name: str) -> str:
    return (api_name or "").strip().lstrip(".")


def _build_doc_url(api_name: str, base: str, suffix: str) -> str:
    path = api_name.lstrip(".").replace(".", "/")
    return f"{base}{path}{suffix}"


def _check_page_valid(
    raw_html: str,
    title: str,
    description: str,
    api_name: str,
    min_page_chars: int,
    min_desc_chars: int,
) -> Tuple[bool, str]:
    title_stripped = title.strip()

    if title_stripped in PADDLE_GENERIC_PAGE_TITLES:
        return False, "soft_404_generic_page"

    if PADDLE_API_TITLE_SUFFIX not in title and PADDLE_API_TITLE_SUFFIX_CN not in title:
        return False, "not_api_page"

    if len(raw_html) < min_page_chars:
        return False, "doc_too_short"

    last_part = api_name.split(".")[-1]
    api_match = (
        api_name in title
        or api_name in raw_html
        or last_part in title
        or last_part in raw_html
    )
    if not api_match and len(description) < min_desc_chars:
        return False, "no_api_match"

    return True, "ok"


def is_doc_valid(
    crawler: PaddleDocCrawler,
    api_name: str,
    min_page_chars: int,
    min_desc_chars: int,
    delay: float,
) -> Tuple[bool, str]:
    """Check whether a PaddlePaddle API doc page looks reliable."""
    normalized = normalize_api_name(api_name)
    if not normalized:
        return False, "empty_api"

    # First, use the unified Paddle crawler to fetch the main doc content.
    # Note: the Paddle site has soft-404 pages, so keep the URL-level checks.
    time.sleep(delay)
    crawler_doc = crawler.crawl(normalized)
    if crawler_doc:
        crawler_raw_html = crawler_doc.get("raw_html", "") or ""
        crawler_description = crawler_doc.get("description", "") or ""
        crawler_title = crawler_doc.get("title", "") or ""
        crawler_ok, _ = _check_page_valid(
            crawler_raw_html,
            crawler_title,
            crawler_description,
            normalized,
            min_page_chars,
            min_desc_chars,
        )
        if crawler_ok:
            return True, "ok_crawler"

    last_reason = "doc_not_found"
    for lang, base_url, suffix in PADDLE_DOC_BASES:
        url = _build_doc_url(normalized, base_url, suffix)
        time.sleep(delay)

        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        except Exception as error:
            last_reason = f"request_error_{lang}:{str(error)[:60]}"
            continue

        if resp.status_code != 200:
            last_reason = f"http_{resp.status_code}_{lang}"

        soup = BeautifulSoup(resp.content, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.text if title_tag else ""

        first_p = soup.find("main").find("p") if soup.find("main") else None
        if first_p is None:
            section = soup.find("div", class_="section")
            first_p = section.find("p") if section else None
        if first_p is None:
            body = soup.find("body")
            first_p = body.find("p") if body else None
        description = first_p.get_text(strip=True) if first_p else ""

        main_content = (
            soup.find("main")
            or soup.find("div", class_="section")
            or soup.find("body")
        )
        raw_html = str(main_content) if main_content else ""

        ok, reason = _check_page_valid(
            raw_html,
            title,
            description,
            normalized,
            min_page_chars,
            min_desc_chars,
        )
        if ok:
            return True, f"ok_{lang}"

        last_reason = f"{reason}_{lang}"

    return False, last_reason


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
    parser = argparse.ArgumentParser(description="Validate PaddlePaddle API docs and fix mapping")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input MS->PD mapping CSV path")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output fixed CSV path")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"Delay seconds per request (default {DEFAULT_DELAY})")
    parser.add_argument(
        "--min-page-chars",
        "--min-html-chars",
        dest="min_page_chars",
        type=int,
        default=DEFAULT_MIN_PAGE_CHARS,
        help=f"Minimum page character threshold (default {DEFAULT_MIN_PAGE_CHARS})",
    )
    parser.add_argument("--min-desc-chars", type=int, default=DEFAULT_MIN_DESC_CHARS, help=f"Minimum description character count (default {DEFAULT_MIN_DESC_CHARS})")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return

    rows, fieldnames = load_csv_rows(args.input)
    if not fieldnames:
        print("❌ CSV parse failed: empty header")
        return

    crawler = PaddleDocCrawler()

    total = len(rows)
    checked = 0
    invalid = 0

    for idx, row in enumerate(rows, start=1):
        pd_api = normalize_api_name(row.get("paddle-api", ""))
        if not pd_api or pd_api == "no_matching_impl":
            continue

        ok, reason = is_doc_valid(
            crawler, pd_api,
            args.min_page_chars, args.min_desc_chars, args.delay,
        )
        checked += 1

        if ok:
            print(f"  ✅ [{idx}/{total}] {pd_api}")
            continue

        invalid += 1
        row["paddle-api"] = "no_matching_impl"
        row["reason"] = build_reason(row.get("reason", ""), f"paddle_doc_invalid:{reason}")
        print(f"  ❌ [{idx}/{total}] {pd_api} ({reason})")

    save_csv_rows(args.output, rows, fieldnames)

    print("=" * 80)
    print("Validation complete (MS->PD)")
    print("=" * 80)
    print(f"Total rows: {total}")
    print(f"Checked entries: {checked}")
    print(f"Invalid entries: {invalid}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
