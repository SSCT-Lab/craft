#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check whether TF APIs exist (based on official English docs)

Features:
- Read API list from tf_pt_test/data/tf_apis_new.json
- Visit TensorFlow official English docs per API
- Use redirect detection, page length, and API keyword matching to validate
- Output a file that keeps only valid APIs

Usage:
    conda activate tf_env
    python tf_pt_test/filter_existing_tf_apis.py \
        --input tf_pt_test/data/tf_apis_new.json \
        --output tf_pt_test/data/tf_apis_existing.json
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from bs4 import BeautifulSoup

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_crawler_tensorflow import TensorFlowDocCrawler

DEFAULT_INPUT = str(ROOT / "tf_pt_test" / "data" / "tf_apis_new.json")
DEFAULT_OUTPUT = str(ROOT / "tf_pt_test" / "data" / "tf_apis_existing.json")
DEFAULT_DELAY = 0.5
DEFAULT_MIN_PAGE_CHARS = 2000
DEFAULT_MIN_DESC_CHARS = 50


def load_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(filepath: str, data: Dict[str, Any]) -> None:
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_expected_path(url: str) -> str:
    """Extract expected path from URL to detect redirects to homepage."""
    if "tensorflow.org" not in url:
        return ""
    return url.split("tensorflow.org", 1)[-1].rstrip("/")


def fetch_tf_doc(
    crawler: TensorFlowDocCrawler,
    api_name: str,
    delay: float,
    min_page_chars: int,
    min_desc_chars: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Visit the doc page and determine if the API exists."""
    normalized_api = crawler.normalize_api_name(api_name)
    url = crawler.build_doc_url(normalized_api)

    try:
        time.sleep(delay)
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
    except Exception as exc:
        return False, {
            "api": api_name,
            "url": url,
            "reason": f"request_failed: {exc}",
        }

    if response.status_code != 200:
        return False, {
            "api": api_name,
            "url": url,
            "status_code": response.status_code,
            "reason": "non_200_status",
        }

    page_text = response.text or ""
    page_len = len(page_text)
    final_url = response.url or url
    redirected = final_url.rstrip("/") != url.rstrip("/")
    expected_path = build_expected_path(url)
    final_path = build_expected_path(final_url)

    soup = BeautifulSoup(page_text, "html.parser")
    doc_content = crawler.parse_doc_content(soup, normalized_api, final_url)

    title = doc_content.get("title", "")
    desc = doc_content.get("description", "")
    raw_html = doc_content.get("raw_html", "")
    signature = doc_content.get("signature", "")

    api_match = (
        api_name in title
        or api_name in signature
        or api_name in raw_html
        or normalized_api in title
        or normalized_api in signature
        or normalized_api in raw_html
    )

    last_part = normalized_api.split(".")[-1] if normalized_api else ""
    last_match = last_part and (last_part in title or last_part in signature)

    if page_len < min_page_chars and len(desc) < min_desc_chars:
        return False, {
            "api": api_name,
            "url": url,
            "final_url": final_url,
            "page_len": page_len,
            "reason": "page_too_short",
        }

    if redirected and expected_path and expected_path != final_path and not api_match:
        return False, {
            "api": api_name,
            "url": url,
            "final_url": final_url,
            "page_len": page_len,
            "reason": "redirect_no_match",
        }

    if not api_match and not last_match and len(desc) < min_desc_chars:
        return False, {
            "api": api_name,
            "url": url,
            "final_url": final_url,
            "page_len": page_len,
            "reason": "no_api_match",
        }

    return True, {
        "api": api_name,
        "url": url,
        "final_url": final_url,
        "page_len": page_len,
        "reason": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter to existing TensorFlow APIs")
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT,
        help="Input API list JSON file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help="Output filtered API list JSON file path",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay seconds per request (default {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--min-page-chars",
        type=int,
        default=DEFAULT_MIN_PAGE_CHARS,
        help=f"Minimum page length threshold (default {DEFAULT_MIN_PAGE_CHARS})",
    )
    parser.add_argument(
        "--min-desc-chars",
        type=int,
        default=DEFAULT_MIN_DESC_CHARS,
        help=f"Minimum description length threshold (default {DEFAULT_MIN_DESC_CHARS})",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return

    data = load_json(args.input)
    apis: List[Dict[str, Any]] = data.get("apis", [])
    print(f"📋 Loaded {len(apis)} TF APIs")

    crawler = TensorFlowDocCrawler()

    valid_apis: List[Dict[str, Any]] = []
    invalid_apis: List[Dict[str, Any]] = []

    for idx, item in enumerate(apis, start=1):
        api_name = item.get("tf_api", "").strip()
        if not api_name:
            invalid_apis.append({"api": api_name, "reason": "empty_api"})
            continue

        is_valid, detail = fetch_tf_doc(
            crawler,
            api_name,
            args.delay,
            args.min_page_chars,
            args.min_desc_chars,
        )

        if is_valid:
            valid_apis.append(item)
            print(f"  ✅ [{idx}/{len(apis)}] {api_name}")
        else:
            invalid_apis.append(detail)
            print(f"  ❌ [{idx}/{len(apis)}] {api_name} ({detail.get('reason')})")

    output_data = {
        "total_apis": len(valid_apis),
        "source_file": args.input,
        "filtered_out": len(invalid_apis),
        "min_page_chars": args.min_page_chars,
        "min_desc_chars": args.min_desc_chars,
        "apis": valid_apis,
        "invalid_apis": invalid_apis,
    }

    save_json(args.output, output_data)

    print("=" * 80)
    print("Filtering complete")
    print("=" * 80)
    print(f"Kept API count: {len(valid_apis)}")
    print(f"Removed API count: {len(invalid_apis)}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
