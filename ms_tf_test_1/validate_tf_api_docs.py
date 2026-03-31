#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3.5b: Validate filtered TensorFlow API docs

Purpose:
- Read ms_tf_mapping_high_confidence.csv from Step 3.5a
- Use TensorFlowDocCrawler to validate each TF API doc
- If docs cannot be fetched, try direct TF official doc URL
- Convert invalid "has-match" mappings to "no_matching_impl"
- Output the validated CSV

Direct URL fallback:
    For TF APIs (e.g., tf.abs), build URL:
    https://www.tensorflow.org/api_docs/python/tf/abs
    Request the URL and verify the HTML contains real doc content.

Usage:
        conda activate tf_env
        python ms_tf_test_1/validate_tf_api_docs.py [--input] [--output]

Output: ms_tf_test_1/data/ms_tf_mapping_validated.csv
"""

import os
import sys
import csv
import re
import time
import argparse
from typing import Dict, List, Tuple

import requests
from bs4 import BeautifulSoup

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from component.doc.doc_crawler_tensorflow import TensorFlowDocCrawler


TF_DOC_BASE = "https://www.tensorflow.org/api_docs/python/"
REQUEST_TIMEOUT = 15
DEFAULT_MIN_HTML_CHARS = 2000
DEFAULT_MIN_DESC_CHARS = 30

# Common keywords for TF soft-404 / generic pages
TF_INVALID_TITLE_PATTERNS = [
    r"\b404\b",
    r"not\s+found",
    r"page\s+not\s+found",
    r"error",
]


def normalize_api_name(api_name: str) -> str:
    return (api_name or "").strip()


def build_reason(original_reason: str, new_reason: str) -> str:
    original_reason = (original_reason or "").strip()
    if not original_reason:
        return new_reason
    if new_reason in original_reason:
        return original_reason
    return f"{original_reason}; {new_reason}"


def _contains_invalid_title(title: str) -> bool:
    title = (title or "").strip().lower()
    if not title:
        return True
    return any(re.search(pattern, title, flags=re.IGNORECASE) for pattern in TF_INVALID_TITLE_PATTERNS)


def _has_api_identity(api_name: str, title: str, h1_text: str, text_blob: str) -> bool:
    normalized_api = normalize_api_name(api_name)
    if not normalized_api:
        return False

    normalized_texts = "\n".join([title or "", h1_text or "", text_blob or ""])
    lower_texts = normalized_texts.lower()
    api_lower = normalized_api.lower()
    api_tail = normalized_api.split(".")[-1].lower()

    # Require full API name or tail + tf semantics to avoid generic pages
    if api_lower in lower_texts:
        return True
    if api_tail and api_tail in lower_texts and "tf" in lower_texts:
        return True
    if f"{normalized_api}(".lower() in lower_texts:
        return True
    return False


def _is_doc_payload_valid(
    api_name: str,
    raw_html: str,
    description: str,
    title: str,
    h1_text: str,
    min_html_chars: int,
    min_desc_chars: int,
) -> Tuple[bool, str]:
    if _contains_invalid_title(title):
        return False, "invalid_title"

    if len((raw_html or "").strip()) < min_html_chars and len((description or "").strip()) < min_desc_chars:
        return False, "doc_too_short"

    text_blob = f"{description}\n{raw_html}"[:12000]
    if not _has_api_identity(api_name, title, h1_text, text_blob):
        return False, "api_identity_mismatch"

    return True, "ok"


def validate_tf_api_doc(
    api_name: str,
    crawler: TensorFlowDocCrawler,
    min_html_chars: int = DEFAULT_MIN_HTML_CHARS,
    min_desc_chars: int = DEFAULT_MIN_DESC_CHARS,
) -> Tuple[bool, str]:
    """
    Validate TF API docs via TensorFlowDocCrawler.

    Args:
        api_name: TF API name (e.g., tf.abs, tf.keras.layers.Conv2D)
        crawler: TensorFlowDocCrawler instance
        min_html_chars: Minimum HTML length to consider valid
        min_desc_chars: Minimum description length to consider valid

    Returns:
        (is_valid, reason)
    """
    normalized_api = normalize_api_name(api_name)
    if not normalized_api or normalized_api == "no_matching_impl":
        return False, "empty_or_none"

    try:
        result = crawler.crawl(normalized_api)
        if result and isinstance(result, dict):
            raw_html = result.get("raw_html", "") or ""
            desc = result.get("description", "") or ""
            title = result.get("title", "") or ""
            h1_text = ""

            ok, reason = _is_doc_payload_valid(
                normalized_api,
                raw_html,
                desc,
                title,
                h1_text,
                min_html_chars,
                min_desc_chars,
            )
            if ok:
                return True, "ok_crawler"
            return False, f"crawler_{reason}"
    except Exception:
        return False, "crawler_exception"

    return False, "crawler_empty"


def validate_tf_api_doc_via_url(
    api_name: str,
    min_html_chars: int = DEFAULT_MIN_HTML_CHARS,
    min_desc_chars: int = DEFAULT_MIN_DESC_CHARS,
) -> Tuple[bool, str]:
    """
    Validate API existence by directly requesting the TF official doc URL.

    Returns:
        (is_valid, reason)
    """
    normalized_api = normalize_api_name(api_name)
    if not normalized_api or normalized_api == "no_matching_impl":
        return False, "empty_or_none"
    if not normalized_api.startswith("tf."):
        return False, "not_tf_namespace"

    url_path = normalized_api.replace(".", "/")
    expected_url = f"{TF_DOC_BASE}{url_path}"

    try:
        resp = requests.get(expected_url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    except Exception as e:
        return False, f"request_exception:{str(e)[:80]}"

    if resp.status_code != 200:
        return False, f"http_{resp.status_code}"

    final_url = (resp.url or "").rstrip("/")
    if TF_DOC_BASE.rstrip("/") not in final_url:
        # Avoid redirects to non-API pages
        return False, f"unexpected_redirect:{final_url}"

    html = resp.text or ""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    h1 = soup.find("h1")
    h1_text = h1.get_text(strip=True) if h1 else ""

    main_content = soup.find("main") or soup.find("div", class_="devsite-article-body") or soup.find("body")
    raw_html = str(main_content) if main_content else ""

    description = ""
    if main_content:
        p_tag = main_content.find("p")
        if p_tag:
            description = p_tag.get_text(strip=True)

    ok, reason = _is_doc_payload_valid(
        normalized_api,
        raw_html,
        description,
        title,
        h1_text,
        min_html_chars,
        min_desc_chars,
    )

    if ok:
        return True, "ok_url"
    return False, f"url_{reason}"


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


def main():
    parser = argparse.ArgumentParser(description="Validate TensorFlow API docs and fix mappings")
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(ROOT_DIR, "ms_tf_test_1", "data", "ms_tf_mapping_high_confidence.csv"),
        help="Input filtered CSV path",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "ms_tf_test_1", "data", "ms_tf_mapping_validated.csv"),
        help="Output validated CSV path",
    )
    parser.add_argument(
        "--delay", "-d", type=float, default=0.3,
        help="Delay between doc validations in seconds (default 0.3)",
    )
    parser.add_argument(
        "--min-html-chars", type=int, default=DEFAULT_MIN_HTML_CHARS,
        help=f"Minimum HTML char threshold (default {DEFAULT_MIN_HTML_CHARS})",
    )
    parser.add_argument(
        "--min-desc-chars", type=int, default=DEFAULT_MIN_DESC_CHARS,
        help=f"Minimum description char threshold (default {DEFAULT_MIN_DESC_CHARS})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return

    rows, fieldnames = load_csv_rows(args.input)
    if not fieldnames:
        print("❌ CSV parse failed: header is empty")
        return

    # Filter rows that need doc validation
    crawler = TensorFlowDocCrawler()

    total = len(rows)
    checked = 0
    invalid = 0
    validated_count = 0
    fallback_count = 0

    for idx, row in enumerate(rows, start=1):
        tf_api = normalize_api_name(row.get("tensorflow-api", ""))
        if not tf_api or tf_api == "no_matching_impl":
            continue

        checked += 1

        is_valid, reason = validate_tf_api_doc(
            tf_api,
            crawler,
            min_html_chars=args.min_html_chars,
            min_desc_chars=args.min_desc_chars,
        )
        time.sleep(args.delay)

        if is_valid:
            validated_count += 1
            print(f"  ✅ [{idx}/{total}] {tf_api}")
        else:
            # Fallback: request TF official URL directly
            is_valid_url, reason_url = validate_tf_api_doc_via_url(
                tf_api,
                min_html_chars=args.min_html_chars,
                min_desc_chars=args.min_desc_chars,
            )
            time.sleep(args.delay)

            if is_valid_url:
                fallback_count += 1
                print(f"  ✅ [{idx}/{total}] {tf_api} (url_fallback)")
            else:
                invalid += 1
                row["tensorflow-api"] = "no_matching_impl"
                row["reason"] = build_reason(
                    row.get("reason", ""),
                    f"tf_doc_invalid:crawler={reason},url={reason_url},original={tf_api}",
                )
                print(f"  ❌ [{idx}/{total}] {tf_api} ({reason}; {reason_url})")

    save_csv_rows(args.output, rows, fieldnames)

    print("=" * 80)
    print("Validation complete (MS -> TF)")
    print("=" * 80)
    print(f"Total rows: {total}")
    print(f"Checked entries: {checked}")
    print(f"Invalid entries: {invalid}")
    print(f"URL fallback passed: {fallback_count}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
