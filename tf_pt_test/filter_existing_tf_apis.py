#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 TF API 是否真实存在（基于官方英文文档页面）

功能：
- 读取 tf_pt_test/data/tf_apis_new.json 中的 API 列表
- 逐个访问 TensorFlow 官方英文文档页面
- 结合重定向与页面内容长度、API 关键词匹配判断是否真实存在
- 输出仅保留真实存在 API 的结果文件

用法：
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
    """从完整 URL 中提取期望路径，用于判断是否重定向到首页"""
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
    """访问文档页面并判断 API 是否存在"""
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
    parser = argparse.ArgumentParser(description="过滤出真实存在的 TensorFlow API")
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT,
        help="输入 API 列表 JSON 文件路径",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help="输出过滤后的 API 列表 JSON 文件路径",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"每次请求延迟秒数（默认 {DEFAULT_DELAY}）",
    )
    parser.add_argument(
        "--min-page-chars",
        type=int,
        default=DEFAULT_MIN_PAGE_CHARS,
        help=f"页面最小字符数阈值（默认 {DEFAULT_MIN_PAGE_CHARS}）",
    )
    parser.add_argument(
        "--min-desc-chars",
        type=int,
        default=DEFAULT_MIN_DESC_CHARS,
        help=f"描述最小字符数阈值（默认 {DEFAULT_MIN_DESC_CHARS}）",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return

    data = load_json(args.input)
    apis: List[Dict[str, Any]] = data.get("apis", [])
    print(f"📋 共加载 {len(apis)} 个 TF API")

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
    print("过滤完成")
    print("=" * 80)
    print(f"保留 API 数量: {len(valid_apis)}")
    print(f"剔除 API 数量: {len(invalid_apis)}")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
