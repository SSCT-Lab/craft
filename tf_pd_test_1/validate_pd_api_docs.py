#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3.5b: 验证 PaddlePaddle API 是否真实存在（基于官方文档页面）

功能：
- 读取 TF→PD 映射 CSV
- 对每个 paddle-api 拉取官方文档
- 若文档不存在或内容异常，则将 paddle-api 改为“无对应实现”
- 输出新的 CSV

用法：
    conda activate tf_env
    python tf_pd_test_1/validate_pd_api_docs.py \
        --input tf_pd_test_1/data/tf_pd_mapping_high.csv \
        --output tf_pd_test_1/data/tf_pd_mapping_validated.csv
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_crawler_paddle import PaddleDocCrawler

DEFAULT_INPUT = os.path.join("tf_pd_test_1", "data", "tf_pd_mapping_high.csv")
DEFAULT_OUTPUT = os.path.join("tf_pd_test_1", "data", "tf_pd_mapping_validated.csv")
DEFAULT_DELAY = 0.5
DEFAULT_MIN_PAGE_CHARS = 2000
DEFAULT_MIN_DESC_CHARS = 50
REQUEST_TIMEOUT = 10

# Paddle 文档不存在时常返回软 404 页面（HTTP 200），需额外检测标题
PADDLE_GENERIC_PAGE_TITLES = [
    "Guides-Document-PaddlePaddle Deep Learning Platform",
    "使用指南-文档-PaddlePaddle深度学习平台",
]

PADDLE_API_TITLE_SUFFIX = "API Document-PaddlePaddle Deep Learning Platform"
PADDLE_API_TITLE_SUFFIX_CN = "API文档-PaddlePaddle深度学习平台"

PADDLE_DOC_BASES = [
    ("en", "https://www.paddlepaddle.org.cn/documentation/docs/en/api/", "_en.html"),
    ("zh", "https://www.paddlepaddle.org.cn/documentation/docs/zh/api/", "_cn.html"),
]


def normalize_api_name(api_name: str) -> str:
    return (api_name or "").strip().lstrip(".")


def build_reason(original_reason: str, new_reason: str) -> str:
    original_reason = (original_reason or "").strip()
    if not original_reason:
        return new_reason
    if new_reason in original_reason:
        return original_reason
    return f"{original_reason}; {new_reason}"


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
    normalized = normalize_api_name(api_name)
    if not normalized:
        return False, "empty_api"

    # 先使用统一的 Paddle 爬取器抓取文档主内容
    # 注意：Paddle 站点有软404，因此仍需保留后续 URL 级别的软404校验
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
            continue

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


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 PaddlePaddle API 文档并修正映射")
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT,
        help="输入 TF→PD 映射 CSV 路径",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help="输出修正后的 CSV 路径",
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

    rows, fieldnames = load_csv_rows(args.input)
    if not fieldnames:
        print("❌ CSV 解析失败：表头为空")
        return

    total = len(rows)
    checked = 0
    invalid = 0
    crawler = PaddleDocCrawler()

    for idx, row in enumerate(rows, start=1):
        pd_api = normalize_api_name(row.get("paddle-api", ""))
        if not pd_api or pd_api == "无对应实现":
            continue

        ok, reason = is_doc_valid(
            crawler,
            pd_api,
            args.min_page_chars,
            args.min_desc_chars,
            args.delay,
        )
        checked += 1

        if ok:
            print(f"  ✅ [{idx}/{total}] {pd_api}")
            continue

        invalid += 1
        row["paddle-api"] = "无对应实现"
        row["reason"] = build_reason(row.get("reason", ""), f"paddle_doc_invalid:{reason}")
        print(f"  ❌ [{idx}/{total}] {pd_api} ({reason})")

    save_csv_rows(args.output, rows, fieldnames)

    print("=" * 80)
    print("验证完成")
    print("=" * 80)
    print(f"总行数: {total}")
    print(f"检查条目数: {checked}")
    print(f"无效条目数: {invalid}")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
