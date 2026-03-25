#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3.5b: 验证 MindSpore API 是否真实存在（基于官方文档页面）

功能：
- 读取 TF→MS 映射 CSV
- 对每个 mindspore-api 拉取官方文档
- 若文档不存在、软404或内容异常，则将 mindspore-api 改为“无对应实现”
- 输出新的 CSV

用法：
    conda activate tf_env
    python tf_ms_test_1/validate_ms_api_docs.py \
        --input tf_ms_test_1/data/tf_ms_mapping_high.csv \
        --output tf_ms_test_1/data/tf_ms_mapping_validated.csv
"""

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_crawler_mindspore import MindSporeDocCrawler

DEFAULT_INPUT = os.path.join("tf_ms_test_1", "data", "tf_ms_mapping_high.csv")
DEFAULT_OUTPUT = os.path.join("tf_ms_test_1", "data", "tf_ms_mapping_validated.csv")
DEFAULT_DELAY = 0.5
DEFAULT_MIN_HTML_CHARS = 2000
DEFAULT_MIN_DESC_CHARS = 50

# MindSpore 文档站点可能返回 HTTP 200 但页面实际不存在（软404）
MS_SOFT_404_TITLE_PATTERNS = [
    "404",
    "not found",
    "page not found",
    "页面不存在",
    "找不到页面",
    "error",
]

MS_SOFT_404_BODY_PATTERNS = [
    "page not found",
    "not found",
    "404",
    "the page you are looking for",
    "抱歉",
    "页面不存在",
    "无法找到",
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


def _contains_pattern(text: str, patterns: List[str]) -> bool:
    normalized_text = (text or "").strip().lower()
    return any(pattern in normalized_text for pattern in patterns)


def _is_soft_404(title: str, raw_html: str, api_name: str) -> bool:
    title_text = (title or "").strip()
    body_text = re.sub(r"\s+", " ", (raw_html or "")).lower()

    title_flag = _contains_pattern(title_text, MS_SOFT_404_TITLE_PATTERNS)
    body_flag = _contains_pattern(body_text, MS_SOFT_404_BODY_PATTERNS)

    api_last = api_name.split(".")[-1].lower() if api_name else ""
    has_api_hint = (api_name.lower() in body_text) or (api_last and api_last in body_text)

    # 标题明确是404，且正文没有 API 内容，判定软404
    if title_flag and not has_api_hint:
        return True

    # 正文具备典型404语义且标题缺少 API 名，也判定软404
    if body_flag and api_name.lower() not in (title_text or "").lower():
        return True

    return False


def is_doc_valid(
    crawler: MindSporeDocCrawler,
    api_name: str,
    min_html_chars: int,
    min_desc_chars: int,
    delay: float,
) -> Tuple[bool, str]:
    """检查 MindSpore API 文档是否可信（含软404校验）"""
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

    if _is_soft_404(title, raw_html, normalized):
        return False, "soft_404"

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


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 MindSpore API 文档并修正映射")
    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT,
        help="输入 TF→MS 映射 CSV 路径",
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
        "--min-html-chars",
        type=int,
        default=DEFAULT_MIN_HTML_CHARS,
        help=f"文档主区域最小字符数阈值（默认 {DEFAULT_MIN_HTML_CHARS}）",
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
    crawler = MindSporeDocCrawler()

    for idx, row in enumerate(rows, start=1):
        ms_api = normalize_api_name(row.get("mindspore-api", ""))
        if not ms_api or ms_api == "无对应实现":
            continue

        ok, reason = is_doc_valid(
            crawler,
            ms_api,
            args.min_html_chars,
            args.min_desc_chars,
            args.delay,
        )
        checked += 1

        if ok:
            print(f"  ✅ [{idx}/{total}] {ms_api}")
            continue

        invalid += 1
        row["mindspore-api"] = "无对应实现"
        row["reason"] = build_reason(row.get("reason", ""), f"mindspore_doc_invalid:{reason}")
        print(f"  ❌ [{idx}/{total}] {ms_api} ({reason})")

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
