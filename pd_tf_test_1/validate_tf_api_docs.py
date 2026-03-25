#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3.5b: 验证 TensorFlow API 是否真实存在（基于官方文档页面）

功能：
- 读取 PD→TF 映射 CSV
- 对每个 tensorflow-api 拉取官方文档
- 若文档不存在或内容过短，则将 tensorflow-api 改为"无对应实现"
- 输出新的 CSV

TensorFlow 文档特点：
- 文档地址格式: https://www.tensorflow.org/api_docs/python/tf/xxx/yyy
- TF 的 API 采用斜线分隔路径（而非点号），如 tf.keras.layers.Conv2D → tf/keras/layers/Conv2D
- TF 有官方 API 搜索页面可用于兜底验证

用法：
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
# TF 使用直接的文档 URL 进行验证，同时可用 _api/xxx 端点做兜底
TF_API_BASE = "https://www.tensorflow.org/api_docs/python/"
REQUEST_TIMEOUT = 15


def normalize_api_name(api_name: str) -> str:
    return (api_name or "").strip().lstrip(".")


def _build_tf_doc_url(api_name: str) -> str:
    """
    构建 TF API 的官方文档 URL。
    例如：tf.keras.layers.Conv2D → https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
    """
    normalized = normalize_api_name(api_name)
    # TF 文档路径用斜线分隔
    path = normalized.replace(".", "/")
    return f"{TF_API_BASE}{path}"


@lru_cache(maxsize=16)
def _check_tf_url_exists(url: str) -> Tuple[bool, str]:
    """
    直接通过 HTTP HEAD/GET 请求检查 TF 文档 URL 是否存在。
    TF 文档服务器对不存在的 API 返回 404 或重定向到搜索页面。
    """
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if response.status_code == 200:
            # 检查是否被重定向到搜索页面或 404 页面
            if "Page not found" in response.text or "404" in response.text[:500]:
                return False, "redirected_to_404"
            return True, "ok_direct"
        return False, f"http_{response.status_code}"
    except Exception as e:
        return False, f"request_error:{str(e)[:50]}"


def has_tf_direct_url_match(api_name: str) -> bool:
    """
    通过直接请求 TF 文档 URL 验证 API 是否存在。
    这是 TF 框架特有的兜底方式（TF 文档 URL 结构清晰且稳定）。
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
    """检查 TensorFlow API 文档是否可信"""
    normalized = normalize_api_name(api_name)
    if not normalized:
        return False, "empty_api"

    time.sleep(delay)
    doc = crawler.crawl(normalized)
    if not doc:
        # 爬取失败，尝试直接 URL 访问兜底
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

    # 检查页面是否真的包含该 API 的相关内容
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
    parser = argparse.ArgumentParser(description="验证 TensorFlow API 文档并修正映射")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="输入 PD→TF 映射 CSV 路径")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="输出修正后的 CSV 路径")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help=f"每次请求延迟秒数（默认 {DEFAULT_DELAY}）")
    parser.add_argument("--min-html-chars", type=int, default=DEFAULT_MIN_HTML_CHARS, help=f"文档最小字符数（默认 {DEFAULT_MIN_HTML_CHARS}）")
    parser.add_argument("--min-desc-chars", type=int, default=DEFAULT_MIN_DESC_CHARS, help=f"描述最小字符数（默认 {DEFAULT_MIN_DESC_CHARS}）")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return

    rows, fieldnames = load_csv_rows(args.input)
    if not fieldnames:
        print("❌ CSV 解析失败：表头为空")
        return

    crawler = TensorFlowDocCrawler()

    total = len(rows)
    checked = 0
    invalid = 0

    for idx, row in enumerate(rows, start=1):
        tf_api = normalize_api_name(row.get("tensorflow-api", ""))
        if not tf_api or tf_api == "无对应实现":
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
        row["tensorflow-api"] = "无对应实现"
        row["reason"] = build_reason(row.get("reason", ""), f"tensorflow_doc_invalid:{reason}")
        print(f"  ❌ [{idx}/{total}] {tf_api} ({reason})")

    save_csv_rows(args.output, rows, fieldnames)

    print("=" * 80)
    print("验证完成（PD→TF）")
    print("=" * 80)
    print(f"总行数: {total}")
    print(f"检查条目数: {checked}")
    print(f"无效条目数: {invalid}")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
