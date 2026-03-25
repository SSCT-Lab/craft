#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3.5b: 验证 MindSpore API 是否真实存在（基于官方文档页面）

功能：
- 读取 PD→MS 映射 CSV
- 对每个 mindspore-api 拉取官方文档
- 若文档不存在或内容过短，则将 mindspore-api 改为"无对应实现"
- 输出新的 CSV

MindSpore 文档特点：
- 文档地址格式: https://www.mindspore.cn/docs/en/r2.8.0/api_python/{sub_module}/{api_name}.html
- 没有 genindex 页面，但可以通过 API 搜索页面进行兜底验证
- MindSpore 的 API 文档结构与 PyTorch 不同，需要适配

用法：
    conda activate tf_env
    python pd_ms_test_1/validate_ms_api_docs.py \
        --input pd_ms_test_1/data/pd_ms_mapping_high.csv \
        --output pd_ms_test_1/data/pd_ms_mapping_validated.csv
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

from component.doc.doc_crawler_mindspore import MindSporeDocCrawler

DEFAULT_INPUT = str(ROOT / "pd_ms_test_1" / "data" / "pd_ms_mapping_high.csv")
DEFAULT_OUTPUT = str(ROOT / "pd_ms_test_1" / "data" / "pd_ms_mapping_validated.csv")
DEFAULT_DELAY = 0.5
DEFAULT_MIN_HTML_CHARS = 1500
DEFAULT_MIN_DESC_CHARS = 20
# MindSpore 搜索 API 页面（用于兜底验证）
MS_SEARCH_URL = "https://www.mindspore.cn/docs/en/r2.8.0/api_python/mindspore/mindspore.ops.html"
REQUEST_TIMEOUT = 15


def normalize_api_name(api_name: str) -> str:
    return (api_name or "").strip().lstrip(".")


@lru_cache(maxsize=8)
def _load_ms_module_page(module_name: str) -> str:
    """
    加载 MindSpore 某个模块的 API 列表页面，用于兜底验证。

    MindSpore 没有全局 genindex，但每个子模块有自己的文档页面列出所有 API。
    例如：
    - mindspore.ops: https://www.mindspore.cn/docs/en/r2.8.0/api_python/mindspore.ops.html
    - mindspore.nn:  https://www.mindspore.cn/docs/en/r2.8.0/api_python/mindspore.nn.html
    """
    base_url = "https://www.mindspore.cn/docs/en/r2.8.0/api_python"
    url = f"{base_url}/{module_name}.html"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
    except Exception:
        return ""
    if response.status_code != 200:
        return ""
    return response.text


def _get_module_name(api_name: str) -> str:
    """
    从 API 名中提取模块名。
    例如：mindspore.ops.abs → mindspore.ops
          mindspore.nn.Conv2d → mindspore.nn
          mindspore.Tensor → mindspore
    """
    parts = api_name.split(".")
    if len(parts) <= 2:
        return "mindspore"
    # 取前两个部分作为模块名
    return ".".join(parts[:2])


def has_module_page_match(api_name: str) -> bool:
    """
    通过 MindSpore 模块文档页面做精确符号匹配，兜底文档爬取失败的场景。
    """
    normalized = normalize_api_name(api_name)
    if not normalized:
        return False

    module_name = _get_module_name(normalized)
    html = _load_ms_module_page(module_name)
    if not html:
        return False

    # 精确匹配 API 名称
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_\.]){re.escape(normalized)}(?![A-Za-z0-9_])",
        flags=re.IGNORECASE,
    )
    return bool(pattern.search(html))


def is_doc_valid(
    crawler: MindSporeDocCrawler,
    api_name: str,
    min_html_chars: int,
    min_desc_chars: int,
    delay: float,
) -> Tuple[bool, str]:
    """检查 MindSpore API 文档是否可信"""
    normalized = normalize_api_name(api_name)
    if not normalized:
        return False, "empty_api"

    time.sleep(delay)
    doc = crawler.crawl(normalized)
    if not doc:
        # 爬取失败，尝试模块页面兜底
        if has_module_page_match(normalized):
            return True, "ok_module_page_fallback"
        return False, "doc_not_found"

    raw_html = doc.get("raw_html", "") or ""
    description = doc.get("description", "") or ""
    title = doc.get("title", "") or ""

    # MindSpore 文档页面可能较短（相比 PyTorch），适当降低阈值
    if len(raw_html) < min_html_chars and len(description) < min_desc_chars:
        if has_module_page_match(normalized):
            return True, "ok_module_page_fallback"
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
        if has_module_page_match(normalized):
            return True, "ok_module_page_fallback"
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
    parser = argparse.ArgumentParser(description="验证 MindSpore API 文档并修正映射")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="输入 PD→MS 映射 CSV 路径")
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

    crawler = MindSporeDocCrawler()

    total = len(rows)
    checked = 0
    invalid = 0

    for idx, row in enumerate(rows, start=1):
        ms_api = normalize_api_name(row.get("mindspore-api", ""))
        if not ms_api or ms_api == "无对应实现":
            continue

        ok, reason = is_doc_valid(
            crawler, ms_api,
            args.min_html_chars, args.min_desc_chars, args.delay,
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
    print("验证完成（PD→MS）")
    print("=" * 80)
    print(f"总行数: {total}")
    print(f"检查条目数: {checked}")
    print(f"无效条目数: {invalid}")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
