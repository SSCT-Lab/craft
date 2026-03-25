#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1.5: 过滤不存在的 Paddle API

功能：
- 读取 Step 1 输出的 API 列表
- 访问 PaddlePaddle 官方文档页面，验证 API 是否真实存在
- 输出仅保留验证通过的 API 列表

用法：
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
    """规范化 API 名称"""
    return (api_name or "").strip().lstrip(".")


# Paddle 文档网站对不存在的 URL 返回 HTTP 200（软 404），
# 页面标题为通用介绍页，需要特征检测来识别。
PADDLE_GENERIC_PAGE_TITLES = [
    "Guides-Document-PaddlePaddle Deep Learning Platform",
    "使用指南-文档-PaddlePaddle深度学习平台",
]

# 真实 API 页面标题的后缀模式（如 "add-API Document-PaddlePaddle ..."）
PADDLE_API_TITLE_SUFFIX = "API Document-PaddlePaddle Deep Learning Platform"
PADDLE_API_TITLE_SUFFIX_CN = "API文档-PaddlePaddle深度学习平台"

# 英文和中文文档的 base URL（部分 API 只有中文文档）
PADDLE_DOC_BASES = [
    ("en", "https://www.paddlepaddle.org.cn/documentation/docs/en/api/", "_en.html"),
    ("zh", "https://www.paddlepaddle.org.cn/documentation/docs/zh/api/", "_cn.html"),
]

REQUEST_TIMEOUT = 10


def _build_doc_url(api_name: str, base: str, suffix: str) -> str:
    """构建 Paddle 文档 URL"""
    path = api_name.lstrip(".").replace(".", "/")
    return f"{base}{path}{suffix}"


def _check_page_valid(
    raw_html: str, title: str, api_name: str, min_page_chars: int,
) -> Tuple[bool, str]:
    """
    判断一个 Paddle 文档页面是否为真实 API 页面

    返回 (valid, reason)
    """
    title_stripped = title.strip()

    # ===== 检查 1: 检测软 404 通用页面 =====
    if title_stripped in PADDLE_GENERIC_PAGE_TITLES:
        return False, "soft_404_generic_page"

    # ===== 检查 2: 标题是否包含 "API Document" 或 "API文档" =====
    if (PADDLE_API_TITLE_SUFFIX not in title
            and PADDLE_API_TITLE_SUFFIX_CN not in title):
        return False, "not_api_page"

    # ===== 检查 3: 内容长度 =====
    if len(raw_html) < min_page_chars:
        return False, "doc_too_short"

    # ===== 检查 4: API 名称匹配 =====
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
    检查 Paddle API 文档是否存在且有效

    验证策略：
    1. 依次尝试英文文档、中文文档（部分 API 只有中文文档）
    2. 对每个 URL 检测软 404 特征（通用页面标题）
    3. 验证内容长度和 API 名称匹配度
    4. 不使用 crawler.crawl() 避免缓存了错误的通用页面
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
        description="Step 1.5: 过滤不存在的 Paddle API"
    )
    parser.add_argument(
        "--input", "-i", default=DEFAULT_INPUT,
        help="Step 1 输出文件路径"
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help="过滤后输出文件路径"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"每次请求延迟秒数（默认 {DEFAULT_DELAY}）"
    )
    parser.add_argument(
        "--min-page-chars", type=int, default=DEFAULT_MIN_PAGE_CHARS,
        help=f"页面最小字符数阈值（默认 {DEFAULT_MIN_PAGE_CHARS}）"
    )
    parser.add_argument(
        "--min-desc-chars", type=int, default=DEFAULT_MIN_DESC_CHARS,
        help=f"描述最小字符数阈值（默认 {DEFAULT_MIN_DESC_CHARS}）"
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="清除之前的 Paddle 文档缓存（推荐在修改验证逻辑后使用）"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Step 1.5: 过滤不存在的 Paddle API")
    print("=" * 80)

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        sys.exit(1)

    # 清除旧缓存（避免之前缓存的通用 404 页面影响结果）
    if args.clear_cache:
        import glob
        cache_pattern = os.path.join("data", "docs_cache", "paddle_*.json")
        cached_files = glob.glob(cache_pattern)
        if cached_files:
            for cf in cached_files:
                os.remove(cf)
            print(f"🗑️ 已清除 {len(cached_files)} 个 Paddle 文档缓存文件")
        else:
            print("ℹ️ 无 Paddle 文档缓存需要清除")

    with open(args.input, 'r', encoding='utf-8') as f:
        api_data = json.load(f)

    all_apis = api_data.get("apis", [])
    print(f"📋 共加载 {len(all_apis)} 个 Paddle API")

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

    # 保存结果
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
    print(f"📊 过滤结果")
    print(f"{'=' * 80}")
    print(f"  原始API数: {len(all_apis)}")
    print(f"  有效API数: {len(valid_apis)}")
    print(f"  过滤掉: {len(invalid_apis)}")
    print(f"  💾 已保存到: {args.output}")


if __name__ == "__main__":
    main()
