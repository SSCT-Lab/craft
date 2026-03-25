"""批量爬取所有相关 TF/PT API 文档并写入缓存 + 汇总文件.

目标：
- 根据现有映射数据，枚举所有可能用到的 TensorFlow / PyTorch API
- 调用现有的文档爬取器（带本地缓存）批量拉取文档
- 生成一个汇总 JSONL，方便后续离线查看或做统计

使用方式（示例）::

    # 推荐：只爬映射中出现过的 API
    python3 component/doc/batch_crawl_all_apis.py \
        --pairs data/components/component_pairs.jsonl \
        --out data/analysis/api_docs.jsonl

    # 如果想限制数量（调试用）
    python3 component/doc/batch_crawl_all_apis.py \
        --pairs data/components/component_pairs.jsonl \
        --out data/analysis/api_docs.sample.jsonl \
        --limit 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Set, Iterable

from tqdm import tqdm

# 添加项目根目录到 sys.path，保证直接运行时可导入
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_crawler_factory import crawl_doc  # noqa: E402


def load_jsonl(path: Path) -> Iterable[Dict]:
    """安全加载 JSONL 文件."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def collect_apis_from_pairs(pairs_path: Path) -> Dict[str, Set[str]]:
    """从 component_pairs.jsonl 中收集 TF / PT API."""
    tf_apis: Set[str] = set()
    pt_apis: Set[str] = set()

    for item in load_jsonl(pairs_path):
        tf_api = item.get("tf_api")
        pt_api = item.get("pt_api")
        if tf_api:
            tf_apis.add(tf_api)
        if pt_api:
            pt_apis.add(pt_api)

    return {"tensorflow": tf_apis, "pytorch": pt_apis}


def main() -> None:
    ap = argparse.ArgumentParser(description="批量爬取 TF/PT 文档（带缓存）")
    ap.add_argument(
        "--pairs",
        default="data/components/component_pairs.jsonl",
        help="TF/PT API 映射文件（含 tf_api / pt_api 字段）",
    )
    ap.add_argument(
        "--out",
        default="data/analysis/api_docs.jsonl",
        help="汇总输出 JSONL 路径",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="最多爬取多少个 API（-1 表示全部）",
    )
    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not pairs_path.exists():
        print(f"[ERROR] 映射文件不存在: {pairs_path}")
        return

    api_sets = collect_apis_from_pairs(pairs_path)
    tf_apis = sorted(api_sets["tensorflow"])
    pt_apis = sorted(api_sets["pytorch"])

    total_apis = len(tf_apis) + len(pt_apis)
    if args.limit > 0:
        # 简单限制：在 TF/PT 合集中截断
        tf_quota = min(len(tf_apis), args.limit // 2 if args.limit > 1 else args.limit)
        pt_quota = min(len(pt_apis), args.limit - tf_quota)
        tf_apis = tf_apis[:tf_quota]
        pt_apis = pt_apis[:pt_quota]
        print(
            f"[INFO] 总 API 数 {total_apis}，限制为 {args.limit} "
            f"(TF: {len(tf_apis)}, PT: {len(pt_apis)})"
        )
    else:
        print(f"[INFO] 将爬取全部 API，TF: {len(tf_apis)}, PT: {len(pt_apis)}")

    with out_path.open("w", encoding="utf-8") as fout:
        # 先爬 TF
        for api in tqdm(tf_apis, desc="Crawling TensorFlow docs"):
            try:
                doc = crawl_doc(api, "tensorflow")
                rec = {
                    "framework": "tensorflow",
                    "api": api,
                    "ok": bool(doc),
                    "doc": doc or {},
                }
            except Exception as e:  # noqa: BLE001
                rec = {
                    "framework": "tensorflow",
                    "api": api,
                    "ok": False,
                    "error": str(e),
                    "doc": {},
                }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

        # 再爬 PT
        for api in tqdm(pt_apis, desc="Crawling PyTorch docs"):
            try:
                doc = crawl_doc(api, "pytorch")
                rec = {
                    "framework": "pytorch",
                    "api": api,
                    "ok": bool(doc),
                    "doc": doc or {},
                }
            except Exception as e:  # noqa: BLE001
                rec = {
                    "framework": "pytorch",
                    "api": api,
                    "ok": False,
                    "error": str(e),
                    "doc": {},
                }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[DONE] 文档抓取完成，结果已写入: {out_path}")


if __name__ == "__main__":
    main()


