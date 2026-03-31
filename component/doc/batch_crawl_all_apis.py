"""Batch crawl all related TF/PT API docs and write cache + summary file.

Goals:
- Enumerate all possible TensorFlow / PyTorch APIs from existing mappings
- Use the existing doc crawler (with local cache) to fetch docs in batch
- Generate a summary JSONL for offline viewing or statistics

Usage (examples)::

    # Recommended: only crawl APIs appearing in mappings
    python3 component/doc/batch_crawl_all_apis.py \
        --pairs data/components/component_pairs.jsonl \
        --out data/analysis/api_docs.jsonl

    # Limit count (for debugging)
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

# Add project root to sys.path to allow direct execution imports
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_crawler_factory import crawl_doc  # noqa: E402


def load_jsonl(path: Path) -> Iterable[Dict]:
    """Safely load a JSONL file."""
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
    """Collect TF / PT APIs from component_pairs.jsonl."""
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
    ap = argparse.ArgumentParser(description="Batch crawl TF/PT docs (with cache)")
    ap.add_argument(
        "--pairs",
        default="data/components/component_pairs.jsonl",
        help="TF/PT API mapping file (with tf_api / pt_api fields)",
    )
    ap.add_argument(
        "--out",
        default="data/analysis/api_docs.jsonl",
        help="Summary JSONL output path",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Max APIs to crawl (-1 means all)",
    )
    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not pairs_path.exists():
        print(f"[ERROR] Mapping file not found: {pairs_path}")
        return

    api_sets = collect_apis_from_pairs(pairs_path)
    tf_apis = sorted(api_sets["tensorflow"])
    pt_apis = sorted(api_sets["pytorch"])

    total_apis = len(tf_apis) + len(pt_apis)
    if args.limit > 0:
        # Simple limit: truncate combined TF/PT lists
        tf_quota = min(len(tf_apis), args.limit // 2 if args.limit > 1 else args.limit)
        pt_quota = min(len(pt_apis), args.limit - tf_quota)
        tf_apis = tf_apis[:tf_quota]
        pt_apis = pt_apis[:pt_quota]
        print(
            f"[INFO] Total APIs {total_apis}, limited to {args.limit} "
            f"(TF: {len(tf_apis)}, PT: {len(pt_apis)})"
        )
    else:
        print(f"[INFO] Crawling all APIs, TF: {len(tf_apis)}, PT: {len(pt_apis)}")

    with out_path.open("w", encoding="utf-8") as fout:
        # Crawl TF first
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

        # Crawl PT next
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

    print(f"[DONE] Doc crawl completed, results written to: {out_path}")


if __name__ == "__main__":
    main()


