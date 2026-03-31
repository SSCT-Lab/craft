"""Batch download docs for APIs extracted from test files.

Uses API list from extract_apis_from_tests.py to download docs in batch.
"""
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_crawler_factory import crawl_doc  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Batch download docs from test API list")
    parser.add_argument(
        "--input",
        default="data/analysis/test_apis.jsonl",
        help="Input API list file (JSONL, each line has framework and api)"
    )
    parser.add_argument(
        "--output",
        default="data/analysis/test_api_docs.jsonl",
        help="Output doc file (JSONL)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit download count (-1 means all)"
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return
    
    # Read API list
    apis = []
    for line in input_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            apis.append(json.loads(line))
        except Exception:
            continue
    
    if args.limit > 0:
        apis = apis[:args.limit]
    
    print(f"[INFO] Will download docs for {len(apis)} APIs")
    
    # Download docs
    with output_path.open("w", encoding="utf-8") as fout:
        for item in tqdm(apis, desc="Downloading docs"):
            framework = item.get("framework", "")
            api = item.get("api", "")
            
            if not api:
                continue
            
            try:
                doc = crawl_doc(api, framework)
                rec = {
                    "framework": framework,
                    "api": api,
                    "ok": bool(doc),
                    "doc": doc or {},
                }
            except Exception as e:
                rec = {
                    "framework": framework,
                    "api": api,
                    "ok": False,
                    "error": str(e),
                    "doc": {},
                }
            
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
    
    print(f"\n[DONE] Doc download complete: {output_path}")


if __name__ == "__main__":
    main()


