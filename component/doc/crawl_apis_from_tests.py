"""从测试文件中提取的 API 批量下载文档

使用 extract_apis_from_tests.py 提取的 API 列表，批量下载文档。
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
    parser = argparse.ArgumentParser(description="从测试 API 列表批量下载文档")
    parser.add_argument(
        "--input",
        default="data/analysis/test_apis.jsonl",
        help="输入 API 列表文件（JSONL，每行包含 framework 和 api）"
    )
    parser.add_argument(
        "--output",
        default="data/analysis/test_api_docs.jsonl",
        help="输出文档文件（JSONL）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="限制下载数量（-1 表示全部）"
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}")
        return
    
    # 读取 API 列表
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
    
    print(f"[INFO] 将下载 {len(apis)} 个 API 的文档")
    
    # 下载文档
    with output_path.open("w", encoding="utf-8") as fout:
        for item in tqdm(apis, desc="下载文档"):
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
    
    print(f"\n[DONE] 文档下载完成: {output_path}")


if __name__ == "__main__":
    main()


