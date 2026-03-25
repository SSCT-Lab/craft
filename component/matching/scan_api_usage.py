# component/scan_api_usage_parallel.py
import ast
import argparse
import ujson as json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def extract_apis_from_file(path: Path):
    """扫描单个文件，返回该文件中用到的 TF API 列表"""
    try:
        source = path.read_text(errors="ignore")
        tree = ast.parse(source)
    except Exception:
        return path, []

    apis = set()

    for node in ast.walk(tree):
        # 调用表达式
        if isinstance(node, ast.Call):
            f = node.func
            # 形态：tf.xxx.yyy
            if isinstance(f, ast.Attribute):
                full = []
                cur = f
                while isinstance(cur, ast.Attribute):
                    full.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    full.append(cur.id)
                    full = list(reversed(full))
                    if full[0] == "tf":
                        apis.add(".".join(full))

    return path, sorted(list(apis))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="framework/tensorflow-master")
    parser.add_argument("--out", default="data/tf_test_api_usage.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=500)
    args = parser.parse_args()

    root = Path(args.root)
    py_files = list(root.rglob("*.py"))
    print(f"[INFO] 共发现 Python 文件 {len(py_files)} 个")

    fout = open(args.out, "w", encoding="utf8")

    # ===== 多进程并发扫描 =====
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for i in range(0, len(py_files), args.batch):
            batch = py_files[i:i + args.batch]
            for p in batch:
                futures[ex.submit(extract_apis_from_file, p)] = p

            for fut in tqdm(as_completed(futures), total=len(batch), desc=f"Batch {i//args.batch}"):
                path, apis = fut.result()
                rec = {
                    "file": str(path),
                    "apis": apis
                }
                fout.write(json.dumps(rec) + "\n")

            futures.clear()

    fout.close()
    print(f"[DONE] 输出写入 {args.out}")


if __name__ == "__main__":
    main()
