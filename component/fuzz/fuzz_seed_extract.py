# ./component/fuzz_seed_extract.py
import ast
from pathlib import Path
import json

MIG_DIR = Path("migrated_tests")
OUT = Path("data/seed_patterns.jsonl")

def extract_patterns_from_file(path):
    text = path.read_text()
    tree = ast.parse(text)
    patterns = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # 抽取 API 名称
            try:
                if isinstance(node.func, ast.Attribute):
                    api = f"{ast.unparse(node.func)}"
                else:
                    api = ast.unparse(node.func)
            except:
                continue

            # 抽取参数 literal 值
            args = []
            for a in node.args:
                if isinstance(a, ast.Constant):
                    args.append(a.value)
                elif isinstance(a, ast.Tuple):
                    try:
                        args.append(tuple([x.value for x in a.elts if isinstance(x, ast.Constant)]))
                    except:
                        pass

            patterns.append({
                "api": api,
                "args": args,
                "file": str(path)
            })

    return patterns


def main():
    out = open(OUT, "w")
    for f in MIG_DIR.glob("*.py"):
        patterns = extract_patterns_from_file(f)
        for p in patterns:
            out.write(json.dumps(p) + "\n")
    out.close()

    print(f"[DONE] seed patterns saved to {OUT}")

if __name__ == "__main__":
    main()
