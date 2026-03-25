# ./component/scan_components.py
import ast
import argparse
import json
import yaml
from pathlib import Path
from tqdm import tqdm

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def is_python_file(path, exclude_globs):
    if not path.name.endswith(".py"):
        return False
    p_str = str(path)
    for g in exclude_globs:
        if path.match(g) or p_str.endswith(g):
            return False
    return True

def extract_defs(path: Path, framework: str, repo_root: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []

    comps = []

    module_rel = path.relative_to(repo_root).as_posix()
    module_name = module_rel[:-3].replace("/", ".")

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            sig = f"{node.name}({', '.join([a.arg for a in node.args.args])})"
            api = f"{framework}.{module_name}.{node.name}"
            comps.append({
                "api": api,
                "kind": "function",
                "module": module_name,
                "file": module_rel,
                "sig": sig,
            })
        elif isinstance(node, ast.ClassDef):
            sig = f"class {node.name}"
            api = f"{framework}.{module_name}.{node.name}"
            comps.append({
                "api": api,
                "kind": "class",
                "module": module_name,
                "file": module_rel,
                "sig": sig,
            })

    return comps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--out-tf", type=str, default="data/tf_components.jsonl")
    parser.add_argument("--out-pt", type=str, default="data/pt_components.jsonl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tf_root = Path(cfg["repos"]["tf"])
    pt_root = Path(cfg["repos"]["pt"])
    exclude = cfg.get("exclude_globs", [])

    all_tf = []
    all_pt = []

    print("Scanning TensorFlow source...")
    for path in tqdm(tf_root.rglob("*.py"), unit="file"):
        if is_python_file(path, exclude):
            all_tf.extend(extract_defs(path, "tf", tf_root))

    print("Scanning PyTorch source...")
    for path in tqdm(pt_root.rglob("*.py"), unit="file"):
        if is_python_file(path, exclude):
            all_pt.extend(extract_defs(path, "pt", pt_root))

    Path("data").mkdir(exist_ok=True)

    with open(args.out_tf, "w") as f:
        for c in all_tf:
            f.write(json.dumps(c) + "\n")

    with open(args.out_pt, "w") as f:
        for c in all_pt:
            f.write(json.dumps(c) + "\n")

    print(f"[DONE] TF components={len(all_tf)}, PT components={len(all_pt)}")

if __name__ == "__main__":
    main()
