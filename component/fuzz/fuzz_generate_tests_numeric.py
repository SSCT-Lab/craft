# ./component/fuzz_generate_tests_numeric.py
import json
import random
from pathlib import Path

SEED_FILE = Path("data/seed_patterns.jsonl")
OUT_DIR = Path("fuzz_tests")
OUT_DIR.mkdir(exist_ok=True)


def mutate_value(v):
    if isinstance(v, int):
        return max(1, v + random.randint(-v//5 - 1, v//5 + 1))
    if isinstance(v, float):
        return v * (1 + random.uniform(-0.3, 0.3))
    if isinstance(v, tuple):
        return tuple(mutate_value(x) for x in v)
    return v


def generate_numeric_test(api, args, idx):
    mut_args = [mutate_value(a) for a in args]

    code = f"""
import torch
import tensorflow as tf

def run_pt():
    return {api}({", ".join(map(str, mut_args))})

def run_tf():
    return {api.replace("torch.", "tf.")}({", ".join(map(str, mut_args))})
"""
    return code


def main():
    patterns = [json.loads(x) for x in open(SEED_FILE)]
    idx = 0

    for p in patterns:
        for _ in range(3):
            code = generate_numeric_test(p["api"], p["args"], idx)
            (OUT_DIR / f"test_fuzz_{idx}.py").write_text(code)
            idx += 1

    print(f"[DONE] generated {idx} numeric fuzz tests -> {OUT_DIR}")


if __name__ == "__main__":
    main()
