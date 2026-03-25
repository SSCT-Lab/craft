# ./component/fuzz_run_numeric.py
import importlib.util
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

RTOL = 1e-3
ATOL = 3e-5

TEST_DIR = Path("fuzz_tests")
OUT = Path("data/fuzz_numeric.jsonl")


def safe_run(func):
    """运行 run_pt / run_tf，捕获异常"""
    try:
        out = func()
        return out, None
    except Exception as e:
        return None, str(e)


def to_numpy(x):
    import torch
    import tensorflow as tf

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    if isinstance(x, tf.Tensor):
        return x.numpy()

    return np.array(x)


def compare(a, b, rtol=RTOL, atol=ATOL):
    """数值一致性比较"""
    a = to_numpy(a)
    b = to_numpy(b)

    if a.shape != b.shape:
        return False, float("inf"), float("inf")

    diff = np.abs(a - b)
    max_abs = float(diff.max())
    denom = np.maximum(np.abs(b), np.finfo(a.dtype).eps)
    max_rel = float((diff / denom).max())
    ok = np.all(diff <= (atol + rtol * np.abs(b)))

    return ok, max_abs, max_rel


def load_test_module(path):
    """动态加载 fuzz 测试模块"""
    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    results = []
    tests = sorted(TEST_DIR.glob("*.py"))

    fout = open(OUT, "w")

    for t in tqdm(tests, desc="Numeric fuzzing"):
        mod = load_test_module(t)

        if not hasattr(mod, "run_pt") or not hasattr(mod, "run_tf"):
            # 如果测试文件中不包含 run_pt/run_tf，则跳过
            continue

        pt_out, pt_err = safe_run(mod.run_pt)
        tf_out, tf_err = safe_run(mod.run_tf)

        if pt_err or tf_err:
            fout.write(json.dumps({
                "file": t.name,
                "status": "runtime_error",
                "pt_error": pt_err,
                "tf_error": tf_err
            }) + "\n")
            continue

        ok, max_abs, max_rel = compare(pt_out, tf_out)

        fout.write(json.dumps({
            "file": t.name,
            "status": "pass" if ok else "value_mismatch",
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "allclose": ok
        }) + "\n")

        fout.flush()

    fout.close()
    print(f"[DONE] numeric results saved to {OUT}")


if __name__ == "__main__":
    main()
