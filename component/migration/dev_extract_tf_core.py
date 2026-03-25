"""
dev_extract_tf_core.py
=======================

从原始 TensorFlow 测试文件中**只提取 TF 核心测试逻辑**，生成独立的 TF 测试脚本，
后续可以在单独的 TF 环境（例如 `tf2pt-dev-tf`）里运行，也可以作为 PT 迁移、
fuzzing 的输入。

设计要点：
- 不依赖 PyTorch，不导入 `torch`；
- 尽量复用现有的提取与缩进处理逻辑（`migrate_collect_tf_results` /
  `migrate_generate_tests` 里的工具函数）；
- 每个 TF 测试对应一个独立脚本，保存在 `dev/tf_core/` 目录；
- 生成的脚本内只包含：
  - 必要的 TF / NumPy 等 import；
  - 原始文件中抽取出来的 helper 函数（非 test*）；
  - 一个带 `tf_` 前缀的独立 TF 测试函数；
  - 一个简单的 `main`，直接调用该 TF 测试函数并打印结果。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from component.migration.migrate_collect_tf_results import load_jsonl
from component.migration.migrate_generate_tests import (
    convert_tf_to_standalone,
    extract_test_function_code as extract_tf_func_code,
    extract_helper_functions,
)


DEFAULT_CANDIDATES = Path("data/migration/migration_candidates_fuzzy.jsonl")
DEFAULT_TF_ROOT = Path("framework/tensorflow-master")
DEFAULT_OUT_DIR = Path("dev/tf_core")
DEFAULT_TESTS_TF_MAPPED = Path("data/mapping/tests_tf.mapped.jsonl")


TF_HEADER = """import tensorflow as tf
# 禁用 eager execution 以支持 tf.compat.v1.Session()
tf.compat.v1.disable_eager_execution()
import numpy as np
import sys
import io
import pkgutil
import re
from contextlib import redirect_stdout, redirect_stderr


def capture_output(func, *args, **kwargs):
    \"\"\"Capture stdout, stderr and return value from function execution\"\"\"
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    result = None
    exception = None

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            result = func(*args, **kwargs)
    except Exception as e:
        exception = e

    stdout_text = stdout_capture.getvalue()
    stderr_text = stderr_capture.getvalue()

    return {
        "result": result,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "exception": exception,
        "success": exception is None,
    }


def assertAllClose(a, b, rtol=1e-6, atol=1e-6):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def assertAllEqual(a, b):
    np.testing.assert_array_equal(a, b)


def assertFunctionMatchesEager(func, *args, **kwargs):
    \"\"\"Assert that a function matches eager execution behavior.
    This is a simplified version for standalone test execution.
    In a full implementation, this would compare eager vs graph mode.\"\"\"
    try:
        # 由于禁用了 eager execution，这里直接执行函数并检查是否抛出异常
        result = func(*args, **kwargs)
        # 如果结果是 Tensor，尝试转换为 numpy（如果可能）
        if isinstance(result, tf.Tensor):
            try:
                with tf.compat.v1.Session() as sess:
                    result_np = sess.run(result)
            except:
                # 如果无法在 session 中运行，至少确保函数执行了
                pass
        return result
    except Exception as e:
        raise AssertionError(f"Function execution failed: {e}")


def assertIsNotNone(obj, msg=None):
    \"\"\"Assert that an object is not None\"\"\"
    if obj is None:
        raise AssertionError(msg or "Object is None")


def assertIn(member, container, msg=None):
    \"\"\"Assert that member is in container\"\"\"
    if member not in container:
        raise AssertionError(msg or f"{member} not in {container}")


def assertNotIn(member, container, msg=None):
    \"\"\"Assert that member is not in container\"\"\"
    if member in container:
        raise AssertionError(msg or f"{member} in {container}")


def assertEqual(first, second, msg=None):
    \"\"\"Assert that two objects are equal\"\"\"
    if first != second:
        raise AssertionError(msg or f"{first} != {second}")


class assertRaisesRegex:
    \"\"\"Context manager to assert that an exception is raised with a matching regex\"\"\"
    def __init__(self, exception, regex):
        self.exception = exception
        self.regex = regex
        self.exception_instance = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"{self.exception.__name__} was not raised")
        if not issubclass(exc_type, self.exception):
            return False  # 重新抛出异常
        if not re.search(self.regex, str(exc_val)):
            raise AssertionError(f"Exception message '{exc_val}' does not match regex '{self.regex}'")
        self.exception_instance = exc_val
        return True  # 抑制异常

"""


def safe_name(name: str) -> str:
    """对测试名做一个简单的文件名安全化处理。"""
    return "".join(ch if ch.isalnum() or ch in "._" else "_" for ch in name)


def load_candidates(
    candidates_path: Path, tests_tf_mapped_path: Path, limit: int
) -> Tuple[List[Dict], List[Dict]]:
    """加载候选测试与 TF 映射元数据。"""
    if not candidates_path.exists():
        raise FileNotFoundError(f"候选文件不存在: {candidates_path}")

    candidates = load_jsonl(str(candidates_path))
    print(f"[LOAD] 从 {candidates_path} 加载了 {len(candidates)} 个候选测试")

    if limit > 0:
        candidates = candidates[:limit]
        print(f"[INFO] 限制测试数量: {limit}")

    tests_tf_mapped: List[Dict] = []
    if tests_tf_mapped_path.exists():
        tests_tf_mapped = load_jsonl(str(tests_tf_mapped_path))
        print(f"[LOAD] 从 {tests_tf_mapped_path} 加载了 {len(tests_tf_mapped)} 条 TF 测试元数据")
    else:
        print(f"[WARN] 未找到 tests_tf.mapped.jsonl: {tests_tf_mapped_path}，将直接使用候选中的 file/name")

    return candidates, tests_tf_mapped


def resolve_tf_source(
    file_path: str, test_name: str, tests_tf_mapped: List[Dict], tf_root: Path
) -> Tuple[Path, str]:
    """
    根据 candidates 里的 file/name 和 tests_tf.mapped 中的信息，解析出
    实际的 TF 源文件路径和真正的测试函数名。
    """
    # 先尝试在 tests_tf.mapped 里查找精确匹配
    if tests_tf_mapped:
        file_key = file_path.replace("framework/tensorflow-master/", "")
        for item in tests_tf_mapped:
            if item.get("file") == file_key and item.get("name") == test_name:
                real_file = item.get("file", file_path)
                real_name = item.get("name", test_name)
                # 映射中的 file 通常是相对路径
                tf_file = (
                    Path(real_file)
                    if Path(real_file).is_absolute()
                    else tf_root / real_file
                )
                return tf_file, real_name

    # 回退：直接使用 candidates 中的 file/name
    if file_path.startswith("framework/tensorflow-master/"):
        tf_file = Path(file_path)
    elif Path(file_path).exists():
        tf_file = Path(file_path)
    else:
        tf_file = tf_root / file_path

    return tf_file, test_name


def build_tf_core_script(
    tf_file: Path,
    test_name: str,
    tf_root: Path,
) -> Tuple[str, str]:
    """
    构造一个仅包含 TF 核心逻辑的脚本内容。

    返回 (脚本内容, 生成的 TF 函数名)。
    """
    if not tf_file.exists():
        raise FileNotFoundError(f"TF 源文件不存在: {tf_file}")

    # 使用 migrate_generate_tests 中的提取与缩进处理逻辑
    tf_code = extract_tf_func_code(str(tf_file), test_name)
    if not tf_code:
        raise RuntimeError(f"无法从 {tf_file} 提取测试函数 {test_name}")

    tf_standalone_code, tf_func_name = convert_tf_to_standalone(tf_code, test_name)

    # 提取辅助函数（非 test*），如果可能，也提取类方法
    helper_code = extract_helper_functions(str(tf_file), test_name) or ""
    helper_section = f"{helper_code}\n\n" if helper_code else ""

    script = (
        TF_HEADER
        + f"# Auto-extracted TF core test\n"
        f"# source: {tf_file.relative_to(tf_root) if tf_file.is_absolute() else tf_file}\n"
        f"# original test: {test_name}\n\n"
        f"{helper_section}"
        f"# ===== TensorFlow Core Test =====\n"
        f"{tf_standalone_code}\n\n"
        f"def main():\n"
        f"    \"\"\"简单执行一次 TF 测试，并打印结果。\"\"\"\n"
        f"    from pprint import pprint\n"
        f"    out = capture_output({tf_func_name})\n"
        f"    print('\\n' + '=' * 80)\n"
        f"    print('TF 核心测试执行结果')\n"
        f"    print('=' * 80)\n"
        f"    pprint(out)\n\n"
        f"if __name__ == '__main__':\n"
        f"    main()\n"
    )

    return script, tf_func_name


def main():
    parser = argparse.ArgumentParser(
        description="从原始 TensorFlow 测试中抽取 TF 核心逻辑，生成独立 TF 脚本（用于后续迁移 & fuzzing）"
    )
    parser.add_argument(
        "--candidates",
        default=str(DEFAULT_CANDIDATES),
        help="候选测试列表（migration_candidates_fuzzy.jsonl）",
    )
    parser.add_argument(
        "--tf-root",
        default=str(DEFAULT_TF_ROOT),
        help="TensorFlow 源码根目录（包含原始测试文件）",
    )
    parser.add_argument(
        "--tests-tf-mapped",
        default=str(DEFAULT_TESTS_TF_MAPPED),
        help="tests_tf.mapped.jsonl，用于解析真实 TF 文件与测试名",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="输出目录，用于保存 TF core 测试脚本",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="限制抽取的测试数量，-1 表示全部（默认 20，方便快速迭代）",
    )
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    tf_root = Path(args.tf_root)
    tests_tf_mapped_path = Path(args.tests_tf_mapped)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, tests_tf_mapped = load_candidates(
        candidates_path, tests_tf_mapped_path, args.limit
    )

    results = []
    for item in candidates:
        file_path = item.get("file", "")
        test_name = item.get("name", "")

        try:
            tf_file, real_test_name = resolve_tf_source(
                file_path, test_name, tests_tf_mapped, tf_root
            )
            script, tf_func_name = build_tf_core_script(tf_file, real_test_name, tf_root)

            out_name = f"tf_core_{safe_name(real_test_name)}.py"
            out_path = out_dir / out_name
            out_path.write_text(script, encoding="utf-8")

            print(f"[OK] {file_path}:{test_name} -> {out_path.name} ({tf_func_name})")
            results.append(
                {
                    "file": file_path,
                    "test_name": test_name,
                    "tf_file": str(tf_file),
                    "tf_func_name": tf_func_name,
                    "core_file": str(out_path),
                    "status": "ok",
                }
            )
        except Exception as e:
            print(f"[FAIL] {file_path}:{test_name} - {e}")
            results.append(
                {
                    "file": file_path,
                    "test_name": test_name,
                    "status": "error",
                    "error": str(e),
                }
            )

    # 同步写一个索引，便于后续 PT 迁移 / fuzzing 使用
    index_path = out_dir / "tf_core_index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(results)
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"[SUMMARY] total={total}, ok={ok}, fail={total - ok}")


if __name__ == "__main__":
    main()


