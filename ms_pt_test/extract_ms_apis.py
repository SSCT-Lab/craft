#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1: Extract the tested MS API list from MindSpore official test files

Purpose:
- Scan test files under testcases_ms/
- Use regex + AST to extract public MindSpore API names
- Deduplicate and output a structured API list (JSON)

Usage:
    conda activate tf_env
    python ms_pt_test/extract_ms_apis.py [--ms-dir testcases_ms] [--output ms_pt_test/data/ms_apis.json]

Output: ms_pt_test/data/ms_apis.json
"""

import os
import sys
import io
import ast

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import json
import re
import argparse
from datetime import datetime
from typing import List, Dict, Set, Any, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== Excluded API names ====================
# Common helper APIs, not primary tested operators
HELPER_API_NAMES: Set[str] = {
    # General utilities
    "Cell", "Parameter", "ParameterTuple", "Tensor",
    # Gradient-related
    "GradOperation", "grad", "value_and_grad",
    # Context
    "set_context", "get_context",
    # Training
    "TrainOneStepCell", "WithLossCell", "WithGradCell",
    # Dtypes
    "float16", "float32", "float64", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "bool_", "complex64", "complex128",
}

# Common helper ops (data prep/testing scaffolding, not core test targets)
AUXILIARY_OPS: Set[str] = {
    "Reshape", "Shape", "Fill", "OnesLike", "ZerosLike",
    "StopGradient", "depend", "TupleGetItem",
}

# Excluded file patterns
EXCLUDE_FILE_PATTERNS = [
    "__init__.py", "__pycache__", "conftest.py",
    "test_cpu_type.py",  # Non-operator test
    "test_primitive_cache.py",  # Non-operator test
    "test_rl_buffer",  # RL buffer, not an operator
    "test_priority_replay_buffer",  # RL buffer, not an operator
    "test_onednn_dfx.py",  # ONEDNN diagnostics, not an operator
]


def should_skip_file(filename: str) -> bool:
    """Check whether a file should be skipped."""
    for pattern in EXCLUDE_FILE_PATTERNS:
        if pattern in filename:
            return True
    return False


def classify_api(api_name: str) -> str:
    """
    Classify an API.

    Returns:
        "ops" | "nn" | "functional" | "tensor_method"
    """
    if api_name.startswith("mindspore.nn."):
        return "nn"
    elif api_name.startswith("mindspore.Tensor."):
        return "tensor_method"
    else:
        # mindspore.ops.xxx
        parts = api_name.split(".")
        if len(parts) >= 3:
            last_part = parts[-1]
            if last_part and last_part[0].islower():
                return "functional"
        return "ops"


def extract_apis_from_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Extract API names from a single MindSpore test file.

    Strategy:
    1. Detect import statements and aliases (P, ops, F, nn, etc.)
    2. Search usage patterns by alias
    3. Extract from direct imports
    4. Use filename inference as a supplement

    Returns:
        Extracted API list
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return []

    apis: Dict[str, Dict[str, Any]] = {}  # api_name -> info

    # ==================== 1. Detect import patterns ====================
    # Detect P alias (mindspore.ops.operations)
    has_P = bool(re.search(
        r'(?:import\s+mindspore\.ops\.operations\s+as\s+P|'
        r'from\s+mindspore\.ops\s+import\s+operations\s+as\s+P|'
        r'from\s+mindspore\.ops\.operations\s+import\s)',
        content
    ))

    # Detect ops alias
    has_ops = bool(re.search(
        r'(?:import\s+mindspore\.ops\s+as\s+ops|'
        r'from\s+mindspore\s+import\s+ops)',
        content
    ))

    # Detect F alias (functional)
    has_F = bool(re.search(
        r'(?:import\s+mindspore\.ops\.functional\s+as\s+F|'
        r'from\s+mindspore\.ops\s+import\s+functional\s+as\s+F)',
        content
    ))

    # Detect nn alias
    has_nn = bool(re.search(
        r'(?:import\s+mindspore\.nn\s+as\s+nn|'
        r'from\s+mindspore\s+import\s+nn)',
        content
    ))

    filename = os.path.basename(filepath)

    # ==================== 2. Extract ops class APIs from P.Xxx() ====================
    if has_P:
        # P.Xxx( or P.Xxx() - capture class name
        matches = re.findall(r'\bP\.([A-Z]\w*)\s*\(', content)
        for name in set(matches):
            if name in HELPER_API_NAMES or name in AUXILIARY_OPS:
                continue
            api_name = f"mindspore.ops.{name}"
            if api_name not in apis:
                apis[api_name] = {
                    "ms_api": api_name,
                    "source_file": filename,
                    "extraction_method": "regex_P",
                    "api_type": "ops",
                }

    # ==================== 3. Extract from ops.Xxx() ====================
    if has_ops:
        # ops.Xxx( - uppercase for class, lowercase for function
        matches_class = re.findall(r'\bops\.([A-Z]\w*)\s*\(', content)
        for name in set(matches_class):
            if name in HELPER_API_NAMES or name in AUXILIARY_OPS:
                continue
            api_name = f"mindspore.ops.{name}"
            if api_name not in apis:
                apis[api_name] = {
                    "ms_api": api_name,
                    "source_file": filename,
                    "extraction_method": "regex_ops_class",
                    "api_type": "ops",
                }

        matches_func = re.findall(r'\bops\.([a-z]\w*)\s*\(', content)
        for name in set(matches_func):
            if name in HELPER_API_NAMES or name in {"operations", "functional"}:
                continue
            # Exclude common non-operator methods
            if name in {"depend", "composite", "prim_attr_register", "stop_gradient"}:
                continue
            api_name = f"mindspore.ops.{name}"
            if api_name not in apis:
                apis[api_name] = {
                    "ms_api": api_name,
                    "source_file": filename,
                    "extraction_method": "regex_ops_func",
                    "api_type": "functional",
                }

    # ==================== 4. Extract functional APIs from F.xxx() ====================
    if has_F:
        matches = re.findall(r'\bF\.([a-z_]\w*)\s*\(', content)
        for name in set(matches):
            if name in HELPER_API_NAMES:
                continue
            if name in {"depend", "stop_gradient", "typeof", "shape", "rank", "dtype"}:
                continue
            api_name = f"mindspore.ops.{name}"
            if api_name not in apis:
                apis[api_name] = {
                    "ms_api": api_name,
                    "source_file": filename,
                    "extraction_method": "regex_F",
                    "api_type": "functional",
                }

    # ==================== 5. Extract NN layers from nn.Xxx() ====================
    if has_nn:
        matches = re.findall(r'\bnn\.([A-Z]\w*)\s*\(', content)
        for name in set(matches):
            if name in HELPER_API_NAMES:
                continue
            # Exclude common non-operator NN classes
            if name in {"Cell", "CellList", "SequentialCell", "TrainOneStepCell",
                         "WithLossCell", "WithGradCell"}:
                continue
            api_name = f"mindspore.nn.{name}"
            if api_name not in apis:
                apis[api_name] = {
                    "ms_api": api_name,
                    "source_file": filename,
                    "extraction_method": "regex_nn",
                    "api_type": "nn",
                }

    # ==================== 6. Extract from direct imports ====================
    # from mindspore.ops.operations import Abs, Add, ...
    direct_imports = re.findall(
        r'from\s+mindspore\.ops\.operations(?:\.\w+)*\s+import\s+([^\n]+)',
        content
    )
    for import_line in direct_imports:
        # Handle multiple imported names
        names = [n.strip().split(" as ")[0].strip() for n in import_line.split(",")]
        for name in names:
            name = name.strip()
            if not name or not name[0].isupper():
                continue
            if name in HELPER_API_NAMES or name in AUXILIARY_OPS:
                continue
            # Ensure the name is actually used in code
            if re.search(rf'\b{re.escape(name)}\s*\(', content):
                api_name = f"mindspore.ops.{name}"
                if api_name not in apis:
                    apis[api_name] = {
                        "ms_api": api_name,
                        "source_file": filename,
                        "extraction_method": "direct_import",
                        "api_type": "ops",
                    }

    # from mindspore.ops import Xxx
    ops_imports = re.findall(
        r'from\s+mindspore\.ops\s+import\s+([^\n]+)',
        content
    )
    for import_line in ops_imports:
        names = [n.strip().split(" as ")[0].strip() for n in import_line.split(",")]
        for name in names:
            name = name.strip()
            if not name or name in {"operations", "functional", "composite",
                                     "prim_attr_register", "PrimitiveWithInfer"}:
                continue
            if name in HELPER_API_NAMES:
                continue
            if name[0].isupper() and re.search(rf'\b{re.escape(name)}\s*\(', content):
                api_name = f"mindspore.ops.{name}"
                if api_name not in apis:
                    apis[api_name] = {
                        "ms_api": api_name,
                        "source_file": filename,
                        "extraction_method": "direct_import_ops",
                        "api_type": "ops",
                    }

    # ==================== 7. Extract Tensor method APIs ====================
    # Look for explicit Tensor method call patterns
    # e.g., input_x.add(input_y), tensor.abs(), x.matmul(y)
    tensor_method_patterns = re.findall(
        r'(?:input_\w+|tensor\w*|x\d*|output\w*|result\w*)\.'
        r'([a-z]\w*)\s*\(',
        content, re.IGNORECASE
    )
    # Keep only known Tensor API method names
    KNOWN_TENSOR_METHODS = {
        "add", "sub", "mul", "div", "matmul", "abs", "neg", "pow", "exp",
        "log", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
        "sigmoid", "tanh", "relu", "softmax", "argmax", "argmin",
        "sum", "mean", "max", "min", "prod", "all", "any",
        "reshape", "transpose", "flatten", "squeeze", "unsqueeze",
        "expand", "permute", "contiguous", "clamp", "clip",
        "cross", "bmm", "addmm", "addmv", "addr",
        "bitwise_and", "bitwise_or", "bitwise_xor",
        "cumsum", "cumprod", "sort", "topk", "gather",
        "scatter", "index_add", "fill", "masked_fill",
    }
    for name in set(tensor_method_patterns):
        name_lower = name.lower()
        if name_lower in KNOWN_TENSOR_METHODS:
            api_name = f"mindspore.Tensor.{name_lower}"
            if api_name not in apis:
                apis[api_name] = {
                    "ms_api": api_name,
                    "source_file": filename,
                    "extraction_method": "tensor_method",
                    "api_type": "tensor_method",
                }

    return list(apis.values())


def extract_test_info(filepath: str) -> List[str]:
    """Extract test function names (AST analysis)."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            tree = ast.parse(f.read())
        test_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                test_names.append(node.name)
            elif isinstance(node, ast.ClassDef) and "Test" in node.name:
                test_names.append(node.name)
        return test_names
    except Exception:
        return []


def infer_api_from_filename(filename: str) -> Optional[str]:
    """
    Infer a possible API name from filename (heuristic fallback).

    test_abs_op.py -> mindspore.ops.Abs
    test_relu6_op.py -> mindspore.ops.ReLU6
    test_batch_matmul.py -> mindspore.ops.BatchMatMul
    """
    name = filename
    # Remove prefix and suffix
    if name.startswith("test_"):
        name = name[5:]
    for suffix in ["_op.py", "_ops.py", ".py"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    if not name:
        return None

    # Convert to CamelCase
    parts = name.split("_")
    camel_name = "".join(p.capitalize() for p in parts if p)

    return f"mindspore.ops.{camel_name}"


def extract_all_apis(ms_dir: str) -> List[Dict[str, Any]]:
    """
    Extract all MindSpore APIs from the testcases_ms directory.

    Args:
        ms_dir: testcases_ms directory path

    Returns:
        API list
    """
    if not os.path.isdir(ms_dir):
        print(f"❌ Directory does not exist: {ms_dir}")
        return []

    all_apis: Dict[str, Dict[str, Any]] = {}  # api_name -> info
    files_processed = 0
    files_with_apis = 0

    for filename in sorted(os.listdir(ms_dir)):
        if not filename.endswith(".py"):
            continue
        if should_skip_file(filename):
            continue

        filepath = os.path.join(ms_dir, filename)
        if not os.path.isfile(filepath):
            continue

        files_processed += 1
        apis = extract_apis_from_file(filepath)

        if apis:
            files_with_apis += 1
            for api_info in apis:
                api_name = api_info["ms_api"]
                # If already exists, keep the first occurrence (usually more accurate)
                if api_name not in all_apis:
                    # Add test function info
                    test_names = extract_test_info(filepath)
                    api_info["test_functions"] = test_names[:5]  # Save up to 5
                    all_apis[api_name] = api_info
            print(f"  ✅ {filename} -> {len(apis)} APIs")
        else:
            # Try to infer from filename
            inferred = infer_api_from_filename(filename)
            if inferred and inferred not in all_apis:
                test_names = extract_test_info(filepath)
                all_apis[inferred] = {
                    "ms_api": inferred,
                    "source_file": filename,
                    "extraction_method": "filename_inferred",
                    "api_type": "ops",
                    "test_functions": test_names[:5],
                }
                print(f"  🔍 {filename} -> {inferred} (inferred from filename)")
            else:
                print(f"  ⏭️ {filename} -> No API recognized")

    print(f"\n📊 File stats: processed {files_processed} files, extracted APIs from {files_with_apis} files")

    # Convert to list and sort
    result = sorted(all_apis.values(), key=lambda x: x["ms_api"])
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Extract tested MS APIs from MindSpore official tests"
    )
    parser.add_argument(
        "--ms-dir", default=os.path.join(ROOT_DIR, "testcases_ms"),
        help="MindSpore test file directory path (default: <repo>/testcases_ms)"
    )
    parser.add_argument(
        "--output", "-o", default=os.path.join(ROOT_DIR, "ms_pt_test", "data", "ms_apis.json"),
        help="Output JSON file path"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Step 1: Extract MS API list from MindSpore official tests")
    print("=" * 80)
    print(f"- Test file directory: {args.ms_dir}")
    print(f"- Output file: {args.output}")

    if not os.path.isdir(args.ms_dir):
        print(f"- Directory does not exist: {args.ms_dir}")
        sys.exit(1)

    # Extract API list
    print(f"\n- Start scanning directory: {args.ms_dir}\n")
    apis = extract_all_apis(args.ms_dir)

    # Summary
    type_counts: Dict[str, int] = {}
    method_counts: Dict[str, int] = {}
    for api in apis:
        api_type = api.get("api_type", "unknown")
        type_counts[api_type] = type_counts.get(api_type, 0) + 1
        method = api.get("extraction_method", "unknown")
        method_counts[method] = method_counts.get(method, 0) + 1

    print(f"\n{'=' * 80}")
    print(f"- Extraction summary")
    print(f"{'=' * 80}")
    print(f"  Total APIs: {len(apis)}")
    print(f"\n  By type:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")
    print(f"\n  By extraction method:")
    for m, c in sorted(method_counts.items()):
        print(f"    {m}: {c}")

    # Save results
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    output_data = {
        "total_apis": len(apis),
        "extraction_time": datetime.now().isoformat(),
        "source_dir": args.ms_dir,
        "type_distribution": type_counts,
        "method_distribution": method_counts,
        "apis": apis,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n- Saved to: {args.output}")
    print(f"- First 10 APIs: {', '.join(a['ms_api'] for a in apis[:10])}")


if __name__ == "__main__":
    main()
