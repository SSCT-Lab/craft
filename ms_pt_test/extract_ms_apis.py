#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1: 从 MindSpore 官方测试文件中提取被测试的 MS API 列表

功能：
- 扫描 testcases_ms/ 目录下的测试文件
- 通过正则表达式 + AST 分析提取 MindSpore 公开 API 名称
- 去重并输出结构化的 API 列表（JSON 格式）

用法：
    conda activate tf_env
    python ms_pt_test/extract_ms_apis.py [--ms-dir testcases_ms] [--output ms_pt_test/data/ms_apis.json]

输出：ms_pt_test/data/ms_apis.json
"""

import os
import sys
import io
import ast

# Windows 环境下强制使用 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import json
import re
import argparse
from datetime import datetime
from typing import List, Dict, Set, Any, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== 排除的 API 名称 ====================
# 这些是常用的辅助API，不是被测试的主要算子
HELPER_API_NAMES: Set[str] = {
    # 通用工具
    "Cell", "Parameter", "ParameterTuple", "Tensor",
    # 梯度相关
    "GradOperation", "grad", "value_and_grad",
    # 上下文
    "set_context", "get_context",
    # 训练相关
    "TrainOneStepCell", "WithLossCell", "WithGradCell",
    # 数据类型
    "float16", "float32", "float64", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "bool_", "complex64", "complex128",
}

# 常见辅助算子（辅助数据准备或测试结构，非核心测试对象）
AUXILIARY_OPS: Set[str] = {
    "Reshape", "Shape", "Fill", "OnesLike", "ZerosLike",
    "StopGradient", "depend", "TupleGetItem",
}

# 排除的文件模式
EXCLUDE_FILE_PATTERNS = [
    "__init__.py", "__pycache__", "conftest.py",
    "test_cpu_type.py",  # 非算子测试
    "test_primitive_cache.py",  # 非算子测试
    "test_rl_buffer",  # 强化学习 buffer，非算子
    "test_priority_replay_buffer",  # 强化学习 buffer，非算子
    "test_onednn_dfx.py",  # ONEDNN 诊断，非算子
]


def should_skip_file(filename: str) -> bool:
    """判断是否应跳过某个文件"""
    for pattern in EXCLUDE_FILE_PATTERNS:
        if pattern in filename:
            return True
    return False


def classify_api(api_name: str) -> str:
    """
    对 API 进行分类

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
    从单个 MindSpore 测试文件中提取 API 名称

    策略：
    1. 检测 import 语句，确定别名（P, ops, F, nn 等）
    2. 根据别名搜索使用模式
    3. 从 direct import 中提取
    4. 文件名推断作为补充

    Returns:
        提取到的 API 列表
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return []

    apis: Dict[str, Dict[str, Any]] = {}  # api_name -> info

    # ==================== 1. 检测 import 模式 ====================
    # 检测 P 别名 (mindspore.ops.operations)
    has_P = bool(re.search(
        r'(?:import\s+mindspore\.ops\.operations\s+as\s+P|'
        r'from\s+mindspore\.ops\s+import\s+operations\s+as\s+P|'
        r'from\s+mindspore\.ops\.operations\s+import\s)',
        content
    ))

    # 检测 ops 别名
    has_ops = bool(re.search(
        r'(?:import\s+mindspore\.ops\s+as\s+ops|'
        r'from\s+mindspore\s+import\s+ops)',
        content
    ))

    # 检测 F 别名 (functional)
    has_F = bool(re.search(
        r'(?:import\s+mindspore\.ops\.functional\s+as\s+F|'
        r'from\s+mindspore\.ops\s+import\s+functional\s+as\s+F)',
        content
    ))

    # 检测 nn 别名
    has_nn = bool(re.search(
        r'(?:import\s+mindspore\.nn\s+as\s+nn|'
        r'from\s+mindspore\s+import\s+nn)',
        content
    ))

    filename = os.path.basename(filepath)

    # ==================== 2. 从 P.Xxx() 提取 ops 类API ====================
    if has_P:
        # P.Xxx( 或 P.Xxx() — 捕获类名
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

    # ==================== 3. 从 ops.Xxx() 提取 ====================
    if has_ops:
        # ops.Xxx( — 大写开头为类，小写开头为函数
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
            # 排除常见的非算子方法
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

    # ==================== 4. 从 F.xxx() 提取 functional API ====================
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

    # ==================== 5. 从 nn.Xxx() 提取 NN 层 ====================
    if has_nn:
        matches = re.findall(r'\bnn\.([A-Z]\w*)\s*\(', content)
        for name in set(matches):
            if name in HELPER_API_NAMES:
                continue
            # 排除常见的非算子 NN 类
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

    # ==================== 6. 从 direct import 提取 ====================
    # from mindspore.ops.operations import Abs, Add, ...
    direct_imports = re.findall(
        r'from\s+mindspore\.ops\.operations(?:\.\w+)*\s+import\s+([^\n]+)',
        content
    )
    for import_line in direct_imports:
        # 处理多个导入名称
        names = [n.strip().split(" as ")[0].strip() for n in import_line.split(",")]
        for name in names:
            name = name.strip()
            if not name or not name[0].isupper():
                continue
            if name in HELPER_API_NAMES or name in AUXILIARY_OPS:
                continue
            # 验证该名称在代码中被实际使用
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

    # ==================== 7. 提取 Tensor 方法 API ====================
    # 寻找明确的 Tensor 方法调用模式
    # 如 input_x.add(input_y), tensor.abs(), x.matmul(y)
    tensor_method_patterns = re.findall(
        r'(?:input_\w+|tensor\w*|x\d*|output\w*|result\w*)\.'
        r'([a-z]\w*)\s*\(',
        content, re.IGNORECASE
    )
    # 只保留已知的 Tensor API 方法名
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
    """提取测试函数名（AST 分析）"""
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
    从文件名推断可能的 API 名称（启发式兜底）

    test_abs_op.py → mindspore.ops.Abs
    test_relu6_op.py → mindspore.ops.ReLU6
    test_batch_matmul.py → mindspore.ops.BatchMatMul
    """
    name = filename
    # 去掉前缀和后缀
    if name.startswith("test_"):
        name = name[5:]
    for suffix in ["_op.py", "_ops.py", ".py"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    if not name:
        return None

    # 转换为 CamelCase
    parts = name.split("_")
    camel_name = "".join(p.capitalize() for p in parts if p)

    return f"mindspore.ops.{camel_name}"


def extract_all_apis(ms_dir: str) -> List[Dict[str, Any]]:
    """
    从 testcases_ms 目录提取所有 MindSpore API

    Args:
        ms_dir: testcases_ms 目录路径

    Returns:
        API 列表
    """
    if not os.path.isdir(ms_dir):
        print(f"❌ 目录不存在: {ms_dir}")
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
                # 如果已存在，保留第一次发现的（通常更准确）
                if api_name not in all_apis:
                    # 补充测试函数信息
                    test_names = extract_test_info(filepath)
                    api_info["test_functions"] = test_names[:5]  # 最多保存5个
                    all_apis[api_name] = api_info
            print(f"  ✅ {filename} → {len(apis)} 个API")
        else:
            # 尝试从文件名推断
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
                print(f"  🔍 {filename} → {inferred}（文件名推断）")
            else:
                print(f"  ⏭️ {filename} → 未识别到API")

    print(f"\n📊 文件统计: 处理 {files_processed} 个文件，{files_with_apis} 个文件提取到API")

    # 转为列表并排序
    result = sorted(all_apis.values(), key=lambda x: x["ms_api"])
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: 从 MindSpore 官方测试文件中提取被测试的 MS API 列表"
    )
    parser.add_argument(
        "--ms-dir", default=os.path.join(ROOT_DIR, "testcases_ms"),
        help="MindSpore 测试文件目录路径（默认: 项目根目录/testcases_ms）"
    )
    parser.add_argument(
        "--output", "-o", default=os.path.join(ROOT_DIR, "ms_pt_test", "data", "ms_apis.json"),
        help="输出的 JSON 文件路径"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Step 1: 从 MindSpore 官方测试文件中提取 MS API 列表")
    print("=" * 80)
    print(f"📁 测试文件目录: {args.ms_dir}")
    print(f"📁 输出文件: {args.output}")

    if not os.path.isdir(args.ms_dir):
        print(f"❌ 目录不存在: {args.ms_dir}")
        sys.exit(1)

    # 提取 API 列表
    print(f"\n📂 开始扫描目录: {args.ms_dir}\n")
    apis = extract_all_apis(args.ms_dir)

    # 统计
    type_counts: Dict[str, int] = {}
    method_counts: Dict[str, int] = {}
    for api in apis:
        api_type = api.get("api_type", "unknown")
        type_counts[api_type] = type_counts.get(api_type, 0) + 1
        method = api.get("extraction_method", "unknown")
        method_counts[method] = method_counts.get(method, 0) + 1

    print(f"\n{'=' * 80}")
    print(f"📊 提取结果汇总")
    print(f"{'=' * 80}")
    print(f"  总API数量: {len(apis)}")
    print(f"\n  按类型分布:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c} 个")
    print(f"\n  按提取方法分布:")
    for m, c in sorted(method_counts.items()):
        print(f"    {m}: {c} 个")

    # 保存结果
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

    print(f"\n💾 已保存到: {args.output}")
    print(f"📋 前10个API: {', '.join(a['ms_api'] for a in apis[:10])}")


if __name__ == "__main__":
    main()
