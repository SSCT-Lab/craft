#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1: 从 TensorFlow 官方测试文件中提取被测试的 TF API 列表

功能：
- 扫描 tf_testcases/ 目录下的测试文件目录（默认自动发现所有子目录）
- 通过预定义映射 + AST 辅助分析提取 TF 公开 API 名称
- 去重并输出结构化的 API 列表（JSON 格式）

用法：
    conda activate tf_env
    python tf_pt_test/extract_tf_apis.py [--tf-dir tf_testcases] [--output tf_pt_test/data/tf_apis_new.json]

输出：tf_pt_test/data/tf_apis_new.json
"""

import os
import sys
import ast
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Any, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== 预定义的测试文件 → TF API 映射 ====================
# 聚焦于核心计算类 API（nn_ops, math_ops, array_ops, linalg, signal, image_ops）
# 每个条目格式: "relative_path": ["tf.api1", "tf.api2", ...]
KNOWN_FILE_APIS: Dict[str, List[str]] = {
    # ==================== nn_ops ====================
    "nn_ops/relu_op_test.py": [
        "tf.nn.relu", "tf.nn.relu6", "tf.nn.leaky_relu", "tf.nn.elu", "tf.nn.selu",
    ],
    "nn_ops/softmax_op_test.py": ["tf.nn.softmax", "tf.nn.log_softmax"],
    "nn_ops/conv1d_test.py": ["tf.nn.conv1d"],
    "nn_ops/conv_ops_test.py": ["tf.nn.conv2d"],
    "nn_ops/conv_ops_3d_test.py": ["tf.nn.conv3d"],
    "nn_ops/conv1d_transpose_test.py": ["tf.nn.conv1d_transpose"],
    "nn_ops/conv2d_transpose_test.py": ["tf.nn.conv2d_transpose"],
    "nn_ops/conv3d_transpose_test.py": ["tf.nn.conv3d_transpose"],
    "nn_ops/pooling_ops_test.py": ["tf.nn.max_pool", "tf.nn.avg_pool"],
    "nn_ops/pooling_ops_3d_test.py": ["tf.nn.max_pool3d", "tf.nn.avg_pool3d"],
    "nn_ops/pool_test.py": ["tf.nn.pool"],
    "nn_ops/embedding_ops_test.py": ["tf.nn.embedding_lookup"],
    "nn_ops/softplus_op_test.py": ["tf.math.softplus"],
    "nn_ops/softsign_op_test.py": ["tf.nn.softsign"],
    "nn_ops/bias_op_test.py": ["tf.nn.bias_add"],
    "nn_ops/lrn_op_test.py": ["tf.nn.local_response_normalization"],
    "nn_ops/ctc_loss_op_test.py": ["tf.nn.ctc_loss"],
    "nn_ops/losses_test.py": ["tf.nn.l2_loss"],
    "nn_ops/xent_op_test.py": [
        "tf.nn.softmax_cross_entropy_with_logits",
        "tf.nn.sparse_softmax_cross_entropy_with_logits",
    ],
    "nn_ops/depthwise_conv_op_test.py": ["tf.nn.depthwise_conv2d"],
    "nn_ops/morphological_ops_test.py": ["tf.nn.dilation2d", "tf.nn.erosion2d"],
    "nn_ops/betainc_op_test.py": ["tf.math.betainc"],
    "nn_ops/atrous_conv2d_test.py": ["tf.nn.atrous_conv2d"],

    # ==================== math_ops ====================
    "math_ops/cwise_ops_unary_test.py": [
        "tf.math.abs", "tf.math.negative", "tf.math.square", "tf.math.sqrt",
        "tf.math.rsqrt", "tf.math.exp", "tf.math.expm1", "tf.math.log", "tf.math.log1p",
        "tf.math.ceil", "tf.math.floor", "tf.math.round",
        "tf.math.sign", "tf.math.sin", "tf.math.cos", "tf.math.tan",
        "tf.math.asin", "tf.math.acos", "tf.math.atan",
        "tf.math.sinh", "tf.math.cosh", "tf.math.tanh",
        "tf.math.asinh", "tf.math.acosh", "tf.math.atanh",
        "tf.math.sigmoid", "tf.math.reciprocal",
        "tf.math.erf", "tf.math.erfc",
        "tf.math.lgamma", "tf.math.digamma",
    ],
    "math_ops/cwise_ops_binary_test.py": [
        "tf.math.add", "tf.math.subtract", "tf.math.multiply", "tf.math.divide",
        "tf.math.floordiv", "tf.math.floormod", "tf.math.pow",
        "tf.math.maximum", "tf.math.minimum",
        "tf.math.equal", "tf.math.not_equal",
        "tf.math.greater", "tf.math.greater_equal",
        "tf.math.less", "tf.math.less_equal",
    ],
    "math_ops/reduction_ops_test.py": [
        "tf.math.reduce_sum", "tf.math.reduce_mean", "tf.math.reduce_max",
        "tf.math.reduce_min", "tf.math.reduce_prod",
        "tf.math.reduce_all", "tf.math.reduce_any",
    ],
    "math_ops/matmul_op_test.py": ["tf.matmul"],
    "math_ops/batch_matmul_op_test.py": ["tf.matmul"],
    "math_ops/argmax_op_test.py": ["tf.math.argmax", "tf.math.argmin"],
    "math_ops/topk_op_test.py": ["tf.math.top_k"],
    "math_ops/clip_ops_test.py": ["tf.clip_by_value", "tf.clip_by_norm"],
    "math_ops/transpose_op_test.py": ["tf.transpose"],
    "math_ops/tensordot_op_test.py": ["tf.tensordot"],
    "math_ops/segment_reduction_ops_test.py": [
        "tf.math.segment_sum", "tf.math.segment_mean",
        "tf.math.segment_max", "tf.math.segment_min",
        "tf.math.segment_prod", "tf.math.unsorted_segment_sum",
    ],
    "math_ops/bincount_op_test.py": ["tf.math.bincount"],
    "math_ops/cumulative_logsumexp_test.py": ["tf.math.cumulative_logsumexp"],
    "math_ops/in_topk_op_test.py": ["tf.math.in_top_k"],
    "math_ops/confusion_matrix_test.py": ["tf.math.confusion_matrix"],
    "math_ops/approx_topk_test.py": ["tf.math.approx_max_k", "tf.math.approx_min_k"],

    # ==================== array_ops ====================
    "array_ops/concat_op_test.py": ["tf.concat"],
    "array_ops/reshape_op_test.py": ["tf.reshape"],
    "array_ops/gather_op_test.py": ["tf.gather"],
    "array_ops/gather_nd_op_test.py": ["tf.gather_nd"],
    "array_ops/slice_op_test.py": ["tf.slice", "tf.strided_slice"],
    "array_ops/stack_op_test.py": ["tf.stack"],
    "array_ops/unstack_op_test.py": ["tf.unstack"],
    "array_ops/split_op_test.py": ["tf.split"],
    "array_ops/pad_op_test.py": ["tf.pad"],
    "array_ops/where_op_test.py": ["tf.where"],
    "array_ops/one_hot_op_test.py": ["tf.one_hot"],
    "array_ops/cast_op_test.py": ["tf.cast"],
    "array_ops/diag_op_test.py": ["tf.linalg.diag"],
    "array_ops/broadcast_to_ops_test.py": ["tf.broadcast_to"],
    "array_ops/scatter_nd_ops_test.py": ["tf.scatter_nd"],
    "array_ops/unique_op_test.py": ["tf.unique"],
    "array_ops/shape_ops_test.py": ["tf.shape", "tf.size", "tf.rank"],
    "array_ops/constant_op_test.py": ["tf.constant", "tf.zeros", "tf.ones", "tf.fill"],
    "array_ops/identity_op_py_test.py": ["tf.identity"],
    "array_ops/batchtospace_op_test.py": ["tf.batch_to_space"],
    "array_ops/spacetobatch_op_test.py": ["tf.space_to_batch"],
    "array_ops/depthtospace_op_test.py": ["tf.nn.depth_to_space"],
    "array_ops/spacetodepth_op_test.py": ["tf.nn.space_to_depth"],
    "array_ops/reverse_sequence_op_test.py": ["tf.reverse_sequence"],
    "array_ops/manip_ops_test.py": ["tf.roll", "tf.tile"],
    "array_ops/bitcast_op_test.py": ["tf.bitcast"],
    "array_ops/matrix_band_part_op_test.py": ["tf.linalg.band_part"],
    "array_ops/batch_gather_op_test.py": ["tf.gather"],

    # ==================== linalg ====================
    "linalg/cholesky_op_test.py": ["tf.linalg.cholesky"],
    "linalg/determinant_op_test.py": ["tf.linalg.det", "tf.linalg.slogdet"],
    "linalg/eig_op_test.py": ["tf.linalg.eig"],
    "linalg/einsum_op_test.py": ["tf.einsum"],
    "linalg/qr_op_test.py": ["tf.linalg.qr"],
    "linalg/svd_op_test.py": ["tf.linalg.svd"],
    "linalg/matrix_inverse_op_test.py": ["tf.linalg.inv"],
    "linalg/matrix_solve_op_test.py": ["tf.linalg.solve"],
    "linalg/matrix_triangular_solve_op_test.py": ["tf.linalg.triangular_solve"],
    "linalg/norm_op_test.py": ["tf.linalg.norm"],
    "linalg/lu_op_test.py": ["tf.linalg.lu"],
    "linalg/matrix_exponential_op_test.py": ["tf.linalg.expm"],
    "linalg/matrix_logarithm_op_test.py": ["tf.linalg.logm"],
    "linalg/matrix_solve_ls_op_test.py": ["tf.linalg.lstsq"],
    "linalg/self_adjoint_eig_op_test.py": ["tf.linalg.eigh"],
    "linalg/normalize_op_test.py": ["tf.linalg.normalize"],
    "linalg/tridiagonal_matmul_op_test.py": ["tf.linalg.tridiagonal_matmul"],
    "linalg/tridiagonal_solve_op_test.py": ["tf.linalg.tridiagonal_solve"],
    "linalg/banded_triangular_solve_op_test.py": ["tf.linalg.banded_triangular_solve"],

    # ==================== signal ====================
    "signal/fft_ops_test.py": ["tf.signal.fft", "tf.signal.ifft", "tf.signal.rfft", "tf.signal.irfft"],
    "signal/dct_ops_test.py": ["tf.signal.dct", "tf.signal.idct"],
    "signal/window_ops_test.py": ["tf.signal.hann_window", "tf.signal.hamming_window"],
    "signal/mel_ops_test.py": ["tf.signal.linear_to_mel_weight_matrix"],
    "signal/shape_ops_test.py": ["tf.signal.frame"],

    # ==================== image_ops ====================
    "image_ops/decode_image_op_test.py": ["tf.image.decode_image"],
    "image_ops/image_ops_test.py": [
        "tf.image.resize", "tf.image.flip_left_right", "tf.image.flip_up_down",
        "tf.image.rot90", "tf.image.transpose",
        "tf.image.rgb_to_grayscale", "tf.image.adjust_brightness",
        "tf.image.adjust_contrast",
    ],

    # ==================== io_ops ====================
    "io_ops/decode_csv_op_test.py": ["tf.io.decode_csv"],
    "io_ops/parse_single_example_op_test.py": ["tf.io.parse_single_example"],
    "io_ops/parsing_ops_test.py": ["tf.io.parse_example"],

    # ==================== strings_ops ====================
    "strings_ops/as_string_op_test.py": ["tf.strings.as_string"],
    "strings_ops/base64_ops_test.py": ["tf.io.encode_base64", "tf.io.decode_base64"],
    "strings_ops/reduce_join_op_test.py": ["tf.strings.reduce_join"],
    "strings_ops/regex_full_match_op_test.py": ["tf.strings.regex_full_match"],
    "strings_ops/regex_replace_op_test.py": ["tf.strings.regex_replace"],
    "strings_ops/string_bytes_split_op_test.py": ["tf.strings.bytes_split"],
    "strings_ops/string_format_op_test.py": ["tf.strings.format"],
    "strings_ops/string_join_op_test.py": ["tf.strings.join"],
    "strings_ops/string_length_op_test.py": ["tf.strings.length"],
    "strings_ops/string_lower_op_test.py": ["tf.strings.lower"],
    "strings_ops/string_split_op_test.py": ["tf.strings.split"],
    "strings_ops/string_strip_op_test.py": ["tf.strings.strip"],
    "strings_ops/string_to_number_op_test.py": ["tf.strings.to_number"],
    "strings_ops/string_upper_op_test.py": ["tf.strings.upper"],
    "strings_ops/substr_op_test.py": ["tf.strings.substr"],
    "strings_ops/unicode_decode_op_test.py": ["tf.strings.unicode_decode"],
    "strings_ops/unicode_encode_op_test.py": ["tf.strings.unicode_encode"],
    "strings_ops/unicode_script_op_test.py": ["tf.strings.unicode_script"],
    "strings_ops/unicode_transcode_op_test.py": ["tf.strings.unicode_transcode"],
}

# 聚焦的核心目录（作为默认展示，不限制实际扫描范围）
DEFAULT_CORE_DIRS = [
    "nn_ops", "math_ops", "array_ops", "linalg", "signal", "image_ops",
    "io_ops", "strings_ops", "random", "sparse_ops", "quantization_ops",
    "summary_ops", "variables", "control_flow",
]

# 目录 → TF 命名空间的默认映射（用于 AST 推断）
DIR_TO_NAMESPACE = {
    "nn_ops": "tf.nn",
    "math_ops": "tf.math",
    "array_ops": "tf",
    "linalg": "tf.linalg",
    "signal": "tf.signal",
    "image_ops": "tf.image",
    "io_ops": "tf.io",
    "strings_ops": "tf.strings",
    "random": "tf.random",
    "sparse_ops": "tf",
    "quantization_ops": "tf.quantization",
    "summary_ops": "tf.summary",
    "variables": "tf",
    "control_flow": "tf",
    "data_structures": "tf",
    "distributions": "tf",
    "custom_ops": "tf",
    "proto": "tf",
}

# 排除的文件模式（不包含有用的算子测试）
EXCLUDE_PATTERNS = [
    "BUILD", "__init__.py", "_base.py", "_d9m_test.py",
    "benchmark_test.py", "v1_compat_tests",
    "cudnn_", "deterministic_",
]

EXCLUDE_DIR_NAMES = ["__pycache__", "v1_compat_tests"]


def should_skip_file(filename: str) -> bool:
    """判断是否应跳过某个文件"""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in filename:
            return True
    return False


def get_default_categories(tf_dir: str) -> List[str]:
    """默认扫描 tf_testcases 下除排除目录外的所有子目录"""
    categories: List[str] = []
    try:
        for entry in os.listdir(tf_dir):
            entry_path = os.path.join(tf_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            if entry in EXCLUDE_DIR_NAMES:
                continue
            categories.append(entry)
    except Exception:
        return DEFAULT_CORE_DIRS

    return sorted(categories)


def extract_test_class_names(filepath: str) -> List[str]:
    """通过 AST 提取测试类名"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            tree = ast.parse(f.read())
        class_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if "Test" in node.name or "test" in node.name.lower():
                    class_names.append(node.name)
        return class_names
    except Exception:
        return []


def infer_api_from_filename(filepath: str, directory: str) -> List[str]:
    """
    从文件名和目录推断可能的 TF API 名称
    
    规则：
    - 去除 _op_test.py, _ops_test.py, _test.py 后缀
    - 根据目录映射到对应的 tf.* 命名空间
    """
    filename = os.path.basename(filepath)
    
    # 去除后缀
    api_name = filename
    for suffix in ["_op_test.py", "_ops_test.py", "_test.py", ".py"]:
        if api_name.endswith(suffix):
            api_name = api_name[:-len(suffix)]
            break
    
    if not api_name:
        return []
    
    # 获取命名空间
    namespace = DIR_TO_NAMESPACE.get(directory, "tf")
    
    return [f"{namespace}.{api_name}"]


def extract_apis_from_directory(tf_dir: str, categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    从 tf_testcases 目录中提取所有 TF API
    
    Args:
        tf_dir: tf_testcases 目录路径
        categories: 要扫描的子目录列表，None 表示扫描所有聚焦目录
    
    Returns:
        API 列表，每个元素包含 tf_api, source_file, category 等信息
    """
    if categories is None:
        categories = get_default_categories(tf_dir)
    
    all_apis: Dict[str, Dict[str, Any]] = {}  # tf_api -> info (去重用)
    
    for category in categories:
        category_dir = os.path.join(tf_dir, category)
        if not os.path.isdir(category_dir):
            print(f"⚠️ 目录不存在，跳过: {category_dir}")
            continue
        
        print(f"\n📂 扫描目录: {category}")
        
        for filename in sorted(os.listdir(category_dir)):
            if not filename.endswith(".py"):
                continue
            if should_skip_file(filename):
                continue
            
            rel_path = f"{category}/{filename}"
            abs_path = os.path.join(category_dir, filename)
            
            # 优先使用预定义映射
            if rel_path in KNOWN_FILE_APIS:
                apis = KNOWN_FILE_APIS[rel_path]
                for api_name in apis:
                    if api_name not in all_apis:
                        all_apis[api_name] = {
                            "tf_api": api_name,
                            "source_file": rel_path,
                            "category": category,
                            "extraction_method": "predefined",
                        }
                print(f"  ✅ {filename} → {len(apis)} 个API（预定义映射）")
            else:
                # AST + 文件名推断
                inferred = infer_api_from_filename(abs_path, category)
                class_names = extract_test_class_names(abs_path)
                
                for api_name in inferred:
                    if api_name not in all_apis:
                        all_apis[api_name] = {
                            "tf_api": api_name,
                            "source_file": rel_path,
                            "category": category,
                            "extraction_method": "inferred",
                            "test_classes": class_names,
                        }
                if inferred:
                    print(f"  🔍 {filename} → {len(inferred)} 个API（启发式推断）")
                else:
                    print(f"  ⏭️ {filename} → 未识别到API")
    
    # 转为列表并排序
    result = sorted(all_apis.values(), key=lambda x: x["tf_api"])
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: 从 TensorFlow 官方测试文件中提取被测试的 TF API 列表"
    )
    parser.add_argument(
        "--tf-dir", default=os.path.join(ROOT_DIR, "tf_testcases"),
        help="tf_testcases 目录路径（默认: 项目根目录/tf_testcases）"
    )
    parser.add_argument(
        "--output", "-o", default=os.path.join(ROOT_DIR, "tf_pt_test", "data", "tf_apis_new.json"),
        help="输出的 JSON 文件路径"
    )
    parser.add_argument(
        "--categories", "-c", nargs="*", default=None,
        help=(
            "要扫描的子目录（默认: 自动发现 tf_testcases 下所有子目录；"
            f"核心目录参考: {DEFAULT_CORE_DIRS}）"
        )
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Step 1: 从 TensorFlow 官方测试文件中提取 TF API 列表")
    print("=" * 80)
    print(f"📁 测试文件目录: {args.tf_dir}")
    print(f"📁 输出文件: {args.output}")
    
    if not os.path.isdir(args.tf_dir):
        print(f"❌ 目录不存在: {args.tf_dir}")
        sys.exit(1)
    
    # 提取 API 列表
    apis = extract_apis_from_directory(args.tf_dir, args.categories)
    
    # 统计
    predefined_count = sum(1 for a in apis if a.get("extraction_method") == "predefined")
    inferred_count = sum(1 for a in apis if a.get("extraction_method") == "inferred")
    categories_found = set(a["category"] for a in apis)
    
    print(f"\n{'=' * 80}")
    print(f"📊 提取结果汇总")
    print(f"{'=' * 80}")
    print(f"  总API数量: {len(apis)}")
    print(f"  预定义映射: {predefined_count}")
    print(f"  启发式推断: {inferred_count}")
    print(f"  覆盖目录: {', '.join(sorted(categories_found))}")
    
    # 按目录统计
    for cat in sorted(categories_found):
        count = sum(1 for a in apis if a["category"] == cat)
        print(f"    {cat}: {count} 个API")
    
    # 保存结果
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    output_data = {
        "total_apis": len(apis),
        "extraction_time": __import__("datetime").datetime.now().isoformat(),
        "source_dir": args.tf_dir,
        "apis": apis,
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 已保存到: {args.output}")
    print(f"📋 前10个API: {', '.join(a['tf_api'] for a in apis[:10])}")


if __name__ == "__main__":
    main()
