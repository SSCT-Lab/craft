#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1: Extract Paddle API list from PaddlePaddle official test files

Function:
- Scan testcases_pd/ test_*_op.py files
- Use predefined mappings + AST to extract public Paddle API names
- Deduplicate and output a structured API list (JSON)

Usage:
    conda activate tf_env
    python pd_pt_test/extract_pd_apis.py [--pd-dir testcases_pd] [--output pd_pt_test/data/pd_apis_new.json]

Output: pd_pt_test/data/pd_apis_new.json
"""

import os
import sys
import ast
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Set, Any, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== Predefined test file -> Paddle API mapping ====================
# Paddle test files are usually named test_xxx_op.py; one file may cover multiple APIs
KNOWN_FILE_APIS: Dict[str, List[str]] = {
    # ==================== Activation ====================
    "test_activation_op.py": [
        "paddle.nn.functional.relu", "paddle.nn.functional.relu6",
        "paddle.nn.functional.leaky_relu", "paddle.nn.functional.elu",
        "paddle.nn.functional.selu", "paddle.nn.functional.silu",
        "paddle.nn.functional.gelu", "paddle.nn.functional.celu",
        "paddle.nn.functional.mish", "paddle.nn.functional.swish",
        "paddle.nn.functional.hardsigmoid", "paddle.nn.functional.hardswish",
        "paddle.nn.functional.hardtanh", "paddle.nn.functional.softshrink",
        "paddle.nn.functional.hardshrink", "paddle.nn.functional.softplus",
        "paddle.nn.functional.softsign", "paddle.nn.functional.log_sigmoid",
        "paddle.exp", "paddle.expm1",
        "paddle.tanh", "paddle.sigmoid",
        "paddle.sqrt", "paddle.rsqrt",
        "paddle.abs", "paddle.ceil", "paddle.floor",
        "paddle.cos", "paddle.sin", "paddle.tan",
        "paddle.acos", "paddle.asin", "paddle.atan",
        "paddle.cosh", "paddle.sinh",
        "paddle.acosh", "paddle.asinh", "paddle.atanh",
        "paddle.sign", "paddle.reciprocal",
        "paddle.square", "paddle.log", "paddle.log2", "paddle.log10",
        "paddle.log1p",
    ],
    "test_activation_stride_op.py": [],

    # ==================== Arithmetic ====================
    "test_add_op.py": ["paddle.add"],
    "test_add_n_op.py": ["paddle.add_n"],
    "test_addmm_op.py": ["paddle.addmm"],
    "test_subtract_op.py": ["paddle.subtract"],
    "test_multiply_op.py": ["paddle.multiply"],
    "test_elementwise_mul_op.py": ["paddle.multiply"],
    "test_elementwise_add_op.py": ["paddle.add"],
    "test_elementwise_sub_op.py": ["paddle.subtract"],
    "test_elementwise_div_op.py": ["paddle.divide"],
    "test_elementwise_mod_op.py": ["paddle.remainder"],
    "test_elementwise_pow_op.py": ["paddle.pow"],
    "test_elementwise_max_op.py": ["paddle.maximum"],
    "test_elementwise_min_op.py": ["paddle.minimum"],
    "test_elementwise_floordiv_op.py": ["paddle.floor_divide"],
    "test_floor_divide_op.py": ["paddle.floor_divide"],
    "test_modulo_op.py": ["paddle.remainder"],

    # ==================== Matrix ====================
    "test_matmul_v2_op.py": ["paddle.matmul"],
    "test_matmul_op.py": ["paddle.matmul"],
    "test_dot_op.py": ["paddle.dot"],
    "test_cross_op.py": ["paddle.cross"],
    "test_bmm_op.py": ["paddle.bmm"],
    "test_mv_op.py": ["paddle.mv"],
    "test_mm_op.py": ["paddle.mm"],
    "test_einsum_v2_op.py": ["paddle.einsum"],
    "test_einsum_op.py": ["paddle.einsum"],
    "test_tensordot_op.py": ["paddle.tensordot"],
    "test_trace_op.py": ["paddle.trace"],
    "test_inner_op.py": ["paddle.inner"],
    "test_outer_op.py": ["paddle.outer"],
    "test_kron_op.py": ["paddle.kron"],

    # ==================== Reduction ====================
    "test_reduce_op.py": [
        "paddle.sum", "paddle.mean", "paddle.max", "paddle.min",
        "paddle.prod", "paddle.all", "paddle.any",
    ],
    "test_sum_op.py": ["paddle.sum"],
    "test_mean_op.py": ["paddle.mean"],
    "test_max_op.py": ["paddle.max"],
    "test_min_op.py": ["paddle.min"],
    "test_prod_op.py": ["paddle.prod"],
    "test_cumsum_op.py": ["paddle.cumsum"],
    "test_cumprod_op.py": ["paddle.cumprod"],
    "test_logsumexp_op.py": ["paddle.logsumexp"],
    "test_logcumsumexp_op.py": ["paddle.logcumsumexp"],

    # ==================== Comparison ====================
    "test_compare_op.py": [
        "paddle.equal", "paddle.not_equal",
        "paddle.greater_than", "paddle.greater_equal",
        "paddle.less_than", "paddle.less_equal",
    ],
    "test_allclose_op.py": ["paddle.allclose"],
    "test_isnan_op.py": ["paddle.isnan"],
    "test_isinf_v2_op.py": ["paddle.isinf"],

    # ==================== Array ops ====================
    "test_concat_op.py": ["paddle.concat"],
    "test_stack_op.py": ["paddle.stack"],
    "test_unstack_op.py": ["paddle.unstack"],
    "test_split_op.py": ["paddle.split"],
    "test_reshape_op.py": ["paddle.reshape"],
    "test_squeeze_op.py": ["paddle.squeeze"],
    "test_unsqueeze_op.py": ["paddle.unsqueeze"],
    "test_flatten_op.py": ["paddle.flatten"],
    "test_transpose_op.py": ["paddle.transpose"],
    "test_gather_op.py": ["paddle.gather"],
    "test_gather_nd_op.py": ["paddle.gather_nd"],
    "test_scatter_op.py": ["paddle.scatter"],
    "test_scatter_nd_op.py": ["paddle.scatter_nd"],
    "test_fill_constant_op.py": ["paddle.full"],
    "test_fill_op.py": ["paddle.fill_"],
    "test_zeros_like_op.py": ["paddle.zeros_like"],
    "test_ones_like_op.py": ["paddle.ones_like"],
    "test_full_like_op.py": ["paddle.full_like"],
    "test_full_op.py": ["paddle.full"],
    "test_arange_op.py": ["paddle.arange"],
    "test_linspace_op.py": ["paddle.linspace"],
    "test_meshgrid_op.py": ["paddle.meshgrid"],
    "test_expand_op.py": ["paddle.expand"],
    "test_expand_as_op.py": ["paddle.expand_as"],
    "test_tile_op.py": ["paddle.tile"],
    "test_repeat_interleave_op.py": ["paddle.repeat_interleave"],
    "test_slice_op.py": ["paddle.slice"],
    "test_strided_slice_op.py": ["paddle.strided_slice"],
    "test_flip_op.py": ["paddle.flip"],
    "test_roll_op.py": ["paddle.roll"],
    "test_reverse_op.py": ["paddle.reverse"],
    "test_unique_op.py": ["paddle.unique"],
    "test_cast_op.py": ["paddle.cast"],
    "test_pad_op.py": ["paddle.nn.functional.pad"],
    "test_pad3d_op.py": ["paddle.nn.functional.pad"],
    "test_where_op.py": ["paddle.where"],
    "test_one_hot_op.py": ["paddle.nn.functional.one_hot"],
    "test_top_k_op.py": ["paddle.topk"],
    "test_top_k_v2_op.py": ["paddle.topk"],
    "test_index_select_op.py": ["paddle.index_select"],
    "test_masked_select_op.py": ["paddle.masked_select"],
    "test_nonzero_op.py": ["paddle.nonzero"],
    "test_sort_op.py": ["paddle.sort"],
    "test_argsort_op.py": ["paddle.argsort"],
    "test_arg_min_max_op.py": ["paddle.argmax", "paddle.argmin"],
    "test_arg_min_max_v2_op.py": ["paddle.argmax", "paddle.argmin"],
    "test_diag_v2_op.py": ["paddle.diag"],
    "test_diag_embed_op.py": ["paddle.diag_embed"],
    "test_diagonal_op.py": ["paddle.diagonal"],
    "test_triu_op.py": ["paddle.triu"],
    "test_tril_op.py": ["paddle.tril"],
    "test_broadcast_to_op.py": ["paddle.broadcast_to"],
    "test_eye_op.py": ["paddle.eye"],
    "test_empty_op.py": ["paddle.empty"],
    "test_assign_op.py": ["paddle.assign"],
    "test_clip_op.py": ["paddle.clip"],
    "test_size_op.py": ["paddle.numel"],
    "test_shape_op.py": ["paddle.shape"],
    "test_chunk_op.py": ["paddle.chunk"],
    "test_unbind_op.py": ["paddle.unbind"],

    # ==================== Convolution ====================
    "test_conv2d_op.py": ["paddle.nn.functional.conv2d"],
    "test_conv3d_op.py": ["paddle.nn.functional.conv3d"],
    "test_conv1d_op.py": ["paddle.nn.functional.conv1d"],
    "test_conv2d_transpose_op.py": ["paddle.nn.functional.conv2d_transpose"],
    "test_conv3d_transpose_op.py": ["paddle.nn.functional.conv3d_transpose"],
    "test_conv1d_transpose_op.py": ["paddle.nn.functional.conv1d_transpose"],
    "test_depthwise_conv2d_op.py": ["paddle.nn.functional.conv2d"],
    "test_deformable_conv_op.py": ["paddle.vision.ops.deform_conv2d"],

    # ==================== Pooling ====================
    "test_pool2d_op.py": ["paddle.nn.functional.max_pool2d", "paddle.nn.functional.avg_pool2d"],
    "test_pool3d_op.py": ["paddle.nn.functional.max_pool3d", "paddle.nn.functional.avg_pool3d"],
    "test_pool1d_op.py": ["paddle.nn.functional.max_pool1d", "paddle.nn.functional.avg_pool1d"],
    "test_adaptive_avg_pool2d_op.py": ["paddle.nn.functional.adaptive_avg_pool2d"],
    "test_adaptive_avg_pool3d_op.py": ["paddle.nn.functional.adaptive_avg_pool3d"],
    "test_adaptive_max_pool2d_op.py": ["paddle.nn.functional.adaptive_max_pool2d"],

    # ==================== Normalization ====================
    "test_batch_norm_op.py": ["paddle.nn.functional.batch_norm"],
    "test_layer_norm_op.py": ["paddle.nn.functional.layer_norm"],
    "test_group_norm_op.py": ["paddle.nn.functional.group_norm"],
    "test_instance_norm_op.py": ["paddle.nn.functional.instance_norm"],
    "test_norm_op.py": ["paddle.norm"],

    # ==================== Loss functions ====================
    "test_softmax_op.py": ["paddle.nn.functional.softmax"],
    "test_log_softmax_op.py": ["paddle.nn.functional.log_softmax"],
    "test_cross_entropy_op.py": ["paddle.nn.functional.cross_entropy"],
    "test_bce_loss_op.py": ["paddle.nn.functional.binary_cross_entropy"],
    "test_bce_with_logits_loss_op.py": ["paddle.nn.functional.binary_cross_entropy_with_logits"],
    "test_mse_loss_op.py": ["paddle.nn.functional.mse_loss"],
    "test_smooth_l1_loss_op.py": ["paddle.nn.functional.smooth_l1_loss"],
    "test_l1_loss_op.py": ["paddle.nn.functional.l1_loss"],
    "test_nll_loss_op.py": ["paddle.nn.functional.nll_loss"],
    "test_kldiv_loss_op.py": ["paddle.nn.functional.kl_div"],
    "test_huber_loss_op.py": ["paddle.nn.functional.smooth_l1_loss"],
    "test_margin_ranking_loss_op.py": ["paddle.nn.functional.margin_ranking_loss"],
    "test_triplet_margin_with_distance_loss_op.py": ["paddle.nn.functional.triplet_margin_with_distance_loss"],

    # ==================== Dropout ====================
    "test_dropout_op.py": ["paddle.nn.functional.dropout"],

    # ==================== Linear algebra ====================
    "test_cholesky_op.py": ["paddle.linalg.cholesky"],
    "test_det_op.py": ["paddle.linalg.det"],
    "test_slogdet_op.py": ["paddle.linalg.slogdet"],
    "test_eig_op.py": ["paddle.linalg.eig"],
    "test_eigh_op.py": ["paddle.linalg.eigh"],
    "test_eigvalsh_op.py": ["paddle.linalg.eigvalsh"],
    "test_qr_op.py": ["paddle.linalg.qr"],
    "test_svd_op.py": ["paddle.linalg.svd"],
    "test_lu_op.py": ["paddle.linalg.lu"],
    "test_matrix_power_op.py": ["paddle.linalg.matrix_power"],
    "test_inverse_op.py": ["paddle.linalg.inv"],
    "test_solve_op.py": ["paddle.linalg.solve"],
    "test_triangular_solve_op.py": ["paddle.linalg.triangular_solve"],
    "test_norm_op.py": ["paddle.linalg.norm"],
    "test_lstsq_op.py": ["paddle.linalg.lstsq"],

    # ==================== FFT ====================
    "test_fft_op.py": [
        "paddle.fft.fft", "paddle.fft.ifft",
        "paddle.fft.rfft", "paddle.fft.irfft",
        "paddle.fft.fft2", "paddle.fft.ifft2",
    ],

    # ==================== Random ====================
    "test_uniform_random_op.py": ["paddle.uniform"],
    "test_gaussian_random_op.py": ["paddle.randn"],
    "test_randint_op.py": ["paddle.randint"],
    "test_randperm_op.py": ["paddle.randperm"],
    "test_bernoulli_op.py": ["paddle.bernoulli"],

    # ==================== Logical ====================
    "test_logical_op.py": [
        "paddle.logical_and", "paddle.logical_or",
        "paddle.logical_not", "paddle.logical_xor",
    ],
    "test_bitwise_op.py": [
        "paddle.bitwise_and", "paddle.bitwise_or",
        "paddle.bitwise_not", "paddle.bitwise_xor",
    ],

    # ==================== Other common operators ====================
    "test_interpolate_v2_op.py": ["paddle.nn.functional.interpolate"],
    "test_interpolate_op.py": ["paddle.nn.functional.interpolate"],
    "test_grid_sampler_op.py": ["paddle.nn.functional.grid_sample"],
    "test_affine_grid_op.py": ["paddle.nn.functional.affine_grid"],
    "test_embedding_op.py": ["paddle.nn.functional.embedding"],
    "test_linear_op.py": ["paddle.nn.functional.linear"],
    "test_bilinear_tensor_product_op.py": ["paddle.nn.functional.bilinear"],
    "test_pixel_shuffle_op.py": ["paddle.nn.functional.pixel_shuffle"],
    "test_atan2_op.py": ["paddle.atan2"],
    "test_floor_op.py": ["paddle.floor"],
    "test_ceil_op.py": ["paddle.ceil"],
    "test_round_op.py": ["paddle.round"],
    "test_abs_op.py": ["paddle.abs"],
    "test_sign_op.py": ["paddle.sign"],
    "test_neg_op.py": ["paddle.neg"],
    "test_erfinv_op.py": ["paddle.erfinv"],
    "test_erf_op.py": ["paddle.erf"],
    "test_lgamma_op.py": ["paddle.lgamma"],
    "test_digamma_op.py": ["paddle.digamma"],
    "test_lerp_op.py": ["paddle.lerp"],
    "test_trunc_op.py": ["paddle.trunc"],
    "test_frac_op.py": ["paddle.frac"],
    "test_dist_op.py": ["paddle.dist"],
    "test_cov_op.py": ["paddle.linalg.cov"],
    "test_corrcoef_op.py": ["paddle.linalg.corrcoef"],
    "test_histogram_op.py": ["paddle.histogram"],
    "test_bincount_op.py": ["paddle.bincount"],
    "test_searchsorted_op.py": ["paddle.searchsorted"],
    "test_bucketize_op.py": ["paddle.bucketize"],
    "test_take_along_axis_op.py": ["paddle.take_along_axis"],
    "test_put_along_axis_op.py": ["paddle.put_along_axis"],
    "test_complex_op.py": ["paddle.complex"],
    "test_angle_op.py": ["paddle.angle"],
    "test_conj_op.py": ["paddle.conj"],
    "test_real_op.py": ["paddle.real"],
    "test_imag_op.py": ["paddle.imag"],
}

# Excluded file patterns (do not include useful operator tests)
# Note: substring match (pattern in filename); avoid overly broad patterns
# Pattern for test_*_op.py files (e.g., "op.py"). Files not starting with test_
# are already excluded by the should_skip_file prefix check; no need to repeat.
EXCLUDE_PATTERNS = [
    "BUILD", "__init__.py", "_base.py",
    "benchmark", "dist_", "fleet_", "parallel_",
    "nproc_", "multi_process", "spawn_",
    "decorator_helper", "gradient_checker",
    "feed_data", "fake_reader", "ctr_dataset",
    "prim_op_test.py", "nets.py",
    "detected_gpu", "detected_xpu",
    "hubconf.py", "find_ports",
    "hdfs_test", "launch_function",
    "ir_memory_optimize", "jit_load",
    "simple_nets.py", "seresnext_net.py",
]


def should_skip_file(filename: str) -> bool:
    """Determine whether to skip a file."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in filename:
            return True
    # Only keep .py files starting with test_
    if not filename.startswith("test_"):
        return True
    if not filename.endswith(".py"):
        return True
    return False


def extract_python_api_from_ast(filepath: str) -> List[str]:
    """
    Use AST to extract self.python_api assignments from test files.

    In Paddle test files, setUp usually includes:
        self.python_api = paddle.nn.functional.relu
        self.public_python_api = paddle.concat
    """
    apis = set()
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Regex extract self.python_api = paddle.xxx.yyy or F.xxx
        patterns = [
            r'self\.python_api\s*=\s*(paddle\.[a-zA-Z_][a-zA-Z0-9_.]*)',
            r'self\.public_python_api\s*=\s*(paddle\.[a-zA-Z_][a-zA-Z0-9_.]*)',
            r'self\.python_api\s*=\s*F\.([a-zA-Z_][a-zA-Z0-9_]*)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content):
                api_name = match.group(1)
                # F.xxx -> paddle.nn.functional.xxx
                if api_name.startswith("F."):
                    api_name = f"paddle.nn.functional.{api_name[2:]}"
                # Exclude helper methods
                if not api_name.endswith("_wrapper") and "test" not in api_name.lower():
                    apis.add(api_name)

    except Exception:
        pass

    return list(apis)


def extract_op_type_from_ast(filepath: str) -> List[str]:
    """
    Use AST/regex to extract self.op_type from test files.

    Used as a fallback to infer API names.
    """
    op_types = set()
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        pattern = r'self\.op_type\s*=\s*["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']'
        for match in re.finditer(pattern, content):
            op_types.add(match.group(1))

    except Exception:
        pass

    return list(op_types)


def infer_api_from_filename(filename: str) -> Optional[str]:
    """
    Infer a possible Paddle API name from a filename.

    Rule: test_xxx_op.py -> paddle.xxx
    """
    name = filename
    for suffix in ["_op.py", "_ops.py", ".py"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    if name.startswith("test_"):
        name = name[5:]

    if not name:
        return None

    return f"paddle.{name}"


def extract_apis_from_directory(pd_dir: str) -> List[Dict[str, Any]]:
    """
    Extract all Paddle APIs from the testcases_pd directory.

    Strategy:
    1. Prefer predefined mapping
    2. If not found, use AST to extract self.python_api / self.public_python_api
    3. Finally infer from filename (fallback only)
    """
    all_apis: Dict[str, Dict[str, Any]] = {}

    py_files = sorted([
        f for f in os.listdir(pd_dir)
        if f.endswith(".py") and not should_skip_file(f)
    ])

    print(f"\n📂 Scan directory: {pd_dir}")
    print(f"📋 Candidate test files: {len(py_files)}")

    for filename in py_files:
        abs_path = os.path.join(pd_dir, filename)

        # 1. Predefined mapping
        if filename in KNOWN_FILE_APIS:
            apis = KNOWN_FILE_APIS[filename]
            for api_name in apis:
                if api_name and api_name not in all_apis:
                    all_apis[api_name] = {
                        "pd_api": api_name,
                        "source_file": filename,
                        "extraction_method": "predefined",
                    }
            if apis:
                print(f"  ✅ {filename} -> {len(apis)} APIs (predefined mapping)")
            continue

        # 2. AST extract python_api
        ast_apis = extract_python_api_from_ast(abs_path)
        if ast_apis:
            for api_name in ast_apis:
                if api_name not in all_apis:
                    all_apis[api_name] = {
                        "pd_api": api_name,
                        "source_file": filename,
                        "extraction_method": "ast_python_api",
                    }
            print(f"  🔍 {filename} -> {len(ast_apis)} APIs (AST python_api)")
            continue

        # 3. Infer from filename (fallback)
        inferred = infer_api_from_filename(filename)
        if inferred:
            # Check op_type to confirm
            op_types = extract_op_type_from_ast(abs_path)
            if op_types:
                for op_type in op_types:
                    api_name = f"paddle.{op_type}"
                    if api_name not in all_apis:
                        all_apis[api_name] = {
                            "pd_api": api_name,
                            "source_file": filename,
                            "extraction_method": "inferred_op_type",
                        }
                print(f"  🔍 {filename} -> {len(op_types)} APIs (op_type inference)")
            else:
                if inferred not in all_apis:
                    all_apis[inferred] = {
                        "pd_api": inferred,
                        "source_file": filename,
                        "extraction_method": "inferred_filename",
                    }
                print(f"  🔍 {filename} -> 1 API (filename inference)")

    result = sorted(all_apis.values(), key=lambda x: x["pd_api"])
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Extract Paddle APIs from PaddlePaddle official test files"
    )
    parser.add_argument(
        "--pd-dir", default=os.path.join(ROOT_DIR, "testcases_pd"),
        help="testcases_pd directory path (default: project_root/testcases_pd)"
    )
    parser.add_argument(
        "--output", "-o", default=os.path.join(ROOT_DIR, "pd_pt_test", "data", "pd_apis_new.json"),
        help="Output JSON filepath"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Step 1: Extract Paddle APIs from PaddlePaddle official test files")
    print("=" * 80)
    print(f"📁 Test file directory: {args.pd_dir}")
    print(f"📁 Output file: {args.output}")

    if not os.path.isdir(args.pd_dir):
        print(f"❌ Directory does not exist: {args.pd_dir}")
        sys.exit(1)

    # Extract API list
    apis = extract_apis_from_directory(args.pd_dir)

    # Summary
    method_counts: Dict[str, int] = {}
    for a in apis:
        method = a.get("extraction_method", "unknown")
        method_counts[method] = method_counts.get(method, 0) + 1

    print(f"\n{'=' * 80}")
    print("📊 Extract summary")
    print(f"{'=' * 80}")
    print(f"  Total APIs: {len(apis)}")
    for method, count in sorted(method_counts.items()):
        print(f"  {method}: {count}")

    # Save results
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    output_data = {
        "total_apis": len(apis),
        "extraction_time": __import__("datetime").datetime.now().isoformat(),
        "source_dir": args.pd_dir,
        "apis": apis,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved to: {args.output}")
    print(f"📋 First 10 APIs: {', '.join(a['pd_api'] for a in apis[:10])}")


if __name__ == "__main__":
    main()

