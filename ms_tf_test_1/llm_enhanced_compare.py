#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4: 基于 LLM 的 MindSpore 与 TensorFlow 算子差分测试框架

功能：
- 从 JSON 文件加载 MS 测试用例和 MS→TF 映射
- 对每对等价算子，分别在 MindSpore 和 TensorFlow 中执行并比较结果
- 使用 LLM 进行测试用例的修复（repair）、变异（mutation）和跳过（skip）
- LLM 调用支持并发，算子执行顺序串行，保证输入值一致
- 保存详细测试结果和批量日志

用法：
    conda activate tf_env
    python ms_tf_test_1/llm_enhanced_compare.py \
        [--max-iterations 3] [--num-cases 3] [--workers 6] \
        [--start 1] [--end N] [--operators mindspore.ops.Abs mindspore.ops.Add]

前置条件：
    1. 已运行 Step 1 extract_ms_apis.py
    2. 已运行 Step 2 extract_ms_test_cases.py
    3. 已运行 Step 3 extract_ms_tf_mapping.py
"""

import os
# 兼容 MindSpore 与 protobuf 版本差异，避免 Descriptors cannot be created directly
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import sys

import json
import csv
import copy
import time
import re
import traceback
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from threading import Lock, RLock
from concurrent.futures import ThreadPoolExecutor, as_completed

import tensorflow as tf
from openai import OpenAI

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from component.doc.doc_crawler_factory import get_doc_content

# ==================== 常量定义 ====================
DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_NUM_CASES = 5
DEFAULT_WORKERS = 6

DATA_DIR = os.path.join(ROOT_DIR, "ms_tf_test_1", "data")
DEFAULT_TEST_CASES_FILE = os.path.join(DATA_DIR, "ms_test_cases.json")
DEFAULT_MAPPING_FILE = os.path.join(DATA_DIR, "ms_tf_mapping_validated.csv")


# ==================== MindSpore 延迟加载 ====================
_mindspore = None
_ms_context_set = False
_tensorflow = None


def get_mindspore():
    """延迟加载 MindSpore，并设置上下文"""
    global _mindspore, _ms_context_set
    if _mindspore is None:
        import mindspore
        _mindspore = mindspore
    if not _ms_context_set:
        try:
            _mindspore.set_device("CPU")
            _mindspore.context.set_context(mode=_mindspore.context.PYNATIVE_MODE)
        except Exception:
            _mindspore.context.set_context(
                mode=_mindspore.context.PYNATIVE_MODE, device_target="CPU"
            )
        _ms_context_set = True
    return _mindspore


def get_tensorflow():
    """延迟加载 TensorFlow"""
    global _tensorflow
    if _tensorflow is None:
        _tensorflow = tf
    return _tensorflow


class LLMEnhancedComparator:
    """基于 LLM 的 MindSpore 与 TensorFlow 差分测试框架"""

    def __init__(
        self,
        test_cases_file: str = DEFAULT_TEST_CASES_FILE,
        mapping_file: str = DEFAULT_MAPPING_FILE,
        key_path: str = DEFAULT_KEY_PATH,
        model: str = DEFAULT_MODEL,
        print_lock: Lock = None,
        llm_workers: int = DEFAULT_WORKERS,
    ):
        self.model = model
        self.print_lock = print_lock or Lock()
        self.llm_workers = max(1, int(llm_workers))
        self.execution_lock = RLock()
        self.stats_lock = Lock()

        api_key = self._load_api_key(key_path)
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        self.test_cases_data = self._load_test_cases(test_cases_file)
        self._safe_print(f"📋 已加载 {len(self.test_cases_data)} 个 MS API 的测试用例")

        self.api_mapping = self._load_mapping(mapping_file)
        has_impl = sum(1 for v in self.api_mapping.values() if v != "无对应实现")
        self._safe_print(f"📋 已加载 {len(self.api_mapping)} 个映射（{has_impl} 个有对应实现）")

        self.result_dir = os.path.join(ROOT_DIR, "ms_tf_test_1", "ms_tf_log_1")
        os.makedirs(self.result_dir, exist_ok=True)
        self._safe_print(f"📁 结果存储目录: {self.result_dir}")

        self.random_seed = 42
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

    # ==================== 辅助方法 ====================

    def _safe_print(self, msg: str, end: str = "\n"):
        with self.print_lock:
            print(msg, end=end, flush=True)

    def _load_api_key(self, key_path: str) -> str:
        if not os.path.isabs(key_path):
            key_file = os.path.join(ROOT_DIR, key_path)
        else:
            key_file = key_path
        if os.path.exists(key_file):
            with open(key_file, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            if api_key:
                return api_key
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            return api_key
        self._safe_print("❌ 未找到 API 密钥")
        return ""

    def _load_test_cases(self, filepath: str) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            self._safe_print(f"⚠️ 测试用例文件不存在: {filepath}")
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("test_cases", {})

    def _load_mapping(self, filepath: str) -> Dict[str, str]:
        if not os.path.exists(filepath):
            self._safe_print(f"⚠️ 映射文件不存在: {filepath}")
            return {}
        mapping = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ms_api = row.get("mindspore-api", "").strip()
                tf_api = row.get("tensorflow-api", "").strip()
                if ms_api and tf_api:
                    mapping[ms_api] = tf_api
        return mapping

    # ==================== API 工具方法 ====================

    def is_class_based_api(self, api_name: str) -> bool:
        """判断 API 是否是基于类的（首字母大写）"""
        parts = api_name.split(".")
        if len(parts) >= 2:
            last_part = parts[-1]
            if last_part and last_part[0].isupper():
                return True
        return False

    def get_ms_function(self, api_name: str):
        """获取 MindSpore 算子函数/类对象"""
        ms = get_mindspore()
        try:
            parts = api_name.split(".")
            # parts[0] = "mindspore"
            obj = ms
            for part in parts[1:]:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            # 尝试在 ops.operations 中查找
            try:
                obj = ms.ops.operations
                op_name = parts[-1]
                return getattr(obj, op_name)
            except AttributeError:
                pass
            return None

    def get_tf_function(self, api_name: str):
        """获取 TensorFlow 算子函数/类对象"""
        try:
            tf_module = get_tensorflow()
            parts = api_name.split(".")
            if not parts:
                return None

            if parts[0] in {"tf", "tensorflow"}:
                obj = tf_module
                walk_parts = parts[1:]
            else:
                obj = tf_module
                walk_parts = parts

            for part in walk_parts:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return None

    def convert_api_name(self, ms_api: str) -> Tuple[Optional[str], Optional[str], str]:
        """查找 MS API 对应的 TensorFlow API"""
        if ms_api in self.api_mapping:
            tf_api = self.api_mapping[ms_api]
            if tf_api and tf_api != "无对应实现":
                return ms_api, tf_api, "映射表"
            else:
                return ms_api, None, "无对应实现"
        return ms_api, None, "映射表中未找到"

    # ==================== 数据转换 ====================

    def generate_numpy_data(self, data: Any) -> np.ndarray:
        """从描述生成 numpy 数组"""
        if isinstance(data, dict):
            if "shape" in data:
                raw_shape = data["shape"]
                if isinstance(raw_shape, int):
                    shape = (raw_shape,)
                elif isinstance(raw_shape, (list, tuple)):
                    shape = tuple(raw_shape)
                else:
                    shape = tuple(np.array(raw_shape).tolist())
                dtype_str = str(data.get("dtype", "float32"))

                for prefix in ["torch.", "tensorflow.", "tf.", "mindspore.", "ms.", "np.", "numpy."]:
                    if dtype_str.startswith(prefix):
                        dtype_str = dtype_str[len(prefix):]

                dtype_map = {
                    "float32": np.float32, "float64": np.float64,
                    "float16": np.float16, "float": np.float32,
                    "int32": np.int32, "int64": np.int64,
                    "int16": np.int16, "int8": np.int8,
                    "uint8": np.uint8, "bool": np.bool_,
                    "complex64": np.complex64, "complex128": np.complex128,
                    "bfloat16": np.float32,
                }
                np_dtype = dtype_map.get(dtype_str, np.float32)

                if any(s == 0 for s in shape):
                    return np.empty(shape, dtype=np_dtype)

                data_range = data.get("range", None)

                if np_dtype == np.bool_:
                    return np.asarray(np.random.choice([True, False], size=shape), dtype=np.bool_)
                elif np.issubdtype(np_dtype, np.integer):
                    low = int(data_range[0]) if data_range else 0
                    high = int(data_range[1]) if data_range else 10
                    return np.asarray(np.random.randint(low, high, size=shape), dtype=np_dtype)
                elif np.issubdtype(np_dtype, np.complexfloating):
                    real = np.asarray(np.random.randn(*shape), dtype=np.float32)
                    imag = np.asarray(np.random.randn(*shape), dtype=np.float32)
                    return np.asarray(real + 1j * imag, dtype=np_dtype)
                else:
                    if data_range:
                        low, high = float(data_range[0]), float(data_range[1])
                        return np.asarray(np.random.uniform(low, high, size=shape), dtype=np_dtype)
                    else:
                        return np.asarray(np.random.randn(*shape), dtype=np_dtype)
            else:
                return np.array(list(data.values()))

        elif isinstance(data, (int, float)):
            return np.array(data)
        elif isinstance(data, list):
            return np.array(data)
        else:
            return np.array(data)

    def convert_to_ms_tensor(self, data: Any, numpy_data: np.ndarray = None):
        """转换为 MindSpore 张量"""
        ms = get_mindspore()
        if numpy_data is not None:
            return ms.Tensor(numpy_data)
        if isinstance(data, dict):
            np_data = self.generate_numpy_data(data)
            return ms.Tensor(np_data)
        if isinstance(data, (int, float)):
            return ms.Tensor(data)
        if isinstance(data, list):
            return ms.Tensor(np.array(data))
        return ms.Tensor(data)

    def convert_to_tf_tensor(self, data: Any, numpy_data: np.ndarray = None):
        """转换为 TensorFlow 张量"""
        if numpy_data is not None:
            return tf.convert_to_tensor(numpy_data.copy())
        if isinstance(data, dict):
            np_data = self.generate_numpy_data(data)
            return tf.convert_to_tensor(np_data.copy())
        if isinstance(data, (int, float)):
            return tf.convert_to_tensor(data)
        if isinstance(data, list):
            return tf.convert_to_tensor(data)
        return tf.convert_to_tensor(data)

    # ==================== 参数准备 ====================

    def should_skip_param(self, key: str, api_name: str, framework: str) -> bool:
        """判断是否应跳过某个参数"""
        common_skip = {"description", "api", "init_params", "is_class_api"}
        if key in common_skip:
            return True

        tf_skip = {"layout", "requires_grad", "out", "memory_format", "pin_memory", "device"}
        if framework == "tf" and key in tf_skip:
            return True

        ms_skip = {"name"}
        if framework == "ms" and key in ms_skip:
            return True

        return False

    def prepare_arguments(
        self, test_case: Dict[str, Any], framework: str = "ms"
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        为指定框架准备参数

        Args:
            test_case: 测试用例（包含共享的 numpy 数据）
            framework: "ms" 或 "tf"

        Returns:
            (args, kwargs)
        """
        args = []
        kwargs = {}

        ms = get_mindspore() if framework == "ms" else None
        tf_module = get_tensorflow() if framework == "tf" else None

        def convert_value(value: Any) -> Any:
            if isinstance(value, dict):
                if "shape" in value:
                    np_data = self.generate_numpy_data(value)
                    if framework == "tf":
                        return tf.convert_to_tensor(np_data.copy())
                    return ms.Tensor(np_data)
                return {k: convert_value(v) for k, v in value.items()}
            if isinstance(value, np.ndarray):
                if framework == "tf":
                    return tf.convert_to_tensor(value.copy())
                return ms.Tensor(value)
            if isinstance(value, list):
                return [convert_value(v) for v in value]
            if isinstance(value, tuple):
                return tuple(convert_value(v) for v in value)
            return value

        def normalize_dtype(dtype_value: Any) -> Any:
            if not isinstance(dtype_value, str):
                return dtype_value
            token = dtype_value.strip()
            for prefix in ["torch.", "tensorflow.", "tf.", "mindspore.", "ms.", "np.", "numpy."]:
                if token.startswith(prefix):
                    token = token[len(prefix):]
            if framework == "tf":
                return getattr(tf_module, token, dtype_value)
            return getattr(ms, token, dtype_value)

        explicit_args = test_case.get("args")
        explicit_kwargs = test_case.get("kwargs")
        if isinstance(explicit_args, list) or isinstance(explicit_kwargs, dict):
            if isinstance(explicit_args, list):
                args = [convert_value(item) for item in explicit_args]
            if isinstance(explicit_kwargs, dict):
                for key, value in explicit_kwargs.items():
                    if self.should_skip_param(key, test_case.get("api", ""), framework):
                        continue
                    if key == "dtype":
                        kwargs[key] = normalize_dtype(value)
                    else:
                        kwargs[key] = convert_value(value)
            return args, kwargs

        positional_params = [
            "inputs", "x", "input", "condition", "y", "other", "a", "b",
            "start", "end", "step", "stop",
        ]

        # 可变长参数处理
        varargs_key = None
        for key in test_case.keys():
            if key.startswith("*"):
                varargs_key = key
                break

        if varargs_key:
            varargs_data = test_case[varargs_key]
            if isinstance(varargs_data, list):
                for item in varargs_data:
                    if isinstance(item, dict) and "shape" in item:
                        np_data = self.generate_numpy_data(item)
                        if framework == "tf":
                            args.append(tf.convert_to_tensor(np_data.copy()))
                        else:
                            args.append(ms.Tensor(np_data))
                    else:
                        args.append(item)
            return args, kwargs

        # 按顺序处理位置参数
        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if param_name == "dtype":
                    args.append(normalize_dtype(value))
                else:
                    args.append(convert_value(value))

        # 处理关键字参数
        for key, value in test_case.items():
            if (
                key in positional_params
                or key in ("api", "args", "kwargs")
                or self.should_skip_param(key, test_case.get("api", ""), framework)
            ):
                continue
            if key.startswith("*"):
                continue

            if key == "dtype":
                kwargs[key] = normalize_dtype(value)
            else:
                kwargs[key] = convert_value(value)

        return args, kwargs

    # ==================== 结果比较 ====================

    def compare_tensors(
        self, ms_result, tf_result, tolerance: float = 1e-5
    ) -> Tuple[bool, str]:
        """比较 MindSpore 和 TensorFlow 的计算结果"""
        try:
            def to_numpy(value: Any):
                if hasattr(value, "asnumpy"):
                    return value.asnumpy()
                if tf.is_tensor(value):
                    return value.numpy()
                if isinstance(value, np.ndarray):
                    return value
                if isinstance(value, (list, tuple)):
                    return [to_numpy(item) for item in value]
                if isinstance(value, dict):
                    return {key: to_numpy(item) for key, item in value.items()}
                return np.array(value)

            def compare_value(left: Any, right: Any, prefix: str = "") -> Tuple[bool, str]:
                if isinstance(left, list) and isinstance(right, list):
                    if len(left) != len(right):
                        return False, f"{prefix}长度不匹配: {len(left)} vs {len(right)}"
                    for index, (left_item, right_item) in enumerate(zip(left, right)):
                        ok, msg = compare_value(left_item, right_item, f"{prefix}[{index}]")
                        if not ok:
                            return ok, msg
                    return True, "结果一致（列表逐项一致）"

                if isinstance(left, dict) and isinstance(right, dict):
                    if set(left.keys()) != set(right.keys()):
                        return False, f"{prefix}字典键不匹配"
                    for key in left.keys():
                        ok, msg = compare_value(left[key], right[key], f"{prefix}.{key}" if prefix else str(key))
                        if not ok:
                            return ok, msg
                    return True, "结果一致（字典逐项一致）"

                left_np = np.array(left)
                right_np = np.array(right)

                if left_np.shape != right_np.shape:
                    return False, f"{prefix}形状不匹配: MS={left_np.shape} vs TF={right_np.shape}"

                if left_np.dtype == np.bool_ or right_np.dtype == np.bool_:
                    match = np.array_equal(left_np, right_np)
                    if match:
                        return True, "布尔结果完全一致"
                    diff_count = np.sum(left_np != right_np)
                    return False, f"{prefix}布尔结果不一致，差异元素数: {diff_count}"

                if np.allclose(left_np, right_np, atol=tolerance, rtol=tolerance, equal_nan=True):
                    return True, "结果一致（在容差范围内）"

                max_diff = np.max(
                    np.abs(left_np.astype(np.float64) - right_np.astype(np.float64))
                )
                return False, f"{prefix}结果不一致，最大差异: {max_diff:.8f}"

            ms_np = to_numpy(ms_result)
            tf_np = to_numpy(tf_result)
            return compare_value(ms_np, tf_np)

        except Exception as e:
            return False, f"比较异常: {str(e)}"

    # ==================== 测试执行 ====================

    def execute_test_case(
        self,
        ms_api: str,
        tensorflow_api: str,
        ms_test_case: Dict[str, Any],
        tensorflow_test_case: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """执行单个测试用例"""
        def pick_first(case: Dict[str, Any], *keys: str):
            for key in keys:
                if isinstance(case, dict) and key in case and case[key] is not None:
                    return case[key]
            return None

        if tensorflow_test_case is None:
            tensorflow_test_case = ms_test_case

        effective_ms_api = ms_test_case.get("api", ms_api) if isinstance(ms_test_case, dict) else ms_api
        effective_tf_api = (
            tensorflow_test_case.get("api", tensorflow_api)
            if isinstance(tensorflow_test_case, dict)
            else tensorflow_api
        )

        result = {
            "ms_api": effective_ms_api,
            "tensorflow_api": effective_tf_api,
            "ms_success": False,
            "tensorflow_success": False,
            "results_match": False,
            "ms_error": None,
            "tensorflow_error": None,
            "comparison_error": None,
            "ms_shape": None,
            "tensorflow_shape": None,
            "ms_dtype": None,
            "tensorflow_dtype": None,
            "status": "unknown",
        }

        # 统一生成共享张量
        ms_test_case, tensorflow_test_case = self._materialize_shared_tensors(
            effective_ms_api, effective_tf_api, ms_test_case, tensorflow_test_case
        )

        is_class_ms = self.is_class_based_api(effective_ms_api)
        is_class_tf = self.is_class_based_api(effective_tf_api)

        # ---- 执行 MindSpore ----
        ms_result = None
        try:
            ms_func = self.get_ms_function(effective_ms_api)
            if ms_func is None:
                raise AttributeError(f"无法找到 MS API: {effective_ms_api}")

            if is_class_ms:
                init_kwargs = {
                    k: v for k, v in ms_test_case.items()
                    if k not in ["api", "input", "x", "init_params"]
                    and not isinstance(v, np.ndarray)
                    and not (isinstance(v, dict) and "shape" in v)
                }
                if isinstance(ms_test_case.get("init_params"), dict):
                    init_kwargs.update(ms_test_case["init_params"])
                op_instance = ms_func(**init_kwargs)

                # 获取输入
                input_data = pick_first(ms_test_case, "input", "x")
                if input_data is not None:
                    if isinstance(input_data, dict) and "shape" in input_data:
                        np_data = self.generate_numpy_data(input_data)
                        ms_input = self.convert_to_ms_tensor(None, np_data)
                    elif isinstance(input_data, np.ndarray):
                        ms_input = self.convert_to_ms_tensor(input_data)
                    else:
                        ms_input = self.convert_to_ms_tensor(input_data)
                    ms_result = op_instance(ms_input)
                else:
                    ms_result = op_instance(self.convert_to_ms_tensor(np.random.randn(2, 3).astype(np.float32)))
            else:
                # 函数式 API / Tensor 方法
                if "Tensor" in effective_ms_api:
                    # Tensor 方法：tensor.method(args)
                    method_name = effective_ms_api.split(".")[-1]
                    input_data = pick_first(ms_test_case, "x", "input")
                    if isinstance(input_data, np.ndarray):
                        ms_tensor = self.convert_to_ms_tensor(input_data)
                    else:
                        input_desc = input_data if input_data is not None else {"shape": [2, 3], "dtype": "float32"}
                        np_data = self.generate_numpy_data(input_desc)
                        ms_tensor = self.convert_to_ms_tensor(np_data)

                    method = getattr(ms_tensor, method_name)
                    # 获取除了 x/input 之外的参数
                    other_args = []
                    other_kwargs = {}
                    for key, value in ms_test_case.items():
                        if key in {"x", "input", "api", "init_params", "is_class_api", "description"}:
                            continue
                        if isinstance(value, np.ndarray):
                            other_args.append(self.convert_to_ms_tensor(value))
                        elif isinstance(value, dict) and "shape" in value:
                            np_data = self.generate_numpy_data(value)
                            other_args.append(self.convert_to_ms_tensor(np_data))
                        elif key in {"y", "other", "b"}:
                            other_args.append(value)
                        else:
                            other_kwargs[key] = value
                    ms_result = method(*other_args, **other_kwargs)
                else:
                    ms_args, ms_kwargs = self.prepare_arguments(ms_test_case, "ms")
                    ms_result = ms_func(*ms_args, **ms_kwargs)

            result["ms_success"] = True
            if hasattr(ms_result, "shape"):
                result["ms_shape"] = list(ms_result.shape)
            if hasattr(ms_result, "dtype"):
                result["ms_dtype"] = str(ms_result.dtype)

        except Exception as e:
            result["ms_error"] = f"{type(e).__name__}: {str(e)}"

        # ---- 执行 TensorFlow ----
        tf_result = None
        try:
            tf_func = self.get_tf_function(effective_tf_api)
            if tf_func is None:
                raise AttributeError(f"无法找到 TF API: {effective_tf_api}")

            if is_class_tf:
                init_kwargs = {
                    k: v for k, v in tensorflow_test_case.items()
                    if k not in ["api", "input", "x", "init_params"]
                    and not isinstance(v, np.ndarray)
                    and not (isinstance(v, dict) and "shape" in v)
                }
                if isinstance(tensorflow_test_case.get("init_params"), dict):
                    init_kwargs.update(tensorflow_test_case["init_params"])
                layer_or_op = tf_func(**init_kwargs)

                input_data = pick_first(tensorflow_test_case, "input", "x")
                if input_data is not None:
                    if isinstance(input_data, dict) and "shape" in input_data:
                        np_data = self.generate_numpy_data(input_data)
                        tf_input = self.convert_to_tf_tensor(None, np_data)
                    elif isinstance(input_data, np.ndarray):
                        tf_input = self.convert_to_tf_tensor(input_data)
                    else:
                        tf_input = self.convert_to_tf_tensor(input_data)
                    tf_result = layer_or_op(tf_input)
                else:
                    tf_result = layer_or_op(tf.random.normal([2, 3], dtype=tf.float32))
            else:
                if "Tensor" in effective_tf_api:
                    method_name = effective_tf_api.split(".")[-1]
                    input_data = pick_first(tensorflow_test_case, "x", "input")
                    if isinstance(input_data, np.ndarray):
                        tf_tensor = tf.convert_to_tensor(input_data.copy())
                    else:
                        input_desc = input_data if input_data is not None else {"shape": [2, 3], "dtype": "float32"}
                        np_data = self.generate_numpy_data(input_desc)
                        tf_tensor = tf.convert_to_tensor(np_data.copy())

                    method = getattr(tf_tensor, method_name)
                    other_args = []
                    other_kwargs = {}
                    for key, value in tensorflow_test_case.items():
                        if key in {"x", "input", "api", "init_params", "is_class_api", "description"}:
                            continue
                        if isinstance(value, np.ndarray):
                            other_args.append(tf.convert_to_tensor(value.copy()))
                        elif isinstance(value, dict) and "shape" in value:
                            np_data = self.generate_numpy_data(value)
                            other_args.append(tf.convert_to_tensor(np_data.copy()))
                        elif key in {"y", "other", "b"}:
                            other_args.append(value)
                        else:
                            other_kwargs[key] = value
                    tf_result = method(*other_args, **other_kwargs)
                else:
                    tf_args, tf_kwargs = self.prepare_arguments(tensorflow_test_case, "tf")
                    tf_result = tf_func(*tf_args, **tf_kwargs)

            if not result["tensorflow_success"]:
                result["tensorflow_success"] = True
                if hasattr(tf_result, "shape"):
                    result["tensorflow_shape"] = list(tf_result.shape)
                if hasattr(tf_result, "dtype"):
                    result["tensorflow_dtype"] = str(tf_result.dtype)

        except Exception as e:
            result["tensorflow_error"] = f"{type(e).__name__}: {str(e)}"

        # ---- 比较结果 ----
        if result["ms_success"] and result["tensorflow_success"]:
            try:
                match, detail = self.compare_tensors(ms_result, tf_result)
                result["results_match"] = match
                result["comparison_error"] = None if match else detail
                result["status"] = "consistent" if match else "inconsistent"
            except Exception as e:
                result["comparison_error"] = str(e)
                result["status"] = "comparison_error"
        elif result["ms_success"] and not result["tensorflow_success"]:
            result["status"] = "tensorflow_error"
        elif not result["ms_success"] and result["tensorflow_success"]:
            result["status"] = "mindspore_error"
        else:
            result["status"] = "both_error"

        return result

    def _execute_test_case_sequential(
        self, ms_api, tensorflow_api, ms_test_case, tensorflow_test_case=None
    ) -> Dict[str, Any]:
        """通过锁保证执行不并发"""
        with self.execution_lock:
            return self.execute_test_case(ms_api, tensorflow_api, ms_test_case, tensorflow_test_case)

    def _materialize_shared_tensors(
        self,
        ms_api: str,
        tensorflow_api: str,
        ms_test_case: Dict[str, Any],
        tensorflow_test_case: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """统一生成共享张量，保证两框架输入数值一致"""
        ms_case = copy.deepcopy(ms_test_case)
        tf_case = copy.deepcopy(tensorflow_test_case)

        is_class_ms = self.is_class_based_api(ms_api)
        is_class_tf = self.is_class_based_api(tensorflow_api)
        if (is_class_ms or is_class_tf) and not (
            "input" in ms_case or "x" in ms_case
            or "input" in tf_case or "x" in tf_case
        ):
            default_desc = self._default_input_desc_for_class(ms_api or tensorflow_api)
            ms_case.setdefault("input", default_desc)
            tf_case.setdefault("input", default_desc)

        def is_tensor_desc(value: Any) -> bool:
            return isinstance(value, dict) and "shape" in value

        def clone_array(value: np.ndarray) -> np.ndarray:
            return value.copy()

        def materialize_pair(ms_val: Any, tf_val: Any) -> Tuple[Any, Any]:
            if isinstance(ms_val, np.ndarray):
                return clone_array(ms_val), clone_array(ms_val)
            if isinstance(tf_val, np.ndarray):
                return clone_array(tf_val), clone_array(tf_val)

            if is_tensor_desc(ms_val) or is_tensor_desc(tf_val):
                tensor_desc = ms_val if is_tensor_desc(ms_val) else tf_val
                shared = self.generate_numpy_data(tensor_desc)
                return clone_array(shared), clone_array(shared)

            if isinstance(ms_val, list) or isinstance(tf_val, list):
                ms_list = ms_val if isinstance(ms_val, list) else []
                tf_list = tf_val if isinstance(tf_val, list) else []
                size = max(len(ms_list), len(tf_list))
                out_ms = []
                out_tf = []
                for index in range(size):
                    left = ms_list[index] if index < len(ms_list) else None
                    right = tf_list[index] if index < len(tf_list) else None
                    new_left, new_right = materialize_pair(left, right)
                    if index < len(ms_list):
                        out_ms.append(new_left)
                    if index < len(tf_list):
                        out_tf.append(new_right)
                return (
                    out_ms if isinstance(ms_val, list) else ms_val,
                    out_tf if isinstance(tf_val, list) else tf_val,
                )

            if isinstance(ms_val, dict) or isinstance(tf_val, dict):
                ms_dict = ms_val if isinstance(ms_val, dict) else {}
                tf_dict = tf_val if isinstance(tf_val, dict) else {}
                keys = set(ms_dict.keys()) | set(tf_dict.keys())
                out_ms = {}
                out_tf = {}
                for key in keys:
                    if key == "api":
                        if key in ms_dict:
                            out_ms[key] = ms_dict[key]
                        if key in tf_dict:
                            out_tf[key] = tf_dict[key]
                        continue
                    new_left, new_right = materialize_pair(ms_dict.get(key), tf_dict.get(key))
                    if key in ms_dict:
                        out_ms[key] = new_left
                    if key in tf_dict:
                        out_tf[key] = new_right
                return (
                    out_ms if isinstance(ms_val, dict) else ms_val,
                    out_tf if isinstance(tf_val, dict) else tf_val,
                )

            return ms_val, tf_val

        ms_case, tf_case = materialize_pair(ms_case, tf_case)

        if isinstance(ms_case, dict) and isinstance(tf_case, dict):
            alias_pairs = [
                ("x", "input"),
                ("input", "x"),
                ("y", "other"),
                ("other", "y"),
            ]
            for ms_key, tf_key in alias_pairs:
                if ms_key in ms_case and tf_key in tf_case:
                    ms_item, tf_item = materialize_pair(ms_case[ms_key], tf_case[tf_key])
                    ms_case[ms_key] = ms_item
                    tf_case[tf_key] = tf_item

        return ms_case, tf_case

    def _default_input_desc_for_class(self, api_name: str) -> Dict[str, Any]:
        name = (api_name or "").lower()
        if "3d" in name:
            return {"shape": [2, 3, 4, 4, 4], "dtype": "float32"}
        if "2d" in name:
            return {"shape": [2, 3, 8, 8], "dtype": "float32"}
        if "1d" in name:
            return {"shape": [2, 3, 10], "dtype": "float32"}
        return {"shape": [2, 3], "dtype": "float32"}

    # ==================== API 文档爬取 ====================

    def _fetch_api_docs(self, ms_api: str, tensorflow_api: str) -> Tuple[str, str]:
        MIN_DOC_LENGTH = 300
        ms_doc = ""
        tensorflow_doc = ""

        try:
            raw = get_doc_content(ms_api, "mindspore")
            if raw and len(raw) >= MIN_DOC_LENGTH:
                ms_doc = raw[:3000]
                self._safe_print(f"    📄 MS文档: {len(ms_doc)} 字符")
            else:
                self._safe_print(f"    📄 MS文档: 未获取到有效内容")
        except Exception as e:
            self._safe_print(f"    ⚠️ MS文档爬取失败: {str(e)[:50]}")

        try:
            raw = get_doc_content(tensorflow_api, "tensorflow")
            if raw and len(raw) >= MIN_DOC_LENGTH:
                tensorflow_doc = raw[:3000]
                self._safe_print(f"    📄 TF文档: {len(tensorflow_doc)} 字符")
            else:
                self._safe_print(f"    📄 TF文档: 未获取到有效内容")
        except Exception as e:
            self._safe_print(f"    ⚠️ TF文档爬取失败: {str(e)[:50]}")

        return ms_doc, tensorflow_doc

    # ==================== LLM 交互 ====================

    def _build_llm_prompt(
        self,
        execution_result: Dict[str, Any],
        ms_test_case: Dict[str, Any],
        tensorflow_test_case: Dict[str, Any],
        ms_doc: str = "",
        tensorflow_doc: str = "",
    ) -> str:
        ms_api = execution_result.get("ms_api", "")
        tensorflow_api = execution_result.get("tensorflow_api", "")
        status = execution_result.get("status", "")
        ms_success = execution_result.get("ms_success", False)
        tensorflow_success = execution_result.get("tensorflow_success", False)
        results_match = execution_result.get("results_match", False)
        ms_error = execution_result.get("ms_error", "")
        tensorflow_error = execution_result.get("tensorflow_error", "")
        comparison_error = execution_result.get("comparison_error", "")

        # 简化测试用例
        def simplify_case(case):
            simplified = {}
            for key, value in case.items():
                if isinstance(value, np.ndarray):
                    simplified[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
                else:
                    simplified[key] = value
            return simplified

        simplified_ms = simplify_case(ms_test_case)
        simplified_tf = simplify_case(tensorflow_test_case)

        # 文档部分
        doc_section = ""
        if ms_doc or tensorflow_doc:
            doc_section = "\n## 官方API文档参考\n\n"
            if ms_doc:
                doc_section += f"### MindSpore {ms_api} 文档\n```\n{ms_doc}\n```\n\n"
            if tensorflow_doc:
                doc_section += f"### TensorFlow {tensorflow_api} 文档\n```\n{tensorflow_doc}\n```\n\n"

        # 参数示例
        def build_param_str(simplified):
            examples = []
            for key, value in simplified.items():
                if key in {"api", "init_params", "is_class_api", "description"}:
                    continue
                examples.append(f'    "{key}": {json.dumps(value, ensure_ascii=False)}')
            return ",\n".join(examples) if examples else '    "x": {"shape": [2, 3], "dtype": "float32"}'

        ms_param_str = build_param_str(simplified_ms)
        tf_param_str = build_param_str(simplified_tf)

        prompt = f"""请分析以下算子测试用例在MindSpore和TensorFlow框架中的执行结果，并根据结果进行测试用例的修复或变异（fuzzing）。

## 测试信息
- **MindSpore API**: {ms_api}
- **TensorFlow API**: {tensorflow_api}
{doc_section}
## 执行结果
- **执行状态**: {status}
- **MindSpore执行成功**: {ms_success}
- **TensorFlow执行成功**: {tensorflow_success}
- **结果是否一致**: {results_match}

## 错误信息
- **MindSpore错误**: {ms_error if ms_error else "无"}
- **TensorFlow错误**: {tensorflow_error if tensorflow_error else "无"}
- **比较错误**: {comparison_error if comparison_error else "无"}

## 原始测试用例

### MindSpore测试用例
```json
{json.dumps(simplified_ms, indent=2, ensure_ascii=False)}
```

### TensorFlow测试用例
```json
{json.dumps(simplified_tf, indent=2, ensure_ascii=False)}
```

## 任务要求
请根据以上信息（包括官方API文档），自主判断两框架的比较结果是**一致**、**不一致**还是**执行出错**，并执行以下操作：

1. **如果一致**：对用例进行**变异（fuzzing）**，例如修改输入张量形状、参数值，优先探索极端值和边界值
2. **如果执行出错**：结合报错和文档进行**修复**（调整参数名/类型/取值范围等）或**跳过**（文档缺失、算子已移除、或两算子语义不等价）
3. **如果不一致**：先判断是否为可容忍精度误差（1e-3及以下）；可容忍则**变异**，语义不等价则**跳过**，否则按用例构造问题进行**修复**

## MindSpore API 调用说明
- MindSpore Primitive 算子（如 mindspore.ops.Abs）需要先实例化再调用：`op = ops.Abs(); result = op(input)`
- MindSpore 函数式 API（如 mindspore.ops.abs）直接调用：`result = ops.abs(input)`
- MindSpore NN 层（如 mindspore.nn.Conv2d）：`layer = nn.Conv2d(...); result = layer(input)`
- TensorFlow Keras 层（如 tf.keras.layers.Conv2D）：`layer = tf.keras.layers.Conv2D(...); result = layer(input)`
- TensorFlow 函数式 API（如 tf.math.abs / tf.nn.relu）直接调用
- 注意 MindSpore 默认常见 NCHW、TensorFlow 默认常见 NHWC，必要时显式设置 data_format

## 输出格式要求
请严格按照以下JSON格式输出，不要包含任何其他文字：

{{
  "operation": "mutation",
  "reason": "进行该操作的详细原因（不超过150字）",
  "mindspore_test_case": {{
    "api": "{ms_api}",
{ms_param_str}
  }},
    "tensorflow_test_case": {{
        "api": "{tensorflow_api}",
{tf_param_str}
  }}
}}

**重要说明**：
1. operation 的值必须是 "mutation"、"repair" 或 "skip" 之一
2. 张量参数必须使用 {{"shape": [...], "dtype": "..."}} 格式
3. 标量参数直接使用数值
4. 构造两个框架的用例时必须保证输入相同、参数在语义上严格对应
5. MindSpore 和 TensorFlow 的测试用例允许参数名/参数值/参数数量差异，只要理论输出一致
6. 如果该算子找不到官方文档或文档显示它已从当前版本移除，请将 operation 设为 "skip"，不需要尝试修复
7. 测试用例变异时可优先探索极端情况：空张量、单元素张量、高维张量、不同数据类型、边界值等
8. 请仔细阅读官方API文档，确保参数名称、类型、取值范围与文档一致
9. 注意两个框架的默认数据布局语义，必要时显式设置避免格式歧义
"""
        return prompt

    def call_llm_for_repair_or_mutation(
        self,
        execution_result: Dict[str, Any],
        ms_test_case: Dict[str, Any],
        tensorflow_test_case: Dict[str, Any],
        ms_doc: str = "",
        tensorflow_doc: str = "",
    ) -> Dict[str, Any]:
        """调用 LLM 进行测试用例修复或变异"""
        prompt = self._build_llm_prompt(
            execution_result, ms_test_case, tensorflow_test_case, ms_doc, tensorflow_doc
        )
        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个深度学习框架测试专家，精通MindSpore和TensorFlow框架的API差异。"
                            "你的任务是根据测试用例的执行结果，判断是否需要修复或变异测试用例，"
                            "并返回严格的JSON格式结果。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            raw_response = completion.choices[0].message.content.strip()
            time.sleep(1)

            try:
                return json.loads(raw_response)
            except json.JSONDecodeError:
                self._safe_print(f"    ⚠️ LLM返回不是有效JSON，尝试提取...")
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return {
                    "operation": "skip",
                    "reason": "LLM返回格式错误",
                    "mindspore_test_case": ms_test_case,
                    "tensorflow_test_case": tensorflow_test_case,
                }

        except Exception as e:
            self._safe_print(f"    ❌ 调用LLM失败: {e}")
            return {
                "operation": "skip",
                "reason": f"LLM调用失败: {e}",
                "mindspore_test_case": ms_test_case,
                "tensorflow_test_case": tensorflow_test_case,
            }

    # ==================== 核心测试循环 ====================

    def llm_enhanced_test_operator(
        self,
        ms_api: str,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        num_test_cases: int = None,
        num_workers: int = DEFAULT_WORKERS,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """使用 LLM 增强的方式测试单个算子对"""
        self._safe_print(f"\n{'=' * 80}")
        self._safe_print(f"🎯 开始测试算子: {ms_api}")
        self._safe_print(f"🔄 每个用例最大迭代次数: {max_iterations}")
        self._safe_print(f"{'=' * 80}\n")

        stats = {"llm_generated_cases": 0, "successful_cases": 0}

        _, tensorflow_api, mapping_method = self.convert_api_name(ms_api)
        if tensorflow_api is None:
            self._safe_print(f"❌ {ms_api} 无 TensorFlow 对应实现")
            return [], stats

        self._safe_print(f"✅ MindSpore API: {ms_api}")
        self._safe_print(f"✅ TensorFlow API: {tensorflow_api}")
        self._safe_print(f"✅ 映射方法: {mapping_method}")

        api_data = self.test_cases_data.get(ms_api, {})
        test_cases = api_data.get("test_cases", [])

        if not test_cases:
            self._safe_print(f"⚠️ 未找到 {ms_api} 的测试用例，使用默认用例")
            test_cases = [
                {"description": "默认", "inputs": {"x": {"shape": [2, 3], "dtype": "float32"}}}
            ]

        if num_test_cases is None:
            num_test_cases = len(test_cases)
        else:
            num_test_cases = min(num_test_cases, len(test_cases))

        self._safe_print(f"📋 将测试 {num_test_cases} 个用例 (LLM并发={num_workers}, 执行顺序)")

        # 准备初始用例
        initial_cases = []
        for case_idx in range(num_test_cases):
            tc = test_cases[case_idx]
            if "inputs" in tc:
                flat_case = dict(tc["inputs"])
            else:
                flat_case = {k: v for k, v in tc.items() if k != "description"}
            flat_case["api"] = ms_api
            # 保留 init_params
            if "init_params" in api_data:
                flat_case["init_params"] = api_data["init_params"]
            elif "init_params" in tc:
                flat_case["init_params"] = tc["init_params"]
            initial_cases.append((case_idx + 1, flat_case))

        all_results = []

        # 用例级别的多轮迭代测试
        # 使用 ThreadPoolExecutor 并发 LLM 调用，执行顺序串行
        if num_workers <= 1:
            for case_number, initial_test_case in initial_cases:
                self._safe_print(f"\n📋 用例 {case_number}/{num_test_cases}")
                case_results = self._test_single_case_with_iterations(
                    ms_api, tensorflow_api, initial_test_case,
                    max_iterations, case_number, stats,
                )
                all_results.extend(case_results)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_case = {}
                for case_number, initial_test_case in initial_cases:
                    future = executor.submit(
                        self._test_single_case_with_iterations,
                        ms_api, tensorflow_api, initial_test_case,
                        max_iterations, case_number, stats,
                    )
                    future_to_case[future] = case_number

                for future in as_completed(future_to_case):
                    case_results = future.result()
                    all_results.extend(case_results)

        all_results.sort(key=lambda r: (r.get("case_number", 0), r.get("iteration", 0)))

        self._safe_print(f"\n{'=' * 80}")
        self._safe_print("✅ 所有测试完成")
        self._safe_print(
            f"📊 共测试 {num_test_cases} 个用例，总计 {len(all_results)} 次迭代"
        )
        self._safe_print(f"📊 LLM生成的测试用例数: {stats['llm_generated_cases']}")
        self._safe_print(f"📊 两个框架都执行成功的用例数: {stats['successful_cases']}")
        self._safe_print(f"{'=' * 80}\n")

        return all_results, stats

    def _test_single_case_with_iterations(
        self,
        ms_api: str,
        tensorflow_api: str,
        initial_test_case: Dict[str, Any],
        max_iterations: int,
        case_number: int,
        stats: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """对单个测试用例进行多轮迭代测试"""
        case_results = []

        current_ms_test_case = copy.deepcopy(initial_test_case)
        current_ms_test_case["api"] = ms_api

        current_tf_test_case = copy.deepcopy(initial_test_case)
        current_tf_test_case["api"] = tensorflow_api

        is_llm_generated = False

        self._safe_print(f"  📖 预先爬取API文档...")
        ms_doc, tensorflow_doc = self._fetch_api_docs(ms_api, tensorflow_api)

        for iteration in range(max_iterations):
            source_type = "LLM" if is_llm_generated else "文件"
            self._safe_print(
                f"  🔄 迭代 {iteration + 1}/{max_iterations} ({source_type})", end=""
            )

            current_ms_api = current_ms_test_case.get("api", ms_api) or ms_api
            current_tf_api = current_tf_test_case.get("api", tensorflow_api) or tensorflow_api

            try:
                execution_result = self._execute_test_case_sequential(
                    current_ms_api, current_tf_api, current_ms_test_case, current_tf_test_case
                )

                ms_status = "✓" if execution_result["ms_success"] else "✗"
                tf_status = "✓" if execution_result["tensorflow_success"] else "✗"
                match_status = "✓" if execution_result["results_match"] else "✗"
                self._safe_print(f" | MS:{ms_status} TF:{tf_status} Match:{match_status}")

                if execution_result["ms_error"] and not execution_result["ms_success"]:
                    self._safe_print(
                        f"    ❌ MS错误: {str(execution_result['ms_error'])[:100]}..."
                    )
                if execution_result["tensorflow_error"] and not execution_result["tensorflow_success"]:
                    self._safe_print(
                        f"    ❌ TF错误: {str(execution_result['tensorflow_error'])[:100]}..."
                    )
                if execution_result["comparison_error"]:
                    self._safe_print(
                        f"    ⚠️ 比较: {str(execution_result['comparison_error'])[:100]}..."
                    )

                if is_llm_generated:
                    if execution_result["ms_success"] and execution_result["tensorflow_success"]:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

            except Exception as e:
                self._safe_print(f" | ❌ 严重错误: {str(e)[:80]}...")
                execution_result = {
                    "status": "fatal_error",
                    "ms_success": False, "tensorflow_success": False,
                    "results_match": False,
                    "ms_error": f"Fatal: {str(e)}", "tensorflow_error": None,
                    "comparison_error": None,
                }

            iteration_result = {
                "iteration": iteration + 1,
                "ms_test_case": current_ms_test_case,
                "tensorflow_test_case": current_tf_test_case,
                "execution_result": execution_result,
                "llm_operation": None,
                "is_llm_generated": is_llm_generated,
                "case_number": case_number,
            }

            # 调用 LLM
            try:
                llm_result = self.call_llm_for_repair_or_mutation(
                    execution_result, current_ms_test_case, current_tf_test_case,
                    ms_doc, tensorflow_doc,
                )
            except Exception as e:
                self._safe_print(f"    ❌ LLM调用失败: {str(e)[:80]}...")
                llm_result = {"operation": "skip", "reason": f"LLM调用失败: {str(e)}"}
                iteration_result["llm_operation"] = llm_result
                case_results.append(iteration_result)
                break

            operation = llm_result.get("operation", "skip")
            reason = llm_result.get("reason", "")
            self._safe_print(f"    🤖 LLM: {operation} - {reason[:80]}")

            iteration_result["llm_operation"] = {"operation": operation, "reason": reason}
            case_results.append(iteration_result)

            if operation == "skip":
                break

            if operation in ("mutation", "repair"):
                next_ms_case = llm_result.get("mindspore_test_case", current_ms_test_case)
                next_tf_case = llm_result.get("tensorflow_test_case", current_tf_test_case)
                with self.stats_lock:
                    stats["llm_generated_cases"] += 1
                is_llm_generated = True
            else:
                next_ms_case = current_ms_test_case
                next_tf_case = current_tf_test_case

            current_ms_test_case, current_tf_test_case = self._convert_llm_test_cases(
                next_ms_case, next_tf_case
            )

        # 最后一轮 LLM 生成了新用例但未执行的补充执行
        if case_results:
            last_op = case_results[-1].get("llm_operation", {}).get("operation", "skip")
            if last_op in ("mutation", "repair"):
                self._safe_print(f"  🔄 执行最终LLM用例", end="")
                try:
                    execution_result = self._execute_test_case_sequential(
                        current_ms_test_case.get("api", ms_api) or ms_api,
                        current_tf_test_case.get("api", tensorflow_api) or tensorflow_api,
                        current_ms_test_case,
                        current_tf_test_case,
                    )
                    ms_s = "✓" if execution_result["ms_success"] else "✗"
                    tf_s = "✓" if execution_result["tensorflow_success"] else "✗"
                    m_s = "✓" if execution_result["results_match"] else "✗"
                    self._safe_print(f" | MS:{ms_s} TF:{tf_s} Match:{m_s}")

                    if execution_result["ms_success"] and execution_result["tensorflow_success"]:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

                    case_results.append({
                        "iteration": len(case_results) + 1,
                        "ms_test_case": current_ms_test_case,
                        "tensorflow_test_case": current_tf_test_case,
                        "execution_result": execution_result,
                        "llm_operation": {
                            "operation": "final_execution",
                            "reason": "执行最后一次LLM生成的用例",
                        },
                        "case_number": case_number,
                        "is_llm_generated": True,
                    })
                except Exception as e:
                    self._safe_print(f"  ❌ 最终用例执行失败: {str(e)[:80]}...")

        self._safe_print(f"  ✅ 用例 {case_number} 完成，共 {len(case_results)} 次迭代")
        return case_results

    def _convert_llm_test_cases(
        self,
        ms_test_case: Dict[str, Any],
        tensorflow_test_case: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """将 LLM 返回的测试用例转换为可执行格式，确保共享张量"""
        ms_api = ms_test_case.get("api", "") if isinstance(ms_test_case, dict) else ""
        tf_api = tensorflow_test_case.get("api", "") if isinstance(tensorflow_test_case, dict) else ""
        return self._materialize_shared_tensors(ms_api, tf_api, ms_test_case, tensorflow_test_case)

    # ==================== 结果保存 ====================

    def save_results(
        self, ms_api: str, results: List[Dict[str, Any]], stats: Dict[str, int] = None
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = ms_api.replace(".", "_")
        filename = f"llm_enhanced_{safe_name}_{timestamp}.json"
        filepath = os.path.join(self.result_dir, filename)

        output_results = []
        for result in results:
            out = copy.deepcopy(result)
            for case_key in ["ms_test_case", "tensorflow_test_case"]:
                if case_key in out and isinstance(out[case_key], dict):
                    simplified = {}
                    for k, v in out[case_key].items():
                        if isinstance(v, np.ndarray):
                            simplified[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
                        else:
                            simplified[k] = v
                    out[case_key] = simplified
            output_results.append(out)

        output_data = {
            "ms_api": ms_api,
            "tensorflow_api": self.api_mapping.get(ms_api, ""),
            "timestamp": datetime.now().isoformat(),
            "total_iterations": len(results),
            "llm_generated_test_cases": stats.get("llm_generated_cases", 0) if stats else 0,
            "successful_test_cases": stats.get("successful_cases", 0) if stats else 0,
            "results": output_results,
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self._safe_print(f"💾 结果已保存到: {filepath}")

    def get_all_testable_apis(self) -> List[str]:
        testable = []
        for ms_api in sorted(self.test_cases_data.keys()):
            tf_api = self.api_mapping.get(ms_api, "无对应实现")
            if tf_api and tf_api != "无对应实现":
                testable.append(ms_api)
        return testable

    def close(self):
        pass


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="基于LLM的MindSpore与TensorFlow算子差分测试框架"
    )
    parser.add_argument(
        "--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"每个测试用例的最大迭代次数（默认{DEFAULT_MAX_ITERATIONS}）",
    )
    parser.add_argument(
        "--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES,
        help=f"每个算子要测试的用例数量（默认{DEFAULT_NUM_CASES}）",
    )
    parser.add_argument(
        "--start", type=int, default=1,
        help="起始算子索引（从1开始，默认1）",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="结束算子索引（包含，默认全部）",
    )
    parser.add_argument(
        "--operators", "-o", nargs="*",
        help="指定要测试的算子名称（如 mindspore.ops.Abs）",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"并发线程数（默认{DEFAULT_WORKERS}）",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"LLM模型名称（默认 {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--key-path", "-k", default=DEFAULT_KEY_PATH,
        help=f"API key文件路径（默认 {DEFAULT_KEY_PATH}）",
    )
    parser.add_argument(
        "--test-cases-file", default=DEFAULT_TEST_CASES_FILE,
        help="测试用例 JSON 文件路径",
    )
    parser.add_argument(
        "--mapping-file", default=DEFAULT_MAPPING_FILE,
        help="MS→TF 映射 CSV 文件路径",
    )

    args = parser.parse_args()
    num_workers = max(1, args.workers)

    print("=" * 80)
    print("基于LLM的MindSpore与TensorFlow算子差分测试框架")
    print("=" * 80)
    print(f"📌 每个算子的迭代次数: {args.max_iterations}")
    print(f"📌 每个算子的测试用例数: {args.num_cases}")
    print(f"📌 LLM并发线程数: {num_workers}")
    print(f"📌 LLM模型: {args.model}")
    print("=" * 80)

    comparator = LLMEnhancedComparator(
        test_cases_file=args.test_cases_file,
        mapping_file=args.mapping_file,
        key_path=args.key_path,
        model=args.model,
        llm_workers=num_workers,
    )

    start_time = time.time()
    start_datetime = datetime.now()

    try:
        all_testable = comparator.get_all_testable_apis()
        print(f"\n🔍 可测试的 MS API 总数: {len(all_testable)}")

        if args.operators:
            operator_names = args.operators
            print(f"📋 指定算子数: {len(operator_names)}")
        else:
            start_idx = max(1, args.start) - 1
            end_idx = args.end if args.end is not None else len(all_testable)
            end_idx = min(end_idx, len(all_testable))
            if start_idx >= end_idx:
                raise ValueError(f"起始索引 {args.start} 必须小于结束索引 {end_idx}")
            operator_names = all_testable[start_idx:end_idx]
            print(f"📌 测试范围: 第 {start_idx + 1} 到第 {end_idx} 个算子")
            print(f"📋 将测试 {len(operator_names)} 个算子")

        print(
            f"📋 算子列表: "
            f"{', '.join(operator_names[:10])}{'...' if len(operator_names) > 10 else ''}\n"
        )

        all_operators_summary = []

        batch_log_file = os.path.join(
            comparator.result_dir,
            f"batch_test_log_{start_datetime.strftime('%Y%m%d_%H%M%S')}.txt",
        )
        log_file = open(batch_log_file, 'w', encoding='utf-8')
        log_file.write("=" * 80 + "\n")
        log_file.write("MS→TF 差分测试批量日志\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"测试配置:\n")
        log_file.write(f"  - 迭代次数: {args.max_iterations}\n")
        log_file.write(f"  - 用例数: {args.num_cases}\n")
        log_file.write(f"  - 并发数: {num_workers}\n")
        log_file.write(f"  - 测试算子数: {len(operator_names)}\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.flush()

        for idx, ms_api in enumerate(operator_names, 1):
            print("\n" + "🔷" * 40)
            print(f"🎯 [{idx}/{len(operator_names)}] 开始测试算子: {ms_api}")
            print("🔷" * 40)

            try:
                results, stats = comparator.llm_enhanced_test_operator(
                    ms_api,
                    max_iterations=args.max_iterations,
                    num_test_cases=args.num_cases,
                    num_workers=num_workers,
                )

                if results:
                    comparator.save_results(ms_api, results, stats)
                    all_operators_summary.append({
                        "operator": ms_api,
                        "tensorflow_api": comparator.api_mapping.get(ms_api, ""),
                        "total_iterations": len(results),
                        "llm_generated_cases": stats.get("llm_generated_cases", 0),
                        "successful_cases": stats.get("successful_cases", 0),
                        "status": "completed",
                    })

                    print(f"\n✅ {ms_api} 测试完成")
                    print(f"   - 总迭代次数: {len(results)}")
                    print(f"   - LLM生成用例数: {stats.get('llm_generated_cases', 0)}")
                    print(f"   - 成功执行用例数: {stats.get('successful_cases', 0)}")

                    log_file.write(f"[{idx}/{len(operator_names)}] {ms_api}\n")
                    log_file.write(f"  状态: ✅ 完成\n")
                    log_file.write(f"  总迭代次数: {len(results)}\n")
                    log_file.write(f"  LLM生成用例数: {stats.get('llm_generated_cases', 0)}\n")
                    log_file.write(f"  成功执行用例数: {stats.get('successful_cases', 0)}\n\n")
                    if stats.get("llm_generated_cases", 0) > 0:
                        rate = stats.get("successful_cases", 0) / stats["llm_generated_cases"] * 100
                        log_file.write(f"  成功率: {rate:.2f}%\n")
                    log_file.write("\n")
                    log_file.flush()
                else:
                    all_operators_summary.append({
                        "operator": ms_api, "total_iterations": 0,
                        "llm_generated_cases": 0, "successful_cases": 0,
                        "status": "no_results",
                    })
                    log_file.write(f"[{idx}/{len(operator_names)}] {ms_api}\n")
                    log_file.write(f"  状态: ⚠️ 无结果\n\n")
                    log_file.flush()

            except Exception as e:
                print(f"\n❌ {ms_api} 测试失败: {e}")
                all_operators_summary.append({
                    "operator": ms_api, "total_iterations": 0,
                    "llm_generated_cases": 0, "successful_cases": 0,
                    "status": "failed", "error": str(e),
                })
                log_file.write(f"[{idx}/{len(operator_names)}] {ms_api}\n")
                log_file.write(f"  状态: ❌ 失败\n  错误: {str(e)}\n\n")
                log_file.flush()
                continue

        # ==================== 输出总结 ====================
        end_time = time.time()
        end_datetime = datetime.now()
        total_duration = end_time - start_time
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)

        completed_count = sum(1 for s in all_operators_summary if s["status"] == "completed")
        failed_count = sum(1 for s in all_operators_summary if s["status"] == "failed")
        no_results_count = sum(1 for s in all_operators_summary if s["status"] == "no_results")
        total_llm_cases = sum(s["llm_generated_cases"] for s in all_operators_summary)
        total_successful = sum(s["successful_cases"] for s in all_operators_summary)
        total_iterations = sum(s["total_iterations"] for s in all_operators_summary)

        print("\n" + "=" * 80)
        print("📊 批量测试总体摘要")
        print("=" * 80)
        print(f"总算子数: {len(operator_names)}")
        print(f"✅ 成功完成: {completed_count}")
        print(f"❌ 测试失败: {failed_count}")
        print(f"⚠️ 无结果: {no_results_count}")
        print(f"\n📈 统计数据:")
        print(f"   - LLM生成的测试用例总数: {total_llm_cases}")
        print(f"   - 成功执行的用例总数: {total_successful}")
        if total_llm_cases > 0:
            print(f"   - 成功执行占比: {total_successful / total_llm_cases * 100:.2f}%")
        print(f"   - 总迭代次数: {total_iterations}")
        print(f"\n⏱️ 运行时间: {hours}小时 {minutes}分钟 {seconds}秒")

        log_file.write("=" * 80 + "\n总体统计\n" + "=" * 80 + "\n")
        log_file.write(f"结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"总运行时间: {hours}h {minutes}m {seconds}s ({total_duration:.2f}s)\n\n")
        log_file.write(f"算子结果:\n")
        log_file.write(f"  - 总算子数: {len(operator_names)}\n")
        log_file.write(f"  - 成功: {completed_count}\n")
        log_file.write(f"  - 失败: {failed_count}\n")
        log_file.write(f"  - 无结果: {no_results_count}\n\n")
        log_file.write(f"LLM统计:\n")
        log_file.write(f"  - 生成用例数: {total_llm_cases}\n")
        log_file.write(f"  - 成功执行数: {total_successful}\n")
        if total_llm_cases > 0:
            log_file.write(f"  - 成功率: {total_successful / total_llm_cases * 100:.2f}%\n")
        log_file.write(f"  - 总迭代次数: {total_iterations}\n")
        log_file.close()

        print(f"\n💾 总日志已保存到: {batch_log_file}")

        # JSON 摘要
        summary_file = os.path.join(
            comparator.result_dir,
            f"batch_test_summary_{start_datetime.strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_config": {
                    "max_iterations": args.max_iterations,
                    "num_test_cases": args.num_cases,
                    "workers": num_workers,
                    "model": args.model,
                },
                "time_info": {
                    "start_time": start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    "end_time": end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    "total_duration_seconds": total_duration,
                    "duration_formatted": f"{hours}h {minutes}m {seconds}s",
                },
                "summary": {
                    "tested_operators": len(operator_names),
                    "completed": completed_count,
                    "failed": failed_count,
                    "no_results": no_results_count,
                    "total_llm_generated_cases": total_llm_cases,
                    "total_successful_cases": total_successful,
                    "success_rate": (
                        f"{total_successful / total_llm_cases * 100:.2f}%"
                        if total_llm_cases > 0 else "N/A"
                    ),
                    "total_iterations": total_iterations,
                },
                "operators": all_operators_summary,
            }, f, indent=2, ensure_ascii=False)

        print(f"💾 JSON摘要已保存到: {summary_file}")

    finally:
        comparator.close()
        print("\n✅ 批量测试程序执行完成")


if __name__ == "__main__":
    main()
