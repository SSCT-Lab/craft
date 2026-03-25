#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4: 基于 LLM 的 TensorFlow 与 PyTorch 算子差分测试框架

功能：
- 从 JSON 文件加载 TF 测试用例和 TF→PT 映射
- 对每对等价算子，执行 TF 和 PT 并比较结果
- 使用 LLM 进行测试用例的修复（repair）、变异（mutation）和跳过（skip）
- 支持并发测试多个用例
- 保存详细测试结果和批量日志

用法：
    conda activate tf_env
    python tf_pt_test/llm_enhanced_compare.py \
        [--max-iterations 3] [--num-cases 5] [--workers 6] \
        [--start 1] [--end N] [--operators tf.nn.relu tf.math.abs]

前置条件：
    1. 已运行 Step 1 extract_tf_apis.py
    2. 已运行 Step 2 extract_tf_test_cases.py
    3. 已运行 Step 3 extract_tf_pt_mapping.py
"""

import os
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

import torch
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

# 数据文件路径
DATA_DIR = os.path.join(ROOT_DIR, "tf_pt_test", "data")
DEFAULT_TEST_CASES_FILE = os.path.join(DATA_DIR, "tf_test_cases.json")
DEFAULT_MAPPING_FILE = os.path.join(DATA_DIR, "tf_pt_mapping_validated.csv")


class LLMEnhancedComparator:
    """基于 LLM 的 TensorFlow 与 PyTorch 差分测试框架"""

    def __init__(
        self,
        test_cases_file: str = DEFAULT_TEST_CASES_FILE,
        mapping_file: str = DEFAULT_MAPPING_FILE,
        key_path: str = DEFAULT_KEY_PATH,
        model: str = DEFAULT_MODEL,
        print_lock: Lock = None,
        llm_workers: int = DEFAULT_WORKERS,
    ):
        """
        初始化比较器

        Args:
            test_cases_file: 测试用例 JSON 文件路径（Step 2 输出）
            mapping_file: TF→PT 映射 CSV 文件路径（Step 3 输出）
            key_path: API key 文件路径
            model: LLM 模型名称
            print_lock: 打印锁
            llm_workers: LLM 并发线程数
        """
        self.model = model
        self.print_lock = print_lock or Lock()
        self.llm_workers = max(1, int(llm_workers))
        self.execution_lock = RLock()
        self.stats_lock = Lock()

        # 初始化 LLM 客户端
        api_key = self._load_api_key(key_path)
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 加载测试用例
        self.test_cases_data = self._load_test_cases(test_cases_file)
        self._safe_print(f"📋 已加载 {len(self.test_cases_data)} 个 TF API 的测试用例")

        # 加载 TF→PT 映射
        self.api_mapping = self._load_mapping(mapping_file)
        has_impl = sum(1 for v in self.api_mapping.values() if v != "无对应实现")
        self._safe_print(f"📋 已加载 {len(self.api_mapping)} 个映射（{has_impl} 个有对应实现）")

        # 创建结果存储目录
        self.result_dir = os.path.join(ROOT_DIR, "tf_pt_test", "tf_pt_log_1")
        os.makedirs(self.result_dir, exist_ok=True)
        self._safe_print(f"📁 结果存储目录: {self.result_dir}")

        # 固定随机种子
        self.random_seed = 42
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

    def _safe_print(self, msg: str, end: str = "\n"):
        """线程安全的打印"""
        with self.print_lock:
            print(msg, end=end, flush=True)

    def _load_api_key(self, key_path: str = DEFAULT_KEY_PATH) -> str:
        """加载 API 密钥"""
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
        """加载测试用例"""
        if not os.path.exists(filepath):
            self._safe_print(f"⚠️ 测试用例文件不存在: {filepath}")
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("test_cases", {})

    def _load_mapping(self, filepath: str) -> Dict[str, str]:
        """加载 TF→PT 映射表"""
        if not os.path.exists(filepath):
            self._safe_print(f"⚠️ 映射文件不存在: {filepath}")
            return {}
        mapping = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tf_api = row.get("tensorflow-api", "").strip()
                pt_api = row.get("pytorch-api", "").strip()
                if tf_api and pt_api:
                    mapping[tf_api] = pt_api
        return mapping

    # ==================== API 工具方法 ====================

    def is_class_based_api(self, api_name: str) -> bool:
        """判断 API 是否是基于类的"""
        parts = api_name.split(".")
        if len(parts) >= 2:
            last_part = parts[-1]
            if last_part and last_part[0].isupper():
                return True
        return False

    def get_operator_function(self, api_name: str, framework: str = "tf"):
        """获取算子函数对象"""
        try:
            if framework == "tf":
                module = tf
            elif framework == "torch":
                module = torch
            else:
                return None

            parts = api_name.split(".")
            # 跳过框架前缀（tf. / torch.）
            start_idx = 1
            obj = module
            for part in parts[start_idx:]:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return None

    def convert_api_name(self, tf_api: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        查找 TF API 对应的 PyTorch API

        Returns:
            (tf_api, pytorch_api, mapping_method)
        """
        if tf_api in self.api_mapping:
            pt_api = self.api_mapping[tf_api]
            if pt_api and pt_api != "无对应实现":
                return tf_api, pt_api, "映射表"
            else:
                return tf_api, None, "无对应实现"
        return tf_api, None, "映射表中未找到"

    # ==================== 数据转换 ====================

    def generate_numpy_data(self, data: Any) -> np.ndarray:
        """
        从描述生成 numpy 数组

        支持格式：
        - {"shape": [2, 3], "dtype": "float32"}
        - {"shape": [2, 3], "dtype": "float32", "range": [-1, 1]}
        - 标量值
        - 列表
        """
        if isinstance(data, dict):
            if "shape" in data:
                shape = data["shape"]
                dtype_str = str(data.get("dtype", "float32"))

                # 去除框架前缀
                for prefix in ["torch.", "tf.", "np.", "numpy."]:
                    if dtype_str.startswith(prefix):
                        dtype_str = dtype_str[len(prefix):]

                # dtype 名称映射
                dtype_map = {
                    "float32": np.float32, "float64": np.float64,
                    "float16": np.float16, "float": np.float32,
                    "int32": np.int32, "int64": np.int64,
                    "int16": np.int16, "int8": np.int8,
                    "uint8": np.uint8, "bool": np.bool_,
                    "complex64": np.complex64, "complex128": np.complex128,
                    "bfloat16": np.float32,  # numpy 不支持 bfloat16，用 float32 替代
                }
                np_dtype = dtype_map.get(dtype_str, np.float32)

                # 处理空张量
                if any(s == 0 for s in shape):
                    return np.empty(shape, dtype=np_dtype)

                # 获取数据范围
                data_range = data.get("range", None)

                if np_dtype == np.bool_:
                    return np.random.choice([True, False], size=shape).astype(np.bool_)
                elif np.issubdtype(np_dtype, np.integer):
                    low = int(data_range[0]) if data_range else 0
                    high = int(data_range[1]) if data_range else 10
                    return np.random.randint(low, high, size=shape).astype(np_dtype)
                elif np.issubdtype(np_dtype, np.complexfloating):
                    real = np.random.randn(*shape).astype(np.float32)
                    imag = np.random.randn(*shape).astype(np.float32)
                    return (real + 1j * imag).astype(np_dtype)
                else:
                    if data_range:
                        low, high = float(data_range[0]), float(data_range[1])
                        return np.random.uniform(low, high, size=shape).astype(np_dtype)
                    else:
                        return np.random.randn(*shape).astype(np_dtype)
            else:
                return np.array(list(data.values()))

        elif isinstance(data, (int, float)):
            return np.array(data)
        elif isinstance(data, list):
            return np.array(data)
        else:
            return np.array(data)

    def convert_to_tensor_tf(self, data: Any, numpy_data: np.ndarray = None) -> tf.Tensor:
        """转换为 TensorFlow 张量"""
        if numpy_data is not None:
            return tf.constant(numpy_data)
        if isinstance(data, dict):
            np_data = self.generate_numpy_data(data)
            return tf.constant(np_data)
        elif isinstance(data, (int, float)):
            return tf.constant(data)
        elif isinstance(data, list):
            return tf.constant(np.array(data))
        else:
            return tf.constant(data)

    def convert_to_tensor_torch(self, data: Any, numpy_data: np.ndarray = None) -> torch.Tensor:
        """转换为 PyTorch 张量"""
        if numpy_data is not None:
            return torch.from_numpy(numpy_data.copy())
        if isinstance(data, dict):
            np_data = self.generate_numpy_data(data)
            return torch.from_numpy(np_data.copy())
        elif isinstance(data, (int, float)):
            return torch.tensor(data)
        elif isinstance(data, list):
            return torch.tensor(data)
        else:
            return torch.tensor(data)

    # ==================== 参数准备 ====================

    def should_skip_param(self, key: str, api_name: str, framework: str) -> bool:
        """判断是否应跳过某个参数"""
        # 通用跳过参数
        common_skip = ["description", "api"]
        if key in common_skip:
            return True

        # PyTorch 特有参数（TF→PT 时跳过）
        torch_skip = ["layout", "requires_grad", "out", "memory_format", "pin_memory"]
        if framework == "torch" and key in torch_skip:
            return True

        # TensorFlow 特有参数（PT→TF 时跳过）
        tf_skip = ["name"]
        if framework == "tf" and key in tf_skip:
            return True

        return False

    def prepare_arguments(
        self, test_case: Dict[str, Any], framework: str = "tf"
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        为指定框架准备参数

        Args:
            test_case: 测试用例（包含张量描述和标量参数）
            framework: "tf" 或 "torch"

        Returns:
            (args, kwargs)
        """
        args = []
        kwargs = {}

        # 位置参数名
        positional_params = ["x", "input", "condition", "y", "other", "a", "b"]

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
                        if framework == "torch":
                            args.append(torch.from_numpy(np_data.copy()))
                        else:
                            args.append(tf.constant(np_data))
                    else:
                        args.append(item)
            return args, kwargs

        # 按顺序处理位置参数
        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, dict) and "shape" in value:
                    np_data = self.generate_numpy_data(value)
                    if framework == "torch":
                        args.append(torch.from_numpy(np_data.copy()))
                    else:
                        args.append(tf.constant(np_data))
                elif isinstance(value, np.ndarray):
                    if framework == "torch":
                        args.append(torch.from_numpy(value.copy()))
                    else:
                        args.append(tf.constant(value))
                else:
                    args.append(value)

        # 处理其他参数（关键字参数）
        for key, value in test_case.items():
            if key in positional_params or key == "api" or self.should_skip_param(key, "", framework):
                continue
            if key.startswith("*"):
                continue

            if isinstance(value, dict) and "shape" in value:
                np_data = self.generate_numpy_data(value)
                if framework == "torch":
                    kwargs[key] = torch.from_numpy(np_data.copy())
                else:
                    kwargs[key] = tf.constant(np_data)
            elif isinstance(value, np.ndarray):
                if framework == "torch":
                    kwargs[key] = torch.from_numpy(value.copy())
                else:
                    kwargs[key] = tf.constant(value)
            else:
                kwargs[key] = value

        return args, kwargs

    # ==================== 结果比较 ====================

    def compare_tensors(
        self, tf_result, torch_result, tolerance: float = 1e-5
    ) -> Tuple[bool, str]:
        """比较 TF 和 PT 的计算结果"""
        try:
            # 转换为 numpy
            if isinstance(tf_result, tf.Tensor):
                tf_np = tf_result.numpy()
            elif isinstance(tf_result, np.ndarray):
                tf_np = tf_result
            else:
                tf_np = np.array(tf_result)

            if isinstance(torch_result, torch.Tensor):
                torch_np = torch_result.detach().cpu().numpy()
            elif isinstance(torch_result, np.ndarray):
                torch_np = torch_result
            else:
                torch_np = np.array(torch_result)

            # 形状一致性检查
            if tf_np.shape != torch_np.shape:
                return False, f"形状不匹配: TF={tf_np.shape} vs PT={torch_np.shape}"

            # 布尔类型精确比较
            if tf_np.dtype == np.bool_ or torch_np.dtype == np.bool_:
                match = np.array_equal(tf_np, torch_np)
                if match:
                    return True, "布尔结果完全一致"
                else:
                    diff_count = np.sum(tf_np != torch_np)
                    return False, f"布尔结果不一致，差异元素数: {diff_count}"

            # 数值比较
            if np.allclose(tf_np, torch_np, atol=tolerance, rtol=tolerance, equal_nan=True):
                return True, "结果一致（在容差范围内）"
            else:
                max_diff = np.max(np.abs(tf_np.astype(np.float64) - torch_np.astype(np.float64)))
                return False, f"结果不一致，最大差异: {max_diff:.8f}"

        except Exception as e:
            return False, f"比较异常: {str(e)}"

    # ==================== 测试执行 ====================

    def execute_test_case(
        self,
        tf_api: str,
        pytorch_api: str,
        tf_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        执行单个测试用例

        Args:
            tf_api: TensorFlow API 名称
            pytorch_api: PyTorch API 名称
            tf_test_case: TF 测试用例
            pytorch_test_case: PT 测试用例（None 则使用 TF 用例）
        """
        result = {
            "tf_api": tf_api,
            "pytorch_api": pytorch_api,
            "tf_success": False,
            "pytorch_success": False,
            "results_match": False,
            "tf_error": None,
            "pytorch_error": None,
            "comparison_error": None,
            "tf_shape": None,
            "pytorch_shape": None,
            "tf_dtype": None,
            "pytorch_dtype": None,
            "status": "unknown",
        }

        if pytorch_test_case is None:
            pytorch_test_case = tf_test_case

        # 统一生成输入张量，保证 TF/PT 使用同一份 numpy 数据
        tf_test_case, pytorch_test_case = self._materialize_shared_tensors(
            tf_api, pytorch_api, tf_test_case, pytorch_test_case
        )

        is_class_tf = self.is_class_based_api(tf_api)
        is_class_pt = self.is_class_based_api(pytorch_api)

        # ---- 执行 TensorFlow ----
        tf_result = None
        try:
            tf_func = self.get_operator_function(tf_api, "tf")
            if tf_func is None:
                raise AttributeError(f"无法找到 TF API: {tf_api}")

            if is_class_tf:
                init_kwargs = {
                    k: v for k, v in tf_test_case.items()
                    if k not in ["api", "input", "x"] and not isinstance(v, (np.ndarray,))
                    and not (isinstance(v, dict) and "shape" in v)
                }
                layer = tf_func(**init_kwargs)
                # 获取输入
                input_data = tf_test_case.get("input") or tf_test_case.get("x")
                if input_data is not None:
                    if isinstance(input_data, dict) and "shape" in input_data:
                        np_data = self.generate_numpy_data(input_data)
                        tf_input = tf.constant(np_data)
                    elif isinstance(input_data, np.ndarray):
                        tf_input = tf.constant(input_data)
                    else:
                        tf_input = tf.constant(input_data)
                    tf_result = layer(tf_input)
                else:
                    tf_result = layer(tf.constant(np.random.randn(2, 3).astype(np.float32)))
            else:
                tf_args, tf_kwargs = self.prepare_arguments(tf_test_case, "tf")
                tf_result = tf_func(*tf_args, **tf_kwargs)

            result["tf_success"] = True
            if hasattr(tf_result, "shape"):
                result["tf_shape"] = list(tf_result.shape)
            if hasattr(tf_result, "dtype"):
                result["tf_dtype"] = str(tf_result.dtype)

        except Exception as e:
            result["tf_error"] = f"{type(e).__name__}: {str(e)}"

        # ---- 执行 PyTorch ----
        torch_result = None
        try:
            pt_func = self.get_operator_function(pytorch_api, "torch")
            if pt_func is None:
                raise AttributeError(f"无法找到 PT API: {pytorch_api}")

            if is_class_pt:
                init_kwargs = {
                    k: v for k, v in pytorch_test_case.items()
                    if k not in ["api", "input", "x"] and not isinstance(v, (np.ndarray,))
                    and not (isinstance(v, dict) and "shape" in v)
                }
                module = pt_func(**init_kwargs)
                input_data = pytorch_test_case.get("input") or pytorch_test_case.get("x")
                if input_data is not None:
                    if isinstance(input_data, dict) and "shape" in input_data:
                        np_data = self.generate_numpy_data(input_data)
                        pt_input = torch.from_numpy(np_data.copy())
                    elif isinstance(input_data, np.ndarray):
                        pt_input = torch.from_numpy(input_data.copy())
                    else:
                        pt_input = torch.tensor(input_data)
                    torch_result = module(pt_input)
                else:
                    torch_result = module(torch.randn(2, 3))
            else:
                pt_args, pt_kwargs = self.prepare_arguments(pytorch_test_case, "torch")
                torch_result = pt_func(*pt_args, **pt_kwargs)

            result["pytorch_success"] = True
            if hasattr(torch_result, "shape"):
                result["pytorch_shape"] = list(torch_result.shape)
            if hasattr(torch_result, "dtype"):
                result["pytorch_dtype"] = str(torch_result.dtype)

        except Exception as e:
            result["pytorch_error"] = f"{type(e).__name__}: {str(e)}"

        # ---- 比较结果 ----
        if result["tf_success"] and result["pytorch_success"]:
            try:
                match, detail = self.compare_tensors(tf_result, torch_result)
                result["results_match"] = match
                result["comparison_error"] = None if match else detail
                result["status"] = "consistent" if match else "inconsistent"
            except Exception as e:
                result["comparison_error"] = str(e)
                result["status"] = "comparison_error"
        elif result["tf_success"] and not result["pytorch_success"]:
            result["status"] = "pytorch_error"
        elif not result["tf_success"] and result["pytorch_success"]:
            result["status"] = "tf_error"
        else:
            result["status"] = "both_error"

        return result

    def _execute_test_case_sequential(
        self, tf_api, pytorch_api, tf_test_case, pytorch_test_case=None
    ) -> Dict[str, Any]:
        """通过锁保证执行不并发"""
        with self.execution_lock:
            return self.execute_test_case(tf_api, pytorch_api, tf_test_case, pytorch_test_case)

    def _default_input_desc_for_class(self, api_name: str) -> Dict[str, Any]:
        """为类 API 提供统一的默认输入描述"""
        name = (api_name or "").lower()
        if "3d" in name:
            return {"shape": [2, 3, 4, 4, 4], "dtype": "float32"}
        if "2d" in name:
            return {"shape": [2, 3, 8, 8], "dtype": "float32"}
        if "1d" in name:
            return {"shape": [2, 3, 10], "dtype": "float32"}
        return {"shape": [2, 3], "dtype": "float32"}

    def _materialize_shared_tensors(
        self,
        tf_api: str,
        pytorch_api: str,
        tf_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """统一生成共享张量，保证两框架输入数值一致"""
        tf_case = copy.deepcopy(tf_test_case)
        pt_case = copy.deepcopy(pytorch_test_case)

        is_class_tf = self.is_class_based_api(tf_api)
        is_class_pt = self.is_class_based_api(pytorch_api)
        if (is_class_tf or is_class_pt) and not (
            "input" in tf_case or "x" in tf_case or "input" in pt_case or "x" in pt_case
        ):
            default_desc = self._default_input_desc_for_class(tf_api or pytorch_api)
            tf_case.setdefault("input", default_desc)
            pt_case.setdefault("input", default_desc)

        shared_tensors: Dict[str, np.ndarray] = {}
        all_keys = set(tf_case.keys()) | set(pt_case.keys())

        for key in all_keys:
            if key == "api":
                continue
            tf_val = tf_case.get(key)
            pt_val = pt_case.get(key)

            if isinstance(tf_val, np.ndarray):
                shared_tensors[key] = tf_val
                continue
            if isinstance(pt_val, np.ndarray):
                shared_tensors[key] = pt_val
                continue

            tensor_desc = None
            if isinstance(tf_val, dict) and "shape" in tf_val:
                tensor_desc = tf_val
            elif isinstance(pt_val, dict) and "shape" in pt_val:
                tensor_desc = pt_val
            if tensor_desc:
                shared_tensors[key] = self.generate_numpy_data(tensor_desc)

        def apply_shared(case: Dict[str, Any]) -> Dict[str, Any]:
            converted = {}
            for key, value in case.items():
                if key in shared_tensors:
                    converted[key] = shared_tensors[key].copy()
                else:
                    converted[key] = value
            return converted

        return apply_shared(tf_case), apply_shared(pt_case)

    # ==================== API 文档爬取 ====================

    def _fetch_api_docs(self, tf_api: str, pytorch_api: str) -> Tuple[str, str]:
        """爬取 TF 和 PT 的 API 文档"""
        MIN_DOC_LENGTH = 300
        tf_doc = ""
        pytorch_doc = ""

        try:
            raw = get_doc_content(tf_api, "tensorflow")
            if raw and len(raw) >= MIN_DOC_LENGTH:
                tf_doc = raw[:3000]
                self._safe_print(f"    📄 TF文档: {len(tf_doc)} 字符")
            else:
                self._safe_print(f"    📄 TF文档: 未获取到有效内容")
        except Exception as e:
            self._safe_print(f"    ⚠️ TF文档爬取失败: {str(e)[:50]}")

        try:
            raw = get_doc_content(pytorch_api, "pytorch")
            if raw and len(raw) >= MIN_DOC_LENGTH:
                pytorch_doc = raw[:3000]
                self._safe_print(f"    📄 PT文档: {len(pytorch_doc)} 字符")
            else:
                self._safe_print(f"    📄 PT文档: 未获取到有效内容")
        except Exception as e:
            self._safe_print(f"    ⚠️ PT文档爬取失败: {str(e)[:50]}")

        return tf_doc, pytorch_doc

    # ==================== LLM 交互 ====================

    def _build_llm_prompt(
        self,
        execution_result: Dict[str, Any],
        tf_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any],
        tf_doc: str = "",
        pytorch_doc: str = "",
    ) -> str:
        """构建 LLM 的提示词"""
        tf_api = execution_result.get("tf_api", "")
        pytorch_api = execution_result.get("pytorch_api", "")
        status = execution_result.get("status", "")
        tf_success = execution_result.get("tf_success", False)
        pytorch_success = execution_result.get("pytorch_success", False)
        results_match = execution_result.get("results_match", False)
        tf_error = execution_result.get("tf_error", "")
        pytorch_error = execution_result.get("pytorch_error", "")
        comparison_error = execution_result.get("comparison_error", "")

        # 简化测试用例以减少 token 消耗
        simplified_tf = {}
        for key, value in tf_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_tf[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_tf[key] = value

        simplified_pt = {}
        for key, value in pytorch_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_pt[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_pt[key] = value

        # 构建参数示例字符串
        tf_param_examples = []
        for key, value in simplified_tf.items():
            if key == "api":
                continue
            if isinstance(value, dict) and "shape" in value:
                tf_param_examples.append(f'    "{key}": {json.dumps(value)}')
            elif isinstance(value, (int, float, bool)):
                tf_param_examples.append(f'    "{key}": {json.dumps(value)}')
            else:
                tf_param_examples.append(f'    "{key}": {json.dumps(value)}')

        tf_param_str = ",\n".join(tf_param_examples) if tf_param_examples else '    "x": {"shape": [2, 3], "dtype": "float32"}'

        pt_param_examples = []
        for key, value in simplified_pt.items():
            if key == "api":
                continue
            if isinstance(value, dict) and "shape" in value:
                pt_param_examples.append(f'    "{key}": {json.dumps(value)}')
            elif isinstance(value, (int, float, bool)):
                pt_param_examples.append(f'    "{key}": {json.dumps(value)}')
            else:
                pt_param_examples.append(f'    "{key}": {json.dumps(value)}')

        pt_param_str = ",\n".join(pt_param_examples) if pt_param_examples else '    "input": {"shape": [2, 3], "dtype": "float32"}'

        # 文档部分
        doc_section = ""
        if tf_doc or pytorch_doc:
            doc_section = "\n## 官方API文档参考\n\n"
            if tf_doc:
                doc_section += f"### TensorFlow {tf_api} 文档\n```\n{tf_doc}\n```\n\n"
            if pytorch_doc:
                doc_section += f"### PyTorch {pytorch_api} 文档\n```\n{pytorch_doc}\n```\n\n"

        prompt = f"""请分析以下算子测试用例在TensorFlow和PyTorch框架中的执行结果，并根据结果进行测试用例的修复或变异（fuzzing）。

## 测试信息
- **TensorFlow API**: {tf_api}
- **PyTorch API**: {pytorch_api}
{doc_section}
## 执行结果
- **执行状态**: {status}
- **TensorFlow执行成功**: {tf_success}
- **PyTorch执行成功**: {pytorch_success}
- **结果是否一致**: {results_match}

## 错误信息
- **TensorFlow错误**: {tf_error if tf_error else "无"}
- **PyTorch错误**: {pytorch_error if pytorch_error else "无"}
- **比较错误**: {comparison_error if comparison_error else "无"}

## 原始测试用例

### TensorFlow测试用例
```json
{json.dumps(simplified_tf, indent=2, ensure_ascii=False)}
```

### PyTorch测试用例
```json
{json.dumps(simplified_pt, indent=2, ensure_ascii=False)}
```

## 任务要求
请根据以上信息（包括官方API文档），自主判断两框架的比较结果是**一致**、**不一致**还是**执行出错**，并执行以下操作：

1. **如果一致**：对用例进行**变异（fuzzing）**，例如修改输入张量的形状、修改参数值等（可以考虑一些极端值或边界值）
2. **如果执行出错**：根据报错原因和官方文档对用例进行**修复**（改变参数名称、数量、类型、取值范围等，不同框架可能不完全一样）或者**跳过**（当你认为这两个跨框架算子的功能不完全等价时）
3. **如果不一致**：判断是否为可容忍的精度误差（1e-3及以下）：（1）如果是可容忍精度误差则**变异**；（2）结合算子文档分析后，认为这两个跨框架算子的功能不完全等价时选择**跳过**；（3）如果既不是可容忍精度误差，两个算子功能也等价，那就是测试用例构造问题，请根据算子文档对用例进行**修复**。

## 输出格式要求
请严格按照以下JSON格式输出，不要包含任何其他文字、注释或markdown标记：

{{
  "operation": "mutation",
  "reason": "进行该操作的详细原因（不超过150字）",
  "tensorflow_test_case": {{
    "api": "{tf_api}",
{tf_param_str}
  }},
  "pytorch_test_case": {{
    "api": "{pytorch_api}",
{pt_param_str}
  }}
}}

**重要说明**：
1. operation的值必须是 "mutation"、"repair" 或 "skip" 之一
2. 张量参数必须使用 {{"shape": [...], "dtype": "..."}} 格式
3. 标量参数直接使用数值
4. 构造两个框架的用例时必须保证输入相同(必要时进行张量形状的转换，如NHWC与NCHW转换)、参数在语义上严格对应
5. TensorFlow和PyTorch的测试用例可以有参数名差异（如x vs input）、参数值差异或参数数量的差异，只要保证理论上输出相同就行
6. 如果该算子找不到官方文档或已从当前版本移除，请将 operation 设为 "skip"，不需要尝试修复
7. 测试用例变异时可探索极端情况：空张量、单元素张量、高维张量、不同数据类型、边界值等
8. 请仔细阅读官方API文档，确保参数名称、类型、取值范围与文档一致
"""
        return prompt

    def call_llm_for_repair_or_mutation(
        self,
        execution_result: Dict[str, Any],
        tf_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any],
        tf_doc: str = "",
        pytorch_doc: str = "",
    ) -> Dict[str, Any]:
        """调用 LLM 进行测试用例修复或变异"""
        prompt = self._build_llm_prompt(
            execution_result, tf_test_case, pytorch_test_case, tf_doc, pytorch_doc
        )
        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个深度学习框架测试专家，精通TensorFlow和PyTorch框架的API差异。你的任务是根据测试用例的执行结果，判断是否需要修复或变异测试用例，并返回严格的JSON格式结果。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            raw_response = completion.choices[0].message.content.strip()
            time.sleep(1)

            # 解析 JSON
            try:
                llm_result = json.loads(raw_response)
                return llm_result
            except json.JSONDecodeError:
                self._safe_print(f"    ⚠️ LLM返回不是有效JSON，尝试提取...")
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    llm_result = json.loads(json_match.group())
                    return llm_result
                else:
                    return {
                        "operation": "skip",
                        "reason": "LLM返回格式错误",
                        "tensorflow_test_case": tf_test_case,
                        "pytorch_test_case": pytorch_test_case,
                    }

        except Exception as e:
            self._safe_print(f"    ❌ 调用LLM失败: {e}")
            return {
                "operation": "skip",
                "reason": f"LLM调用失败: {e}",
                "tensorflow_test_case": tf_test_case,
                "pytorch_test_case": pytorch_test_case,
            }

    # ==================== 核心测试循环 ====================

    def llm_enhanced_test_operator(
        self,
        tf_api: str,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        num_test_cases: int = None,
        num_workers: int = DEFAULT_WORKERS,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        使用 LLM 增强的方式测试单个算子对

        Args:
            tf_api: TensorFlow API 名称
            max_iterations: 每个测试用例的最大迭代次数
            num_test_cases: 要测试的用例数量
            num_workers: LLM 并发线程数
        """
        self._safe_print(f"\n{'=' * 80}")
        self._safe_print(f"🎯 开始测试算子: {tf_api}")
        self._safe_print(f"🔄 每个用例最大迭代次数: {max_iterations}")
        self._safe_print(f"{'=' * 80}\n")

        stats = {"llm_generated_cases": 0, "successful_cases": 0}

        # 获取对应的 PyTorch API
        _, pytorch_api, mapping_method = self.convert_api_name(tf_api)
        if pytorch_api is None:
            self._safe_print(f"❌ {tf_api} 无 PyTorch 对应实现")
            return [], stats

        self._safe_print(f"✅ TensorFlow API: {tf_api}")
        self._safe_print(f"✅ PyTorch API: {pytorch_api}")
        self._safe_print(f"✅ 映射方法: {mapping_method}")

        # 获取测试用例
        api_data = self.test_cases_data.get(tf_api, {})
        test_cases = api_data.get("test_cases", [])

        if not test_cases:
            self._safe_print(f"⚠️ 未找到 {tf_api} 的测试用例，使用默认用例")
            test_cases = [{"description": "默认", "inputs": {"x": {"shape": [2, 3], "dtype": "float32"}}}]

        # 确定实际测试数量
        if num_test_cases is None:
            num_test_cases = len(test_cases)
        else:
            num_test_cases = min(num_test_cases, len(test_cases))

        self._safe_print(f"📋 将测试 {num_test_cases} 个用例 (LLM并发={num_workers}, 执行顺序)")

        # 准备初始用例
        initial_cases = []
        for case_idx in range(num_test_cases):
            tc = test_cases[case_idx]
            # 从 inputs 字段提取参数，构建扁平的测试用例
            if "inputs" in tc:
                flat_case = dict(tc["inputs"])
            else:
                flat_case = {k: v for k, v in tc.items() if k != "description"}
            flat_case["api"] = tf_api
            initial_cases.append((case_idx + 1, flat_case))

        all_results = []

        if num_workers <= 1:
            for case_number, initial_test_case in initial_cases:
                self._safe_print(f"\n📋 用例 {case_number}/{num_test_cases}")
                case_results = self._test_single_case_with_iterations(
                    tf_api, pytorch_api, initial_test_case,
                    max_iterations, case_number, stats,
                )
                all_results.extend(case_results)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_case = {}
                for case_number, initial_test_case in initial_cases:
                    future = executor.submit(
                        self._test_single_case_with_iterations,
                        tf_api, pytorch_api, initial_test_case,
                        max_iterations, case_number, stats,
                    )
                    future_to_case[future] = case_number

                for future in as_completed(future_to_case):
                    case_results = future.result()
                    all_results.extend(case_results)

        all_results.sort(key=lambda r: (r.get("case_number", 0), r.get("iteration", 0)))

        self._safe_print(f"\n{'=' * 80}")
        self._safe_print("✅ 所有测试完成")
        self._safe_print(f"📊 共测试 {num_test_cases} 个用例，总计 {len(all_results)} 次迭代")
        self._safe_print(f"📊 LLM生成的测试用例数: {stats['llm_generated_cases']}")
        self._safe_print(f"📊 两个框架都执行成功的用例数: {stats['successful_cases']}")
        self._safe_print(f"{'=' * 80}\n")

        return all_results, stats

    def _test_single_case_with_iterations(
        self,
        tf_api: str,
        pytorch_api: str,
        initial_test_case: Dict[str, Any],
        max_iterations: int,
        case_number: int,
        stats: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """
        对单个测试用例进行多轮迭代测试

        核心循环：执行 → LLM判断 → 修复/变异/跳过 → 再执行 → ...
        """
        case_results = []

        # 构建 TF 和 PT 的初始测试用例
        current_tf_test_case = copy.deepcopy(initial_test_case)
        current_tf_test_case["api"] = tf_api

        current_pt_test_case = copy.deepcopy(initial_test_case)
        current_pt_test_case["api"] = pytorch_api

        is_llm_generated = False

        # 预先爬取 API 文档（只爬一次）
        self._safe_print(f"  📖 预先爬取API文档...")
        tf_doc, pytorch_doc = self._fetch_api_docs(tf_api, pytorch_api)

        # 迭代测试
        for iteration in range(max_iterations):
            source_type = "LLM" if is_llm_generated else "文件"
            self._safe_print(f"  🔄 迭代 {iteration + 1}/{max_iterations} ({source_type})", end="")

            # 执行测试
            try:
                execution_result = self._execute_test_case_sequential(
                    tf_api, pytorch_api, current_tf_test_case, current_pt_test_case
                )

                tf_status = "✓" if execution_result["tf_success"] else "✗"
                pt_status = "✓" if execution_result["pytorch_success"] else "✗"
                match_status = "✓" if execution_result["results_match"] else "✗"
                self._safe_print(f" | TF:{tf_status} PT:{pt_status} Match:{match_status}")

                if execution_result["tf_error"] and not execution_result["tf_success"]:
                    self._safe_print(f"    ❌ TF错误: {str(execution_result['tf_error'])[:100]}...")
                if execution_result["pytorch_error"] and not execution_result["pytorch_success"]:
                    self._safe_print(f"    ❌ PT错误: {str(execution_result['pytorch_error'])[:100]}...")
                if execution_result["comparison_error"]:
                    self._safe_print(f"    ⚠️ 比较: {str(execution_result['comparison_error'])[:100]}...")

                # 统计 LLM 生成的用例
                if is_llm_generated:
                    if execution_result["tf_success"] and execution_result["pytorch_success"]:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

            except Exception as e:
                self._safe_print(f" | ❌ 严重错误: {str(e)[:80]}...")
                execution_result = {
                    "status": "fatal_error",
                    "tf_success": False, "pytorch_success": False,
                    "results_match": False,
                    "tf_error": f"Fatal: {str(e)}", "pytorch_error": None,
                    "comparison_error": None,
                }

            # 保存迭代结果
            iteration_result = {
                "iteration": iteration + 1,
                "tf_test_case": current_tf_test_case,
                "pytorch_test_case": current_pt_test_case,
                "execution_result": execution_result,
                "llm_operation": None,
                "is_llm_generated": is_llm_generated,
                "case_number": case_number,
            }

            # 调用 LLM
            try:
                llm_result = self.call_llm_for_repair_or_mutation(
                    execution_result, current_tf_test_case, current_pt_test_case,
                    tf_doc, pytorch_doc,
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

            # 准备下一轮
            if operation in ("mutation", "repair"):
                next_tf_case = llm_result.get("tensorflow_test_case", current_tf_test_case)
                next_pt_case = llm_result.get("pytorch_test_case", current_pt_test_case)
                with self.stats_lock:
                    stats["llm_generated_cases"] += 1
                is_llm_generated = True
            else:
                next_tf_case = current_tf_test_case
                next_pt_case = current_pt_test_case

            current_tf_test_case, current_pt_test_case = self._convert_llm_test_cases(
                next_tf_case, next_pt_case
            )

        # 如果最后一轮 LLM 生成了新用例但未执行，补充执行
        if case_results:
            last_op = case_results[-1].get("llm_operation", {}).get("operation", "skip")
            if last_op in ("mutation", "repair"):
                self._safe_print(f"  🔄 执行最终LLM用例", end="")
                try:
                    execution_result = self._execute_test_case_sequential(
                        tf_api, pytorch_api, current_tf_test_case, current_pt_test_case
                    )
                    tf_s = "✓" if execution_result["tf_success"] else "✗"
                    pt_s = "✓" if execution_result["pytorch_success"] else "✗"
                    m_s = "✓" if execution_result["results_match"] else "✗"
                    self._safe_print(f" | TF:{tf_s} PT:{pt_s} Match:{m_s}")

                    if execution_result["tf_success"] and execution_result["pytorch_success"]:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

                    case_results.append({
                        "iteration": len(case_results) + 1,
                        "tf_test_case": current_tf_test_case,
                        "pytorch_test_case": current_pt_test_case,
                        "execution_result": execution_result,
                        "llm_operation": {"operation": "final_execution", "reason": "执行最后一次LLM生成的用例"},
                        "case_number": case_number,
                        "is_llm_generated": True,
                    })
                except Exception as e:
                    self._safe_print(f"  ❌ 最终用例执行失败: {str(e)[:80]}...")
                    case_results.append({
                        "iteration": len(case_results) + 1,
                        "tf_test_case": current_tf_test_case,
                        "pytorch_test_case": current_pt_test_case,
                        "execution_result": {
                            "status": "fatal_error", "tf_success": False,
                            "pytorch_success": False, "results_match": False,
                            "error": str(e),
                        },
                        "llm_operation": {"operation": "final_execution", "reason": "最终用例执行失败"},
                        "case_number": case_number,
                        "is_llm_generated": True,
                    })

        self._safe_print(f"  ✅ 用例 {case_number} 完成，共 {len(case_results)} 次迭代")
        return case_results

    def _convert_llm_test_cases(
        self,
        tf_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        将 LLM 返回的测试用例转换为可执行格式
        确保两个框架共享相同的张量数据
        """
        shared_tensors = {}
        all_keys = set(tf_test_case.keys()) | set(pytorch_test_case.keys())

        for key in all_keys:
            if key == "api":
                continue
            tf_val = tf_test_case.get(key)
            pt_val = pytorch_test_case.get(key)
            tensor_desc = None
            if isinstance(tf_val, dict) and "shape" in tf_val:
                tensor_desc = tf_val
            elif isinstance(pt_val, dict) and "shape" in pt_val:
                tensor_desc = pt_val
            if tensor_desc:
                shared_tensors[key] = self.generate_numpy_data(tensor_desc)

        converted_tf = {}
        for key, value in tf_test_case.items():
            if key in shared_tensors:
                converted_tf[key] = shared_tensors[key]
            else:
                converted_tf[key] = value

        converted_pt = {}
        for key, value in pytorch_test_case.items():
            if key in shared_tensors:
                converted_pt[key] = shared_tensors[key]
            else:
                converted_pt[key] = value

        return converted_tf, converted_pt

    # ==================== 结果保存 ====================

    def save_results(
        self, tf_api: str, results: List[Dict[str, Any]], stats: Dict[str, int] = None
    ):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = tf_api.replace(".", "_")
        filename = f"llm_enhanced_{safe_name}_{timestamp}.json"
        filepath = os.path.join(self.result_dir, filename)

        output_results = []
        for result in results:
            out = copy.deepcopy(result)
            # 简化 numpy 数组
            for case_key in ["tf_test_case", "pytorch_test_case"]:
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
            "tf_api": tf_api,
            "pytorch_api": self.api_mapping.get(tf_api, ""),
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
        """获取所有可测试的 TF API（有测试用例且有 PT 映射）"""
        testable = []
        for tf_api in sorted(self.test_cases_data.keys()):
            pt_api = self.api_mapping.get(tf_api, "无对应实现")
            if pt_api and pt_api != "无对应实现":
                testable.append(tf_api)
        return testable

    def close(self):
        """清理资源"""
        pass


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="基于LLM的TensorFlow与PyTorch算子差分测试框架"
    )
    parser.add_argument(
        "--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"每个测试用例的最大迭代次数（默认{DEFAULT_MAX_ITERATIONS}）"
    )
    parser.add_argument(
        "--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES,
        help=f"每个算子要测试的用例数量（默认{DEFAULT_NUM_CASES}）"
    )
    parser.add_argument(
        "--start", type=int, default=1,
        help="起始算子索引（从1开始，默认1）"
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="结束算子索引（包含，默认全部）"
    )
    parser.add_argument(
        "--operators", "-o", nargs="*",
        help="指定要测试的算子名称（TF格式，如 tf.nn.relu）"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"并发线程数（默认{DEFAULT_WORKERS}）"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"LLM模型名称（默认 {DEFAULT_MODEL}）"
    )
    parser.add_argument(
        "--key-path", "-k", default=DEFAULT_KEY_PATH,
        help=f"API key文件路径（默认 {DEFAULT_KEY_PATH}）"
    )
    parser.add_argument(
        "--test-cases-file", default=DEFAULT_TEST_CASES_FILE,
        help="测试用例 JSON 文件路径"
    )
    parser.add_argument(
        "--mapping-file", default=DEFAULT_MAPPING_FILE,
        help="TF→PT 映射 CSV 文件路径"
    )

    args = parser.parse_args()
    num_workers = max(1, args.workers)

    print("=" * 80)
    print("基于LLM的TensorFlow与PyTorch算子差分测试框架")
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
        # 获取所有可测试的 API
        all_testable = comparator.get_all_testable_apis()
        print(f"\n🔍 可测试的 TF API 总数: {len(all_testable)}")

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

        print(f"📋 算子列表: {', '.join(operator_names[:10])}{'...' if len(operator_names) > 10 else ''}\n")

        all_operators_summary = []

        # 批量日志
        batch_log_file = os.path.join(
            comparator.result_dir,
            f"batch_test_log_{start_datetime.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        log_file = open(batch_log_file, 'w', encoding='utf-8')
        log_file.write("=" * 80 + "\n")
        log_file.write("TF→PT 差分测试批量日志\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"测试配置:\n")
        log_file.write(f"  - 迭代次数: {args.max_iterations}\n")
        log_file.write(f"  - 用例数: {args.num_cases}\n")
        log_file.write(f"  - 并发数: {num_workers}\n")
        log_file.write(f"  - 测试算子数: {len(operator_names)}\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.flush()

        for idx, tf_api in enumerate(operator_names, 1):
            print("\n" + "🔷" * 40)
            print(f"🎯 [{idx}/{len(operator_names)}] 开始测试算子: {tf_api}")
            print("🔷" * 40)

            try:
                results, stats = comparator.llm_enhanced_test_operator(
                    tf_api,
                    max_iterations=args.max_iterations,
                    num_test_cases=args.num_cases,
                    num_workers=num_workers,
                )

                if results:
                    comparator.save_results(tf_api, results, stats)
                    all_operators_summary.append({
                        "operator": tf_api,
                        "pytorch_api": comparator.api_mapping.get(tf_api, ""),
                        "total_iterations": len(results),
                        "llm_generated_cases": stats.get("llm_generated_cases", 0),
                        "successful_cases": stats.get("successful_cases", 0),
                        "status": "completed",
                    })
                    print(f"\n✅ {tf_api} 测试完成")
                    print(f"   - 总迭代次数: {len(results)}")
                    print(f"   - LLM生成用例数: {stats.get('llm_generated_cases', 0)}")
                    print(f"   - 成功执行用例数: {stats.get('successful_cases', 0)}")

                    log_file.write(f"[{idx}/{len(operator_names)}] {tf_api}\n")
                    log_file.write(f"  状态: ✅ 完成\n")
                    log_file.write(f"  总迭代次数: {len(results)}\n")
                    log_file.write(f"  LLM生成用例数: {stats.get('llm_generated_cases', 0)}\n")
                    log_file.write(f"  成功执行用例数: {stats.get('successful_cases', 0)}\n")
                    if stats.get("llm_generated_cases", 0) > 0:
                        rate = stats.get("successful_cases", 0) / stats["llm_generated_cases"] * 100
                        log_file.write(f"  成功率: {rate:.2f}%\n")
                    log_file.write("\n")
                    log_file.flush()
                else:
                    all_operators_summary.append({
                        "operator": tf_api, "total_iterations": 0,
                        "llm_generated_cases": 0, "successful_cases": 0,
                        "status": "no_results",
                    })
                    log_file.write(f"[{idx}/{len(operator_names)}] {tf_api}\n")
                    log_file.write(f"  状态: ⚠️ 无结果\n\n")
                    log_file.flush()

            except Exception as e:
                print(f"\n❌ {tf_api} 测试失败: {e}")
                all_operators_summary.append({
                    "operator": tf_api, "total_iterations": 0,
                    "llm_generated_cases": 0, "successful_cases": 0,
                    "status": "failed", "error": str(e),
                })
                log_file.write(f"[{idx}/{len(operator_names)}] {tf_api}\n")
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

        # 写入日志
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
            f"batch_test_summary_{start_datetime.strftime('%Y%m%d_%H%M%S')}.json"
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
                    "success_rate": f"{total_successful / total_llm_cases * 100:.2f}%" if total_llm_cases > 0 else "N/A",
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
