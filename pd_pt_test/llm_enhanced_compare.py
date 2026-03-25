#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4: 基于 LLM 的 PaddlePaddle 与 PyTorch 算子差分测试框架

功能：
- 从 JSON 文件加载 Paddle 测试用例和 PD→PT 映射
- 对每对等价算子，执行 Paddle 和 PT 并比较结果
- 使用 LLM 进行测试用例的修复（repair）、变异（mutation）和跳过（skip）
- 支持并发测试多个用例
- 保存详细测试结果和批量日志

用法：
    conda activate tf_env
    python pd_pt_test/llm_enhanced_compare.py \
        [--max-iterations 3] [--num-cases 5] [--workers 6] \
        [--start 1] [--end N] [--operators paddle.nn.ReLU paddle.abs]

前置条件：
    1. 已运行 Step 1 extract_pd_apis.py
    2. 已运行 Step 2 extract_pd_test_cases.py
    3. 已运行 Step 3 extract_pd_pt_mapping.py
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
import paddle

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
DATA_DIR = os.path.join(ROOT_DIR, "pd_pt_test", "data")
DEFAULT_TEST_CASES_FILE = os.path.join(DATA_DIR, "pd_test_cases.json")
DEFAULT_MAPPING_FILE = os.path.join(DATA_DIR, "pd_pt_mapping_validated.csv")


class LLMEnhancedComparator:
    """基于 LLM 的 PaddlePaddle 与 PyTorch 差分测试框架"""

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
            mapping_file: PD→PT 映射 CSV 文件路径（Step 3 输出）
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
        self._safe_print(f"📋 已加载 {len(self.test_cases_data)} 个 Paddle API 的测试用例")

        # 加载 PD→PT 映射
        self.api_mapping = self._load_mapping(mapping_file)
        has_impl = sum(1 for v in self.api_mapping.values() if v != "无对应实现")
        self._safe_print(f"📋 已加载 {len(self.api_mapping)} 个映射（{has_impl} 个有对应实现）")

        # 创建结果存储目录
        self.result_dir = os.path.join(ROOT_DIR, "pd_pt_test", "pd_pt_log_1")
        os.makedirs(self.result_dir, exist_ok=True)
        self._safe_print(f"📁 结果存储目录: {self.result_dir}")

        # 固定随机种子
        self.random_seed = 42
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        paddle.seed(self.random_seed)

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
        """加载 PD→PT 映射表"""
        if not os.path.exists(filepath):
            self._safe_print(f"⚠️ 映射文件不存在: {filepath}")
            return {}
        mapping = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pd_api = row.get("paddle-api", "").strip()
                pt_api = row.get("pytorch-api", "").strip()
                if pd_api and pt_api:
                    mapping[pd_api] = pt_api
        return mapping

    # ==================== API 工具方法 ====================

    def is_class_based_api(self, api_name: str) -> bool:
        """判断 API 是否是基于类的（如 paddle.nn.ReLU, torch.nn.ReLU）"""
        parts = api_name.split(".")
        if len(parts) >= 2:
            last_part = parts[-1]
            if last_part and last_part[0].isupper():
                return True
        return False

    def get_operator_function(self, api_name: str, framework: str = "paddle"):
        """
        获取算子函数对象

        Args:
            api_name: API 名称（如 paddle.abs, torch.abs）
            framework: "paddle" 或 "torch"
        """
        try:
            if framework == "paddle":
                module = paddle
            elif framework == "torch":
                module = torch
            else:
                return None

            parts = api_name.split(".")
            # 跳过框架前缀（paddle. / torch.）
            start_idx = 1
            obj = module
            for part in parts[start_idx:]:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return None

    def convert_api_name(self, pd_api: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        查找 Paddle API 对应的 PyTorch API

        Returns:
            (pd_api, pytorch_api, mapping_method)
        """
        if pd_api in self.api_mapping:
            pt_api = self.api_mapping[pd_api]
            if pt_api and pt_api != "无对应实现":
                return pd_api, pt_api, "映射表"
            else:
                return pd_api, None, "无对应实现"
        return pd_api, None, "映射表中未找到"

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
                raw_shape = data["shape"]
                if isinstance(raw_shape, int):
                    shape = (raw_shape,)
                elif isinstance(raw_shape, (list, tuple)):
                    shape = tuple(raw_shape)
                else:
                    shape = tuple(np.array(raw_shape).tolist())
                dtype_str = str(data.get("dtype", "float32"))

                # 去除框架前缀
                for prefix in ["torch.", "paddle.", "np.", "numpy."]:
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

    def convert_to_tensor_pd(self, data: Any, numpy_data: np.ndarray = None) -> paddle.Tensor:
        """转换为 PaddlePaddle 张量"""
        if numpy_data is not None:
            return paddle.to_tensor(numpy_data)
        if isinstance(data, dict):
            np_data = self.generate_numpy_data(data)
            return paddle.to_tensor(np_data)
        elif isinstance(data, (int, float)):
            return paddle.to_tensor(data)
        elif isinstance(data, list):
            return paddle.to_tensor(np.array(data))
        else:
            return paddle.to_tensor(data)

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

        # PyTorch 特有参数（PD→PT 时跳过）
        torch_skip = ["layout", "requires_grad", "out", "memory_format", "pin_memory"]
        if framework == "torch" and key in torch_skip:
            return True

        # Paddle 特有参数（PT→PD 时跳过）
        paddle_skip = ["name", "place"]
        if framework == "paddle" and key in paddle_skip:
            return True

        return False

    def prepare_arguments(
        self, test_case: Dict[str, Any], framework: str = "paddle"
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        为指定框架准备参数

        Args:
            test_case: 测试用例（包含张量描述和标量参数）
            framework: "paddle" 或 "torch"

        Returns:
            (args, kwargs)
        """
        args = []
        kwargs = {}

        def convert_value(value: Any) -> Any:
            if isinstance(value, dict):
                if "shape" in value:
                    np_data = self.generate_numpy_data(value)
                    if framework == "torch":
                        return torch.from_numpy(np_data.copy())
                    return paddle.to_tensor(np_data)
                return {k: convert_value(v) for k, v in value.items()}
            if isinstance(value, np.ndarray):
                if framework == "torch":
                    return torch.from_numpy(value.copy())
                return paddle.to_tensor(value)
            if isinstance(value, list):
                return [convert_value(v) for v in value]
            if isinstance(value, tuple):
                return tuple(convert_value(v) for v in value)
            return value

        def normalize_dtype(dtype_value: Any) -> Any:
            if not isinstance(dtype_value, str):
                return dtype_value

            token = dtype_value.strip()
            for prefix in ["torch.", "paddle.", "np.", "numpy."]:
                if token.startswith(prefix):
                    token = token[len(prefix):]

            if framework == "torch":
                return getattr(torch, token, dtype_value)
            if framework == "paddle":
                return getattr(paddle, token, dtype_value)
            return dtype_value

        # LLM 显式 args/kwargs 风格
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

        # 位置参数名
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
                        if framework == "torch":
                            args.append(torch.from_numpy(np_data.copy()))
                        else:
                            args.append(paddle.to_tensor(np_data))
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

        # 处理其他参数（关键字参数）
        for key, value in test_case.items():
            if key in positional_params or key in ("api", "args", "kwargs") or self.should_skip_param(key, "", framework):
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
        self, pd_result, torch_result, tolerance: float = 1e-5
    ) -> Tuple[bool, str]:
        """比较 Paddle 和 PT 的计算结果"""
        try:
            # 转换为 numpy
            if isinstance(pd_result, paddle.Tensor):
                pd_np = pd_result.numpy()
            elif isinstance(pd_result, np.ndarray):
                pd_np = pd_result
            else:
                pd_np = np.array(pd_result)

            if isinstance(torch_result, torch.Tensor):
                torch_np = torch_result.detach().cpu().numpy()
            elif isinstance(torch_result, np.ndarray):
                torch_np = torch_result
            else:
                torch_np = np.array(torch_result)

            # 形状一致性检查
            if pd_np.shape != torch_np.shape:
                return False, f"形状不匹配: PD={pd_np.shape} vs PT={torch_np.shape}"

            # 布尔类型精确比较
            if pd_np.dtype == np.bool_ or torch_np.dtype == np.bool_:
                match = np.array_equal(pd_np, torch_np)
                if match:
                    return True, "布尔结果完全一致"
                else:
                    diff_count = np.sum(pd_np != torch_np)
                    return False, f"布尔结果不一致，差异元素数: {diff_count}"

            # 数值比较
            if np.allclose(pd_np, torch_np, atol=tolerance, rtol=tolerance, equal_nan=True):
                return True, "结果一致（在容差范围内）"
            else:
                max_diff = np.max(np.abs(pd_np.astype(np.float64) - torch_np.astype(np.float64)))
                return False, f"结果不一致，最大差异: {max_diff:.8f}"

        except Exception as e:
            return False, f"比较异常: {str(e)}"

    # ==================== 测试执行 ====================

    def execute_test_case(
        self,
        pd_api: str,
        pytorch_api: str,
        pd_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        执行单个测试用例

        Args:
            pd_api: PaddlePaddle API 名称
            pytorch_api: PyTorch API 名称
            pd_test_case: Paddle 测试用例
            pytorch_test_case: PT 测试用例（None 则使用 Paddle 用例）
        """
        if pytorch_test_case is None:
            pytorch_test_case = pd_test_case

        effective_pd_api = pd_test_case.get("api", pd_api) if isinstance(pd_test_case, dict) else pd_api
        effective_pt_api = pytorch_test_case.get("api", pytorch_api) if isinstance(pytorch_test_case, dict) else pytorch_api

        result = {
            "pd_api": effective_pd_api,
            "pytorch_api": effective_pt_api,
            "pd_success": False,
            "pytorch_success": False,
            "results_match": False,
            "pd_error": None,
            "pytorch_error": None,
            "comparison_error": None,
            "pd_shape": None,
            "pytorch_shape": None,
            "pd_dtype": None,
            "pytorch_dtype": None,
            "status": "unknown",
        }

        # 统一生成输入张量，保证 PD/PT 使用同一份 numpy 数据
        pd_test_case, pytorch_test_case = self._materialize_shared_tensors(
            effective_pd_api, effective_pt_api, pd_test_case, pytorch_test_case
        )

        is_class_pd = self.is_class_based_api(effective_pd_api)
        is_class_pt = self.is_class_based_api(effective_pt_api)

        # ---- 执行 PaddlePaddle ----
        pd_result = None
        try:
            pd_func = self.get_operator_function(effective_pd_api, "paddle")
            if pd_func is None:
                raise AttributeError(f"无法找到 Paddle API: {effective_pd_api}")

            if is_class_pd:
                init_kwargs = {
                    k: v for k, v in pd_test_case.items()
                    if k not in ["api", "input", "x"] and not isinstance(v, (np.ndarray,))
                    and not (isinstance(v, dict) and "shape" in v)
                }
                layer = pd_func(**init_kwargs)
                # 获取输入
                input_data = pd_test_case.get("input") or pd_test_case.get("x")
                if input_data is not None:
                    if isinstance(input_data, dict) and "shape" in input_data:
                        np_data = self.generate_numpy_data(input_data)
                        pd_input = paddle.to_tensor(np_data)
                    elif isinstance(input_data, np.ndarray):
                        pd_input = paddle.to_tensor(input_data)
                    else:
                        pd_input = paddle.to_tensor(input_data)
                    pd_result = layer(pd_input)
                else:
                    pd_result = layer(paddle.to_tensor(np.random.randn(2, 3).astype(np.float32)))
            else:
                pd_args, pd_kwargs = self.prepare_arguments(pd_test_case, "paddle")
                pd_result = pd_func(*pd_args, **pd_kwargs)

            result["pd_success"] = True
            if hasattr(pd_result, "shape"):
                result["pd_shape"] = list(pd_result.shape)
            if hasattr(pd_result, "dtype"):
                result["pd_dtype"] = str(pd_result.dtype)

        except Exception as e:
            result["pd_error"] = f"{type(e).__name__}: {str(e)}"

        # ---- 执行 PyTorch ----
        torch_result = None
        try:
            pt_func = self.get_operator_function(effective_pt_api, "torch")
            if pt_func is None:
                raise AttributeError(f"无法找到 PT API: {effective_pt_api}")

            if effective_pd_api == "paddle.add_n" and effective_pt_api == "torch.sum":
                candidate_inputs = pytorch_test_case.get("inputs") or pd_test_case.get("inputs")
                if isinstance(candidate_inputs, list) and len(candidate_inputs) >= 1:
                    tensor_list = []
                    for item in candidate_inputs:
                        if isinstance(item, torch.Tensor):
                            tensor_list.append(item)
                        elif isinstance(item, np.ndarray):
                            tensor_list.append(torch.from_numpy(item.copy()))
                        elif isinstance(item, paddle.Tensor):
                            tensor_list.append(torch.from_numpy(item.numpy().copy()))
                        elif isinstance(item, dict) and "shape" in item:
                            np_data = self.generate_numpy_data(item)
                            tensor_list.append(torch.from_numpy(np_data.copy()))
                        else:
                            tensor_list.append(torch.tensor(item))
                    torch_result = torch.stack(tensor_list, dim=0).sum(dim=0)
                    result["pytorch_success"] = True
                    if hasattr(torch_result, "shape"):
                        result["pytorch_shape"] = list(torch_result.shape)
                    if hasattr(torch_result, "dtype"):
                        result["pytorch_dtype"] = str(torch_result.dtype)
                else:
                    raise ValueError("torch.sum 等价执行需要 inputs 列表")

            elif is_class_pt:
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

            if not result["pytorch_success"]:
                result["pytorch_success"] = True
                if hasattr(torch_result, "shape"):
                    result["pytorch_shape"] = list(torch_result.shape)
                if hasattr(torch_result, "dtype"):
                    result["pytorch_dtype"] = str(torch_result.dtype)

        except Exception as e:
            result["pytorch_error"] = f"{type(e).__name__}: {str(e)}"

        # ---- 比较结果 ----
        if result["pd_success"] and result["pytorch_success"]:
            try:
                match, detail = self.compare_tensors(pd_result, torch_result)
                result["results_match"] = match
                result["comparison_error"] = None if match else detail
                result["status"] = "consistent" if match else "inconsistent"
            except Exception as e:
                result["comparison_error"] = str(e)
                result["status"] = "comparison_error"
        elif result["pd_success"] and not result["pytorch_success"]:
            result["status"] = "pytorch_error"
        elif not result["pd_success"] and result["pytorch_success"]:
            result["status"] = "paddle_error"
        else:
            result["status"] = "both_error"

        return result

    def _execute_test_case_sequential(
        self, pd_api, pytorch_api, pd_test_case, pytorch_test_case=None
    ) -> Dict[str, Any]:
        """通过锁保证执行不并发"""
        with self.execution_lock:
            return self.execute_test_case(pd_api, pytorch_api, pd_test_case, pytorch_test_case)

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
        pd_api: str,
        pytorch_api: str,
        pd_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """统一生成共享张量，保证两框架输入数值一致"""
        pd_case = copy.deepcopy(pd_test_case)
        pt_case = copy.deepcopy(pytorch_test_case)

        is_class_pd = self.is_class_based_api(pd_api)
        is_class_pt = self.is_class_based_api(pytorch_api)
        if (is_class_pd or is_class_pt) and not (
            "input" in pd_case or "x" in pd_case or "input" in pt_case or "x" in pt_case
        ):
            default_desc = self._default_input_desc_for_class(pd_api or pytorch_api)
            pd_case.setdefault("input", default_desc)
            pt_case.setdefault("input", default_desc)

        def is_tensor_desc(value: Any) -> bool:
            return isinstance(value, dict) and "shape" in value

        def clone_array(value: np.ndarray) -> np.ndarray:
            return value.copy()

        def materialize_pair(pd_val: Any, pt_val: Any) -> Tuple[Any, Any]:
            if isinstance(pd_val, np.ndarray):
                return clone_array(pd_val), clone_array(pd_val)
            if isinstance(pt_val, np.ndarray):
                return clone_array(pt_val), clone_array(pt_val)

            if is_tensor_desc(pd_val) or is_tensor_desc(pt_val):
                tensor_desc = pd_val if is_tensor_desc(pd_val) else pt_val
                shared = self.generate_numpy_data(tensor_desc)
                return clone_array(shared), clone_array(shared)

            if isinstance(pd_val, list) or isinstance(pt_val, list):
                pd_list = pd_val if isinstance(pd_val, list) else []
                pt_list = pt_val if isinstance(pt_val, list) else []
                size = max(len(pd_list), len(pt_list))
                out_pd = []
                out_pt = []
                for index in range(size):
                    left = pd_list[index] if index < len(pd_list) else None
                    right = pt_list[index] if index < len(pt_list) else None
                    new_left, new_right = materialize_pair(left, right)
                    if index < len(pd_list):
                        out_pd.append(new_left)
                    if index < len(pt_list):
                        out_pt.append(new_right)
                return out_pd if isinstance(pd_val, list) else pd_val, out_pt if isinstance(pt_val, list) else pt_val

            if isinstance(pd_val, dict) or isinstance(pt_val, dict):
                pd_dict = pd_val if isinstance(pd_val, dict) else {}
                pt_dict = pt_val if isinstance(pt_val, dict) else {}
                keys = set(pd_dict.keys()) | set(pt_dict.keys())
                out_pd = {}
                out_pt = {}
                for key in keys:
                    if key == "api":
                        if key in pd_dict:
                            out_pd[key] = pd_dict[key]
                        if key in pt_dict:
                            out_pt[key] = pt_dict[key]
                        continue
                    new_left, new_right = materialize_pair(pd_dict.get(key), pt_dict.get(key))
                    if key in pd_dict:
                        out_pd[key] = new_left
                    if key in pt_dict:
                        out_pt[key] = new_right
                return out_pd if isinstance(pd_val, dict) else pd_val, out_pt if isinstance(pt_val, dict) else pt_val

            return pd_val, pt_val

        pd_case, pt_case = materialize_pair(pd_case, pt_case)

        # 通用跨框架同义参数映射
        if isinstance(pd_case, dict) and isinstance(pt_case, dict):
            alias_pairs = [
                ("x", "input"),
                ("input", "x"),
                ("y", "other"),
                ("other", "y"),
            ]
            for pd_key, pt_key in alias_pairs:
                if pd_key in pd_case and pt_key in pt_case:
                    pd_item, pt_item = materialize_pair(pd_case[pd_key], pt_case[pt_key])
                    pd_case[pd_key] = pd_item
                    pt_case[pt_key] = pt_item

        # add_n 常见跨键映射：PD inputs[0/1] <-> PT input/other
        pd_inputs = pd_case.get("inputs") if isinstance(pd_case, dict) else None
        if isinstance(pd_inputs, list) and isinstance(pt_case, dict):
            if len(pd_inputs) >= 1 and "input" in pt_case:
                pd_item, pt_item = materialize_pair(pd_inputs[0], pt_case.get("input"))
                pd_inputs[0] = pd_item
                pt_case["input"] = pt_item
            if len(pd_inputs) >= 2 and "other" in pt_case:
                pd_item, pt_item = materialize_pair(pd_inputs[1], pt_case.get("other"))
                pd_inputs[1] = pd_item
                pt_case["other"] = pt_item

        return pd_case, pt_case

    # ==================== API 文档爬取 ====================

    def _fetch_api_docs(self, pd_api: str, pytorch_api: str) -> Tuple[str, str]:
        """爬取 Paddle 和 PT 的 API 文档"""
        MIN_DOC_LENGTH = 300
        pd_doc = ""
        pytorch_doc = ""

        try:
            raw = get_doc_content(pd_api, "paddle")
            if raw and len(raw) >= MIN_DOC_LENGTH:
                pd_doc = raw[:3000]
                self._safe_print(f"    📄 PD文档: {len(pd_doc)} 字符")
            else:
                self._safe_print(f"    📄 PD文档: 未获取到有效内容")
        except Exception as e:
            self._safe_print(f"    ⚠️ PD文档爬取失败: {str(e)[:50]}")

        try:
            raw = get_doc_content(pytorch_api, "pytorch")
            if raw and len(raw) >= MIN_DOC_LENGTH:
                pytorch_doc = raw[:3000]
                self._safe_print(f"    📄 PT文档: {len(pytorch_doc)} 字符")
            else:
                self._safe_print(f"    📄 PT文档: 未获取到有效内容")
        except Exception as e:
            self._safe_print(f"    ⚠️ PT文档爬取失败: {str(e)[:50]}")

        return pd_doc, pytorch_doc

    # ==================== LLM 交互 ====================

    def _build_llm_prompt(
        self,
        execution_result: Dict[str, Any],
        pd_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any],
        pd_doc: str = "",
        pytorch_doc: str = "",
    ) -> str:
        """构建 LLM 的提示词"""
        pd_api = execution_result.get("pd_api", "")
        pytorch_api = execution_result.get("pytorch_api", "")
        status = execution_result.get("status", "")
        pd_success = execution_result.get("pd_success", False)
        pytorch_success = execution_result.get("pytorch_success", False)
        results_match = execution_result.get("results_match", False)
        pd_error = execution_result.get("pd_error", "")
        pytorch_error = execution_result.get("pytorch_error", "")
        comparison_error = execution_result.get("comparison_error", "")

        # 简化测试用例以减少 token 消耗
        simplified_pd = {}
        for key, value in pd_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_pd[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_pd[key] = value

        simplified_pt = {}
        for key, value in pytorch_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_pt[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_pt[key] = value

        # 构建参数示例字符串
        pd_param_examples = []
        for key, value in simplified_pd.items():
            if key == "api":
                continue
            pd_param_examples.append(f'    "{key}": {json.dumps(value)}')

        pd_param_str = ",\n".join(pd_param_examples) if pd_param_examples else '    "x": {"shape": [2, 3], "dtype": "float32"}'

        pt_param_examples = []
        for key, value in simplified_pt.items():
            if key == "api":
                continue
            pt_param_examples.append(f'    "{key}": {json.dumps(value)}')

        pt_param_str = ",\n".join(pt_param_examples) if pt_param_examples else '    "input": {"shape": [2, 3], "dtype": "float32"}'

        # 文档部分
        doc_section = ""
        if pd_doc or pytorch_doc:
            doc_section = "\n## 官方API文档参考\n\n"
            if pd_doc:
                doc_section += f"### PaddlePaddle {pd_api} 文档\n```\n{pd_doc}\n```\n\n"
            if pytorch_doc:
                doc_section += f"### PyTorch {pytorch_api} 文档\n```\n{pytorch_doc}\n```\n\n"

        prompt = f"""请分析以下算子测试用例在PaddlePaddle和PyTorch框架中的执行结果，并根据结果进行测试用例的修复或变异（fuzzing）。

## 测试信息
- **PaddlePaddle API**: {pd_api}
- **PyTorch API**: {pytorch_api}
{doc_section}
## 执行结果
- **执行状态**: {status}
- **PaddlePaddle执行成功**: {pd_success}
- **PyTorch执行成功**: {pytorch_success}
- **结果是否一致**: {results_match}

## 错误信息
- **PaddlePaddle错误**: {pd_error if pd_error else "无"}
- **PyTorch错误**: {pytorch_error if pytorch_error else "无"}
- **比较错误**: {comparison_error if comparison_error else "无"}

## 原始测试用例

### PaddlePaddle测试用例
```json
{json.dumps(simplified_pd, indent=2, ensure_ascii=False)}
```

### PyTorch测试用例
```json
{json.dumps(simplified_pt, indent=2, ensure_ascii=False)}
```

## 任务要求
请根据以上信息（包括官方API文档），自主判断两框架的比较结果是**一致**、**不一致**还是**执行出错**，并执行以下操作：

1. **如果一致**：对用例进行**变异（fuzzing）**，例如修改输入张量的形状、修改参数值等（可以考虑一些极端值或边界值）
2. **如果执行出错**：根据报错原因和官方文档对用例进行**修复**（改变参数名称、数量、类型、取值范围等，不同框架可能不完全一样）或者**跳过**（当：1. 该算子的文档不存在或文档显示它已从当前版本移除；2. 你认为这两个跨框架算子的功能不完全等价时）
3. **如果不一致**：判断是否为可容忍的精度误差（1e-3及以下）：（1）如果是可容忍精度误差则**变异**；（2）结合算子文档分析后，认为这两个跨框架算子的功能不完全等价时选择**跳过**；（3）如果既不是可容忍精度误差，两个算子功能也等价，那就是测试用例构造问题，请根据算子文档对用例进行**修复**。

## 输出格式要求
请严格按照以下JSON格式输出，不要包含任何其他文字、注释或markdown标记：

{{
  "operation": "mutation",
  "reason": "进行该操作的详细原因（不超过150字）",
  "paddle_test_case": {{
    "api": "{pd_api}",
{pd_param_str}
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
4. 构造两个框架的用例时必须保证输入相同、参数在语义上严格对应
5. PaddlePaddle和PyTorch的测试用例可以有参数名差异（如x vs input）、参数值差异或参数数量的差异，只要保证理论上输出相同就行
6. 如果该算子找不到官方文档或文档显示它已从当前版本移除，请将 operation 设为 "skip"，不需要尝试修复
7. 测试用例变异时可探索极端情况：空张量、单元素张量、高维张量、不同数据类型、边界值等
8. 请仔细阅读官方API文档，确保参数名称、类型、取值范围与文档一致
9. PaddlePaddle 和 PyTorch 的数据格式默认都是 NCHW（不像 TensorFlow 默认 NHWC），注意这一点
"""
        return prompt

    def call_llm_for_repair_or_mutation(
        self,
        execution_result: Dict[str, Any],
        pd_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any],
        pd_doc: str = "",
        pytorch_doc: str = "",
    ) -> Dict[str, Any]:
        """调用 LLM 进行测试用例修复或变异"""
        prompt = self._build_llm_prompt(
            execution_result, pd_test_case, pytorch_test_case, pd_doc, pytorch_doc
        )
        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个深度学习框架测试专家，精通PaddlePaddle和PyTorch框架的API差异。你的任务是根据测试用例的执行结果，判断是否需要修复或变异测试用例，并返回严格的JSON格式结果。",
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
                        "paddle_test_case": pd_test_case,
                        "pytorch_test_case": pytorch_test_case,
                    }

        except Exception as e:
            self._safe_print(f"    ❌ 调用LLM失败: {e}")
            return {
                "operation": "skip",
                "reason": f"LLM调用失败: {e}",
                "paddle_test_case": pd_test_case,
                "pytorch_test_case": pytorch_test_case,
            }

    # ==================== 核心测试循环 ====================

    def llm_enhanced_test_operator(
        self,
        pd_api: str,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        num_test_cases: int = None,
        num_workers: int = DEFAULT_WORKERS,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        使用 LLM 增强的方式测试单个算子对

        Args:
            pd_api: PaddlePaddle API 名称
            max_iterations: 每个测试用例的最大迭代次数
            num_test_cases: 要测试的用例数量
            num_workers: LLM 并发线程数
        """
        self._safe_print(f"\n{'=' * 80}")
        self._safe_print(f"🎯 开始测试算子: {pd_api}")
        self._safe_print(f"🔄 每个用例最大迭代次数: {max_iterations}")
        self._safe_print(f"{'=' * 80}\n")

        stats = {"llm_generated_cases": 0, "successful_cases": 0}

        # 获取对应的 PyTorch API
        _, pytorch_api, mapping_method = self.convert_api_name(pd_api)
        if pytorch_api is None:
            self._safe_print(f"❌ {pd_api} 无 PyTorch 对应实现")
            return [], stats

        self._safe_print(f"✅ PaddlePaddle API: {pd_api}")
        self._safe_print(f"✅ PyTorch API: {pytorch_api}")
        self._safe_print(f"✅ 映射方法: {mapping_method}")

        # 获取测试用例
        api_data = self.test_cases_data.get(pd_api, {})
        test_cases = api_data.get("test_cases", [])

        if not test_cases:
            self._safe_print(f"⚠️ 未找到 {pd_api} 的测试用例，使用默认用例")
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
            flat_case["api"] = pd_api
            initial_cases.append((case_idx + 1, flat_case))

        all_results = []

        if num_workers <= 1:
            for case_number, initial_test_case in initial_cases:
                self._safe_print(f"\n📋 用例 {case_number}/{num_test_cases}")
                case_results = self._test_single_case_with_iterations(
                    pd_api, pytorch_api, initial_test_case,
                    max_iterations, case_number, stats,
                )
                all_results.extend(case_results)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_case = {}
                for case_number, initial_test_case in initial_cases:
                    future = executor.submit(
                        self._test_single_case_with_iterations,
                        pd_api, pytorch_api, initial_test_case,
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
        pd_api: str,
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

        # 构建 PD 和 PT 的初始测试用例
        current_pd_test_case = copy.deepcopy(initial_test_case)
        current_pd_test_case["api"] = pd_api

        current_pt_test_case = copy.deepcopy(initial_test_case)
        current_pt_test_case["api"] = pytorch_api

        is_llm_generated = False

        # 预先爬取 API 文档（只爬一次）
        self._safe_print(f"  📖 预先爬取API文档...")
        pd_doc, pytorch_doc = self._fetch_api_docs(pd_api, pytorch_api)

        # 迭代测试
        for iteration in range(max_iterations):
            source_type = "LLM" if is_llm_generated else "文件"
            self._safe_print(f"  🔄 迭代 {iteration + 1}/{max_iterations} ({source_type})", end="")

            current_pd_api = current_pd_test_case.get("api", pd_api) or pd_api
            current_pt_api = current_pt_test_case.get("api", pytorch_api) or pytorch_api

            # 执行测试
            try:
                execution_result = self._execute_test_case_sequential(
                    current_pd_api, current_pt_api, current_pd_test_case, current_pt_test_case
                )

                pd_status = "✓" if execution_result["pd_success"] else "✗"
                pt_status = "✓" if execution_result["pytorch_success"] else "✗"
                match_status = "✓" if execution_result["results_match"] else "✗"
                self._safe_print(f" | PD:{pd_status} PT:{pt_status} Match:{match_status}")

                if execution_result["pd_error"] and not execution_result["pd_success"]:
                    self._safe_print(f"    ❌ PD错误: {str(execution_result['pd_error'])[:100]}...")
                if execution_result["pytorch_error"] and not execution_result["pytorch_success"]:
                    self._safe_print(f"    ❌ PT错误: {str(execution_result['pytorch_error'])[:100]}...")
                if execution_result["comparison_error"]:
                    self._safe_print(f"    ⚠️ 比较: {str(execution_result['comparison_error'])[:100]}...")

                # 统计 LLM 生成的用例
                if is_llm_generated:
                    if execution_result["pd_success"] and execution_result["pytorch_success"]:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

            except Exception as e:
                self._safe_print(f" | ❌ 严重错误: {str(e)[:80]}...")
                execution_result = {
                    "status": "fatal_error",
                    "pd_success": False, "pytorch_success": False,
                    "results_match": False,
                    "pd_error": f"Fatal: {str(e)}", "pytorch_error": None,
                    "comparison_error": None,
                }

            # 保存迭代结果
            iteration_result = {
                "iteration": iteration + 1,
                "pd_test_case": current_pd_test_case,
                "pytorch_test_case": current_pt_test_case,
                "execution_result": execution_result,
                "llm_operation": None,
                "is_llm_generated": is_llm_generated,
                "case_number": case_number,
            }

            # 调用 LLM
            try:
                llm_result = self.call_llm_for_repair_or_mutation(
                    execution_result, current_pd_test_case, current_pt_test_case,
                    pd_doc, pytorch_doc,
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
                next_pd_case = llm_result.get("paddle_test_case", current_pd_test_case)
                next_pt_case = llm_result.get("pytorch_test_case", current_pt_test_case)
                with self.stats_lock:
                    stats["llm_generated_cases"] += 1
                is_llm_generated = True
            else:
                next_pd_case = current_pd_test_case
                next_pt_case = current_pt_test_case

            current_pd_test_case, current_pt_test_case = self._convert_llm_test_cases(
                next_pd_case, next_pt_case
            )

        # 如果最后一轮 LLM 生成了新用例但未执行，补充执行
        if case_results:
            last_op = case_results[-1].get("llm_operation", {}).get("operation", "skip")
            if last_op in ("mutation", "repair"):
                self._safe_print(f"  🔄 执行最终LLM用例", end="")
                try:
                    execution_result = self._execute_test_case_sequential(
                        current_pd_test_case.get("api", pd_api) or pd_api,
                        current_pt_test_case.get("api", pytorch_api) or pytorch_api,
                        current_pd_test_case,
                        current_pt_test_case,
                    )
                    pd_s = "✓" if execution_result["pd_success"] else "✗"
                    pt_s = "✓" if execution_result["pytorch_success"] else "✗"
                    m_s = "✓" if execution_result["results_match"] else "✗"
                    self._safe_print(f" | PD:{pd_s} PT:{pt_s} Match:{m_s}")

                    if execution_result["pd_success"] and execution_result["pytorch_success"]:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

                    case_results.append({
                        "iteration": len(case_results) + 1,
                        "pd_test_case": current_pd_test_case,
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
                        "pd_test_case": current_pd_test_case,
                        "pytorch_test_case": current_pt_test_case,
                        "execution_result": {
                            "status": "fatal_error", "pd_success": False,
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
        pd_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        将 LLM 返回的测试用例转换为可执行格式
        确保两个框架共享相同的张量数据
        """
        pd_api = pd_test_case.get("api", "") if isinstance(pd_test_case, dict) else ""
        pt_api = pytorch_test_case.get("api", "") if isinstance(pytorch_test_case, dict) else ""
        return self._materialize_shared_tensors(pd_api, pt_api, pd_test_case, pytorch_test_case)

    # ==================== 结果保存 ====================

    def save_results(
        self, pd_api: str, results: List[Dict[str, Any]], stats: Dict[str, int] = None
    ):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = pd_api.replace(".", "_")
        filename = f"llm_enhanced_{safe_name}_{timestamp}.json"
        filepath = os.path.join(self.result_dir, filename)

        output_results = []
        for result in results:
            out = copy.deepcopy(result)
            # 简化 numpy 数组
            for case_key in ["pd_test_case", "pytorch_test_case"]:
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
            "pd_api": pd_api,
            "pytorch_api": self.api_mapping.get(pd_api, ""),
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
        """获取所有可测试的 Paddle API（有测试用例且有 PT 映射）"""
        testable = []
        for pd_api in sorted(self.test_cases_data.keys()):
            pt_api = self.api_mapping.get(pd_api, "无对应实现")
            if pt_api and pt_api != "无对应实现":
                testable.append(pd_api)
        return testable

    def close(self):
        """清理资源"""
        pass


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="基于LLM的PaddlePaddle与PyTorch算子差分测试框架"
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
        help="指定要测试的算子名称（Paddle格式，如 paddle.nn.ReLU）"
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
        help="PD→PT 映射 CSV 文件路径"
    )

    args = parser.parse_args()
    num_workers = max(1, args.workers)

    print("=" * 80)
    print("基于LLM的PaddlePaddle与PyTorch算子差分测试框架")
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
        print(f"\n🔍 可测试的 Paddle API 总数: {len(all_testable)}")

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
        log_file.write("PD→PT 差分测试批量日志\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"测试配置:\n")
        log_file.write(f"  - 迭代次数: {args.max_iterations}\n")
        log_file.write(f"  - 用例数: {args.num_cases}\n")
        log_file.write(f"  - 并发数: {num_workers}\n")
        log_file.write(f"  - 测试算子数: {len(operator_names)}\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.flush()

        for idx, pd_api in enumerate(operator_names, 1):
            print("\n" + "🔷" * 40)
            print(f"🎯 [{idx}/{len(operator_names)}] 开始测试算子: {pd_api}")
            print("🔷" * 40)

            try:
                results, stats = comparator.llm_enhanced_test_operator(
                    pd_api,
                    max_iterations=args.max_iterations,
                    num_test_cases=args.num_cases,
                    num_workers=num_workers,
                )

                if results:
                    comparator.save_results(pd_api, results, stats)
                    all_operators_summary.append({
                        "operator": pd_api,
                        "pytorch_api": comparator.api_mapping.get(pd_api, ""),
                        "total_iterations": len(results),
                        "llm_generated_cases": stats.get("llm_generated_cases", 0),
                        "successful_cases": stats.get("successful_cases", 0),
                        "status": "completed",
                    })
                    print(f"\n✅ {pd_api} 测试完成")
                    print(f"   - 总迭代次数: {len(results)}")
                    print(f"   - LLM生成用例数: {stats.get('llm_generated_cases', 0)}")
                    print(f"   - 成功执行用例数: {stats.get('successful_cases', 0)}")

                    log_file.write(f"[{idx}/{len(operator_names)}] {pd_api}\n")
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
                        "operator": pd_api, "total_iterations": 0,
                        "llm_generated_cases": 0, "successful_cases": 0,
                        "status": "no_results",
                    })
                    log_file.write(f"[{idx}/{len(operator_names)}] {pd_api}\n")
                    log_file.write(f"  状态: ⚠️ 无结果\n\n")
                    log_file.flush()

            except Exception as e:
                print(f"\n❌ {pd_api} 测试失败: {e}")
                all_operators_summary.append({
                    "operator": pd_api, "total_iterations": 0,
                    "llm_generated_cases": 0, "successful_cases": 0,
                    "status": "failed", "error": str(e),
                })
                log_file.write(f"[{idx}/{len(operator_names)}] {pd_api}\n")
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
