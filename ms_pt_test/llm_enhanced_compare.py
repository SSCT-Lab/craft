#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4: LLM-based MindSpore vs PyTorch operator differential testing framework

Purpose:
- Load MS test cases and MS->PT mappings from JSON files
- Execute each operator pair in MindSpore and PyTorch and compare results
- Use the LLM to repair/mutate/skip test cases
- LLM calls are concurrent; operator execution is sequential to keep inputs aligned
- Save detailed test results and batch logs

Usage:
    conda activate tf_env
    python ms_pt_test/llm_enhanced_compare.py \
        [--max-iterations 3] [--num-cases 3] [--workers 6] \
        [--start 1] [--end N] [--operators mindspore.ops.Abs mindspore.ops.Add]

Prerequisites:
    1. Run Step 1 extract_ms_apis.py
    2. Run Step 2 extract_ms_test_cases.py
    3. Run Step 3 extract_ms_pt_mapping.py
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
from openai import OpenAI

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from component.doc.doc_crawler_factory import get_doc_content

# ==================== Constants ====================
DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_NUM_CASES = 5
DEFAULT_WORKERS = 6

DATA_DIR = os.path.join(ROOT_DIR, "ms_pt_test", "data")
DEFAULT_TEST_CASES_FILE = os.path.join(DATA_DIR, "ms_test_cases.json")
DEFAULT_MAPPING_FILE = os.path.join(DATA_DIR, "ms_pt_mapping_validated.csv")


# ==================== MindSpore lazy load ====================
_mindspore = None
_ms_context_set = False


def get_mindspore():
    """Lazy-load MindSpore and set context."""
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


class LLMEnhancedComparator:
    """LLM-based MindSpore vs PyTorch differential testing framework."""

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
        self._safe_print(f"📋 Loaded {len(self.test_cases_data)} MS API test cases")

        self.api_mapping = self._load_mapping(mapping_file)
        has_impl = sum(1 for v in self.api_mapping.values() if v != "no_matching_impl")
        self._safe_print(f"📋 Loaded {len(self.api_mapping)} mappings ({has_impl} with matching impl)")

        self.result_dir = os.path.join(ROOT_DIR, "ms_pt_test", "ms_pt_log_1")
        os.makedirs(self.result_dir, exist_ok=True)
        self._safe_print(f"📁 Result directory: {self.result_dir}")

        self.random_seed = 42
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    # ==================== Helpers ====================

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
        self._safe_print("❌ API key not found")
        return ""

    def _load_test_cases(self, filepath: str) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            self._safe_print(f"⚠️ Test case file does not exist: {filepath}")
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("test_cases", {})

    def _load_mapping(self, filepath: str) -> Dict[str, str]:
        if not os.path.exists(filepath):
            self._safe_print(f"⚠️ Mapping file does not exist: {filepath}")
            return {}
        mapping = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ms_api = row.get("mindspore-api", "").strip()
                pt_api = row.get("pytorch-api", "").strip()
                if ms_api and pt_api:
                    mapping[ms_api] = pt_api
        return mapping

    # ==================== API helpers ====================

    def is_class_based_api(self, api_name: str) -> bool:
        """Check whether an API is class-based (capitalized)."""
        parts = api_name.split(".")
        if len(parts) >= 2:
            last_part = parts[-1]
            if last_part and last_part[0].isupper():
                return True
        return False

    def get_ms_function(self, api_name: str):
        """Get MindSpore operator function/class."""
        ms = get_mindspore()
        try:
            parts = api_name.split(".")
            # parts[0] = "mindspore"
            obj = ms
            for part in parts[1:]:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            # Try ops.operations as a fallback
            try:
                obj = ms.ops.operations
                op_name = parts[-1]
                return getattr(obj, op_name)
            except AttributeError:
                pass
            return None

    def get_pt_function(self, api_name: str):
        """Get PyTorch operator function/class."""
        try:
            parts = api_name.split(".")
            # parts[0] = "torch"
            obj = torch
            for part in parts[1:]:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return None

    def convert_api_name(self, ms_api: str) -> Tuple[Optional[str], Optional[str], str]:
        """Find the PyTorch API for the given MindSpore API."""
        if ms_api in self.api_mapping:
            pt_api = self.api_mapping[ms_api]
            if pt_api and pt_api != "no_matching_impl":
                return ms_api, pt_api, "mapping_table"
            else:
                return ms_api, None, "no_matching_impl"
        return ms_api, None, "not_found_in_mapping"

    # ==================== Data conversion ====================

    def generate_numpy_data(self, data: Any) -> np.ndarray:
        """Generate a numpy array from a descriptor."""
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

                for prefix in ["torch.", "mindspore.", "ms.", "np.", "numpy."]:
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
        """Convert to a MindSpore tensor."""
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

    def convert_to_pt_tensor(self, data: Any, numpy_data: np.ndarray = None) -> torch.Tensor:
        """Convert to a PyTorch tensor."""
        if numpy_data is not None:
            return torch.from_numpy(numpy_data.copy())
        if isinstance(data, dict):
            np_data = self.generate_numpy_data(data)
            return torch.from_numpy(np_data.copy())
        if isinstance(data, (int, float)):
            return torch.tensor(data)
        if isinstance(data, list):
            return torch.tensor(data)
        return torch.tensor(data)

    # ==================== Argument preparation ====================

    def should_skip_param(self, key: str, api_name: str, framework: str) -> bool:
        """Decide whether to skip a parameter."""
        common_skip = {"description", "api", "init_params", "is_class_api"}
        if key in common_skip:
            return True

        torch_skip = {"layout", "requires_grad", "out", "memory_format", "pin_memory"}
        if framework == "torch" and key in torch_skip:
            return True

        ms_skip = {"name"}
        if framework == "ms" and key in ms_skip:
            return True

        return False

    def prepare_arguments(
        self, test_case: Dict[str, Any], framework: str = "ms"
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Prepare arguments for a target framework.

        Args:
            test_case: Test case (with shared numpy data)
            framework: "ms" or "torch"

        Returns:
            (args, kwargs)
        """
        args = []
        kwargs = {}

        ms = get_mindspore() if framework == "ms" else None

        def convert_value(value: Any) -> Any:
            if isinstance(value, dict):
                if "shape" in value:
                    np_data = self.generate_numpy_data(value)
                    if framework == "torch":
                        return torch.from_numpy(np_data.copy())
                    return ms.Tensor(np_data)
                return {k: convert_value(v) for k, v in value.items()}
            if isinstance(value, np.ndarray):
                if framework == "torch":
                    return torch.from_numpy(value.copy())
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
            for prefix in ["torch.", "mindspore.", "ms.", "np.", "numpy."]:
                if token.startswith(prefix):
                    token = token[len(prefix):]
            if framework == "torch":
                return getattr(torch, token, dtype_value)
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

        # Variadic argument handling
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
                            args.append(ms.Tensor(np_data))
                    else:
                        args.append(item)
            return args, kwargs

        # Process positional parameters in order
        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if param_name == "dtype":
                    args.append(normalize_dtype(value))
                else:
                    args.append(convert_value(value))

        # Process keyword arguments
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

    # ==================== Result comparison ====================

    def compare_tensors(
        self, ms_result, torch_result, tolerance: float = 1e-5
    ) -> Tuple[bool, str]:
        """Compare computation results between MindSpore and PyTorch."""
        try:
            # Convert to numpy
            if hasattr(ms_result, 'asnumpy'):
                ms_np = ms_result.asnumpy()
            elif isinstance(ms_result, np.ndarray):
                ms_np = ms_result
            else:
                ms_np = np.array(ms_result)

            if isinstance(torch_result, torch.Tensor):
                torch_np = torch_result.detach().cpu().numpy()
            elif isinstance(torch_result, np.ndarray):
                torch_np = torch_result
            else:
                torch_np = np.array(torch_result)

            # Shape consistency check
            if ms_np.shape != torch_np.shape:
                return False, f"Shape mismatch: MS={ms_np.shape} vs PT={torch_np.shape}"

            # Exact comparison for boolean types
            if ms_np.dtype == np.bool_ or torch_np.dtype == np.bool_:
                match = np.array_equal(ms_np, torch_np)
                if match:
                    return True, "Boolean results match exactly"
                else:
                    diff_count = np.sum(ms_np != torch_np)
                    return False, f"Boolean results differ; diff count: {diff_count}"

            # Numeric comparison
            if np.allclose(ms_np, torch_np, atol=tolerance, rtol=tolerance, equal_nan=True):
                return True, "Results match (within tolerance)"
            else:
                max_diff = np.max(
                    np.abs(ms_np.astype(np.float64) - torch_np.astype(np.float64))
                )
                return False, f"Results differ; max diff: {max_diff:.8f}"

        except Exception as e:
            return False, f"Comparison error: {str(e)}"

    # ==================== Test execution ====================

    def execute_test_case(
        self,
        ms_api: str,
        pytorch_api: str,
        ms_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Execute a single test case."""
        def pick_first(case: Dict[str, Any], *keys: str):
            for key in keys:
                if isinstance(case, dict) and key in case and case[key] is not None:
                    return case[key]
            return None

        if pytorch_test_case is None:
            pytorch_test_case = ms_test_case

        effective_ms_api = ms_test_case.get("api", ms_api) if isinstance(ms_test_case, dict) else ms_api
        effective_pt_api = (
            pytorch_test_case.get("api", pytorch_api)
            if isinstance(pytorch_test_case, dict)
            else pytorch_api
        )

        result = {
            "ms_api": effective_ms_api,
            "pytorch_api": effective_pt_api,
            "ms_success": False,
            "pytorch_success": False,
            "results_match": False,
            "ms_error": None,
            "pytorch_error": None,
            "comparison_error": None,
            "ms_shape": None,
            "pytorch_shape": None,
            "ms_dtype": None,
            "pytorch_dtype": None,
            "status": "unknown",
        }

        # Materialize shared tensors
        ms_test_case, pytorch_test_case = self._materialize_shared_tensors(
            effective_ms_api, effective_pt_api, ms_test_case, pytorch_test_case
        )

        is_class_ms = self.is_class_based_api(effective_ms_api)
        is_class_pt = self.is_class_based_api(effective_pt_api)

        # ---- Execute MindSpore ----
        ms_result = None
        try:
            ms_func = self.get_ms_function(effective_ms_api)
            if ms_func is None:
                raise AttributeError(f"MS API not found: {effective_ms_api}")

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

                # Get input
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
                # Functional API / Tensor method
                if "Tensor" in effective_ms_api:
                    # Tensor method: tensor.method(args)
                    method_name = effective_ms_api.split(".")[-1]
                    input_data = pick_first(ms_test_case, "x", "input")
                    if isinstance(input_data, np.ndarray):
                        ms_tensor = self.convert_to_ms_tensor(input_data)
                    else:
                        input_desc = input_data if input_data is not None else {"shape": [2, 3], "dtype": "float32"}
                        np_data = self.generate_numpy_data(input_desc)
                        ms_tensor = self.convert_to_ms_tensor(np_data)

                    method = getattr(ms_tensor, method_name)
                    # Collect parameters besides x/input
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

        # ---- Execute PyTorch ----
        torch_result = None
        try:
            pt_func = self.get_pt_function(effective_pt_api)
            if pt_func is None:
                raise AttributeError(f"PT API not found: {effective_pt_api}")

            if is_class_pt:
                init_kwargs = {
                    k: v for k, v in pytorch_test_case.items()
                    if k not in ["api", "input", "x", "init_params"]
                    and not isinstance(v, np.ndarray)
                    and not (isinstance(v, dict) and "shape" in v)
                }
                if isinstance(pytorch_test_case.get("init_params"), dict):
                    init_kwargs.update(pytorch_test_case["init_params"])
                module = pt_func(**init_kwargs)

                input_data = pick_first(pytorch_test_case, "input", "x")
                if input_data is not None:
                    if isinstance(input_data, dict) and "shape" in input_data:
                        np_data = self.generate_numpy_data(input_data)
                        pt_input = self.convert_to_pt_tensor(None, np_data)
                    elif isinstance(input_data, np.ndarray):
                        pt_input = self.convert_to_pt_tensor(input_data)
                    else:
                        pt_input = self.convert_to_pt_tensor(input_data)
                    torch_result = module(pt_input)
                else:
                    torch_result = module(torch.randn(2, 3))
            else:
                if "Tensor" in effective_pt_api:
                    method_name = effective_pt_api.split(".")[-1]
                    input_data = pick_first(pytorch_test_case, "x", "input")
                    if isinstance(input_data, np.ndarray):
                        pt_tensor = torch.from_numpy(input_data.copy())
                    else:
                        input_desc = input_data if input_data is not None else {"shape": [2, 3], "dtype": "float32"}
                        np_data = self.generate_numpy_data(input_desc)
                        pt_tensor = torch.from_numpy(np_data.copy())

                    method = getattr(pt_tensor, method_name)
                    other_args = []
                    other_kwargs = {}
                    for key, value in pytorch_test_case.items():
                        if key in {"x", "input", "api", "init_params", "is_class_api", "description"}:
                            continue
                        if isinstance(value, np.ndarray):
                            other_args.append(torch.from_numpy(value.copy()))
                        elif isinstance(value, dict) and "shape" in value:
                            np_data = self.generate_numpy_data(value)
                            other_args.append(torch.from_numpy(np_data.copy()))
                        elif key in {"y", "other", "b"}:
                            other_args.append(value)
                        else:
                            other_kwargs[key] = value
                    torch_result = method(*other_args, **other_kwargs)
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

        # ---- Compare results ----
        if result["ms_success"] and result["pytorch_success"]:
            try:
                match, detail = self.compare_tensors(ms_result, torch_result)
                result["results_match"] = match
                result["comparison_error"] = None if match else detail
                result["status"] = "consistent" if match else "inconsistent"
            except Exception as e:
                result["comparison_error"] = str(e)
                result["status"] = "comparison_error"
        elif result["ms_success"] and not result["pytorch_success"]:
            result["status"] = "pytorch_error"
        elif not result["ms_success"] and result["pytorch_success"]:
            result["status"] = "mindspore_error"
        else:
            result["status"] = "both_error"

        return result

    def _execute_test_case_sequential(
        self, ms_api, pytorch_api, ms_test_case, pytorch_test_case=None
    ) -> Dict[str, Any]:
        """Use a lock to keep execution sequential."""
        with self.execution_lock:
            return self.execute_test_case(ms_api, pytorch_api, ms_test_case, pytorch_test_case)

    def _materialize_shared_tensors(
        self,
        ms_api: str,
        pytorch_api: str,
        ms_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Materialize shared tensors to keep inputs aligned across frameworks."""
        ms_case = copy.deepcopy(ms_test_case)
        pt_case = copy.deepcopy(pytorch_test_case)

        is_class_ms = self.is_class_based_api(ms_api)
        is_class_pt = self.is_class_based_api(pytorch_api)
        if (is_class_ms or is_class_pt) and not (
            "input" in ms_case or "x" in ms_case
            or "input" in pt_case or "x" in pt_case
        ):
            default_desc = self._default_input_desc_for_class(ms_api or pytorch_api)
            ms_case.setdefault("input", default_desc)
            pt_case.setdefault("input", default_desc)

        def is_tensor_desc(value: Any) -> bool:
            return isinstance(value, dict) and "shape" in value

        def clone_array(value: np.ndarray) -> np.ndarray:
            return value.copy()

        def materialize_pair(ms_val: Any, pt_val: Any) -> Tuple[Any, Any]:
            if isinstance(ms_val, np.ndarray):
                return clone_array(ms_val), clone_array(ms_val)
            if isinstance(pt_val, np.ndarray):
                return clone_array(pt_val), clone_array(pt_val)

            if is_tensor_desc(ms_val) or is_tensor_desc(pt_val):
                tensor_desc = ms_val if is_tensor_desc(ms_val) else pt_val
                shared = self.generate_numpy_data(tensor_desc)
                return clone_array(shared), clone_array(shared)

            if isinstance(ms_val, list) or isinstance(pt_val, list):
                ms_list = ms_val if isinstance(ms_val, list) else []
                pt_list = pt_val if isinstance(pt_val, list) else []
                size = max(len(ms_list), len(pt_list))
                out_ms = []
                out_pt = []
                for index in range(size):
                    left = ms_list[index] if index < len(ms_list) else None
                    right = pt_list[index] if index < len(pt_list) else None
                    new_left, new_right = materialize_pair(left, right)
                    if index < len(ms_list):
                        out_ms.append(new_left)
                    if index < len(pt_list):
                        out_pt.append(new_right)
                return (
                    out_ms if isinstance(ms_val, list) else ms_val,
                    out_pt if isinstance(pt_val, list) else pt_val,
                )

            if isinstance(ms_val, dict) or isinstance(pt_val, dict):
                ms_dict = ms_val if isinstance(ms_val, dict) else {}
                pt_dict = pt_val if isinstance(pt_val, dict) else {}
                keys = set(ms_dict.keys()) | set(pt_dict.keys())
                out_ms = {}
                out_pt = {}
                for key in keys:
                    if key == "api":
                        if key in ms_dict:
                            out_ms[key] = ms_dict[key]
                        if key in pt_dict:
                            out_pt[key] = pt_dict[key]
                        continue
                    new_left, new_right = materialize_pair(ms_dict.get(key), pt_dict.get(key))
                    if key in ms_dict:
                        out_ms[key] = new_left
                    if key in pt_dict:
                        out_pt[key] = new_right
                return (
                    out_ms if isinstance(ms_val, dict) else ms_val,
                    out_pt if isinstance(pt_val, dict) else pt_val,
                )

            return ms_val, pt_val

        ms_case, pt_case = materialize_pair(ms_case, pt_case)

        if isinstance(ms_case, dict) and isinstance(pt_case, dict):
            alias_pairs = [
                ("x", "input"),
                ("input", "x"),
                ("y", "other"),
                ("other", "y"),
            ]
            for ms_key, pt_key in alias_pairs:
                if ms_key in ms_case and pt_key in pt_case:
                    ms_item, pt_item = materialize_pair(ms_case[ms_key], pt_case[pt_key])
                    ms_case[ms_key] = ms_item
                    pt_case[pt_key] = pt_item

        return ms_case, pt_case

    def _default_input_desc_for_class(self, api_name: str) -> Dict[str, Any]:
        name = (api_name or "").lower()
        if "3d" in name:
            return {"shape": [2, 3, 4, 4, 4], "dtype": "float32"}
        if "2d" in name:
            return {"shape": [2, 3, 8, 8], "dtype": "float32"}
        if "1d" in name:
            return {"shape": [2, 3, 10], "dtype": "float32"}
        return {"shape": [2, 3], "dtype": "float32"}

    # ==================== API doc crawling ====================

    def _fetch_api_docs(self, ms_api: str, pytorch_api: str) -> Tuple[str, str]:
        MIN_DOC_LENGTH = 300
        ms_doc = ""
        pytorch_doc = ""

        try:
            raw = get_doc_content(ms_api, "mindspore")
            if raw and len(raw) >= MIN_DOC_LENGTH:
                ms_doc = raw[:3000]
                self._safe_print(f"    📄 MS doc: {len(ms_doc)} chars")
            else:
                self._safe_print(f"    📄 MS doc: no valid content")
        except Exception as e:
            self._safe_print(f"    ⚠️ MS doc crawl failed: {str(e)[:50]}")

        try:
            raw = get_doc_content(pytorch_api, "pytorch")
            if raw and len(raw) >= MIN_DOC_LENGTH:
                pytorch_doc = raw[:3000]
                self._safe_print(f"    📄 PT doc: {len(pytorch_doc)} chars")
            else:
                self._safe_print(f"    📄 PT doc: no valid content")
        except Exception as e:
            self._safe_print(f"    ⚠️ PT doc crawl failed: {str(e)[:50]}")

        return ms_doc, pytorch_doc

    # ==================== LLM interaction ====================

    def _build_llm_prompt(
        self,
        execution_result: Dict[str, Any],
        ms_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any],
        ms_doc: str = "",
        pytorch_doc: str = "",
    ) -> str:
        ms_api = execution_result.get("ms_api", "")
        pytorch_api = execution_result.get("pytorch_api", "")
        status = execution_result.get("status", "")
        ms_success = execution_result.get("ms_success", False)
        pytorch_success = execution_result.get("pytorch_success", False)
        results_match = execution_result.get("results_match", False)
        ms_error = execution_result.get("ms_error", "")
        pytorch_error = execution_result.get("pytorch_error", "")
        comparison_error = execution_result.get("comparison_error", "")

        # Simplify test cases
        def simplify_case(case):
            simplified = {}
            for key, value in case.items():
                if isinstance(value, np.ndarray):
                    simplified[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
                else:
                    simplified[key] = value
            return simplified

        simplified_ms = simplify_case(ms_test_case)
        simplified_pt = simplify_case(pytorch_test_case)

        # Doc section
        doc_section = ""
        if ms_doc or pytorch_doc:
            doc_section = "\n## Official API docs reference\n\n"
            if ms_doc:
                doc_section += f"### MindSpore {ms_api} docs\n```\n{ms_doc}\n```\n\n"
            if pytorch_doc:
                doc_section += f"### PyTorch {pytorch_api} docs\n```\n{pytorch_doc}\n```\n\n"

        # Parameter examples
        def build_param_str(simplified):
            examples = []
            for key, value in simplified.items():
                if key in {"api", "init_params", "is_class_api", "description"}:
                    continue
                examples.append(f'    "{key}": {json.dumps(value, ensure_ascii=False)}')
            return ",\n".join(examples) if examples else '    "x": {"shape": [2, 3], "dtype": "float32"}'

        ms_param_str = build_param_str(simplified_ms)
        pt_param_str = build_param_str(simplified_pt)

        prompt = f"""Analyze the following operator test cases executed in MindSpore and PyTorch, then repair or mutate (fuzz) the test cases based on the results.

## Test info
- **MindSpore API**: {ms_api}
- **PyTorch API**: {pytorch_api}
{doc_section}
## Execution results
- **Status**: {status}
- **MindSpore success**: {ms_success}
- **PyTorch success**: {pytorch_success}
- **Results match**: {results_match}

## Error info
- **MindSpore error**: {ms_error if ms_error else "none"}
- **PyTorch error**: {pytorch_error if pytorch_error else "none"}
- **Comparison error**: {comparison_error if comparison_error else "none"}

## Original test cases

### MindSpore test case
```json
{json.dumps(simplified_ms, indent=2, ensure_ascii=False)}
```

### PyTorch test case
```json
{json.dumps(simplified_pt, indent=2, ensure_ascii=False)}
```

## Task requirements
Based on the info above (including official API docs), decide whether the results are **consistent**, **inconsistent**, or **execution error**, then follow these rules:

1. **If consistent**: **mutate (fuzz)** the test case, e.g., change input shapes or parameter values, focusing on edge/extreme cases.
2. **If execution error**: **repair** (adjust parameter names/types/ranges) or **skip** (docs missing, operator removed, or semantics not equivalent).
3. **If inconsistent**: check if it is tolerable numerical error (<= 1e-3). If tolerable, **mutate**; if semantics are not equivalent, **skip**; otherwise **repair**.

## MindSpore API call notes
- MindSpore Primitive ops (e.g., mindspore.ops.Abs) must be instantiated: `op = ops.Abs(); result = op(input)`
- MindSpore functional APIs (e.g., mindspore.ops.abs) are called directly: `result = ops.abs(input)`
- MindSpore NN layers (e.g., mindspore.nn.Conv2d) are similar to PyTorch: `layer = nn.Conv2d(...); result = layer(input)`
- MindSpore and PyTorch parameter names/semantics may differ and need alignment

## Output format
Return JSON only in the following format, with no extra text:

{{
  "operation": "mutation",
    "reason": "Detailed reason for this operation (<= 150 words)",
  "mindspore_test_case": {{
    "api": "{ms_api}",
{ms_param_str}
  }},
  "pytorch_test_case": {{
    "api": "{pytorch_api}",
{pt_param_str}
  }}
}}

**Important**:
1. `operation` must be one of "mutation", "repair", or "skip"
2. Tensor parameters must use {{"shape": [...], "dtype": "..."}} format
3. Scalar parameters should be plain values
4. Inputs must be identical across frameworks and parameters must align semantically
5. MindSpore and PyTorch test cases may differ in parameter names/values/counts as long as the expected outputs match
6. If no official docs exist or the operator was removed, set `operation` to "skip" and do not attempt repair
7. Prefer extreme cases when mutating: empty tensors, single-element tensors, high-rank tensors, different dtypes, boundary values
8. Follow the official docs for parameter names/types/ranges
9. Mind the default data layout semantics and set them explicitly if needed
"""
        return prompt

    def call_llm_for_repair_or_mutation(
        self,
        execution_result: Dict[str, Any],
        ms_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any],
        ms_doc: str = "",
        pytorch_doc: str = "",
    ) -> Dict[str, Any]:
        """Call the LLM to repair or mutate test cases."""
        prompt = self._build_llm_prompt(
            execution_result, ms_test_case, pytorch_test_case, ms_doc, pytorch_doc
        )
        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a deep learning framework testing expert with strong knowledge of MindSpore and PyTorch API differences."
                            "Based on the execution results, decide whether to repair or mutate the test case,"
                            "and return a strict JSON-only response."
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
                self._safe_print("    ⚠️ LLM response is not valid JSON; attempting to extract...")
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return {
                    "operation": "skip",
                    "reason": "LLM returned invalid format",
                    "mindspore_test_case": ms_test_case,
                    "pytorch_test_case": pytorch_test_case,
                }

        except Exception as e:
            self._safe_print(f"    ❌ LLM call failed: {e}")
            return {
                "operation": "skip",
                "reason": f"LLM call failed: {e}",
                "mindspore_test_case": ms_test_case,
                "pytorch_test_case": pytorch_test_case,
            }

    # ==================== Core test loop ====================

    def llm_enhanced_test_operator(
        self,
        ms_api: str,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        num_test_cases: int = None,
        num_workers: int = DEFAULT_WORKERS,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Test a single operator pair with LLM enhancement."""
        self._safe_print(f"\n{'=' * 80}")
        self._safe_print(f"🎯 Start testing operator: {ms_api}")
        self._safe_print(f"🔄 Max iterations per case: {max_iterations}")
        self._safe_print(f"{'=' * 80}\n")

        stats = {"llm_generated_cases": 0, "successful_cases": 0}

        _, pytorch_api, mapping_method = self.convert_api_name(ms_api)
        if pytorch_api is None:
            self._safe_print(f"❌ {ms_api} has no matching PyTorch implementation")
            return [], stats

        self._safe_print(f"✅ MindSpore API: {ms_api}")
        self._safe_print(f"✅ PyTorch API: {pytorch_api}")
        self._safe_print(f"✅ Mapping method: {mapping_method}")

        api_data = self.test_cases_data.get(ms_api, {})
        test_cases = api_data.get("test_cases", [])

        if not test_cases:
            self._safe_print(f"⚠️ No test cases found for {ms_api}; using defaults")
            test_cases = [
                {"description": "default", "inputs": {"x": {"shape": [2, 3], "dtype": "float32"}}}
            ]

        if num_test_cases is None:
            num_test_cases = len(test_cases)
        else:
            num_test_cases = min(num_test_cases, len(test_cases))

        self._safe_print(
            f"📋 Testing {num_test_cases} cases (LLM workers={num_workers}, sequential execution)"
        )

        # Prepare initial cases
        initial_cases = []
        for case_idx in range(num_test_cases):
            tc = test_cases[case_idx]
            if "inputs" in tc:
                flat_case = dict(tc["inputs"])
            else:
                flat_case = {k: v for k, v in tc.items() if k != "description"}
            flat_case["api"] = ms_api
            # Preserve init_params
            if "init_params" in api_data:
                flat_case["init_params"] = api_data["init_params"]
            elif "init_params" in tc:
                flat_case["init_params"] = tc["init_params"]
            initial_cases.append((case_idx + 1, flat_case))

        all_results = []

        # Multi-round iteration per case
        # Use ThreadPoolExecutor for LLM calls; execution stays sequential
        if num_workers <= 1:
            for case_number, initial_test_case in initial_cases:
                self._safe_print(f"\n📋 Case {case_number}/{num_test_cases}")
                case_results = self._test_single_case_with_iterations(
                    ms_api, pytorch_api, initial_test_case,
                    max_iterations, case_number, stats,
                )
                all_results.extend(case_results)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_case = {}
                for case_number, initial_test_case in initial_cases:
                    future = executor.submit(
                        self._test_single_case_with_iterations,
                        ms_api, pytorch_api, initial_test_case,
                        max_iterations, case_number, stats,
                    )
                    future_to_case[future] = case_number

                for future in as_completed(future_to_case):
                    case_results = future.result()
                    all_results.extend(case_results)

        all_results.sort(key=lambda r: (r.get("case_number", 0), r.get("iteration", 0)))

        self._safe_print(f"\n{'=' * 80}")
        self._safe_print("✅ All tests completed")
        self._safe_print(
            f"📊 Tested {num_test_cases} cases, total {len(all_results)} iterations"
        )
        self._safe_print(f"📊 LLM-generated test cases: {stats['llm_generated_cases']}")
        self._safe_print(f"📊 Cases where both frameworks succeeded: {stats['successful_cases']}")
        self._safe_print(f"{'=' * 80}\n")

        return all_results, stats

    def _test_single_case_with_iterations(
        self,
        ms_api: str,
        pytorch_api: str,
        initial_test_case: Dict[str, Any],
        max_iterations: int,
        case_number: int,
        stats: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Run multiple iterations for a single test case."""
        case_results = []

        current_ms_test_case = copy.deepcopy(initial_test_case)
        current_ms_test_case["api"] = ms_api

        current_pt_test_case = copy.deepcopy(initial_test_case)
        current_pt_test_case["api"] = pytorch_api

        is_llm_generated = False

        self._safe_print("  📖 Pre-fetching API docs...")
        ms_doc, pytorch_doc = self._fetch_api_docs(ms_api, pytorch_api)

        for iteration in range(max_iterations):
            source_type = "LLM" if is_llm_generated else "file"
            self._safe_print(
                f"  🔄 Iteration {iteration + 1}/{max_iterations} ({source_type})", end=""
            )

            current_ms_api = current_ms_test_case.get("api", ms_api) or ms_api
            current_pt_api = current_pt_test_case.get("api", pytorch_api) or pytorch_api

            try:
                execution_result = self._execute_test_case_sequential(
                    current_ms_api, current_pt_api, current_ms_test_case, current_pt_test_case
                )

                ms_status = "✓" if execution_result["ms_success"] else "✗"
                pt_status = "✓" if execution_result["pytorch_success"] else "✗"
                match_status = "✓" if execution_result["results_match"] else "✗"
                self._safe_print(f" | MS:{ms_status} PT:{pt_status} Match:{match_status}")

                if execution_result["ms_error"] and not execution_result["ms_success"]:
                    self._safe_print(
                        f"    ❌ MS error: {str(execution_result['ms_error'])[:100]}..."
                    )
                if execution_result["pytorch_error"] and not execution_result["pytorch_success"]:
                    self._safe_print(
                        f"    ❌ PT error: {str(execution_result['pytorch_error'])[:100]}..."
                    )
                if execution_result["comparison_error"]:
                    self._safe_print(
                        f"    ⚠️ Comparison: {str(execution_result['comparison_error'])[:100]}..."
                    )

                if is_llm_generated:
                    if execution_result["ms_success"] and execution_result["pytorch_success"]:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

            except Exception as e:
                self._safe_print(f" | ❌ Fatal error: {str(e)[:80]}...")
                execution_result = {
                    "status": "fatal_error",
                    "ms_success": False, "pytorch_success": False,
                    "results_match": False,
                    "ms_error": f"Fatal: {str(e)}", "pytorch_error": None,
                    "comparison_error": None,
                }

            iteration_result = {
                "iteration": iteration + 1,
                "ms_test_case": current_ms_test_case,
                "pytorch_test_case": current_pt_test_case,
                "execution_result": execution_result,
                "llm_operation": None,
                "is_llm_generated": is_llm_generated,
                "case_number": case_number,
            }

            # Call LLM
            try:
                llm_result = self.call_llm_for_repair_or_mutation(
                    execution_result, current_ms_test_case, current_pt_test_case,
                    ms_doc, pytorch_doc,
                )
            except Exception as e:
                self._safe_print(f"    ❌ LLM call failed: {str(e)[:80]}...")
                llm_result = {"operation": "skip", "reason": f"LLM call failed: {str(e)}"}
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
                next_pt_case = llm_result.get("pytorch_test_case", current_pt_test_case)
                with self.stats_lock:
                    stats["llm_generated_cases"] += 1
                is_llm_generated = True
            else:
                next_ms_case = current_ms_test_case
                next_pt_case = current_pt_test_case

            current_ms_test_case, current_pt_test_case = self._convert_llm_test_cases(
                next_ms_case, next_pt_case
            )

        # If the last LLM round generated a new case, execute it once
        if case_results:
            last_op = case_results[-1].get("llm_operation", {}).get("operation", "skip")
            if last_op in ("mutation", "repair"):
                self._safe_print("  🔄 Executing final LLM case", end="")
                try:
                    execution_result = self._execute_test_case_sequential(
                        current_ms_test_case.get("api", ms_api) or ms_api,
                        current_pt_test_case.get("api", pytorch_api) or pytorch_api,
                        current_ms_test_case,
                        current_pt_test_case,
                    )
                    ms_s = "✓" if execution_result["ms_success"] else "✗"
                    pt_s = "✓" if execution_result["pytorch_success"] else "✗"
                    m_s = "✓" if execution_result["results_match"] else "✗"
                    self._safe_print(f" | MS:{ms_s} PT:{pt_s} Match:{m_s}")

                    if execution_result["ms_success"] and execution_result["pytorch_success"]:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

                    case_results.append({
                        "iteration": len(case_results) + 1,
                        "ms_test_case": current_ms_test_case,
                        "pytorch_test_case": current_pt_test_case,
                        "execution_result": execution_result,
                        "llm_operation": {
                            "operation": "final_execution",
                            "reason": "execute final LLM-generated case",
                        },
                        "case_number": case_number,
                        "is_llm_generated": True,
                    })
                except Exception as e:
                    self._safe_print(f"  ❌ Final case execution failed: {str(e)[:80]}...")

        self._safe_print(f"  ✅ Case {case_number} complete, {len(case_results)} iterations")
        return case_results

    def _convert_llm_test_cases(
        self,
        ms_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Convert LLM test cases to executable format and share tensors."""
        ms_api = ms_test_case.get("api", "") if isinstance(ms_test_case, dict) else ""
        pt_api = pytorch_test_case.get("api", "") if isinstance(pytorch_test_case, dict) else ""
        return self._materialize_shared_tensors(ms_api, pt_api, ms_test_case, pytorch_test_case)

    # ==================== Results saving ====================

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
            for case_key in ["ms_test_case", "pytorch_test_case"]:
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
            "pytorch_api": self.api_mapping.get(ms_api, ""),
            "timestamp": datetime.now().isoformat(),
            "total_iterations": len(results),
            "llm_generated_test_cases": stats.get("llm_generated_cases", 0) if stats else 0,
            "successful_test_cases": stats.get("successful_cases", 0) if stats else 0,
            "results": output_results,
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self._safe_print(f"💾 Results saved to: {filepath}")

    def get_all_testable_apis(self) -> List[str]:
        testable = []
        for ms_api in sorted(self.test_cases_data.keys()):
            pt_api = self.api_mapping.get(ms_api, "no_matching_impl")
            if pt_api and pt_api != "no_matching_impl":
                testable.append(ms_api)
        return testable

    def close(self):
        pass


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="LLM-based MindSpore vs PyTorch operator differential testing framework"
    )
    parser.add_argument(
        "--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"Max iterations per test case (default {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES,
        help=f"Test cases per operator (default {DEFAULT_NUM_CASES})",
    )
    parser.add_argument(
        "--start", type=int, default=1,
        help="Start operator index (1-based, default 1)",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="End operator index (inclusive, default all)",
    )
    parser.add_argument(
        "--operators", "-o", nargs="*",
        help="Operator names to test (e.g., mindspore.ops.Abs)",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"Worker threads (default {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"LLM model name (default {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--key-path", "-k", default=DEFAULT_KEY_PATH,
        help=f"API key file path (default {DEFAULT_KEY_PATH})",
    )
    parser.add_argument(
        "--test-cases-file", default=DEFAULT_TEST_CASES_FILE,
        help="Test case JSON file path",
    )
    parser.add_argument(
        "--mapping-file", default=DEFAULT_MAPPING_FILE,
        help="MS->PT mapping CSV file path",
    )

    args = parser.parse_args()
    num_workers = max(1, args.workers)

    print("=" * 80)
    print("LLM-based MindSpore vs PyTorch operator differential testing framework")
    print("=" * 80)
    print(f"📌 Iterations per operator: {args.max_iterations}")
    print(f"📌 Test cases per operator: {args.num_cases}")
    print(f"📌 LLM worker threads: {num_workers}")
    print(f"📌 LLM model: {args.model}")
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
        print(f"\n🔍 Total testable MS APIs: {len(all_testable)}")

        if args.operators:
            operator_names = args.operators
            print(f"📋 Selected operators: {len(operator_names)}")
        else:
            start_idx = max(1, args.start) - 1
            end_idx = args.end if args.end is not None else len(all_testable)
            end_idx = min(end_idx, len(all_testable))
            if start_idx >= end_idx:
                raise ValueError(f"Start index {args.start} must be less than end index {end_idx}")
            operator_names = all_testable[start_idx:end_idx]
            print(f"📌 Test range: operator {start_idx + 1} to {end_idx}")
            print(f"📋 Operators to test: {len(operator_names)}")

        print(
            f"📋 Operator list: "
            f"{', '.join(operator_names[:10])}{'...' if len(operator_names) > 10 else ''}\n"
        )

        all_operators_summary = []

        batch_log_file = os.path.join(
            comparator.result_dir,
            f"batch_test_log_{start_datetime.strftime('%Y%m%d_%H%M%S')}.txt",
        )
        log_file = open(batch_log_file, 'w', encoding='utf-8')
        log_file.write("=" * 80 + "\n")
        log_file.write("MS->PT batch differential test log\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("Test config:\n")
        log_file.write(f"  - Iterations: {args.max_iterations}\n")
        log_file.write(f"  - Cases: {args.num_cases}\n")
        log_file.write(f"  - Workers: {num_workers}\n")
        log_file.write(f"  - Operators: {len(operator_names)}\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.flush()

        for idx, ms_api in enumerate(operator_names, 1):
            print("\n" + "🔷" * 40)
            print(f"🎯 [{idx}/{len(operator_names)}] Start testing operator: {ms_api}")
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
                        "pytorch_api": comparator.api_mapping.get(ms_api, ""),
                        "total_iterations": len(results),
                        "llm_generated_cases": stats.get("llm_generated_cases", 0),
                        "successful_cases": stats.get("successful_cases", 0),
                        "status": "completed",
                    })

                    print(f"\n✅ {ms_api} completed")
                    print(f"   - Total iterations: {len(results)}")
                    print(f"   - LLM-generated cases: {stats.get('llm_generated_cases', 0)}")
                    print(f"   - Successful cases: {stats.get('successful_cases', 0)}")

                    log_file.write(f"[{idx}/{len(operator_names)}] {ms_api}\n")
                    log_file.write("  Status: ✅ completed\n")
                    log_file.write(f"  Total iterations: {len(results)}\n")
                    log_file.write(f"  LLM-generated cases: {stats.get('llm_generated_cases', 0)}\n")
                    log_file.write(f"  Successful cases: {stats.get('successful_cases', 0)}\n\n")
                    if stats.get("llm_generated_cases", 0) > 0:
                        rate = stats.get("successful_cases", 0) / stats["llm_generated_cases"] * 100
                        log_file.write(f"  Success rate: {rate:.2f}%\n")
                    log_file.write("\n")
                    log_file.flush()
                else:
                    all_operators_summary.append({
                        "operator": ms_api, "total_iterations": 0,
                        "llm_generated_cases": 0, "successful_cases": 0,
                        "status": "no_results",
                    })
                    log_file.write(f"[{idx}/{len(operator_names)}] {ms_api}\n")
                    log_file.write("  Status: ⚠️ no results\n\n")
                    log_file.flush()

            except Exception as e:
                print(f"\n❌ {ms_api} failed: {e}")
                all_operators_summary.append({
                    "operator": ms_api, "total_iterations": 0,
                    "llm_generated_cases": 0, "successful_cases": 0,
                    "status": "failed", "error": str(e),
                })
                log_file.write(f"[{idx}/{len(operator_names)}] {ms_api}\n")
                log_file.write(f"  Status: ❌ failed\n  Error: {str(e)}\n\n")
                log_file.flush()
                continue

        # ==================== Summary output ====================
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
        print("📊 Batch test summary")
        print("=" * 80)
        print(f"Total operators: {len(operator_names)}")
        print(f"✅ Completed: {completed_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"⚠️ No results: {no_results_count}")
        print("\n📈 Stats:")
        print(f"   - Total LLM-generated cases: {total_llm_cases}")
        print(f"   - Total successful cases: {total_successful}")
        if total_llm_cases > 0:
            print(f"   - Success rate: {total_successful / total_llm_cases * 100:.2f}%")
        print(f"   - Total iterations: {total_iterations}")
        print(f"\n⏱️ Duration: {hours}h {minutes}m {seconds}s")

        log_file.write("=" * 80 + "\nOverall summary\n" + "=" * 80 + "\n")
        log_file.write(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total duration: {hours}h {minutes}m {seconds}s ({total_duration:.2f}s)\n\n")
        log_file.write("Operator results:\n")
        log_file.write(f"  - Total operators: {len(operator_names)}\n")
        log_file.write(f"  - Completed: {completed_count}\n")
        log_file.write(f"  - Failed: {failed_count}\n")
        log_file.write(f"  - No results: {no_results_count}\n\n")
        log_file.write("LLM stats:\n")
        log_file.write(f"  - Generated cases: {total_llm_cases}\n")
        log_file.write(f"  - Successful cases: {total_successful}\n")
        if total_llm_cases > 0:
            log_file.write(f"  - Success rate: {total_successful / total_llm_cases * 100:.2f}%\n")
        log_file.write(f"  - Total iterations: {total_iterations}\n")
        log_file.close()

        print(f"\n💾 Batch log saved to: {batch_log_file}")

        # JSON summary
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

        print(f"💾 JSON summary saved to: {summary_file}")

    finally:
        comparator.close()
        print("\n✅ Batch test finished")


if __name__ == "__main__":
    main()
