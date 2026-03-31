#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4: LLM-based PaddlePaddle vs MindSpore operator differential testing framework

Function:
- Load Paddle test cases and PD->MS mappings from JSON files
- For each equivalent operator, execute Paddle and MindSpore and compare results
- Use LLM to repair, mutate, or skip test cases
- Support concurrent testing of multiple cases
- Save detailed test results and batch logs

Usage:
    conda activate tf_env
    python pd_ms_test_1/llm_enhanced_compare.py \
        [--max-iterations 3] [--num-cases 5] [--workers 6] \
        [--start 1] [--end N] [--operators paddle.nn.ReLU paddle.abs]

Prerequisites:
    1. Step 1 extract_pd_apis.py has been run (outputs pd_apis_existing.json)
    2. Step 2 extract_pd_test_cases.py has been run (outputs pd_test_cases.json)
    3. Step 3 extract_pd_ms_mapping.py has been run (outputs pd_ms_mapping.csv / pd_ms_mapping_validated.csv)
"""

# ==================== Environment variables (must be set before imports) ====================
import os

# Resolve protobuf version conflicts between PaddlePaddle and MindSpore
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
# MKL thread settings (avoid Intel MKL crash under multithreading)
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
# OpenMP thread settings
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OMP_DYNAMIC'] = 'FALSE'
# Intel-related settings
os.environ['KMP_BLOCKTIME'] = '0'
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Reduce PaddlePaddle / MindSpore log output
os.environ['GLOG_v'] = '3'
# MindSpore exits when logtostderr=0 and GLOG_log_dir is unset
os.environ['GLOG_logtostderr'] = '1'
os.environ['GLOG_stderrthreshold'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['MS_DEV_LOG_LEVEL'] = '3'
# =========================================================================

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

import paddle

from openai import OpenAI


def _configure_console_encoding() -> None:
    """Try to set stdout/stderr to UTF-8 to avoid emoji crashes on Windows GBK."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_console_encoding()

# Add project root directory to path
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

# Data file paths
DATA_DIR = os.path.join(ROOT_DIR, "pd_ms_test_1", "data")
DEFAULT_TEST_CASES_FILE = os.path.join(DATA_DIR, "pd_test_cases.json")
DEFAULT_MAPPING_FILE = os.path.join(DATA_DIR, "pd_ms_mapping_validated.csv")

# Global execution lock (avoid MKL concurrency conflicts between PaddlePaddle and MindSpore)
_BLAS_EXECUTION_LOCK = RLock()


# ==================== MindSpore deferred load ====================
_mindspore = None
_ms_context_set = False


def get_mindspore():
        """
        Defer loading MindSpore and set context to PyNative mode + CPU.

        Why defer loading?
        - Importing MindSpore initializes the backend immediately; importing alongside PaddlePaddle
            can trigger protobuf or MKL conflicts.
        - Deferred loading ensures environment variables are set before import.
        """
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
    """LLM-based PaddlePaddle vs MindSpore differential testing framework."""

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
        Initialize comparator.

        Args:
            test_cases_file: Test cases JSON filepath (Step 2 output)
            mapping_file: PD->MS mapping CSV filepath (Step 3 output)
            key_path: API key filepath
            model: LLM model name
            print_lock: Print lock
            llm_workers: LLM concurrent worker count
        """
        self.model = model
        self.print_lock = print_lock or Lock()
        self.llm_workers = max(1, int(llm_workers))
        self.execution_lock = RLock()
        self.stats_lock = Lock()

        # Initialize LLM client
        api_key = self._load_api_key(key_path)
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # Load test cases
        self.test_cases_data = self._load_test_cases(test_cases_file)
        self._safe_print(f"📋 Loaded {len(self.test_cases_data)} Paddle API test cases")

        # Load PD->MS mappings
        self.api_mapping = self._load_mapping(mapping_file)
        has_impl = sum(1 for v in self.api_mapping.values() if v != "No corresponding implementation")
        self._safe_print(f"📋 Loaded {len(self.api_mapping)} mappings ({has_impl} with implementations)")

        # Create result storage directory
        self.result_dir = os.path.join(ROOT_DIR, "pd_ms_test_1", "pd_ms_log_1")
        os.makedirs(self.result_dir, exist_ok=True)
        self._safe_print(f"📁 Result storage directory: {self.result_dir}")

        # Fix random seed
        self.random_seed = 42
        np.random.seed(self.random_seed)
        paddle.seed(self.random_seed)
        ms = get_mindspore()
        ms.set_seed(self.random_seed)

    def _safe_print(self, msg: str, end: str = "\n"):
        """Thread-safe print"""
        with self.print_lock:
            print(msg, end=end, flush=True)

    def _load_api_key(self, key_path: str = DEFAULT_KEY_PATH) -> str:
        """Load API key."""
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
        """Load test cases."""
        if not os.path.exists(filepath):
            self._safe_print(f"⚠️ Test cases file not found: {filepath}")
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("test_cases", {})

    def _load_mapping(self, filepath: str) -> Dict[str, str]:
        """Load PD->MS mapping table."""
        if not os.path.exists(filepath):
            self._safe_print(f"⚠️ Mapping file does not exist: {filepath}")
            return {}
        mapping = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pd_api = row.get("paddle-api", "").strip()
                ms_api = row.get("mindspore-api", "").strip()
                if pd_api and ms_api:
                    mapping[pd_api] = ms_api
        return mapping

    # ==================== API utilities ====================

    def is_class_based_api(self, api_name: str) -> bool:
        """Determine whether an API is class-based (e.g., paddle.nn.ReLU, mindspore.nn.ReLU)."""
        parts = api_name.split(".")
        if len(parts) >= 2:
            last_part = parts[-1]
            if last_part and last_part[0].isupper():
                return True
        return False

    def get_operator_function(self, api_name: str, framework: str = "paddle"):
        """
        Get the operator callable.

        Args:
            api_name: API name (e.g., paddle.abs, mindspore.ops.abs)
            framework: "paddle" or "mindspore"
        """
        try:
            if framework == "paddle":
                module = paddle
            elif framework == "mindspore":
                module = get_mindspore()
            else:
                return None

            parts = api_name.split(".")
            # Skip framework prefix (paddle. / mindspore.)
            start_idx = 1
            obj = module
            for part in parts[start_idx:]:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            # MindSpore fallback: try ops.operations
            if framework == "mindspore":
                try:
                    ms = get_mindspore()
                    obj = ms.ops.operations
                    op_name = parts[-1]
                    return getattr(obj, op_name)
                except (AttributeError, IndexError):
                    pass
            return None

    def convert_api_name(self, pd_api: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        Find the corresponding MindSpore API for a Paddle API.

        Returns:
            (pd_api, mindspore_api, mapping_method)
        """
        if pd_api in self.api_mapping:
            ms_api = self.api_mapping[pd_api]
            if ms_api and ms_api != "No corresponding implementation":
                return pd_api, ms_api, "Mapping table"
            else:
                return pd_api, None, "No corresponding implementation"
        return pd_api, None, "Not found in mapping table"

    # ==================== Data conversion ====================

    def generate_numpy_data(self, data: Any) -> np.ndarray:
        """
        Generate a numpy array from a description.

        Supported formats:
        - {"shape": [2, 3], "dtype": "float32"}
        - {"shape": [2, 3], "dtype": "float32", "range": [-1, 1]}
        - Scalar value
        - List
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

                # Strip framework prefix
                for prefix in ["mindspore.", "ms.", "paddle.", "torch.", "np.", "numpy."]:
                    if dtype_str.startswith(prefix):
                        dtype_str = dtype_str[len(prefix):]

                # dtype name mapping
                dtype_map = {
                    "float32": np.float32, "float64": np.float64,
                    "float16": np.float16, "float": np.float32,
                    "int32": np.int32, "int64": np.int64,
                    "int16": np.int16, "int8": np.int8,
                    "uint8": np.uint8, "bool": np.bool_,
                    "complex64": np.complex64, "complex128": np.complex128,
                    "bfloat16": np.float32,  # numpy does not support bfloat16, use float32
                }
                np_dtype = dtype_map.get(dtype_str, np.float32)

                # Process empty tensor
                if any(s == 0 for s in shape):
                    return np.empty(shape, dtype=np_dtype)

                # Get data range
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
        """Convert to PaddlePaddle tensor."""
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

    def convert_to_tensor_ms(self, data: Any, numpy_data: np.ndarray = None):
        """Convert to MindSpore tensor."""
        ms = get_mindspore()
        if numpy_data is not None:
            return ms.Tensor(numpy_data.copy())
        if isinstance(data, dict):
            np_data = self.generate_numpy_data(data)
            return ms.Tensor(np_data.copy())
        elif isinstance(data, (int, float)):
            return ms.Tensor(data)
        elif isinstance(data, list):
            return ms.Tensor(np.array(data))
        else:
            return ms.Tensor(data)

    # ==================== parameterPrepare ====================

    def should_skip_param(self, key: str, api_name: str, framework: str) -> bool:
        """Determine whether to skip a parameter."""
        # Generic skip parameters
        common_skip = ["description", "api"]
        if key in common_skip:
            return True

        # MindSpore-specific parameters (skip in PD->MS)
        ms_skip = ["has_bias"]
        if framework == "mindspore" and key in ms_skip:
            return True

        # Paddle-specific parameters
        paddle_skip = ["name", "place"]
        if framework == "paddle" and key in paddle_skip:
            return True

        return False

    def prepare_arguments(
        self, test_case: Dict[str, Any], framework: str = "paddle"
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Prepare parameters for a specific framework.

        Args:
            test_case: Test cases (tensor descriptions and scalar parameters)
            framework: "paddle" or "mindspore"

        Returns:
            (args, kwargs)
        """
        ms = get_mindspore()
        args = []
        kwargs = {}

        def convert_value(value: Any) -> Any:
            if isinstance(value, dict):
                if "shape" in value:
                    np_data = self.generate_numpy_data(value)
                    if framework == "mindspore":
                        return ms.Tensor(np_data.copy())
                    return paddle.to_tensor(np_data)
                return {k: convert_value(v) for k, v in value.items()}
            if isinstance(value, np.ndarray):
                if framework == "mindspore":
                    return ms.Tensor(value.copy())
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
            for prefix in ["mindspore.", "ms.", "paddle.", "torch.", "np.", "numpy."]:
                if token.startswith(prefix):
                    token = token[len(prefix):]

            if framework == "mindspore":
                return getattr(ms, token, dtype_value)
            if framework == "paddle":
                return getattr(paddle, token, dtype_value)
            return dtype_value

        # LLM explicit args/kwargs style
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

        # Positional parameter names
        positional_params = [
            "inputs", "x", "input", "condition", "y", "other", "a", "b",
            "start", "end", "step", "stop",
        ]

        # Variable-length parameter handling
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
                        if framework == "mindspore":
                            args.append(ms.Tensor(np_data.copy()))
                        else:
                            args.append(paddle.to_tensor(np_data))
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

        # Process other parameters (keyword params)
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

    # ==================== Result comparison ====================

    def compare_tensors(
        self, pd_result, ms_result, tolerance: float = 1e-5
    ) -> Tuple[bool, str]:
        """Compare Paddle and MindSpore computation results."""
        try:
            # Convert to numpy: Paddle uses .numpy(), MindSpore uses .asnumpy()
            if isinstance(pd_result, paddle.Tensor):
                pd_np = pd_result.numpy()
            elif isinstance(pd_result, np.ndarray):
                pd_np = pd_result
            else:
                pd_np = np.array(pd_result)

            ms = get_mindspore()
            if isinstance(ms_result, ms.Tensor):
                ms_np = ms_result.asnumpy()
            elif hasattr(ms_result, 'asnumpy'):
                ms_np = ms_result.asnumpy()
            elif hasattr(ms_result, 'numpy'):
                ms_np = ms_result.numpy()
            elif isinstance(ms_result, np.ndarray):
                ms_np = ms_result
            else:
                ms_np = np.array(ms_result)

            # Shape consistency check
            if pd_np.shape != ms_np.shape:
                return False, f"Shape mismatch: PD={pd_np.shape} vs MS={ms_np.shape}"

            # Exact compare for boolean types
            if pd_np.dtype == np.bool_ or ms_np.dtype == np.bool_:
                match = np.array_equal(pd_np, ms_np)
                if match:
                    return True, "Boolean results fully consistent"
                else:
                    diff_count = np.sum(pd_np != ms_np)
                    return False, f"Boolean results inconsistent, diff count: {diff_count}"

            # Numeric compare
            if np.allclose(pd_np, ms_np, atol=tolerance, rtol=tolerance, equal_nan=True):
                return True, "Results consistent (within tolerance)"
            else:
                max_diff = np.max(np.abs(pd_np.astype(np.float64) - ms_np.astype(np.float64)))
                return False, f"Results inconsistent, max diff: {max_diff:.8f}"

        except Exception as e:
            return False, f"Compare error: {str(e)}"

    # ==================== Test execution ====================

    def execute_test_case(
        self,
        pd_api: str,
        ms_api: str,
        pd_test_case: Dict[str, Any],
        ms_test_case: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single test case.

        Args:
            pd_api: PaddlePaddle API name
            ms_api: MindSpore API name
            pd_test_case: Paddle Test cases
            ms_test_case: MindSpore test case (if None, use Paddle case)
        """
        ms = get_mindspore()

        if ms_test_case is None:
            ms_test_case = pd_test_case

        effective_pd_api = pd_test_case.get("api", pd_api) if isinstance(pd_test_case, dict) else pd_api
        effective_ms_api = ms_test_case.get("api", ms_api) if isinstance(ms_test_case, dict) else ms_api

        result = {
            "pd_api": effective_pd_api,
            "ms_api": effective_ms_api,
            "pd_success": False,
            "ms_success": False,
            "results_match": False,
            "pd_error": None,
            "ms_error": None,
            "comparison_error": None,
            "pd_shape": None,
            "ms_shape": None,
            "pd_dtype": None,
            "ms_dtype": None,
            "status": "unknown",
        }

        # Materialize shared input tensors to keep PD/MS inputs identical
        pd_test_case, ms_test_case = self._materialize_shared_tensors(
            effective_pd_api, effective_ms_api, pd_test_case, ms_test_case
        )

        is_class_pd = self.is_class_based_api(effective_pd_api)
        is_class_ms = self.is_class_based_api(effective_ms_api)

        # ---- Execute PaddlePaddle ----
        pd_result = None
        try:
            with _BLAS_EXECUTION_LOCK:
                pd_func = self.get_operator_function(effective_pd_api, "paddle")
                if pd_func is None:
                    raise AttributeError(f"Failed to find Paddle API: {effective_pd_api}")

                if is_class_pd:
                    init_kwargs = {
                        k: v for k, v in pd_test_case.items()
                        if k not in ["api", "input", "x"] and not isinstance(v, (np.ndarray,))
                        and not (isinstance(v, dict) and "shape" in v)
                    }
                    layer = pd_func(**init_kwargs)
                    # Get input
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

        # ---- Execute MindSpore ----
        ms_result = None
        try:
            with _BLAS_EXECUTION_LOCK:
                ms_func = self.get_operator_function(effective_ms_api, "mindspore")
                if ms_func is None:
                    raise AttributeError(f"Failed to find MindSpore API: {effective_ms_api}")

                if is_class_ms:
                    init_kwargs = {
                        k: v for k, v in ms_test_case.items()
                        if k not in ["api", "input", "x"] and not isinstance(v, (np.ndarray,))
                        and not (isinstance(v, dict) and "shape" in v)
                    }
                    module_instance = ms_func(**init_kwargs)
                    input_data = ms_test_case.get("input") or ms_test_case.get("x")
                    if input_data is not None:
                        if isinstance(input_data, dict) and "shape" in input_data:
                            np_data = self.generate_numpy_data(input_data)
                            ms_input = ms.Tensor(np_data.copy())
                        elif isinstance(input_data, np.ndarray):
                            ms_input = ms.Tensor(input_data.copy())
                        else:
                            ms_input = ms.Tensor(input_data)
                        ms_result = module_instance(ms_input)
                    else:
                        ms_result = module_instance(ms.Tensor(np.random.randn(2, 3).astype(np.float32)))
                else:
                    ms_args, ms_kwargs = self.prepare_arguments(ms_test_case, "mindspore")
                    ms_result = ms_func(*ms_args, **ms_kwargs)

            result["ms_success"] = True
            if hasattr(ms_result, "shape"):
                result["ms_shape"] = list(ms_result.shape)
            if hasattr(ms_result, "dtype"):
                result["ms_dtype"] = str(ms_result.dtype)

        except Exception as e:
            result["ms_error"] = f"{type(e).__name__}: {str(e)}"

        # ---- Compare results ----
        if result["pd_success"] and result["ms_success"]:
            try:
                match, detail = self.compare_tensors(pd_result, ms_result)
                result["results_match"] = match
                result["comparison_error"] = None if match else detail
                result["status"] = "consistent" if match else "inconsistent"
            except Exception as e:
                result["comparison_error"] = str(e)
                result["status"] = "comparison_error"
        elif result["pd_success"] and not result["ms_success"]:
            result["status"] = "mindspore_error"
        elif not result["pd_success"] and result["ms_success"]:
            result["status"] = "paddle_error"
        else:
            result["status"] = "both_error"

        return result

    def _execute_test_case_sequential(
        self, pd_api, ms_api, pd_test_case, ms_test_case=None
    ) -> Dict[str, Any]:
        """Use a lock to ensure non-concurrent execution."""
        with self.execution_lock:
            return self.execute_test_case(pd_api, ms_api, pd_test_case, ms_test_case)

    def _default_input_desc_for_class(self, api_name: str) -> Dict[str, Any]:
        """Provide default input description for class APIs."""
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
        ms_api: str,
        pd_test_case: Dict[str, Any],
        ms_test_case: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Create shared tensors to keep both frameworks inputs consistent."""
        pd_case = copy.deepcopy(pd_test_case)
        ms_case = copy.deepcopy(ms_test_case)

        is_class_pd = self.is_class_based_api(pd_api)
        is_class_ms = self.is_class_based_api(ms_api)
        if (is_class_pd or is_class_ms) and not (
            "input" in pd_case or "x" in pd_case or "input" in ms_case or "x" in ms_case
        ):
            default_desc = self._default_input_desc_for_class(pd_api or ms_api)
            pd_case.setdefault("input", default_desc)
            ms_case.setdefault("input", default_desc)

        def is_tensor_desc(value: Any) -> bool:
            return isinstance(value, dict) and "shape" in value

        def clone_array(value: np.ndarray) -> np.ndarray:
            return value.copy()

        def materialize_pair(pd_val: Any, ms_val: Any) -> Tuple[Any, Any]:
            if isinstance(pd_val, np.ndarray):
                return clone_array(pd_val), clone_array(pd_val)
            if isinstance(ms_val, np.ndarray):
                return clone_array(ms_val), clone_array(ms_val)

            if is_tensor_desc(pd_val) or is_tensor_desc(ms_val):
                tensor_desc = pd_val if is_tensor_desc(pd_val) else ms_val
                shared = self.generate_numpy_data(tensor_desc)
                return clone_array(shared), clone_array(shared)

            if isinstance(pd_val, list) or isinstance(ms_val, list):
                pd_list = pd_val if isinstance(pd_val, list) else []
                ms_list = ms_val if isinstance(ms_val, list) else []
                size = max(len(pd_list), len(ms_list))
                out_pd = []
                out_ms = []
                for index in range(size):
                    left = pd_list[index] if index < len(pd_list) else None
                    right = ms_list[index] if index < len(ms_list) else None
                    new_left, new_right = materialize_pair(left, right)
                    if index < len(pd_list):
                        out_pd.append(new_left)
                    if index < len(ms_list):
                        out_ms.append(new_right)
                return out_pd if isinstance(pd_val, list) else pd_val, out_ms if isinstance(ms_val, list) else ms_val

            if isinstance(pd_val, dict) or isinstance(ms_val, dict):
                pd_dict = pd_val if isinstance(pd_val, dict) else {}
                ms_dict = ms_val if isinstance(ms_val, dict) else {}
                keys = set(pd_dict.keys()) | set(ms_dict.keys())
                out_pd = {}
                out_ms = {}
                for key in keys:
                    if key == "api":
                        if key in pd_dict:
                            out_pd[key] = pd_dict[key]
                        if key in ms_dict:
                            out_ms[key] = ms_dict[key]
                        continue
                    new_left, new_right = materialize_pair(pd_dict.get(key), ms_dict.get(key))
                    if key in pd_dict:
                        out_pd[key] = new_left
                    if key in ms_dict:
                        out_ms[key] = new_right
                return out_pd if isinstance(pd_val, dict) else pd_val, out_ms if isinstance(ms_val, dict) else ms_val

            return pd_val, ms_val

        pd_case, ms_case = materialize_pair(pd_case, ms_case)

        # Generic cross-framework alias mapping
        if isinstance(pd_case, dict) and isinstance(ms_case, dict):
            alias_pairs = [
                ("x", "input"),
                ("input", "x"),
                ("y", "other"),
                ("other", "y"),
            ]
            for pd_key, ms_key in alias_pairs:
                if pd_key in pd_case and ms_key in ms_case:
                    pd_item, ms_item = materialize_pair(pd_case[pd_key], ms_case[ms_key])
                    pd_case[pd_key] = pd_item
                    ms_case[ms_key] = ms_item

        return pd_case, ms_case

    # ==================== API document fetch ====================

    def _fetch_api_docs(self, pd_api: str, ms_api: str) -> Tuple[str, str]:
        """Fetch Paddle and MindSpore API docs."""
        MIN_DOC_LENGTH = 300
        pd_doc = ""
        ms_doc = ""

        try:
            raw = get_doc_content(pd_api, "paddle")
            if raw and len(raw) >= MIN_DOC_LENGTH:
                pd_doc = raw[:3000]
                self._safe_print(f"    📄 PDDocument: {len(pd_doc)} chars")
            else:
                self._safe_print("    📄 PDDocument: no valid content")
        except Exception as e:
            self._safe_print(f"    ⚠️ PDDocumentFetchfailed: {str(e)[:50]}")

        try:
            raw = get_doc_content(ms_api, "mindspore")
            if raw and len(raw) >= MIN_DOC_LENGTH:
                ms_doc = raw[:3000]
                self._safe_print(f"    📄 MSDocument: {len(ms_doc)} chars")
            else:
                self._safe_print("    📄 MSDocument: no valid content")
        except Exception as e:
            self._safe_print(f"    ⚠️ MSDocumentFetchfailed: {str(e)[:50]}")

        return pd_doc, ms_doc

    # ==================== LLM interaction ====================

    def _build_llm_prompt(
        self,
        execution_result: Dict[str, Any],
        pd_test_case: Dict[str, Any],
        ms_test_case: Dict[str, Any],
        pd_doc: str = "",
        ms_doc: str = "",
    ) -> str:
        """Build the LLM prompt."""
        pd_api = execution_result.get("pd_api", "")
        ms_api = execution_result.get("ms_api", "")
        status = execution_result.get("status", "")
        pd_success = execution_result.get("pd_success", False)
        ms_success = execution_result.get("ms_success", False)
        results_match = execution_result.get("results_match", False)
        pd_error = execution_result.get("pd_error", "")
        ms_error = execution_result.get("ms_error", "")
        comparison_error = execution_result.get("comparison_error", "")

        # Simplify test cases to reduce token usage
        simplified_pd = {}
        for key, value in pd_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_pd[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_pd[key] = value

        simplified_ms = {}
        for key, value in ms_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_ms[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_ms[key] = value

        # Build parameter example strings
        pd_param_examples = []
        for key, value in simplified_pd.items():
            if key == "api":
                continue
            pd_param_examples.append(f'    "{key}": {json.dumps(value)}')

        pd_param_str = ",\n".join(pd_param_examples) if pd_param_examples else '    "x": {"shape": [2, 3], "dtype": "float32"}'

        ms_param_examples = []
        for key, value in simplified_ms.items():
            if key == "api":
                continue
            ms_param_examples.append(f'    "{key}": {json.dumps(value)}')

        ms_param_str = ",\n".join(ms_param_examples) if ms_param_examples else '    "input": {"shape": [2, 3], "dtype": "float32"}'

        # Document section
        doc_section = ""
        if pd_doc or ms_doc:
            doc_section = "\n## Official API document reference\n\n"
            if pd_doc:
                doc_section += f"### PaddlePaddle {pd_api} Document\n```\n{pd_doc}\n```\n\n"
            if ms_doc:
                doc_section += f"### MindSpore {ms_api} Document\n```\n{ms_doc}\n```\n\n"

                prompt = f"""Please analyze the execution results of the following operator test cases in PaddlePaddle and MindSpore, and repair or mutate (fuzz) the test cases based on the results.

## Test information
- **PaddlePaddle API**: {pd_api}
- **MindSpore API**: {ms_api}
{doc_section}
## Execution results
- **Execute status**: {status}
- **PaddlePaddle execute successful**: {pd_success}
- **MindSpore execute successful**: {ms_success}
- **Results consistent**: {results_match}

## Error information
- **PaddlePaddle error**: {pd_error if pd_error else "none"}
- **MindSpore error**: {ms_error if ms_error else "none"}
- **Compare error**: {comparison_error if comparison_error else "none"}

## Original test cases

### PaddlePaddle test cases
```json
{json.dumps(simplified_pd, indent=2, ensure_ascii=False)}
```

### MindSpore test cases
```json
{json.dumps(simplified_ms, indent=2, ensure_ascii=False)}
```

## Task requirements
Based on the information above (including official API docs), decide whether the comparison result is **consistent**, **inconsistent**, or **execution error**, then perform one of the following actions:

1. **If consistent**: perform **mutation (fuzzing)**, e.g., modify input shapes or parameter values (consider extreme/boundary values).
2. **If execution error**: perform **repair** (adjust parameter names, counts, types, or ranges; frameworks may differ) or **skip** when:
     - the operator doc is missing or indicates removal, or
     - the cross-framework operators are not truly equivalent.
3. **If inconsistent**: check whether the difference is a tolerable precision error (<= 1e-3).
     - If tolerable, use **mutation**.
     - If the operators are not equivalent, choose **skip**.
     - Otherwise, treat it as test case construction issues and **repair** using docs.

## Output format requirements
Please strictly output the following JSON format. Do not include any other text, comments, or markdown markers:

{{
    "operation": "mutation",
    "reason": "Detailed reason in English (<= 150 chars)",
    "paddle_test_case": {{
        "api": "{pd_api}",
{pd_param_str}
    }},
    "mindspore_test_case": {{
        "api": "{ms_api}",
{ms_param_str}
    }}
}}

**Important notes**:
1. The operation value must be one of "mutation", "repair", or "skip".
2. Tensor parameters must use {{"shape": [...], "dtype": "..."}}.
3. Scalar parameters should be plain values.
4. Keep inputs identical and parameters semantically aligned across frameworks.
5. PaddlePaddle and MindSpore test cases may differ in parameter names/values/counts as long as outputs are equivalent.
6. If docs are missing or indicate removal, set operation to "skip" and do not attempt repair.
7. For mutation, explore edge cases: empty tensor, single-element tensor, high-dimensional tensor, different dtypes, boundary values.
8. Read official docs carefully to ensure parameter names, types, and ranges match.
9. PaddlePaddle and MindSpore data format defaults are NCHW (not TensorFlow's NHWC).
10. Many MindSpore operators are under mindspore.ops; class APIs are often under mindspore.nn.
"""
        return prompt

    def call_llm_for_repair_or_mutation(
        self,
        execution_result: Dict[str, Any],
        pd_test_case: Dict[str, Any],
        ms_test_case: Dict[str, Any],
        pd_doc: str = "",
        ms_doc: str = "",
    ) -> Dict[str, Any]:
        """Call the LLM to repair or mutate test cases."""
        prompt = self._build_llm_prompt(
            execution_result, pd_test_case, ms_test_case, pd_doc, ms_doc
        )
        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a deep learning framework testing expert familiar with PaddlePaddle and MindSpore API differences. Determine whether to repair or mutate test cases based on execution results and return strict JSON output.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            raw_response = completion.choices[0].message.content.strip()
            time.sleep(1)

            # Parse JSON
            try:
                llm_result = json.loads(raw_response)
                return llm_result
            except json.JSONDecodeError:
                self._safe_print(f"    ⚠️ LLMReturned content is not valid JSON, trying to extract...")
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    llm_result = json.loads(json_match.group())
                    return llm_result
                else:
                    return {
                        "operation": "skip",
                        "reason": "LLMReturned format error",
                        "paddle_test_case": pd_test_case,
                        "mindspore_test_case": ms_test_case,
                    }

        except Exception as e:
            self._safe_print(f"    ❌ LLM call failed: {e}")
            return {
                "operation": "skip",
                "reason": f"LLM call failed: {e}",
                "paddle_test_case": pd_test_case,
                "mindspore_test_case": ms_test_case,
            }

    # ==================== Core testing loop ====================

    def llm_enhanced_test_operator(
        self,
        pd_api: str,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        num_test_cases: int = None,
        num_workers: int = DEFAULT_WORKERS,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Test a single operator pair with LLM enhancement.

        Args:
            pd_api: PaddlePaddle API name
            max_iterations: Max iterations per test case
            num_test_cases: Number of cases to test
            num_workers: LLM concurrent worker count
        """
        self._safe_print(f"\n{'=' * 80}")
        self._safe_print(f"🎯 Starting operator test: {pd_api}")
        self._safe_print(f"🔄 Max iterations per case: {max_iterations}")
        self._safe_print(f"{'=' * 80}\n")

        stats = {"llm_generated_cases": 0, "successful_cases": 0}

        # Get corresponding MindSpore API
        _, ms_api, mapping_method = self.convert_api_name(pd_api)
        if ms_api is None:
            self._safe_print(f"❌ {pd_api} has no corresponding MindSpore implementation")
            return [], stats

        self._safe_print(f"✅ PaddlePaddle API: {pd_api}")
        self._safe_print(f"✅ MindSpore API: {ms_api}")
        self._safe_print(f"✅ Mapping method: {mapping_method}")

        # Get test cases
        api_data = self.test_cases_data.get(pd_api, {})
        test_cases = api_data.get("test_cases", [])

        if not test_cases:
            self._safe_print(f"⚠️ No test cases found for {pd_api}, using default case")
            test_cases = [{"description": "default", "inputs": {"x": {"shape": [2, 3], "dtype": "float32"}}}]

        # Determine actual test count
        if num_test_cases is None:
            num_test_cases = len(test_cases)
        else:
            num_test_cases = min(num_test_cases, len(test_cases))

        self._safe_print(f"📋 Will test {num_test_cases} cases (LLM concurrency={num_workers}, execution sequential)")

        # Prepare initial cases
        initial_cases = []
        for case_idx in range(num_test_cases):
            tc = test_cases[case_idx]
            # Extract parameters from inputs and build a flat test case
            if "inputs" in tc:
                flat_case = dict(tc["inputs"])
            else:
                flat_case = {k: v for k, v in tc.items() if k != "description"}
            flat_case["api"] = pd_api
            initial_cases.append((case_idx + 1, flat_case))

        all_results = []

        if num_workers <= 1:
            for case_number, initial_test_case in initial_cases:
                self._safe_print(f"\n📋 Case {case_number}/{num_test_cases}")
                case_results = self._test_single_case_with_iterations(
                    pd_api, ms_api, initial_test_case,
                    max_iterations, case_number, stats,
                )
                all_results.extend(case_results)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_case = {}
                for case_number, initial_test_case in initial_cases:
                    future = executor.submit(
                        self._test_single_case_with_iterations,
                        pd_api, ms_api, initial_test_case,
                        max_iterations, case_number, stats,
                    )
                    future_to_case[future] = case_number

                for future in as_completed(future_to_case):
                    case_results = future.result()
                    all_results.extend(case_results)

        all_results.sort(key=lambda r: (r.get("case_number", 0), r.get("iteration", 0)))

        self._safe_print(f"\n{'=' * 80}")
        self._safe_print("✅ All tests completed")
        self._safe_print(f"📊 Tested {num_test_cases} cases, total {len(all_results)} iterations")
        self._safe_print(f"📊 LLM-generated test case count: {stats['llm_generated_cases']}")
        self._safe_print(f"📊 Cases where both frameworks executed successfully: {stats['successful_cases']}")
        self._safe_print(f"{'=' * 80}\n")

        return all_results, stats

    def _test_single_case_with_iterations(
        self,
        pd_api: str,
        ms_api: str,
        initial_test_case: Dict[str, Any],
        max_iterations: int,
        case_number: int,
        stats: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """
        Run multi-iteration tests for a single test case.

        Core loop: execute -> LLM decide -> repair/mutation/skip -> execute again -> ...
        """
        case_results = []

        # Build initial PD and MS test cases
        current_pd_test_case = copy.deepcopy(initial_test_case)
        current_pd_test_case["api"] = pd_api

        current_ms_test_case = copy.deepcopy(initial_test_case)
        current_ms_test_case["api"] = ms_api

        is_llm_generated = False

        # Pre-fetch API docs (once)
        self._safe_print("  📖 Pre-fetch API docs...")
        pd_doc, ms_doc = self._fetch_api_docs(pd_api, ms_api)

        # Iteration testing
        for iteration in range(max_iterations):
            source_type = "LLM" if is_llm_generated else "file"
            self._safe_print(f"  🔄 Iteration {iteration + 1}/{max_iterations} ({source_type})", end="")

            current_pd_api = current_pd_test_case.get("api", pd_api) or pd_api
            current_ms_api = current_ms_test_case.get("api", ms_api) or ms_api

            # Execute test
            try:
                execution_result = self._execute_test_case_sequential(
                    current_pd_api, current_ms_api, current_pd_test_case, current_ms_test_case
                )

                pd_status = "✓" if execution_result["pd_success"] else "✗"
                ms_status = "✓" if execution_result["ms_success"] else "✗"
                match_status = "✓" if execution_result["results_match"] else "✗"
                self._safe_print(f" | PD:{pd_status} MS:{ms_status} Match:{match_status}")

                if execution_result["pd_error"] and not execution_result["pd_success"]:
                    self._safe_print(f"    ❌ PDerror: {str(execution_result['pd_error'])[:100]}...")
                if execution_result["ms_error"] and not execution_result["ms_success"]:
                    self._safe_print(f"    ❌ MSerror: {str(execution_result['ms_error'])[:100]}...")
                if execution_result["comparison_error"]:
                    self._safe_print(f"    ⚠️ Compare: {str(execution_result['comparison_error'])[:100]}...")

                # Count LLM-generated cases
                if is_llm_generated:
                    if execution_result["pd_success"] and execution_result["ms_success"]:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

            except Exception as e:
                self._safe_print(f" | ❌ Severe error: {str(e)[:80]}...")
                execution_result = {
                    "status": "fatal_error",
                    "pd_success": False, "ms_success": False,
                    "results_match": False,
                    "pd_error": f"Fatal: {str(e)}", "ms_error": None,
                    "comparison_error": None,
                }

            # Save iteration result
            iteration_result = {
                "iteration": iteration + 1,
                "pd_test_case": current_pd_test_case,
                "ms_test_case": current_ms_test_case,
                "execution_result": execution_result,
                "llm_operation": None,
                "is_llm_generated": is_llm_generated,
                "case_number": case_number,
            }

            # Call LLM
            try:
                llm_result = self.call_llm_for_repair_or_mutation(
                    execution_result, current_pd_test_case, current_ms_test_case,
                    pd_doc, ms_doc,
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

            # Prepare next iteration
            if operation in ("mutation", "repair"):
                next_pd_case = llm_result.get("paddle_test_case", current_pd_test_case)
                next_ms_case = llm_result.get("mindspore_test_case", current_ms_test_case)
                with self.stats_lock:
                    stats["llm_generated_cases"] += 1
                is_llm_generated = True
            else:
                next_pd_case = current_pd_test_case
                next_ms_case = current_ms_test_case

            current_pd_test_case, current_ms_test_case = self._convert_llm_test_cases(
                next_pd_case, next_ms_case
            )

        # If the last LLM-generated case was not executed, run a final execution
        if case_results:
            last_op = case_results[-1].get("llm_operation", {}).get("operation", "skip")
            if last_op in ("mutation", "repair"):
                self._safe_print("  🔄 Execute final LLM case", end="")
                try:
                    execution_result = self._execute_test_case_sequential(
                        current_pd_test_case.get("api", pd_api) or pd_api,
                        current_ms_test_case.get("api", ms_api) or ms_api,
                        current_pd_test_case,
                        current_ms_test_case,
                    )
                    pd_s = "✓" if execution_result["pd_success"] else "✗"
                    ms_s = "✓" if execution_result["ms_success"] else "✗"
                    m_s = "✓" if execution_result["results_match"] else "✗"
                    self._safe_print(f" | PD:{pd_s} MS:{ms_s} Match:{m_s}")

                    if execution_result["pd_success"] and execution_result["ms_success"]:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

                    case_results.append({
                        "iteration": len(case_results) + 1,
                        "pd_test_case": current_pd_test_case,
                        "ms_test_case": current_ms_test_case,
                        "execution_result": execution_result,
                        "llm_operation": {"operation": "final_execution", "reason": "Execute the last LLM-generated case"},
                        "case_number": case_number,
                        "is_llm_generated": True,
                    })
                except Exception as e:
                    self._safe_print(f"  ❌ Final case execution failed: {str(e)[:80]}...")
                    case_results.append({
                        "iteration": len(case_results) + 1,
                        "pd_test_case": current_pd_test_case,
                        "ms_test_case": current_ms_test_case,
                        "execution_result": {
                            "status": "fatal_error", "pd_success": False,
                            "ms_success": False, "results_match": False,
                            "error": str(e),
                        },
                        "llm_operation": {"operation": "final_execution", "reason": "Final case execution failed"},
                        "case_number": case_number,
                        "is_llm_generated": True,
                    })

        self._safe_print(f"  ✅ Case {case_number} completed, total {len(case_results)} iterations")
        return case_results

    def _convert_llm_test_cases(
        self,
        pd_test_case: Dict[str, Any],
        ms_test_case: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Convert LLM-returned test cases to executable format.
        Ensure both frameworks share identical tensor data.
        """
        pd_api = pd_test_case.get("api", "") if isinstance(pd_test_case, dict) else ""
        ms_api = ms_test_case.get("api", "") if isinstance(ms_test_case, dict) else ""
        return self._materialize_shared_tensors(pd_api, ms_api, pd_test_case, ms_test_case)

    # ==================== Result saving ====================

    def save_results(
        self, pd_api: str, results: List[Dict[str, Any]], stats: Dict[str, int] = None
    ):
        """Save test results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = pd_api.replace(".", "_")
        filename = f"llm_enhanced_{safe_name}_{timestamp}.json"
        filepath = os.path.join(self.result_dir, filename)

        output_results = []
        for result in results:
            out = copy.deepcopy(result)
            # Simplify numpy arrays
            for case_key in ["pd_test_case", "ms_test_case"]:
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
            "ms_api": self.api_mapping.get(pd_api, ""),
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
        """Get all testable Paddle APIs (with test cases and MS mapping)."""
        testable = []
        for pd_api in sorted(self.test_cases_data.keys()):
            ms_api = self.api_mapping.get(pd_api, "No corresponding implementation")
            if ms_api and ms_api != "No corresponding implementation":
                testable.append(pd_api)
        return testable

    def close(self):
        """Clean up resources"""
        pass


# ==================== Main function ====================

def main():
    parser = argparse.ArgumentParser(
        description="LLM-based PaddlePaddle vs MindSpore operator differential testing framework"
    )
    parser.add_argument(
        "--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"Max iterations per test case (default {DEFAULT_MAX_ITERATIONS})"
    )
    parser.add_argument(
        "--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES,
        help=f"Number of cases per operator (default {DEFAULT_NUM_CASES})"
    )
    parser.add_argument(
        "--start", type=int, default=1,
        help="Start operator index (1-based, default 1)"
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="End operator index (inclusive, default all)"
    )
    parser.add_argument(
        "--operators", "-o", nargs="*",
        help="Operator names to test (Paddle format, e.g., paddle.nn.ReLU)"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"concurrent worker count (default {DEFAULT_WORKERS})"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"LLM model name (default {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--key-path", "-k", default=DEFAULT_KEY_PATH,
        help=f"API key filepath (default {DEFAULT_KEY_PATH})"
    )
    parser.add_argument(
        "--test-cases-file", default=DEFAULT_TEST_CASES_FILE,
        help="Test cases JSON filepath"
    )
    parser.add_argument(
        "--mapping-file", default=DEFAULT_MAPPING_FILE,
        help="PD→MS mapping CSV file path"
    )

    args = parser.parse_args()
    num_workers = max(1, args.workers)

    print("=" * 80)
    print("LLM-based PaddlePaddle vs MindSpore operator differential testing framework")
    print("=" * 80)
    print(f"📌 Iterations per operator: {args.max_iterations}")
    print(f"📌 Test cases per operator: {args.num_cases}")
    print(f"📌 LLMconcurrent worker count: {num_workers}")
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
        # Get all testable APIs
        all_testable = comparator.get_all_testable_apis()
        print(f"\n🔍 Testable Paddle API count: {len(all_testable)}")

        if args.operators:
            operator_names = args.operators
            print(f"📋 Specified operator count: {len(operator_names)}")
        else:
            start_idx = max(1, args.start) - 1
            end_idx = args.end if args.end is not None else len(all_testable)
            end_idx = min(end_idx, len(all_testable))
            if start_idx >= end_idx:
                raise ValueError(f"start index {args.start} must be less than end index {end_idx}")
            operator_names = all_testable[start_idx:end_idx]
            print(f"📌 Test range: {start_idx + 1} to {end_idx} operators")
            print(f"📋 Will test {len(operator_names)} operators")

        print(f"📋 Operator list: {', '.join(operator_names[:10])}{'...' if len(operator_names) > 10 else ''}\n")

        all_operators_summary = []

        # Batch log
        batch_log_file = os.path.join(
            comparator.result_dir,
            f"batch_test_log_{start_datetime.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        log_file = open(batch_log_file, 'w', encoding='utf-8')
        log_file.write("=" * 80 + "\n")
        log_file.write("PD→MS differential testingBatch log\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Test configuration:\n")
        log_file.write(f"  - iterations: {args.max_iterations}\n")
        log_file.write(f"  - case count: {args.num_cases}\n")
        log_file.write(f"  - concurrency: {num_workers}\n")
        log_file.write(f"  - operator count: {len(operator_names)}\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.flush()

        for idx, pd_api in enumerate(operator_names, 1):
            print("\n" + "🔷" * 40)
            print(f"🎯 [{idx}/{len(operator_names)}] Starting operator test: {pd_api}")
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
                        "ms_api": comparator.api_mapping.get(pd_api, ""),
                        "total_iterations": len(results),
                        "llm_generated_cases": stats.get("llm_generated_cases", 0),
                        "successful_cases": stats.get("successful_cases", 0),
                        "status": "completed",
                    })
                    print(f"\n✅ {pd_api} test completed")
                    print(f"   - Total iterations: {len(results)}")
                    print(f"   - LLM-generated case count: {stats.get('llm_generated_cases', 0)}")
                    print(f"   - Successful execute case count: {stats.get('successful_cases', 0)}")

                    log_file.write(f"[{idx}/{len(operator_names)}] {pd_api}\n")
                    log_file.write(f"  status: ✅ completed\n")
                    log_file.write(f"  Total iterations: {len(results)}\n")
                    log_file.write(f"  LLM-generated case count: {stats.get('llm_generated_cases', 0)}\n")
                    log_file.write(f"  Successful execute case count: {stats.get('successful_cases', 0)}\n")
                    if stats.get("llm_generated_cases", 0) > 0:
                        rate = stats.get("successful_cases", 0) / stats["llm_generated_cases"] * 100
                        log_file.write(f"  Success rate: {rate:.2f}%\n")
                    log_file.write("\n")
                    log_file.flush()
                else:
                    all_operators_summary.append({
                        "operator": pd_api, "total_iterations": 0,
                        "llm_generated_cases": 0, "successful_cases": 0,
                        "status": "no_results",
                    })
                    log_file.write(f"[{idx}/{len(operator_names)}] {pd_api}\n")
                    log_file.write("  status: ⚠️ no results\n\n")
                    log_file.flush()

            except Exception as e:
                print(f"\n❌ {pd_api} test failed: {e}")
                all_operators_summary.append({
                    "operator": pd_api, "total_iterations": 0,
                    "llm_generated_cases": 0, "successful_cases": 0,
                    "status": "failed", "error": str(e),
                })
                log_file.write(f"[{idx}/{len(operator_names)}] {pd_api}\n")
                log_file.write(f"  status: ❌ failed\n  error: {str(e)}\n\n")
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
        print("📊 Batch testing overall summary")
        print("=" * 80)
        print(f"Total operators: {len(operator_names)}")
        print(f"✅ successfulcompleted: {completed_count}")
        print(f"❌ Tests failed: {failed_count}")
        print(f"⚠️ No results: {no_results_count}")
        print(f"\n📈 Statistics:")
        print(f"   - LLM-generated test case total: {total_llm_cases}")
        print(f"   - Successful execute case total: {total_successful}")
        if total_llm_cases > 0:
            print(f"   - Success rate: {total_successful / total_llm_cases * 100:.2f}%")
        print(f"   - Total iterations: {total_iterations}")
        print(f"\n⏱️ Runtime: {hours}h {minutes}m {seconds}s")

        # Write log
        log_file.write("=" * 80 + "\nOverall statistics\n" + "=" * 80 + "\n")
        log_file.write(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total runtime: {hours}h {minutes}m {seconds}s ({total_duration:.2f}s)\n\n")
        log_file.write("Operator results:\n")
        log_file.write(f"  - Total operators: {len(operator_names)}\n")
        log_file.write(f"  - successful: {completed_count}\n")
        log_file.write(f"  - failed: {failed_count}\n")
        log_file.write(f"  - no results: {no_results_count}\n\n")
        log_file.write("LLM stats:\n")
        log_file.write(f"  - generated case count: {total_llm_cases}\n")
        log_file.write(f"  - successful execute count: {total_successful}\n")
        if total_llm_cases > 0:
            log_file.write(f"  - success rate: {total_successful / total_llm_cases * 100:.2f}%\n")
        log_file.write(f"  - Total iterations: {total_iterations}\n")
        log_file.close()

        print(f"\n💾 Full log saved to: {batch_log_file}")

        # JSON summary
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

        print(f"💾 JSON summary saved to: {summary_file}")

    finally:
        comparator.close()
        print("\n✅ Batch testing completed")


if __name__ == "__main__":
    main()

