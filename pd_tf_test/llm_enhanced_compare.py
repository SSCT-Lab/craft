#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4: LLM-based differential testing framework for PaddlePaddle vs TensorFlow operators.

Functions:
- Load Paddle test cases and PD→TF mappings from JSON/CSV.
- Execute each equivalent operator in Paddle and TF and compare results.
- Use LLM to repair, mutate, or skip test cases.
- Support concurrent testing (execution stage is serialized to reduce BLAS/MKL conflicts).
- Save detailed results and batch logs.

Usage:
    conda activate tf_env
    python pd_tf_test_1/llm_enhanced_compare.py \
        [--max-iterations 3] [--num-cases 5] [--workers 6] \
        [--start 1] [--end N] [--operators paddle.nn.ReLU paddle.abs]

Prerequisites:
    1. Run Step 1 extract_pd_apis.py
    2. Run Step 2 extract_pd_test_cases.py
    3. Run Step 3 extract_pd_tf_mapping.py
"""

import os

# ==================== Environment vars (set before importing TensorFlow/Paddle) ====================
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
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
import tensorflow as tf

from openai import OpenAI

# Add project root to sys.path
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
DATA_DIR = os.path.join(ROOT_DIR, "pd_tf_test_1", "data")
DEFAULT_TEST_CASES_FILE = os.path.join(DATA_DIR, "pd_test_cases.json")
DEFAULT_MAPPING_FILE = os.path.join(DATA_DIR, "pd_tf_mapping_validated.csv")


class LLMEnhancedComparator:
    """LLM-based differential testing framework for PaddlePaddle and TensorFlow."""

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
        Initialize the comparator.

        Args:
            test_cases_file: Test case JSON file path (Step 2 output)
            mapping_file: PD→TF mapping CSV file path (Step 3 output)
            key_path: API key file path
            model: LLM model name
            print_lock: Print lock
            llm_workers: LLM worker threads
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
        self._safe_print(f"📋 Loaded test cases for {len(self.test_cases_data)} Paddle APIs")

        # Load PD→TF mappings
        self.api_mapping = self._load_mapping(mapping_file)
        has_impl = sum(1 for v in self.api_mapping.values() if v != "no_equivalent_impl")
        self._safe_print(f"📋 Loaded {len(self.api_mapping)} mappings ({has_impl} with equivalents)")

        # Create result directory
        self.result_dir = os.path.join(ROOT_DIR, "pd_tf_test_1", "pd_tf_log_1")
        os.makedirs(self.result_dir, exist_ok=True)
        self._safe_print(f"📁 Result directory: {self.result_dir}")

        # Fix random seed
        self.random_seed = 42
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        paddle.seed(self.random_seed)

    def _safe_print(self, msg: str, end: str = "\n"):
        """Thread-safe print."""
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
            self._safe_print(f"⚠️ Test case file not found: {filepath}")
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("test_cases", {})

    def _load_mapping(self, filepath: str) -> Dict[str, str]:
        """Load PD→TF mapping table."""
        if not os.path.exists(filepath):
            self._safe_print(f"⚠️ Mapping file not found: {filepath}")
            return {}
        mapping = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pd_api = row.get("paddle-api", "").strip()
                tf_api = row.get("tensorflow-api", "").strip()
                if pd_api and tf_api:
                    mapping[pd_api] = tf_api
        return mapping

    # ==================== API Utilities ====================

    def is_class_based_api(self, api_name: str) -> bool:
        """Check whether an API is class-based (e.g., paddle.nn.ReLU, tf.keras.layers.ReLU)."""
        parts = api_name.split(".")
        if len(parts) >= 2:
            last_part = parts[-1]
            if last_part and last_part[0].isupper():
                return True
        return False

    def get_operator_function(self, api_name: str, framework: str = "paddle"):
        """
        Get the operator function object.

        Args:
            api_name: API name (e.g., paddle.abs, tf.abs)
            framework: "paddle" or "tensorflow"
        """
        try:
            if framework == "paddle":
                module = paddle
            elif framework == "tensorflow":
                module = tf
            else:
                return None

            parts = api_name.split(".")
            # Skip framework prefix (paddle. / tf.)
            start_idx = 1
            obj = module
            for part in parts[start_idx:]:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return None

    def convert_api_name(self, pd_api: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        Find the TensorFlow API corresponding to the Paddle API.

        Returns:
            (pd_api, tensorflow_api, mapping_method)
        """
        if pd_api in self.api_mapping:
            tf_api = self.api_mapping[pd_api]
            if tf_api and tf_api != "no_equivalent_impl":
                return pd_api, tf_api, "mapping_table"
            else:
                return pd_api, None, "no_equivalent_impl"
        return pd_api, None, "not_in_mapping_table"

    # ==================== Data Conversion ====================

    def generate_numpy_data(self, data: Any) -> np.ndarray:
        """
        Generate a numpy array from a description.

        Supported formats:
        - {"shape": [2, 3], "dtype": "float32"}
        - {"shape": [2, 3], "dtype": "float32", "range": [-1, 1]}
        - scalar values
        - lists
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

                # Remove framework prefix
                for prefix in ["tf.", "tensorflow.", "torch.", "paddle.", "np.", "numpy."]:
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
                    "bfloat16": np.float32,  # numpy does not support bfloat16; use float32
                }
                np_dtype = dtype_map.get(dtype_str, np.float32)

                # Handle empty tensors
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
        """Convert to a PaddlePaddle tensor."""
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

    def convert_to_tensor_tf(self, data: Any, numpy_data: np.ndarray = None):
        """Convert to a TensorFlow tensor."""
        if numpy_data is not None:
            return tf.constant(numpy_data)
        if isinstance(data, dict):
            np_data = self.generate_numpy_data(data)
            return tf.constant(np_data)
        elif isinstance(data, (int, float)):
            return tf.constant(data)
        elif isinstance(data, list):
            return tf.constant(data)
        else:
            return tf.constant(data)

    # ==================== Argument Preparation ====================

    def should_skip_param(self, key: str, api_name: str, framework: str) -> bool:
        """Decide whether a parameter should be skipped."""
        # Common skip params
        common_skip = ["description", "api"]
        if key in common_skip:
            return True

        # TensorFlow-specific params (skip for PD→TF)
        tensorflow_skip = ["name"]
        if framework == "tensorflow" and key in tensorflow_skip:
            return True

        # Paddle-specific params (skip for PT→PD)
        paddle_skip = ["name", "place"]
        if framework == "paddle" and key in paddle_skip:
            return True

        return False

    def prepare_arguments(
        self, test_case: Dict[str, Any], framework: str = "paddle"
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Prepare arguments for a specific framework.

        Args:
            test_case: Test case (includes tensor descriptions and scalar params)
            framework: "paddle" or "tensorflow"

        Returns:
            (args, kwargs)
        """
        args = []
        kwargs = {}

        def convert_value(value: Any) -> Any:
            if isinstance(value, dict):
                if "shape" in value:
                    np_data = self.generate_numpy_data(value)
                    if framework == "tensorflow":
                        return tf.constant(np_data)
                    return paddle.to_tensor(np_data)
                return {k: convert_value(v) for k, v in value.items()}
            if isinstance(value, np.ndarray):
                if framework == "tensorflow":
                    return tf.constant(value)
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
            for prefix in ["tf.", "tensorflow.", "torch.", "paddle.", "np.", "numpy."]:
                if token.startswith(prefix):
                    token = token[len(prefix):]

            if framework == "tensorflow":
                return getattr(tf, token, dtype_value)
            if framework == "paddle":
                return getattr(paddle, token, dtype_value)
            return dtype_value

        # Explicit args/kwargs style from LLM
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

        # Varargs handling
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
                        if framework == "tensorflow":
                            args.append(tf.constant(np_data))
                        else:
                            args.append(paddle.to_tensor(np_data))
                    else:
                        args.append(item)
            return args, kwargs

        # Process positional params in order
        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if param_name == "dtype":
                    args.append(normalize_dtype(value))
                else:
                    args.append(convert_value(value))

        # Process other params (keyword args)
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

    # ==================== Result Comparison ====================

    def compare_tensors(
        self, pd_result, tf_result, tolerance: float = 1e-5
    ) -> Tuple[bool, str]:
        """Compare Paddle and TensorFlow results."""
        try:
            # Convert to numpy
            if isinstance(pd_result, paddle.Tensor):
                pd_np = pd_result.numpy()
            elif isinstance(pd_result, np.ndarray):
                pd_np = pd_result
            else:
                pd_np = np.array(pd_result)

            if isinstance(tf_result, tf.Tensor):
                tf_np = tf_result.numpy()
            elif isinstance(tf_result, np.ndarray):
                tf_np = tf_result
            else:
                tf_np = np.array(tf_result)

            # Shape consistency check
            if pd_np.shape != tf_np.shape:
                return False, f"Shape mismatch: PD={pd_np.shape} vs TF={tf_np.shape}"

            # Exact comparison for boolean types
            if pd_np.dtype == np.bool_ or tf_np.dtype == np.bool_:
                match = np.array_equal(pd_np, tf_np)
                if match:
                    return True, "Boolean results match exactly"
                else:
                    diff_count = np.sum(pd_np != tf_np)
                    return False, f"Boolean results differ; differing elements: {diff_count}"

            # Numeric comparison
            if np.allclose(pd_np, tf_np, atol=tolerance, rtol=tolerance, equal_nan=True):
                return True, "Results match (within tolerance)"
            else:
                max_diff = np.max(np.abs(pd_np.astype(np.float64) - tf_np.astype(np.float64)))
                return False, f"Results inconsistent, max diff: {max_diff:.8f}"

        except Exception as e:
            return False, f"Comparison error: {str(e)}"

    # ==================== Test Execution ====================

    def execute_test_case(
        self,
        pd_api: str,
        tensorflow_api: str,
        pd_test_case: Dict[str, Any],
        tensorflow_test_case: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single test case.

        Args:
            pd_api: PaddlePaddle API name
            tensorflow_api: TensorFlow API name
            pd_test_case: Paddle test case
            tensorflow_test_case: TF test case (if None, use Paddle case)
        """
        if tensorflow_test_case is None:
            tensorflow_test_case = pd_test_case

        effective_pd_api = pd_test_case.get("api", pd_api) if isinstance(pd_test_case, dict) else pd_api
        effective_tf_api = tensorflow_test_case.get("api", tensorflow_api) if isinstance(tensorflow_test_case, dict) else tensorflow_api

        result = {
            "pd_api": effective_pd_api,
            "tensorflow_api": effective_tf_api,
            "pd_success": False,
            "tensorflow_success": False,
            "results_match": False,
            "pd_error": None,
            "tensorflow_error": None,
            "comparison_error": None,
            "pd_shape": None,
            "tensorflow_shape": None,
            "pd_dtype": None,
            "tensorflow_dtype": None,
            "status": "unknown",
        }

        # Materialize shared input tensors to keep PD/TF inputs identical
        pd_test_case, tensorflow_test_case = self._materialize_shared_tensors(
            effective_pd_api, effective_tf_api, pd_test_case, tensorflow_test_case
        )

        is_class_pd = self.is_class_based_api(effective_pd_api)
        is_class_tf = self.is_class_based_api(effective_tf_api)

        # ---- Execute PaddlePaddle ----
        pd_result = None
        try:
            pd_func = self.get_operator_function(effective_pd_api, "paddle")
            if pd_func is None:
                raise AttributeError(f"Paddle API not found: {effective_pd_api}")

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

        # ---- Execute TensorFlow ----
        tf_result = None
        try:
            tf_func = self.get_operator_function(effective_tf_api, "tensorflow")
            if tf_func is None:
                raise AttributeError(f"TF API not found: {effective_tf_api}")

            if effective_pd_api == "paddle.add_n" and effective_tf_api in ("tf.math.add_n", "tf.add_n"):
                candidate_inputs = tensorflow_test_case.get("inputs") or pd_test_case.get("inputs")
                if isinstance(candidate_inputs, list) and len(candidate_inputs) >= 1:
                    tensor_list = []
                    for item in candidate_inputs:
                        if isinstance(item, tf.Tensor):
                            tensor_list.append(item)
                        elif isinstance(item, np.ndarray):
                            tensor_list.append(tf.constant(item))
                        elif isinstance(item, paddle.Tensor):
                            tensor_list.append(tf.constant(item.numpy()))
                        elif isinstance(item, dict) and "shape" in item:
                            np_data = self.generate_numpy_data(item)
                            tensor_list.append(tf.constant(np_data))
                        else:
                            tensor_list.append(tf.constant(item))
                    tf_result = tf.math.add_n(tensor_list)
                    result["tensorflow_success"] = True
                    if hasattr(tf_result, "shape"):
                        result["tensorflow_shape"] = list(tf_result.shape)
                    if hasattr(tf_result, "dtype"):
                        result["tensorflow_dtype"] = str(tf_result.dtype)
                else:
                    raise ValueError("tf.math.add_n equivalent execution requires an inputs list")

            elif is_class_tf:
                init_kwargs = {
                    k: v for k, v in tensorflow_test_case.items()
                    if k not in ["api", "input", "x"] and not isinstance(v, (np.ndarray,))
                    and not (isinstance(v, dict) and "shape" in v)
                }
                module = tf_func(**init_kwargs)
                input_data = tensorflow_test_case.get("input") or tensorflow_test_case.get("x")
                if input_data is not None:
                    if isinstance(input_data, dict) and "shape" in input_data:
                        np_data = self.generate_numpy_data(input_data)
                        tf_input = tf.constant(np_data)
                    elif isinstance(input_data, np.ndarray):
                        tf_input = tf.constant(input_data)
                    else:
                        tf_input = tf.constant(input_data)
                    tf_result = module(tf_input)
                else:
                    tf_result = module(tf.constant(np.random.randn(2, 3).astype(np.float32)))
            else:
                tf_args, tf_kwargs = self.prepare_arguments(tensorflow_test_case, "tensorflow")
                tf_result = tf_func(*tf_args, **tf_kwargs)

            if not result["tensorflow_success"]:
                result["tensorflow_success"] = True
                if hasattr(tf_result, "shape"):
                    result["tensorflow_shape"] = list(tf_result.shape)
                if hasattr(tf_result, "dtype"):
                    result["tensorflow_dtype"] = str(tf_result.dtype)

        except Exception as e:
            result["tensorflow_error"] = f"{type(e).__name__}: {str(e)}"

        # ---- Compare results ----
        if result["pd_success"] and result["tensorflow_success"]:
            try:
                match, detail = self.compare_tensors(pd_result, tf_result)
                result["results_match"] = match
                result["comparison_error"] = None if match else detail
                result["status"] = "consistent" if match else "inconsistent"
            except Exception as e:
                result["comparison_error"] = str(e)
                result["status"] = "comparison_error"
        elif result["pd_success"] and not result["tensorflow_success"]:
            result["status"] = "tensorflow_error"
        elif not result["pd_success"] and result["tensorflow_success"]:
            result["status"] = "paddle_error"
        else:
            result["status"] = "both_error"

        return result

    def _execute_test_case_sequential(
        self, pd_api, tensorflow_api, pd_test_case, tensorflow_test_case=None
    ) -> Dict[str, Any]:
        """Use a lock to ensure serialized execution."""
        with self.execution_lock:
            return self.execute_test_case(pd_api, tensorflow_api, pd_test_case, tensorflow_test_case)

    def _default_input_desc_for_class(self, api_name: str) -> Dict[str, Any]:
        """Provide a default input description for class APIs."""
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
        tensorflow_api: str,
        pd_test_case: Dict[str, Any],
        tensorflow_test_case: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Materialize shared tensors to keep inputs numerically identical."""
        pd_case = copy.deepcopy(pd_test_case)
        tf_case = copy.deepcopy(tensorflow_test_case)

        is_class_pd = self.is_class_based_api(pd_api)
        is_class_tf = self.is_class_based_api(tensorflow_api)
        if (is_class_pd or is_class_tf) and not (
            "input" in pd_case or "x" in pd_case or "input" in tf_case or "x" in tf_case
        ):
            default_desc = self._default_input_desc_for_class(pd_api or tensorflow_api)
            pd_case.setdefault("input", default_desc)
            tf_case.setdefault("input", default_desc)

        def is_tensor_desc(value: Any) -> bool:
            return isinstance(value, dict) and "shape" in value

        def clone_array(value: np.ndarray) -> np.ndarray:
            return value.copy()

        def materialize_pair(pd_val: Any, tf_val: Any) -> Tuple[Any, Any]:
            if isinstance(pd_val, np.ndarray):
                return clone_array(pd_val), clone_array(pd_val)
            if isinstance(tf_val, np.ndarray):
                return clone_array(tf_val), clone_array(tf_val)

            if is_tensor_desc(pd_val) or is_tensor_desc(tf_val):
                tensor_desc = pd_val if is_tensor_desc(pd_val) else tf_val
                shared = self.generate_numpy_data(tensor_desc)
                return clone_array(shared), clone_array(shared)

            if isinstance(pd_val, list) or isinstance(tf_val, list):
                pd_list = pd_val if isinstance(pd_val, list) else []
                tf_list = tf_val if isinstance(tf_val, list) else []
                size = max(len(pd_list), len(tf_list))
                out_pd = []
                out_tf = []
                for index in range(size):
                    left = pd_list[index] if index < len(pd_list) else None
                    right = tf_list[index] if index < len(tf_list) else None
                    new_left, new_right = materialize_pair(left, right)
                    if index < len(pd_list):
                        out_pd.append(new_left)
                    if index < len(tf_list):
                        out_tf.append(new_right)
                return out_pd if isinstance(pd_val, list) else pd_val, out_tf if isinstance(tf_val, list) else tf_val

            if isinstance(pd_val, dict) or isinstance(tf_val, dict):
                pd_dict = pd_val if isinstance(pd_val, dict) else {}
                tf_dict = tf_val if isinstance(tf_val, dict) else {}
                keys = set(pd_dict.keys()) | set(tf_dict.keys())
                out_pd = {}
                out_tf = {}
                for key in keys:
                    if key == "api":
                        if key in pd_dict:
                            out_pd[key] = pd_dict[key]
                        if key in tf_dict:
                            out_tf[key] = tf_dict[key]
                        continue
                    new_left, new_right = materialize_pair(pd_dict.get(key), tf_dict.get(key))
                    if key in pd_dict:
                        out_pd[key] = new_left
                    if key in tf_dict:
                        out_tf[key] = new_right
                return out_pd if isinstance(pd_val, dict) else pd_val, out_tf if isinstance(tf_val, dict) else tf_val

            return pd_val, tf_val

        pd_case, tf_case = materialize_pair(pd_case, tf_case)

        # Common cross-framework alias parameter mapping
        if isinstance(pd_case, dict) and isinstance(tf_case, dict):
            alias_pairs = [
                ("x", "input"),
                ("input", "x"),
                ("y", "other"),
                ("other", "y"),
            ]
            for pd_key, tf_key in alias_pairs:
                if pd_key in pd_case and tf_key in tf_case:
                    pd_item, tf_item = materialize_pair(pd_case[pd_key], tf_case[tf_key])
                    pd_case[pd_key] = pd_item
                    tf_case[tf_key] = tf_item

        # Common add_n cross-key mapping: PD inputs[0/1] <-> TF input/other
        pd_inputs = pd_case.get("inputs") if isinstance(pd_case, dict) else None
        if isinstance(pd_inputs, list) and isinstance(tf_case, dict):
            if len(pd_inputs) >= 1 and "input" in tf_case:
                pd_item, tf_item = materialize_pair(pd_inputs[0], tf_case.get("input"))
                pd_inputs[0] = pd_item
                tf_case["input"] = tf_item
            if len(pd_inputs) >= 2 and "other" in tf_case:
                pd_item, tf_item = materialize_pair(pd_inputs[1], tf_case.get("other"))
                pd_inputs[1] = pd_item
                tf_case["other"] = tf_item

        return pd_case, tf_case

    # ==================== API Docs Fetching ====================

    def _fetch_api_docs(self, pd_api: str, tensorflow_api: str) -> Tuple[str, str]:
        """Fetch Paddle and TensorFlow API docs."""
        MIN_DOC_LENGTH = 300
        pd_doc = ""
        tensorflow_doc = ""

        try:
            raw = get_doc_content(pd_api, "paddle")
            if raw and len(raw) >= MIN_DOC_LENGTH:
                pd_doc = raw[:3000]
                self._safe_print(f"    📄 PD docs: {len(pd_doc)} chars")
            else:
                self._safe_print(f"    📄 PD docs: no valid content")
        except Exception as e:
            self._safe_print(f"    ⚠️ PD docs fetch failed: {str(e)[:50]}")

        try:
            raw = get_doc_content(tensorflow_api, "tensorflow")
            if raw and len(raw) >= MIN_DOC_LENGTH:
                tensorflow_doc = raw[:3000]
                self._safe_print(f"    📄 TF docs: {len(tensorflow_doc)} chars")
            else:
                self._safe_print(f"    📄 TF docs: no valid content")
        except Exception as e:
            self._safe_print(f"    ⚠️ TF docs fetch failed: {str(e)[:50]}")

        return pd_doc, tensorflow_doc

    # ==================== LLM Interaction ====================

    def _build_llm_prompt(
        self,
        execution_result: Dict[str, Any],
        pd_test_case: Dict[str, Any],
        tensorflow_test_case: Dict[str, Any],
        pd_doc: str = "",
        tensorflow_doc: str = "",
    ) -> str:
        """Build the LLM prompt."""
        pd_api = execution_result.get("pd_api", "")
        tensorflow_api = execution_result.get("tensorflow_api", "")
        status = execution_result.get("status", "")
        pd_success = execution_result.get("pd_success", False)
        tensorflow_success = execution_result.get("tensorflow_success", False)
        results_match = execution_result.get("results_match", False)
        pd_error = execution_result.get("pd_error", "")
        tensorflow_error = execution_result.get("tensorflow_error", "")
        comparison_error = execution_result.get("comparison_error", "")

        # Simplify test cases to reduce token usage
        simplified_pd = {}
        for key, value in pd_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_pd[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_pd[key] = value

        simplified_tf = {}
        for key, value in tensorflow_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_tf[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_tf[key] = value

        # Build parameter example strings
        pd_param_examples = []
        for key, value in simplified_pd.items():
            if key == "api":
                continue
            pd_param_examples.append(f'    "{key}": {json.dumps(value)}')

        pd_param_str = ",\n".join(pd_param_examples) if pd_param_examples else '    "x": {"shape": [2, 3], "dtype": "float32"}'

        tf_param_examples = []
        for key, value in simplified_tf.items():
            if key == "api":
                continue
            tf_param_examples.append(f'    "{key}": {json.dumps(value)}')

        tf_param_str = ",\n".join(tf_param_examples) if tf_param_examples else '    "input": {"shape": [2, 3], "dtype": "float32"}'

        # Docs section
        doc_section = ""
        if pd_doc or tensorflow_doc:
            doc_section = "\n## Official API Docs Reference\n\n"
            if pd_doc:
                doc_section += f"### PaddlePaddle {pd_api} Docs\n```\n{pd_doc}\n```\n\n"
            if tensorflow_doc:
                doc_section += f"### TensorFlow {tensorflow_api} Docs\n```\n{tensorflow_doc}\n```\n\n"

                prompt = f"""Please analyze the execution results of the following operator test case in PaddlePaddle and TensorFlow, and then decide whether to repair or mutate (fuzz) the test case.

## Test Info
- **PaddlePaddle API**: {pd_api}
- **TensorFlow API**: {tensorflow_api}
{doc_section}
## Execution Results
- **Status**: {status}
- **PaddlePaddle success**: {pd_success}
- **TensorFlow success**: {tensorflow_success}
- **Results match**: {results_match}

## Error Info
- **PaddlePaddle error**: {pd_error if pd_error else "none"}
- **TensorFlow error**: {tensorflow_error if tensorflow_error else "none"}
- **Comparison error**: {comparison_error if comparison_error else "none"}

## Original Test Cases

### PaddlePaddle test case
```json
{json.dumps(simplified_pd, indent=2, ensure_ascii=False)}
```

### TensorFlow test case
```json
{json.dumps(simplified_tf, indent=2, ensure_ascii=False)}
```

## Requirements
Based on the information above (including official API docs), decide whether the results are **consistent**, **inconsistent**, or **execution error**, and then do the following:

1. **If consistent**: perform **mutation (fuzzing)**, e.g., change input tensor shapes or parameter values (consider extreme/boundary values).
2. **If execution error**: **repair** the test case based on the error and docs (change parameter names, counts, types, ranges; frameworks may differ) or **skip** when: (1) docs are missing or indicate removal; (2) the two APIs are not fully equivalent.
3. **If inconsistent**: judge whether it is a tolerable precision error (<= 1e-3). (1) If tolerable, **mutate**; (2) if docs suggest the APIs are not equivalent, **skip**; (3) otherwise treat it as test construction error and **repair** based on docs.
4. **Strict parameter alignment (must follow)**: if parameters cannot be aligned one-to-one, you must **manually add/remove/modify parameters** to make both sides as equivalent as possible before comparing. Do not merely copy parameters.

## Output Format
Return strictly in the following JSON format with no extra text, comments, or markdown:

{{
    "operation": "mutation",
    "reason": "Detailed reason (<=150 chars)",
    "paddle_test_case": {{
        "api": "{pd_api}",
{pd_param_str}
    }},
        "tensorflow_test_case": {{
                "api": "{tensorflow_api}",
{tf_param_str}
    }}
}}

**Important**:
1. operation must be one of "mutation", "repair", or "skip".
2. Tensor params must use {{"shape": [...], "dtype": "..."}}.
3. Scalar params should be literal values.
4. Inputs must be identical and parameters semantically aligned across frameworks.
5. Paddle and TensorFlow cases can differ in param names (e.g., x vs input), values, or counts as long as outputs are theoretically identical.
6. If docs are missing or indicate removal, set operation to "skip" and do not attempt repair.
7. Mutation can explore extremes: empty tensors, single-element tensors, high-dimensional tensors, different dtypes, boundary values.
8. Read official docs carefully to ensure parameter names/types/ranges are valid.
9. For convolution ops, note TF NHWC vs Paddle NCHW and keep semantics aligned.
10. If parameters do not align, prefer semantic-equivalent rewrites over mechanical same-name mapping. **Example**: `paddle.nn.functional.conv2d` may have `groups`, while `tf.nn.conv2d` has no `groups`. Remove/transform that parameter and adjust inputs/kernels or use equivalent expressions; explicitly handle `data_format` (NCHW/NHWC) to preserve equivalence.
"""
        return prompt

    def call_llm_for_repair_or_mutation(
        self,
        execution_result: Dict[str, Any],
        pd_test_case: Dict[str, Any],
        tensorflow_test_case: Dict[str, Any],
        pd_doc: str = "",
        tensorflow_doc: str = "",
    ) -> Dict[str, Any]:
        """Call the LLM to repair or mutate test cases."""
        prompt = self._build_llm_prompt(
            execution_result, pd_test_case, tensorflow_test_case, pd_doc, tensorflow_doc
        )
        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a deep learning framework testing expert familiar with PaddlePaddle and TensorFlow API differences. Based on execution results, decide whether to repair or mutate the test case and return strict JSON output.",
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
                self._safe_print("    ⚠️ LLM response is not valid JSON; attempting extraction...")
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    llm_result = json.loads(json_match.group())
                    return llm_result
                else:
                    return {
                        "operation": "skip",
                        "reason": "LLM response format error",
                        "paddle_test_case": pd_test_case,
                        "tensorflow_test_case": tensorflow_test_case,
                    }

        except Exception as e:
            self._safe_print(f"    ❌ LLM call failed: {e}")
            return {
                "operation": "skip",
                "reason": f"LLM call failed: {e}",
                "paddle_test_case": pd_test_case,
                "tensorflow_test_case": tensorflow_test_case,
            }

    # ==================== Core Test Loop ====================

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
            num_workers: LLM worker threads
        """
        self._safe_print(f"\n{'=' * 80}")
        self._safe_print(f"🎯 Start testing operator: {pd_api}")
        self._safe_print(f"🔄 Max iterations per case: {max_iterations}")
        self._safe_print(f"{'=' * 80}\n")

        stats = {"llm_generated_cases": 0, "successful_cases": 0}

        # Get corresponding TensorFlow API
        _, tensorflow_api, mapping_method = self.convert_api_name(pd_api)
        if tensorflow_api is None:
            self._safe_print(f"❌ {pd_api} has no TensorFlow equivalent")
            return [], stats

        self._safe_print(f"✅ PaddlePaddle API: {pd_api}")
        self._safe_print(f"✅ TensorFlow API: {tensorflow_api}")
        self._safe_print(f"✅ Mapping method: {mapping_method}")

        # Get test cases
        api_data = self.test_cases_data.get(pd_api, {})
        test_cases = api_data.get("test_cases", [])

        if not test_cases:
            self._safe_print(f"⚠️ No test cases found for {pd_api}; using defaults")
            test_cases = [{"description": "default", "inputs": {"x": {"shape": [2, 3], "dtype": "float32"}}}]

        # Determine actual test count
        if num_test_cases is None:
            num_test_cases = len(test_cases)
        else:
            num_test_cases = min(num_test_cases, len(test_cases))

        self._safe_print(f"📋 Will test {num_test_cases} cases (LLM workers={num_workers}, serialized execution)")

        # Prepare initial cases
        initial_cases = []
        for case_idx in range(num_test_cases):
            tc = test_cases[case_idx]
            # Extract params from inputs to build a flat test case
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
                    pd_api, tensorflow_api, initial_test_case,
                    max_iterations, case_number, stats,
                )
                all_results.extend(case_results)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_case = {}
                for case_number, initial_test_case in initial_cases:
                    future = executor.submit(
                        self._test_single_case_with_iterations,
                        pd_api, tensorflow_api, initial_test_case,
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
        self._safe_print(f"📊 LLM-generated test cases: {stats['llm_generated_cases']}")
        self._safe_print(f"📊 Cases where both frameworks succeeded: {stats['successful_cases']}")
        self._safe_print(f"{'=' * 80}\n")

        return all_results, stats

    def _test_single_case_with_iterations(
        self,
        pd_api: str,
        tensorflow_api: str,
        initial_test_case: Dict[str, Any],
        max_iterations: int,
        case_number: int,
        stats: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """
        Run multiple iterations for a single test case.

        Core loop: execute -> LLM decision -> repair/mutate/skip -> execute -> ...
        """
        case_results = []

        # Build initial PD and TF test cases
        current_pd_test_case = copy.deepcopy(initial_test_case)
        current_pd_test_case["api"] = pd_api

        current_tf_test_case = copy.deepcopy(initial_test_case)
        current_tf_test_case["api"] = tensorflow_api

        is_llm_generated = False

        # Prefetch API docs (once)
        self._safe_print("  📖 Prefetching API docs...")
        pd_doc, tensorflow_doc = self._fetch_api_docs(pd_api, tensorflow_api)

        # Iterative testing
        for iteration in range(max_iterations):
            source_type = "LLM" if is_llm_generated else "file"
            self._safe_print(f"  🔄 Iteration {iteration + 1}/{max_iterations} ({source_type})", end="")

            current_pd_api = current_pd_test_case.get("api", pd_api) or pd_api
            current_tf_api = current_tf_test_case.get("api", tensorflow_api) or tensorflow_api

            # Execute test
            try:
                execution_result = self._execute_test_case_sequential(
                    current_pd_api, current_tf_api, current_pd_test_case, current_tf_test_case
                )

                pd_status = "✓" if execution_result["pd_success"] else "✗"
                tf_status = "✓" if execution_result["tensorflow_success"] else "✗"
                match_status = "✓" if execution_result["results_match"] else "✗"
                self._safe_print(f" | PD:{pd_status} TF:{tf_status} Match:{match_status}")

                if execution_result["pd_error"] and not execution_result["pd_success"]:
                    self._safe_print(f"    ❌ PD error: {str(execution_result['pd_error'])[:100]}...")
                if execution_result["tensorflow_error"] and not execution_result["tensorflow_success"]:
                    self._safe_print(f"    ❌ TF error: {str(execution_result['tensorflow_error'])[:100]}...")
                if execution_result["comparison_error"]:
                    self._safe_print(f"    ⚠️ Comparison: {str(execution_result['comparison_error'])[:100]}...")

                # Count LLM-generated cases
                if is_llm_generated:
                    if execution_result["pd_success"] and execution_result["tensorflow_success"]:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

            except Exception as e:
                self._safe_print(f" | ❌ Fatal error: {str(e)[:80]}...")
                execution_result = {
                    "status": "fatal_error",
                    "pd_success": False, "tensorflow_success": False,
                    "results_match": False,
                    "pd_error": f"Fatal: {str(e)}", "tensorflow_error": None,
                    "comparison_error": None,
                }

            # Save iteration result
            iteration_result = {
                "iteration": iteration + 1,
                "pd_test_case": current_pd_test_case,
                "tensorflow_test_case": current_tf_test_case,
                "execution_result": execution_result,
                "llm_operation": None,
                "is_llm_generated": is_llm_generated,
                "case_number": case_number,
            }

            # Call LLM
            try:
                llm_result = self.call_llm_for_repair_or_mutation(
                    execution_result, current_pd_test_case, current_tf_test_case,
                    pd_doc, tensorflow_doc,
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

            # Prepare next round
            if operation in ("mutation", "repair"):
                next_pd_case = llm_result.get("paddle_test_case", current_pd_test_case)
                next_tf_case = llm_result.get("tensorflow_test_case", current_tf_test_case)
                with self.stats_lock:
                    stats["llm_generated_cases"] += 1
                is_llm_generated = True
            else:
                next_pd_case = current_pd_test_case
                next_tf_case = current_tf_test_case

            current_pd_test_case, current_tf_test_case = self._convert_llm_test_cases(
                next_pd_case, next_tf_case
            )

        # If last LLM round generated a new case but wasn't executed, run it
        if case_results:
            last_op = case_results[-1].get("llm_operation", {}).get("operation", "skip")
            if last_op in ("mutation", "repair"):
                self._safe_print("  🔄 Executing final LLM case", end="")
                try:
                    execution_result = self._execute_test_case_sequential(
                        current_pd_test_case.get("api", pd_api) or pd_api,
                        current_tf_test_case.get("api", tensorflow_api) or tensorflow_api,
                        current_pd_test_case,
                        current_tf_test_case,
                    )
                    pd_s = "✓" if execution_result["pd_success"] else "✗"
                    tf_s = "✓" if execution_result["tensorflow_success"] else "✗"
                    m_s = "✓" if execution_result["results_match"] else "✗"
                    self._safe_print(f" | PD:{pd_s} TF:{tf_s} Match:{m_s}")

                    if execution_result["pd_success"] and execution_result["tensorflow_success"]:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

                    case_results.append({
                        "iteration": len(case_results) + 1,
                        "pd_test_case": current_pd_test_case,
                        "tensorflow_test_case": current_tf_test_case,
                        "execution_result": execution_result,
                        "llm_operation": {"operation": "final_execution", "reason": "Execute the final LLM-generated case"},
                        "case_number": case_number,
                        "is_llm_generated": True,
                    })
                except Exception as e:
                    self._safe_print(f"  ❌ Final case execution failed: {str(e)[:80]}...")
                    case_results.append({
                        "iteration": len(case_results) + 1,
                        "pd_test_case": current_pd_test_case,
                        "tensorflow_test_case": current_tf_test_case,
                        "execution_result": {
                            "status": "fatal_error", "pd_success": False,
                            "tensorflow_success": False, "results_match": False,
                            "error": str(e),
                        },
                        "llm_operation": {"operation": "final_execution", "reason": "Final case execution failed"},
                        "case_number": case_number,
                        "is_llm_generated": True,
                    })

        self._safe_print(f"  ✅ Case {case_number} completed, {len(case_results)} iterations")
        return case_results

    def _convert_llm_test_cases(
        self,
        pd_test_case: Dict[str, Any],
        tensorflow_test_case: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Convert LLM-returned test cases into executable format,
        ensuring both frameworks share identical tensor data.
        """
        pd_api = pd_test_case.get("api", "") if isinstance(pd_test_case, dict) else ""
        tf_api = tensorflow_test_case.get("api", "") if isinstance(tensorflow_test_case, dict) else ""
        return self._materialize_shared_tensors(pd_api, tf_api, pd_test_case, tensorflow_test_case)

    # ==================== Result Saving ====================

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
            for case_key in ["pd_test_case", "tensorflow_test_case"]:
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
            "tensorflow_api": self.api_mapping.get(pd_api, ""),
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
        """Get all testable Paddle APIs (have test cases and TF mappings)."""
        testable = []
        for pd_api in sorted(self.test_cases_data.keys()):
            tf_api = self.api_mapping.get(pd_api, "no_equivalent_impl")
            if tf_api and tf_api != "no_equivalent_impl":
                testable.append(pd_api)
        return testable

    def close(self):
        """Clean up resources."""
        pass


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="LLM-based differential testing framework for PaddlePaddle and TensorFlow"
    )
    parser.add_argument(
        "--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"Max iterations per test case (default {DEFAULT_MAX_ITERATIONS})"
    )
    parser.add_argument(
        "--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES,
        help=f"Test cases per operator (default {DEFAULT_NUM_CASES})"
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
        help=f"Worker threads (default {DEFAULT_WORKERS})"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"LLM model name (default {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--key-path", "-k", default=DEFAULT_KEY_PATH,
        help=f"API key file path (default {DEFAULT_KEY_PATH})"
    )
    parser.add_argument(
        "--test-cases-file", default=DEFAULT_TEST_CASES_FILE,
        help="Test case JSON file path"
    )
    parser.add_argument(
        "--mapping-file", default=DEFAULT_MAPPING_FILE,
        help="PD→TF mapping CSV file path"
    )

    args = parser.parse_args()
    num_workers = max(1, args.workers)

    print("=" * 80)
    print("LLM-based differential testing framework for PaddlePaddle and TensorFlow")
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
        # Get all testable APIs
        all_testable = comparator.get_all_testable_apis()
        print(f"\n🔍 Total testable Paddle APIs: {len(all_testable)}")

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
            print(f"📌 Test range: operators {start_idx + 1} to {end_idx}")
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
        log_file.write("PD→TF differential test batch log\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("Test config:\n")
        log_file.write(f"  - Iterations: {args.max_iterations}\n")
        log_file.write(f"  - Cases: {args.num_cases}\n")
        log_file.write(f"  - Workers: {num_workers}\n")
        log_file.write(f"  - Operators: {len(operator_names)}\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.flush()

        for idx, pd_api in enumerate(operator_names, 1):
            print("\n" + "🔷" * 40)
            print(f"🎯 [{idx}/{len(operator_names)}] Start testing operator: {pd_api}")
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
                        "tensorflow_api": comparator.api_mapping.get(pd_api, ""),
                        "total_iterations": len(results),
                        "llm_generated_cases": stats.get("llm_generated_cases", 0),
                        "successful_cases": stats.get("successful_cases", 0),
                        "status": "completed",
                    })
                    print(f"\n✅ {pd_api} test completed")
                    print(f"   - Total iterations: {len(results)}")
                    print(f"   - LLM-generated cases: {stats.get('llm_generated_cases', 0)}")
                    print(f"   - Successful cases: {stats.get('successful_cases', 0)}")

                    log_file.write(f"[{idx}/{len(operator_names)}] {pd_api}\n")
                    log_file.write("  Status: ✅ completed\n")
                    log_file.write(f"  Total iterations: {len(results)}\n")
                    log_file.write(f"  LLM-generated cases: {stats.get('llm_generated_cases', 0)}\n")
                    log_file.write(f"  Successful cases: {stats.get('successful_cases', 0)}\n")
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
                    log_file.write("  Status: ⚠️ no results\n\n")
                    log_file.flush()

            except Exception as e:
                print(f"\n❌ {pd_api} test failed: {e}")
                all_operators_summary.append({
                    "operator": pd_api, "total_iterations": 0,
                    "llm_generated_cases": 0, "successful_cases": 0,
                    "status": "failed", "error": str(e),
                })
                log_file.write(f"[{idx}/{len(operator_names)}] {pd_api}\n")
                log_file.write(f"  Status: ❌ failed\n  Error: {str(e)}\n\n")
                log_file.flush()
                continue

        # ==================== Summary Output ====================
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
        print(f"\n⏱️ Runtime: {hours}h {minutes}m {seconds}s")

        # Write logs
        log_file.write("=" * 80 + "\nOverall Summary\n" + "=" * 80 + "\n")
        log_file.write(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total runtime: {hours}h {minutes}m {seconds}s ({total_duration:.2f}s)\n\n")
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
        print("\n✅ Batch test run completed")


if __name__ == "__main__":
    main()
