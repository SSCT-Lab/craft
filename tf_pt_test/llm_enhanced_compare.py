#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4: LLM-based TensorFlow ↔ PyTorch operator differential testing framework

Features:
- Load TF test cases and TF→PT mappings from JSON/CSV
- Execute TF and PT for each equivalent operator pair and compare results
- Use LLM to repair/mutate/skip test cases
- Support concurrent testing
- Save detailed results and batch logs

Usage:
    conda activate tf_env
    python tf_pt_test/llm_enhanced_compare.py \
        [--max-iterations 3] [--num-cases 5] [--workers 6] \
        [--start 1] [--end N] [--operators tf.nn.relu tf.math.abs]

Prerequisites:
    1. Run Step 1 extract_tf_apis.py
    2. Run Step 2 extract_tf_test_cases.py
    3. Run Step 3 extract_tf_pt_mapping.py
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

# Add project root to path.
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

# Data file paths.
DATA_DIR = os.path.join(ROOT_DIR, "tf_pt_test", "data")
DEFAULT_TEST_CASES_FILE = os.path.join(DATA_DIR, "tf_test_cases.json")
DEFAULT_MAPPING_FILE = os.path.join(DATA_DIR, "tf_pt_mapping_validated.csv")


class LLMEnhancedComparator:
    """LLM-based TensorFlow ↔ PyTorch differential testing framework."""

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
            test_cases_file: Test-case JSON file path (Step 2 output)
            mapping_file: TF→PT mapping CSV file path (Step 3 output)
            key_path: API key file path
            model: LLM model name
            print_lock: Print lock
            llm_workers: LLM concurrent workers
        """
        self.model = model
        self.print_lock = print_lock or Lock()
        self.llm_workers = max(1, int(llm_workers))
        self.execution_lock = RLock()
        self.stats_lock = Lock()

        # Initialize LLM client.
        api_key = self._load_api_key(key_path)
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # Load test cases.
        self.test_cases_data = self._load_test_cases(test_cases_file)
        self._safe_print(f"📋 Loaded test cases for {len(self.test_cases_data)} TF APIs")

        # Load TF→PT mapping.
        self.api_mapping = self._load_mapping(mapping_file)
        has_impl = sum(1 for v in self.api_mapping.values() if v != "无对应实现")
        self._safe_print(f"📋 Loaded {len(self.api_mapping)} mappings ({has_impl} with implementations)")

        # Create result directory.
        self.result_dir = os.path.join(ROOT_DIR, "tf_pt_test", "tf_pt_log_1")
        os.makedirs(self.result_dir, exist_ok=True)
        self._safe_print(f"📁 Result directory: {self.result_dir}")

        # Fix random seeds.
        self.random_seed = 42
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

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
            self._safe_print(f"⚠️ Test cases file does not exist: {filepath}")
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("test_cases", {})

    def _load_mapping(self, filepath: str) -> Dict[str, str]:
        """Load TF→PT mapping table."""
        if not os.path.exists(filepath):
            self._safe_print(f"⚠️ Mapping file does not exist: {filepath}")
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

    # ==================== API helpers ====================

    def is_class_based_api(self, api_name: str) -> bool:
        """Return True if the API is class-based."""
        parts = api_name.split(".")
        if len(parts) >= 2:
            last_part = parts[-1]
            if last_part and last_part[0].isupper():
                return True
        return False

    def get_operator_function(self, api_name: str, framework: str = "tf"):
        """Get the operator function object."""
        try:
            if framework == "tf":
                module = tf
            elif framework == "torch":
                module = torch
            else:
                return None

            parts = api_name.split(".")
            # Skip framework prefix (tf. / torch.).
            start_idx = 1
            obj = module
            for part in parts[start_idx:]:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return None

    def convert_api_name(self, tf_api: str) -> Tuple[Optional[str], Optional[str], str]:
        """Find the corresponding PyTorch API for a TF API."""
        if tf_api in self.api_mapping:
            pt_api = self.api_mapping[tf_api]
            if pt_api and pt_api != "无对应实现":
                return tf_api, pt_api, "mapping_table"
            else:
                return tf_api, None, "no_implementation"
        return tf_api, None, "not_found_in_mapping"

    # ==================== Data conversion ====================

    def generate_numpy_data(self, data: Any) -> np.ndarray:
        """
        Generate a numpy array from a description.

        Supported formats:
        - {"shape": [2, 3], "dtype": "float32"}
        - {"shape": [2, 3], "dtype": "float32", "range": [-1, 1]}
        - scalar value
        - list
        """
        if isinstance(data, dict):
            if "shape" in data:
                shape = data["shape"]
                dtype_str = str(data.get("dtype", "float32"))

                # Remove framework prefix.
                for prefix in ["torch.", "tf.", "np.", "numpy."]:
                    if dtype_str.startswith(prefix):
                        dtype_str = dtype_str[len(prefix):]

                # dtype mapping.
                dtype_map = {
                    "float32": np.float32, "float64": np.float64,
                    "float16": np.float16, "float": np.float32,
                    "int32": np.int32, "int64": np.int64,
                    "int16": np.int16, "int8": np.int8,
                    "uint8": np.uint8, "bool": np.bool_,
                    "complex64": np.complex64, "complex128": np.complex128,
                    "bfloat16": np.float32,  # numpy does not support bfloat16; use float32.
                }
                np_dtype = dtype_map.get(dtype_str, np.float32)

                # Handle empty tensors.
                if any(s == 0 for s in shape):
                    return np.empty(shape, dtype=np_dtype)

                # Get data range.
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
        """Convert to TensorFlow tensor."""
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
        """Convert to PyTorch tensor."""
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

    # ==================== Argument preparation ====================

    def should_skip_param(self, key: str, api_name: str, framework: str) -> bool:
        """Return True if a parameter should be skipped."""
        # Common skipped parameters.
        common_skip = ["description", "api"]
        if key in common_skip:
            return True

        # PyTorch-specific parameters (skip for TF→PT).
        torch_skip = ["layout", "requires_grad", "out", "memory_format", "pin_memory"]
        if framework == "torch" and key in torch_skip:
            return True

        # TensorFlow-specific parameters (skip for PT→TF).
        tf_skip = ["name"]
        if framework == "tf" and key in tf_skip:
            return True

        return False

    def prepare_arguments(
        self, test_case: Dict[str, Any], framework: str = "tf"
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Prepare arguments for a target framework.

        Args:
            test_case: Test case (tensor descriptions and scalar params)
            framework: "tf" or "torch"

        Returns:
            (args, kwargs)
        """
        args = []
        kwargs = {}

        # Positional parameter names.
        positional_params = ["x", "input", "condition", "y", "other", "a", "b"]

        # Varargs handling.
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

        # Process positional parameters in order.
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

        # Process other parameters (kwargs).
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

    # ==================== Result comparison ====================

    def compare_tensors(
        self, tf_result, torch_result, tolerance: float = 1e-5
    ) -> Tuple[bool, str]:
        """Compare TF and PT results."""
        try:
            # Convert to numpy.
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

            # Shape check.
            if tf_np.shape != torch_np.shape:
                return False, f"Shape mismatch: TF={tf_np.shape} vs PT={torch_np.shape}"

            # Exact comparison for boolean tensors.
            if tf_np.dtype == np.bool_ or torch_np.dtype == np.bool_:
                match = np.array_equal(tf_np, torch_np)
                if match:
                    return True, "Boolean results match exactly"
                else:
                    diff_count = np.sum(tf_np != torch_np)
                    return False, f"Boolean results differ, diff count: {diff_count}"

            # Numeric comparison.
            if np.allclose(tf_np, torch_np, atol=tolerance, rtol=tolerance, equal_nan=True):
                return True, "Results match within tolerance"
            else:
                max_diff = np.max(np.abs(tf_np.astype(np.float64) - torch_np.astype(np.float64)))
                return False, f"Results inconsistent, max diff: {max_diff:.8f}"

        except Exception as e:
            return False, f"Comparison error: {str(e)}"

    # ==================== Test execution ====================

    def execute_test_case(
        self,
        tf_api: str,
        pytorch_api: str,
        tf_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single test case.

        Args:
            tf_api: TensorFlow API name
            pytorch_api: PyTorch API name
            tf_test_case: TF test case
            pytorch_test_case: PT test case (uses TF case if None)
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

        # Materialize shared tensors so TF/PT use the same numpy data.
        tf_test_case, pytorch_test_case = self._materialize_shared_tensors(
            tf_api, pytorch_api, tf_test_case, pytorch_test_case
        )

        is_class_tf = self.is_class_based_api(tf_api)
        is_class_pt = self.is_class_based_api(pytorch_api)

        # ---- Execute TensorFlow ----
        tf_result = None
        try:
            tf_func = self.get_operator_function(tf_api, "tf")
            if tf_func is None:
                raise AttributeError(f"TF API not found: {tf_api}")

            if is_class_tf:
                init_kwargs = {
                    k: v for k, v in tf_test_case.items()
                    if k not in ["api", "input", "x"] and not isinstance(v, (np.ndarray,))
                    and not (isinstance(v, dict) and "shape" in v)
                }
                layer = tf_func(**init_kwargs)
                # Get input.
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

        # ---- Execute PyTorch ----
        torch_result = None
        try:
            pt_func = self.get_operator_function(pytorch_api, "torch")
            if pt_func is None:
                raise AttributeError(f"PT API not found: {pytorch_api}")

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

        # ---- Compare results ----
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
        """Run with a lock to avoid concurrent execution."""
        with self.execution_lock:
            return self.execute_test_case(tf_api, pytorch_api, tf_test_case, pytorch_test_case)

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
        tf_api: str,
        pytorch_api: str,
        tf_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate shared tensors so both frameworks receive identical inputs."""
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

    # ==================== API doc fetching ====================

    def _fetch_api_docs(self, tf_api: str, pytorch_api: str) -> Tuple[str, str]:
        """Fetch TF and PT API docs."""
        MIN_DOC_LENGTH = 300
        tf_doc = ""
        pytorch_doc = ""

        try:
            raw = get_doc_content(tf_api, "tensorflow")
            if raw and len(raw) >= MIN_DOC_LENGTH:
                tf_doc = raw[:3000]
                self._safe_print(f"    📄 TF docs: {len(tf_doc)} chars")
            else:
                self._safe_print(f"    📄 TF docs: no valid content")
        except Exception as e:
            self._safe_print(f"    ⚠️ TF doc fetch failed: {str(e)[:50]}")

        try:
            raw = get_doc_content(pytorch_api, "pytorch")
            if raw and len(raw) >= MIN_DOC_LENGTH:
                pytorch_doc = raw[:3000]
                self._safe_print(f"    📄 PT docs: {len(pytorch_doc)} chars")
            else:
                self._safe_print(f"    📄 PT docs: no valid content")
        except Exception as e:
            self._safe_print(f"    ⚠️ PT doc fetch failed: {str(e)[:50]}")

        return tf_doc, pytorch_doc

    # ==================== LLM interaction ====================

    def _build_llm_prompt(
        self,
        execution_result: Dict[str, Any],
        tf_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any],
        tf_doc: str = "",
        pytorch_doc: str = "",
    ) -> str:
        """Build the LLM prompt."""
        tf_api = execution_result.get("tf_api", "")
        pytorch_api = execution_result.get("pytorch_api", "")
        status = execution_result.get("status", "")
        tf_success = execution_result.get("tf_success", False)
        pytorch_success = execution_result.get("pytorch_success", False)
        results_match = execution_result.get("results_match", False)
        tf_error = execution_result.get("tf_error", "")
        pytorch_error = execution_result.get("pytorch_error", "")
        comparison_error = execution_result.get("comparison_error", "")

        # Simplify test cases to reduce token usage.
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

        # Build parameter example strings.
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

                # Doc section.
        doc_section = ""
        if tf_doc or pytorch_doc:
                        doc_section = "\n## Official API docs reference\n\n"
            if tf_doc:
                                doc_section += f"### TensorFlow {tf_api} docs\n```\n{tf_doc}\n```\n\n"
            if pytorch_doc:
                                doc_section += f"### PyTorch {pytorch_api} docs\n```\n{pytorch_doc}\n```\n\n"

                prompt = f"""Analyze the operator test case results in TensorFlow and PyTorch, then repair or mutate (fuzz) the test cases based on the results.

## Test info
- **TensorFlow API**: {tf_api}
- **PyTorch API**: {pytorch_api}
{doc_section}
## Execution results
- **Status**: {status}
- **TensorFlow success**: {tf_success}
- **PyTorch success**: {pytorch_success}
- **Results match**: {results_match}

## Error info
- **TensorFlow error**: {tf_error if tf_error else "none"}
- **PyTorch error**: {pytorch_error if pytorch_error else "none"}
- **Comparison error**: {comparison_error if comparison_error else "none"}

## Original test cases

### TensorFlow test case
```json
{json.dumps(simplified_tf, indent=2, ensure_ascii=False)}
```

### PyTorch test case
```json
{json.dumps(simplified_pt, indent=2, ensure_ascii=False)}
```

## Task requirements
Based on the info above (including official API docs), judge whether the results are **consistent**, **inconsistent**, or **execution error**, then do:

1. **If consistent**: **mutate (fuzz)** the case, e.g., change input shapes or parameter values (consider edge/extreme values).
2. **If execution error**: **repair** the case based on the error and docs (parameter names/count/types/ranges may differ across frameworks), or **skip** if you think the APIs are not equivalent.
3. **If inconsistent**: decide if the difference is tolerable (<= 1e-3). If tolerable, **mutate**. If APIs are not equivalent, **skip**. If neither, treat it as a test construction issue and **repair**.

## Output format
Output strictly the following JSON with no extra text, comments, or markdown:

{{
    "operation": "mutation",
    "reason": "Detailed reason (<= 150 words)",
    "tensorflow_test_case": {{
        "api": "{tf_api}",
{tf_param_str}
    }},
    "pytorch_test_case": {{
        "api": "{pytorch_api}",
{pt_param_str}
    }}
}}

**Important**:
1. operation must be one of "mutation", "repair", or "skip".
2. Tensor params must use {{"shape": [...], "dtype": "..."}}.
3. Scalar params use numeric values.
4. Inputs must be identical across frameworks (convert shapes like NHWC↔NCHW if needed), and params must be semantically aligned.
5. TF/PT test cases may differ in param names (x vs input), values, or counts, as long as outputs should match in theory.
6. If docs are missing or the operator is removed, set operation to "skip"; no repair needed.
7. When mutating, explore edge cases: empty tensors, single-element tensors, high-rank tensors, different dtypes, boundary values, etc.
8. Read official docs carefully and keep parameter names/types/ranges aligned.
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
        """Call LLM to repair or mutate test cases."""
        prompt = self._build_llm_prompt(
            execution_result, tf_test_case, pytorch_test_case, tf_doc, pytorch_doc
        )
        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a deep learning framework testing expert. Based on execution results, decide whether to repair or mutate the test case and return strict JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            raw_response = completion.choices[0].message.content.strip()
            time.sleep(1)

            # Parse JSON.
            try:
                llm_result = json.loads(raw_response)
                return llm_result
            except json.JSONDecodeError:
                self._safe_print("    ⚠️ LLM did not return valid JSON; attempting extraction...")
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    llm_result = json.loads(json_match.group())
                    return llm_result
                else:
                    return {
                        "operation": "skip",
                        "reason": "LLM returned an invalid format",
                        "tensorflow_test_case": tf_test_case,
                        "pytorch_test_case": pytorch_test_case,
                    }

        except Exception as e:
            self._safe_print(f"    ❌ LLM call failed: {e}")
            return {
                "operation": "skip",
                "reason": f"LLM call failed: {e}",
                "tensorflow_test_case": tf_test_case,
                "pytorch_test_case": pytorch_test_case,
            }

    # ==================== Core test loop ====================

    def llm_enhanced_test_operator(
        self,
        tf_api: str,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        num_test_cases: int = None,
        num_workers: int = DEFAULT_WORKERS,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Test a single operator pair with LLM enhancement.

        Args:
            tf_api: TensorFlow API name
            max_iterations: Max iterations per test case
            num_test_cases: Number of cases to test
            num_workers: LLM concurrent workers
        """
        self._safe_print(f"\n{'=' * 80}")
        self._safe_print(f"🎯 Start testing operator: {tf_api}")
        self._safe_print(f"🔄 Max iterations per case: {max_iterations}")
        self._safe_print(f"{'=' * 80}\n")

        stats = {"llm_generated_cases": 0, "successful_cases": 0}

        # Get corresponding PyTorch API.
        _, pytorch_api, mapping_method = self.convert_api_name(tf_api)
        if pytorch_api is None:
            self._safe_print(f"❌ {tf_api} has no PyTorch implementation")
            return [], stats

        self._safe_print(f"✅ TensorFlow API: {tf_api}")
        self._safe_print(f"✅ PyTorch API: {pytorch_api}")
        self._safe_print(f"✅ Mapping method: {mapping_method}")

        # Get test cases.
        api_data = self.test_cases_data.get(tf_api, {})
        test_cases = api_data.get("test_cases", [])

        if not test_cases:
            self._safe_print(f"⚠️ No test cases found for {tf_api}; using default")
            test_cases = [{"description": "default", "inputs": {"x": {"shape": [2, 3], "dtype": "float32"}}}]

        # Determine actual number of cases to test.
        if num_test_cases is None:
            num_test_cases = len(test_cases)
        else:
            num_test_cases = min(num_test_cases, len(test_cases))

        self._safe_print(f"📋 Testing {num_test_cases} cases (LLM workers={num_workers}, sequential execution)")

        # Prepare initial cases.
        initial_cases = []
        for case_idx in range(num_test_cases):
            tc = test_cases[case_idx]
            # Extract params from inputs to build a flat test case.
            if "inputs" in tc:
                flat_case = dict(tc["inputs"])
            else:
                flat_case = {k: v for k, v in tc.items() if k != "description"}
            flat_case["api"] = tf_api
            initial_cases.append((case_idx + 1, flat_case))

        all_results = []

        if num_workers <= 1:
            for case_number, initial_test_case in initial_cases:
                self._safe_print(f"\n📋 Case {case_number}/{num_test_cases}")
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
        self._safe_print("✅ All tests completed")
        self._safe_print(f"📊 Tested {num_test_cases} cases, total {len(all_results)} iterations")
        self._safe_print(f"📊 LLM-generated cases: {stats['llm_generated_cases']}")
        self._safe_print(f"📊 Cases where both frameworks succeeded: {stats['successful_cases']}")
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
        Iterate a single test case multiple rounds.

        Core loop: execute → LLM decision → repair/mutate/skip → execute → ...
        """
        case_results = []

        # Build initial TF/PT test cases.
        current_tf_test_case = copy.deepcopy(initial_test_case)
        current_tf_test_case["api"] = tf_api

        current_pt_test_case = copy.deepcopy(initial_test_case)
        current_pt_test_case["api"] = pytorch_api

        is_llm_generated = False

        # Pre-fetch API docs (once).
        self._safe_print("  📖 Pre-fetching API docs...")
        tf_doc, pytorch_doc = self._fetch_api_docs(tf_api, pytorch_api)

        # Iterative testing.
        for iteration in range(max_iterations):
            source_type = "LLM" if is_llm_generated else "file"
            self._safe_print(f"  🔄 Iteration {iteration + 1}/{max_iterations} ({source_type})", end="")

            # Execute test.
            try:
                execution_result = self._execute_test_case_sequential(
                    tf_api, pytorch_api, current_tf_test_case, current_pt_test_case
                )

                tf_status = "✓" if execution_result["tf_success"] else "✗"
                pt_status = "✓" if execution_result["pytorch_success"] else "✗"
                match_status = "✓" if execution_result["results_match"] else "✗"
                self._safe_print(f" | TF:{tf_status} PT:{pt_status} Match:{match_status}")

                if execution_result["tf_error"] and not execution_result["tf_success"]:
                    self._safe_print(f"    ❌ TF error: {str(execution_result['tf_error'])[:100]}...")
                if execution_result["pytorch_error"] and not execution_result["pytorch_success"]:
                    self._safe_print(f"    ❌ PT error: {str(execution_result['pytorch_error'])[:100]}...")
                if execution_result["comparison_error"]:
                    self._safe_print(f"    ⚠️ Compare: {str(execution_result['comparison_error'])[:100]}...")

                # Count LLM-generated cases.
                if is_llm_generated:
                    if execution_result["tf_success"] and execution_result["pytorch_success"]:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

            except Exception as e:
                self._safe_print(f" | ❌ Fatal error: {str(e)[:80]}...")
                execution_result = {
                    "status": "fatal_error",
                    "tf_success": False, "pytorch_success": False,
                    "results_match": False,
                    "tf_error": f"Fatal: {str(e)}", "pytorch_error": None,
                    "comparison_error": None,
                }

            # Save iteration result.
            iteration_result = {
                "iteration": iteration + 1,
                "tf_test_case": current_tf_test_case,
                "pytorch_test_case": current_pt_test_case,
                "execution_result": execution_result,
                "llm_operation": None,
                "is_llm_generated": is_llm_generated,
                "case_number": case_number,
            }

            # Call LLM.
            try:
                llm_result = self.call_llm_for_repair_or_mutation(
                    execution_result, current_tf_test_case, current_pt_test_case,
                    tf_doc, pytorch_doc,
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

            # Prepare next round.
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

        # If the last round generated a new case but didn't execute, run it.
        if case_results:
            last_op = case_results[-1].get("llm_operation", {}).get("operation", "skip")
            if last_op in ("mutation", "repair"):
            self._safe_print("  🔄 Executing final LLM case", end="")
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
                        "llm_operation": {"operation": "final_execution", "reason": "execute the last LLM-generated case"},
                        "case_number": case_number,
                        "is_llm_generated": True,
                    })
                except Exception as e:
                    self._safe_print(f"  ❌ Final case execution failed: {str(e)[:80]}...")
                    case_results.append({
                        "iteration": len(case_results) + 1,
                        "tf_test_case": current_tf_test_case,
                        "pytorch_test_case": current_pt_test_case,
                        "execution_result": {
                            "status": "fatal_error", "tf_success": False,
                            "pytorch_success": False, "results_match": False,
                            "error": str(e),
                        },
                        "llm_operation": {"operation": "final_execution", "reason": "final case execution failed"},
                        "case_number": case_number,
                        "is_llm_generated": True,
                    })

        self._safe_print(f"  ✅ Case {case_number} completed, {len(case_results)} iterations")
        return case_results

    def _convert_llm_test_cases(
        self,
        tf_test_case: Dict[str, Any],
        pytorch_test_case: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Convert LLM test cases into executable format.
        Ensure both frameworks share the same tensor data.
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

    # ==================== Result saving ====================

    def save_results(
        self, tf_api: str, results: List[Dict[str, Any]], stats: Dict[str, int] = None
    ):
        """Save test results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = tf_api.replace(".", "_")
        filename = f"llm_enhanced_{safe_name}_{timestamp}.json"
        filepath = os.path.join(self.result_dir, filename)

        output_results = []
        for result in results:
            out = copy.deepcopy(result)
            # Simplify numpy arrays.
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

        self._safe_print(f"💾 Results saved to: {filepath}")

    def get_all_testable_apis(self) -> List[str]:
        """Get all testable TF APIs (with test cases and PT mapping)."""
        testable = []
        for tf_api in sorted(self.test_cases_data.keys()):
            pt_api = self.api_mapping.get(tf_api, "无对应实现")
            if pt_api and pt_api != "无对应实现":
                testable.append(tf_api)
        return testable

    def close(self):
        """Clean up resources."""
        pass


# ==================== Main function ====================

def main():
    parser = argparse.ArgumentParser(
        description="LLM-based TensorFlow ↔ PyTorch operator differential testing framework"
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
        help="End operator index (inclusive; default all)"
    )
    parser.add_argument(
        "--operators", "-o", nargs="*",
        help="Operator names to test (TF format, e.g., tf.nn.relu)"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"Concurrent workers (default {DEFAULT_WORKERS})"
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
        help="Test cases JSON file path"
    )
    parser.add_argument(
        "--mapping-file", default=DEFAULT_MAPPING_FILE,
        help="TF→PT mapping CSV file path"
    )

    args = parser.parse_args()
    num_workers = max(1, args.workers)

    print("=" * 80)
    print("LLM-based TensorFlow ↔ PyTorch operator differential testing framework")
    print("=" * 80)
    print(f"📌 Iterations per operator: {args.max_iterations}")
    print(f"📌 Test cases per operator: {args.num_cases}")
    print(f"📌 LLM workers: {num_workers}")
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
        # Get all testable APIs.
        all_testable = comparator.get_all_testable_apis()
        print(f"\n🔍 Total testable TF APIs: {len(all_testable)}")

        if args.operators:
            operator_names = args.operators
            print(f"📋 Specified operators: {len(operator_names)}")
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

        # Batch log.
        batch_log_file = os.path.join(
            comparator.result_dir,
            f"batch_test_log_{start_datetime.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        log_file = open(batch_log_file, 'w', encoding='utf-8')
        log_file.write("=" * 80 + "\n")
        log_file.write("TF→PT differential test batch log\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("Test config:\n")
        log_file.write(f"  - Iterations: {args.max_iterations}\n")
        log_file.write(f"  - Cases: {args.num_cases}\n")
        log_file.write(f"  - Workers: {num_workers}\n")
        log_file.write(f"  - Operators: {len(operator_names)}\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.flush()

        for idx, tf_api in enumerate(operator_names, 1):
            print("\n" + "🔷" * 40)
            print(f"🎯 [{idx}/{len(operator_names)}] Start testing operator: {tf_api}")
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
                    print(f"\n✅ {tf_api} testing completed")
                    print(f"   - Total iterations: {len(results)}")
                    print(f"   - LLM-generated cases: {stats.get('llm_generated_cases', 0)}")
                    print(f"   - Successful cases: {stats.get('successful_cases', 0)}")

                    log_file.write(f"[{idx}/{len(operator_names)}] {tf_api}\n")
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
                        "operator": tf_api, "total_iterations": 0,
                        "llm_generated_cases": 0, "successful_cases": 0,
                        "status": "no_results",
                    })
                    log_file.write(f"[{idx}/{len(operator_names)}] {tf_api}\n")
                    log_file.write("  Status: ⚠️ no results\n\n")
                    log_file.flush()

            except Exception as e:
                print(f"\n❌ {tf_api} testing failed: {e}")
                all_operators_summary.append({
                    "operator": tf_api, "total_iterations": 0,
                    "llm_generated_cases": 0, "successful_cases": 0,
                    "status": "failed", "error": str(e),
                })
                log_file.write(f"[{idx}/{len(operator_names)}] {tf_api}\n")
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
        print(f"   - LLM-generated cases: {total_llm_cases}")
        print(f"   - Successful cases: {total_successful}")
        if total_llm_cases > 0:
            print(f"   - Success rate: {total_successful / total_llm_cases * 100:.2f}%")
        print(f"   - Total iterations: {total_iterations}")
        print(f"\n⏱️ Runtime: {hours}h {minutes}m {seconds}s")

        # Write log.
        log_file.write("=" * 80 + "\nOverall summary\n" + "=" * 80 + "\n")
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
        print("\n✅ Batch test completed")


if __name__ == "__main__":
    main()
