#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4: LLM-based differential testing framework for TensorFlow vs Paddle operators

Purpose:
- Load TF test cases and TF→Paddle mappings from JSON
- Run TF and Paddle for each equivalent operator pair and compare results
- Use the LLM to repair, mutate (fuzz), or skip test cases
- Support concurrent testing (execution is serialized with a lock to avoid BLAS/MKL conflicts)
- Save detailed results and batch logs

Usage:
    conda activate tf_env
    python tf_pd_test_1/llm_enhanced_compare.py \
        [--max-iterations 3] [--num-cases 5] [--workers 6] \
        [--start 1] [--end N] [--operators tf.math.abs tf.concat]
"""

import os

# ==================== Environment variables (set before importing TF/Paddle) ====================
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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from component.doc.doc_crawler_factory import get_doc_content

DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_NUM_CASES = 5
DEFAULT_WORKERS = 6

DATA_DIR = os.path.join(ROOT_DIR, "tf_pd_test_1", "data")
DEFAULT_TEST_CASES_FILE = os.path.join(DATA_DIR, "tf_test_cases.json")
DEFAULT_MAPPING_FILE = os.path.join(DATA_DIR, "tf_pd_mapping_validated.csv")


class LLMEnhancedComparator:
    """LLM-based differential testing framework for TensorFlow and Paddle."""

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

        self.problematic_apis = {
            "tf.nn.conv3d": "Known to be unstable on some CPU/MKL environments",
        }

        api_key = self._load_api_key(key_path)
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        self.test_cases_data = self._load_test_cases(test_cases_file)
        self._safe_print(f"📋 Loaded test cases for {len(self.test_cases_data)} TF APIs")

        self.api_mapping = self._load_mapping(mapping_file)
        has_impl = sum(1 for value in self.api_mapping.values() if value != "无对应实现")
        self._safe_print(f"📋 Loaded {len(self.api_mapping)} mappings ({has_impl} with equivalents)")

        self.result_dir = os.path.join(ROOT_DIR, "tf_pd_test_1", "tf_pd_log_1")
        os.makedirs(self.result_dir, exist_ok=True)
        self._safe_print(f"📁 Result directory: {self.result_dir}")

        self.random_seed = 42
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        paddle.seed(self.random_seed)

    def _safe_print(self, msg: str, end: str = "\n"):
        with self.print_lock:
            print(msg, end=end, flush=True)

    def _load_api_key(self, key_path: str = DEFAULT_KEY_PATH) -> str:
        key_file = os.path.join(ROOT_DIR, key_path) if not os.path.isabs(key_path) else key_path
        if os.path.exists(key_file):
            with open(key_file, "r", encoding="utf-8") as file:
                api_key = file.read().strip()
            if api_key:
                return api_key

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            return api_key

        self._safe_print("❌ API key not found")
        return ""

    def _load_test_cases(self, filepath: str) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            self._safe_print(f"⚠️ Test case file not found: {filepath}")
            return {}
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data.get("test_cases", {})

    def _load_mapping(self, filepath: str) -> Dict[str, str]:
        if not os.path.exists(filepath):
            self._safe_print(f"⚠️ Mapping file not found: {filepath}")
            return {}
        mapping: Dict[str, str] = {}
        with open(filepath, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                tf_api = row.get("tensorflow-api", "").strip()
                paddle_api = row.get("paddle-api", "").strip()
                if tf_api and paddle_api:
                    mapping[tf_api] = paddle_api
        return mapping

    def is_class_based_api(self, api_name: str) -> bool:
        parts = api_name.split(".")
        if len(parts) >= 2:
            last_part = parts[-1]
            return bool(last_part and last_part[0].isupper())
        return False

    def get_operator_function(self, api_name: str, framework: str = "tf"):
        try:
            module = tf if framework == "tf" else paddle if framework == "paddle" else None
            if module is None:
                return None
            obj = module
            for part in api_name.split(".")[1:]:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return None

    def convert_api_name(self, tf_api: str) -> Tuple[Optional[str], Optional[str], str]:
        if tf_api in self.api_mapping:
            paddle_api = self.api_mapping[tf_api]
            if paddle_api and paddle_api != "无对应实现":
                return tf_api, paddle_api, "mapping_file"
            return tf_api, None, "no_equivalent"
        return tf_api, None, "not_in_mapping"

    def generate_numpy_data(self, data: Any) -> np.ndarray:
        if isinstance(data, dict):
            if "shape" in data:
                shape = data["shape"]
                dtype_str = str(data.get("dtype", "float32"))
                for prefix in ["paddle.", "torch.", "tf.", "np.", "numpy."]:
                    if dtype_str.startswith(prefix):
                        dtype_str = dtype_str[len(prefix):]

                dtype_map = {
                    "float32": np.float32,
                    "float64": np.float64,
                    "float16": np.float16,
                    "float": np.float32,
                    "int32": np.int32,
                    "int64": np.int64,
                    "int16": np.int16,
                    "int8": np.int8,
                    "uint8": np.uint8,
                    "bool": np.bool_,
                    "bool_": np.bool_,
                    "complex64": np.complex64,
                    "complex128": np.complex128,
                    "bfloat16": np.float32,
                    "string": np.str_,
                }
                np_dtype = dtype_map.get(dtype_str, np.float32)

                if isinstance(shape, list) and any(dim == 0 for dim in shape):
                    return np.empty(shape, dtype=np_dtype)

                if np_dtype == np.str_:
                    size = int(np.prod(shape)) if isinstance(shape, list) and shape else 1
                    content = np.array(["a"] * size, dtype=np.str_)
                    return content.reshape(shape) if isinstance(shape, list) and shape else content[0]

                data_range = data.get("range", None)
                if np_dtype == np.bool_:
                    return np.random.choice([True, False], size=shape).astype(np.bool_)
                if np.issubdtype(np_dtype, np.integer):
                    low = int(data_range[0]) if data_range else 0
                    high = int(data_range[1]) if data_range else 10
                    high = high if high > low else low + 1
                    return np.random.randint(low, high, size=shape).astype(np_dtype)
                if np.issubdtype(np_dtype, np.complexfloating):
                    real = np.random.randn(*shape).astype(np.float32)
                    imag = np.random.randn(*shape).astype(np.float32)
                    return (real + 1j * imag).astype(np_dtype)
                if data_range:
                    low, high = float(data_range[0]), float(data_range[1])
                    return np.random.uniform(low, high, size=shape).astype(np_dtype)
                return np.random.randn(*shape).astype(np_dtype)

            return np.array(list(data.values()))

        if isinstance(data, (int, float, bool, str)):
            return np.array(data)
        if isinstance(data, list):
            return np.array(data)
        return np.array(data)

    def convert_to_tensor_tf(self, data: Any, numpy_data: np.ndarray = None):
        if numpy_data is not None:
            return tf.constant(numpy_data)
        if isinstance(data, dict):
            return tf.constant(self.generate_numpy_data(data))
        if isinstance(data, list):
            return tf.constant(np.array(data))
        return tf.constant(data)

    def convert_to_tensor_paddle(self, data: Any, numpy_data: np.ndarray = None):
        if numpy_data is not None:
            return paddle.to_tensor(numpy_data.copy())
        if isinstance(data, dict):
            return paddle.to_tensor(self.generate_numpy_data(data).copy())
        if isinstance(data, list):
            return paddle.to_tensor(np.array(data))
        return paddle.to_tensor(data)

    def should_skip_param(self, key: str, framework: str) -> bool:
        if key in ["description", "api"]:
            return True
        if framework == "paddle" and key in ["place", "name", "stop_gradient"]:
            return True
        if framework == "tf" and key in ["name"]:
            return True
        return False

    def prepare_arguments(self, test_case: Dict[str, Any], framework: str) -> Tuple[List[Any], Dict[str, Any]]:
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        positional_params = ["x", "input", "condition", "y", "other", "a", "b"]

        varargs_key = next((key for key in test_case if key.startswith("*")), None)
        if varargs_key:
            varargs_data = test_case[varargs_key]
            if isinstance(varargs_data, list):
                for item in varargs_data:
                    if isinstance(item, np.ndarray):
                        args.append(self.convert_to_tensor_paddle(None, item) if framework == "paddle" else self.convert_to_tensor_tf(None, item))
                    elif isinstance(item, dict) and "shape" in item:
                        np_data = self.generate_numpy_data(item)
                        args.append(self.convert_to_tensor_paddle(None, np_data) if framework == "paddle" else self.convert_to_tensor_tf(None, np_data))
                    else:
                        args.append(item)
            return args, kwargs

        for param_name in positional_params:
            if param_name not in test_case:
                continue
            value = test_case[param_name]
            if isinstance(value, np.ndarray):
                args.append(self.convert_to_tensor_paddle(None, value) if framework == "paddle" else self.convert_to_tensor_tf(None, value))
            elif isinstance(value, dict) and "shape" in value:
                np_data = self.generate_numpy_data(value)
                args.append(self.convert_to_tensor_paddle(None, np_data) if framework == "paddle" else self.convert_to_tensor_tf(None, np_data))
            else:
                args.append(value)

        for key, value in test_case.items():
            if key in positional_params or key.startswith("*") or self.should_skip_param(key, framework):
                continue
            if isinstance(value, np.ndarray):
                kwargs[key] = self.convert_to_tensor_paddle(None, value) if framework == "paddle" else self.convert_to_tensor_tf(None, value)
            elif isinstance(value, dict) and "shape" in value:
                np_data = self.generate_numpy_data(value)
                kwargs[key] = self.convert_to_tensor_paddle(None, np_data) if framework == "paddle" else self.convert_to_tensor_tf(None, np_data)
            else:
                kwargs[key] = value

        return args, kwargs

    def compare_tensors(self, tf_result, paddle_result, tolerance: float = 1e-5) -> Tuple[bool, str]:
        try:
            tf_np = tf_result.numpy() if isinstance(tf_result, tf.Tensor) else tf_result if isinstance(tf_result, np.ndarray) else np.array(tf_result)
            paddle_np = paddle_result.numpy() if isinstance(paddle_result, paddle.Tensor) else paddle_result if isinstance(paddle_result, np.ndarray) else np.array(paddle_result)

            if tf_np.shape != paddle_np.shape:
                return False, f"Shape mismatch: TF={tf_np.shape} vs Paddle={paddle_np.shape}"

            if tf_np.dtype == np.bool_ or paddle_np.dtype == np.bool_:
                match = np.array_equal(tf_np, paddle_np)
                return (True, "Boolean results match") if match else (False, f"Boolean mismatch, differing elements: {int(np.sum(tf_np != paddle_np))}")

            if np.issubdtype(tf_np.dtype, np.str_) or np.issubdtype(paddle_np.dtype, np.str_):
                return (True, "String results match") if np.array_equal(tf_np, paddle_np) else (False, "String results differ")

            if np.allclose(tf_np, paddle_np, atol=tolerance, rtol=tolerance, equal_nan=True):
                return True, "Results match (within tolerance)"

            max_diff = np.max(np.abs(tf_np.astype(np.float64) - paddle_np.astype(np.float64)))
            return False, f"Results differ, max diff: {max_diff:.8f}"
        except Exception as error:
            return False, f"Comparison error: {str(error)}"

    def execute_test_case(
        self,
        tf_api: str,
        paddle_api: str,
        tf_test_case: Dict[str, Any],
        paddle_test_case: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "tf_api": tf_api,
            "paddle_api": paddle_api,
            "tf_success": False,
            "paddle_success": False,
            "results_match": False,
            "tf_error": None,
            "paddle_error": None,
            "comparison_error": None,
            "tf_shape": None,
            "paddle_shape": None,
            "tf_dtype": None,
            "paddle_dtype": None,
            "status": "unknown",
        }

        if paddle_test_case is None:
            paddle_test_case = tf_test_case

        tf_test_case, paddle_test_case = self._materialize_shared_tensors(
            tf_api, paddle_api, tf_test_case, paddle_test_case
        )

        is_class_tf = self.is_class_based_api(tf_api)
        is_class_paddle = self.is_class_based_api(paddle_api)

        tf_result = None
        try:
            tf_func = self.get_operator_function(tf_api, "tf")
            if tf_func is None:
                raise AttributeError(f"TF API not found: {tf_api}")

            if is_class_tf:
                init_kwargs = {
                    key: value
                    for key, value in tf_test_case.items()
                    if key not in ["api", "input", "x"] and not isinstance(value, np.ndarray) and not (isinstance(value, dict) and "shape" in value)
                }
                layer = tf_func(**init_kwargs)
                input_data = tf_test_case.get("input") or tf_test_case.get("x")
                if isinstance(input_data, np.ndarray):
                    tf_input = tf.constant(input_data)
                elif isinstance(input_data, dict) and "shape" in input_data:
                    tf_input = tf.constant(self.generate_numpy_data(input_data))
                elif input_data is None:
                    tf_input = tf.constant(np.random.randn(2, 3).astype(np.float32))
                else:
                    tf_input = tf.constant(input_data)
                tf_result = layer(tf_input)
            else:
                tf_args, tf_kwargs = self.prepare_arguments(tf_test_case, "tf")
                tf_result = tf_func(*tf_args, **tf_kwargs)

            result["tf_success"] = True
            if hasattr(tf_result, "shape"):
                result["tf_shape"] = list(tf_result.shape)
            if hasattr(tf_result, "dtype"):
                result["tf_dtype"] = str(tf_result.dtype)
        except Exception as error:
            result["tf_error"] = f"{type(error).__name__}: {str(error)}"

        paddle_result = None
        try:
            paddle_func = self.get_operator_function(paddle_api, "paddle")
            if paddle_func is None:
                raise AttributeError(f"Paddle API not found: {paddle_api}")

            if is_class_paddle:
                init_kwargs = {
                    key: value
                    for key, value in paddle_test_case.items()
                    if key not in ["api", "input", "x"] and not isinstance(value, np.ndarray) and not (isinstance(value, dict) and "shape" in value)
                }
                module = paddle_func(**init_kwargs)
                input_data = paddle_test_case.get("input") or paddle_test_case.get("x")
                if isinstance(input_data, np.ndarray):
                    paddle_input = paddle.to_tensor(input_data.copy())
                elif isinstance(input_data, dict) and "shape" in input_data:
                    paddle_input = paddle.to_tensor(self.generate_numpy_data(input_data).copy())
                elif input_data is None:
                    paddle_input = paddle.randn([2, 3], dtype="float32")
                else:
                    paddle_input = paddle.to_tensor(input_data)
                paddle_result = module(paddle_input)
            else:
                paddle_args, paddle_kwargs = self.prepare_arguments(paddle_test_case, "paddle")
                paddle_result = paddle_func(*paddle_args, **paddle_kwargs)

            result["paddle_success"] = True
            if hasattr(paddle_result, "shape"):
                result["paddle_shape"] = list(paddle_result.shape)
            if hasattr(paddle_result, "dtype"):
                result["paddle_dtype"] = str(paddle_result.dtype)
        except Exception as error:
            result["paddle_error"] = f"{type(error).__name__}: {str(error)}"

        if result["tf_success"] and result["paddle_success"]:
            match, detail = self.compare_tensors(tf_result, paddle_result)
            result["results_match"] = match
            result["comparison_error"] = None if match else detail
            result["status"] = "consistent" if match else "inconsistent"
        elif result["tf_success"] and not result["paddle_success"]:
            result["status"] = "paddle_error"
        elif not result["tf_success"] and result["paddle_success"]:
            result["status"] = "tf_error"
        else:
            result["status"] = "both_error"

        return result

    def _execute_test_case_sequential(self, tf_api, paddle_api, tf_test_case, paddle_test_case=None) -> Dict[str, Any]:
        with self.execution_lock:
            return self.execute_test_case(tf_api, paddle_api, tf_test_case, paddle_test_case)

    def _default_input_desc_for_class(self, api_name: str) -> Dict[str, Any]:
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
        paddle_api: str,
        tf_test_case: Dict[str, Any],
        paddle_test_case: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        tf_case = copy.deepcopy(tf_test_case)
        paddle_case = copy.deepcopy(paddle_test_case)

        is_class_tf = self.is_class_based_api(tf_api)
        is_class_paddle = self.is_class_based_api(paddle_api)
        if (is_class_tf or is_class_paddle) and not ("input" in tf_case or "x" in tf_case or "input" in paddle_case or "x" in paddle_case):
            default_desc = self._default_input_desc_for_class(tf_api or paddle_api)
            tf_case.setdefault("input", default_desc)
            paddle_case.setdefault("input", default_desc)

        shared_tensors: Dict[str, np.ndarray] = {}
        all_keys = set(tf_case.keys()) | set(paddle_case.keys())

        for key in all_keys:
            if key == "api":
                continue
            tf_value = tf_case.get(key)
            paddle_value = paddle_case.get(key)

            if isinstance(tf_value, np.ndarray):
                shared_tensors[key] = tf_value
                continue
            if isinstance(paddle_value, np.ndarray):
                shared_tensors[key] = paddle_value
                continue

            tensor_desc = None
            if isinstance(tf_value, dict) and "shape" in tf_value:
                tensor_desc = tf_value
            elif isinstance(paddle_value, dict) and "shape" in paddle_value:
                tensor_desc = paddle_value
            if tensor_desc:
                shared_tensors[key] = self.generate_numpy_data(tensor_desc)

        def apply_shared(case: Dict[str, Any]) -> Dict[str, Any]:
            converted = {}
            for key, value in case.items():
                converted[key] = shared_tensors[key].copy() if key in shared_tensors else value
            return converted

        return apply_shared(tf_case), apply_shared(paddle_case)

    def _fetch_api_docs(self, tf_api: str, paddle_api: str) -> Tuple[str, str]:
        min_doc_length = 300
        tf_doc = ""
        paddle_doc = ""

        try:
            raw = get_doc_content(tf_api, "tensorflow")
            if raw and len(raw) >= min_doc_length:
                tf_doc = raw[:3000]
                self._safe_print(f"    📄 TF docs: {len(tf_doc)} chars")
            else:
                self._safe_print("    📄 TF docs: no valid content")
        except Exception as error:
            self._safe_print(f"    ⚠️ TF docs fetch failed: {str(error)[:50]}")

        try:
            raw = get_doc_content(paddle_api, "paddle")
            if raw and len(raw) >= min_doc_length:
                paddle_doc = raw[:3000]
                self._safe_print(f"    📄 Paddle docs: {len(paddle_doc)} chars")
            else:
                self._safe_print("    📄 Paddle docs: no valid content")
        except Exception as error:
            self._safe_print(f"    ⚠️ Paddle docs fetch failed: {str(error)[:50]}")

        return tf_doc, paddle_doc

    def _build_llm_prompt(
        self,
        execution_result: Dict[str, Any],
        tf_test_case: Dict[str, Any],
        paddle_test_case: Dict[str, Any],
        tf_doc: str = "",
        paddle_doc: str = "",
    ) -> str:
        tf_api = execution_result.get("tf_api", "")
        paddle_api = execution_result.get("paddle_api", "")
        status = execution_result.get("status", "")
        tf_success = execution_result.get("tf_success", False)
        paddle_success = execution_result.get("paddle_success", False)
        results_match = execution_result.get("results_match", False)
        tf_error = execution_result.get("tf_error", "")
        paddle_error = execution_result.get("paddle_error", "")
        comparison_error = execution_result.get("comparison_error", "")

        simplified_tf = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)} if isinstance(value, np.ndarray) else value
            for key, value in tf_test_case.items()
        }
        simplified_paddle = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)} if isinstance(value, np.ndarray) else value
            for key, value in paddle_test_case.items()
        }

        tf_param_examples = [f'    "{k}": {json.dumps(v, ensure_ascii=False)}' for k, v in simplified_tf.items() if k != "api"]
        paddle_param_examples = [f'    "{k}": {json.dumps(v, ensure_ascii=False)}' for k, v in simplified_paddle.items() if k != "api"]
        tf_param_str = ",\n".join(tf_param_examples) if tf_param_examples else '    "x": {"shape": [2, 3], "dtype": "float32"}'
        paddle_param_str = ",\n".join(paddle_param_examples) if paddle_param_examples else '    "x": {"shape": [2, 3], "dtype": "float32"}'

                doc_section = ""
                if tf_doc or paddle_doc:
                        doc_section = "\n## Official API docs\n\n"
                        if tf_doc:
                                doc_section += f"### TensorFlow {tf_api} docs\n```\n{tf_doc}\n```\n\n"
                        if paddle_doc:
                                doc_section += f"### PaddlePaddle {paddle_api} docs\n```\n{paddle_doc}\n```\n\n"

                return f"""Please analyze the execution results of the following operator test case in TensorFlow and PaddlePaddle, and then repair or mutate (fuzz) the test case based on the results.

## Test Info
- **TensorFlow API**: {tf_api}
- **PaddlePaddle API**: {paddle_api}
{doc_section}
## Execution Results
- **Status**: {status}
- **TensorFlow success**: {tf_success}
- **PaddlePaddle success**: {paddle_success}
- **Results match**: {results_match}

## Errors
- **TensorFlow error**: {tf_error if tf_error else "none"}
- **PaddlePaddle error**: {paddle_error if paddle_error else "none"}
- **Comparison error**: {comparison_error if comparison_error else "none"}

## Original Test Cases

### TensorFlow Test Case
```json
{json.dumps(simplified_tf, indent=2, ensure_ascii=False)}
```

### PaddlePaddle Test Case
```json
{json.dumps(simplified_paddle, indent=2, ensure_ascii=False)}
```

## Task
Based on the above information (including the official API docs), determine whether the comparison is **consistent**, **inconsistent**, or **execution error**, and then do the following:

1. **If consistent**: **Mutate (fuzz)** the case, e.g., change input tensor shapes or parameter values (consider edge/boundary values).
2. **If execution error**: **Repair** the case based on the error and docs (change parameter names/count/types/ranges; frameworks may differ), or **skip** (if the operator doesn't exist or the cross-framework APIs are not truly equivalent).
3. **If inconsistent**: decide whether it is a tolerable precision error (≤ 1e-3). (1) If tolerable, **mutate**; (2) if docs suggest the APIs are not truly equivalent, **skip**; (3) otherwise, treat it as test-case construction and **repair** according to docs.

## Output Format
Return strictly the following JSON format, with no extra text, comments, or markdown:

{{
    "operation": "mutation",
    "reason": "Detailed reason for the operation (<= 150 words)",
    "tensorflow_test_case": {{
        "api": "{tf_api}",
{tf_param_str}
    }},
    "paddle_test_case": {{
        "api": "{paddle_api}",
{paddle_param_str}
    }}
}}

**Important**:
1. operation must be one of "mutation", "repair", or "skip".
2. Tensor parameters must use {{"shape": [...], "dtype": "..."}} format.
3. Scalar parameters should use numeric values directly.
4. Inputs for both frameworks must be identical (convert shapes if needed, e.g., NHWC vs NCHW), and parameters must be semantically aligned.
5. TensorFlow and Paddle test cases may differ in parameter names (e.g., x vs input), values, or counts as long as outputs are theoretically the same.
6. If the operator has no official docs or is removed in the current version, set operation to "skip" and do not attempt repair.
7. When mutating, explore edge cases: empty tensors, single-element tensors, high-rank tensors, different dtypes, boundary values, etc.
8. Read the official API docs carefully and ensure parameter names/types/ranges are consistent with the docs.
"""

    def call_llm_for_repair_or_mutation(
        self,
        execution_result: Dict[str, Any],
        tf_test_case: Dict[str, Any],
        paddle_test_case: Dict[str, Any],
        tf_doc: str = "",
        paddle_doc: str = "",
    ) -> Dict[str, Any]:
        prompt = self._build_llm_prompt(execution_result, tf_test_case, paddle_test_case, tf_doc, paddle_doc)
        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a deep learning framework testing expert. Only return strict JSON.",
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
                self._safe_print("    ⚠️ LLM output is not valid JSON, attempting extraction...")
                json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return {
                    "operation": "skip",
                    "reason": "LLM returned invalid format",
                    "tensorflow_test_case": tf_test_case,
                    "paddle_test_case": paddle_test_case,
                }
        except Exception as error:
            self._safe_print(f"    ❌ LLM call failed: {error}")
            return {
                "operation": "skip",
                "reason": f"LLM call failed: {error}",
                "tensorflow_test_case": tf_test_case,
                "paddle_test_case": paddle_test_case,
            }

    def llm_enhanced_test_operator(
        self,
        tf_api: str,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        num_test_cases: int = None,
        num_workers: int = DEFAULT_WORKERS,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        self._safe_print(f"\n{'=' * 80}")
        self._safe_print(f"🎯 Start testing operator: {tf_api}")
        self._safe_print(f"🔄 Max iterations per case: {max_iterations}")
        self._safe_print(f"{'=' * 80}\n")

        stats = {"llm_generated_cases": 0, "successful_cases": 0}

        if tf_api in self.problematic_apis:
            self._safe_print(f"⏭️ Skip {tf_api}: {self.problematic_apis[tf_api]}")
            return [], stats

        _, paddle_api, mapping_method = self.convert_api_name(tf_api)
        if paddle_api is None:
            self._safe_print(f"❌ {tf_api} has no Paddle equivalent")
            return [], stats

        self._safe_print(f"✅ TensorFlow API: {tf_api}")
        self._safe_print(f"✅ Paddle API: {paddle_api}")
        self._safe_print(f"✅ Mapping method: {mapping_method}")

        api_data = self.test_cases_data.get(tf_api, {})
        test_cases = api_data.get("test_cases", [])
        if not test_cases:
            self._safe_print(f"⚠️ No test cases for {tf_api}; using default case")
            test_cases = [{"description": "default", "inputs": {"x": {"shape": [2, 3], "dtype": "float32"}}}]

        num_test_cases = len(test_cases) if num_test_cases is None else min(num_test_cases, len(test_cases))
        self._safe_print(f"📋 Testing {num_test_cases} cases (LLM workers={num_workers}, sequential execution)")

        initial_cases = []
        for case_idx in range(num_test_cases):
            case_data = test_cases[case_idx]
            flat_case = dict(case_data["inputs"]) if "inputs" in case_data else {k: v for k, v in case_data.items() if k != "description"}
            flat_case["api"] = tf_api
            initial_cases.append((case_idx + 1, flat_case))

        all_results: List[Dict[str, Any]] = []

        if num_workers <= 1:
            for case_number, initial_test_case in initial_cases:
                self._safe_print(f"\n📋 Case {case_number}/{num_test_cases}")
                case_results = self._test_single_case_with_iterations(
                    tf_api, paddle_api, initial_test_case, max_iterations, case_number, stats
                )
                all_results.extend(case_results)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_case = {
                    executor.submit(
                        self._test_single_case_with_iterations,
                        tf_api,
                        paddle_api,
                        initial_test_case,
                        max_iterations,
                        case_number,
                        stats,
                    ): case_number
                    for case_number, initial_test_case in initial_cases
                }
                for future in as_completed(future_to_case):
                    all_results.extend(future.result())

        all_results.sort(key=lambda item: (item.get("case_number", 0), item.get("iteration", 0)))

        self._safe_print(f"\n{'=' * 80}")
        self._safe_print("✅ All tests complete")
        self._safe_print(f"📊 Tested {num_test_cases} cases, total {len(all_results)} iterations")
        self._safe_print(f"📊 LLM-generated cases: {stats['llm_generated_cases']}")
        self._safe_print(f"📊 Cases where both frameworks succeeded: {stats['successful_cases']}")
        self._safe_print(f"{'=' * 80}\n")

        return all_results, stats

    def _test_single_case_with_iterations(
        self,
        tf_api: str,
        paddle_api: str,
        initial_test_case: Dict[str, Any],
        max_iterations: int,
        case_number: int,
        stats: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        case_results: List[Dict[str, Any]] = []

        current_tf_test_case = copy.deepcopy(initial_test_case)
        current_tf_test_case["api"] = tf_api
        current_paddle_test_case = copy.deepcopy(initial_test_case)
        current_paddle_test_case["api"] = paddle_api
        is_llm_generated = False

        self._safe_print("  📖 Pre-fetching API docs...")
        tf_doc, paddle_doc = self._fetch_api_docs(tf_api, paddle_api)

        for iteration in range(max_iterations):
            source_type = "LLM" if is_llm_generated else "file"
            self._safe_print(f"  🔄 Iteration {iteration + 1}/{max_iterations} ({source_type})", end="")

            try:
                execution_result = self._execute_test_case_sequential(
                    tf_api, paddle_api, current_tf_test_case, current_paddle_test_case
                )
                tf_status = "✓" if execution_result["tf_success"] else "✗"
                paddle_status = "✓" if execution_result["paddle_success"] else "✗"
                match_status = "✓" if execution_result["results_match"] else "✗"
                self._safe_print(f" | TF:{tf_status} Paddle:{paddle_status} Match:{match_status}")

                if execution_result["tf_error"] and not execution_result["tf_success"]:
                    self._safe_print(f"    ❌ TF error: {str(execution_result['tf_error'])[:120]}...")
                if execution_result["paddle_error"] and not execution_result["paddle_success"]:
                    self._safe_print(f"    ❌ Paddle error: {str(execution_result['paddle_error'])[:120]}...")
                if execution_result["comparison_error"]:
                    self._safe_print(f"    ⚠️ Compare: {str(execution_result['comparison_error'])[:120]}...")

                if is_llm_generated and execution_result["tf_success"] and execution_result["paddle_success"]:
                    with self.stats_lock:
                        stats["successful_cases"] += 1
            except Exception as error:
                self._safe_print(f" | ❌ Fatal error: {str(error)[:80]}...")
                execution_result = {
                    "status": "fatal_error",
                    "tf_success": False,
                    "paddle_success": False,
                    "results_match": False,
                    "tf_error": f"Fatal: {str(error)}",
                    "paddle_error": None,
                    "comparison_error": None,
                    "traceback": traceback.format_exc(),
                }

            iteration_result = {
                "iteration": iteration + 1,
                "tf_test_case": current_tf_test_case,
                "paddle_test_case": current_paddle_test_case,
                "execution_result": execution_result,
                "llm_operation": None,
                "is_llm_generated": is_llm_generated,
                "case_number": case_number,
            }

            try:
                llm_result = self.call_llm_for_repair_or_mutation(
                    execution_result,
                    current_tf_test_case,
                    current_paddle_test_case,
                    tf_doc,
                    paddle_doc,
                )
            except Exception as error:
                self._safe_print(f"    ❌ LLM call failed: {str(error)[:80]}...")
                llm_result = {"operation": "skip", "reason": f"LLM call failed: {str(error)}"}
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
                next_tf_case = llm_result.get("tensorflow_test_case", current_tf_test_case)
                next_paddle_case = llm_result.get("paddle_test_case", current_paddle_test_case)
                with self.stats_lock:
                    stats["llm_generated_cases"] += 1
                is_llm_generated = True
            else:
                next_tf_case = current_tf_test_case
                next_paddle_case = current_paddle_test_case

            current_tf_test_case, current_paddle_test_case = self._convert_llm_test_cases(
                next_tf_case, next_paddle_case
            )

        if case_results:
            last_op = case_results[-1].get("llm_operation", {}).get("operation", "skip")
            if last_op in ("mutation", "repair"):
                self._safe_print("  🔄 Running final LLM case", end="")
                try:
                    execution_result = self._execute_test_case_sequential(
                        tf_api, paddle_api, current_tf_test_case, current_paddle_test_case
                    )
                    tf_status = "✓" if execution_result["tf_success"] else "✗"
                    paddle_status = "✓" if execution_result["paddle_success"] else "✗"
                    match_status = "✓" if execution_result["results_match"] else "✗"
                    self._safe_print(f" | TF:{tf_status} Paddle:{paddle_status} Match:{match_status}")

                    if execution_result["tf_success"] and execution_result["paddle_success"]:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

                    case_results.append(
                        {
                            "iteration": len(case_results) + 1,
                            "tf_test_case": current_tf_test_case,
                            "paddle_test_case": current_paddle_test_case,
                            "execution_result": execution_result,
                            "llm_operation": {"operation": "final_execution", "reason": "Run final LLM-generated case"},
                            "case_number": case_number,
                            "is_llm_generated": True,
                        }
                    )
                except Exception as error:
                    self._safe_print(f" | ❌ Final case execution failed: {str(error)[:80]}...")
                    case_results.append(
                        {
                            "iteration": len(case_results) + 1,
                            "tf_test_case": current_tf_test_case,
                            "paddle_test_case": current_paddle_test_case,
                            "execution_result": {
                                "status": "fatal_error",
                                "tf_success": False,
                                "paddle_success": False,
                                "results_match": False,
                                "error": str(error),
                            },
                            "llm_operation": {"operation": "final_execution", "reason": "Final case execution failed"},
                            "case_number": case_number,
                            "is_llm_generated": True,
                        }
                    )

        self._safe_print(f"  ✅ Case {case_number} complete, {len(case_results)} iterations")
        return case_results

    def _convert_llm_test_cases(
        self, tf_test_case: Dict[str, Any], paddle_test_case: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        shared_tensors: Dict[str, np.ndarray] = {}
        all_keys = set(tf_test_case.keys()) | set(paddle_test_case.keys())
        for key in all_keys:
            if key == "api":
                continue
            tf_value = tf_test_case.get(key)
            paddle_value = paddle_test_case.get(key)
            tensor_desc = tf_value if isinstance(tf_value, dict) and "shape" in tf_value else paddle_value if isinstance(paddle_value, dict) and "shape" in paddle_value else None
            if tensor_desc:
                shared_tensors[key] = self.generate_numpy_data(tensor_desc)

        converted_tf = {key: shared_tensors[key] if key in shared_tensors else value for key, value in tf_test_case.items()}
        converted_paddle = {key: shared_tensors[key] if key in shared_tensors else value for key, value in paddle_test_case.items()}
        return converted_tf, converted_paddle

    def save_results(self, tf_api: str, results: List[Dict[str, Any]], stats: Dict[str, int] = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = tf_api.replace(".", "_")
        filename = f"llm_enhanced_{safe_name}_{timestamp}.json"
        filepath = os.path.join(self.result_dir, filename)

        output_results = []
        for result in results:
            output_result = copy.deepcopy(result)
            for case_key in ["tf_test_case", "paddle_test_case"]:
                if case_key in output_result and isinstance(output_result[case_key], dict):
                    simplified = {}
                    for key, value in output_result[case_key].items():
                        simplified[key] = {"shape": list(value.shape), "dtype": str(value.dtype)} if isinstance(value, np.ndarray) else value
                    output_result[case_key] = simplified
            output_results.append(output_result)

        output_data = {
            "tf_api": tf_api,
            "paddle_api": self.api_mapping.get(tf_api, ""),
            "timestamp": datetime.now().isoformat(),
            "total_iterations": len(results),
            "llm_generated_test_cases": stats.get("llm_generated_cases", 0) if stats else 0,
            "successful_test_cases": stats.get("successful_cases", 0) if stats else 0,
            "results": output_results,
        }

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(output_data, file, indent=2, ensure_ascii=False)
        self._safe_print(f"💾 Results saved to: {filepath}")

    def get_all_testable_apis(self) -> List[str]:
        return [tf_api for tf_api in sorted(self.test_cases_data.keys()) if self.api_mapping.get(tf_api, "无对应实现") not in ("", "无对应实现")]

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description="LLM-based differential testing for TensorFlow vs Paddle operators")
    parser.add_argument("--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--operators", "-o", nargs="*")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--key-path", "-k", default=DEFAULT_KEY_PATH)
    parser.add_argument("--test-cases-file", default=DEFAULT_TEST_CASES_FILE)
    parser.add_argument("--mapping-file", default=DEFAULT_MAPPING_FILE)
    args = parser.parse_args()

    num_workers = max(1, args.workers)

    print("=" * 80)
    print("LLM-based differential testing for TensorFlow vs Paddle operators")
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
            print(f"📋 Operators to test: {len(operator_names)}")

        print(f"📋 Operator list: {', '.join(operator_names[:10])}{'...' if len(operator_names) > 10 else ''}\n")

        all_operators_summary = []
        batch_log_file = os.path.join(comparator.result_dir, f"batch_test_log_{start_datetime.strftime('%Y%m%d_%H%M%S')}.txt")
        log_file = open(batch_log_file, "w", encoding="utf-8")
        log_file.write("=" * 80 + "\n")
        log_file.write("TF→Paddle batch differential testing log\n")
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
                    all_operators_summary.append(
                        {
                            "operator": tf_api,
                            "paddle_api": comparator.api_mapping.get(tf_api, ""),
                            "total_iterations": len(results),
                            "llm_generated_cases": stats.get("llm_generated_cases", 0),
                            "successful_cases": stats.get("successful_cases", 0),
                            "status": "completed",
                        }
                    )

                    print(f"\n✅ {tf_api} testing complete")
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
                    all_operators_summary.append(
                        {
                            "operator": tf_api,
                            "total_iterations": 0,
                            "llm_generated_cases": 0,
                            "successful_cases": 0,
                            "status": "no_results",
                        }
                    )
                    log_file.write(f"[{idx}/{len(operator_names)}] {tf_api}\n")
                    log_file.write("  Status: ⚠️ no results\n\n")
                    log_file.flush()
            except Exception as error:
                print(f"\n❌ {tf_api} testing failed: {error}")
                all_operators_summary.append(
                    {
                        "operator": tf_api,
                        "total_iterations": 0,
                        "llm_generated_cases": 0,
                        "successful_cases": 0,
                        "status": "failed",
                        "error": str(error),
                    }
                )
                log_file.write(f"[{idx}/{len(operator_names)}] {tf_api}\n")
                log_file.write(f"  Status: ❌ failed\n  Error: {str(error)}\n\n")
                log_file.flush()

        end_time = time.time()
        end_datetime = datetime.now()
        total_duration = end_time - start_time
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)

        completed_count = sum(1 for summary in all_operators_summary if summary["status"] == "completed")
        failed_count = sum(1 for summary in all_operators_summary if summary["status"] == "failed")
        no_results_count = sum(1 for summary in all_operators_summary if summary["status"] == "no_results")
        total_llm_cases = sum(summary["llm_generated_cases"] for summary in all_operators_summary)
        total_successful = sum(summary["successful_cases"] for summary in all_operators_summary)
        total_iterations = sum(summary["total_iterations"] for summary in all_operators_summary)

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

        summary_file = os.path.join(comparator.result_dir, f"batch_test_summary_{start_datetime.strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "test_config": {
                        "max_iterations": args.max_iterations,
                        "num_test_cases": args.num_cases,
                        "workers": num_workers,
                        "model": args.model,
                    },
                    "time_info": {
                        "start_time": start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                        "end_time": end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
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
                },
                file,
                indent=2,
                ensure_ascii=False,
            )
        print(f"💾 JSON summary saved to: {summary_file}")
    finally:
        comparator.close()
        print("\n✅ Batch test run completed")


if __name__ == "__main__":
    main()