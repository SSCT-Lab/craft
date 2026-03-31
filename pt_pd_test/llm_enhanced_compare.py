#!/usr/bin/env python3
"""
LLM-based PyTorch and PaddlePaddle operator comparison test framework.
Uses a large model to repair and mutate test cases to improve usability and coverage.
"""

import pymongo
import torch
import paddle
import numpy as np
import pandas as pd
import re
import copy
import os
import sys
import json
import traceback
import time
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm

# Add project root to path so we can import the component module.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from component.doc.doc_crawler_factory import get_doc_content


class LLMEnhancedComparator:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "freefuzz-torch"):
        """
        Initialize the LLM-based PyTorch and PaddlePaddle comparator.

        Args:
            mongo_uri: MongoDB connection URI.
            db_name: Database name.
        """
        # MongoDB connection
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["argVS"]

        # Initialize LLM client (Aliyun Qwen).
        # Prefer reading the key from aliyun.key at project root, otherwise use env var.
        api_key = self._load_api_key()
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # Load API mappings
        self.api_mapping = self.load_api_mapping()

        # Create results directory (under pt_pd_test)
        self.result_dir = os.path.join(ROOT_DIR, "pt_pd_test", "pt_pd_log_1")
        os.makedirs(self.result_dir, exist_ok=True)
        print(f"📁 Results directory: {self.result_dir}")

        # Fix random seed for reproducibility
        self.random_seed = 42
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        paddle.seed(self.random_seed)

        # Deprecated PyTorch operator list
        self.deprecated_torch_apis = {
            "torch.symeig": "Removed in PyTorch 1.9. Use torch.linalg.eigh instead."
        }

        # Operators that can hang or crash (skip tests for these)
        self.problematic_apis = {
            "torch.nn.Embedding": "May hang the program",
            "torch.nn.functional.embedding": "May hang the program",
            "torch.nn.functional.max_unpool1d": "May hang the program",
            "torch.nn.functional.max_unpool2d": "May hang the program",
            "torch.nn.functional.max_unpool3d": "May hang the program",
            "torch.nn.MaxUnpool1d": "May hang the program",
            "torch.nn.MaxUnpool2d": "May hang the program",
            "torch.nn.MaxUnpool3d": "May hang the program",
        }

    def _load_api_key(self) -> str:
        """
        Load the Aliyun API key.

        Prefer reading aliyun.key at project root. If missing, use DASHSCOPE_API_KEY.

        Returns:
            API key string.
        """
        key_file = os.path.join(ROOT_DIR, "aliyun.key")

        # Prefer file
        if os.path.exists(key_file):
            try:
                with open(key_file, "r", encoding="utf-8") as f:
                    api_key = f.read().strip()
                if api_key:
                    print(f"✅ Loaded API key from file: {key_file}")
                    return api_key
            except Exception as e:
                print(f"⚠️ Failed to read key file: {e}")

        # Fallback to env var
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            print("✅ Loaded API key from environment: DASHSCOPE_API_KEY")
            return api_key

        # Not found
        print("❌ API key not found. Provide aliyun.key or set DASHSCOPE_API_KEY.")
        return ""

    def load_api_mapping(self) -> Dict[str, Dict[str, str]]:
        """Load the PyTorch-to-PaddlePaddle API mapping table."""
        # Use the updated mapping file
        mapping_file = os.path.join(ROOT_DIR, "component", "data", "pd_api_mappings_final.csv")
        try:
            df = pd.read_csv(mapping_file)
            mapping = {}

            for _, row in df.iterrows():
                # New mapping file columns are pytorch-api and paddle-api
                pt_api = str(row["pytorch-api"]).strip()
                pd_api = str(row["paddle-api"]).strip()
                mapping[pt_api] = {"pd_api": pd_api, "note": ""}

            print(f"✅ Loaded API mapping table with {len(mapping)} entries")
            print(f"📄 Mapping file: {mapping_file}")
            return mapping
        except Exception as e:
            print(f"❌ Failed to load API mapping table: {e}")
            return {}

    def is_class_based_api(self, api_name: str) -> bool:
        """Check whether an API is class-based."""
        parts = api_name.split(".")
        if len(parts) >= 2:
            name = parts[-1]
            return any(c.isupper() for c in name)
        return False

    # def convert_class_to_functional(self, torch_api: str) -> Tuple[Optional[str], Optional[str]]:
    #     """Convert class-style API to functional style."""
    #     if not self.is_class_based_api(torch_api):
    #         return None, None
    #
    #     parts = torch_api.split(".")
    #     if len(parts) >= 3 and parts[1] == "nn":
    #         class_name = parts[-1]
    #
    #         # Improved regex handling for consecutive uppercase letters
    #         # 1. Insert underscore after a lowercase letter followed by uppercase
    #         # 2. Insert underscore before last uppercase in a group if followed by lowercase
    #         func_name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", class_name)  # aB -> a_B
    #         func_name = re.sub("([A-Z]+)([A-Z][a-z])", r"\1_\2", func_name)  # ABCDef -> ABC_Def
    #         func_name = func_name.lower()
    #
    #         torch_func_api = f"torch.nn.functional.{func_name}"
    #         paddle_func_api = f"paddle.nn.functional.{func_name}"
    #
    #         return torch_func_api, paddle_func_api
    #
    #     return None, None

    def convert_api_name(self, torch_api: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        Convert a PyTorch API to a PaddlePaddle API.

        Uses pd_api_mappings_final.csv only; no manual name conversion.

        Returns:
            (converted_torch_api, converted_paddle_api, mapping_method)
            - Found in mapping table with valid Paddle API -> returns mapped APIs
            - Found but value is "无对应实现" -> returns (torch_api, None, "no_corresponding_implementation")
            - Not found -> returns (torch_api, None, "not_found_in_mapping_table")
        """
        # Lookup in mapping table
        if torch_api in self.api_mapping:
            pd_api = self.api_mapping[torch_api]["pd_api"]

            if pd_api == "无对应实现":
                return torch_api, None, "no_corresponding_implementation"
            return torch_api, pd_api, "mapping_table"

        # Not found in mapping table; no manual conversion
        return torch_api, None, "not_found_in_mapping_table"

    def get_operator_function(self, api_name: str, framework: str = "torch"):
        """Get an operator function."""
        try:
            parts = api_name.split(".")
            if len(parts) >= 2:
                if framework == "torch" and parts[0] == "torch":
                    if len(parts) == 2:
                        return getattr(torch, parts[1])
                    if len(parts) == 3:
                        module = getattr(torch, parts[1])
                        return getattr(module, parts[2])
                    if len(parts) == 4:
                        module1 = getattr(torch, parts[1])
                        module2 = getattr(module1, parts[2])
                        return getattr(module2, parts[3])
                elif framework == "paddle" and parts[0] == "paddle":
                    if len(parts) == 2:
                        return getattr(paddle, parts[1])
                    if len(parts) == 3:
                        module = getattr(paddle, parts[1])
                        return getattr(module, parts[2])
                    if len(parts) == 4:
                        module1 = getattr(paddle, parts[1])
                        module2 = getattr(module1, parts[2])
                        return getattr(module2, parts[3])
            return None
        except AttributeError:
            return None

    def convert_key(self, key: str, paddle_api: str = "") -> str:
        """Convert parameter names."""
        key_mapping = {
            "input": "x",
            "other": "y",
        }
        return key_mapping.get(key, key)

    def should_skip_param(self, key: str, paddle_api: str) -> bool:
        """Check whether a parameter should be skipped."""
        common_skip_params = ["layout", "requires_grad", "out"]
        skip_params = {
            "paddle.nn.functional.dropout2d": ["inplace"],
        }

        if key in common_skip_params:
            return True

        if paddle_api in skip_params:
            return key in skip_params[paddle_api]

        return False

    def generate_numpy_data(self, data: Any) -> np.ndarray:
        """
        Generate a numpy array as shared data source.

        Supported dtype formats:
        - torch prefixed: torch.float32, torch.bool, torch.int64, etc.
        - no prefix: float32, bool, int64, etc.
        - numpy formats: float32, bool_, int64, etc.
        """
        if isinstance(data, dict):
            # Extended dtype map to support multiple formats
            dtype_map = {
                # torch formats (prefixed)
                "torch.float64": np.float64,
                "torch.float32": np.float32,
                "torch.int64": np.int64,
                "torch.int32": np.int32,
                "torch.bool": np.bool_,
                "torch.uint8": np.uint8,
                # no torch prefix (LLM may return these)
                "float64": np.float64,
                "float32": np.float32,
                "int64": np.int64,
                "int32": np.int32,
                "bool": np.bool_,
                "uint8": np.uint8,
                # numpy formats
                "bool_": np.bool_,
                "float": np.float32,
                "int": np.int64,
            }

            shape = data.get("shape", [])
            dtype_str = data.get("dtype", "torch.float32")
            dtype = dtype_map.get(dtype_str, np.float32)

            # Warn if dtype_str is not in map
            if dtype_str not in dtype_map:
                print(f"      ⚠️ Warning: unrecognized dtype '{dtype_str}', using default float32")
            else:
                print(f"      ✅ dtype mapping: '{dtype_str}' -> {dtype}")

            if shape:
                if dtype == np.bool_:
                    return np.random.randint(0, 2, shape).astype(np.bool_)
                if dtype in [np.int64, np.int32]:
                    return np.random.randint(-10, 10, shape).astype(dtype)
                return np.random.randn(*shape).astype(dtype)
            else:
                if dtype == np.bool_:
                    return np.array(True, dtype=np.bool_)
                if dtype in [np.int64, np.int32]:
                    return np.array(1, dtype=dtype)
                return np.array(1.0, dtype=dtype)
        if isinstance(data, (int, float)):
            return np.array(data)
        if isinstance(data, list):
            return np.array(data)
        return np.array(data)

    def prepare_shared_numpy_data(self, document: Dict[str, Any], case_index: int = 0) -> Dict[str, Any]:
        """Prepare shared numpy data to ensure both frameworks use the same inputs."""
        shared_data = {}
        api_name = document.get("api", "")

        # For class-based APIs without input, create default input
        if self.is_class_based_api(api_name) and "input" not in document:
            if "2d" in api_name.lower() or "2D" in api_name:
                default_shape = {"shape": [2, 3, 4, 4], "dtype": "torch.float32"}
            elif "1d" in api_name.lower() or "1D" in api_name:
                default_shape = {"shape": [2, 3, 10], "dtype": "torch.float32"}
            elif "3d" in api_name.lower() or "3D" in api_name:
                default_shape = {"shape": [2, 3, 4, 4, 4], "dtype": "torch.float32"}
            else:
                default_shape = {"shape": [2, 3], "dtype": "torch.float32"}

            shared_data["input"] = self.generate_numpy_data(default_shape)

        # Handle other parameters in document
        exclude_keys = ["_id", "api"]
        for key, value in document.items():
            if key not in exclude_keys:
                # For variadic params (prefixed with *), keep raw value without conversion
                # Conversion happens in prepare_arguments_torch/paddle
                if key.startswith("*"):
                    if isinstance(value, list) and len(value) > 0:
                        idx = min(case_index, len(value) - 1)
                        shared_data[key] = value[idx]
                    else:
                        shared_data[key] = value
                elif isinstance(value, list):
                    if len(value) > 0:
                        idx = min(case_index, len(value) - 1)
                        param_value = value[idx]
                        if isinstance(param_value, dict):
                            shared_data[key] = self.generate_numpy_data(param_value)
                        else:
                            shared_data[key] = param_value
                else:
                    shared_data[key] = value

        return shared_data

    def convert_to_tensor_torch(self, data: Any, numpy_data: np.ndarray = None) -> torch.Tensor:
        """Convert data to a PyTorch tensor."""
        if numpy_data is not None:
            return torch.from_numpy(numpy_data.copy())

        if isinstance(data, dict):
            numpy_data = self.generate_numpy_data(data)
            return torch.from_numpy(numpy_data.copy())
        if isinstance(data, (int, float)):
            return torch.tensor(data)
        if isinstance(data, list):
            return torch.tensor(data)
        return torch.tensor(data)

    def convert_to_tensor_paddle(self, data: Any, numpy_data: np.ndarray = None) -> paddle.Tensor:
        """Convert data to a PaddlePaddle tensor."""
        if numpy_data is not None:
            return paddle.to_tensor(numpy_data.copy())

        if isinstance(data, dict):
            numpy_data = self.generate_numpy_data(data)
            return paddle.to_tensor(numpy_data.copy())
        if isinstance(data, (int, float)):
            return paddle.to_tensor(data)
        if isinstance(data, list):
            return paddle.to_tensor(data)
        return paddle.to_tensor(data)

    def prepare_arguments_torch(self, test_case: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Prepare arguments for PyTorch.

        Notes:
        1. For torch.where, args must be positional:
           - torch.where(condition, x, y) or torch.where(condition, input, other)
        2. Params prefixed with * (e.g., *tensors) are varargs and must be unpacked
        """
        args = []
        kwargs = {}

        # First check for varargs (prefixed with *)
        varargs_key = None
        for key in test_case.keys():
            if key.startswith("*"):
                varargs_key = key
                break

        # If varargs exist, unpack to positional args
        if varargs_key:
            varargs_value = test_case[varargs_key]
            if isinstance(varargs_value, list):
                for item in varargs_value:
                    if isinstance(item, dict) and "shape" in item:
                        # Tensor description; generate numpy and convert
                        numpy_data = self.generate_numpy_data(item)
                        args.append(self.convert_to_tensor_torch(None, numpy_data))
                    elif isinstance(item, list):
                        # Nested list; process recursively
                        nested_tensors = []
                        for nested_item in item:
                            if isinstance(nested_item, dict) and "shape" in nested_item:
                                numpy_data = self.generate_numpy_data(nested_item)
                                nested_tensors.append(self.convert_to_tensor_torch(None, numpy_data))
                            elif isinstance(nested_item, np.ndarray):
                                nested_tensors.append(self.convert_to_tensor_torch(None, nested_item))
                            else:
                                nested_tensors.append(nested_item)
                        args.extend(nested_tensors)
                    elif isinstance(item, np.ndarray):
                        args.append(self.convert_to_tensor_torch(None, item))
                    else:
                        args.append(item)
            return args, kwargs

        # Process positional params in order: condition, x/input, y/other
        # These must be passed as positional args, not kwargs
        positional_params = ["condition", "x", "y", "input", "other"]

        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, np.ndarray):
                    args.append(self.convert_to_tensor_torch(None, value))
                else:
                    # Scalar values go directly
                    args.append(value)

        # Process remaining params as kwargs
        for key, value in test_case.items():
            if key not in positional_params + ["api"]:
                if isinstance(value, np.ndarray):
                    kwargs[key] = self.convert_to_tensor_torch(None, value)
                else:
                    kwargs[key] = value

        return args, kwargs

    def prepare_arguments_paddle(self, test_case: Dict[str, Any], paddle_api: str) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Prepare arguments for PaddlePaddle.

        Notes:
        1. For paddle.where, args must be positional:
           - paddle.where(condition, x, y)
        2. Params prefixed with * (e.g., *tensors) are varargs and must be unpacked
        """
        args = []
        kwargs = {}

        # First check for varargs (prefixed with *)
        varargs_key = None
        for key in test_case.keys():
            if key.startswith("*"):
                varargs_key = key
                break

        # If varargs exist, unpack to positional args
        if varargs_key:
            varargs_value = test_case[varargs_key]
            if isinstance(varargs_value, list):
                for item in varargs_value:
                    if isinstance(item, dict) and "shape" in item:
                        # Tensor description; generate numpy and convert
                        numpy_data = self.generate_numpy_data(item)
                        args.append(self.convert_to_tensor_paddle(None, numpy_data))
                    elif isinstance(item, list):
                        # Nested list; process recursively
                        nested_tensors = []
                        for nested_item in item:
                            if isinstance(nested_item, dict) and "shape" in nested_item:
                                numpy_data = self.generate_numpy_data(nested_item)
                                nested_tensors.append(self.convert_to_tensor_paddle(None, numpy_data))
                            elif isinstance(nested_item, np.ndarray):
                                nested_tensors.append(self.convert_to_tensor_paddle(None, nested_item))
                            else:
                                nested_tensors.append(nested_item)
                        args.extend(nested_tensors)
                    elif isinstance(item, np.ndarray):
                        args.append(self.convert_to_tensor_paddle(None, item))
                    else:
                        args.append(item)
            return args, kwargs

        # Process positional params in order: condition, x/input, y/other
        positional_params = ["condition", "x", "y", "input", "other"]

        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, np.ndarray):
                    args.append(self.convert_to_tensor_paddle(None, value))
                else:
                    # Scalar values go directly
                    args.append(value)

        # Process remaining params as kwargs
        for key, value in test_case.items():
            if key not in positional_params + ["api"]:
                if self.should_skip_param(key, paddle_api):
                    continue

                if isinstance(value, np.ndarray):
                    kwargs[key] = self.convert_to_tensor_paddle(None, value)
                else:
                    kwargs[key] = value

        return args, kwargs

    def compare_tensors(self, torch_result, paddle_result, tolerance: float = 1e-5) -> Tuple[bool, str]:
        """Compare two tensors for equality."""
        try:
            # Convert to numpy for comparison
            if hasattr(torch_result, "detach"):
                torch_np = torch_result.detach().cpu().numpy()
            else:
                torch_np = np.array(torch_result)

            if hasattr(paddle_result, "numpy"):
                paddle_np = paddle_result.numpy()
            else:
                paddle_np = np.array(paddle_result)

            # Check shape
            if torch_np.shape != paddle_np.shape:
                return False, f"Shape mismatch: PyTorch {torch_np.shape} vs PaddlePaddle {paddle_np.shape}"

            # Check bool dtype
            if torch_np.dtype == np.bool_ or paddle_np.dtype == np.bool_:
                if np.array_equal(torch_np, paddle_np):
                    return True, "Boolean values match"
                diff_count = np.sum(torch_np != paddle_np)
                return False, f"Boolean values mismatch, diff count: {diff_count}"

            # Check numeric values
            if np.allclose(torch_np, paddle_np, atol=tolerance, rtol=tolerance, equal_nan=True):
                return True, "Numeric values match"
            max_diff = np.max(np.abs(torch_np - paddle_np))
            return False, f"Numeric values mismatch, max diff: {max_diff}"

        except Exception as e:
            return False, f"Comparison failed: {str(e)}"

    def execute_test_case(self, torch_api: str, paddle_api: str, torch_test_case: Dict[str, Any], paddle_test_case: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a single test case.

        Args:
            torch_api: PyTorch API name.
            paddle_api: PaddlePaddle API name.
            torch_test_case: PyTorch test case (parameters included).
            paddle_test_case: PaddlePaddle test case (parameters included).

        Returns:
            Execution result dict.
        """
        result = {
            "torch_api": torch_api,
            "paddle_api": paddle_api,
            "torch_success": False,
            "paddle_success": False,
            "results_match": False,
            "torch_error": None,
            "paddle_error": None,
            "comparison_error": None,
            "torch_shape": None,
            "paddle_shape": None,
            "torch_dtype": None,
            "paddle_dtype": None,
            "status": "unknown",
        }

        # If paddle_test_case not provided, reuse torch_test_case (backward compatible)
        if paddle_test_case is None:
            paddle_test_case = torch_test_case

        # Check if class operator
        is_class_api = self.is_class_based_api(torch_api)

        # Test PyTorch
        torch_result = None
        try:
            torch_func = self.get_operator_function(torch_api, "torch")
            if torch_func is None:
                result["torch_error"] = f"PyTorch operator {torch_api} not found"
            else:
                args, kwargs = self.prepare_arguments_torch(torch_test_case)

                if is_class_api:
                    # For class operators: instantiate then call
                    # Extract init args (non-input)
                    init_kwargs = {k: v for k, v in kwargs.items() if k != "input"}
                    # Instantiate class
                    torch_instance = torch_func(**init_kwargs)
                    # Get input data (from args or kwargs)
                    if "input" in kwargs:
                        input_data = kwargs["input"]
                    elif len(args) > 0:
                        input_data = args[0]
                    else:
                        # If no input, try default
                        raise ValueError("Class operator missing input parameter")

                    # Call instance (forward)
                    with torch.no_grad():
                        torch_result = torch_instance(input_data)
                else:
                    # Function operator: call directly
                    with torch.no_grad():
                        torch_result = torch_func(*args, **kwargs)

                result["torch_success"] = True
                result["torch_shape"] = list(torch_result.shape) if hasattr(torch_result, "shape") else None
                result["torch_dtype"] = str(torch_result.dtype) if hasattr(torch_result, "dtype") else None
        except Exception as e:
            result["torch_error"] = str(e)
            result["torch_traceback"] = traceback.format_exc()

        # Test PaddlePaddle
        paddle_result = None
        try:
            paddle_func = self.get_operator_function(paddle_api, "paddle")
            if paddle_func is None:
                result["paddle_error"] = f"PaddlePaddle operator {paddle_api} not found"
            else:
                args, kwargs = self.prepare_arguments_paddle(paddle_test_case, paddle_api)

                if is_class_api:
                    # For class operators: instantiate then call
                    # Extract init args (non-x/input)
                    init_kwargs = {k: v for k, v in kwargs.items() if k not in ["x", "input"]}
                    # Instantiate class
                    paddle_instance = paddle_func(**init_kwargs)
                    # Get input data (from args or kwargs)
                    if "x" in kwargs:
                        input_data = kwargs["x"]
                    elif "input" in kwargs:
                        input_data = kwargs["input"]
                    elif len(args) > 0:
                        input_data = args[0]
                    else:
                        # If no input, try default
                        raise ValueError("Class operator missing input/x parameter")

                    # Call instance (forward)
                    paddle_result = paddle_instance(input_data)
                else:
                    # Function operator: call directly
                    paddle_result = paddle_func(*args, **kwargs)

                result["paddle_success"] = True
                result["paddle_shape"] = list(paddle_result.shape) if hasattr(paddle_result, "shape") else None
                result["paddle_dtype"] = str(paddle_result.dtype) if hasattr(paddle_result, "dtype") else None
        except Exception as e:
            result["paddle_error"] = str(e)
            result["paddle_traceback"] = traceback.format_exc()

        # Compare results
        if result["torch_success"] and result["paddle_success"]:
            try:
                is_match, comparison_msg = self.compare_tensors(torch_result, paddle_result)
                result["results_match"] = is_match
                result["comparison_error"] = comparison_msg if not is_match else None
                result["status"] = "compared"
            except Exception as e:
                result["comparison_error"] = str(e)
                result["status"] = "comparison_failed"
        elif result["torch_success"] and not result["paddle_success"]:
            result["status"] = "paddle_failed"
        elif not result["torch_success"] and result["paddle_success"]:
            result["status"] = "torch_failed"
        else:
            result["status"] = "both_failed"

        return result

    def _fetch_api_docs(self, torch_api: str, paddle_api: str) -> Tuple[str, str]:
        """
        Crawl PyTorch and PaddlePaddle API docs.

        Args:
            torch_api: PyTorch API name.
            paddle_api: PaddlePaddle API name.

        Returns:
            (PyTorch doc text, PaddlePaddle doc text)
        """
        # Minimum length to consider a doc valid
        MIN_DOC_LENGTH = 300

        torch_doc = ""
        paddle_doc = ""

        try:
            print(f"    📖 Fetching PyTorch docs: {torch_api}")
            torch_doc = get_doc_content(torch_api, "pytorch")
            # Doc validity: non-empty, no error text, length above threshold
            if (
                torch_doc
                and "Unable" not in torch_doc
                and "not supported" not in torch_doc
                and len(torch_doc.strip()) > MIN_DOC_LENGTH
            ):
                # Truncate long docs to save tokens
                if len(torch_doc) > 3000:
                    torch_doc = torch_doc[:3000] + "\n... (doc truncated)"
                print(f"    ✅ PyTorch docs fetched, length: {len(torch_doc)}")
            else:
                doc_len = len(torch_doc.strip()) if torch_doc else 0
                torch_doc = (
                    f"Unable to fetch documentation for {torch_api} "
                    f"(length: {doc_len}, min required: {MIN_DOC_LENGTH})"
                )
                print(f"    ⚠️ {torch_doc}")
        except Exception as e:
            torch_doc = f"Failed to fetch documentation: {str(e)}"
            print(f"    ❌ PyTorch docs fetch failed: {e}")

        try:
            print(f"    📖 Fetching PaddlePaddle docs: {paddle_api}")
            paddle_doc = get_doc_content(paddle_api, "paddle")
            # Doc validity: non-empty, no error text, length above threshold
            if (
                paddle_doc
                and "Unable" not in paddle_doc
                and "not supported" not in paddle_doc
                and len(paddle_doc.strip()) > MIN_DOC_LENGTH
            ):
                # Truncate long docs to save tokens
                if len(paddle_doc) > 3000:
                    paddle_doc = paddle_doc[:3000] + "\n... (doc truncated)"
                print(f"    ✅ PaddlePaddle docs fetched, length: {len(paddle_doc)}")
            else:
                doc_len = len(paddle_doc.strip()) if paddle_doc else 0
                paddle_doc = (
                    f"Unable to fetch documentation for {paddle_api} "
                    f"(length: {doc_len}, min required: {MIN_DOC_LENGTH})"
                )
                print(f"    ⚠️ {paddle_doc}")
        except Exception as e:
            paddle_doc = f"Failed to fetch documentation: {str(e)}"
            print(f"    ❌ PaddlePaddle docs fetch failed: {e}")

        return torch_doc, paddle_doc

    def _build_llm_prompt(
        self,
        execution_result: Dict[str, Any],
        torch_test_case: Dict[str, Any],
        paddle_test_case: Dict[str, Any],
        torch_doc: str = "",
        paddle_doc: str = "",
    ) -> str:
        """Build the LLM prompt."""
        torch_api = execution_result.get("torch_api", "")
        paddle_api = execution_result.get("paddle_api", "")
        status = execution_result.get("status", "")
        torch_success = execution_result.get("torch_success", False)
        paddle_success = execution_result.get("paddle_success", False)
        results_match = execution_result.get("results_match", False)
        torch_error = execution_result.get("torch_error", "")
        paddle_error = execution_result.get("paddle_error", "")
        comparison_error = execution_result.get("comparison_error", "")

        # Simplify PyTorch test case to reduce token usage
        simplified_torch_test_case = {}
        for key, value in torch_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_torch_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_torch_test_case[key] = value

        # Simplify PaddlePaddle test case to reduce token usage
        simplified_paddle_test_case = {}
        for key, value in paddle_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_paddle_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_paddle_test_case[key] = value

        # Build PyTorch param examples
        torch_param_examples = []
        for key, value in simplified_torch_test_case.items():
            if key == "api":
                continue
            if isinstance(value, dict) and "shape" in value:
                torch_param_examples.append(f'    "{key}": {json.dumps(value)}')
            elif isinstance(value, (int, float)):
                torch_param_examples.append(f'    "{key}": {value}')
            else:
                torch_param_examples.append(f'    "{key}": {json.dumps(value)}')

        torch_param_example_str = (
            ",\n".join(torch_param_examples)
            if torch_param_examples
            else '    "input": {"shape": [2, 3], "dtype": "torch.float32"}'
        )

        # Build PaddlePaddle param examples
        paddle_param_examples = []
        for key, value in simplified_paddle_test_case.items():
            if key == "api":
                continue
            if isinstance(value, dict) and "shape" in value:
                paddle_param_examples.append(f'    "{key}": {json.dumps(value)}')
            elif isinstance(value, (int, float)):
                paddle_param_examples.append(f'    "{key}": {value}')
            else:
                paddle_param_examples.append(f'    "{key}": {json.dumps(value)}')

        paddle_param_example_str = (
            ",\n".join(paddle_param_examples)
            if paddle_param_examples
            else '    "x": {"shape": [2, 3], "dtype": "float32"}'
        )

        # Build doc section
        doc_section = ""
        if torch_doc or paddle_doc:
            doc_section = f"""
## API Docs Reference
### PyTorch Docs
{torch_doc if torch_doc else "Docs unavailable"}

### PaddlePaddle Docs
{paddle_doc if paddle_doc else "Docs unavailable"}
"""

        prompt = f"""Please analyze the following operator test case results in PyTorch and PaddlePaddle, and repair or mutate (fuzz) the test case based on the outcome.

## Test Info
- **PyTorch API**: {torch_api}
- **PaddlePaddle API**: {paddle_api}
{doc_section}
## Execution Result
- **Status**: {status}
- **PyTorch Success**: {torch_success}
- **PaddlePaddle Success**: {paddle_success}
- **Results Match**: {results_match}

## Error Info
- **PyTorch Error**: {torch_error if torch_error else "None"}
- **PaddlePaddle Error**: {paddle_error if paddle_error else "None"}
- **Comparison Error**: {comparison_error if comparison_error else "None"}

## Original Test Cases

### PyTorch Test Case
```json
{json.dumps(simplified_torch_test_case, indent=2, ensure_ascii=False)}
```

### PaddlePaddle Test Case
```json
{json.dumps(simplified_paddle_test_case, indent=2, ensure_ascii=False)}
```

## Task Requirements
Based on the above (including official API docs), decide whether the results are **consistent**, **inconsistent**, or **execution failed**, and perform one of the following:

1. **If consistent**: **Mutate (fuzz)** the case, e.g., change input shapes or parameter values (consider edge/extreme values)
2. **If execution failed**: **Repair** the case according to the error and docs (change parameter names, counts, types, ranges, etc.; they may differ between frameworks) or **Skip** if the cross-framework operators are not equivalent
3. **If inconsistent**: Decide whether this is a tolerable precision error ($\leq 1e-3$). (1) If tolerable, **Mutate**; (2) If operators are not equivalent per docs, **Skip**; (3) Otherwise, treat it as a test construction issue and **Repair** based on docs.

## Output Format
Strictly output JSON in this format, with no other text, comments, or markdown:

{{
  "operation": "mutation",
  "reason": "Detailed reason for this operation",
  "pytorch_test_case": {{
    "api": "{torch_api}",
{torch_param_example_str}
  }},
  "paddle_test_case": {{
    "api": "{paddle_api}",
{paddle_param_example_str}
  }}
}}

**Important Notes**:
1. `operation` must be one of "mutation", "repair", or "skip"
2. Tensor parameters must use {{"shape": [...], "dtype": "..."}} format
3. Scalar parameters should be direct values, e.g., "y": 0
4. Ensure inputs are identical between frameworks and parameters are semantically equivalent with matching expected outputs.
5. PyTorch and PaddlePaddle test cases may differ in parameter names (input vs x), values, or counts as long as outputs are equivalent.
6. If official docs are unavailable because the operator does not exist or was removed in current versions, set `operation` to "skip" and do not attempt repair.
7. For mutation, explore edge cases such as empty tensors (shape contains 0), single-element tensors (shape=[1] or []), high-dimensional tensors, very large tensors, different dtypes (int/float/bool), boundary values, etc., to improve coverage and find potential bugs.
8. Read official API docs carefully to ensure parameter names, types, and ranges are compliant.
"""
        return prompt

    def call_llm_for_repair_or_mutation(
        self,
        execution_result: Dict[str, Any],
        torch_test_case: Dict[str, Any],
        paddle_test_case: Dict[str, Any],
        torch_doc: str = "",
        paddle_doc: str = "",
    ) -> Dict[str, Any]:
        """Call the LLM to repair or mutate test cases."""
        prompt = self._build_llm_prompt(execution_result, torch_test_case, paddle_test_case, torch_doc, paddle_doc)

        # Print simplified cases passed to LLM
        simplified_torch_test_case = {}
        for key, value in torch_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_torch_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_torch_test_case[key] = value

        simplified_paddle_test_case = {}
        for key, value in paddle_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_paddle_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_paddle_test_case[key] = value

        # Simplified logging: detailed case info is logged elsewhere
        # print(f"\n{'='*40}")
        # print("📋 LLM input test cases (simplified):")
        # print(f"{'='*40}")
        # print("\n🅿️ PyTorch test case:")
        # print(json.dumps(simplified_torch_test_case, indent=2, ensure_ascii=False))
        # print("\n🅰️ PaddlePaddle test case:")
        # print(json.dumps(simplified_paddle_test_case, indent=2, ensure_ascii=False))
        # print(f"\n{'='*40}\n")

        try:
            print("    🤖 Calling LLM for analysis...")
            completion = self.llm_client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a deep learning framework testing expert. "
                            "You are proficient with PyTorch and PaddlePaddle API differences. "
                            "Given test case execution results, decide whether to repair or mutate "
                            "the test cases and return a strict JSON response."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.1,
            )

            raw_response = completion.choices[0].message.content.strip()
            # Simplified logging: detailed LLM responses are logged elsewhere
            # print(f"    🤖 LLM raw response: {raw_response[:200]}...")
            # print(f"    🤖 LLM raw response: {raw_response}")

            # Add 1-second delay to avoid excessive API calls
            time.sleep(1)

            # Try to parse JSON
            try:
                llm_result = json.loads(raw_response)
                return llm_result
            except json.JSONDecodeError as e:
                print("    ⚠️ LLM did not return valid JSON; attempting to extract JSON...")
                json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                if json_match:
                    llm_result = json.loads(json_match.group())
                    return llm_result
                return {
                    "operation": "skip",
                    "reason": f"LLM returned invalid JSON: {e}",
                    "pytorch_test_case": torch_test_case,
                    "paddle_test_case": paddle_test_case,
                }

        except Exception as e:
            print(f"    ❌ LLM call failed: {e}")
            return {
                "operation": "skip",
                "reason": f"LLM call failed: {e}",
                "pytorch_test_case": torch_test_case,
                "paddle_test_case": paddle_test_case,
            }

    def get_num_test_cases_from_document(self, document: Dict[str, Any]) -> int:
        """Get the number of test cases in the document."""
        max_len = 0
        # Iterate all fields and find the max list length
        for key, value in document.items():
            if key not in ["_id", "api"] and isinstance(value, list):
                max_len = max(max_len, len(value))
        return max_len if max_len > 0 else 1

    def llm_enhanced_test_operator(
        self, operator_name: str, max_iterations: int = 3, num_test_cases: int = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Test a single operator with LLM enhancement.

        Args:
            operator_name: Operator name, e.g., "torch.where".
            max_iterations: Max iterations per test case.
            num_test_cases: Number of cases to test; None means all cases.

        Returns:
            (list of all iteration results for all cases, stats dict)
        """
        print(f"\n{'='*80}")
        print(f"🎯 Start testing operator: {operator_name}")
        print(f"🔄 Max iterations per case: {max_iterations}")
        print(f"{'='*80}\n")

        # Init counters
        stats = {
            "llm_generated_cases": 0,  # total LLM-generated cases
            "successful_cases": 0,  # cases where both frameworks succeeded
        }

        # Check if operator can hang
        if operator_name in self.problematic_apis:
            reason = self.problematic_apis[operator_name]
            print(f"⏭️ Skip operator {operator_name}: {reason}")
            return [], stats

        # Fetch test cases from MongoDB
        document = self.collection.find_one({"api": operator_name})
        if document is None:
            print(f"❌ No test cases found for operator {operator_name}")
            return [], stats

        # Total cases
        total_cases = self.get_num_test_cases_from_document(document)
        print(f"📊 Total test cases in DB: {total_cases}")

        # Determine how many to run
        if num_test_cases is None:
            num_test_cases = total_cases
            print(f"📝 Testing all {num_test_cases} cases")
        else:
            num_test_cases = min(num_test_cases, total_cases)
            print(f"📝 Testing first {num_test_cases} cases (of {total_cases})")

        # Get mapped APIs
        torch_api, paddle_api, mapping_method = self.convert_api_name(operator_name)
        if paddle_api is None:
            print(f"❌ Operator {operator_name} has no PaddlePaddle implementation")
            return [], stats

        # Show mapping info
        if torch_api != operator_name:
            print(f"✅ Original PyTorch API: {operator_name}")
            print(f"✅ Converted PyTorch API: {torch_api}")
        else:
            print(f"✅ PyTorch API: {torch_api}")
        print(f"✅ PaddlePaddle API: {paddle_api}")
        print(f"✅ Mapping method: {mapping_method}\n")

        # Store all results
        all_results = []

        # Iterate cases
        for case_idx in range(num_test_cases):
            print(f"\n{'#'*80}")
            print(f"📋 Test case {case_idx + 1}/{num_test_cases}")
            print(f"{'#'*80}")

            # Prepare data for this case
            print(f"  📦 Preparing data for test case {case_idx + 1}...")
            initial_test_case = self.prepare_shared_numpy_data(document, case_index=case_idx)
            # Use PyTorch API
            initial_test_case["api"] = torch_api

            # Print parameters
            print("  📝 Test case parameters:")
            for key, value in initial_test_case.items():
                if key == "api":
                    continue
                if isinstance(value, np.ndarray):
                    print(f"    - {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"    - {key}: {value}")

            # Iterative testing for this case
            case_results = self._test_single_case_with_iterations(
                torch_api,
                paddle_api,
                initial_test_case,
                max_iterations,
                case_idx + 1,
                stats,
            )

            # Save results for this case
            all_results.extend(case_results)

        print(f"\n{'='*80}")
        print("✅ All tests completed")
        print(f"📊 Tested {num_test_cases} cases, total {len(all_results)} iterations")
        print(f"📊 LLM-generated cases: {stats['llm_generated_cases']}")
        print(f"📊 Cases where both frameworks succeeded: {stats['successful_cases']}")
        print(f"{'='*80}\n")

        return all_results, stats

    def _test_single_case_with_iterations(
        self,
        operator_name: str,
        paddle_api: str,
        initial_test_case: Dict[str, Any],
        max_iterations: int,
        case_number: int,
        stats: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """
        Run multiple iterations for one test case.

        Args:
            operator_name: PyTorch operator name.
            paddle_api: PaddlePaddle operator name.
            initial_test_case: Initial test case.
            max_iterations: Max iterations.
            case_number: Case index (for display).
            stats: Stats dict (LLM-generated cases and success count).

        Returns:
            All iteration results for this case.
        """
        # Store all iterations for this case
        case_results = []

        # Current test case
        # PyTorch uses original case (api already set to torch_api)
        # PaddlePaddle uses a copy with the correct API
        current_torch_test_case = initial_test_case
        current_paddle_test_case = copy.deepcopy(initial_test_case)
        current_paddle_test_case["api"] = paddle_api  # set PaddlePaddle API

        # Whether current case was generated by LLM (first iteration is DB original)
        is_llm_generated = False

        # Pre-fetch docs once per case
        print("\n  📖 Pre-fetching API docs...")
        torch_doc, paddle_doc = self._fetch_api_docs(operator_name, paddle_api)

        # Start iterations
        for iteration in range(max_iterations):
            print(f"\n{'─'*80}")
            print(f"🔄 Iteration {iteration + 1}/{max_iterations}")
            if is_llm_generated:
                print("   (LLM-generated case)")
            else:
                print("   (DB original case)")
            print(f"{'─'*80}")

            # Execute test case
            try:
                # print("  📝 Executing test case...")
                execution_result = self.execute_test_case(
                    operator_name, paddle_api, current_torch_test_case, current_paddle_test_case
                )

                # Simplified logging: only key status
                status = execution_result["status"]
                torch_ok = "✓" if execution_result["torch_success"] else "✗"
                paddle_ok = "✓" if execution_result["paddle_success"] else "✗"
                match_ok = "✓" if execution_result["results_match"] else "✗"
                print(
                    f"  📊 Result: status={status}, PyTorch={torch_ok}, Paddle={paddle_ok}, match={match_ok}"
                )

                # Detailed result logging is commented out (logged elsewhere)
                # print(f"  📊 Status: {execution_result['status']}")
                # print(f"  🅿️ PyTorch success: {execution_result['torch_success']}")
                # print(f"  🅰️ PaddlePaddle success: {execution_result['paddle_success']}")
                # print(f"  ❓ Results match: {execution_result['results_match']}")

                # Only print errors if present
                if execution_result["torch_error"]:
                    print(
                        f"  ❌ PyTorch error: {execution_result['torch_error'][:100]}..."
                        if len(str(execution_result["torch_error"])) > 100
                        else f"  ❌ PyTorch error: {execution_result['torch_error']}"
                    )
                if execution_result["paddle_error"]:
                    print(
                        f"  ❌ PaddlePaddle error: {execution_result['paddle_error'][:100]}..."
                        if len(str(execution_result["paddle_error"])) > 100
                        else f"  ❌ PaddlePaddle error: {execution_result['paddle_error']}"
                    )
                # if execution_result['comparison_error']:
                #     print(f"  ⚠️ Comparison error: {execution_result['comparison_error']}")

                # Only count LLM-generated cases (exclude DB original)
                if is_llm_generated:
                    # Count cases where both frameworks succeeded
                    if execution_result["torch_success"] and execution_result["paddle_success"]:
                        stats["successful_cases"] += 1
                        # print(f"  📊 LLM-generated success count: {stats['successful_cases']}")

            except Exception as e:
                print(
                    f"  ❌ Fatal error when executing test case: {str(e)[:100]}..."
                    if len(str(e)) > 100
                    else f"  ❌ Fatal error when executing test case: {e}"
                )
                # print("  ❌ Error details:")
                # traceback.print_exc()

                # Create an error result
                execution_result = {
                    "status": "fatal_error",
                    "torch_success": False,
                    "paddle_success": False,
                    "results_match": False,
                    "torch_error": f"Fatal error: {str(e)}",
                    "paddle_error": None,
                    "comparison_error": None,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }

            # Save this iteration result
            iteration_result = {
                "iteration": iteration + 1,
                "torch_test_case": current_torch_test_case,
                "paddle_test_case": current_paddle_test_case,
                "execution_result": execution_result,
                "llm_operation": None,
                "is_llm_generated": is_llm_generated,
            }

            # Call LLM for repair or mutation (pass docs)
            try:
                # print("\n  🤖 Calling LLM for test case analysis...")
                llm_result = self.call_llm_for_repair_or_mutation(
                    execution_result,
                    current_torch_test_case,
                    current_paddle_test_case,
                    torch_doc,
                    paddle_doc,
                )
            except Exception as e:
                print(
                    f"  ❌ Error calling LLM: {str(e)[:100]}..."
                    if len(str(e)) > 100
                    else f"  ❌ Error calling LLM: {e}"
                )
                # print("  ❌ Error details:")
                # traceback.print_exc()
                print("  ⏭️ Skip LLM analysis and end iterations for this case")

                # Create a skip operation
                llm_result = {
                    "operation": "skip",
                    "reason": f"LLM call failed: {str(e)}",
                }

                iteration_result["llm_operation"] = llm_result
                iteration_result["case_number"] = case_number
                case_results.append(iteration_result)
                break

            operation = llm_result.get("operation", "skip")
            reason = llm_result.get("reason", "")

            # Simplified logging: only operation type
            print(f"  🤖 LLM decision: {operation}")
            # print(f"  🤖 LLM operation: {operation}")
            # print(f"  🤖 LLM reason: {reason}")

            iteration_result["llm_operation"] = {
                "operation": operation,
                "reason": reason,
            }

            # Add case number
            iteration_result["case_number"] = case_number
            case_results.append(iteration_result)

            # If LLM suggests skip, end iterations
            if operation == "skip":
                # print("  ⏭️ LLM suggests skip, ending iterations")
                break

            # Prepare next iteration test cases
            if operation == "mutation":
                # print("  🔀 Using LLM-mutated test cases")
                next_pytorch_test_case = llm_result.get("pytorch_test_case", current_torch_test_case)
                next_paddle_test_case = llm_result.get("paddle_test_case", current_paddle_test_case)
                # Count LLM-generated cases
                stats["llm_generated_cases"] += 1
                # print(f"  📊 LLM-generated case count: {stats['llm_generated_cases']}")
                is_llm_generated = True
            elif operation == "repair":
                # print("  🔧 Using LLM-repaired test cases")
                next_pytorch_test_case = llm_result.get("pytorch_test_case", current_torch_test_case)
                next_paddle_test_case = llm_result.get("paddle_test_case", current_paddle_test_case)
                # Count LLM-generated cases
                stats["llm_generated_cases"] += 1
                # print(f"  📊 LLM-generated case count: {stats['llm_generated_cases']}")
                is_llm_generated = True
            else:
                next_pytorch_test_case = current_torch_test_case
                next_paddle_test_case = current_paddle_test_case

            # Convert LLM-returned cases to executable format (share tensors)
            current_torch_test_case, current_paddle_test_case = self._convert_llm_test_cases(
                next_pytorch_test_case, next_paddle_test_case
            )

        # Fix issue: if last LLM output was mutation/repair, execute that case
        if len(case_results) > 0:
            last_iteration = case_results[-1]
            last_operation = last_iteration["llm_operation"].get("operation", "skip")

            if last_operation in ["mutation", "repair"]:
                print("\n  🔄 Executing the last LLM-generated case...")

                try:
                    # Execute the last LLM-generated test case
                    # print("  📝 Executing test case...")
                    execution_result = self.execute_test_case(
                        operator_name, paddle_api, current_torch_test_case, current_paddle_test_case
                    )

                    # Simplified logging: only key status
                    status = execution_result["status"]
                    torch_ok = "✓" if execution_result["torch_success"] else "✗"
                    paddle_ok = "✓" if execution_result["paddle_success"] else "✗"
                    match_ok = "✓" if execution_result["results_match"] else "✗"
                    print(
                        f"  📊 Final result: status={status}, PyTorch={torch_ok}, Paddle={paddle_ok}, match={match_ok}"
                    )

                    # Detailed result logging is commented out (logged elsewhere)
                    # print(f"  📊 Status: {execution_result['status']}")
                    # print(f"  🅿️ PyTorch success: {execution_result['torch_success']}")
                    # print(f"  🅰️ PaddlePaddle success: {execution_result['paddle_success']}")
                    # print(f"  ❓ Results match: {execution_result['results_match']}")

                    # Only print errors if present
                    if execution_result["torch_error"]:
                        print(
                            f"  ❌ PyTorch error: {execution_result['torch_error'][:100]}..."
                            if len(str(execution_result["torch_error"])) > 100
                            else f"  ❌ PyTorch error: {execution_result['torch_error']}"
                        )
                    if execution_result["paddle_error"]:
                        print(
                            f"  ❌ PaddlePaddle error: {execution_result['paddle_error'][:100]}..."
                            if len(str(execution_result["paddle_error"])) > 100
                            else f"  ❌ PaddlePaddle error: {execution_result['paddle_error']}"
                        )
                    # if execution_result['comparison_error']:
                    #     print(f"  ⚠️ Comparison error: {execution_result['comparison_error']}")

                    # Count cases where both frameworks succeeded (LLM-generated)
                    if execution_result["torch_success"] and execution_result["paddle_success"]:
                        stats["successful_cases"] += 1
                        # print(f"  📊 LLM-generated success count: {stats['successful_cases']}")

                    # Save this extra execution
                    final_iteration_result = {
                        "iteration": len(case_results) + 1,
                        "torch_test_case": current_torch_test_case,
                        "paddle_test_case": current_paddle_test_case,
                        "execution_result": execution_result,
                        "llm_operation": {
                            "operation": "final_execution",
                            "reason": "Execute the last LLM-generated case",
                        },
                        "case_number": case_number,
                        "is_llm_generated": True,
                    }
                    case_results.append(final_iteration_result)

                except Exception as e:
                    print(
                        f"  ❌ Fatal error executing the last LLM-generated case: {str(e)[:100]}..."
                        if len(str(e)) > 100
                        else f"  ❌ Fatal error executing the last LLM-generated case: {e}"
                    )
                    # print("  ❌ Error details:")
                    # traceback.print_exc()

                    # Record the failed attempt
                    final_iteration_result = {
                        "iteration": len(case_results) + 1,
                        "torch_test_case": current_torch_test_case,
                        "paddle_test_case": current_paddle_test_case,
                        "execution_result": {
                            "status": "fatal_error",
                            "torch_success": False,
                            "paddle_success": False,
                            "results_match": False,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        },
                        "llm_operation": {
                            "operation": "final_execution",
                            "reason": "Execute the last LLM-generated case (fatal error)",
                        },
                        "case_number": case_number,
                        "is_llm_generated": True,
                    }
                    case_results.append(final_iteration_result)

        print(f"\n  {'─'*76}")
        print(f"  ✅ Test case {case_number} completed, {len(case_results)} iterations executed")
        print(f"  {'─'*76}")

        return case_results

    def _convert_llm_test_cases(
        self, pytorch_test_case: Dict[str, Any], paddle_test_case: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Convert LLM-returned PyTorch and PaddlePaddle test cases into executable format.
        Ensure both frameworks share tensor data, while allowing other params to differ.

        Args:
            pytorch_test_case: LLM-returned PyTorch test case.
            paddle_test_case: LLM-returned PaddlePaddle test case.

        Returns:
            (converted PyTorch test case, converted PaddlePaddle test case)
        """
        # Simplified logging: detailed conversion is logged elsewhere
        # print("    🔄 Converting LLM test cases...")

        # Step 1: Collect tensor parameter names and create shared numpy arrays
        shared_tensors = {}

        # Find all tensor params (in either test case)
        all_keys = set(pytorch_test_case.keys()) | set(paddle_test_case.keys())

        for key in all_keys:
            if key == "api":
                continue

            # Check for tensor descriptions
            pytorch_value = pytorch_test_case.get(key)
            paddle_value = paddle_test_case.get(key)

            is_tensor = False
            tensor_desc = None

            if isinstance(pytorch_value, dict) and "shape" in pytorch_value:
                is_tensor = True
                tensor_desc = pytorch_value
            elif isinstance(paddle_value, dict) and "shape" in paddle_value:
                is_tensor = True
                tensor_desc = paddle_value

            if is_tensor:
                # Generate shared numpy array
                # print(f"      - {key}: shape={tensor_desc.get('shape')}, dtype={tensor_desc.get('dtype')}")
                numpy_array = self.generate_numpy_data(tensor_desc)
                shared_tensors[key] = numpy_array
                # print(f"        Shared numpy: shape={numpy_array.shape}, dtype={numpy_array.dtype}")

        # Step 2: Build PyTorch and PaddlePaddle test cases
        converted_pytorch = {}
        converted_paddle = {}

        # print("    📦 Building PyTorch test case:")
        for key, value in pytorch_test_case.items():
            if key in shared_tensors:
                converted_pytorch[key] = shared_tensors[key]
                # print(f"      - {key}: shared tensor")
            else:
                converted_pytorch[key] = value
                # print(f"      - {key}: value={value}")

        # print("    📦 Building PaddlePaddle test case:")
        for key, value in paddle_test_case.items():
            if key in shared_tensors:
                converted_paddle[key] = shared_tensors[key]
                # print(f"      - {key}: shared tensor")
            else:
                converted_paddle[key] = value
                # print(f"      - {key}: value={value}")

        return converted_pytorch, converted_paddle

    def save_results(self, operator_name: str, results: List[Dict[str, Any]], stats: Dict[str, int] = None):
        """
        Save test results to JSON.

        Args:
            operator_name: Operator name.
            results: Test results list.
            stats: Stats (LLM-generated cases and successful cases).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_enhanced_{operator_name.replace('.', '_')}_{timestamp}.json"
        filepath = os.path.join(self.result_dir, filename)

        # Prepare output data (strip numpy arrays for JSON)
        output_results = []
        for result in results:
            output_result = copy.deepcopy(result)

            # Simplify numpy arrays in old format: test_case
            if "test_case" in output_result:
                simplified_case = {}
                for key, value in output_result["test_case"].items():
                    if isinstance(value, np.ndarray):
                        simplified_case[key] = {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "sample_values": value.flatten()[:10].tolist() if value.size > 0 else [],
                        }
                    else:
                        simplified_case[key] = value
                output_result["test_case"] = simplified_case

            # Simplify numpy arrays in new format: torch_test_case / paddle_test_case
            for test_case_key in ["torch_test_case", "paddle_test_case"]:
                if test_case_key in output_result:
                    simplified_case = {}
                    for key, value in output_result[test_case_key].items():
                        if isinstance(value, np.ndarray):
                            simplified_case[key] = {
                                "shape": list(value.shape),
                                "dtype": str(value.dtype),
                                "sample_values": value.flatten()[:10].tolist() if value.size > 0 else [],
                            }
                        else:
                            simplified_case[key] = value
                    output_result[test_case_key] = simplified_case

            output_results.append(output_result)

        output_data = {
            "operator": operator_name,
            "timestamp": datetime.now().isoformat(),
            "total_iterations": len(results),
            "llm_generated_test_cases": stats.get("llm_generated_cases", 0) if stats else 0,
            "successful_test_cases": stats.get("successful_cases", 0) if stats else 0,
            "results": output_results,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Results saved to: {filepath}")

    def close(self):
        """Close the MongoDB connection."""
        self.client.close()


def main():
    """
    Main function.

    Two modes:
    1. Single operator test (uncomment mode 1, comment out mode 2)
    2. Batch test all operators (current mode)
    """
    # ==================== Test Config ====================
    max_iterations = 3  # Max iterations per test case
    num_test_cases = 3  # Test cases per operator

    # Batch range config (mode 2 only)
    # None means all operators
    # (start, end) means test operators start..end (1-based, inclusive)
    # Examples:
    #   operator_range = None          # test all operators
    #   operator_range = (1, 10)       # test 1..10
    #   operator_range = (10, 20)      # test 10..20
    #   operator_range = (50, 100)     # test 50..100
    operator_range = (301, 465)
    # ====================================================

    # ==================== Mode 1: Single operator ====================
    # To test a single operator, uncomment the triple-quoted block below
    # and comment out "Mode 2" code.
    """
    operator_name = "torch.nn.Dropout2d"  # Operator to test

    print("="*80)
    print("LLM-based PyTorch and PaddlePaddle operator comparison framework")
    print("="*80)
    print(f"📌 Operator: {operator_name}")
    print(f"📌 Iterations per case: {max_iterations}")
    print(f"📌 Test cases: {num_test_cases}")
    print("="*80)

    # Initialize comparator
    comparator = LLMEnhancedComparator()

    try:
        # Run LLM-enhanced tests
        results, stats = comparator.llm_enhanced_test_operator(
            operator_name,
            max_iterations=max_iterations,
            num_test_cases=num_test_cases,
        )

        # Save results
        comparator.save_results(operator_name, results, stats)

        # Print summary
        print("\n" + "="*80)
        print("📊 Test Summary")
        print("="*80)
        print(f"Operator: {operator_name}")
        print(f"Total iterations: {len(results)}")

        # Group by test case
        case_groups = {}
        for result in results:
            case_num = result.get("case_number", 0)
            if case_num not in case_groups:
                case_groups[case_num] = []
            case_groups[case_num].append(result)

        print(f"\nTested {len(case_groups)} cases:")
        for case_num in sorted(case_groups.keys()):
            case_results = case_groups[case_num]
            print(f"\nCase {case_num} ({len(case_results)} iterations):")
            for i, result in enumerate(case_results):
                exec_result = result["execution_result"]
                llm_op = result.get("llm_operation", {})
                print(f"  Iteration {i+1}:")
                print(f"    - Status: {exec_result['status']}")
                print(f"    - PyTorch success: {exec_result['torch_success']}")
                print(f"    - PaddlePaddle success: {exec_result['paddle_success']}")
                print(f"    - Results match: {exec_result['results_match']}")
                print(f"    - LLM operation: {llm_op.get('operation', 'N/A')}")

    finally:
        # Close connection
        comparator.close()
        print("\n✅ Program completed")
    """
    # ==================== End Mode 1 ====================

    # ==================== Mode 2: Batch operators ====================
    # To batch test all operators, uncomment the block below
    # and comment out "Mode 1".

    print("="*80)
    print("LLM-based PyTorch and PaddlePaddle operator batch comparison framework")
    print("="*80)
    print(f"📌 Iterations per operator: {max_iterations}")
    print(f"📌 Test cases per operator: {num_test_cases}")
    if operator_range is not None:
        print(f"📌 Range: operators {operator_range[0]} to {operator_range[1]}")
    else:
        print("📌 Range: all operators")
    print("="*80)

    # Initialize comparator
    comparator = LLMEnhancedComparator()

    # Record start time
    import time

    start_time = time.time()
    start_datetime = datetime.now()

    try:
        # Fetch all operators from DB
        print("\n🔍 Fetching all operators from database...")
        all_operators = list(comparator.collection.find({}, {"api": 1}))
        all_operator_names = [doc["api"] for doc in all_operators if "api" in doc]

        print(f"✅ Database has {len(all_operator_names)} operators")

        # Filter by operator_range
        if operator_range is not None:
            start_idx, end_idx = operator_range
            # Convert to 0-based indices
            start_idx = max(1, start_idx) - 1
            end_idx = min(len(all_operator_names), end_idx)
            operator_names = all_operator_names[start_idx:end_idx]
            print(f"📌 Range: operators {start_idx + 1} to {end_idx}")
            print(f"📋 Will test {len(operator_names)} operators")
        else:
            operator_names = all_operator_names
            print(f"📋 Will test all {len(operator_names)} operators")

        # Filter out operators with no Paddle implementation
        print("\n🔍 Filtering operators without PaddlePaddle implementations...")
        original_count = len(operator_names)
        filtered_operator_names = []
        skipped_operators = []

        for op_name in operator_names:
            _, pd_api, mapping_method = comparator.convert_api_name(op_name)
            if pd_api is not None:
                filtered_operator_names.append(op_name)
            else:
                skipped_operators.append((op_name, mapping_method))

        operator_names = filtered_operator_names
        skipped_count = original_count - len(operator_names)

        print(
            f"✅ Filtering done: original {original_count}, skipped {skipped_count}, remaining {len(operator_names)}"
        )
        if skipped_operators:
            skipped_preview = ", ".join([f"{op}({reason})" for op, reason in skipped_operators[:10]])
            print(f"⏭️ Skipped operators (first 10): {skipped_preview}{'...' if len(skipped_operators) > 10 else ''}")

        print(
            f"📋 Operator list: {', '.join(operator_names[:10])}{'...' if len(operator_names) > 10 else ''}\n"
        )

        # Summary across all operators
        all_operators_summary = []

        # Create master log file
        batch_log_file = os.path.join(
            comparator.result_dir, f"batch_test_log_{start_datetime.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        log_file = open(batch_log_file, "w", encoding="utf-8")

        # Log header
        log_file.write("=" * 80 + "\n")
        log_file.write("Batch Test Master Log\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("Test config:\n")
        log_file.write(f"  - Max iterations per operator: {max_iterations}\n")
        log_file.write(f"  - Test cases per operator: {num_test_cases}\n")
        log_file.write(f"  - Total operators in DB: {len(all_operator_names)}\n")
        if operator_range is not None:
            log_file.write(f"  - Range: {operator_range[0]} to {operator_range[1]}\n")
        log_file.write(f"  - Skipped (no implementation): {skipped_count}\n")
        log_file.write(f"  - Operators tested: {len(operator_names)}\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.flush()

        # Test each operator
        for idx, operator_name in enumerate(operator_names, 1):
            print("\n" + "🔷" * 40)
            print(f"🎯 [{idx}/{len(operator_names)}] Testing operator: {operator_name}")
            print("🔷" * 40)

            try:
                # Run LLM-enhanced tests
                results, stats = comparator.llm_enhanced_test_operator(
                    operator_name,
                    max_iterations=max_iterations,
                    num_test_cases=num_test_cases,
                )

                # Save results
                if results:
                    comparator.save_results(operator_name, results, stats)

                    # Record summary
                    all_operators_summary.append(
                        {
                            "operator": operator_name,
                            "total_iterations": len(results),
                            "llm_generated_cases": stats.get("llm_generated_cases", 0),
                            "successful_cases": stats.get("successful_cases", 0),
                            "status": "completed",
                        }
                    )

                    print(f"\n✅ Operator {operator_name} completed")
                    print(f"   - Total iterations: {len(results)}")
                    print(f"   - LLM-generated cases: {stats.get('llm_generated_cases', 0)}")
                    print(f"   - Successful cases: {stats.get('successful_cases', 0)}")

                    # Write log
                    log_file.write(f"[{idx}/{len(operator_names)}] {operator_name}\n")
                    log_file.write("  Status: ✅ Completed\n")
                    log_file.write(f"  Total iterations: {len(results)}\n")
                    log_file.write(f"  LLM-generated cases: {stats.get('llm_generated_cases', 0)}\n")
                    log_file.write(f"  Successful cases: {stats.get('successful_cases', 0)}\n")
                    if stats.get("llm_generated_cases", 0) > 0:
                        success_rate = (
                            stats.get("successful_cases", 0) / stats.get("llm_generated_cases", 0)
                        ) * 100
                        log_file.write(f"  Success rate: {success_rate:.2f}%\n")
                    log_file.write("\n")
                    log_file.flush()
                else:
                    all_operators_summary.append(
                        {
                            "operator": operator_name,
                            "total_iterations": 0,
                            "llm_generated_cases": 0,
                            "successful_cases": 0,
                            "status": "no_results",
                        }
                    )
                    print(f"\n⚠️ Operator {operator_name} produced no results")

                    # Write log
                    log_file.write(f"[{idx}/{len(operator_names)}] {operator_name}\n")
                    log_file.write("  Status: ⚠️ No results\n\n")
                    log_file.flush()

            except Exception as e:
                print(f"\n❌ Operator {operator_name} failed: {e}")
                all_operators_summary.append(
                    {
                        "operator": operator_name,
                        "total_iterations": 0,
                        "llm_generated_cases": 0,
                        "successful_cases": 0,
                        "status": "failed",
                        "error": str(e),
                    }
                )

                # Write log
                log_file.write(f"[{idx}/{len(operator_names)}] {operator_name}\n")
                log_file.write("  Status: ❌ Failed\n")
                log_file.write(f"  Error: {str(e)}\n\n")
                log_file.flush()

                # Continue to next operator
                continue

        # Calculate runtime
        end_time = time.time()
        end_datetime = datetime.now()
        total_duration = end_time - start_time
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)

        # Print overall summary
        print("\n" + "=" * 80)
        print("📊 Batch Test Summary")
        print("=" * 80)
        print(f"Total operators: {len(operator_names)}")

        completed_count = sum(1 for s in all_operators_summary if s["status"] == "completed")
        failed_count = sum(1 for s in all_operators_summary if s["status"] == "failed")
        no_results_count = sum(1 for s in all_operators_summary if s["status"] == "no_results")

        print(f"✅ Completed: {completed_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"⚠️ No results: {no_results_count}")

        total_llm_cases = sum(s["llm_generated_cases"] for s in all_operators_summary)
        total_successful_cases = sum(s["successful_cases"] for s in all_operators_summary)
        total_iterations = sum(s["total_iterations"] for s in all_operators_summary)

        print("\n📈 Stats:")
        print(f"   - Total LLM-generated cases: {total_llm_cases}")
        print(f"   - Total successful cases: {total_successful_cases}")
        if total_llm_cases > 0:
            success_rate = (total_successful_cases / total_llm_cases) * 100
            print(f"   - Success rate: {success_rate:.2f}%")
        print(f"   - Total iterations: {total_iterations}")
        print(f"\n⏱️ Runtime: {hours}h {minutes}m {seconds}s")

        # Write summary to log
        log_file.write("=" * 80 + "\n")
        log_file.write("Overall Summary\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(
            f"Total runtime: {hours}h {minutes}m {seconds}s ({total_duration:.2f} seconds)\n\n"
        )

        log_file.write("Operator results:\n")
        log_file.write(f"  - Total operators: {len(operator_names)}\n")
        log_file.write(f"  - Completed: {completed_count} ({completed_count/len(operator_names)*100:.2f}%)\n")
        log_file.write(f"  - Failed: {failed_count} ({failed_count/len(operator_names)*100:.2f}%)\n")
        log_file.write(f"  - No results: {no_results_count} ({no_results_count/len(operator_names)*100:.2f}%)\n\n")

        log_file.write("LLM-generated case stats:\n")
        log_file.write(f"  - Total LLM-generated cases: {total_llm_cases}\n")
        log_file.write(f"  - Successful cases: {total_successful_cases}\n")
        if total_llm_cases > 0:
            success_rate = (total_successful_cases / total_llm_cases) * 100
            log_file.write(f"  - Success rate: {success_rate:.2f}%\n")
        log_file.write(f"  - Total iterations: {total_iterations}\n")
        if completed_count > 0:
            avg_llm_cases = total_llm_cases / completed_count
            avg_successful = total_successful_cases / completed_count
            log_file.write(f"  - Avg LLM cases per operator: {avg_llm_cases:.2f}\n")
            log_file.write(f"  - Avg successful cases per operator: {avg_successful:.2f}\n")

        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("See individual operator logs for details\n")
        log_file.write("=" * 80 + "\n")
        log_file.close()

        print(f"\n💾 Master log saved to: {batch_log_file}")

        # Save overall summary to JSON
        summary_file = os.path.join(
            comparator.result_dir, f"batch_test_summary_{start_datetime.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "test_config": {
                        "max_iterations": max_iterations,
                        "num_test_cases": num_test_cases,
                        "operator_range": f"{operator_range[0]}-{operator_range[1]}" if operator_range else "all",
                        "total_operators_in_db": len(all_operator_names),
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
                        "total_successful_cases": total_successful_cases,
                        "success_rate": f"{(total_successful_cases / total_llm_cases * 100):.2f}%"
                        if total_llm_cases > 0
                        else "N/A",
                        "total_iterations": total_iterations,
                        "avg_llm_cases_per_operator": f"{total_llm_cases / completed_count:.2f}"
                        if completed_count > 0
                        else "N/A",
                        "avg_successful_per_operator": f"{total_successful_cases / completed_count:.2f}"
                        if completed_count > 0
                        else "N/A",
                    },
                    "operators": all_operators_summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"💾 JSON summary saved to: {summary_file}")

    finally:
        # Close connection
        comparator.close()
        print("\n✅ Batch test completed")

    # ==================== End Mode 2 ====================


if __name__ == "__main__":
    main()
