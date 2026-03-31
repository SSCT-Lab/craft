#!/usr/bin/env python3
"""
PyTorch and PaddlePaddle operator comparison test framework.
Compare whether the first 20 operators execute consistently across both frameworks.
"""

import pymongo
import torch
import paddle
import numpy as np
import pandas as pd
import re
import copy
from typing import Dict, List, Any, Tuple, Optional
import traceback
import json
from datetime import datetime
import os
from collections import defaultdict

class PyTorchPaddleComparator:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "freefuzz-torch"):
        """
        Initialize PyTorch and PaddlePaddle comparator.
        
        Args:
            mongo_uri: MongoDB connection URI
            db_name: Database name
        """
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["argVS"]
        
        # Load API mapping table
        self.api_mapping = self.load_api_mapping()
        
        # Create result directory (absolute path)
        self.result_dir = os.path.abspath("pt_pd_log")
        os.makedirs(self.result_dir, exist_ok=True)
        print(f"📁 Result directory: {self.result_dir}")
        
        # Result stats
        self.comparison_results = []
        
        # Deprecated PyTorch operator list
        self.deprecated_torch_apis = {
            "torch.symeig": "Removed in PyTorch 1.9; use torch.linalg.eigh instead"
        }
        
        # Fixed random seed for reproducibility
        self.random_seed = 42
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        paddle.seed(self.random_seed)
        
    def load_api_mapping(self) -> Dict[str, Dict[str, str]]:
        """Load PyTorch-to-PaddlePaddle API mapping table."""
        mapping_file = "api_mapping/pt_pd_mapping.csv"
        try:
            df = pd.read_csv(mapping_file)
            mapping = {}
            
            for _, row in df.iterrows():
                pt_api = str(row["PyTorch APIs"]).strip()
                pd_api = str(row["PaddlePaddle APIs"]).strip()
                note = str(row.get("说明", "")).strip()
                mapping[pt_api] = {"pd_api": pd_api, "note": note}
            
            print(f"✅ API mapping table loaded: {len(mapping)} entries")
            return mapping
        except Exception as e:
            print(f"❌ Failed to load API mapping table: {e}")
            return {}
    
    def is_class_based_api(self, api_name: str) -> bool:
        """
        Determine whether an API is class-based (by checking for uppercase letters).
        Example: torch.nn.Dropout2d, torch.nn.AvgPool2d
        """
        # Get the last part of the API (function/class name)
        parts = api_name.split(".")
        if len(parts) >= 2:
            name = parts[-1]
            # Check whether it contains uppercase letters
            return any(c.isupper() for c in name)
        return False
    
    def convert_class_to_functional(self, torch_api: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Convert class-based API to functional API.
        Example: torch.nn.Dropout2d -> torch.nn.functional.dropout2d
                 torch.nn.AvgPool2d -> torch.nn.functional.avg_pool2d
        Returns (torch_functional_api, paddle_functional_api)
        """
        if not self.is_class_based_api(torch_api):
            return None, None
        
        parts = torch_api.split(".")
        if len(parts) >= 3 and parts[1] == "nn":
            # Convert class name to snake_case
            class_name = parts[-1]
            
            # Convert CamelCase to snake_case
            # Example: AvgPool2d -> avg_pool2d, Dropout2d -> dropout2d
            # Step 1: insert underscore before uppercase letters (not at start or after digits)
            # (?<!^) means not at start
            # (?<![0-9]) means previous is not a digit
            # ([A-Z]) matches uppercase letters
            func_name = re.sub(r'(?<!^)(?<![0-9])([A-Z])', r'_\1', class_name).lower()
            
            # Build torch functional API
            torch_func_api = f"torch.nn.functional.{func_name}"
            
            # Build paddle functional API (keep lowercase)
            paddle_func_api = f"paddle.nn.functional.{func_name}"
            
            return torch_func_api, paddle_func_api
        
        return None, None
    
    def convert_api_name(self, torch_api: str) -> Tuple[Optional[str], str]:
        """
        Convert PyTorch API to PaddlePaddle API.
        Returns (converted API name, method used)
        """
        # 0. If class-based, convert to functional
        if self.is_class_based_api(torch_api):
            torch_func, paddle_func = self.convert_class_to_functional(torch_api)
            if torch_func and paddle_func:
                # Validate functions exist
                torch_func_obj = self.get_operator_function(torch_func, "torch")
                paddle_func_obj = self.get_operator_function(paddle_func, "paddle")
                if torch_func_obj and paddle_func_obj:
                    return paddle_func, "Class-to-function"
        
        # 1. Check mapping table first
        if torch_api in self.api_mapping:
            note = self.api_mapping[torch_api]["note"]
            pd_api = self.api_mapping[torch_api]["pd_api"]
            
            # Check whether implementation exists
            if "无对应实现" in note:
                return None, "No implementation"
            elif "功能一致" in note:
                return pd_api, "Mapping table (equivalent)"
            else:
                return pd_api, "Mapping table (differences)"
        
        # 2. Default conversion rule (no lowercase-to-uppercase transform)
        api = torch_api.replace("torch", "paddle", 1)
        
        return api, "Name conversion"
    
    def convert_dtype(self, torch_dtype_str: str) -> str:
        """Convert torch dtype string to paddle dtype string."""
        if isinstance(torch_dtype_str, str):
            if torch_dtype_str.startswith("torch."):
                return torch_dtype_str.replace("torch.", "paddle.")
        return torch_dtype_str
    
    def convert_key(self, key: str, paddle_api: str = "") -> str:
        """Convert parameter name."""
        # Common parameter mapping
        key_mapping = {
            "input": "x",
            "other": "y",
            # "n": "num_rows",      # paddle.eye (needed in 3.1, alias supported in 3.2+)
            # "m": "num_columns"    # paddle.eye (needed in 3.1, alias supported in 3.2+)
        }
        
        # API-specific mapping
        if paddle_api == "paddle.nn.functional.avg_pool2d":
            if key == "count_include_pad":
                return "exclusive"  # PaddlePaddle uses exclusive
        
        return key_mapping.get(key, key)
    
    def should_skip_param(self, key: str, paddle_api: str) -> bool:
        """Check whether a parameter should be skipped (unsupported/incompatible)."""
        # Common unsupported parameters
        common_skip_params = ["layout", "requires_grad", "out"]
        
        # API-specific unsupported parameters
        skip_params = {
            "paddle.nn.functional.selu": ["inplace"],
            "paddle.nn.functional.avg_pool2d": ["divisor_override"],  # PaddlePaddle does not support divisor_override
            # "paddle.eye": ["device"],  # Some versions do not support device
        }
        
        # Check common skip params
        if key in common_skip_params:
            return True
        
        # Check API-specific skip params
        if paddle_api in skip_params:
            return key in skip_params[paddle_api]
        
        return False
    
    def should_convert_scalar_to_tensor(self, key: str, paddle_api: str, value: Any) -> bool:
        """Check whether a scalar should be converted to tensor."""
        # If value is not scalar, no conversion
        if not isinstance(value, (int, float)):
            return False
        
        # APIs/params requiring scalar -> tensor conversion
        scalar_to_tensor_apis = {
            "paddle.floor_divide": ["other", "y"],
            "paddle.remainder": ["other", "y"],
            "paddle.fmod": ["other", "y"],
            "paddle.pow": ["other", "y"],
            "paddle.atan2": ["other", "y"],
        }
        
        if paddle_api in scalar_to_tensor_apis:
            return key in scalar_to_tensor_apis[paddle_api]
        
        return False
    
    def convert_scalar_to_paddle_tensor(self, value: Any) -> paddle.Tensor:
        """Convert scalar to PaddlePaddle tensor."""
        if isinstance(value, (int, float)):
            return paddle.to_tensor(value)
        else:
            return paddle.to_tensor([value])
    
    def get_default_param_value(self, param_name: str, api_name: str) -> Any:
        """Get default parameter value."""
        # API-specific default values
        api_defaults = {
            "torch.nn.AvgPool2d": {
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            "torch.nn.functional.avg_pool2d": {
                "ceil_mode": False,
                "count_include_pad": True,
                "divisor_override": None
            },
            "torch.nn.MaxPool2d": {
                "ceil_mode": False,
                "dilation": 1,
                "return_indices": False
            },
            "torch.nn.functional.max_pool2d": {
                "ceil_mode": False,
                "dilation": 1,
                "return_indices": False
            }
        }
        
        if api_name in api_defaults:
            return api_defaults[api_name].get(param_name)
        
        return None
    
    def convert_param_for_paddle(self, param_name: str, value: Any, paddle_api: str) -> Any:
        """Convert special parameter values for PaddlePaddle."""        
        # Handle boolean parameter name differences
        if paddle_api == "paddle.nn.functional.avg_pool2d":
            if param_name == "exclusive":  # param_name has already been converted by convert_key
                # PaddlePaddle exclusive = not count_include_pad
                if isinstance(value, bool):
                    return not value  # exclusive = not count_include_pad
        
        return value
    
    def generate_numpy_data(self, data: Any) -> np.ndarray:
        """Generate numpy array as shared data source for both frameworks."""
        if isinstance(data, dict):
            dtype_map = {
                "torch.float64": np.float64,
                "torch.float32": np.float32,
                "torch.int64": np.int64,
                "torch.int32": np.int32,
                "torch.bool": np.bool_,
                "torch.uint8": np.uint8
            }
            
            shape = data.get("shape", [])
            dtype_str = data.get("dtype", "torch.float32")
            dtype = dtype_map.get(dtype_str, np.float32)
            
            if shape:
                if dtype == np.bool_:
                    return np.random.randint(0, 2, shape).astype(np.bool_)
                elif dtype in [np.int64, np.int32]:
                    return np.random.randint(-10, 10, shape).astype(dtype)
                else:
                    return np.random.randn(*shape).astype(dtype)
            else:
                # Scalar
                if dtype == np.bool_:
                    return np.array(True, dtype=np.bool_)
                elif dtype in [np.int64, np.int32]:
                    return np.array(1, dtype=dtype)
                else:
                    return np.array(1.0, dtype=dtype)
        elif isinstance(data, (int, float)):
            return np.array(data)
        elif isinstance(data, list):
            return np.array(data)
        else:
            return np.array(data)
    
    def convert_to_tensor_torch(self, data: Any, numpy_data: np.ndarray = None) -> torch.Tensor:
        """
        Convert data to PyTorch tensor.
        If numpy_data is provided, convert from it to keep data consistent with PaddlePaddle.
        """
        if numpy_data is not None:
            # Convert from numpy to keep data consistent
            return torch.from_numpy(numpy_data.copy())
        
            # Compatibility path (generate new data)
        if isinstance(data, dict):
            # Generate numpy data then convert
            numpy_data = self.generate_numpy_data(data)
            return torch.from_numpy(numpy_data.copy())
        elif isinstance(data, (int, float)):
            return torch.tensor(data)
        elif isinstance(data, list):
            return torch.tensor(data)
        else:
            return torch.tensor(data)
    
    def convert_to_tensor_paddle(self, data: Any, numpy_data: np.ndarray = None) -> paddle.Tensor:
        """
        Convert data to PaddlePaddle tensor.
        If numpy_data is provided, convert from it to keep data consistent with PyTorch.
        """
        if numpy_data is not None:
            # Convert from numpy to keep data consistent
            return paddle.to_tensor(numpy_data.copy())
        
            # Compatibility path (generate new data)
        if isinstance(data, dict):
            # Generate numpy data then convert
            numpy_data = self.generate_numpy_data(data)
            return paddle.to_tensor(numpy_data.copy())
        elif isinstance(data, (int, float)):
            return paddle.to_tensor(data)
        elif isinstance(data, list):
            return paddle.to_tensor(data)
        else:
            return paddle.to_tensor(data)
    
    def prepare_shared_numpy_data(self, document: Dict[str, Any], case_index: int) -> Dict[str, Any]:
        """
        Prepare shared numpy data so PyTorch and PaddlePaddle use the same inputs.
        Returns dict: param name -> numpy array.
        """
        shared_data = {}
        api_name = document.get("api", "")
        
        # For class-based APIs without input, generate default input
        if self.is_class_based_api(api_name) and "input" not in document:
            # Generate default input shape based on API type
            if "2d" in api_name.lower() or "2D" in api_name:
                # 2D op: 4D tensor (batch, channel, height, width)
                default_shape = {"shape": [2, 3, 4, 4], "dtype": "torch.float32"}
            elif "1d" in api_name.lower() or "1D" in api_name:
                # 1D op: 3D tensor (batch, channel, length)
                default_shape = {"shape": [2, 3, 10], "dtype": "torch.float32"}
            elif "3d" in api_name.lower() or "3D" in api_name:
                # 3D op: 5D tensor (batch, channel, depth, height, width)
                default_shape = {"shape": [2, 3, 4, 4, 4], "dtype": "torch.float32"}
            else:
                # Default: 2D tensor
                default_shape = {"shape": [2, 3], "dtype": "torch.float32"}
            
            shared_data["input"] = self.generate_numpy_data(default_shape)
        
        # Handle *size param (no numpy conversion needed)
        if "*size" in document:
            size_data = document["*size"]
            if isinstance(size_data, list) and len(size_data) > case_index:
                shared_data["*size"] = size_data[case_index]
        
        # Handle *tensors param
        if "*tensors" in document:
            tensors_data = document["*tensors"]
            if isinstance(tensors_data, list) and len(tensors_data) > case_index:
                tensor_list = tensors_data[case_index]
                if isinstance(tensor_list, list):
                    shared_data["*tensors"] = [self.generate_numpy_data(t) for t in tensor_list]
                else:
                    shared_data["*tensors"] = [self.generate_numpy_data(tensor_list)]
        
        # Handle other tensor params
        for param_name in ["condition", "x", "y", "input", "other"]:
            if param_name in document:
                param_data = document[param_name]
                if isinstance(param_data, list) and len(param_data) > case_index:
                    param_value = param_data[case_index]
                    if isinstance(param_value, dict):
                        shared_data[param_name] = self.generate_numpy_data(param_value)
                    else:
                        # For scalars, store directly and handle later in prepare_arguments
                        shared_data[param_name] = param_value
        
        # Handle other params
        exclude_keys = ["_id", "api", "condition", "x", "y", "input", "other", "*size", "*tensors", "tensors", "out", "eigenvectors", "upper"]
        for key, value in document.items():
            if key not in exclude_keys:
                if isinstance(value, list):
                    if len(value) > 0:
                        idx = min(case_index, len(value) - 1)
                        param_value = value[idx]
                        if isinstance(param_value, dict):
                            shared_data[key] = self.generate_numpy_data(param_value)
                        else:
                            shared_data[key] = param_value
                    else:
                        # Handle empty list: use API-specific default
                        default_value = self.get_default_param_value(key, api_name)
                        if default_value is not None:
                            shared_data[key] = default_value
                else:
                    shared_data[key] = value
        
        return shared_data
    
    def prepare_arguments_torch(self, document: Dict[str, Any], case_index: int, shared_data: Dict[str, Any] = None) -> Tuple[List[Any], Dict[str, Any]]:
        """Prepare arguments for PyTorch."""
        args = []
        kwargs = {}
        api_name = document.get("api", "")
        
        # Generate shared data if not provided
        if shared_data is None:
            shared_data = self.prepare_shared_numpy_data(document, case_index)
        
        # Handle *size param
        if "*size" in shared_data:
            size_value = shared_data["*size"]
            if isinstance(size_value, list):
                if size_value:
                    args.append(tuple(size_value))
                else:
                    args.append(())
            elif isinstance(size_value, int):
                args.append((size_value,))
            else:
                args.append(size_value)
        
        # Handle *tensors param
        if "*tensors" in shared_data:
            for numpy_tensor in shared_data["*tensors"]:
                args.append(self.convert_to_tensor_torch(None, numpy_tensor))
        
        # Handle other tensor params
        for param_name in ["condition", "x", "y", "input", "other"]:
            if param_name in shared_data:
                value = shared_data[param_name]
                if isinstance(value, np.ndarray):
                    tensor = self.convert_to_tensor_torch(None, value)
                    args.append(tensor)
                else:
                    # PyTorch can use scalars directly
                    args.append(value)
        
        # Handle other params
        for key, value in shared_data.items():
            if key not in ["*size", "*tensors", "condition", "x", "y", "input", "other"]:
                if isinstance(value, np.ndarray):
                    kwargs[key] = self.convert_to_tensor_torch(None, value)
                else:
                    kwargs[key] = value
        
        # Convert dtype and device params
        kwargs = self.convert_dtype_device_params_torch(kwargs)
        
        return args, kwargs
    
    def prepare_arguments_paddle(self, document: Dict[str, Any], case_index: int, paddle_api: str, shared_data: Dict[str, Any] = None) -> Tuple[List[Any], Dict[str, Any]]:
        """Prepare arguments for PaddlePaddle."""
        args = []
        kwargs = {}
        
        # Generate shared data if not provided
        if shared_data is None:
            shared_data = self.prepare_shared_numpy_data(document, case_index)
        
        # Handle *size param
        if "*size" in shared_data:
            size_value = shared_data["*size"]
            if isinstance(size_value, list):
                if size_value:
                    args.append(tuple(size_value))
                else:
                    args.append(())
            elif isinstance(size_value, int):
                args.append((size_value,))
            else:
                args.append(size_value)
        
        # Handle *tensors param
        if "*tensors" in shared_data:
            for numpy_tensor in shared_data["*tensors"]:
                args.append(self.convert_to_tensor_paddle(None, numpy_tensor))
        
        # Handle other tensor params
        for param_name in ["condition", "x", "y", "input", "other"]:
            if param_name in shared_data:
                value = shared_data[param_name]
                if isinstance(value, np.ndarray):
                    tensor = self.convert_to_tensor_paddle(None, value)
                    args.append(tensor)
                else:
                    # For scalars, check whether conversion is needed
                    if self.should_convert_scalar_to_tensor(param_name, paddle_api, value):
                        tensor = self.convert_scalar_to_paddle_tensor(value)
                        args.append(tensor)
                    else:
                        args.append(value)
        
        # Handle other params
        for key, value in shared_data.items():
            if key not in ["*size", "*tensors", "condition", "x", "y", "input", "other"]:
                # Skip unsupported params
                if self.should_skip_param(key, paddle_api):
                    continue
                
                # Convert parameter name
                new_key = self.convert_key(key, paddle_api)
                if isinstance(value, np.ndarray):
                    kwargs[new_key] = self.convert_to_tensor_paddle(None, value)
                else:
                    # Some PaddlePaddle ops require all params to be tensors
                    if self.should_convert_scalar_to_tensor(key, paddle_api, value):
                        # Convert scalar to tensor
                        kwargs[new_key] = self.convert_scalar_to_paddle_tensor(value)
                    else:
                        # Handle PaddlePaddle special param values
                        converted_value = self.convert_param_for_paddle(key, value, paddle_api)
                        kwargs[new_key] = converted_value
        
        # Convert dtype and device params
        kwargs = self.convert_dtype_device_params_paddle(kwargs)
        
        return args, kwargs
    
    def convert_dtype_device_params_torch(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert PyTorch dtype and device params."""
        if "dtype" in kwargs:
            dtype_value = kwargs["dtype"]
            if isinstance(dtype_value, str):
                if dtype_value == "torchdtype":
                    kwargs["dtype"] = torch.float32
                else:
                    dtype_map = {
                        "torch.float32": torch.float32,
                        "torch.float64": torch.float64,
                        "torch.int32": torch.int32,
                        "torch.int64": torch.int64,
                        "torch.bool": torch.bool,
                        "torch.uint8": torch.uint8
                    }
                    kwargs["dtype"] = dtype_map.get(dtype_value, torch.float32)
            elif isinstance(dtype_value, int):
                int_dtype_map = {
                    0: torch.float32, 1: torch.float64, 2: torch.int32,
                    3: torch.int64, 4: torch.bool, 5: torch.uint8, 8: torch.float32,
                }
                kwargs["dtype"] = int_dtype_map.get(dtype_value, torch.float32)
        
        if "device" in kwargs:
            device_str = kwargs["device"]
            if isinstance(device_str, str):
                if device_str == "cpu":
                    kwargs["device"] = torch.device("cpu")
                elif device_str.startswith("cuda"):
                    kwargs["device"] = torch.device(device_str)
        
        return kwargs
    
    def convert_dtype_device_params_paddle(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert PaddlePaddle dtype and device params."""
        if "dtype" in kwargs:
            dtype_value = kwargs["dtype"]
            if isinstance(dtype_value, str):
                if dtype_value == "torchdtype":
                    kwargs["dtype"] = paddle.float32
                else:
                    # Convert torch dtype to paddle dtype
                    torch_dtype = dtype_value
                    if torch_dtype.startswith("torch."):
                        paddle_dtype = torch_dtype.replace("torch.", "paddle.")
                        dtype_map = {
                            "paddle.float32": paddle.float32,
                            "paddle.float64": paddle.float64,
                            "paddle.int32": paddle.int32,
                            "paddle.int64": paddle.int64,
                            "paddle.bool": paddle.bool,
                            "paddle.uint8": paddle.uint8
                        }
                        kwargs["dtype"] = dtype_map.get(paddle_dtype, paddle.float32)
                    else:
                        kwargs["dtype"] = paddle.float32
            elif isinstance(dtype_value, int):
                int_dtype_map = {
                    0: paddle.float32, 1: paddle.float64, 2: paddle.int32,
                    3: paddle.int64, 4: paddle.bool, 5: paddle.uint8, 8: paddle.float32,
                }
                kwargs["dtype"] = int_dtype_map.get(dtype_value, paddle.float32)
        
        # PaddlePaddle uses string device names
        if "device" in kwargs:
            device_str = kwargs["device"]
            if isinstance(device_str, str):
                kwargs["device"] = device_str  # paddle uses string directly
        
        return kwargs
    
    def get_operator_function(self, api_name: str, framework: str = "torch"):
        """Get operator function."""
        try:
            parts = api_name.split(".")
            if len(parts) >= 2:
                if framework == "torch" and parts[0] == "torch":
                    if len(parts) == 2:
                        return getattr(torch, parts[1])
                    elif len(parts) == 3:
                        module = getattr(torch, parts[1])
                        return getattr(module, parts[2])
                    elif len(parts) == 4:
                        module1 = getattr(torch, parts[1])
                        module2 = getattr(module1, parts[2])
                        return getattr(module2, parts[3])
                elif framework == "paddle" and parts[0] == "paddle":
                    if len(parts) == 2:
                        return getattr(paddle, parts[1])
                    elif len(parts) == 3:
                        module = getattr(paddle, parts[1])
                        return getattr(module, parts[2])
                    elif len(parts) == 4:
                        module1 = getattr(paddle, parts[1])
                        module2 = getattr(module1, parts[2])
                        return getattr(module2, parts[3])
            return None
        except AttributeError:
            return None
    
    def compare_tensors(self, torch_result, paddle_result, tolerance: float = 1e-5) -> Tuple[bool, str]:
        """Compare two tensors for equality."""
        try:
            # Convert to numpy for comparison
            if hasattr(torch_result, 'detach'):
                torch_np = torch_result.detach().cpu().numpy()
            else:
                torch_np = np.array(torch_result)
            
            if hasattr(paddle_result, 'numpy'):
                paddle_np = paddle_result.numpy()
            else:
                paddle_np = np.array(paddle_result)
            
            # Check shape
            if torch_np.shape != paddle_np.shape:
                return False, f"Shape mismatch: PyTorch {torch_np.shape} vs PaddlePaddle {paddle_np.shape}"
            
            # Check for bool or non-numeric dtypes
            if torch_np.dtype == np.bool_ or paddle_np.dtype == np.bool_:
                # For bools, compare directly
                if np.array_equal(torch_np, paddle_np):
                    return True, "Boolean values match"
                else:
                    diff_count = np.sum(torch_np != paddle_np)
                    return False, f"Boolean mismatch, diff count: {diff_count}"
            
            # Check for object or non-numeric dtypes
            if not np.issubdtype(torch_np.dtype, np.number) or not np.issubdtype(paddle_np.dtype, np.number):
                # For non-numeric, use exact comparison
                if np.array_equal(torch_np, paddle_np):
                    return True, "Values match"
                else:
                    return False, f"Value mismatch (dtype: torch={torch_np.dtype}, paddle={paddle_np.dtype})"
            
            # Check numeric values
            if np.allclose(torch_np, paddle_np, atol=tolerance, rtol=tolerance, equal_nan=True):
                return True, "Numeric values match"
            else:
                max_diff = np.max(np.abs(torch_np - paddle_np))
                return False, f"Numeric mismatch, max diff: {max_diff}"
        
        except Exception as e:
            return False, f"Comparison error: {str(e)}"
    
    def test_single_case(self, document: Dict[str, Any], case_index: int) -> Dict[str, Any]:
        """Test a single case."""
        torch_api = document.get("api", "unknown")
        test_id = str(document.get("_id", "unknown"))
        
        # If class-based, convert to functional
        original_torch_api = torch_api
        if self.is_class_based_api(torch_api):
            torch_func, _ = self.convert_class_to_functional(torch_api)
            if torch_func:
                # Verify function exists
                torch_func_obj = self.get_operator_function(torch_func, "torch")
                if torch_func_obj:
                    torch_api = torch_func
                    print(f"    🔄 Class-to-function: {original_torch_api} -> {torch_api}")
        
        result = {
            "test_id": test_id,
            "torch_api": original_torch_api,  # Keep original API name for logging
            "torch_api_used": torch_api,  # Actual API name used
            "case_index": case_index + 1,
            "status": "unknown",
            "paddle_api": None,
            "mapping_method": None,
            "torch_success": False,
            "paddle_success": False,
            "results_match": False,
            "torch_error": None,
            "paddle_error": None,
            "comparison_error": None,
            "torch_shape": None,
            "paddle_shape": None,
            "torch_dtype": None,
            "paddle_dtype": None
        }
        
        # Check deprecated operators
        if torch_api in self.deprecated_torch_apis:
            result["status"] = "deprecated"
            result["torch_error"] = self.deprecated_torch_apis[torch_api]
            result["paddle_error"] = "Corresponding PyTorch operator is deprecated, skipping"
            return result
        
        # Get corresponding PaddlePaddle API
        paddle_api, mapping_method = self.convert_api_name(torch_api)
        result["paddle_api"] = paddle_api
        result["mapping_method"] = mapping_method
        
        if paddle_api is None:
            result["status"] = "no_paddle_equivalent"
            return result
        
        # Generate shared numpy data for consistent inputs
        shared_data = self.prepare_shared_numpy_data(document, case_index)
        
        # Test PyTorch
        torch_result = None
        try:
            torch_func = self.get_operator_function(torch_api, "torch")
            if torch_func is None:
                result["torch_error"] = f"PyTorch operator {torch_api} not found"
            else:
                args, kwargs = self.prepare_arguments_torch(document, case_index, shared_data)
                with torch.no_grad():
                    torch_result = torch_func(*args, **kwargs)
                    result["torch_success"] = True
                    result["torch_shape"] = list(torch_result.shape) if hasattr(torch_result, 'shape') else None
                    result["torch_dtype"] = str(torch_result.dtype) if hasattr(torch_result, 'dtype') else None
        except Exception as e:
            error_msg = str(e)
            result["torch_error"] = error_msg
            
            # Check for deprecated function error
            if "deprecated" in error_msg.lower() or "removed" in error_msg.lower():
                if torch_api not in self.deprecated_torch_apis:
                    # Add to deprecated list dynamically
                    self.deprecated_torch_apis[torch_api] = f"Deprecated at runtime: {error_msg[:100]}..."
                result["status"] = "deprecated"
                result["paddle_error"] = "Corresponding PyTorch operator is deprecated, skipping"
                return result
        
        # Test PaddlePaddle (use same shared data)
        paddle_result = None
        try:
            paddle_func = self.get_operator_function(paddle_api, "paddle")
            if paddle_func is None:
                result["paddle_error"] = f"PaddlePaddle operator {paddle_api} not found"
            else:
                args, kwargs = self.prepare_arguments_paddle(document, case_index, paddle_api, shared_data)
                paddle_result = paddle_func(*args, **kwargs)
                result["paddle_success"] = True
                result["paddle_shape"] = list(paddle_result.shape) if hasattr(paddle_result, 'shape') else None
                result["paddle_dtype"] = str(paddle_result.dtype) if hasattr(paddle_result, 'dtype') else None
        except Exception as e:
            result["paddle_error"] = str(e)
        
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
    
    def get_num_test_cases(self, document: Dict[str, Any]) -> int:
        """Get test case count in the document."""
        max_len = 0
        for key, value in document.items():
            if key not in ["_id", "api"] and isinstance(value, list):
                max_len = max(max_len, len(value))
        return max_len if max_len > 0 else 1
    
    def compare_first_n_operators(self, n: int = 20) -> List[Dict[str, Any]]:
        """Compare the first N operators."""
        print(f"🚀 Starting PyTorch vs PaddlePaddle operator comparison (first {n} operators)...")
        print(f"📊 Connected to MongoDB: {self.client.address}")
        
        # Get first N documents
        documents = list(self.collection.find().limit(n))
        
        print(f"📋 Found {len(documents)} operators:")
        total_cases = 0
        for i, doc in enumerate(documents):
            api_name = doc.get("api", "unknown")
            num_cases = self.get_num_test_cases(doc)
            total_cases += num_cases
            print(f"  {i+1}. {api_name} ({num_cases} test cases)")
        
        print(f"\n🎯 Total test cases to run: {total_cases}")
        
        results = []
        current_case = 1
        
        for doc_idx, doc in enumerate(documents):
            api_name = doc.get("api", "unknown")
            num_cases = self.get_num_test_cases(doc)
            
            print(f"\n🔧 Testing operator {doc_idx+1}/{len(documents)}: {api_name} ({num_cases} cases)")
            
            # Test each case for this operator
            for case_idx in range(num_cases):
                print(f"  Case {current_case}/{total_cases} (operator case: {case_idx+1}/{num_cases}): {api_name}")
                
                result = self.test_single_case(doc, case_idx)
                result["operator"] = api_name
                result["total_cases_for_operator"] = num_cases
                results.append(result)
                
                # Show results
                if result["status"] == "compared":
                    if result["results_match"]:
                        print("    ✅ Results match")
                    else:
                        print(f"    ❌ Results mismatch: {result['comparison_error']}")
                elif result["status"] == "deprecated":
                    print("    ⏭️ Deprecated operator, skipping")
                elif result["status"] == "no_paddle_equivalent":
                    print("    ⚠️ No PaddlePaddle implementation")
                else:
                    print(f"    ❌ Test failed: {result['status']}")
                
                current_case += 1
        
        # Summary
        self.print_summary(results)
        return results
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print test result summary."""
        total = len(results)
        compared = len([r for r in results if r["status"] == "compared"])
        matched = len([r for r in results if r["results_match"]])
        deprecated = len([r for r in results if r["status"] == "deprecated"])
        no_paddle = len([r for r in results if r["status"] == "no_paddle_equivalent"])
        torch_failed = len([r for r in results if r["status"] in ["torch_failed", "both_failed"]])
        paddle_failed = len([r for r in results if r["status"] in ["paddle_failed", "both_failed"]])
        
        print(f"\n📊 Test summary:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📈 Total cases: {total}")
        print(f"🔍 Compared: {compared}")
        print(f"✅ Matches: {matched}")
        print(f"❌ Mismatches: {compared - matched}")
        print(f"⏭️ Deprecated operators: {deprecated}")
        print(f"⚠️ No PaddlePaddle implementation: {no_paddle}")
        print(f"🔴 PyTorch failures: {torch_failed}")
        print(f"🟠 PaddlePaddle failures: {paddle_failed}")
        
        if compared > 0:
            match_rate = matched / compared * 100
            print(f"📊 Match rate: {match_rate:.1f}%")
        
        # Show deprecated operators
        if deprecated > 0:
            deprecated_apis = set([r["torch_api"] for r in results if r["status"] == "deprecated"])
            print(f"\n⏭️ Deprecated operators ({len(deprecated_apis)}):")
            for api in sorted(deprecated_apis):
                print(f"  - {api}")
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = None):
        """Save test results."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pt_pd_comparison_{timestamp}.json"
        
        filepath = os.path.join(self.result_dir, filename)
        
        # Prepare summary
        total = len(results)
        compared = len([r for r in results if r["status"] == "compared"])
        matched = len([r for r in results if r["results_match"]])
        deprecated = len([r for r in results if r["status"] == "deprecated"])
        no_paddle = len([r for r in results if r["status"] == "no_paddle_equivalent"])
        torch_failed = len([r for r in results if r["status"] in ["torch_failed", "both_failed"]])
        paddle_failed = len([r for r in results if r["status"] in ["paddle_failed", "both_failed"]])
        
        output_data = {
            "summary": {
                "total_tests": total,
                "compared": compared,
                "matched": matched,
                "mismatch": compared - matched,
                "deprecated": deprecated,
                "no_paddle_equivalent": no_paddle,
                "torch_failed": torch_failed,
                "paddle_failed": paddle_failed,
                "match_rate": matched / compared * 100 if compared > 0 else 0,
                "timestamp": datetime.now().isoformat()
            },
            "results": results,
            "deprecated_apis": list(self.deprecated_torch_apis.keys())
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filepath}")
        
        # Save mismatched cases
        mismatched_cases = [r for r in results if r["status"] == "compared" and not r["results_match"]]
        if mismatched_cases:
            mismatch_filename = f"pt_pd_mismatches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            mismatch_filepath = os.path.join(self.result_dir, mismatch_filename)
            
            with open(mismatch_filepath, 'w', encoding='utf-8') as f:
                json.dump(mismatched_cases, f, indent=2, ensure_ascii=False)
            
            print(f"⚠️ Mismatched cases saved to: {mismatch_filepath}")
    
    def close(self):
        """Close MongoDB connection."""
        self.client.close()

def main():
    """Main function."""
    # Initialize comparator
    comparator = PyTorchPaddleComparator()
    
    try:
        # Compare first N operators
        results = comparator.compare_first_n_operators(5)
        
        # Save results
        comparator.save_results(results)
        
    finally:
        # Close connection
        comparator.close()

if __name__ == "__main__":
    main()
