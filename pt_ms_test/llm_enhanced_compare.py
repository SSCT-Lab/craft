#!/usr/bin/env python3
"""
LLM-based PyTorch vs MindSpore operator comparison test framework.
Uses a large model to repair and mutate test cases to improve usability and coverage.
"""

import pymongo
import torch
import mindspore
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

# Add project root to sys.path so component modules can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from component.doc.doc_crawler_factory import get_doc_content

class LLMEnhancedComparator:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "freefuzz-torch"):
        """
        Initialize an LLM-based PyTorch and MindSpore comparator.
        
        Args:
            mongo_uri: MongoDB connection URI
            db_name: database name
        """
        # MongoDB connection
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["argVS"]
        
        # Initialize LLM client (Aliyun Qwen)
        # Prefer key from project root aliyun.key, otherwise use env var
        api_key = self._load_api_key()
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # Load API mapping table
        self.api_mapping = self.load_api_mapping()
        
        # Create result directory (under pt_ms_test)
        self.result_dir = os.path.join(ROOT_DIR, "pt_ms_test", "pt_ms_log_1")
        os.makedirs(self.result_dir, exist_ok=True)
        print(f"📁 Result directory: {self.result_dir}")
        
        # Fix random seed for reproducibility
        self.random_seed = 42
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        mindspore.set_seed(self.random_seed)
        
        # Set MindSpore to dynamic (pynative) mode
        mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
        
        # Deprecated PyTorch operators
        self.deprecated_torch_apis = {
            "torch.symeig": "Removed in PyTorch 1.9; use torch.linalg.eigh instead"
        }
        
        # Operators that may hang or crash (skip their tests)
        self.problematic_apis = {
            "torch.triu": "May cause the program to hang",
            "torch.as_tensor": "May crash MindSpore runtime",
        }
    
    def _load_api_key(self) -> str:
        """
        Load Aliyun API key.
        
        Prefer aliyun.key in project root; otherwise use DASHSCOPE_API_KEY env var.
        
        Returns:
            API key string
        """
        key_file = os.path.join(ROOT_DIR, "aliyun.key")
        
        # Prefer file-based key
        if os.path.exists(key_file):
            try:
                with open(key_file, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
                if api_key:
                    print(f"✅ Loaded API key from file: {key_file}")
                    return api_key
            except Exception as e:
                print(f"⚠️ Failed to read key file: {e}")
        
        # Fallback to environment variable
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            print(f"✅ Loaded API key from env var: DASHSCOPE_API_KEY")
            return api_key
        
        # Not found
        print("❌ API key not found. Ensure aliyun.key exists or set DASHSCOPE_API_KEY.")
        return ""
    
    def load_api_mapping(self) -> Dict[str, Dict[str, str]]:
        """Load PyTorch-to-MindSpore API mapping table."""
        # Use new mapping file: ms_api_mappings_final.csv
        mapping_file = os.path.join(ROOT_DIR, "component", "data", "ms_api_mappings_final.csv")
        try:
            df = pd.read_csv(mapping_file)
            mapping = {}
            
            for _, row in df.iterrows():
                # New mapping columns: pytorch-api, mindspore-api
                pt_api = str(row["pytorch-api"]).strip()
                ms_api = str(row["mindspore-api"]).strip()
                
                # Keep all mappings (including "无对应实现"), handled in convert_api_name
                mapping[pt_api] = {"ms_api": ms_api, "note": ""}
            
            print(f"✅ Loaded API mapping table with {len(mapping)} entries")
            print(f"📄 Mapping file: {mapping_file}")
            return mapping
        except Exception as e:
            print(f"❌ Failed to load API mapping table: {e}")
            return {}
    
    def is_class_based_api(self, api_name: str) -> bool:
        """Check whether the API is class-based."""
        parts = api_name.split(".")
        if len(parts) >= 2:
            name = parts[-1]
            return any(c.isupper() for c in name)
        return False
    
    # def convert_class_to_functional(self, torch_api: str) -> Tuple[Optional[str], Optional[str]]:
    #     """Convert class-based API to functional API."""
    #     if not self.is_class_based_api(torch_api):
    #         return None, None
    #     
    #     parts = torch_api.split(".")
    #     if len(parts) >= 3 and parts[1] == "nn":
    #         class_name = parts[-1]
    #         
    #         # Improved regex to handle consecutive uppercase letters
    #         # 1) Insert underscore between lowercase and uppercase
    #         # 2) Insert underscore before the last uppercase when followed by lowercase
    #         func_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', class_name)  # aB -> a_B
    #         func_name = re.sub('([A-Z]+)([A-Z][a-z])', r'\1_\2', func_name)  # ABCDef -> ABC_Def
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
        Convert PyTorch API to MindSpore API.
        
        Returns:
            (converted PyTorch API, converted MindSpore API, mapping method)
            For class operators, map directly to MindSpore class operator.
        """
        # # Class-to-functional logic disabled
        # # Check whether the API is class-based
        # if self.is_class_based_api(torch_api):
        #     torch_func, paddle_func = self.convert_class_to_functional(torch_api)
        #     if torch_func and paddle_func:
        #         torch_func_obj = self.get_operator_function(torch_func, "torch")
        #         paddle_func_obj = self.get_operator_function(paddle_func, "paddle")
        #         if torch_func_obj and paddle_func_obj:
        #             return torch_func, paddle_func, "class-to-function"
        
        # Check mapping table
        if torch_api in self.api_mapping:
            ms_api = self.api_mapping[torch_api]["ms_api"]
            
            # Check for "no equivalent" literal value
            if ms_api == "无对应实现":
                return torch_api, None, "无对应实现"
            else:
                return torch_api, ms_api, "mapping table"
        
        # Not in mapping table; do not perform manual conversion
        return torch_api, None, "not found in mapping table"
    
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
                elif framework == "mindspore" and parts[0] == "mindspore":
                    if len(parts) == 2:
                        return getattr(mindspore, parts[1])
                    elif len(parts) == 3:
                        module = getattr(mindspore, parts[1])
                        return getattr(module, parts[2])
                    elif len(parts) == 4:
                        module1 = getattr(mindspore, parts[1])
                        module2 = getattr(module1, parts[2])
                        return getattr(module2, parts[3])
            return None
        except AttributeError:
            return None
    
    def convert_key(self, key: str, mindspore_api: str = "") -> str:
        """Convert parameter name."""
        key_mapping = {
            "input": "x",
            "other": "y",
        }
        return key_mapping.get(key, key)
    
    def should_skip_param(self, key: str, mindspore_api: str) -> bool:
        """Determine whether a parameter should be skipped."""
        common_skip_params = ["layout", "requires_grad", "out"]
        skip_params = {
            # Add API-specific skip parameters if needed
        }
        
        if key in common_skip_params:
            return True
        
        if mindspore_api in skip_params:
            return key in skip_params[mindspore_api]
        
        return False
    
    def generate_numpy_data(self, data: Any) -> np.ndarray:
        """
        Generate numpy arrays as shared data source.
        
        Supported dtypes:
        - With torch prefix: torch.float32, torch.bool, torch.int64, etc.
        - Without prefix: float32, bool, int64, etc.
        - Numpy format: float32, bool_, int64, etc.
        """
        if isinstance(data, dict):
            # Extended dtype mapping with multiple formats
            dtype_map = {
                # Torch format (with prefix)
                "torch.float64": np.float64,
                "torch.float32": np.float32,
                "torch.int64": np.int64,
                "torch.int32": np.int32,
                "torch.bool": np.bool_,
                "torch.uint8": np.uint8,
                # Without torch prefix (LLM may return these)
                "float64": np.float64,
                "float32": np.float32,
                "int64": np.int64,
                "int32": np.int32,
                "bool": np.bool_,
                "uint8": np.uint8,
                # Numpy format
                "bool_": np.bool_,
                "float": np.float32,
                "int": np.int64,
            }
            
            shape = data.get("shape", [])
            dtype_str = data.get("dtype", "torch.float32")
            dtype = dtype_map.get(dtype_str, np.float32)
            
            # Warn if dtype_str is not in mapping
            if dtype_str not in dtype_map:
                print(f"      ⚠️ Warning: unrecognized dtype '{dtype_str}', using default float32")
            else:
                print(f"      ✅ dtype mapping: '{dtype_str}' -> {dtype}")
            
            if shape:
                if dtype == np.bool_:
                    return np.random.randint(0, 2, shape).astype(np.bool_)
                elif dtype in [np.int64, np.int32]:
                    return np.random.randint(-10, 10, shape).astype(dtype)
                else:
                    return np.random.randn(*shape).astype(dtype)
            else:
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
    
    def prepare_shared_numpy_data(self, document: Dict[str, Any], case_index: int = 0) -> Dict[str, Any]:
        """Prepare shared numpy data so PyTorch and MindSpore use identical inputs."""
        shared_data = {}
        api_name = document.get("api", "")
        
        # For class-based APIs, generate default input if none provided
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
        
        # Process other parameters in document
        exclude_keys = ["_id", "api"]
        for key, value in document.items():
            if key not in exclude_keys:
            # For varargs (prefixed with *), keep raw values without conversion
            # Conversion will happen in prepare_arguments_torch/paddle
                if key.startswith('*'):
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
        """Convert data to PyTorch tensor."""
        if numpy_data is not None:
            return torch.from_numpy(numpy_data.copy())
        
        if isinstance(data, dict):
            numpy_data = self.generate_numpy_data(data)
            return torch.from_numpy(numpy_data.copy())
        elif isinstance(data, (int, float)):
            return torch.tensor(data)
        elif isinstance(data, list):
            return torch.tensor(data)
        else:
            return torch.tensor(data)
    
    def convert_to_tensor_mindspore(self, data: Any, numpy_data: np.ndarray = None) -> mindspore.Tensor:
        """Convert data to MindSpore tensor."""
        if numpy_data is not None:
            return mindspore.Tensor(numpy_data.copy())
        
        if isinstance(data, dict):
            numpy_data = self.generate_numpy_data(data)
            return mindspore.Tensor(numpy_data.copy())
        elif isinstance(data, (int, float)):
            return mindspore.Tensor(data)
        elif isinstance(data, list):
            return mindspore.Tensor(data)
        else:
            return mindspore.Tensor(data)
    
    def prepare_arguments_torch(self, test_case: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Prepare arguments for PyTorch.
    
        Notes:
        1) For functions like torch.where, parameters must be positional in order:
            - torch.where(condition, x, y) or torch.where(condition, input, other)
        2) Parameters starting with * (e.g., *tensors) are varargs and must be unpacked
        """
        args = []
        kwargs = {}
        
        # Check for varargs (prefixed with *)
        varargs_key = None
        for key in test_case.keys():
            if key.startswith('*'):
                varargs_key = key
                break
        
        # If varargs exist, unpack to positional args
        if varargs_key:
            varargs_value = test_case[varargs_key]
            if isinstance(varargs_value, list):
                for item in varargs_value:
                    if isinstance(item, dict) and "shape" in item:
                        # Tensor descriptor: generate numpy data and convert
                        numpy_data = self.generate_numpy_data(item)
                        args.append(self.convert_to_tensor_torch(None, numpy_data))
                    elif isinstance(item, list):
                        # Nested list, process recursively
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
        # These must be passed as positional args
        positional_params = ["condition", "x", "y", "input", "other"]
        
        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, np.ndarray):
                    args.append(self.convert_to_tensor_torch(None, value))
                else:
                    # Add scalar directly
                    args.append(value)
        
        # Handle other parameters as keyword args
        for key, value in test_case.items():
            if key not in positional_params + ["api"]:
                if isinstance(value, np.ndarray):
                    kwargs[key] = self.convert_to_tensor_torch(None, value)
                else:
                    kwargs[key] = value
        
        return args, kwargs
    
    def prepare_arguments_mindspore(self, test_case: Dict[str, Any], mindspore_api: str) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Prepare arguments for MindSpore.
        
        Notes:
        1) For functions like mindspore.mint.where, parameters must be positional in order
        2) Parameters starting with * (e.g., *tensors) are varargs and must be unpacked
        """
        args = []
        kwargs = {}
        
        # Check for varargs (prefixed with *)
        varargs_key = None
        for key in test_case.keys():
            if key.startswith('*'):
                varargs_key = key
                break
        
        # If varargs exist, unpack to positional args
        if varargs_key:
            varargs_value = test_case[varargs_key]
            if isinstance(varargs_value, list):
                for item in varargs_value:
                    if isinstance(item, dict) and "shape" in item:
                        # Tensor descriptor: generate numpy data and convert
                        numpy_data = self.generate_numpy_data(item)
                        args.append(self.convert_to_tensor_mindspore(None, numpy_data))
                    elif isinstance(item, list):
                        # Nested list, process recursively
                        nested_tensors = []
                        for nested_item in item:
                            if isinstance(nested_item, dict) and "shape" in nested_item:
                                numpy_data = self.generate_numpy_data(nested_item)
                                nested_tensors.append(self.convert_to_tensor_mindspore(None, numpy_data))
                            elif isinstance(nested_item, np.ndarray):
                                nested_tensors.append(self.convert_to_tensor_mindspore(None, nested_item))
                            else:
                                nested_tensors.append(nested_item)
                        args.extend(nested_tensors)
                    elif isinstance(item, np.ndarray):
                        args.append(self.convert_to_tensor_mindspore(None, item))
                    else:
                        args.append(item)
            return args, kwargs
        
        # Process positional params in order: condition, x/input, y/other
        positional_params = ["condition", "x", "y", "input", "other"]
        
        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, np.ndarray):
                    args.append(self.convert_to_tensor_mindspore(None, value))
                else:
                    # Add scalar directly
                    args.append(value)
        
        # Handle other parameters as keyword args
        for key, value in test_case.items():
            if key not in positional_params + ["api"]:
                if self.should_skip_param(key, mindspore_api):
                    continue
                
                if isinstance(value, np.ndarray):
                    kwargs[key] = self.convert_to_tensor_mindspore(None, value)
                else:
                    kwargs[key] = value
        
        return args, kwargs
    
    def compare_tensors(self, torch_result, mindspore_result, tolerance: float = 1e-5) -> Tuple[bool, str]:
        """Compare two tensors for equality."""
        try:
            # Convert to numpy for comparison
            if hasattr(torch_result, 'detach'):
                torch_np = torch_result.detach().cpu().numpy()
            else:
                torch_np = np.array(torch_result)
            
            if hasattr(mindspore_result, 'asnumpy'):
                mindspore_np = mindspore_result.asnumpy()
            else:
                mindspore_np = np.array(mindspore_result)
            
            # Check shape
            if torch_np.shape != mindspore_np.shape:
                return False, f"Shape mismatch: PyTorch {torch_np.shape} vs MindSpore {mindspore_np.shape}"
            
            # Check whether dtype is boolean
            if torch_np.dtype == np.bool_ or mindspore_np.dtype == np.bool_:
                if np.array_equal(torch_np, mindspore_np):
                    return True, "Boolean values match"
                else:
                    diff_count = np.sum(torch_np != mindspore_np)
                    return False, f"Boolean mismatch, diff count: {diff_count}"
            
            # Check numeric values
            if np.allclose(torch_np, mindspore_np, atol=tolerance, rtol=tolerance, equal_nan=True):
                return True, "Numeric values match"
            else:
                max_diff = np.max(np.abs(torch_np - mindspore_np))
                return False, f"Numeric mismatch, max diff: {max_diff}"
        
        except Exception as e:
            return False, f"Comparison error: {str(e)}"
    
    def execute_test_case(self, torch_api: str, mindspore_api: str, torch_test_case: Dict[str, Any], mindspore_test_case: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a single test case.
        
        Args:
            torch_api: PyTorch API name
            mindspore_api: MindSpore API name
            torch_test_case: PyTorch test case (with params)
            mindspore_test_case: MindSpore test case (with params)
        
        Returns:
            execution result dict
        """
        result = {
            "torch_api": torch_api,
            "mindspore_api": mindspore_api,
            "torch_success": False,
            "mindspore_success": False,
            "results_match": False,
            "torch_error": None,
            "mindspore_error": None,
            "comparison_error": None,
            "torch_shape": None,
            "mindspore_shape": None,
            "torch_dtype": None,
            "mindspore_dtype": None,
            "status": "unknown"
        }
        
        # If mindspore_test_case is not provided, use torch_test_case (backward compatible)
        if mindspore_test_case is None:
            mindspore_test_case = torch_test_case
        
        # Determine whether this is a class operator
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
                    # For class operators, instantiate first then call
                    # Extract init params (non-input)
                    init_kwargs = {k: v for k, v in kwargs.items() if k != 'input'}
                    # Instantiate class
                    torch_instance = torch_func(**init_kwargs)
                    # Get input data (from args or kwargs)
                    if 'input' in kwargs:
                        input_data = kwargs['input']
                    elif len(args) > 0:
                        input_data = args[0]
                    else:
                        # If no input, try using default input
                        raise ValueError("Class operator missing input parameter")
                    
                    # Call instance (forward)
                    with torch.no_grad():
                        torch_result = torch_instance(input_data)
                else:
                    # For function operators, call directly
                    with torch.no_grad():
                        torch_result = torch_func(*args, **kwargs)
                
                result["torch_success"] = True
                result["torch_shape"] = list(torch_result.shape) if hasattr(torch_result, 'shape') else None
                result["torch_dtype"] = str(torch_result.dtype) if hasattr(torch_result, 'dtype') else None
        except Exception as e:
            result["torch_error"] = str(e)
            result["torch_traceback"] = traceback.format_exc()
        
        # Test MindSpore
        mindspore_result = None
        try:
            mindspore_func = self.get_operator_function(mindspore_api, "mindspore")
            if mindspore_func is None:
                result["mindspore_error"] = f"MindSpore operator {mindspore_api} not found"
            else:
                args, kwargs = self.prepare_arguments_mindspore(mindspore_test_case, mindspore_api)
                
                if is_class_api:
                    # For class operators, instantiate first then call
                    # Extract init params (non-x/input)
                    init_kwargs = {k: v for k, v in kwargs.items() if k not in ['x', 'input']}
                    # Instantiate class
                    mindspore_instance = mindspore_func(**init_kwargs)
                    # Get input data (from args or kwargs)
                    if 'x' in kwargs:
                        input_data = kwargs['x']
                    elif 'input' in kwargs:
                        input_data = kwargs['input']
                    elif len(args) > 0:
                        input_data = args[0]
                    else:
                        # If no input, try using default input
                        raise ValueError("Class operator missing input/x parameter")
                    
                    # Call instance (forward)
                    mindspore_result = mindspore_instance(input_data)
                else:
                    # For function operators, call directly
                    mindspore_result = mindspore_func(*args, **kwargs)
                
                result["mindspore_success"] = True
                result["mindspore_shape"] = list(mindspore_result.shape) if hasattr(mindspore_result, 'shape') else None
                result["mindspore_dtype"] = str(mindspore_result.dtype) if hasattr(mindspore_result, 'dtype') else None
        except Exception as e:
            result["mindspore_error"] = str(e)
            result["mindspore_traceback"] = traceback.format_exc()
        
        # Compare results
        if result["torch_success"] and result["mindspore_success"]:
            try:
                is_match, comparison_msg = self.compare_tensors(torch_result, mindspore_result)
                result["results_match"] = is_match
                result["comparison_error"] = comparison_msg if not is_match else None
                result["status"] = "compared"
            except Exception as e:
                result["comparison_error"] = str(e)
                result["status"] = "comparison_failed"
        elif result["torch_success"] and not result["mindspore_success"]:
            result["status"] = "mindspore_failed"
        elif not result["torch_success"] and result["mindspore_success"]:
            result["status"] = "torch_failed"
        else:
            result["status"] = "both_failed"
        
        return result
    
    def _fetch_api_docs(self, torch_api: str, mindspore_api: str) -> Tuple[str, str]:
        """
        Fetch API docs for PyTorch and MindSpore.
        
        Args:
            torch_api: PyTorch API name
            mindspore_api: MindSpore API name
        
        Returns:
            (PyTorch doc content, MindSpore doc content)
        """
        # Minimum length threshold for a valid doc
        MIN_DOC_LENGTH = 300
        
        torch_doc = ""
        mindspore_doc = ""
        
        try:
            print(f"    📖 Fetching PyTorch doc: {torch_api}")
            torch_doc = get_doc_content(torch_api, "pytorch")
            # Doc is valid if: non-empty, no error text, and above length threshold
            if (torch_doc 
                and "Unable" not in torch_doc 
                and "not supported" not in torch_doc
                and len(torch_doc.strip()) > MIN_DOC_LENGTH):
                # Truncate long docs to save tokens
                if len(torch_doc) > 3000:
                    torch_doc = torch_doc[:3000] + "\n... (doc truncated)"
                print(f"    ✅ PyTorch doc fetched, length: {len(torch_doc)}")
            else:
                doc_len = len(torch_doc.strip()) if torch_doc else 0
                torch_doc = f"Unable to fetch documentation for {torch_api} (length: {doc_len}, min required: {MIN_DOC_LENGTH})"
                print(f"    ⚠️ {torch_doc}")
        except Exception as e:
            torch_doc = f"Failed to fetch documentation: {str(e)}"
            print(f"    ❌ Failed to fetch PyTorch doc: {e}")
        
        try:
            print(f"    📖 Fetching MindSpore doc: {mindspore_api}")
            mindspore_doc = get_doc_content(mindspore_api, "mindspore")
            # Doc is valid if: non-empty, no error text, and above length threshold
            if (mindspore_doc 
                and "Unable" not in mindspore_doc 
                and "not supported" not in mindspore_doc
                and len(mindspore_doc.strip()) > MIN_DOC_LENGTH):
                # Truncate long docs to save tokens
                if len(mindspore_doc) > 3000:
                    mindspore_doc = mindspore_doc[:3000] + "\n... (doc truncated)"
                print(f"    ✅ MindSpore doc fetched, length: {len(mindspore_doc)}")
            else:
                doc_len = len(mindspore_doc.strip()) if mindspore_doc else 0
                mindspore_doc = f"Unable to fetch documentation for {mindspore_api} (length: {doc_len}, min required: {MIN_DOC_LENGTH})"
                print(f"    ⚠️ {mindspore_doc}")
        except Exception as e:
            mindspore_doc = f"Failed to fetch documentation: {str(e)}"
            print(f"    ❌ Failed to fetch MindSpore doc: {e}")
        
        return torch_doc, mindspore_doc
    
    def _build_llm_prompt(self, execution_result: Dict[str, Any], torch_test_case: Dict[str, Any], mindspore_test_case: Dict[str, Any], torch_doc: str = "", mindspore_doc: str = "") -> str:
        """Build prompt for LLM."""
        torch_api = execution_result.get("torch_api", "")
        mindspore_api = execution_result.get("mindspore_api", "")
        status = execution_result.get("status", "")
        torch_success = execution_result.get("torch_success", False)
        mindspore_success = execution_result.get("mindspore_success", False)
        results_match = execution_result.get("results_match", False)
        torch_error = execution_result.get("torch_error", "")
        mindspore_error = execution_result.get("mindspore_error", "")
        comparison_error = execution_result.get("comparison_error", "")
        
        # Simplify PyTorch test case to reduce tokens
        simplified_torch_test_case = {}
        for key, value in torch_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_torch_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_torch_test_case[key] = value
        
        # Simplify MindSpore test case to reduce tokens
        simplified_mindspore_test_case = {}
        for key, value in mindspore_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_mindspore_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_mindspore_test_case[key] = value
        
        # Build PyTorch parameter examples
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
        
        torch_param_example_str = ",\n".join(torch_param_examples) if torch_param_examples else '    "input": {"shape": [2, 3], "dtype": "torch.float32"}'
        
        # Build MindSpore parameter examples
        mindspore_param_examples = []
        for key, value in simplified_mindspore_test_case.items():
            if key == "api":
                continue
            if isinstance(value, dict) and "shape" in value:
                mindspore_param_examples.append(f'    "{key}": {json.dumps(value)}')
            elif isinstance(value, (int, float)):
                mindspore_param_examples.append(f'    "{key}": {value}')
            else:
                mindspore_param_examples.append(f'    "{key}": {json.dumps(value)}')
        
        mindspore_param_example_str = ",\n".join(mindspore_param_examples) if mindspore_param_examples else '    "x": {"shape": [2, 3], "dtype": "float32"}'
        
        # Build API doc section
        doc_section = ""
        if torch_doc or mindspore_doc:
            doc_section = f"""
    ## API Documentation Reference
    ### PyTorch Documentation
    {torch_doc if torch_doc else "Documentation unavailable"}

    ### MindSpore Documentation
    {mindspore_doc if mindspore_doc else "Documentation unavailable"}
"""
        
        prompt = f"""Analyze the following operator test case results in PyTorch and MindSpore, then repair or mutate (fuzz) the test case based on the results.

    ## Test Info
    - **PyTorch API**: {torch_api}
    - **MindSpore API**: {mindspore_api}
    {doc_section}
    ## Execution Results
    - **Status**: {status}
    - **PyTorch succeeded**: {torch_success}
    - **MindSpore succeeded**: {mindspore_success}
    - **Results match**: {results_match}

    ## Error Info
    - **PyTorch error**: {torch_error if torch_error else "None"}
    - **MindSpore error**: {mindspore_error if mindspore_error else "None"}
    - **Comparison error**: {comparison_error if comparison_error else "None"}

    ## Original Test Cases

    ### PyTorch Test Case
```json
{json.dumps(simplified_torch_test_case, indent=2, ensure_ascii=False)}
```

    ### MindSpore Test Case
```json
{json.dumps(simplified_mindspore_test_case, indent=2, ensure_ascii=False)}
```

    ## Task Requirements
    Based on the above information (including official API docs), decide whether the cross-framework results are **consistent**, **inconsistent**, or **execution failed**, then perform one of the following:

    1. **If consistent**: **mutate (fuzz)** the case, e.g., change input shapes, tweak parameter values, or explore edge cases.
    2. **If execution failed**: **repair** the case based on error and docs (change param names/count/types/ranges as needed; frameworks may differ), or **skip** if you believe the operators are not equivalent.
    3. **If inconsistent**: determine whether the difference is tolerable precision (<= 1e-3). If tolerable, **mutate**. If operators are not equivalent based on docs, **skip**. Otherwise, treat as test-case construction issue and **repair** based on docs.

    ## Output Format
    Strictly output JSON in the following format with no extra text, comments, or markdown:

{{
  "operation": "mutation",
    "reason": "detailed reason for the operation",
  "pytorch_test_case": {{
    "api": "{torch_api}",
{torch_param_example_str}
  }},
  "mindspore_test_case": {{
    "api": "{mindspore_api}",
{mindspore_param_example_str}
  }}
}}

**Important Notes**:
1. `operation` must be one of "mutation", "repair", or "skip"
2. Tensor params must use {"shape": [...], "dtype": "..."}
3. Scalar params use numeric values, e.g., "y": 0
4. Inputs for both frameworks must be identical and semantically equivalent, with theoretically identical outputs.
5. PyTorch and MindSpore test cases may differ in param names (e.g., input vs x), values, or counts, as long as outputs should match.
6. If the operator docs are missing because the operator does not exist or was removed, set `operation` to "skip" and do not attempt repair.
7. When mutating, explore edge cases such as empty tensors (shape includes 0), single-element tensors (shape=[1] or []), high-rank tensors, huge tensors, different dtypes (int/float/bool), and boundary values to improve coverage.
8. Read the official API docs to ensure param names, types, and value ranges are consistent with documentation.
"""
        return prompt
    
    def call_llm_for_repair_or_mutation(self, execution_result: Dict[str, Any], torch_test_case: Dict[str, Any], mindspore_test_case: Dict[str, Any], torch_doc: str = "", mindspore_doc: str = "") -> Dict[str, Any]:
        """Call LLM to repair or mutate test cases."""
        prompt = self._build_llm_prompt(execution_result, torch_test_case, mindspore_test_case, torch_doc, mindspore_doc)
        
        # Print simplified test cases sent to LLM
        simplified_torch_test_case = {}
        for key, value in torch_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_torch_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_torch_test_case[key] = value
        
        simplified_mindspore_test_case = {}
        for key, value in mindspore_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_mindspore_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_mindspore_test_case[key] = value
        
        # Simplified printing: detailed test case info is logged elsewhere
        # print(f"\n{'='*40}")
        # print(f"📋 Test cases sent to LLM (simplified):")
        # print(f"{'='*40}")
        # print(f"\n🅿️ PyTorch测试用例:")
        # print(json.dumps(simplified_torch_test_case, indent=2, ensure_ascii=False))
        # print(f"\n🅰️ MindSpore测试用例:")
        # print(json.dumps(simplified_mindspore_test_case, indent=2, ensure_ascii=False))
        # print(f"\n{'='*40}\n")
        
        try:
            print(f"    🤖 Calling LLM for analysis...")
            completion = self.llm_client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a deep learning framework testing expert, familiar with API differences between PyTorch and MindSpore. Based on execution results, decide whether to repair or mutate test cases and return a strict JSON result."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
            )
            
            raw_response = completion.choices[0].message.content.strip()
            # Simplified printing: raw LLM response is logged elsewhere
            # print(f"    🤖 LLM原始响应: {raw_response[:200]}...")
            # print(f"    🤖 LLM原始响应: {raw_response}")
            
            # Add 1s delay to avoid rate limits
            time.sleep(1)
            
            # Try parse JSON
            try:
                llm_result = json.loads(raw_response)
                return llm_result
            except json.JSONDecodeError as e:
                print(f"    ⚠️ LLM did not return valid JSON; attempting to extract JSON...")
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    llm_result = json.loads(json_match.group())
                    return llm_result
                else:
                    return {
                        "operation": "skip",
                        "reason": f"LLM response format error: {e}",
                        "pytorch_test_case": torch_test_case,
                        "mindspore_test_case": mindspore_test_case
                    }
        
        except Exception as e:
            print(f"    ❌ LLM call failed: {e}")
            return {
                "operation": "skip",
                "reason": f"LLM call failed: {e}",
                "pytorch_test_case": torch_test_case,
                "mindspore_test_case": mindspore_test_case
            }
    
    def get_num_test_cases_from_document(self, document: Dict[str, Any]) -> int:
        """Get number of test cases in the document."""
        max_len = 0
        # Find max length among list-type fields
        for key, value in document.items():
            if key not in ["_id", "api"] and isinstance(value, list):
                max_len = max(max_len, len(value))
        return max_len if max_len > 0 else 1
    
    def llm_enhanced_test_operator(self, operator_name: str, max_iterations: int = 3, num_test_cases: int = None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Test a single operator using LLM enhancement.
        
        Args:
            operator_name: operator name, e.g., "torch.where"
            max_iterations: max iterations per test case
            num_test_cases: number of cases to test; None means all
        
        Returns:
            (all iteration results for all cases, stats dict)
        """
        print(f"\n{'='*80}")
        print(f"🎯 Starting operator test: {operator_name}")
        print(f"🔄 Max iterations per case: {max_iterations}")
        print(f"{'='*80}\n")
        
        # Initialize stats counters
        stats = {
            "llm_generated_cases": 0,      # Total LLM-generated cases
            "successful_cases": 0           # Cases where both frameworks succeeded
        }
        
        # Skip operators that may hang
        if operator_name in self.problematic_apis:
            reason = self.problematic_apis[operator_name]
            print(f"⏭️ Skipping operator {operator_name}: {reason}")
            return [], stats
        
        # Fetch test cases from MongoDB
        document = self.collection.find_one({"api": operator_name})
        if document is None:
            print(f"❌ No test cases found for operator {operator_name}")
            return [], stats
        
        # Get total test case count
        total_cases = self.get_num_test_cases_from_document(document)
        print(f"📊 Total test cases in DB: {total_cases}")
        
        # Determine number of cases to test
        if num_test_cases is None:
            num_test_cases = total_cases
            print(f"📝 Testing all {num_test_cases} cases")
        else:
            num_test_cases = min(num_test_cases, total_cases)
            print(f"📝 Testing first {num_test_cases} cases (total {total_cases})")
        
        # Get converted PyTorch and MindSpore APIs
        torch_api, mindspore_api, mapping_method = self.convert_api_name(operator_name)
        if mindspore_api is None:
            print(f"❌ Operator {operator_name} has no MindSpore equivalent")
            return [], stats
        
        # Show API mapping info
        if torch_api != operator_name:
            print(f"✅ Original PyTorch API: {operator_name}")
            print(f"✅ Converted PyTorch API: {torch_api}")
        else:
            print(f"✅ PyTorch API: {torch_api}")
        print(f"✅ MindSpore API: {mindspore_api}")
        print(f"✅ Mapping method: {mapping_method}\n")
        
        # Store all iteration results for all cases
        all_results = []
        
        # Test each case
        for case_idx in range(num_test_cases):
            print(f"\n{'#'*80}")
            print(f"📋 Test case {case_idx + 1}/{num_test_cases}")
            print(f"{'#'*80}")
            
            # Prepare initial data for this case
            print(f"  📦 Preparing data for case {case_idx + 1}...")
            initial_test_case = self.prepare_shared_numpy_data(document, case_index=case_idx)
            # Use PyTorch API
            initial_test_case["api"] = torch_api
            
            # Print test case params
            print(f"  📝 Test case parameters:")
            for key, value in initial_test_case.items():
                if key == "api":
                    continue
                if isinstance(value, np.ndarray):
                    print(f"    - {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"    - {key}: {value}")
            
            # Iterate testing for current case
            # Use converted APIs
            case_results = self._test_single_case_with_iterations(
                torch_api, 
                mindspore_api, 
                initial_test_case, 
                max_iterations,
                case_idx + 1,
                stats
            )
            
            # Save current case results
            all_results.extend(case_results)
        
        print(f"\n{'='*80}")
        print(f"✅ All tests completed")
        print(f"📊 Tested {num_test_cases} cases, total {len(all_results)} iterations")
        print(f"📊 LLM-generated cases: {stats['llm_generated_cases']}")
        print(f"📊 Cases where both frameworks succeeded: {stats['successful_cases']}")
        print(f"{'='*80}\n")
        
        return all_results, stats
    
    def _test_single_case_with_iterations(self, operator_name: str, mindspore_api: str, 
                                          initial_test_case: Dict[str, Any], 
                                          max_iterations: int,
                                          case_number: int,
                                          stats: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        Run multiple iterations for a single test case.
        
        Args:
            operator_name: PyTorch operator name
            mindspore_api: MindSpore operator name
            initial_test_case: initial test case
            max_iterations: max iterations
            case_number: test case index (for display)
            stats: stats dict (counts LLM-generated and successful cases)
        
        Returns:
            all iteration results for this case
        """
        # Store iteration results for this case
        case_results = []
        
        # Current test case
        # PyTorch uses original test case (api already set to torch_api)
        # MindSpore needs a copy with correct api
        current_torch_test_case = initial_test_case
        current_mindspore_test_case = copy.deepcopy(initial_test_case)
        current_mindspore_test_case["api"] = mindspore_api  # Set correct MindSpore API
        
        # Mark whether current case is LLM-generated (first iteration is DB case)
        is_llm_generated = False
        
        # Pre-fetch API docs (once per case)
        print(f"\n  📖 Pre-fetching API docs...")
        torch_doc, mindspore_doc = self._fetch_api_docs(operator_name, mindspore_api)
        
        # Start iterative testing
        for iteration in range(max_iterations):
            print(f"\n{'─'*80}")
            print(f"🔄 Iteration {iteration + 1}/{max_iterations}")
            if is_llm_generated:
                print(f"   (LLM-generated case)")
            else:
                print(f"   (DB original case)")
            print(f"{'─'*80}")
            
            # Execute test case
            try:
                # print(f"  📝 Executing test case...")
                execution_result = self.execute_test_case(operator_name, mindspore_api, current_torch_test_case, current_mindspore_test_case)
                
                # Simplified output: key status only
                status = execution_result['status']
                torch_ok = "✓" if execution_result['torch_success'] else "✗"
                mindspore_ok = "✓" if execution_result['mindspore_success'] else "✗"
                match_ok = "✓" if execution_result['results_match'] else "✗"
                print(f"  📊 Result: status={status}, PyTorch={torch_ok}, MindSpore={mindspore_ok}, match={match_ok}")
                
                # Detailed output commented out (logged elsewhere)
                # print(f"  📊 Status: {execution_result['status']}")
                # print(f"  🅿️ PyTorch success: {execution_result['torch_success']}")
                # print(f"  🅰️ MindSpore success: {execution_result['mindspore_success']}")
                # print(f"  ❓ Results match: {execution_result['results_match']}")
                
                # Print errors only when present
                if execution_result['torch_error']:
                    print(f"  ❌ PyTorch error: {execution_result['torch_error'][:100]}..." if len(str(execution_result['torch_error'])) > 100 else f"  ❌ PyTorch error: {execution_result['torch_error']}")
                if execution_result['mindspore_error']:
                    print(f"  ❌ MindSpore error: {execution_result['mindspore_error'][:100]}..." if len(str(execution_result['mindspore_error'])) > 100 else f"  ❌ MindSpore error: {execution_result['mindspore_error']}")
                # if execution_result['comparison_error']:
                #     print(f"  ⚠️ Comparison error: {execution_result['comparison_error']}")
                
                # Count only LLM-generated cases (exclude DB original)
                if is_llm_generated:
                    # Count cases where both frameworks succeeded
                    if execution_result['torch_success'] and execution_result['mindspore_success']:
                        stats["successful_cases"] += 1
                        # print(f"  📊 LLM successful case count: {stats['successful_cases']}")
                
            except Exception as e:
                print(f"  ❌ Fatal error during test execution: {str(e)[:100]}..." if len(str(e)) > 100 else f"  ❌ Fatal error during test execution: {e}")
                # print(f"  ❌ Error details:")
                # traceback.print_exc()
                
                # Create an error result
                execution_result = {
                    "status": "fatal_error",
                    "torch_success": False,
                    "mindspore_success": False,
                    "results_match": False,
                    "torch_error": f"Fatal error: {str(e)}",
                    "mindspore_error": None,
                    "comparison_error": None,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            
            # Save iteration result
            iteration_result = {
                "iteration": iteration + 1,
                "torch_test_case": current_torch_test_case,
                "mindspore_test_case": current_mindspore_test_case,
                "execution_result": execution_result,
                "llm_operation": None,
                "is_llm_generated": is_llm_generated
            }
            
            # Call LLM to repair or mutate (pass test cases and docs)
            try:
                # print(f"\n  🤖 Calling LLM to analyze test case...")
                llm_result = self.call_llm_for_repair_or_mutation(
                    execution_result, 
                    current_torch_test_case, 
                    current_mindspore_test_case,
                    torch_doc,
                    mindspore_doc
                )
            except Exception as e:
                print(f"  ❌ Error calling LLM: {str(e)[:100]}..." if len(str(e)) > 100 else f"  ❌ Error calling LLM: {e}")
                # print(f"  ❌ Error details:")
                # traceback.print_exc()
                print(f"  ⏭️ Skipping LLM analysis; ending iterations for this case")
                
                # Create a skip operation
                llm_result = {
                    "operation": "skip",
                    "reason": f"LLM call failed: {str(e)}"
                }
                
                iteration_result["llm_operation"] = llm_result
                iteration_result["case_number"] = case_number
                case_results.append(iteration_result)
                break  # 结束迭代循环
            
            operation = llm_result.get("operation", "skip")
            reason = llm_result.get("reason", "")
            
            # Simplified output: operation only
            print(f"  🤖 LLM decision: {operation}")
            # print(f"  🤖 LLM operation: {operation}")
            # print(f"  🤖 LLM reason: {reason}")
            
            iteration_result["llm_operation"] = {
                "operation": operation,
                "reason": reason
            }
            
            # Add case number info
            iteration_result["case_number"] = case_number
            case_results.append(iteration_result)
            
            # If LLM recommends skip, end iterations
            if operation == "skip":
                # print(f"  ⏭️ LLM suggested skip, ending iterations")
                break
            
            # Prepare test case for next iteration
            if operation == "mutation":
                # print(f"  🔀 Using LLM-mutated test case")
                next_pytorch_test_case = llm_result.get("pytorch_test_case", current_torch_test_case)
                next_mindspore_test_case = llm_result.get("mindspore_test_case", current_mindspore_test_case)
                # Count LLM-generated cases
                stats["llm_generated_cases"] += 1
                # print(f"  📊 LLM-generated case count: {stats['llm_generated_cases']}")
                is_llm_generated = True
            elif operation == "repair":
                # print(f"  🔧 Using LLM-repaired test case")
                next_pytorch_test_case = llm_result.get("pytorch_test_case", current_torch_test_case)
                next_mindspore_test_case = llm_result.get("mindspore_test_case", current_mindspore_test_case)
                # Count LLM-generated cases
                stats["llm_generated_cases"] += 1
                # print(f"  📊 LLM-generated case count: {stats['llm_generated_cases']}")
                is_llm_generated = True
            else:
                next_pytorch_test_case = current_torch_test_case
                next_mindspore_test_case = current_mindspore_test_case
            
            # Convert LLM test cases (PyTorch and MindSpore), sharing tensor data
            current_torch_test_case, current_mindspore_test_case = self._convert_llm_test_cases(next_pytorch_test_case, next_mindspore_test_case)
        
        # Fix: if last iteration produced new case (mutation/repair), execute it
        if len(case_results) > 0:
            last_iteration = case_results[-1]
            last_operation = last_iteration["llm_operation"].get("operation", "skip")
            
            if last_operation in ["mutation", "repair"]:
                print(f"\n  🔄 Executing last LLM-generated case...")
                
                try:
                    # Execute last LLM-generated test case
                    # print(f"  📝 Executing test case...")
                    execution_result = self.execute_test_case(operator_name, mindspore_api, current_torch_test_case, current_mindspore_test_case)
                    
                    # Simplified output: key status only
                    status = execution_result['status']
                    torch_ok = "✓" if execution_result['torch_success'] else "✗"
                    mindspore_ok = "✓" if execution_result['mindspore_success'] else "✗"
                    match_ok = "✓" if execution_result['results_match'] else "✗"
                    print(f"  📊 Final result: status={status}, PyTorch={torch_ok}, MindSpore={mindspore_ok}, match={match_ok}")
                    
                    # Detailed output commented out (logged elsewhere)
                    # print(f"  📊 Status: {execution_result['status']}")
                    # print(f"  🅿️ PyTorch success: {execution_result['torch_success']}")
                    # print(f"  🅰️ MindSpore success: {execution_result['mindspore_success']}")
                    # print(f"  ❓ Results match: {execution_result['results_match']}")
                    
                    # Print errors only when present
                    if execution_result['torch_error']:
                        print(f"  ❌ PyTorch error: {execution_result['torch_error'][:100]}..." if len(str(execution_result['torch_error'])) > 100 else f"  ❌ PyTorch error: {execution_result['torch_error']}")
                    if execution_result['mindspore_error']:
                        print(f"  ❌ MindSpore error: {execution_result['mindspore_error'][:100]}..." if len(str(execution_result['mindspore_error'])) > 100 else f"  ❌ MindSpore error: {execution_result['mindspore_error']}")
                    # if execution_result['comparison_error']:
                    #     print(f"  ⚠️ Comparison error: {execution_result['comparison_error']}")
                    
                    # Count cases where both frameworks succeeded (LLM-generated)
                    if execution_result['torch_success'] and execution_result['mindspore_success']:
                        stats["successful_cases"] += 1
                        # print(f"  📊 LLM successful case count: {stats['successful_cases']}")
                    
                    # Save this extra execution result
                    final_iteration_result = {
                        "iteration": len(case_results) + 1,
                        "torch_test_case": current_torch_test_case,
                        "mindspore_test_case": current_mindspore_test_case,
                        "execution_result": execution_result,
                        "llm_operation": {
                            "operation": "final_execution",
                            "reason": "Execute last LLM-generated case"
                        },
                        "case_number": case_number,
                        "is_llm_generated": True
                    }
                    case_results.append(final_iteration_result)
                    
                except Exception as e:
                    print(f"  ❌ Fatal error during last LLM-generated case: {str(e)[:100]}..." if len(str(e)) > 100 else f"  ❌ Fatal error during last LLM-generated case: {e}")
                    # print(f"  ❌ Error details:")
                    # traceback.print_exc()
                    
                    # Record the attempt even on error
                    final_iteration_result = {
                        "iteration": len(case_results) + 1,
                        "torch_test_case": current_torch_test_case,
                        "mindspore_test_case": current_mindspore_test_case,
                        "execution_result": {
                            "status": "fatal_error",
                            "torch_success": False,
                            "mindspore_success": False,
                            "results_match": False,
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        },
                        "llm_operation": {
                            "operation": "final_execution",
                            "reason": "Execute last LLM-generated case (fatal error)"
                        },
                        "case_number": case_number,
                        "is_llm_generated": True
                    }
                    case_results.append(final_iteration_result)
        
        print(f"\n  {'─'*76}")
        print(f"  ✅ Test case {case_number} completed, {len(case_results)} iterations")
        print(f"  {'─'*76}")
        
        return case_results
    
    def _convert_llm_test_cases(self, pytorch_test_case: Dict[str, Any], mindspore_test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Convert LLM-returned PyTorch and MindSpore test cases to executable format.
        Ensure shared tensor data between frameworks while allowing other param differences.
        
        Args:
            pytorch_test_case: LLM-returned PyTorch test case
            mindspore_test_case: LLM-returned MindSpore test case
        
        Returns:
            (converted PyTorch test case, converted MindSpore test case)
        """
        # Simplified printing: detailed conversion info is logged elsewhere
        # print(f"    🔄 Converting LLM test case format...")
        
        # Step 1: collect tensor params and generate shared numpy arrays
        shared_tensors = {}  # Store shared numpy arrays
        
        # Identify tensor params (in pytorch or mindspore test cases)
        all_keys = set(pytorch_test_case.keys()) | set(mindspore_test_case.keys())
        
        for key in all_keys:
            if key == "api":
                continue
            
            # Check for tensor descriptor
            pytorch_value = pytorch_test_case.get(key)
            mindspore_value = mindspore_test_case.get(key)
            
            is_tensor = False
            tensor_desc = None
            
            if isinstance(pytorch_value, dict) and "shape" in pytorch_value:
                is_tensor = True
                tensor_desc = pytorch_value
            elif isinstance(mindspore_value, dict) and "shape" in mindspore_value:
                is_tensor = True
                tensor_desc = mindspore_value
            
            if is_tensor:
                # Generate shared numpy array
                # print(f"      - {key}: tensor desc shape={tensor_desc.get('shape')}, dtype={tensor_desc.get('dtype')}")
                numpy_array = self.generate_numpy_data(tensor_desc)
                shared_tensors[key] = numpy_array
                # print(f"        Generated shared numpy array: shape={numpy_array.shape}, dtype={numpy_array.dtype}")
        
        # Step 2: build PyTorch and MindSpore test cases
        converted_pytorch = {}
        converted_mindspore = {}
        
        # print(f"    📦 Building PyTorch test case:")
        for key, value in pytorch_test_case.items():
            if key in shared_tensors:
                converted_pytorch[key] = shared_tensors[key]
                # print(f"      - {key}: using shared tensor")
            else:
                converted_pytorch[key] = value
                # print(f"      - {key}: using value={value}")
        
        # print(f"    📦 Building MindSpore test case:")
        for key, value in mindspore_test_case.items():
            if key in shared_tensors:
                converted_mindspore[key] = shared_tensors[key]
                # print(f"      - {key}: using shared tensor")
            else:
                converted_mindspore[key] = value
                # print(f"      - {key}: using value={value}")
        
        return converted_pytorch, converted_mindspore
    
    def save_results(self, operator_name: str, results: List[Dict[str, Any]], stats: Dict[str, int] = None):
        """
        Save test results to JSON file.
        
        Args:
            operator_name: operator name
            results: list of test results
            stats: stats dict (LLM-generated cases and successful cases)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_enhanced_{operator_name.replace('.', '_')}_{timestamp}.json"
        filepath = os.path.join(self.result_dir, filename)
        
        # Prepare output data (remove numpy arrays for JSON serialization)
        output_results = []
        for result in results:
            output_result = copy.deepcopy(result)
            
            # Simplify numpy arrays in test_case (legacy format)
            if "test_case" in output_result:
                simplified_case = {}
                for key, value in output_result["test_case"].items():
                    if isinstance(value, np.ndarray):
                        simplified_case[key] = {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "sample_values": value.flatten()[:10].tolist() if value.size > 0 else []
                        }
                    else:
                        simplified_case[key] = value
                output_result["test_case"] = simplified_case
            
            # Simplify numpy arrays in torch_test_case/mindspore_test_case (new format)
            for test_case_key in ["torch_test_case", "mindspore_test_case"]:
                if test_case_key in output_result:
                    simplified_case = {}
                    for key, value in output_result[test_case_key].items():
                        if isinstance(value, np.ndarray):
                            simplified_case[key] = {
                                "shape": list(value.shape),
                                "dtype": str(value.dtype),
                                "sample_values": value.flatten()[:10].tolist() if value.size > 0 else []
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
            "results": output_results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filepath}")
    
    def close(self):
        """Close MongoDB connection."""
        self.client.close()


def main():
    """
    Main entry point.
    
    Two modes:
    1) Single-operator test (uncomment mode 1, comment mode 2)
    2) Batch test all operators (current mode)
    """
    # ==================== Test config ====================
    max_iterations = 3  # Max iterations per case
    num_test_cases = 3  # Cases per operator
    
    # Batch range config (mode 2 only)
    # None means all operators
    # (start, end) means operators start..end inclusive, 1-based
    # Examples:
    #   operator_range = None          # test all operators
    #   operator_range = (1, 10)       # test operators 1..10
    #   operator_range = (10, 20)      # test operators 10..20
    #   operator_range = (50, 100)     # test operators 50..100
    operator_range = (251, 465)
    # ====================================================
    
    # ==================== Mode 1: single-operator test ====================
    # To run a single operator, uncomment the triple-quoted block below and comment mode 2
    """
    operator_name = "torch.nn.Dropout2d"  # Operator to test
    
    print("="*80)
    print("LLM-based PyTorch vs MindSpore operator comparison test framework")
    print("="*80)
    print(f"📌 Operator: {operator_name}")
    print(f"📌 Iterations per case: {max_iterations}")
    print(f"📌 Test case count: {num_test_cases}")
    print("="*80)
    
    # Initialize comparator
    comparator = LLMEnhancedComparator()
    
    try:
        # Run LLM-enhanced test
        results, stats = comparator.llm_enhanced_test_operator(
            operator_name, 
            max_iterations=max_iterations,
            num_test_cases=num_test_cases
        )
        
        # Save results
        comparator.save_results(operator_name, results, stats)
        
        # Print detailed summary
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
        
        print(f"\nTested {len(case_groups)} test cases:")
        for case_num in sorted(case_groups.keys()):
            case_results = case_groups[case_num]
            print(f"\nTest case {case_num} ({len(case_results)} iterations):")
            for i, result in enumerate(case_results):
                exec_result = result["execution_result"]
                llm_op = result.get("llm_operation", {})
                print(f"  Iteration {i+1}:")
                print(f"    - Status: {exec_result['status']}")
                print(f"    - PyTorch success: {exec_result['torch_success']}")
                print(f"    - MindSpore success: {exec_result['mindspore_success']}")
                print(f"    - Results match: {exec_result['results_match']}")
                print(f"    - LLM operation: {llm_op.get('operation', 'N/A')}")
        
    finally:
        # Close connection
        comparator.close()
        print("\n✅ Program completed")
    """
    # ==================== End mode 1 ====================
    
    # ==================== Mode 2: batch test all operators ====================
    # To batch test all operators, uncomment mode 2 and comment mode 1
    
    print("="*80)
    print("LLM-based PyTorch vs MindSpore batch operator comparison framework")
    print("="*80)
    print(f"📌 Iterations per operator: {max_iterations}")
    print(f"📌 Test cases per operator: {num_test_cases}")
    if operator_range is not None:
        print(f"📌 Test range: operator {operator_range[0]} to {operator_range[1]}")
    else:
        print(f"📌 Test range: all operators")
    print("="*80)
    
    # Initialize comparator
    comparator = LLMEnhancedComparator()
    
    # Record start time
    import time
    start_time = time.time()
    start_datetime = datetime.now()
    
    try:
        # Fetch all PyTorch operators from DB
        print("\n🔍 Fetching all operators from database...")
        all_operators = list(comparator.collection.find({}, {"api": 1}))
        all_operator_names = [doc["api"] for doc in all_operators if "api" in doc]
        
        print(f"✅ Total operators in DB: {len(all_operator_names)}")
        
        # Filter operators by range
        if operator_range is not None:
            start_idx, end_idx = operator_range
            # Convert to 0-based indices
            start_idx = max(1, start_idx) - 1
            end_idx = min(len(all_operator_names), end_idx)
            operator_names = all_operator_names[start_idx:end_idx]
            print(f"📌 Test range: operator {start_idx + 1} to {end_idx}")
            print(f"📋 Will test {len(operator_names)} operators")
        else:
            operator_names = all_operator_names
            print(f"📋 Will test all {len(operator_names)} operators")
        
        # Filter operators without MindSpore equivalents (from mapping table)
        print(f"\n🔍 Filtering operators without MindSpore equivalents...")
        original_count = len(operator_names)
        filtered_operator_names = []
        skipped_operators = []
        
        for op_name in operator_names:
            _, ms_api, mapping_method = comparator.convert_api_name(op_name)
            if ms_api is not None:
                filtered_operator_names.append(op_name)
            else:
                skipped_operators.append((op_name, mapping_method))
        
        operator_names = filtered_operator_names
        skipped_count = original_count - len(operator_names)
        
        print(f"✅ Filtered: {original_count} original, {skipped_count} skipped, {len(operator_names)} remaining")
        if skipped_operators:
            print(f"⏭️ Skipped operators (first 10): {', '.join([f'{op}({reason})' for op, reason in skipped_operators[:10]])}{'...' if len(skipped_operators) > 10 else ''}")
        
        print(f"📋 Operator list: {', '.join(operator_names[:10])}{'...' if len(operator_names) > 10 else ''}\n")
        
        # Summary of all operator results
        all_operators_summary = []
        
        # Create batch log file
        batch_log_file = os.path.join(comparator.result_dir, f"batch_test_log_{start_datetime.strftime('%Y%m%d_%H%M%S')}.txt")
        log_file = open(batch_log_file, 'w', encoding='utf-8')
        
        # Write log header
        log_file.write("="*80 + "\n")
        log_file.write("Batch Test Log\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("Test config:\n")
        log_file.write(f"  - Iterations per operator: {max_iterations}\n")
        log_file.write(f"  - Test cases per operator: {num_test_cases}\n")
        log_file.write(f"  - Total operators in DB: {len(all_operator_names)}\n")
        if operator_range is not None:
            log_file.write(f"  - Test range: {operator_range[0]} to {operator_range[1]}\n")
        log_file.write(f"  - Skipped operators without equivalents: {skipped_count}\n")
        log_file.write(f"  - Operators actually tested: {len(operator_names)}\n")
        log_file.write("="*80 + "\n\n")
        log_file.flush()
        
        # Test each operator
        for idx, operator_name in enumerate(operator_names, 1):
            print("\n" + "🔷"*40)
            print(f"🎯 [{idx}/{len(operator_names)}] Starting operator: {operator_name}")
            print("🔷"*40)
            
            try:
                # Run LLM-enhanced test
                results, stats = comparator.llm_enhanced_test_operator(
                    operator_name, 
                    max_iterations=max_iterations,
                    num_test_cases=num_test_cases
                )
                
                # Save results
                if results:
                    comparator.save_results(operator_name, results, stats)
                    
                    # Record summary info
                    all_operators_summary.append({
                        "operator": operator_name,
                        "total_iterations": len(results),
                        "llm_generated_cases": stats.get("llm_generated_cases", 0),
                        "successful_cases": stats.get("successful_cases", 0),
                        "status": "completed"
                    })
                    
                    print(f"\n✅ Operator {operator_name} completed")
                    print(f"   - Total iterations: {len(results)}")
                    print(f"   - LLM-generated cases: {stats.get('llm_generated_cases', 0)}")
                    print(f"   - Successful cases: {stats.get('successful_cases', 0)}")
                    
                    # Write log
                    log_file.write(f"[{idx}/{len(operator_names)}] {operator_name}\n")
                    log_file.write("  Status: ✅ completed\n")
                    log_file.write(f"  Total iterations: {len(results)}\n")
                    log_file.write(f"  LLM-generated cases: {stats.get('llm_generated_cases', 0)}\n")
                    log_file.write(f"  Successful cases: {stats.get('successful_cases', 0)}\n")
                    if stats.get('llm_generated_cases', 0) > 0:
                        success_rate = (stats.get('successful_cases', 0) / stats.get('llm_generated_cases', 0)) * 100
                        log_file.write(f"  Success rate: {success_rate:.2f}%\n")
                    log_file.write("\n")
                    log_file.flush()
                else:
                    all_operators_summary.append({
                        "operator": operator_name,
                        "total_iterations": 0,
                        "llm_generated_cases": 0,
                        "successful_cases": 0,
                        "status": "no_results"
                    })
                    print(f"\n⚠️ Operator {operator_name} has no results")
                    
                    # Write log
                    log_file.write(f"[{idx}/{len(operator_names)}] {operator_name}\n")
                    log_file.write("  Status: ⚠️ no results\n\n")
                    log_file.flush()
                    
            except Exception as e:
                print(f"\n❌ Operator {operator_name} failed: {e}")
                all_operators_summary.append({
                    "operator": operator_name,
                    "total_iterations": 0,
                    "llm_generated_cases": 0,
                    "successful_cases": 0,
                    "status": "failed",
                    "error": str(e)
                })
                
                # Write log
                log_file.write(f"[{idx}/{len(operator_names)}] {operator_name}\n")
                log_file.write("  Status: ❌ failed\n")
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
        print("\n" + "="*80)
        print("📊 Batch Test Overall Summary")
        print("="*80)
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
        
        print(f"\n📈 Statistics:")
        print(f"   - Total LLM-generated cases: {total_llm_cases}")
        print(f"   - Total successful cases: {total_successful_cases}")
        if total_llm_cases > 0:
            success_rate = (total_successful_cases / total_llm_cases) * 100
            print(f"   - Success rate: {success_rate:.2f}%")
        print(f"   - Total iterations: {total_iterations}")
        print(f"\n⏱️ Runtime: {hours}h {minutes}m {seconds}s")
        
        # Write log stats
        log_file.write("="*80 + "\n")
        log_file.write("Overall Stats\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total runtime: {hours}h {minutes}m {seconds}s ({total_duration:.2f}s)\n\n")
        
        log_file.write("Operator Results:\n")
        log_file.write(f"  - Total operators: {len(operator_names)}\n")
        log_file.write(f"  - Completed: {completed_count} ({completed_count/len(operator_names)*100:.2f}%)\n")
        log_file.write(f"  - Failed: {failed_count} ({failed_count/len(operator_names)*100:.2f}%)\n")
        log_file.write(f"  - No results: {no_results_count} ({no_results_count/len(operator_names)*100:.2f}%)\n\n")
        
        log_file.write("LLM Case Stats:\n")
        log_file.write(f"  - Total LLM-generated cases: {total_llm_cases}\n")
        log_file.write(f"  - Cases where both frameworks succeeded: {total_successful_cases}\n")
        if total_llm_cases > 0:
            success_rate = (total_successful_cases / total_llm_cases) * 100
            log_file.write(f"  - Success rate: {success_rate:.2f}%\n")
        log_file.write(f"  - Total iterations: {total_iterations}\n")
        if completed_count > 0:
            avg_llm_cases = total_llm_cases / completed_count
            avg_successful = total_successful_cases / completed_count
            log_file.write(f"  - Avg LLM cases per operator: {avg_llm_cases:.2f}\n")
            log_file.write(f"  - Avg successful cases per operator: {avg_successful:.2f}\n")
        
        log_file.write("\n" + "="*80 + "\n")
        log_file.write("See per-operator logs for details\n")
        log_file.write("="*80 + "\n")
        log_file.close()
        
        print(f"\n💾 Batch log saved to: {batch_log_file}")
        
        # Save overall summary to JSON
        summary_file = os.path.join(comparator.result_dir, f"batch_test_summary_{start_datetime.strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_config": {
                    "max_iterations": max_iterations,
                    "num_test_cases": num_test_cases,
                    "operator_range": f"{operator_range[0]}-{operator_range[1]}" if operator_range else "all",
                    "total_operators_in_db": len(all_operator_names)
                },
                "time_info": {
                    "start_time": start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    "end_time": end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    "total_duration_seconds": total_duration,
                    "duration_formatted": f"{hours}h {minutes}m {seconds}s"
                },
                "summary": {
                    "tested_operators": len(operator_names),
                    "completed": completed_count,
                    "failed": failed_count,
                    "no_results": no_results_count,
                    "total_llm_generated_cases": total_llm_cases,
                    "total_successful_cases": total_successful_cases,
                    "success_rate": f"{(total_successful_cases / total_llm_cases * 100):.2f}%" if total_llm_cases > 0 else "N/A",
                    "total_iterations": total_iterations,
                    "avg_llm_cases_per_operator": f"{total_llm_cases / completed_count:.2f}" if completed_count > 0 else "N/A",
                    "avg_successful_per_operator": f"{total_successful_cases / completed_count:.2f}" if completed_count > 0 else "N/A"
                },
                "operators": all_operators_summary
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 JSON summary saved to: {summary_file}")
        
    finally:
        # Close connection
        comparator.close()
        print("\n✅ Batch test completed")
    
    # ==================== End mode 2 ====================


if __name__ == "__main__":
    main()
