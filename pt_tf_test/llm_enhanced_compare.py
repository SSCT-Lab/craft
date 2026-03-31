# ./pt_tf_test/llm_enhanced_compare.py
"""
PyTorch and TensorFlow operator comparison testing framework based on LLM Use large models for test case repair and mutation to improve test case availability and coverage
"""

import pymongo
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import copy
import os
import sys
import json
import argparse
import traceback
import time
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, RLock

# Add the project root directory to the path so that component modules can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from component.doc.doc_crawler_factory import get_doc_content

# ==================== constant definition ====================
DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_NUM_CASES = 3
DEFAULT_WORKERS = 4
DEFAULT_LLM_WORKERS = DEFAULT_WORKERS

class LLMEnhancedComparator:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "freefuzz-torch",
                 key_path: str = DEFAULT_KEY_PATH,
                 model: str = DEFAULT_MODEL,
                 print_lock: Lock = None, llm_workers: int = DEFAULT_LLM_WORKERS):
        """
        Initializing LLM-based PyTorch and TensorFlow comparators
        
        Args:
            mongo_uri: MongoDBconnectURI
            db_name: Database name
            key_path: API keyfile path
            model: LLMModel name
            print_lock: Print lock (for thread-safe output during concurrency)）
            llm_workers: LLMNumber of concurrent calling threads
        """
        self.model = model
        self.print_lock = print_lock or Lock()
        self.llm_workers = max(1, int(llm_workers))
        self.execution_lock = RLock()
        self.stats_lock = Lock()

        # MongoDBconnect
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["argVS"]
        
        # Initialize LLM client (Alibaba Qianwen large model）
        # Read the key from the specified key file first, otherwise use environment variables
        api_key = self._load_api_key(key_path)
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # Load API mapping table
        self.api_mapping = self.load_api_mapping()
        
        # Create a result storage directory (under the pt_tf_test directory）
        self.result_dir = os.path.join(ROOT_DIR, "pt_tf_test", "pt_tf_log_1")
        os.makedirs(self.result_dir, exist_ok=True)
        self._safe_print(f"📁 Results storage directory: {self.result_dir}")
        
        # Fixed random seed to ensure reproducibility
        self.random_seed = 42
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        
        # List of obsolete PyTorch operators
        self.deprecated_torch_apis = {
            "torch.symeig": "Already inPyTorch 1.9removed from the version, please usetorch.linalg.eighsubstitute"
        }

    def _safe_print(self, msg: str, end: str = "\n"):
        """Thread-safe printing"""
        with self.print_lock:
            print(msg, end=end, flush=True)
    
    def _load_api_key(self, key_path: str = DEFAULT_KEY_PATH) -> str:
        """
        Load Alibaba Cloud API key                  Read from the specified file first, if the file does not exist, use environment variables DASHSCOPE_API_KEY
        
        Args:
            key_path: API keyfile path
        
        Returns:
            API key string
        """
        if not os.path.isabs(key_path):
            key_file = os.path.join(ROOT_DIR, key_path)
        else:
            key_file = key_path
        
        # Read from file first
        if os.path.exists(key_file):
            try:
                with open(key_file, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
                if api_key:
                    self._safe_print(f"✅ Load API key from file: {key_file}")
                    return api_key
            except Exception as e:
                self._safe_print(f"⚠️ Failed to read key file: {e}")
        
        # Fallback to environment variables
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            self._safe_print(f"✅ Load API key from environment variable: DASHSCOPE_API_KEY")
            return api_key
        
        # None found
        self._safe_print("❌ API key not found, please make sure aliyun.key File exists or the DASHSCOPE_API_KEY environment variable is set")
        return ""
    
    def load_api_mapping(self) -> Dict[str, Dict[str, str]]:
        """Load the PyTorch to TensorFlow API mapping table"""
        # Use updated mapping file
        mapping_file = os.path.join(ROOT_DIR, "component", "data", "api_mappings_final.csv")
        try:
            df = pd.read_csv(mapping_file)
            mapping = {}
            
            for _, row in df.iterrows():
                # The column names of the new mapping file are pytorch-api and tensorflow-api
                pt_api = str(row["pytorch-api"]).strip()
                tf_api = str(row["tensorflow-api"]).strip()
                mapping[pt_api] = {"tf_api": tf_api, "note": ""}
            
            self._safe_print(f"✅ Successfully loaded API mapping table, total {len(mapping)} bar mapping")
            self._safe_print(f"📄 mapping file: {mapping_file}")
            return mapping
        except Exception as e:
            self._safe_print(f"❌ Failed to load API mapping table: {e}")
            return {}
    
    def is_class_based_api(self, api_name: str) -> bool:
        """Determine whether the API is class-based"""
        parts = api_name.split(".")
        if len(parts) >= 2:
            name = parts[-1]
            return any(c.isupper() for c in name)
        return False
    
    # def convert_class_to_functional(self, torch_api: str) -> Tuple[Optional[str], Optional[str]]:
    #     """Convert class form API to function form"""
    #     if not self.is_class_based_api(torch_api):
    #         return None, None
    #     
    #     parts = torch_api.split(".")
    #     if len(parts) >= 3 and parts[1] == "nn":
    #         class_name = parts[-1]
    #         
    #         # Improved regular expressions: correctly handle consecutive uppercase letters
    #         # 1. first"Lowercase letters followed by uppercase letters"Insert an underline at the position
    #         # 2. again"In consecutive capital letters, before the last capital letter"Insert an underscore (if followed by a lowercase letter）
    #         func_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', class_name)  # aB -> a_B
    #         func_name = re.sub('([A-Z]+)([A-Z][a-z])', r'\1_\2', func_name)  # ABCDef -> ABC_Def
    #         func_name = func_name.lower()
    #         
    #         torch_func_api = f"torch.nn.functional.{func_name}"
    #         tf_func_api = f"tf.nn.functional.{func_name}"
    #         
    #         return torch_func_api, tf_func_api
    #     
    #     return None, None
    
    def convert_api_name(self, torch_api: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        Convert PyTorch API to TensorFlow API                  Completely based on api_mappings_final.csv Mapping table lookup, no more manual name translation。
        
        Returns:
            (convertedPyTorch API, convertedTensorFlow API, Mapping method description)             - If found in the mapping table and there is a valid TensorFlow API → Return the mapped API             - If found in the mapping table but the value is "No corresponding implementation" → return (torch_api, None, "No corresponding implementation")
            - If it is not found in the mapping table API → return (torch_api, None, "Not found in mapping table")
        """
        # Check mapping table
        if torch_api in self.api_mapping:
            tf_api = self.api_mapping[torch_api]["tf_api"]
            
            # Check if it is "No corresponding implementation"
            if tf_api == "No corresponding implementation" or tf_api == "NONE" or not tf_api:
                return torch_api, None, "No corresponding implementation"
            else:
                return torch_api, tf_api, "mapping table"
        
        # There is no such API in the mapping table, so manual conversion will no longer be performed and will be returned directly. None
        return torch_api, None, "Not found in mapping table"

    
    def get_operator_function(self, api_name: str, framework: str = "torch"):
        """Get operator function"""
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
                elif framework == "tensorflow" and parts[0] == "tf":
                    if len(parts) == 2:
                        return getattr(tf, parts[1])
                    elif len(parts) == 3:
                        module = getattr(tf, parts[1])
                        return getattr(module, parts[2])
                    elif len(parts) == 4:
                        module1 = getattr(tf, parts[1])
                        module2 = getattr(module1, parts[2])
                        return getattr(module2, parts[3])
                    elif len(parts) == 5:
                        module1 = getattr(tf, parts[1])
                        module2 = getattr(module1, parts[2])
                        module3 = getattr(module2, parts[3])
                        return getattr(module3, parts[4])
            return None
        except AttributeError:
            return None
    
    def convert_key(self, key: str, tensorflow_api: str = "") -> str:
        """Conversion parameter name"""
        key_mapping = {
            "input": "x",
            "other": "y",
        }
        return key_mapping.get(key, key)
    
    def should_skip_param(self, key: str, tensorflow_api: str) -> bool:
        """Determine whether a parameter should be skipped"""
        common_skip_params = ["layout", "requires_grad", "out"]
        skip_params = {
            # Skip parameters for specific APIs can be added as needed
        }
        
        if key in common_skip_params:
            return True
        
        if tensorflow_api in skip_params:
            return key in skip_params[tensorflow_api]
        
        return False
    
    def generate_numpy_data(self, data: Any) -> np.ndarray:
        """
        Generate numpy array as shared data source                  Supported dtype formats:         -With torch prefix：torch.float32, torch.bool, torch.int64Wait         - without prefix：float32, bool, int64Wait         - numpy format：float32, bool_, int64wait
        """
        if isinstance(data, dict):
            # Extended dtype mapping table, supporting multiple formats
            dtype_map = {
                # torchFormat (with prefix）
                "torch.float64": np.float64,
                "torch.float32": np.float32,
                "torch.int64": np.int64,
                "torch.int32": np.int32,
                "torch.bool": np.bool_,
                "torch.uint8": np.uint8,
                # Format without torch prefix (LLM may return this format）
                "float64": np.float64,
                "float32": np.float32,
                "int64": np.int64,
                "int32": np.int32,
                "bool": np.bool_,
                "uint8": np.uint8,
                # numpyFormat
                "bool_": np.bool_,
                "float": np.float32,
                "int": np.int64,
            }
            
            shape = data.get("shape", [])
            dtype_str = data.get("dtype", "torch.float32")
            dtype = dtype_map.get(dtype_str, np.float32)
            
            # Keep silent: automatically fall back to default value when dtype is not recognized
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
        """Prepare shared numpy data, ensuring PyTorch and TensorFlow use the same inputs"""
        shared_data = {}
        api_name = document.get("api", "")
        
        # For class-form APIs, if there is no input parameter, a default input is generated.
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
        
        # Handle other parameters in the document
        exclude_keys = ["_id", "api"]
        for key, value in document.items():
            if key not in exclude_keys:
                # For variadic arguments (starting with*at the beginning), save the original value directly without conversion
                # The conversion will be done inprepare_arguments_torch/tfcompleted in
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
        """Convert data to PyTorch tensors"""
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
    
    def convert_to_tensor_tensorflow(self, data: Any, numpy_data: np.ndarray = None) -> tf.Tensor:
        """Convert data to TensorFlow tensors"""
        if numpy_data is not None:
            return tf.convert_to_tensor(numpy_data.copy())
        
        if isinstance(data, dict):
            numpy_data = self.generate_numpy_data(data)
            return tf.convert_to_tensor(numpy_data.copy())
        elif isinstance(data, (int, float)):
            return tf.convert_to_tensor(data)
        elif isinstance(data, list):
            return tf.convert_to_tensor(data)
        else:
            return tf.convert_to_tensor(data)
    
    def prepare_arguments_torch(self, test_case: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Prepare parameters for PyTorch                  Note：
        1. fortorch.whereand other functions, the parameters need to be passed in order as positional parameters.：
           - torch.where(condition, x, y) or torch.where(condition, input, other)
        2. For*Parameters at the beginning (such as*tensors），Indicates variable parameters, which need to be unpacked as positional parameters.
        """
        args = []
        kwargs = {}
        
        # First check if there are variadic arguments (starting with*Parameters at the beginning）
        varargs_key = None
        for key in test_case.keys():
            if key.startswith('*'):
                varargs_key = key
                break
        
        # If there are variadic arguments, unpack them as positional arguments
        if varargs_key:
            varargs_value = test_case[varargs_key]
            if isinstance(varargs_value, list):
                for item in varargs_value:
                    if isinstance(item, dict) and "shape" in item:
                        # This is a tensor description, generate numpy data and convert
                        numpy_data = self.generate_numpy_data(item)
                        args.append(self.convert_to_tensor_torch(None, numpy_data))
                    elif isinstance(item, list):
                        # Nested lists, recursive processing
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
        
        # Process positional arguments sequentially：condition, x/input, y/other
        # These parameters need to be passed as positional parameters, not keyword parameters
        positional_params = ["condition", "x", "y", "input", "other"]
        
        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, np.ndarray):
                    args.append(self.convert_to_tensor_torch(None, value))
                else:
                    # Scalar values ​​are added directly
                    args.append(value)
        
        # Handle additional parameters (as keyword arguments）
        for key, value in test_case.items():
            if key not in positional_params + ["api"]:
                if isinstance(value, np.ndarray):
                    kwargs[key] = self.convert_to_tensor_torch(None, value)
                else:
                    kwargs[key] = value
        
        return args, kwargs
    
    def prepare_arguments_tensorflow(self, test_case: Dict[str, Any], tensorflow_api: str) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Prepare parameters for TensorFlow                  Note：
        1. fortf.wherefunction, the parameters also need to be passed in order as positional parameters.
        2. For*Parameters at the beginning (such as*tensors），Indicates variable parameters, which need to be unpacked as positional parameters.
        """
        args = []
        kwargs = {}
        
        # First check if there are variadic arguments (starting with*Parameters at the beginning）
        varargs_key = None
        for key in test_case.keys():
            if key.startswith('*'):
                varargs_key = key
                break
        
        # If there are variadic arguments, unpack them as positional arguments
        if varargs_key:
            varargs_value = test_case[varargs_key]
            if isinstance(varargs_value, list):
                for item in varargs_value:
                    if isinstance(item, dict) and "shape" in item:
                        # This is a tensor description, generate numpy data and convert
                        numpy_data = self.generate_numpy_data(item)
                        args.append(self.convert_to_tensor_tensorflow(None, numpy_data))
                    elif isinstance(item, list):
                        # Nested lists, recursive processing
                        nested_tensors = []
                        for nested_item in item:
                            if isinstance(nested_item, dict) and "shape" in nested_item:
                                numpy_data = self.generate_numpy_data(nested_item)
                                nested_tensors.append(self.convert_to_tensor_tensorflow(None, numpy_data))
                            elif isinstance(nested_item, np.ndarray):
                                nested_tensors.append(self.convert_to_tensor_tensorflow(None, nested_item))
                            else:
                                nested_tensors.append(nested_item)
                        args.extend(nested_tensors)
                    elif isinstance(item, np.ndarray):
                        args.append(self.convert_to_tensor_tensorflow(None, item))
                    else:
                        args.append(item)
            return args, kwargs
        
        # Process positional arguments sequentially：condition, x/input, y/other
        positional_params = ["condition", "x", "y", "input", "other"]
        
        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, np.ndarray):
                    args.append(self.convert_to_tensor_tensorflow(None, value))
                else:
                    # Scalar values ​​are added directly
                    args.append(value)
        
        # Handle additional parameters (as keyword arguments）
        for key, value in test_case.items():
            if key not in positional_params + ["api"]:
                if self.should_skip_param(key, tensorflow_api):
                    continue
                
                if isinstance(value, np.ndarray):
                    kwargs[key] = self.convert_to_tensor_tensorflow(None, value)
                else:
                    kwargs[key] = value
        
        return args, kwargs
    
    def compare_tensors(self, torch_result, tensorflow_result, tolerance: float = 1e-5) -> Tuple[bool, str]:
        """Compare two tensors for equality"""
        try:
            # Convert to numpy for comparison
            if hasattr(torch_result, 'detach'):
                torch_np = torch_result.detach().cpu().numpy()
            else:
                torch_np = np.array(torch_result)
            
            if hasattr(tensorflow_result, 'numpy'):
                tensorflow_np = tensorflow_result.numpy()
            else:
                tensorflow_np = np.array(tensorflow_result)
            
            # Check shape
            if torch_np.shape != tensorflow_np.shape:
                return False, f"Shape mismatch: PyTorch {torch_np.shape} vs TensorFlow {tensorflow_np.shape}"
            
            # Check if dtype is a boolean type
            if torch_np.dtype == np.bool_ or tensorflow_np.dtype == np.bool_:
                if np.array_equal(torch_np, tensorflow_np):
                    return True, "boolean match"
                else:
                    diff_count = np.sum(torch_np != tensorflow_np)
                    return False, f"boolean mismatch, number of differences: {diff_count}"
            
            # Check value
            if np.allclose(torch_np, tensorflow_np, atol=tolerance, rtol=tolerance, equal_nan=True):
                return True, "numerical matching"
            else:
                max_diff = np.max(np.abs(torch_np - tensorflow_np))
                return False, f"Numerical mismatch, maximum difference: {max_diff}"
        
        except Exception as e:
            return False, f"An error occurred during comparison: {str(e)}"
    
    def execute_test_case(self, torch_api: str, tensorflow_api: str, torch_test_case: Dict[str, Any], tensorflow_test_case: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a single test case
        
        Args:
            torch_api: PyTorch APIname
            tensorflow_api: TensorFlow APIname
            torch_test_case: PyTorchTest case (contains parameter information）
            tensorflow_test_case: TensorFlowTest case (contains parameter information）
        
        Returns:
            Execution result dictionary
        """
        result = {
            "torch_api": torch_api,
            "tensorflow_api": tensorflow_api,
            "torch_success": False,
            "tensorflow_success": False,
            "results_match": False,
            "torch_error": None,
            "tensorflow_error": None,
            "comparison_error": None,
            "torch_shape": None,
            "tensorflow_shape": None,
            "torch_dtype": None,
            "tensorflow_dtype": None,
            "status": "unknown"
        }
        
        # If tensorflow_test_case is not provided, torch_test_case is used (backwards compatible）
        if tensorflow_test_case is None:
            tensorflow_test_case = torch_test_case
        
        # Determine whether it is a class operator
        is_class_api = self.is_class_based_api(torch_api)
        
        # testPyTorch
        torch_result = None
        try:
            torch_func = self.get_operator_function(torch_api, "torch")
            if torch_func is None:
                result["torch_error"] = f"PyTorchoperator {torch_api} not found"
            else:
                args, kwargs = self.prepare_arguments_torch(torch_test_case)
                
                if is_class_api:
                    # For class operators, they need to be instantiated first and then called
                    # Extract initialization parameters (non-input parameters) from kwargs）
                    init_kwargs = {k: v for k, v in kwargs.items() if k != 'input'}
                    # instantiate class
                    torch_instance = torch_func(**init_kwargs)
                    # Get input data (maybe in args or kwargs）
                    if 'input' in kwargs:
                        input_data = kwargs['input']
                    elif len(args) > 0:
                        input_data = args[0]
                    else:
                        # If there is no input parameter, try to use the default input
                        raise ValueError("Class operator is missing input parameters")
                    
                    # Call instance (forward propagation）
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
        
        # testTensorFlow
        tensorflow_result = None
        try:
            tensorflow_func = self.get_operator_function(tensorflow_api, "tensorflow")
            if tensorflow_func is None:
                result["tensorflow_error"] = f"TensorFlowoperator {tensorflow_api} not found"
            else:
                args, kwargs = self.prepare_arguments_tensorflow(tensorflow_test_case, tensorflow_api)
                
                if is_class_api:
                    # For class operators, they need to be instantiated first and then called
                    # Extract initialization parameters from kwargs (notx/inputparameter）
                    init_kwargs = {k: v for k, v in kwargs.items() if k not in ['x', 'input']}
                    # instantiate class
                    tensorflow_instance = tensorflow_func(**init_kwargs)
                    # Get input data (maybe in args or kwargs）
                    if 'x' in kwargs:
                        input_data = kwargs['x']
                    elif 'input' in kwargs:
                        input_data = kwargs['input']
                    elif len(args) > 0:
                        input_data = args[0]
                    else:
                        # If there is no input parameter, try to use the default input
                        raise ValueError("Class operator missinginput/xparameter")
                    
                    # Call instance (forward propagation）
                    tensorflow_result = tensorflow_instance(input_data)
                else:
                    # For function operators, call directly
                    tensorflow_result = tensorflow_func(*args, **kwargs)
                
                result["tensorflow_success"] = True
                result["tensorflow_shape"] = list(tensorflow_result.shape) if hasattr(tensorflow_result, 'shape') else None
                result["tensorflow_dtype"] = str(tensorflow_result.dtype) if hasattr(tensorflow_result, 'dtype') else None
        except Exception as e:
            result["tensorflow_error"] = str(e)
            result["tensorflow_traceback"] = traceback.format_exc()
        
        # Compare results
        if result["torch_success"] and result["tensorflow_success"]:
            try:
                is_match, comparison_msg = self.compare_tensors(torch_result, tensorflow_result)
                result["results_match"] = is_match
                result["comparison_error"] = comparison_msg if not is_match else None
                result["status"] = "compared"
            except Exception as e:
                result["comparison_error"] = str(e)
                result["status"] = "comparison_failed"
        elif result["torch_success"] and not result["tensorflow_success"]:
            result["status"] = "tensorflow_failed"
        elif not result["torch_success"] and result["tensorflow_success"]:
            result["status"] = "torch_failed"
        else:
            result["status"] = "both_failed"
        
        return result

    def _execute_test_case_sequential(self, torch_api: str, tensorflow_api: str,
                                      torch_test_case: Dict[str, Any],
                                      tensorflow_test_case: Dict[str, Any] = None) -> Dict[str, Any]:
        """Sequential execution of operators (guaranteeing non-concurrent execution through locks)）"""
        with self.execution_lock:
            return self.execute_test_case(torch_api, tensorflow_api, torch_test_case, tensorflow_test_case)
    
    def _fetch_api_docs(self, torch_api: str, tensorflow_api: str) -> Tuple[str, str]:
        """
        Crawl the API documentation of PyTorch and TensorFlow
        
        Args:
            torch_api: PyTorch APIname
            tensorflow_api: TensorFlow APIname
        
        Returns:
            (PyTorchDocument content, TensorFlowDocument content)
        """
        # Minimum length threshold for document validity judgment
        MIN_DOC_LENGTH = 300
        
        torch_doc = ""
        tensorflow_doc = ""
        
        try:
            self._safe_print(f"    📖 Crawling PyTorch documentation: {torch_api}")
            torch_doc = get_doc_content(torch_api, "pytorch")
            # Determine whether the document is valid：1. Content is not empty 2. No error message included 3. length exceeds threshold
            if (torch_doc 
                and "Unable" not in torch_doc 
                and "not supported" not in torch_doc
                and len(torch_doc.strip()) > MIN_DOC_LENGTH):
                # Truncate overly long documents to savetoken
                if len(torch_doc) > 3000:
                    torch_doc = torch_doc[:3000] + "\n... (doc truncated)"
                self._safe_print(f"    ✅ PyTorch Document obtained successfully ({len(torch_doc)} character)")
            else:
                doc_len = len(torch_doc.strip()) if torch_doc else 0
                torch_doc = f"Unable to fetch documentation for {torch_api} (length: {doc_len}, min required: {MIN_DOC_LENGTH})"
                self._safe_print(f"    ⚠️ PyTorch Document is invalid or too short")
        except Exception as e:
            torch_doc = f"Failed to fetch documentation: {str(e)}"
            self._safe_print(f"    ❌ PyTorch Document crawling failed: {e}")
        
        try:
            self._safe_print(f"    📖 Crawling TensorFlow documentation: {tensorflow_api}")
            tensorflow_doc = get_doc_content(tensorflow_api, "tensorflow")
            # Determine whether the document is valid：1. Content is not empty 2. No error message included 3. length exceeds threshold
            if (tensorflow_doc 
                and "Unable" not in tensorflow_doc 
                and "not supported" not in tensorflow_doc
                and len(tensorflow_doc.strip()) > MIN_DOC_LENGTH):
                # Truncate overly long documents to savetoken
                if len(tensorflow_doc) > 3000:
                    tensorflow_doc = tensorflow_doc[:3000] + "\n... (doc truncated)"
                self._safe_print(f"    ✅ TensorFlow Document obtained successfully ({len(tensorflow_doc)} character)")
            else:
                doc_len = len(tensorflow_doc.strip()) if tensorflow_doc else 0
                tensorflow_doc = f"Unable to fetch documentation for {tensorflow_api} (length: {doc_len}, min required: {MIN_DOC_LENGTH})"
                self._safe_print(f"    ⚠️ TensorFlow Document is invalid or too short")
        except Exception as e:
            tensorflow_doc = f"Failed to fetch documentation: {str(e)}"
            self._safe_print(f"    ❌ TensorFlow Document crawling failed: {e}")
        
        return torch_doc, tensorflow_doc
        
        return torch_doc, tensorflow_doc
    
    def _build_llm_prompt(self, execution_result: Dict[str, Any], torch_test_case: Dict[str, Any], tensorflow_test_case: Dict[str, Any], torch_doc: str = "", tensorflow_doc: str = "") -> str:
        """Tips for building LLM"""
        torch_api = execution_result.get("torch_api", "")
        tensorflow_api = execution_result.get("tensorflow_api", "")
        status = execution_result.get("status", "")
        torch_success = execution_result.get("torch_success", False)
        tensorflow_success = execution_result.get("tensorflow_success", False)
        results_match = execution_result.get("results_match", False)
        torch_error = execution_result.get("torch_error", "")
        tensorflow_error = execution_result.get("tensorflow_error", "")
        comparison_error = execution_result.get("comparison_error", "")
        
        # Simplify PyTorch test cases to reduce token consumption
        simplified_torch_test_case = {}
        for key, value in torch_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_torch_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_torch_test_case[key] = value
        
        # Simplify TensorFlow test cases to reduce token consumption
        simplified_tensorflow_test_case = {}
        for key, value in tensorflow_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_tensorflow_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_tensorflow_test_case[key] = value
        
        # Build PyTorch parameter example
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
        
        # Build TensorFlow parameter example
        tf_param_examples = []
        for key, value in simplified_tensorflow_test_case.items():
            if key == "api":
                continue
            if isinstance(value, dict) and "shape" in value:
                tf_param_examples.append(f'    "{key}": {json.dumps(value)}')
            elif isinstance(value, (int, float)):
                tf_param_examples.append(f'    "{key}": {value}')
            else:
                tf_param_examples.append(f'    "{key}": {json.dumps(value)}')
        
        tf_param_example_str = ",\n".join(tf_param_examples) if tf_param_examples else '    "x": {"shape": [2, 3], "dtype": "float32"}'
        
        # Build API documentation section
        doc_section = ""
        if torch_doc or tensorflow_doc:
            doc_section = "\n## Official API document reference\n\n"
            if torch_doc:
                doc_section += f"### PyTorch {torch_api} document\n```\n{torch_doc}\n```\n\n"
            if tensorflow_doc:
                doc_section += f"### TensorFlow {tensorflow_api} document\n```\n{tensorflow_doc}\n```\n\n"
        
        prompt = f"""Please analyze the execution results of the following operator test cases in PyTorch and TensorFlow frameworks, and make repairs or mutations of the test cases based on the results.（fuzzing）。

## Test information
- **PyTorch API**: {torch_api}
- **TensorFlow API**: {tensorflow_api}
{doc_section}
## Execution result
- **Execution status**: {status}
- **PyTorchExecuted successfully**: {torch_success}
- **TensorFlowExecuted successfully**: {tensorflow_success}
- **Are the results consistent?**: {results_match}

## error message
- **PyTorchmistake**: {torch_error if torch_error else "none"}
- **TensorFlowmistake**: {tensorflow_error if tensorflow_error else "none"}
- **comparison error**: {comparison_error if comparison_error else "none"}

## Original test case

### PyTorchtest case
```json
{json.dumps(simplified_torch_test_case, indent=2, ensure_ascii=False)}
```

### TensorFlowtest case
```json
{json.dumps(simplified_tensorflow_test_case, indent=2, ensure_ascii=False)}
```

## Mission requirements Please judge the comparison result between the two frameworks based on the above information (including official API documents).**consistent**、**inconsistent**still**Execution error**，and do the following：

1. **if consistent**：Perform on use cases**Mutations（fuzzing）**，For example, modify the shape of the input tensor, modify parameter values, etc. (you can consider some extreme values ​​or boundary values）
2. **If an error occurs during execution**：Conduct use cases based on error reasons and official documents**repair**（Change parameter names, quantities, types, value ranges, etc. (different frameworks may not be exactly the same) or**jump over**（When you think that the functions of these two cross-framework operators are not completely equivalent）
3. **if inconsistent**：Determine whether it is a tolerable precision error (1e-3 and below): (1) If it is a tolerable precision error, then**Mutations**；（2）After combined with the operator document analysis, select when it is believed that the functions of the two cross-framework operators are not completely equivalent.**jump over**；（3）If the precision error is neither tolerable nor the functions of the two operators are equivalent, it is a test case construction problem. Please conduct the test case according to the operator documentation.**repair**。

## Output format requirements Please strictly follow the following JSON format and do not include any other text, comments or markdown tags：

{{
  "operation": "mutation",
  "reason": "Detailed reasons for doing this",
  "pytorch_test_case": {{
    "api": "{torch_api}",
{torch_param_example_str}
  }},
  "tensorflow_test_case": {{
    "api": "{tensorflow_api}",
{tf_param_example_str}
  }}
}}

**Important note**：
1. operationThe value must be "mutation"、"repair" or "skip" one
2. Tensor parameters must be used {{"shape": [...], "dtype": "..."}} Format
3. Scalar parameters use numeric values ​​directly, e.g. "y": 0
4. When constructing use cases for the two frameworks, you must ensure that the inputs are the same (convert the tensor shape if necessary, such as NHWC and NCHW conversion), and the parameters are strictly semantically corresponding. For example, the "pad" parameter, in PyTorchpadding=1Does not strictly correspond to TensorFlowpadding='SAME'，An equivalent pad operation must be performed。
5. PyTorchTest cases with TensorFlow can have differences in parameter names (such as input vs x), parameter values, or number of parameters, as long as the theoretical output is the same.。
6. If the official document cannot be found for this operator, please determine whether it is because the operator does not exist or has been removed from the current version of PyTorch or TensorFlow. If so, please set operation to "skip"，No need to try to fix。
7. When test cases are mutated, some extreme cases can be appropriately explored, such as: empty tensor (shape contains 0), single-element tensor（shape=[1]or []), high-dimensional tensors, very large tensors, different data types (int, float, bool), boundary values, etc., to improve test coverage and discover potentialbug
8. Please read the official API documentation carefully to ensure that parameter names, parameter types, parameter value ranges, etc. are consistent with the documentation.
"""
        return prompt
    
    def call_llm_for_repair_or_mutation(self, execution_result: Dict[str, Any], torch_test_case: Dict[str, Any], tensorflow_test_case: Dict[str, Any], torch_doc: str = "", tensorflow_doc: str = "") -> Dict[str, Any]:
        """Call LLM for test case repair or mutation"""
        prompt = self._build_llm_prompt(execution_result, torch_test_case, tensorflow_test_case, torch_doc, tensorflow_doc)
        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in deep learning framework testing and are proficient in the API differences between PyTorch and TensorFlow frameworks. Your task is to determine whether the test case needs to be repaired or mutated based on the execution results of the test case, and return the results in strict JSON format。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
            )
            
            raw_response = completion.choices[0].message.content.strip()
            
            # Add a 1 second time interval to avoid too frequent API calls
            time.sleep(1)
            
            # try to parseJSON
            try:
                llm_result = json.loads(raw_response)
                return llm_result
            except json.JSONDecodeError as e:
                self._safe_print(f"    ⚠️ LLMThe return is not valid JSON, try to extract the JSON content...")
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    llm_result = json.loads(json_match.group())
                    return llm_result
                else:
                    return {
                        "operation": "skip",
                        "reason": f"LLMReturn format error: {e}",
                        "pytorch_test_case": torch_test_case,
                        "tensorflow_test_case": tensorflow_test_case
                    }
        
        except Exception as e:
            self._safe_print(f"    ❌ Calling LLM failed: {e}")
            return {
                "operation": "skip",
                "reason": f"LLMcall failed: {e}",
                "pytorch_test_case": torch_test_case,
                "tensorflow_test_case": tensorflow_test_case
            }
    
    def get_num_test_cases_from_document(self, document: Dict[str, Any]) -> int:
        """Get the number of test cases in the document"""
        max_len = 0
        # Iterate through all fields in the document and find the maximum length of list type fields
        for key, value in document.items():
            if key not in ["_id", "api"] and isinstance(value, list):
                max_len = max(max_len, len(value))
        return max_len if max_len > 0 else 1
    
    def llm_enhanced_test_operator(self, operator_name: str, max_iterations: int = 3,
                                   num_test_cases: int = None,
                                   num_workers: int = DEFAULT_LLM_WORKERS) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Use LLM enhancement to test a single operator
        
        Args:
            operator_name: Operator name, for example "torch.where"
            max_iterations: Maximum number of iterations per test case
            num_test_cases: The number of use cases to be tested, None means testing all use cases
            num_workers: LLMNumber of concurrent calling threads (operator execution is still sequential）
        
        Returns:
            (List of all iteration results for all test cases, Statistics Dictionary)
        """
        self._safe_print(f"\n{'='*80}")
        self._safe_print(f"🎯 Start testing operators: {operator_name}")
        self._safe_print(f"🔄 Maximum number of iterations per use case: {max_iterations}")
        self._safe_print(f"{'='*80}\n")
        
        # Initialize statistics counter
        stats = {
            "llm_generated_cases": 0,      # LLMTotal number of test cases generated
            "successful_cases": 0           # The number of test cases executed successfully by both frameworks
        }
        
        # Get operator test cases from MongoDB
        document = self.collection.find_one({"api": operator_name})
        if document is None:
            self._safe_print(f"❌ operator not found {operator_name} test cases")
            return [], stats
        
        # Get the total number of test cases
        total_cases = self.get_num_test_cases_from_document(document)
        
        # Determine the actual number of use cases to test
        if num_test_cases is None:
            num_test_cases = total_cases
        else:
            num_test_cases = min(num_test_cases, total_cases)
        
        # Get the converted PyTorch sumTensorFlow API
        torch_api, tensorflow_api, mapping_method = self.convert_api_name(operator_name)
        if tensorflow_api is None:
            self._safe_print(f"❌ operator {operator_name} No TensorFlow corresponding implementation")
            return [], stats
        
        # Display API mapping information
        if torch_api != operator_name:
            self._safe_print(f"✅ original PyTorch API: {operator_name}")
            self._safe_print(f"✅ After conversion PyTorch API: {torch_api}")
        else:
            self._safe_print(f"✅ PyTorch API: {torch_api}")
        self._safe_print(f"✅ TensorFlow API: {tensorflow_api}")
        self._safe_print(f"✅ Mapping method: {mapping_method}")
        self._safe_print(f"📋 will test {num_test_cases} use cases (LLM concurrency={num_workers}, Execution order)")
        
        all_results = []
        initial_cases = []
        for case_idx in range(num_test_cases):
            initial_test_case = self.prepare_shared_numpy_data(document, case_index=case_idx)
            initial_test_case["api"] = torch_api
            initial_cases.append((case_idx + 1, initial_test_case))

        if num_workers <= 1:
            for case_number, initial_test_case in initial_cases:
                self._safe_print(f"\n📋 use case {case_number}/{num_test_cases}")
                case_results = self._test_single_case_with_iterations(
                    torch_api,
                    tensorflow_api,
                    initial_test_case,
                    max_iterations,
                    case_number,
                    stats
                )
                all_results.extend(case_results)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_case = {}
                for case_number, initial_test_case in initial_cases:
                    future = executor.submit(
                        self._test_single_case_with_iterations,
                        torch_api,
                        tensorflow_api,
                        initial_test_case,
                        max_iterations,
                        case_number,
                        stats
                    )
                    future_to_case[future] = case_number

                for future in as_completed(future_to_case):
                    case_results = future.result()
                    all_results.extend(case_results)

        all_results.sort(key=lambda r: (r.get("case_number", 0), r.get("iteration", 0)))

        self._safe_print(f"\n{'='*80}")
        self._safe_print("✅ All tests completed")
        self._safe_print(f"📊 Total tests {num_test_cases} use cases, total {len(all_results)} iterations")
        self._safe_print(f"📊 LLMNumber of test cases generated: {stats['llm_generated_cases']}")
        self._safe_print(f"📊 Number of use cases executed successfully by both frameworks: {stats['successful_cases']}")
        self._safe_print(f"{'='*80}\n")
        
        return all_results, stats
    
    def _test_single_case_with_iterations(self, operator_name: str, tensorflow_api: str, 
                                          initial_test_case: Dict[str, Any], 
                                          max_iterations: int,
                                          case_number: int,
                                          stats: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        Multiple rounds of iterative testing on a single test case
        
        Args:
            operator_name: PyTorchOperator name
            tensorflow_api: TensorFlowOperator name
            initial_test_case: Initial test case
            max_iterations: Maximum number of iterations
            case_number: Test case number (used to display）
            stats: Statistics dictionary (used to record the number of use cases generated by LLM and the number of successfully executed use cases）
        
        Returns:
            All iteration results of this test case
        """
        # Stores all iteration results of the current test case
        case_results = []
        
        # Current test case
        # PyTorch Using the original test case (the api has been set to torch_api）
        # TensorFlow Need to create a copy and set the correct api
        current_torch_test_case = initial_test_case
        current_tensorflow_test_case = copy.deepcopy(initial_test_case)
        current_tensorflow_test_case["api"] = tensorflow_api  # Set the correct TensorFlow API
        
        # Mark whether the current use case was generated by LLM (the first iteration is the database original use case）
        is_llm_generated = False
        
        # Crawl API documents in advance (crawl only once and reuse them in subsequent iterations)）
        self._safe_print(f"  📖 Pre-crawl API documentation...")
        torch_doc, tensorflow_doc = self._fetch_api_docs(operator_name, tensorflow_api)
        
        # Start iterative testing
        for iteration in range(max_iterations):
            source_type = "LLM" if is_llm_generated else "DB"
            self._safe_print(f"  🔄 iterate {iteration + 1}/{max_iterations} ({source_type})", end="")
            
            # Execute test cases
            try:
                execution_result = self._execute_test_case_sequential(
                    operator_name, tensorflow_api,
                    current_torch_test_case, current_tensorflow_test_case
                )

                pt_status = "✓" if execution_result['torch_success'] else "✗"
                tf_status = "✓" if execution_result['tensorflow_success'] else "✗"
                match_status = "✓" if execution_result['results_match'] else "✗"
                self._safe_print(f" | PT:{pt_status} TF:{tf_status} Match:{match_status}")

                if execution_result['torch_error'] and not execution_result['torch_success']:
                    err_short = str(execution_result['torch_error'])[:100]
                    self._safe_print(f"    ❌ PyTorchmistake: {err_short}...")
                if execution_result['tensorflow_error'] and not execution_result['tensorflow_success']:
                    err_short = str(execution_result['tensorflow_error'])[:100]
                    self._safe_print(f"    ❌ TensorFlowmistake: {err_short}...")
                if execution_result['comparison_error']:
                    err_short = str(execution_result['comparison_error'])[:100]
                    self._safe_print(f"    ⚠️ comparison error: {err_short}...")

                # Only count the use cases generated by LLM (excluding original database use cases)）
                if is_llm_generated:
                    if execution_result['torch_success'] and execution_result['tensorflow_success']:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

            except Exception as e:
                self._safe_print(f" | ❌ serious error: {str(e)[:80]}...")
                
                # Create an error result
                execution_result = {
                    "status": "fatal_error",
                    "torch_success": False,
                    "tensorflow_success": False,
                    "results_match": False,
                    "torch_error": f"Fatal error: {str(e)}",
                    "tensorflow_error": None,
                    "comparison_error": None,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            
            # Save the results of this iteration
            iteration_result = {
                "iteration": iteration + 1,
                "torch_test_case": current_torch_test_case,
                "tensorflow_test_case": current_tensorflow_test_case,
                "execution_result": execution_result,
                "llm_operation": None,
                "is_llm_generated": is_llm_generated
            }
            
            # Call LLM for repair or mutation (pass in PyTorch and TensorFlow test cases and API documentation）
            try:
                llm_result = self.call_llm_for_repair_or_mutation(
                    execution_result,
                    current_torch_test_case,
                    current_tensorflow_test_case,
                    torch_doc,
                    tensorflow_doc
                )
            except Exception as e:
                self._safe_print(f"    ❌ LLMcall failed: {str(e)[:80]}...")
                
                # Create a skip operation
                llm_result = {
                    "operation": "skip",
                    "reason": f"LLMcall failed: {str(e)}"
                }
                
                iteration_result["llm_operation"] = llm_result
                iteration_result["case_number"] = case_number
                case_results.append(iteration_result)
                break  # end iteration loop
            
            operation = llm_result.get("operation", "skip")
            reason = llm_result.get("reason", "")
            reason_short = reason[:80]
            self._safe_print(f"    🤖 LLM: {operation} - {reason_short}")
            
            iteration_result["llm_operation"] = {
                "operation": operation,
                "reason": reason
            }
            
            # Add test case number information
            iteration_result["case_number"] = case_number
            case_results.append(iteration_result)
            
            # End iteration if LLM recommends skipping
            if operation == "skip":
                break
            
            # Prepare test cases for the next iteration
            if operation == "mutation":
                next_pytorch_test_case = llm_result.get("pytorch_test_case", current_torch_test_case)
                next_tensorflow_test_case = llm_result.get("tensorflow_test_case", current_tensorflow_test_case)
                with self.stats_lock:
                    stats["llm_generated_cases"] += 1
                is_llm_generated = True
            elif operation == "repair":
                next_pytorch_test_case = llm_result.get("pytorch_test_case", current_torch_test_case)
                next_tensorflow_test_case = llm_result.get("tensorflow_test_case", current_tensorflow_test_case)
                with self.stats_lock:
                    stats["llm_generated_cases"] += 1
                is_llm_generated = True
            else:
                next_pytorch_test_case = current_torch_test_case
                next_tensorflow_test_case = current_tensorflow_test_case
            
            # Convert the test case format returned by LLM (convert PyTorch and TensorFlow test cases respectively, share tensor data）
            current_torch_test_case, current_tensorflow_test_case = self._convert_llm_test_cases(next_pytorch_test_case, next_tensorflow_test_case)
        
        # Fix problem 1: If the last iteration of LLM generates a new use case (mutation or repair), this new use case needs to be executed
        if len(case_results) > 0:
            last_iteration = case_results[-1]
            last_operation = last_iteration["llm_operation"].get("operation", "skip")
            
            if last_operation in ["mutation", "repair"]:
                self._safe_print(f"  🔄 Execute the final LLM use case", end="")

                try:
                    execution_result = self._execute_test_case_sequential(
                        operator_name, tensorflow_api,
                        current_torch_test_case, current_tensorflow_test_case
                    )

                    pt_status = "✓" if execution_result['torch_success'] else "✗"
                    tf_status = "✓" if execution_result['tensorflow_success'] else "✗"
                    match_status = "✓" if execution_result['results_match'] else "✗"
                    self._safe_print(f" | PT:{pt_status} TF:{tf_status} Match:{match_status}")

                    if execution_result['torch_error'] and not execution_result['torch_success']:
                        err_short = str(execution_result['torch_error'])[:100]
                        self._safe_print(f"    ❌ PyTorchmistake: {err_short}...")
                    if execution_result['tensorflow_error'] and not execution_result['tensorflow_success']:
                        err_short = str(execution_result['tensorflow_error'])[:100]
                        self._safe_print(f"    ❌ TensorFlowmistake: {err_short}...")
                    if execution_result['comparison_error']:
                        err_short = str(execution_result['comparison_error'])[:100]
                        self._safe_print(f"    ⚠️ comparison error: {err_short}...")

                    if execution_result['torch_success'] and execution_result['tensorflow_success']:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

                    final_iteration_result = {
                        "iteration": len(case_results) + 1,
                        "torch_test_case": current_torch_test_case,
                        "tensorflow_test_case": current_tensorflow_test_case,
                        "execution_result": execution_result,
                        "llm_operation": {
                            "operation": "final_execution",
                            "reason": "Execute the last use case generated by LLM"
                        },
                        "case_number": case_number,
                        "is_llm_generated": True
                    }
                    case_results.append(final_iteration_result)

                except Exception as e:
                    self._safe_print(f"  ❌ Final use case execution failed: {str(e)[:80]}...")
                    
                    # Record the attempt even if something goes wrong
                    final_iteration_result = {
                        "iteration": len(case_results) + 1,
                        "torch_test_case": current_torch_test_case,
                        "tensorflow_test_case": current_tensorflow_test_case,
                        "execution_result": {
                            "status": "fatal_error",
                            "torch_success": False,
                            "tensorflow_success": False,
                            "results_match": False,
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        },
                        "llm_operation": {
                            "operation": "final_execution",
                            "reason": "Execute the last LLM generated use case (a serious error occurred）"
                        },
                        "case_number": case_number,
                        "is_llm_generated": True
                    }
                    case_results.append(final_iteration_result)
        
        self._safe_print(f"  ✅ use case {case_number} Completed, total {len(case_results)} iterations")
        
        return case_results
    
    def _convert_llm_test_cases(self, pytorch_test_case: Dict[str, Any], tensorflow_test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Convert PyTorch and TensorFlow test cases returned by LLM into executable format         Make sure both frameworks use the same tensor data, but allow other parameters to differ
        
        Args:
            pytorch_test_case: LLMReturned PyTorch test case
            tensorflow_test_case: LLMReturned TensorFlow test case
        
        Returns:
            (Converted PyTorch test cases, Converted TensorFlow test cases)
        """
        # Silent conversion, reducing output
        
        # Step 1: Collect all parameter names that need to be generated as tensors and generate shared numpy arrays
        shared_tensors = {}  # Storing shared numpy array
        
        # Find out all tensor parameters (in pytorch or tensorflow test case）
        all_keys = set(pytorch_test_case.keys()) | set(tensorflow_test_case.keys())
        
        for key in all_keys:
            if key == "api":
                continue
            
            # Check if it is a tensor description
            pytorch_value = pytorch_test_case.get(key)
            tensorflow_value = tensorflow_test_case.get(key)
            
            is_tensor = False
            tensor_desc = None
            
            if isinstance(pytorch_value, dict) and "shape" in pytorch_value:
                is_tensor = True
                tensor_desc = pytorch_value
            elif isinstance(tensorflow_value, dict) and "shape" in tensorflow_value:
                is_tensor = True
                tensor_desc = tensorflow_value
            
            if is_tensor:
                # Generate shared numpy array
                numpy_array = self.generate_numpy_data(tensor_desc)
                shared_tensors[key] = numpy_array
        
        # Step 2: Build PyTorch and TensorFlow test cases separately
        converted_pytorch = {}
        converted_tensorflow = {}
        
        for key, value in pytorch_test_case.items():
            if key in shared_tensors:
                converted_pytorch[key] = shared_tensors[key]
            else:
                converted_pytorch[key] = value
        
        for key, value in tensorflow_test_case.items():
            if key in shared_tensors:
                converted_tensorflow[key] = shared_tensors[key]
            else:
                converted_tensorflow[key] = value
        
        return converted_pytorch, converted_tensorflow
    
    def save_results(self, operator_name: str, results: List[Dict[str, Any]], stats: Dict[str, int] = None):
        """
        Save test results to JSON file
        
        Args:
            operator_name: Operator name
            results: Test result list
            stats: Statistics (number of use cases generated by LLM and number of successfully executed use cases）
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_enhanced_{operator_name.replace('.', '_')}_{timestamp}.json"
        filepath = os.path.join(self.result_dir, filename)
        
        # Prepare output data (remove numpy array for JSON serialization）
        output_results = []
        for result in results:
            output_result = copy.deepcopy(result)
            
            # Simplifying numpy arrays in test cases (handling old format：test_case）
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
            
            # Simplify numpy arrays in test cases (handling new formats: torch_test_case and tensorflow_test_case）
            for test_case_key in ["torch_test_case", "tensorflow_test_case"]:
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
        
        self._safe_print(f"💾 Results have been saved to: {filepath}")
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()


def main():
    """
    main function
    """
    parser = argparse.ArgumentParser(
        description="PyTorch and TensorFlow operator comparison testing framework based on LLM"
    )
    parser.add_argument("--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS,
                        help="Maximum number of iterations per test case (default3）")
    parser.add_argument("--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES,
                        help="The number of test cases to be tested for each operator (default3）")
    parser.add_argument("--start", type=int, default=1,
                        help="Starting operator index (starts from 1, default1）")
    parser.add_argument("--end", type=int, default=None,
                        help="End operator index (including, default all）")
    parser.add_argument("--operators", "-o", nargs="*",
                        help="Specify the name of the operator to be tested (PyTorch format）")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS,
                        help="Number of concurrent threads (default 1, sequential execution is more stable）")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"LLMmodel name (default {DEFAULT_MODEL}）")
    parser.add_argument("--key-path", "-k", default=DEFAULT_KEY_PATH,
                        help=f"API keyFile path (default {DEFAULT_KEY_PATH}）")

    args = parser.parse_args()

    max_iterations = args.max_iterations
    num_test_cases = args.num_cases
    num_workers = max(1, args.workers)

    print("="*80)
    print("PyTorch and TensorFlow operator batch comparison testing framework based on LLM")
    print("="*80)
    print(f"📌 The number of iterations for each operator: {max_iterations}")
    print(f"📌 Number of test cases for each operator: {num_test_cases}")
    print(f"📌 LLMNumber of concurrent threads: {num_workers}")
    print(f"📌 LLMModel: {args.model}")
    print("="*80)

    comparator = LLMEnhancedComparator(
        key_path=args.key_path,
        model=args.model,
        llm_workers=num_workers
    )

    start_time = time.time()
    start_datetime = datetime.now()

    try:
        print("\n🔍 Retrieving all operators in the database...")
        all_operators = list(comparator.collection.find({}, {"api": 1}))
        all_operator_names = [doc["api"] for doc in all_operators if "api" in doc]
        print(f"✅ There are total in the database {len(all_operator_names)} an operator")

        if args.operators:
            operator_names = args.operators
            print(f"📋 Specify the number of operators: {len(operator_names)}")
        else:
            total_available = len(all_operator_names)
            start_idx = max(1, args.start) - 1
            end_idx = args.end if args.end is not None else total_available
            if end_idx > total_available:
                end_idx = total_available
            if start_idx >= end_idx:
                raise ValueError(f"starting index {args.start} Must be less than end index {end_idx}")

            operator_names = all_operator_names[start_idx:end_idx]
            print(f"📌 Test range: No. {start_idx + 1} To the third {end_idx} an operator")
            print(f"📋 will test {len(operator_names)} an operator")

        print("\n🔍 Filter operators without TensorFlow corresponding implementation...")
        original_count = len(operator_names)
        filtered_operator_names = []
        skipped_operators = []

        for op_name in operator_names:
            _, tf_api, mapping_method = comparator.convert_api_name(op_name)
            if tf_api is not None:
                filtered_operator_names.append(op_name)
            else:
                skipped_operators.append((op_name, mapping_method))

        operator_names = filtered_operator_names
        skipped_count = original_count - len(operator_names)

        print(f"✅ Filtering completed: original {original_count} operator, skip {skipped_count} pieces, remaining {len(operator_names)} indivual")
        if skipped_operators:
            print(f"⏭️ Operators to skip (first 10）: {', '.join([f'{op}({reason})' for op, reason in skipped_operators[:10]])}{'...' if len(skipped_operators) > 10 else ''}")

        print(f"📋 Operator list: {', '.join(operator_names[:10])}{'...' if len(operator_names) > 10 else ''}\n")

        all_operators_summary = []

        batch_log_file = os.path.join(comparator.result_dir, f"batch_test_log_{start_datetime.strftime('%Y%m%d_%H%M%S')}.txt")
        log_file = open(batch_log_file, 'w', encoding='utf-8')

        log_file.write("="*80 + "\n")
        log_file.write("Batch test total log\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("Test configuration:\n")
        log_file.write(f"  - The number of iterations for each operator: {max_iterations}\n")
        log_file.write(f"  - Number of test cases for each operator: {num_test_cases}\n")
        log_file.write(f"  - LLMNumber of concurrent threads: {num_workers}\n")
        log_file.write(f"  - Total number of operators in the database: {len(all_operator_names)}\n")
        if not args.operators:
            log_file.write(f"  - Test range: No. {args.start} To the third {args.end if args.end is not None else len(all_operator_names)} indivual\n")
        log_file.write(f"  - Number of uncorresponding implementation operators skipped: {skipped_count}\n")
        log_file.write(f"  - Actual number of test operators: {len(operator_names)}\n")
        log_file.write("="*80 + "\n\n")
        log_file.flush()

        for idx, operator_name in enumerate(operator_names, 1):
            print("\n" + "🔷"*40)
            print(f"🎯 [{idx}/{len(operator_names)}] Start testing operators: {operator_name}")
            print("🔷"*40)

            try:
                results, stats = comparator.llm_enhanced_test_operator(
                    operator_name,
                    max_iterations=max_iterations,
                    num_test_cases=num_test_cases,
                    num_workers=num_workers
                )

                if results:
                    comparator.save_results(operator_name, results, stats)

                    all_operators_summary.append({
                        "operator": operator_name,
                        "total_iterations": len(results),
                        "llm_generated_cases": stats.get("llm_generated_cases", 0),
                        "successful_cases": stats.get("successful_cases", 0),
                        "status": "completed"
                    })

                    print(f"\n✅ operator {operator_name} Test completed")
                    print(f"   - Total number of iterations: {len(results)}")
                    print(f"   - LLMNumber of use cases generated: {stats.get('llm_generated_cases', 0)}")
                    print(f"   - Number of successfully executed use cases: {stats.get('successful_cases', 0)}")

                    log_file.write(f"[{idx}/{len(operator_names)}] {operator_name}\n")
                    log_file.write("  state: ✅ Finish\n")
                    log_file.write(f"  Total number of iterations: {len(results)}\n")
                    log_file.write(f"  LLMNumber of use cases generated: {stats.get('llm_generated_cases', 0)}\n")
                    log_file.write(f"  Number of successfully executed use cases: {stats.get('successful_cases', 0)}\n")
                    if stats.get('llm_generated_cases', 0) > 0:
                        success_rate = (stats.get('successful_cases', 0) / stats.get('llm_generated_cases', 0)) * 100
                        log_file.write(f"  success rate: {success_rate:.2f}%\n")
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
                    print(f"\n⚠️ operator {operator_name} No test results")

                    log_file.write(f"[{idx}/{len(operator_names)}] {operator_name}\n")
                    log_file.write("  state: ⚠️ No result\n\n")
                    log_file.flush()

            except Exception as e:
                print(f"\n❌ operator {operator_name} test failed: {e}")
                all_operators_summary.append({
                    "operator": operator_name,
                    "total_iterations": 0,
                    "llm_generated_cases": 0,
                    "successful_cases": 0,
                    "status": "failed",
                    "error": str(e)
                })

                log_file.write(f"[{idx}/{len(operator_names)}] {operator_name}\n")
                log_file.write("  state: ❌ fail\n")
                log_file.write(f"  mistake: {str(e)}\n\n")
                log_file.flush()
                continue

        end_time = time.time()
        end_datetime = datetime.now()
        total_duration = end_time - start_time
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)

        print("\n" + "="*80)
        print("📊 Overall summary of batch testing")
        print("="*80)
        print(f"Total number of operators: {len(operator_names)}")

        completed_count = sum(1 for s in all_operators_summary if s["status"] == "completed")
        failed_count = sum(1 for s in all_operators_summary if s["status"] == "failed")
        no_results_count = sum(1 for s in all_operators_summary if s["status"] == "no_results")

        print(f"✅ Completed successfully: {completed_count}")
        print(f"❌ test failed: {failed_count}")
        print(f"⚠️ No result: {no_results_count}")
        print(f"⏭️ Skip (no corresponding implementation）: {skipped_count}")

        total_llm_cases = sum(s["llm_generated_cases"] for s in all_operators_summary)
        total_successful_cases = sum(s["successful_cases"] for s in all_operators_summary)
        total_iterations = sum(s["total_iterations"] for s in all_operators_summary)

        print("\n📈 Statistics:")
        print(f"   - LLMTotal number of test cases generated: {total_llm_cases}")
        print(f"   - Total number of successfully executed use cases: {total_successful_cases}")
        if total_llm_cases > 0:
            success_rate = (total_successful_cases / total_llm_cases) * 100
            print(f"   - Proportion of successful executions: {success_rate:.2f}%")
        print(f"   - Total number of iterations: {total_iterations}")
        print(f"\n⏱️ running time: {hours}Hour {minutes}minute {seconds}Second")

        log_file.write("="*80 + "\n")
        log_file.write("Overall statistics\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"end time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"total running time: {hours}Hour {minutes}minute {seconds}Second ({total_duration:.2f}Second)\n\n")

        log_file.write("Operator test results:\n")
        log_file.write(f"  - Total number of operators: {len(operator_names)}\n")
        log_file.write(f"  - Completed successfully: {completed_count} ({completed_count/len(operator_names)*100:.2f}%)\n")
        log_file.write(f"  - test failed: {failed_count} ({failed_count/len(operator_names)*100:.2f}%)\n")
        log_file.write(f"  - No result: {no_results_count} ({no_results_count/len(operator_names)*100:.2f}%)\n\n")

        log_file.write("LLMGenerate use case statistics:\n")
        log_file.write(f"  - LLMTotal number of test cases generated: {total_llm_cases}\n")
        log_file.write(f"  - Number of use cases successfully executed by both frameworks: {total_successful_cases}\n")
        if total_llm_cases > 0:
            success_rate = (total_successful_cases / total_llm_cases) * 100
            log_file.write(f"  - Proportion of successful executions: {success_rate:.2f}%\n")
        log_file.write(f"  - Total number of iterations: {total_iterations}\n")
        if completed_count > 0:
            avg_llm_cases = total_llm_cases / completed_count
            avg_successful = total_successful_cases / completed_count
            log_file.write(f"  - Average number of use cases generated by LLM for each operator: {avg_llm_cases:.2f}\n")
            log_file.write(f"  - Average number of successfully executed use cases for each operator: {avg_successful:.2f}\n")

        log_file.write("\n" + "="*80 + "\n")
        log_file.write("For detailed results, please view the separate log files of each operator.\n")
        log_file.write("="*80 + "\n")
        log_file.close()

        print(f"\n💾 The total log has been saved to: {batch_log_file}")

        summary_file = os.path.join(comparator.result_dir, f"batch_test_summary_{start_datetime.strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_config": {
                    "max_iterations": max_iterations,
                    "num_test_cases": num_test_cases,
                    "operator_range": f"{args.start}-{args.end}" if not args.operators else "custom",
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
                    "skipped_no_tf_impl": skipped_count,
                    "skipped_operators_list": [op for op, _ in skipped_operators],
                    "total_llm_generated_cases": total_llm_cases,
                    "total_successful_cases": total_successful_cases,
                    "success_rate": f"{(total_successful_cases / total_llm_cases * 100):.2f}%" if total_llm_cases > 0 else "N/A",
                    "total_iterations": total_iterations,
                    "avg_llm_cases_per_operator": f"{total_llm_cases / completed_count:.2f}" if completed_count > 0 else "N/A",
                    "avg_successful_per_operator": f"{total_successful_cases / completed_count:.2f}" if completed_count > 0 else "N/A"
                },
                "operators": all_operators_summary
            }, f, indent=2, ensure_ascii=False)

        print(f"💾 JSONSummary saved to: {summary_file}")

    finally:
        comparator.close()
        print("\n✅ Batch test program execution completed")


if __name__ == "__main__":
    main()
