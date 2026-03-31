# ./pt_pd_test/compare_llm_vs_x2paddle.py
"""
LLM method vs X2Paddle method: PyTorch -> PaddlePaddle test case conversion success rate
===========================================================================

Compare two cross-framework test-case migration strategies:
1. LLM method: fetch cases from MongoDB -> build PT/PD cases -> run -> LLM repair/mutation -> run LLM cases
2. X2Paddle method: fetch cases from MongoDB -> wrap as small PyTorch model -> torch.onnx.export -> onnxruntime
     (simulate X2Paddle core path: PyTorch -> ONNX -> PaddlePaddle)

X2Paddle overview:
    - Official Paddle conversion tool supporting Caffe/TensorFlow/ONNX/PyTorch -> PaddlePaddle
    - Core conversion path relies on ONNX intermediate format
    - Supports 130+ PyTorch OPs, 90+ ONNX OPs
    - API: pytorch2paddle(module, save_dir, jit_type="trace", input_examples=[...])

Metric: LLM-generated PD case success rate vs X2Paddle (ONNX) export + inference success rate
(excluding operators skipped by LLM)
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
import argparse
import traceback
import time
import tempfile
import onnx
import onnxruntime as ort
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, RLock

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from component.doc.doc_crawler_factory import get_doc_content

# ==================== Constants ====================
DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"
DEFAULT_MAX_ITERATIONS = 1   # Simplified comparison: one iteration only
DEFAULT_NUM_CASES = 3
DEFAULT_WORKERS = 4


# ==================== Utilities ====================

def safe_print(msg: str, print_lock: Lock = None, end: str = "\n"):
    """Thread-safe print."""
    if print_lock:
        with print_lock:
            print(msg, end=end, flush=True)
    else:
        print(msg, end=end, flush=True)


# ==================== X2Paddle (ONNX) Converter ====================

class X2PaddleConverter:
    """
    Simulate X2Paddle PyTorch -> PaddlePaddle test-case migration via ONNX.
    
    X2Paddle core path:
    1. PyTorch model -> torch.onnx.export -> ONNX model
    2. ONNX model -> X2Paddle internal mapping -> PaddlePaddle model
    
    Simulator flow (operator-level):
    1. Wrap PyTorch operator as a minimal nn.Module
    2. Export ONNX with torch.onnx.export
    3. Run inference with onnxruntime
    
    Success criterion: ONNX export + OnnxRuntime inference success
    """

    def __init__(self, print_lock: Lock = None):
        self.print_lock = print_lock or Lock()
        self.execution_lock = RLock()

    def _safe_print(self, msg: str, end: str = "\n"):
        safe_print(msg, self.print_lock, end)

    def _wrap_as_module(self, torch_api: str, is_class_api: bool,
                        init_kwargs: Dict[str, Any],
                        extra_kwargs: Dict[str, Any] = None) -> Optional[torch.nn.Module]:
        """
        Wrap a PyTorch operator as nn.Module for torch.onnx.export.
        
        Args:
            torch_api: PyTorch API name
            is_class_api: Whether this is a class-based API
            init_kwargs: Init kwargs for class-based operators
            extra_kwargs: Non-tensor kwargs for function operators (captured in closure)
        
        Returns:
            Wrapped nn.Module, or None on failure
        """
        parts = torch_api.split(".")
        extra_kwargs = extra_kwargs or {}

        try:
            if is_class_api:
                cls = self._resolve_attr(parts)
                if cls is None:
                    return None
                instance = cls(**init_kwargs)
                if isinstance(instance, torch.nn.Module):
                    return instance
                return _FuncWrapper(lambda x, _inst=instance: _inst(x))
            else:
                func = self._resolve_attr(parts)
                if func is None:
                    return None
                if extra_kwargs:
                    return _FuncWrapper(lambda *args, _f=func, _kw=extra_kwargs: _f(*args, **_kw))
                return _FuncWrapper(func)
        except Exception:
            return None

    @staticmethod
    def _resolve_attr(parts: List[str]):
        """Resolve attribute path like ['torch', 'nn', 'ReLU']."""
        try:
            obj = torch
            for p in parts[1:]:  # Skip 'torch'
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            return None

    def convert_and_run(self, torch_api: str, test_case: Dict[str, Any],
                        is_class_api: bool) -> Dict[str, Any]:
        """
        Run X2Paddle (ONNX) conversion + inference for one test case.
        
        Flow:
        1. Prepare PyTorch input tensors
        2. Run PyTorch forward to get reference result
        3. torch.onnx.export (simulate X2Paddle ONNX export stage)
        4. onnxruntime inference (simulate Paddle inference)
        5. Compare results
        
        Returns:
            {
                "onnx_export_success": bool,
                "onnx_run_success": bool,
                "pt_success": bool,
                "error": str or None,
                "pt_shape": list or None,
                "onnx_shape": list or None,
            }
        """
        result = {
            "onnx_export_success": False,
            "onnx_run_success": False,
            "pt_success": False,
            "error": None,
            "pt_shape": None,
            "onnx_shape": None,
        }

        # Use execution lock to keep torch.onnx.export thread-safe
        with self.execution_lock:
            # ---------- 1. Prepare inputs ----------
            try:
                input_tensors, init_kwargs, input_names, extra_kwargs = self._prepare_inputs(test_case, is_class_api)
            except Exception as e:
                result["error"] = f"Input preparation failed: {e}"
                return result

            if not input_tensors:
                result["error"] = "No valid input tensors"
                return result

            # ---------- 2. Wrap model ----------
            module = self._wrap_as_module(torch_api, is_class_api, init_kwargs, extra_kwargs)
            if module is None:
                result["error"] = f"Failed to wrap operator {torch_api} as nn.Module"
                return result

            module.eval()

            # ---------- 3. PyTorch forward ----------
            try:
                with torch.no_grad():
                    if len(input_tensors) == 1:
                        pt_output = module(input_tensors[0])
                    else:
                        pt_output = module(*input_tensors)
                result["pt_success"] = True
                if hasattr(pt_output, 'shape'):
                    result["pt_shape"] = list(pt_output.shape)
            except Exception as e:
                result["error"] = f"PyTorch forward failed: {e}"
                return result

            # ---------- 4. ONNX export (simulate X2Paddle export stage) ----------
            onnx_path = None
            try:
                onnx_fd, onnx_path = tempfile.mkstemp(suffix=".onnx")
                os.close(onnx_fd)

                if len(input_tensors) == 1:
                    dummy_input = input_tensors[0]
                else:
                    dummy_input = tuple(input_tensors)

                output_names = ["output"]
                torch.onnx.export(
                    module,
                    dummy_input,
                    onnx_path,
                    input_names=input_names,
                    output_names=output_names,
                    opset_version=14,
                    do_constant_folding=True,
                )
                result["onnx_export_success"] = True
            except Exception as e:
                result["error"] = f"ONNX export failed (X2Paddle core path): {e}"
                if onnx_path and os.path.exists(onnx_path):
                    os.remove(onnx_path)
                return result

            # ---------- 5. OnnxRuntime inference (simulate converted inference) ----------
            try:
                sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                feed = {}
                for i, tensor in enumerate(input_tensors):
                    name = input_names[i] if i < len(input_names) else f"input_{i}"
                    feed[name] = tensor.detach().cpu().numpy()

                onnx_outputs = sess.run(None, feed)
                onnx_result = onnx_outputs[0]

                result["onnx_run_success"] = True
                result["onnx_shape"] = list(onnx_result.shape)

            except Exception as e:
                result["error"] = f"OnnxRuntime inference failed: {e}"
            finally:
                if onnx_path and os.path.exists(onnx_path):
                    os.remove(onnx_path)

        return result

    def _prepare_inputs(self, test_case: Dict[str, Any],
                        is_class_api: bool) -> Tuple[List[torch.Tensor], Dict[str, Any], List[str], Dict[str, Any]]:
        """
        Extract input tensors and init kwargs from a test case.
        
        Returns:
            (input_tensors list, init_kwargs dict, input_names list, extra_kwargs dict)
        """
        input_tensors: List[torch.Tensor] = []
        init_kwargs: Dict[str, Any] = {}
        input_names: List[str] = []
        extra_kwargs: Dict[str, Any] = {}

        positional_tensor_params = ["condition", "input", "x", "y", "other"]

        # Handle varargs
        varargs_key = None
        for key in test_case.keys():
            if key.startswith('*'):
                varargs_key = key
                break

        if varargs_key:
            varargs_value = test_case[varargs_key]
            if isinstance(varargs_value, list):
                for idx, item in enumerate(varargs_value):
                    tensor = self._to_torch_tensor(item)
                    if tensor is not None:
                        input_tensors.append(tensor)
                        input_names.append(f"input_{idx}")
            return input_tensors, init_kwargs, input_names, extra_kwargs

        # Collect positional params in order as input tensors
        for param_name in positional_tensor_params:
            if param_name in test_case:
                value = test_case[param_name]
                tensor = self._to_torch_tensor(value)
                if tensor is not None:
                    input_tensors.append(tensor)
                    input_names.append(param_name)

        # Other parameters
        skip_params = {"layout", "requires_grad", "out", "api"}
        for key, value in test_case.items():
            if key in positional_tensor_params or key in skip_params or key.startswith('*'):
                continue
            if is_class_api:
                if isinstance(value, np.ndarray):
                    input_tensors.append(torch.from_numpy(value.copy()))
                    input_names.append(key)
                else:
                    init_kwargs[key] = value
            else:
                if isinstance(value, np.ndarray):
                    input_tensors.append(torch.from_numpy(value.copy()))
                    input_names.append(key)
                elif isinstance(value, (int, float, bool, str)):
                    extra_kwargs[key] = value

        return input_tensors, init_kwargs, input_names, extra_kwargs

    @staticmethod
    def _to_torch_tensor(value: Any) -> Optional[torch.Tensor]:
        """Convert various value formats to a PyTorch tensor."""
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy())
        elif isinstance(value, dict) and "shape" in value:
            dtype_map = {
                "torch.float64": np.float64, "torch.float32": np.float32,
                "torch.int64": np.int64, "torch.int32": np.int32,
                "torch.bool": np.bool_, "torch.uint8": np.uint8,
                "float64": np.float64, "float32": np.float32,
                "int64": np.int64, "int32": np.int32,
                "bool": np.bool_, "uint8": np.uint8,
            }
            shape = value.get("shape", [])
            dtype_str = value.get("dtype", "torch.float32")
            dtype = dtype_map.get(dtype_str, np.float32)
            if shape:
                if dtype == np.bool_:
                    arr = np.random.randint(0, 2, shape).astype(np.bool_)
                elif dtype in [np.int64, np.int32]:
                    arr = np.random.randint(-10, 10, shape).astype(dtype)
                else:
                    arr = np.random.randn(*shape).astype(dtype)
            else:
                arr = np.array(1.0, dtype=dtype)
            return torch.from_numpy(arr)
        elif isinstance(value, (int, float)):
            return torch.tensor(value)
        elif isinstance(value, list):
            try:
                return torch.tensor(value)
            except Exception:
                return None
        return None


class _FuncWrapper(torch.nn.Module):
    """Wrap a plain function as nn.Module for torch.onnx.export."""

    def __init__(self, func):
        super().__init__()
        self._func = func

    def forward(self, *args):
        return self._func(*args)


# ==================== LLM Method (Simplified) ====================

class LLMMethod:
    """
    LLM-based test case conversion method (simplified, 1 iteration).
    
    Flow: fetch cases from MongoDB -> run in PT/PD -> LLM decides repair/skip -> run LLM cases
    Reuses core logic from llm_enhanced_compare.py with simplified output.
    """

    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/",
                 db_name: str = "freefuzz-torch",
                 key_path: str = DEFAULT_KEY_PATH,
                 model: str = DEFAULT_MODEL,
                 print_lock: Lock = None,
                 llm_workers: int = DEFAULT_WORKERS):
        self.model = model
        self.print_lock = print_lock or Lock()
        self.llm_workers = max(1, int(llm_workers))
        self.execution_lock = RLock()
        self.stats_lock = Lock()

        # MongoDB
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["argVS"]

        # LLM client
        api_key = self._load_api_key(key_path)
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # API mapping
        self.api_mapping = self._load_api_mapping()

        # Random seeds
        np.random.seed(42)
        torch.manual_seed(42)
        paddle.seed(42)

        # Operators that may hang or crash (skip these)
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

    def _safe_print(self, msg: str, end: str = "\n"):
        safe_print(msg, self.print_lock, end)

    def _load_api_key(self, key_path: str) -> str:
        if not os.path.isabs(key_path):
            key_file = os.path.join(ROOT_DIR, key_path)
        else:
            key_file = key_path
        if os.path.exists(key_file):
            try:
                with open(key_file, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
                if api_key:
                    return api_key
            except Exception:
                pass
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            return api_key
        return ""

    def _load_api_mapping(self) -> Dict[str, Dict[str, str]]:
        mapping_file = os.path.join(ROOT_DIR, "component", "data", "pd_api_mappings_final.csv")
        try:
            df = pd.read_csv(mapping_file)
            mapping = {}
            for _, row in df.iterrows():
                pt_api = str(row["pytorch-api"]).strip()
                pd_api = str(row["paddle-api"]).strip()
                mapping[pt_api] = {"pd_api": pd_api}
            return mapping
        except Exception as e:
            self._safe_print(f"❌ Failed to load mapping table: {e}")
            return {}

    def convert_api_name(self, torch_api: str) -> Tuple[Optional[str], Optional[str], str]:
        if torch_api in self.api_mapping:
            pd_api = self.api_mapping[torch_api]["pd_api"]
            if pd_api in ("No implementation", "NONE", ""):
                return torch_api, None, "No implementation"
            return torch_api, pd_api, "Mapping table"
        return torch_api, None, "Not found in mapping table"

    @staticmethod
    def is_class_based_api(api_name: str) -> bool:
        parts = api_name.split(".")
        if len(parts) >= 2:
            return any(c.isupper() for c in parts[-1])
        return False

    @staticmethod
    def get_operator_function(api_name: str, framework: str = "torch"):
        try:
            parts = api_name.split(".")
            if framework == "torch" and parts[0] == "torch":
                obj = torch
                for p in parts[1:]:
                    obj = getattr(obj, p)
                return obj
            elif framework == "paddle" and parts[0] == "paddle":
                obj = paddle
                for p in parts[1:]:
                    obj = getattr(obj, p)
                return obj
        except AttributeError:
            pass
        return None

    # -------- Data generation --------

    @staticmethod
    def generate_numpy_data(data: Any) -> np.ndarray:
        if isinstance(data, dict):
            dtype_map = {
                "torch.float64": np.float64, "torch.float32": np.float32,
                "torch.int64": np.int64, "torch.int32": np.int32,
                "torch.bool": np.bool_, "torch.uint8": np.uint8,
                "float64": np.float64, "float32": np.float32,
                "int64": np.int64, "int32": np.int32,
                "bool": np.bool_, "uint8": np.uint8,
                "bool_": np.bool_, "float": np.float32, "int": np.int64,
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
                return np.array(1.0 if dtype not in [np.bool_, np.int64, np.int32] else 1, dtype=dtype)
        elif isinstance(data, (int, float)):
            return np.array(data)
        elif isinstance(data, list):
            return np.array(data)
        return np.array(data)

    def prepare_shared_numpy_data(self, document: Dict[str, Any], case_index: int = 0) -> Dict[str, Any]:
        shared_data = {}
        api_name = document.get("api", "")
        if self.is_class_based_api(api_name) and "input" not in document:
            if "2d" in api_name.lower():
                default_shape = {"shape": [2, 3, 4, 4], "dtype": "torch.float32"}
            elif "1d" in api_name.lower():
                default_shape = {"shape": [2, 3, 10], "dtype": "torch.float32"}
            elif "3d" in api_name.lower():
                default_shape = {"shape": [2, 3, 4, 4, 4], "dtype": "torch.float32"}
            else:
                default_shape = {"shape": [2, 3], "dtype": "torch.float32"}
            shared_data["input"] = self.generate_numpy_data(default_shape)

        exclude_keys = ["_id", "api"]
        for key, value in document.items():
            if key not in exclude_keys:
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

    def get_num_test_cases(self, document: Dict[str, Any]) -> int:
        max_len = 0
        for key, value in document.items():
            if key not in ["_id", "api"] and isinstance(value, list):
                max_len = max(max_len, len(value))
        return max_len if max_len > 0 else 1

    # -------- Argument preparation --------

    def prepare_arguments_torch(self, test_case: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        args, kwargs = [], {}
        varargs_key = None
        for key in test_case.keys():
            if key.startswith('*'):
                varargs_key = key
                break

        if varargs_key:
            varargs_value = test_case[varargs_key]
            if isinstance(varargs_value, list):
                for item in varargs_value:
                    if isinstance(item, dict) and "shape" in item:
                        arr = self.generate_numpy_data(item)
                        args.append(torch.from_numpy(arr))
                    elif isinstance(item, np.ndarray):
                        args.append(torch.from_numpy(item.copy()))
                    else:
                        args.append(item)
            return args, kwargs

        positional_params = ["condition", "x", "y", "input", "other"]
        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, np.ndarray):
                    args.append(torch.from_numpy(value.copy()))
                else:
                    args.append(value)

        for key, value in test_case.items():
            if key not in positional_params + ["api"]:
                if isinstance(value, np.ndarray):
                    kwargs[key] = torch.from_numpy(value.copy())
                else:
                    kwargs[key] = value
        return args, kwargs

    def prepare_arguments_paddle(self, test_case: Dict[str, Any], paddle_api: str) -> Tuple[List[Any], Dict[str, Any]]:
        args, kwargs = [], {}
        common_skip = ["layout", "requires_grad", "out"]

        varargs_key = None
        for key in test_case.keys():
            if key.startswith('*'):
                varargs_key = key
                break

        if varargs_key:
            varargs_value = test_case[varargs_key]
            if isinstance(varargs_value, list):
                for item in varargs_value:
                    if isinstance(item, dict) and "shape" in item:
                        arr = self.generate_numpy_data(item)
                        args.append(paddle.to_tensor(arr))
                    elif isinstance(item, np.ndarray):
                        args.append(paddle.to_tensor(item.copy()))
                    else:
                        args.append(item)
            return args, kwargs

        positional_params = ["condition", "x", "y", "input", "other"]
        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, np.ndarray):
                    args.append(paddle.to_tensor(value.copy()))
                else:
                    args.append(value)

        for key, value in test_case.items():
            if key not in positional_params + ["api"]:
                if key in common_skip:
                    continue
                if isinstance(value, np.ndarray):
                    kwargs[key] = paddle.to_tensor(value.copy())
                else:
                    kwargs[key] = value
        return args, kwargs

    # -------- Execute test cases --------

    def execute_test_case(self, torch_api: str, paddle_api: str,
                          torch_test_case: Dict[str, Any],
                          paddle_test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Execute test cases in PyTorch and PaddlePaddle separately."""
        result = {
            "torch_api": torch_api, "paddle_api": paddle_api,
            "torch_success": False, "paddle_success": False,
            "results_match": False,
            "torch_error": None, "paddle_error": None,
            "comparison_error": None, "status": "unknown"
        }

        is_class_api = self.is_class_based_api(torch_api)

        # -- PyTorch --
        torch_result = None
        try:
            torch_func = self.get_operator_function(torch_api, "torch")
            if torch_func is None:
                result["torch_error"] = f"Operator {torch_api} not found"
            else:
                args, kwargs = self.prepare_arguments_torch(torch_test_case)
                if is_class_api:
                    init_kwargs = {k: v for k, v in kwargs.items() if k != 'input'}
                    instance = torch_func(**init_kwargs)
                    input_data = kwargs.get('input', args[0] if args else None)
                    if input_data is None:
                        raise ValueError("Class operator missing input parameter")
                    with torch.no_grad():
                        torch_result = instance(input_data)
                else:
                    with torch.no_grad():
                        torch_result = torch_func(*args, **kwargs)
                result["torch_success"] = True
        except Exception as e:
            result["torch_error"] = str(e)

        # -- PaddlePaddle --
        paddle_result = None
        try:
            paddle_func = self.get_operator_function(paddle_api, "paddle")
            if paddle_func is None:
                result["paddle_error"] = f"Operator {paddle_api} not found"
            else:
                args, kwargs = self.prepare_arguments_paddle(paddle_test_case, paddle_api)
                if is_class_api:
                    init_kwargs = {k: v for k, v in kwargs.items() if k not in ['x', 'input']}
                    instance = paddle_func(**init_kwargs)
                    input_data = kwargs.get('x', kwargs.get('input', args[0] if args else None))
                    if input_data is None:
                        raise ValueError("Class operator missing input/x parameter")
                    paddle_result = instance(input_data)
                else:
                    paddle_result = paddle_func(*args, **kwargs)
                result["paddle_success"] = True
        except Exception as e:
            result["paddle_error"] = str(e)

        # -- Compare --
        if result["torch_success"] and result["paddle_success"]:
            try:
                torch_np = torch_result.detach().cpu().numpy() if hasattr(torch_result, 'detach') else np.array(torch_result)
                pd_np = paddle_result.numpy() if hasattr(paddle_result, 'numpy') else np.array(paddle_result)
                if torch_np.shape != pd_np.shape:
                    result["comparison_error"] = f"Shape mismatch: {torch_np.shape} vs {pd_np.shape}"
                elif np.allclose(torch_np, pd_np, atol=1e-5, rtol=1e-5, equal_nan=True):
                    result["results_match"] = True
                else:
                    max_diff = np.max(np.abs(torch_np.astype(float) - pd_np.astype(float)))
                    result["comparison_error"] = f"Max difference: {max_diff}"
                result["status"] = "compared"
            except Exception as e:
                result["comparison_error"] = str(e)
                result["status"] = "comparison_failed"
        elif result["torch_success"]:
            result["status"] = "paddle_failed"
        elif result["paddle_success"]:
            result["status"] = "torch_failed"
        else:
            result["status"] = "both_failed"

        return result

    def _execute_sequential(self, torch_api, paddle_api, torch_tc, pd_tc):
        with self.execution_lock:
            return self.execute_test_case(torch_api, paddle_api, torch_tc, pd_tc)

    # -------- Doc crawling --------

    def fetch_api_docs(self, torch_api: str, paddle_api: str) -> Tuple[str, str]:
        MIN_DOC_LENGTH = 300
        torch_doc, pd_doc = "", ""
        try:
            torch_doc = get_doc_content(torch_api, "pytorch")
            if not (torch_doc and "Unable" not in torch_doc and len(torch_doc.strip()) > MIN_DOC_LENGTH):
                torch_doc = f"Unable to fetch documentation for {torch_api}"
            elif len(torch_doc) > 3000:
                torch_doc = torch_doc[:3000] + "\n... (doc truncated)"
        except Exception:
            torch_doc = ""
        try:
            pd_doc = get_doc_content(paddle_api, "paddle")
            if not (pd_doc and "Unable" not in pd_doc and len(pd_doc.strip()) > MIN_DOC_LENGTH):
                pd_doc = f"Unable to fetch documentation for {paddle_api}"
            elif len(pd_doc) > 3000:
                pd_doc = pd_doc[:3000] + "\n... (doc truncated)"
        except Exception:
            pd_doc = ""
        return torch_doc, pd_doc

    # -------- LLM call --------

    def call_llm(self, execution_result: Dict, torch_tc: Dict, pd_tc: Dict,
                 torch_doc: str = "", pd_doc: str = "") -> Dict[str, Any]:
        """Call LLM to decide repair/mutation/skip."""
        torch_api = execution_result.get("torch_api", "")
        paddle_api = execution_result.get("paddle_api", "")
        status = execution_result.get("status", "")
        torch_success = execution_result.get("torch_success", False)
        paddle_success = execution_result.get("paddle_success", False)
        results_match = execution_result.get("results_match", False)
        torch_error = execution_result.get("torch_error", "")
        paddle_error = execution_result.get("paddle_error", "")
        comparison_error = execution_result.get("comparison_error", "")

        # Simplify PyTorch test case
        simplified_torch_test_case = {}
        for key, value in torch_tc.items():
            if isinstance(value, np.ndarray):
                simplified_torch_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_torch_test_case[key] = value

        # Simplify PaddlePaddle test case
        simplified_paddle_test_case = {}
        for key, value in pd_tc.items():
            if isinstance(value, np.ndarray):
                simplified_paddle_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_paddle_test_case[key] = value

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

        # Build PaddlePaddle parameter examples
        pd_param_examples = []
        for key, value in simplified_paddle_test_case.items():
            if key == "api":
                continue
            if isinstance(value, dict) and "shape" in value:
                pd_param_examples.append(f'    "{key}": {json.dumps(value)}')
            elif isinstance(value, (int, float)):
                pd_param_examples.append(f'    "{key}": {value}')
            else:
                pd_param_examples.append(f'    "{key}": {json.dumps(value)}')

        pd_param_example_str = ",\n".join(pd_param_examples) if pd_param_examples else '    "x": {"shape": [2, 3], "dtype": "float32"}'

        # Build API documentation section
        doc_section = ""
        if torch_doc or pd_doc:
            doc_section = "\n## Official API Documentation\n\n"
            if torch_doc:
                doc_section += f"### PyTorch {torch_api} Docs\n```\n{torch_doc}\n```\n\n"
            if pd_doc:
                doc_section += f"### PaddlePaddle {paddle_api} Docs\n```\n{pd_doc}\n```\n\n"

                prompt = f"""Please analyze the following operator test cases in PyTorch and PaddlePaddle, then repair or mutate (fuzz) the cases based on the results.

## Test Info
- **PyTorch API**: {torch_api}
- **PaddlePaddle API**: {paddle_api}
{doc_section}
## Execution Results
- **Status**: {status}
- **PyTorch success**: {torch_success}
- **PaddlePaddle success**: {paddle_success}
- **Results match**: {results_match}

## Error Info
- **PyTorch error**: {torch_error if torch_error else "None"}
- **PaddlePaddle error**: {paddle_error if paddle_error else "None"}
- **Comparison error**: {comparison_error if comparison_error else "None"}

## Original Test Cases

### PyTorch test case
```json
{json.dumps(simplified_torch_test_case, indent=2, ensure_ascii=False)}
```

### PaddlePaddle test case
```json
{json.dumps(simplified_paddle_test_case, indent=2, ensure_ascii=False)}
```

## Task
Based on the above info (including official API docs), determine whether the cross-framework results are **consistent**, **inconsistent**, or **execution failed**, then take action:

1. **If consistent**: **mutate (fuzz)** the case, e.g., change input shapes or parameter values (try edge cases)
2. **If execution failed**: **repair** the case using error messages and docs (adjust parameter names, counts, types, ranges; frameworks may differ) or **skip** if the operators are not equivalent
3. **If inconsistent**: decide whether it is tolerable precision error (<= 1e-3). (1) If tolerable, **mutate**; (2) If operators are not equivalent, **skip**; (3) If neither, treat as case-construction issue and **repair** according to docs.

## Output Format
Return strictly the following JSON only (no extra text, comments, or markdown):

{{
    "operation": "mutation",
    "reason": "Detailed reason for this action",
    "pytorch_test_case": {{
        "api": "{torch_api}",
{torch_param_example_str}
    }},
    "paddle_test_case": {{
        "api": "{paddle_api}",
{pd_param_example_str}
    }}
}}

**Important**:
1. operation must be one of "mutation", "repair", or "skip"
2. Tensor params must use {{"shape": [...], "dtype": "..."}} format
3. Scalar params use direct values (e.g., "y": 0)
4. Ensure inputs are aligned across frameworks (convert shapes if needed) and parameters are semantically equivalent
5. Parameter names/values/counts may differ between PyTorch and PaddlePaddle as long as outputs should match
6. If official docs are missing because the API is absent or removed in current versions, set operation to "skip" and do not attempt repair
7. For mutation, explore edge cases: empty tensors (shape contains 0), single-element tensors (shape=[1] or []), high-dim tensors, huge tensors, different dtypes (int/float/bool), boundary values
8. Read the official API docs carefully and keep parameter names/types/ranges consistent with the docs
"""

        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a deep learning framework testing expert who knows PyTorch and PaddlePaddle API differences. Return strict JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )
            raw = completion.choices[0].message.content.strip()
            time.sleep(1)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    return json.loads(match.group())
                return {"operation": "skip", "reason": "LLM returned invalid format"}
        except Exception as e:
            return {"operation": "skip", "reason": f"LLM call failed: {e}"}

    # -------- Convert LLM test cases --------

    def convert_llm_test_cases(self, pt_tc: Dict, pd_tc: Dict) -> Tuple[Dict, Dict]:
        """Convert LLM-returned test cases to executable format (share tensor data)."""
        shared_tensors = {}
        all_keys = set(pt_tc.keys()) | set(pd_tc.keys())
        for key in all_keys:
            if key == "api":
                continue
            pt_val = pt_tc.get(key)
            pd_val = pd_tc.get(key)
            tensor_desc = None
            if isinstance(pt_val, dict) and "shape" in pt_val:
                tensor_desc = pt_val
            elif isinstance(pd_val, dict) and "shape" in pd_val:
                tensor_desc = pd_val
            if tensor_desc:
                shared_tensors[key] = self.generate_numpy_data(tensor_desc)

        converted_pt = {}
        for k, v in pt_tc.items():
            converted_pt[k] = shared_tensors.get(k, v)
        converted_pd = {}
        for k, v in pd_tc.items():
            converted_pd[k] = shared_tensors.get(k, v)
        return converted_pt, converted_pd

    def close(self):
        self.client.close()


# ==================== Main comparison logic ====================

class LLMvsX2PaddleComparator:
    """
    Main comparison class: manages LLM and X2Paddle (ONNX) comparison tests.
    
    Runs both methods for each operator and case, then summarizes results.
    """

    def __init__(self, key_path: str = DEFAULT_KEY_PATH,
                 model: str = DEFAULT_MODEL,
                 num_workers: int = DEFAULT_WORKERS):
        self.print_lock = Lock()
        self.num_workers = max(1, num_workers)
        self.execution_lock = RLock()
        self.realtime_lock = Lock()

        self.llm_method = LLMMethod(
            key_path=key_path, model=model,
            print_lock=self.print_lock,
            llm_workers=self.num_workers
        )
        self.x2paddle_converter = X2PaddleConverter(print_lock=self.print_lock)

        # Result directory
        self.result_dir = os.path.join(ROOT_DIR, "pt_pd_test")
        os.makedirs(self.result_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.realtime_file_path = os.path.join(self.result_dir, f"llm_vs_x2paddle_realtime_{timestamp}.jsonl")
        self.realtime_file = open(self.realtime_file_path, "a", encoding="utf-8")

    def _safe_print(self, msg: str, end: str = "\n"):
        safe_print(msg, self.print_lock, end)

    def _append_realtime_record(self, record: Dict[str, Any]):
        """Append a realtime JSONL record."""
        with self.realtime_lock:
            self.realtime_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.realtime_file.flush()

    def run_comparison(self, operator_names: List[str],
                       num_cases: int = DEFAULT_NUM_CASES,
                       max_iterations: int = DEFAULT_MAX_ITERATIONS) -> Dict[str, Any]:
        """
        Run LLM vs X2Paddle comparison for a batch of operators.
        
        Returns summary result dict.
        """
        global_stats = {
            "total_operators": len(operator_names),
            "tested_operators": 0,
            "skipped_operators_llm": 0,
            "skipped_operators_no_pd": 0,
            "skipped_operators_problematic": 0,
            "skipped_operators_deprecated": 0,

            # LLM method stats
            "llm_total_cases": 0,
            "llm_pd_success": 0,

            # X2Paddle method stats
            "x2paddle_total_cases": 0,
            "x2paddle_export_success": 0,
            "x2paddle_run_success": 0,
        }

        operator_details = []

        for idx, op_name in enumerate(operator_names, 1):
            self._safe_print(f"\n{'='*70}")
            self._safe_print(f"[{idx}/{len(operator_names)}] Operator: {op_name}")
            self._safe_print(f"{'='*70}")

            # 0. Skip operators that may hang
            if op_name in self.llm_method.problematic_apis:
                reason = self.llm_method.problematic_apis.get(op_name, "May hang the program")
                self._safe_print(f"  ⏭️ Skipped ({reason})")
                global_stats["skipped_operators_problematic"] += 1
                operator_details.append({
                    "operator": op_name,
                    "status": "skipped_problematic",
                    "reason": reason
                })
                continue

            # 1. Resolve API mapping
            torch_api, pd_api, mapping_method = self.llm_method.convert_api_name(op_name)
            if pd_api is None:
                self._safe_print(f"  ⏭️ No PD implementation ({mapping_method}), skipping")
                global_stats["skipped_operators_no_pd"] += 1
                operator_details.append({
                    "operator": op_name, "status": "skipped_no_pd",
                    "mapping_method": mapping_method
                })
                continue

            self._safe_print(f"  PT: {torch_api} -> PD: {pd_api}")

            # 2. Get document from MongoDB
            document = self.llm_method.collection.find_one({"api": op_name})
            if document is None:
                self._safe_print(f"  ❌ Not found in database")
                operator_details.append({"operator": op_name, "status": "not_found"})
                continue

            total_cases = self.llm_method.get_num_test_cases(document)
            actual_cases = min(num_cases, total_cases)

            # 3. Test both LLM and X2Paddle for each case
            op_result = self._test_operator(
                op_name, torch_api, pd_api, document,
                actual_cases, max_iterations
            )

            operator_details.append(op_result)

            # 4. Summarize stats
            if op_result["status"] == "skipped_by_llm":
                global_stats["skipped_operators_llm"] += 1
            elif op_result["status"] == "skipped_deprecated":
                global_stats["skipped_operators_deprecated"] += 1
            else:
                global_stats["tested_operators"] += 1
                global_stats["llm_total_cases"] += op_result["llm_total"]
                global_stats["llm_pd_success"] += op_result["llm_pd_success"]
                global_stats["x2paddle_total_cases"] += op_result["x2paddle_total"]
                global_stats["x2paddle_export_success"] += op_result["x2paddle_export_success"]
                global_stats["x2paddle_run_success"] += op_result["x2paddle_run_success"]

        return {"global_stats": global_stats, "operator_details": operator_details}

    def _test_operator(self, op_name: str, torch_api: str, pd_api: str,
                       document: Dict, num_cases: int,
                       max_iterations: int) -> Dict[str, Any]:
        """Test LLM and X2Paddle methods for multiple cases of one operator."""
        op_result = {
            "operator": op_name,
            "torch_api": torch_api,
            "pd_api": pd_api,
            "status": "completed",
            "num_cases": num_cases,
            "llm_total": 0,
            "llm_pd_success": 0,
            "x2paddle_total": 0,
            "x2paddle_export_success": 0,
            "x2paddle_run_success": 0,
            "case_details": []
        }

        is_class_api = self.llm_method.is_class_based_api(torch_api)

        # Prepare all test cases
        cases = []
        for case_idx in range(num_cases):
            tc = self.llm_method.prepare_shared_numpy_data(document, case_index=case_idx)
            tc["api"] = torch_api
            cases.append((case_idx + 1, tc))

        # Prefetch docs (only once)
        self._safe_print("  📖 Fetching API docs...")
        torch_doc, pd_doc = self.llm_method.fetch_api_docs(torch_api, pd_api)

        # Process cases concurrently with thread pool
        if self.num_workers <= 1:
            case_results = []
            for case_number, tc in cases:
                cr = self._process_single_case(
                    torch_api, pd_api, tc, case_number,
                    is_class_api, torch_doc, pd_doc, max_iterations
                )
                case_results.append(cr)
        else:
            case_results = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_map = {}
                for case_number, tc in cases:
                    future = executor.submit(
                        self._process_single_case,
                        torch_api, pd_api, tc, case_number,
                        is_class_api, torch_doc, pd_doc, max_iterations
                    )
                    future_map[future] = case_number
                for future in as_completed(future_map):
                    case_results.append(future.result())

        case_results.sort(key=lambda x: x["case_number"])

        # Check whether skipped due to deprecation
        any_deprecated = any(cr.get("deprecated_skip", False) for cr in case_results)
        if any_deprecated:
            op_result["status"] = "skipped_deprecated"
            op_result["case_details"] = case_results
            self._safe_print("  ⏭️ Operator deprecated, skipping")
            return op_result

        # Check whether all cases were skipped by LLM
        all_skipped = all(cr.get("llm_skipped", False) for cr in case_results)
        if all_skipped and len(case_results) > 0:
            op_result["status"] = "skipped_by_llm"
            op_result["case_details"] = case_results
            self._safe_print("  ⏭️ LLM chose to skip this operator")
            return op_result

        # Summarize per-case results
        for cr in case_results:
            if not cr.get("llm_skipped", False):
                op_result["llm_total"] += cr.get("llm_total", 0)
                op_result["llm_pd_success"] += cr.get("llm_pd_success", 0)
                op_result["x2paddle_total"] += cr.get("x2paddle_total", 0)
                op_result["x2paddle_export_success"] += cr.get("x2paddle_export_success", 0)
                op_result["x2paddle_run_success"] += cr.get("x2paddle_run_success", 0)
        op_result["case_details"] = case_results

        return op_result

    def _process_single_case(self, torch_api: str, pd_api: str,
                             test_case: Dict[str, Any], case_number: int,
                             is_class_api: bool,
                             torch_doc: str, pd_doc: str,
                             max_iterations: int) -> Dict[str, Any]:
        """
        Process one case with both LLM and X2Paddle methods.
        
        LLM method (1 iteration):
            1. Run DB case in PT/PD
            2. Call LLM for repair/mutation/skip
            3. If mutation/repair, run LLM-generated case
        
        X2Paddle method:
            1. Wrap DB case as PyTorch module
            2. torch.onnx.export + onnxruntime inference (simulate X2Paddle ONNX core path)
        """
        case_result = {
            "case_number": case_number,
            "llm_skipped": False,
            "deprecated_skip": False,
            "llm_total": 0,
            "llm_pd_success": 0,
            "x2paddle_total": 0,
            "x2paddle_export_success": 0,
            "x2paddle_run_success": 0,
            "llm_detail": None,
            "x2paddle_detail": None,
        }

        # ========== X2Paddle method ==========
        self._safe_print(f"  [Case {case_number}] X2Paddle method...", end="")
        with self.execution_lock:
            x2paddle_result = self.x2paddle_converter.convert_and_run(
                torch_api, test_case, is_class_api
            )
        case_result["x2paddle_total"] = 1
        if x2paddle_result["onnx_export_success"]:
            case_result["x2paddle_export_success"] = 1
        if x2paddle_result["onnx_run_success"]:
            case_result["x2paddle_run_success"] = 1

        x2p_status = "✓" if x2paddle_result["onnx_run_success"] else "✗"
        x2p_err = f" ({x2paddle_result['error'][:60]})" if x2paddle_result.get("error") else ""
        self._safe_print(f" X2Paddle:{x2p_status}{x2p_err}")
        case_result["x2paddle_detail"] = x2paddle_result

        # ========== LLM method ==========
        # Step 1: run DB original case in both frameworks
        torch_test_case = test_case
        pd_test_case = copy.deepcopy(test_case)
        pd_test_case["api"] = pd_api

        self._safe_print(f"  [Case {case_number}] LLM method: initial run...", end="")
        try:
            with self.execution_lock:
                exec_result = self.llm_method._execute_sequential(
                    torch_api, pd_api, torch_test_case, pd_test_case
                )
            pt_s = "✓" if exec_result['torch_success'] else "✗"
            pd_s = "✓" if exec_result['paddle_success'] else "✗"
            self._safe_print(f" PT:{pt_s} PD:{pd_s}")
        except Exception as e:
            self._safe_print(f" ❌ Execution failed: {str(e)[:60]}")
            exec_result = {
                "torch_api": torch_api, "paddle_api": pd_api,
                "torch_success": False, "paddle_success": False,
                "results_match": False, "status": "fatal_error",
                "torch_error": str(e), "paddle_error": None,
                "comparison_error": None
            }

        # If PyTorch error includes deprecated/removed, skip the operator
        torch_error = str(exec_result.get("torch_error", ""))
        if not exec_result.get("torch_success", False) and torch_error:
            if re.search(r"deprecated|removed", torch_error, re.IGNORECASE):
                case_result["deprecated_skip"] = True
                case_result["llm_skipped"] = False
                case_result["llm_detail"] = {
                    "initial_exec": exec_result,
                    "llm_operation": "skip",
                    "llm_reason": "PyTorch operator is deprecated"
                }
                self._safe_print("  Deprecation detected, skipping")
                self._append_realtime_record({
                    "timestamp": datetime.now().isoformat(),
                    "operator": torch_api,
                    "case_number": case_number,
                    "status": "skipped_deprecated",
                    "x2paddle_run_success": bool(case_result.get("x2paddle_run_success", 0)),
                    "llm_pd_success": False,
                    "llm_skipped": False,
                    "deprecated_skip": True
                })
                return case_result

        # Step 2: call LLM
        self._safe_print(f"  [Case {case_number}] Calling LLM...", end="")
        llm_result = self.llm_method.call_llm(
            exec_result, torch_test_case, pd_test_case, torch_doc, pd_doc
        )
        operation = llm_result.get("operation", "skip")
        reason = llm_result.get("reason", "")[:60]
        self._safe_print(f" {operation} - {reason}")

        if operation == "skip":
            case_result["llm_skipped"] = True
            case_result["llm_detail"] = {
                "initial_exec": exec_result,
                "llm_operation": operation,
                "llm_reason": llm_result.get("reason", "")
            }
            self._append_realtime_record({
                "timestamp": datetime.now().isoformat(),
                "operator": torch_api,
                "case_number": case_number,
                "status": "llm_skip",
                "x2paddle_run_success": bool(case_result.get("x2paddle_run_success", 0)),
                "llm_pd_success": False,
                "llm_skipped": True,
                "deprecated_skip": False
            })
            return case_result

        # Step 3: if LLM returned mutation/repair, execute LLM case
        llm_pt_tc = llm_result.get("pytorch_test_case", torch_test_case)
        llm_pd_tc = llm_result.get("paddle_test_case", pd_test_case)

        # Convert to executable format
        try:
            llm_pt_tc, llm_pd_tc = self.llm_method.convert_llm_test_cases(llm_pt_tc, llm_pd_tc)
        except Exception as e:
            case_result["llm_skipped"] = True
            case_result["llm_detail"] = {
                "initial_exec": exec_result,
                "llm_operation": "skip",
                "llm_reason": f"Invalid shape from LLM or failed to create tensor: {e}"
            }
            self._safe_print(f"  [Case {case_number}] LLM case conversion failed, skipping: {str(e)[:60]}")
            self._append_realtime_record({
                "timestamp": datetime.now().isoformat(),
                "operator": torch_api,
                "case_number": case_number,
                "status": "llm_case_convert_failed",
                "x2paddle_run_success": bool(case_result.get("x2paddle_run_success", 0)),
                "llm_pd_success": False,
                "llm_skipped": True,
                "deprecated_skip": False
            })
            return case_result

        case_result["llm_total"] = 1

        self._safe_print(f"  [Case {case_number}] Executing LLM case...", end="")
        try:
            with self.execution_lock:
                llm_exec = self.llm_method._execute_sequential(
                    torch_api, pd_api, llm_pt_tc, llm_pd_tc
                )
            pt_s = "✓" if llm_exec['torch_success'] else "✗"
            pd_s = "✓" if llm_exec['paddle_success'] else "✗"
            match_s = "✓" if llm_exec['results_match'] else "✗"
            self._safe_print(f" PT:{pt_s} PD:{pd_s} Match:{match_s}")

            if llm_exec["paddle_success"]:
                case_result["llm_pd_success"] = 1

        except Exception as e:
            self._safe_print(f" ❌ {str(e)[:60]}")
            llm_exec = {
                "torch_success": False, "paddle_success": False,
                "results_match": False, "status": "fatal_error",
                "error": str(e)
            }

        case_result["llm_detail"] = {
            "initial_exec": exec_result,
            "llm_operation": operation,
            "llm_reason": llm_result.get("reason", ""),
            "llm_exec": llm_exec
        }

        self._append_realtime_record({
            "timestamp": datetime.now().isoformat(),
            "operator": torch_api,
            "case_number": case_number,
            "status": "completed",
            "x2paddle_run_success": bool(case_result.get("x2paddle_run_success", 0)),
            "llm_pd_success": bool(case_result.get("llm_pd_success", 0)),
            "llm_skipped": False,
            "deprecated_skip": False
        })

        return case_result

    def print_and_save_results(self, comparison_result: Dict[str, Any]):
        """Print results and save to file."""
        gs = comparison_result["global_stats"]

        print("\n" + "=" * 80)
        print("📊 LLM vs X2Paddle — test case conversion success rate")
        print("=" * 80)

        print(f"\n📌 Total operators: {gs['total_operators']}")
        print(f"   - Skipped (no PD mapping): {gs['skipped_operators_no_pd']}")
        print(f"   - Skipped (problematic): {gs['skipped_operators_problematic']}")
        print(f"   - Skipped (LLM chose): {gs['skipped_operators_llm']}")
        print(f"   - Skipped (deprecated): {gs['skipped_operators_deprecated']}")
        print(f"   - Actually compared: {gs['tested_operators']}")

        print(f"\n{'─'*40}")
        print("🤖 LLM method (excluding skipped operators):")
        print(f"   Total LLM-generated PD cases: {gs['llm_total_cases']}")
        print(f"   PD success count: {gs['llm_pd_success']}")
        if gs['llm_total_cases'] > 0:
            llm_rate = gs['llm_pd_success'] / gs['llm_total_cases'] * 100
            print(f"   ✅ PD success rate: {llm_rate:.2f}%")
        else:
            llm_rate = 0
            print("   ✅ PD success rate: N/A (no LLM-generated cases)")

        print(f"\n{'─'*40}")
        print("🔄 X2Paddle method (excluding LLM-skipped operators):")
        print(f"   Total X2Paddle attempts: {gs['x2paddle_total_cases']}")
        print(f"   ONNX export success: {gs['x2paddle_export_success']}")
        print(f"   ONNX inference success (= conversion success): {gs['x2paddle_run_success']}")
        if gs['x2paddle_total_cases'] > 0:
            x2p_rate = gs['x2paddle_run_success'] / gs['x2paddle_total_cases'] * 100
            print(f"   ✅ X2Paddle success rate: {x2p_rate:.2f}%")
        else:
            x2p_rate = 0
            print("   ✅ X2Paddle success rate: N/A")

        print(f"\n{'─'*40}")
        print("📈 Comparison conclusion:")
        if gs['llm_total_cases'] > 0 and gs['x2paddle_total_cases'] > 0:
            diff = llm_rate - x2p_rate
            if diff > 0:
                print(f"   LLM outperforms X2Paddle by {diff:.2f} percentage points")
            elif diff < 0:
                print(f"   X2Paddle outperforms LLM by {-diff:.2f} percentage points")
            else:
                print("   Success rates are tied")
        else:
            print("   Insufficient data to compare")
        print("=" * 80)

        # -------- Save to file --------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.result_dir, f"llm_vs_x2paddle_result_{timestamp}.json")

        save_data = self._make_serializable(comparison_result)
        save_data["summary"] = {
            "llm_pd_success_rate": f"{llm_rate:.2f}%" if gs['llm_total_cases'] > 0 else "N/A",
            "x2paddle_success_rate": f"{x2p_rate:.2f}%" if gs['x2paddle_total_cases'] > 0 else "N/A",
            "skipped_operators_deprecated": gs.get("skipped_operators_deprecated", 0),
            "timestamp": datetime.now().isoformat()
        }

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Detailed results saved to: {result_file}")

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """Recursively convert objects to JSON-serializable formats."""
        if isinstance(obj, dict):
            return {k: LLMvsX2PaddleComparator._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [LLMvsX2PaddleComparator._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return {"__type__": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (torch.Tensor,)):
            return {"__type__": "tensor", "shape": list(obj.shape), "dtype": str(obj.dtype)}
        elif isinstance(obj, bytes):
            return "<bytes>"
        else:
            return obj

    def close(self):
        try:
            if self.realtime_file:
                self.realtime_file.flush()
                self.realtime_file.close()
        finally:
            self.llm_method.close()


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="LLM vs X2Paddle: PyTorch->PaddlePaddle test case conversion success rate"
    )
    parser.add_argument("--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES,
                        help=f"Cases per operator (default {DEFAULT_NUM_CASES})")
    parser.add_argument("--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS,
                        help=f"LLM iterations per case (default {DEFAULT_MAX_ITERATIONS})")
    parser.add_argument("--start", type=int, default=1,
                        help="Start operator index (1-based)")
    parser.add_argument("--end", type=int, default=None,
                        help="End operator index (inclusive)")
    parser.add_argument("--operators", "-o", nargs="*",
                        help="Specify operator name list")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS,
                        help=f"LLM worker threads (default {DEFAULT_WORKERS})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"LLM model (default {DEFAULT_MODEL})")
    parser.add_argument("--key-path", "-k", default=DEFAULT_KEY_PATH,
                        help=f"API key path (default {DEFAULT_KEY_PATH})")
    args = parser.parse_args()

    print("=" * 80)
    print("LLM vs X2Paddle — PyTorch->PaddlePaddle test case conversion success rate")
    print("=" * 80)
    print(f"📌 Cases per operator: {args.num_cases}")
    print(f"📌 LLM iterations: {args.max_iterations}")
    print(f"📌 Worker threads: {args.workers}")
    print(f"📌 LLM model: {args.model}")
    print("=" * 80)

    comparator = LLMvsX2PaddleComparator(
        key_path=args.key_path, model=args.model,
        num_workers=args.workers
    )

    start_time = time.time()

    try:
        # Fetch operator list
        all_docs = list(comparator.llm_method.collection.find({}, {"api": 1}))
        all_ops = [doc["api"] for doc in all_docs if "api" in doc]
        print(f"\n📋 Total operators in database: {len(all_ops)}")

        if args.operators:
            operator_names = args.operators
        else:
            start_idx = max(1, args.start) - 1
            end_idx = args.end if args.end is not None else len(all_ops)
            end_idx = min(end_idx, len(all_ops))
            operator_names = all_ops[start_idx:end_idx]
            print(f"📌 Test range: operators {start_idx + 1} ~ {end_idx}")

        print(f"📋 Will test {len(operator_names)} operators")
        print(f"📋 Top 10: {', '.join(operator_names[:10])}{'...' if len(operator_names) > 10 else ''}\n")

        # Run comparison
        result = comparator.run_comparison(
            operator_names,
            num_cases=args.num_cases,
            max_iterations=args.max_iterations
        )

        # Output and save
        comparator.print_and_save_results(result)

        elapsed = time.time() - start_time
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
        print(f"\n⏱️ Total elapsed: {h}h {m}m {s}s")

    finally:
        comparator.close()
        print("✅ Program completed")


if __name__ == "__main__":
    main()
