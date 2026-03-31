# ./pt_ms_test/compare_llm_vs_mindconverter.py
"""
LLM method vs MindConverter method: PyTorch → MindSpore test case conversion success rate
=======================================================================

Compare two cross-framework test case migration approaches:
1. LLM method: extract cases from MongoDB → build PT/MS cases → run → LLM repair/mutation → run LLM-generated cases
2. MindConverter method: extract cases from MongoDB → wrap as a small PyTorch model → torch.onnx.export → onnxruntime inference
     (simulate MindConverter core path: PyTorch → ONNX → MindSpore)

MindConverter overview:
    - Official MindSpore model migration tool that converts PyTorch (ONNX) models to MindSpore
    - Core conversion path relies on ONNX as intermediate: PyTorch → ONNX → MindSpore (model.py + ckpt)
    - Supports CLI (mindconverter --model_file) and API (pytorch2mindspore)
    - Operator-driven mapping between ONNX operators and MindSpore operators
    - Note: MindConverter stopped evolving after 1.9.0; official recommendation is version 1.7.0

Metric: LLM-generated MS case execution success rate vs MindConverter (ONNX) export + inference success rate
(excluding operators skipped by LLM)
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

# Add project root directory to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from component.doc.doc_crawler_factory import get_doc_content

# ==================== Constants ====================
DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"
DEFAULT_MAX_ITERATIONS = 1   # Simplified comparison: only 1 iteration
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


# ==================== MindConverter (ONNX) Converter ====================

class MindConverterSimulator:
    """
    Simulate MindConverter's PyTorch → MindSpore test case migration via ONNX.

    MindConverter core conversion path:
    1. PyTorch model → torch.onnx.export → ONNX model
    2. ONNX model → MindConverter internal operator mapping → MindSpore model (model.py + ckpt)

    Simulation flow (operator-level):
    1. Wrap PyTorch operator as a minimal nn.Module
    2. Export to ONNX via torch.onnx.export
    3. Run onnxruntime inference

    Success criterion: ONNX export + OnnxRuntime inference success == MindConverter conversion success
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
            init_kwargs: Init kwargs for class operators
            extra_kwargs: Non-tensor kwargs for function operators (captured via closure)

        Returns:
            Wrapped nn.Module or None on failure
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
        """Resolve attribute by path like ['torch', 'nn', 'ReLU']."""
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
        Run MindConverter (ONNX) conversion + inference for one test case.

        Steps:
        1. Prepare PyTorch input tensors
        2. Run PyTorch forward pass as reference
        3. torch.onnx.export (simulate MindConverter ONNX export stage)
        4. onnxruntime inference (simulate MindConverter MindSpore inference stage)
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
                result["error"] = f"Unable to wrap operator {torch_api} as nn.Module"
                return result

            module.eval()

            # ---------- 3. PyTorch forward pass ----------
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

            # ---------- 4. ONNX export (simulate MindConverter ONNX export stage) ----------
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
                result["error"] = f"ONNX export failed (MindConverter core path): {e}"
                if onnx_path and os.path.exists(onnx_path):
                    os.remove(onnx_path)
                return result

            # ---------- 5. OnnxRuntime inference (simulate MindConverter inference stage) ----------
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

        # Collect positional params as input tensors in order
        for param_name in positional_tensor_params:
            if param_name in test_case:
                value = test_case[param_name]
                tensor = self._to_torch_tensor(value)
                if tensor is not None:
                    input_tensors.append(tensor)
                    input_names.append(param_name)

        # Other params
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
        """Convert values of various formats to PyTorch tensors."""
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
    """Wrap a regular function as nn.Module for torch.onnx.export."""

    def __init__(self, func):
        super().__init__()
        self._func = func

    def forward(self, *args):
        return self._func(*args)


# ==================== LLM Method (Simplified) ====================

class LLMMethod:
    """
    LLM-based test case conversion method (simplified, 1 iteration).

    Flow: extract cases from MongoDB → run on PT/MS → LLM decides repair/skip → run LLM-generated cases
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
        mindspore.set_seed(42)

        # Set MindSpore to PyNative (eager) mode
        mindspore.set_context(mode=mindspore.PYNATIVE_MODE)

        # Operators that may hang or crash (skip testing these)
        # Note: torch.as_tensor is not skipped
        self.problematic_apis = {
            "torch.triu": "May hang the program",
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
        mapping_file = os.path.join(ROOT_DIR, "component", "data", "ms_api_mappings_final.csv")
        try:
            df = pd.read_csv(mapping_file)
            mapping = {}
            for _, row in df.iterrows():
                pt_api = str(row["pytorch-api"]).strip()
                ms_api = str(row["mindspore-api"]).strip()
                mapping[pt_api] = {"ms_api": ms_api}
            return mapping
        except Exception as e:
            self._safe_print(f"❌ Failed to load mapping table: {e}")
            return {}

    def convert_api_name(self, torch_api: str) -> Tuple[Optional[str], Optional[str], str]:
        if torch_api in self.api_mapping:
            ms_api = self.api_mapping[torch_api]["ms_api"]
            if ms_api in ("No implementation", "NONE", ""):
                return torch_api, None, "No implementation"
            return torch_api, ms_api, "Mapping table"
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
            elif framework == "mindspore" and parts[0] == "mindspore":
                obj = mindspore
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

    def prepare_arguments_mindspore(self, test_case: Dict[str, Any], mindspore_api: str) -> Tuple[List[Any], Dict[str, Any]]:
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
                        args.append(mindspore.Tensor(arr))
                    elif isinstance(item, np.ndarray):
                        args.append(mindspore.Tensor(item.copy()))
                    else:
                        args.append(item)
            return args, kwargs

        positional_params = ["condition", "x", "y", "input", "other"]
        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, np.ndarray):
                    args.append(mindspore.Tensor(value.copy()))
                else:
                    args.append(value)

        for key, value in test_case.items():
            if key not in positional_params + ["api"]:
                if key in common_skip:
                    continue
                if isinstance(value, np.ndarray):
                    kwargs[key] = mindspore.Tensor(value.copy())
                else:
                    kwargs[key] = value
        return args, kwargs

    # -------- Execute test cases --------

    def execute_test_case(self, torch_api: str, mindspore_api: str,
                          torch_test_case: Dict[str, Any],
                          mindspore_test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a test case in PyTorch and MindSpore separately."""
        result = {
            "torch_api": torch_api, "mindspore_api": mindspore_api,
            "torch_success": False, "mindspore_success": False,
            "results_match": False,
            "torch_error": None, "mindspore_error": None,
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
                        raise ValueError("Class operator missing input argument")
                    with torch.no_grad():
                        torch_result = instance(input_data)
                else:
                    with torch.no_grad():
                        torch_result = torch_func(*args, **kwargs)
                result["torch_success"] = True
        except Exception as e:
            result["torch_error"] = str(e)

        # -- MindSpore --
        ms_result = None
        try:
            ms_func = self.get_operator_function(mindspore_api, "mindspore")
            if ms_func is None:
                result["mindspore_error"] = f"Operator {mindspore_api} not found"
            else:
                args, kwargs = self.prepare_arguments_mindspore(mindspore_test_case, mindspore_api)
                if is_class_api:
                    init_kwargs = {k: v for k, v in kwargs.items() if k not in ['x', 'input']}
                    instance = ms_func(**init_kwargs)
                    input_data = kwargs.get('x', kwargs.get('input', args[0] if args else None))
                    if input_data is None:
                        raise ValueError("Class operator missing input/x argument")
                    ms_result = instance(input_data)
                else:
                    ms_result = ms_func(*args, **kwargs)
                result["mindspore_success"] = True
        except Exception as e:
            result["mindspore_error"] = str(e)

        # -- Compare --
        if result["torch_success"] and result["mindspore_success"]:
            try:
                torch_np = torch_result.detach().cpu().numpy() if hasattr(torch_result, 'detach') else np.array(torch_result)
                ms_np = ms_result.asnumpy() if hasattr(ms_result, 'asnumpy') else np.array(ms_result)
                if torch_np.shape != ms_np.shape:
                    result["comparison_error"] = f"Shape mismatch: {torch_np.shape} vs {ms_np.shape}"
                elif np.allclose(torch_np, ms_np, atol=1e-5, rtol=1e-5, equal_nan=True):
                    result["results_match"] = True
                else:
                    max_diff = np.max(np.abs(torch_np.astype(float) - ms_np.astype(float)))
                    result["comparison_error"] = f"Max difference: {max_diff}"
                result["status"] = "compared"
            except Exception as e:
                result["comparison_error"] = str(e)
                result["status"] = "comparison_failed"
        elif result["torch_success"]:
            result["status"] = "mindspore_failed"
        elif result["mindspore_success"]:
            result["status"] = "torch_failed"
        else:
            result["status"] = "both_failed"

        return result

    def _execute_sequential(self, torch_api, mindspore_api, torch_tc, ms_tc):
        with self.execution_lock:
            return self.execute_test_case(torch_api, mindspore_api, torch_tc, ms_tc)

    # -------- Documentation fetch --------

    def fetch_api_docs(self, torch_api: str, mindspore_api: str) -> Tuple[str, str]:
        MIN_DOC_LENGTH = 300
        torch_doc, ms_doc = "", ""
        try:
            torch_doc = get_doc_content(torch_api, "pytorch")
            if not (torch_doc and "Unable" not in torch_doc and len(torch_doc.strip()) > MIN_DOC_LENGTH):
                torch_doc = f"Unable to fetch documentation for {torch_api}"
            elif len(torch_doc) > 3000:
                torch_doc = torch_doc[:3000] + "\n... (doc truncated)"
        except Exception:
            torch_doc = ""
        try:
            ms_doc = get_doc_content(mindspore_api, "mindspore")
            if not (ms_doc and "Unable" not in ms_doc and len(ms_doc.strip()) > MIN_DOC_LENGTH):
                ms_doc = f"Unable to fetch documentation for {mindspore_api}"
            elif len(ms_doc) > 3000:
                ms_doc = ms_doc[:3000] + "\n... (doc truncated)"
        except Exception:
            ms_doc = ""
        return torch_doc, ms_doc

    # -------- LLM call --------

    def call_llm(self, execution_result: Dict, torch_tc: Dict, ms_tc: Dict,
                 torch_doc: str = "", ms_doc: str = "") -> Dict[str, Any]:
        """Call LLM to decide repair/mutation/skip."""
        torch_api = execution_result.get("torch_api", "")
        mindspore_api = execution_result.get("mindspore_api", "")
        status = execution_result.get("status", "")
        torch_success = execution_result.get("torch_success", False)
        mindspore_success = execution_result.get("mindspore_success", False)
        results_match = execution_result.get("results_match", False)
        torch_error = execution_result.get("torch_error", "")
        mindspore_error = execution_result.get("mindspore_error", "")
        comparison_error = execution_result.get("comparison_error", "")

        # Simplify PyTorch test case
        simplified_torch_test_case = {}
        for key, value in torch_tc.items():
            if isinstance(value, np.ndarray):
                simplified_torch_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_torch_test_case[key] = value

        # Simplify MindSpore test case
        simplified_mindspore_test_case = {}
        for key, value in ms_tc.items():
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
        ms_param_examples = []
        for key, value in simplified_mindspore_test_case.items():
            if key == "api":
                continue
            if isinstance(value, dict) and "shape" in value:
                ms_param_examples.append(f'    "{key}": {json.dumps(value)}')
            elif isinstance(value, (int, float)):
                ms_param_examples.append(f'    "{key}": {value}')
            else:
                ms_param_examples.append(f'    "{key}": {json.dumps(value)}')

        ms_param_example_str = ",\n".join(ms_param_examples) if ms_param_examples else '    "x": {"shape": [2, 3], "dtype": "float32"}'

        # Build API documentation section
        doc_section = ""
        if torch_doc or ms_doc:
            doc_section = "\n## Official API Documentation\n\n"
            if torch_doc:
                doc_section += f"### PyTorch {torch_api} Documentation\n```\n{torch_doc}\n```\n\n"
            if ms_doc:
                doc_section += f"### MindSpore {mindspore_api} Documentation\n```\n{ms_doc}\n```\n\n"

                prompt = f"""Please analyze the following operator test cases in PyTorch and MindSpore, and repair or mutate (fuzz) them based on the results.

## Test Information
- **PyTorch API**: {torch_api}
- **MindSpore API**: {mindspore_api}
{doc_section}
## Execution Results
- **Status**: {status}
- **PyTorch success**: {torch_success}
- **MindSpore success**: {mindspore_success}
- **Results match**: {results_match}

## Error Information
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
Based on the above information (including official API docs), decide whether the cross-framework result is **consistent**, **inconsistent**, or **execution error**, and then perform the following:

1. **If consistent**: **Mutate (fuzz)** the case, e.g., change input tensor shapes or parameter values (consider extreme/boundary values).
2. **If execution error**: **Repair** the case according to the error and docs (change parameter names, counts, types, or ranges; frameworks may differ), or **skip** if you believe the two cross-framework operators are not equivalent.
3. **If inconsistent**: Decide whether the difference is tolerable numerical error (<= 1e-3): (1) if tolerable, **mutate**; (2) if docs suggest the operators are not equivalent, **skip**; (3) otherwise, it is a test case construction issue, so **repair** the case based on the docs.

## Output Format
Please strictly output JSON in the following format without any other text, comments, or markdown:

{{
    "operation": "mutation",
    "reason": "Detailed reason for the operation",
    "pytorch_test_case": {{
        "api": "{torch_api}",
{torch_param_example_str}
    }},
    "mindspore_test_case": {{
        "api": "{mindspore_api}",
{ms_param_example_str}
    }}
}}

**Important Notes**:
1. operation must be one of "mutation", "repair", or "skip"
2. Tensor params must use {{"shape": [...], "dtype": "..."}} format
3. Scalar params should be numeric values, e.g., "y": 0
4. When constructing both frameworks' cases, inputs must match (convert shapes if needed) and parameters must be semantically aligned
5. PyTorch and MindSpore cases may differ in parameter names (input vs x), values, or counts, as long as outputs should theoretically match
6. If no official doc exists, decide if the operator is missing or removed in current PyTorch/MindSpore; if so, set operation to "skip" without attempting repair
7. During mutation, explore extreme cases: empty tensors (shape contains 0), single-element tensors (shape=[1] or []), high-dim tensors, very large tensors, different dtypes (int/float/bool), boundary values, etc.
8. Read official API docs carefully to ensure parameter names, types, and ranges are correct
"""

        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in deep learning framework testing and understand PyTorch vs MindSpore API differences. Return strict JSON only."},
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

    def convert_llm_test_cases(self, pt_tc: Dict, ms_tc: Dict) -> Tuple[Dict, Dict]:
        """Convert LLM-returned test cases to executable format (shared tensor data)."""
        shared_tensors = {}
        all_keys = set(pt_tc.keys()) | set(ms_tc.keys())
        for key in all_keys:
            if key == "api":
                continue
            pt_val = pt_tc.get(key)
            ms_val = ms_tc.get(key)
            tensor_desc = None
            if isinstance(pt_val, dict) and "shape" in pt_val:
                tensor_desc = pt_val
            elif isinstance(ms_val, dict) and "shape" in ms_val:
                tensor_desc = ms_val
            if tensor_desc:
                shared_tensors[key] = self.generate_numpy_data(tensor_desc)

        converted_pt = {}
        for k, v in pt_tc.items():
            converted_pt[k] = shared_tensors.get(k, v)
        converted_ms = {}
        for k, v in ms_tc.items():
            converted_ms[k] = shared_tensors.get(k, v)
        return converted_pt, converted_ms

    def close(self):
        self.client.close()


# ==================== Main Comparison Logic ====================

class LLMvsMindConverterComparator:
    """
    Main comparator: orchestrates LLM and MindConverter (ONNX) comparison tests.

    For each operator and each case, run both methods and summarize results.
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
        self.mindconverter = MindConverterSimulator(print_lock=self.print_lock)

        # Result directory
        self.result_dir = os.path.join(ROOT_DIR, "pt_ms_test")
        os.makedirs(self.result_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.realtime_file_path = os.path.join(self.result_dir, f"llm_vs_mindconverter_realtime_{timestamp}.jsonl")
        self.realtime_file = open(self.realtime_file_path, "a", encoding="utf-8")

    def _safe_print(self, msg: str, end: str = "\n"):
        safe_print(msg, self.print_lock, end)

    def _append_realtime_record(self, record: Dict[str, Any]):
        """Write a single statistics record in real time (JSONL)."""
        with self.realtime_lock:
            self.realtime_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.realtime_file.flush()

    def run_comparison(self, operator_names: List[str],
                       num_cases: int = DEFAULT_NUM_CASES,
                       max_iterations: int = DEFAULT_MAX_ITERATIONS) -> Dict[str, Any]:
        """
        Run LLM vs MindConverter comparison on a batch of operators.

        Returns a summary result dict.
        """
        global_stats = {
            "total_operators": len(operator_names),
            "tested_operators": 0,
            "skipped_operators_llm": 0,
            "skipped_operators_no_ms": 0,
            "skipped_operators_problematic": 0,
            "skipped_operators_deprecated": 0,

            # LLM method stats
            "llm_total_cases": 0,
            "llm_ms_success": 0,

            # MindConverter method stats
            "mindconverter_total_cases": 0,
            "mindconverter_export_success": 0,
            "mindconverter_run_success": 0,
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

            # 1. Check API mapping
            torch_api, ms_api, mapping_method = self.llm_method.convert_api_name(op_name)
            if ms_api is None:
                self._safe_print(f"  ⏭️ No MindSpore implementation ({mapping_method}), skipped")
                global_stats["skipped_operators_no_ms"] += 1
                operator_details.append({
                    "operator": op_name, "status": "skipped_no_ms",
                    "mapping_method": mapping_method
                })
                continue

            self._safe_print(f"  PT: {torch_api} → MS: {ms_api}")

            # 2. Fetch document from MongoDB
            document = self.llm_method.collection.find_one({"api": op_name})
            if document is None:
                self._safe_print(f"  ❌ Not found in database")
                operator_details.append({"operator": op_name, "status": "not_found"})
                continue

            total_cases = self.llm_method.get_num_test_cases(document)
            actual_cases = min(num_cases, total_cases)

            # 3. Test LLM and MindConverter for each case
            op_result = self._test_operator(
                op_name, torch_api, ms_api, document,
                actual_cases, max_iterations
            )

            operator_details.append(op_result)

            # 4. Aggregate stats
            if op_result["status"] == "skipped_by_llm":
                global_stats["skipped_operators_llm"] += 1
            elif op_result["status"] == "skipped_deprecated":
                global_stats["skipped_operators_deprecated"] += 1
            else:
                global_stats["tested_operators"] += 1
                global_stats["llm_total_cases"] += op_result["llm_total"]
                global_stats["llm_ms_success"] += op_result["llm_ms_success"]
                global_stats["mindconverter_total_cases"] += op_result["mindconverter_total"]
                global_stats["mindconverter_export_success"] += op_result["mindconverter_export_success"]
                global_stats["mindconverter_run_success"] += op_result["mindconverter_run_success"]

        return {"global_stats": global_stats, "operator_details": operator_details}

    def _test_operator(self, op_name: str, torch_api: str, ms_api: str,
                       document: Dict, num_cases: int,
                       max_iterations: int) -> Dict[str, Any]:
        """Test LLM and MindConverter for multiple cases of one operator."""
        op_result = {
            "operator": op_name,
            "torch_api": torch_api,
            "ms_api": ms_api,
            "status": "completed",
            "num_cases": num_cases,
            "llm_total": 0,
            "llm_ms_success": 0,
            "mindconverter_total": 0,
            "mindconverter_export_success": 0,
            "mindconverter_run_success": 0,
            "case_details": []
        }

        is_class_api = self.llm_method.is_class_based_api(torch_api)

        # Prepare all test cases
        cases = []
        for case_idx in range(num_cases):
            tc = self.llm_method.prepare_shared_numpy_data(document, case_index=case_idx)
            tc["api"] = torch_api
            cases.append((case_idx + 1, tc))

        # Pre-fetch docs (once)
        self._safe_print(f"  📖 Fetching API docs...")
        torch_doc, ms_doc = self.llm_method.fetch_api_docs(torch_api, ms_api)

        # Process cases concurrently with a thread pool
        if self.num_workers <= 1:
            case_results = []
            for case_number, tc in cases:
                cr = self._process_single_case(
                    torch_api, ms_api, tc, case_number,
                    is_class_api, torch_doc, ms_doc, max_iterations
                )
                case_results.append(cr)
        else:
            case_results = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_map = {}
                for case_number, tc in cases:
                    future = executor.submit(
                        self._process_single_case,
                        torch_api, ms_api, tc, case_number,
                        is_class_api, torch_doc, ms_doc, max_iterations
                    )
                    future_map[future] = case_number
                for future in as_completed(future_map):
                    case_results.append(future.result())

        case_results.sort(key=lambda x: x["case_number"])

        # Check if skipped due to deprecation
        any_deprecated = any(cr.get("deprecated_skip", False) for cr in case_results)
        if any_deprecated:
            op_result["status"] = "skipped_deprecated"
            op_result["case_details"] = case_results
            self._safe_print("  ⏭️ Operator deprecated, skipped")
            return op_result

        # Check if all cases were skipped by LLM
        all_skipped = all(cr.get("llm_skipped", False) for cr in case_results)
        if all_skipped and len(case_results) > 0:
            op_result["status"] = "skipped_by_llm"
            op_result["case_details"] = case_results
            self._safe_print(f"  ⏭️ LLM chose to skip this operator")
            return op_result

        # Aggregate results for each case
        for cr in case_results:
            if not cr.get("llm_skipped", False):
                op_result["llm_total"] += cr.get("llm_total", 0)
                op_result["llm_ms_success"] += cr.get("llm_ms_success", 0)
                op_result["mindconverter_total"] += cr.get("mindconverter_total", 0)
                op_result["mindconverter_export_success"] += cr.get("mindconverter_export_success", 0)
                op_result["mindconverter_run_success"] += cr.get("mindconverter_run_success", 0)
        op_result["case_details"] = case_results

        return op_result

    def _process_single_case(self, torch_api: str, ms_api: str,
                             test_case: Dict[str, Any], case_number: int,
                             is_class_api: bool,
                             torch_doc: str, ms_doc: str,
                             max_iterations: int) -> Dict[str, Any]:
                """
                Process one case: run both LLM and MindConverter methods.

                LLM flow (1 iteration):
                    1. Run DB case in PT and MS
                    2. Call LLM for repair/mutation/skip
                    3. If LLM returns mutation/repair, run the generated case

                MindConverter flow:
                    1. Wrap the same DB case as a PyTorch Module
                    2. torch.onnx.export + onnxruntime inference (simulate MindConverter ONNX core path)
                """
        case_result = {
            "case_number": case_number,
            "llm_skipped": False,
            "deprecated_skip": False,
            "llm_total": 0,
            "llm_ms_success": 0,
            "mindconverter_total": 0,
            "mindconverter_export_success": 0,
            "mindconverter_run_success": 0,
            "llm_detail": None,
            "mindconverter_detail": None,
        }

        # ========== MindConverter method ==========
        self._safe_print(f"  [Case {case_number}] MindConverter method...", end="")
        with self.execution_lock:
            mc_result = self.mindconverter.convert_and_run(
                torch_api, test_case, is_class_api
            )
        case_result["mindconverter_total"] = 1
        if mc_result["onnx_export_success"]:
            case_result["mindconverter_export_success"] = 1
        if mc_result["onnx_run_success"]:
            case_result["mindconverter_run_success"] = 1

        mc_status = "✓" if mc_result["onnx_run_success"] else "✗"
        mc_err = f" ({mc_result['error'][:60]})" if mc_result.get("error") else ""
        self._safe_print(f" MindConverter:{mc_status}{mc_err}")
        case_result["mindconverter_detail"] = mc_result

        # ========== LLM method ==========
        # Step 1: run the DB case in both frameworks
        torch_test_case = test_case
        ms_test_case = copy.deepcopy(test_case)
        ms_test_case["api"] = ms_api

        self._safe_print(f"  [Case {case_number}] LLM method: initial run...", end="")
        try:
            with self.execution_lock:
                exec_result = self.llm_method._execute_sequential(
                    torch_api, ms_api, torch_test_case, ms_test_case
                )
            pt_s = "✓" if exec_result['torch_success'] else "✗"
            ms_s = "✓" if exec_result['mindspore_success'] else "✗"
            self._safe_print(f" PT:{pt_s} MS:{ms_s}")
        except Exception as e:
            self._safe_print(f" ❌ Execution failed: {str(e)[:60]}")
            exec_result = {
                "torch_api": torch_api, "mindspore_api": ms_api,
                "torch_success": False, "mindspore_success": False,
                "results_match": False, "status": "fatal_error",
                "torch_error": str(e), "mindspore_error": None,
                "comparison_error": None
            }

        # If PyTorch error contains deprecated/removed, skip this operator
        torch_error = str(exec_result.get("torch_error", ""))
        if not exec_result.get("torch_success", False) and torch_error:
            if re.search(r"deprecated|removed", torch_error, re.IGNORECASE):
                case_result["deprecated_skip"] = True
                case_result["llm_skipped"] = False
                case_result["llm_detail"] = {
                    "initial_exec": exec_result,
                    "llm_operation": "skip",
                    "llm_reason": "PyTorch operator has been deprecated"
                }
                self._safe_print("  Deprecation detected, skipping")
                self._append_realtime_record({
                    "timestamp": datetime.now().isoformat(),
                    "operator": torch_api,
                    "case_number": case_number,
                    "status": "skipped_deprecated",
                    "mindconverter_run_success": bool(case_result.get("mindconverter_run_success", 0)),
                    "llm_ms_success": False,
                    "llm_skipped": False,
                    "deprecated_skip": True
                })
                return case_result

        # Step 2: call LLM
        self._safe_print(f"  [Case {case_number}] Calling LLM...", end="")
        llm_result = self.llm_method.call_llm(
            exec_result, torch_test_case, ms_test_case, torch_doc, ms_doc
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
                "mindconverter_run_success": bool(case_result.get("mindconverter_run_success", 0)),
                "llm_ms_success": False,
                "llm_skipped": True,
                "deprecated_skip": False
            })
            return case_result

        # Step 3: if LLM returns mutation/repair, run generated case
        llm_pt_tc = llm_result.get("pytorch_test_case", torch_test_case)
        llm_ms_tc = llm_result.get("mindspore_test_case", ms_test_case)

        # Convert to executable format
        try:
            llm_pt_tc, llm_ms_tc = self.llm_method.convert_llm_test_cases(llm_pt_tc, llm_ms_tc)
        except Exception as e:
            case_result["llm_skipped"] = True
            case_result["llm_detail"] = {
                "initial_exec": exec_result,
                "llm_operation": "skip",
                "llm_reason": f"LLM returned invalid shape or tensor generation failed: {e}"
            }
            self._safe_print(f"  [Case {case_number}] LLM case conversion failed, skipped: {str(e)[:60]}")
            self._append_realtime_record({
                "timestamp": datetime.now().isoformat(),
                "operator": torch_api,
                "case_number": case_number,
                "status": "llm_case_convert_failed",
                "mindconverter_run_success": bool(case_result.get("mindconverter_run_success", 0)),
                "llm_ms_success": False,
                "llm_skipped": True,
                "deprecated_skip": False
            })
            return case_result

        case_result["llm_total"] = 1

        self._safe_print(f"  [Case {case_number}] Running LLM case...", end="")
        try:
            with self.execution_lock:
                llm_exec = self.llm_method._execute_sequential(
                    torch_api, ms_api, llm_pt_tc, llm_ms_tc
                )
            pt_s = "✓" if llm_exec['torch_success'] else "✗"
            ms_s = "✓" if llm_exec['mindspore_success'] else "✗"
            match_s = "✓" if llm_exec['results_match'] else "✗"
            self._safe_print(f" PT:{pt_s} MS:{ms_s} Match:{match_s}")

            if llm_exec["mindspore_success"]:
                case_result["llm_ms_success"] = 1

        except Exception as e:
            self._safe_print(f" ❌ {str(e)[:60]}")
            llm_exec = {
                "torch_success": False, "mindspore_success": False,
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
            "mindconverter_run_success": bool(case_result.get("mindconverter_run_success", 0)),
            "llm_ms_success": bool(case_result.get("llm_ms_success", 0)),
            "llm_skipped": False,
            "deprecated_skip": False
        })

        return case_result

    def print_and_save_results(self, comparison_result: Dict[str, Any]):
        """Print results and save to file."""
        gs = comparison_result["global_stats"]

        print("\n" + "=" * 80)
        print("📊 LLM vs MindConverter — Test Case Conversion Success Rate")
        print("=" * 80)

        print(f"\n📌 Total operators: {gs['total_operators']}")
        print(f"   - Skipped (no MS mapping): {gs['skipped_operators_no_ms']}")
        print(f"   - Skipped (problematic): {gs['skipped_operators_problematic']}")
        print(f"   - Skipped by LLM: {gs['skipped_operators_llm']}")
        print(f"   - Skipped (deprecated): {gs['skipped_operators_deprecated']}")
        print(f"   - Participated in comparison: {gs['tested_operators']}")

        print(f"\n{'─'*40}")
        print(f"🤖 LLM method (excluding skipped operators):")
        print(f"   Total MS test cases generated by LLM: {gs['llm_total_cases']}")
        print(f"   MS execution successes: {gs['llm_ms_success']}")
        if gs['llm_total_cases'] > 0:
            llm_rate = gs['llm_ms_success'] / gs['llm_total_cases'] * 100
            print(f"   ✅ MS execution success rate: {llm_rate:.2f}%")
        else:
            llm_rate = 0
            print(f"   ✅ MS execution success rate: N/A (no LLM-generated cases)")

        print(f"\n{'─'*40}")
        print(f"🔄 MindConverter method (excluding LLM-skipped operators):")
        print(f"   Total MindConverter conversion attempts: {gs['mindconverter_total_cases']}")
        print(f"   ONNX export successes: {gs['mindconverter_export_success']}")
        print(f"   ONNX inference successes (= conversion success): {gs['mindconverter_run_success']}")
        if gs['mindconverter_total_cases'] > 0:
            mc_rate = gs['mindconverter_run_success'] / gs['mindconverter_total_cases'] * 100
            print(f"   ✅ MindConverter conversion success rate: {mc_rate:.2f}%")
        else:
            mc_rate = 0
            print(f"   ✅ MindConverter conversion success rate: N/A")

        print(f"\n{'─'*40}")
        print(f"📈 Comparison conclusion:")
        if gs['llm_total_cases'] > 0 and gs['mindconverter_total_cases'] > 0:
            diff = llm_rate - mc_rate
            if diff > 0:
                print(f"   LLM outperforms MindConverter by {diff:.2f} percentage points")
            elif diff < 0:
                print(f"   MindConverter outperforms LLM by {-diff:.2f} percentage points")
            else:
                print(f"   Success rates are tied")
        else:
            print(f"   Insufficient data to compare")
        print("=" * 80)

        # -------- Save to file --------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.result_dir, f"llm_vs_mindconverter_result_{timestamp}.json")

        save_data = self._make_serializable(comparison_result)
        save_data["summary"] = {
            "llm_ms_success_rate": f"{llm_rate:.2f}%" if gs['llm_total_cases'] > 0 else "N/A",
            "mindconverter_success_rate": f"{mc_rate:.2f}%" if gs['mindconverter_total_cases'] > 0 else "N/A",
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
            return {k: LLMvsMindConverterComparator._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [LLMvsMindConverterComparator._make_serializable(item) for item in obj]
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


# ==================== Main Function ====================

def main():
    parser = argparse.ArgumentParser(
        description="LLM vs MindConverter: PyTorch→MindSpore test case conversion success rate"
    )
    parser.add_argument("--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES,
                        help=f"Number of cases per operator (default {DEFAULT_NUM_CASES})")
    parser.add_argument("--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS,
                        help=f"LLM iterations per case (default {DEFAULT_MAX_ITERATIONS})")
    parser.add_argument("--start", type=int, default=1,
                        help="Start operator index (1-based)")
    parser.add_argument("--end", type=int, default=None,
                        help="End operator index (inclusive)")
    parser.add_argument("--operators", "-o", nargs="*",
                        help="List of operator names")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS,
                        help=f"LLM concurrent workers (default {DEFAULT_WORKERS})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"LLM model (default {DEFAULT_MODEL})")
    parser.add_argument("--key-path", "-k", default=DEFAULT_KEY_PATH,
                        help=f"API key path (default {DEFAULT_KEY_PATH})")
    args = parser.parse_args()

    print("=" * 80)
    print("LLM vs MindConverter — PyTorch→MindSpore Test Case Conversion Success Rate")
    print("=" * 80)
    print(f"📌 Cases per operator: {args.num_cases}")
    print(f"📌 LLM iterations: {args.max_iterations}")
    print(f"📌 Worker threads: {args.workers}")
    print(f"📌 LLM model: {args.model}")
    print("=" * 80)

    comparator = LLMvsMindConverterComparator(
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
            print(f"📌 Test range: operators {start_idx + 1} to {end_idx}")

        print(f"📋 Testing {len(operator_names)} operators")
        print(f"📋 First 10: {', '.join(operator_names[:10])}{'...' if len(operator_names) > 10 else ''}\n")

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
        print(f"\n⏱️ Total time: {h}h {m}m {s}s")

    finally:
        comparator.close()
        print("✅ Program completed")


if __name__ == "__main__":
    main()
