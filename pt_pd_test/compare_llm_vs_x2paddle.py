# ./pt_pd_test/compare_llm_vs_x2paddle.py
"""
LLM方法 vs X2Paddle方法：PyTorch → PaddlePaddle 测试用例转换成功率对比
====================================================================

对比两种跨框架测试用例迁移方案：
1. LLM方法：从MongoDB提取用例 → 构造PT/PD用例 → 执行 → LLM修复/变异 → 执行LLM生成的用例
2. X2Paddle方法：从MongoDB提取用例 → 包装为PyTorch小模型 → torch.onnx.export → onnxruntime推理
   （模拟X2Paddle的核心路径：PyTorch → ONNX → PaddlePaddle）

X2Paddle 简介：
  - 飞桨官方模型转换工具，支持 Caffe/TensorFlow/ONNX/PyTorch → PaddlePaddle
  - 核心转换路径依赖 ONNX 中间格式
  - 支持130+ PyTorch OP、90+ ONNX OP
  - API: pytorch2paddle(module, save_dir, jit_type="trace", input_examples=[...])

统计指标：LLM生成的PD用例执行成功率 vs X2Paddle(ONNX)导出+推理成功率
（剔除LLM选择跳过的算子）
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

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from component.doc.doc_crawler_factory import get_doc_content

# ==================== 常量定义 ====================
DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"
DEFAULT_MAX_ITERATIONS = 1   # 简化比较：只迭代1次
DEFAULT_NUM_CASES = 3
DEFAULT_WORKERS = 4


# ==================== 工具函数 ====================

def safe_print(msg: str, print_lock: Lock = None, end: str = "\n"):
    """线程安全的打印"""
    if print_lock:
        with print_lock:
            print(msg, end=end, flush=True)
    else:
        print(msg, end=end, flush=True)


# ==================== X2Paddle(ONNX) 转换器 ====================

class X2PaddleConverter:
    """
    通过 ONNX 中间格式模拟 X2Paddle 的 PyTorch → PaddlePaddle 测试用例迁移
    
    X2Paddle 核心转换路径：
    1. PyTorch 模型 → torch.onnx.export → ONNX 模型
    2. ONNX 模型 → X2Paddle 内部映射 → PaddlePaddle 模型
    
    本转换器的模拟流程（算子级）：
    1. 将 PyTorch 算子包装为一个最小化的 nn.Module
    2. 使用 torch.onnx.export 导出为 ONNX 模型
    3. 使用 onnxruntime 推理获得输出
    
    成功标准：ONNX 导出 + OnnxRuntime 推理成功即视为"X2Paddle方法转换成功"
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
        将 PyTorch 算子包装为 nn.Module，以便 torch.onnx.export 使用
        
        Args:
            torch_api: PyTorch API 名称
            is_class_api: 是否为类形式的 API
            init_kwargs: 类算子的初始化参数
            extra_kwargs: 函数算子的非张量关键字参数（通过闭包捕获）
        
        Returns:
            包装后的 nn.Module，失败返回 None
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
        """根据 ['torch', 'nn', 'ReLU'] 这样的路径获取属性"""
        try:
            obj = torch
            for p in parts[1:]:  # 跳过 'torch'
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            return None

    def convert_and_run(self, torch_api: str, test_case: Dict[str, Any],
                        is_class_api: bool) -> Dict[str, Any]:
        """
        对单个测试用例执行 X2Paddle(ONNX) 转换 + 推理
        
        流程：
        1. 准备 PyTorch 输入张量
        2. 用 PyTorch 前向传播，获得参考结果
        3. torch.onnx.export 导出（模拟X2Paddle的ONNX导出阶段）
        4. onnxruntime 推理（模拟X2Paddle转换后的PaddlePaddle推理）
        5. 比较两者结果
        
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

        # 使用执行锁保证 torch.onnx.export 的线程安全
        with self.execution_lock:
            # ---------- 1. 准备输入 ----------
            try:
                input_tensors, init_kwargs, input_names, extra_kwargs = self._prepare_inputs(test_case, is_class_api)
            except Exception as e:
                result["error"] = f"输入准备失败: {e}"
                return result

            if not input_tensors:
                result["error"] = "无有效输入张量"
                return result

            # ---------- 2. 包装模型 ----------
            module = self._wrap_as_module(torch_api, is_class_api, init_kwargs, extra_kwargs)
            if module is None:
                result["error"] = f"无法包装算子 {torch_api} 为 nn.Module"
                return result

            module.eval()

            # ---------- 3. PyTorch 前向传播 ----------
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
                result["error"] = f"PyTorch前向传播失败: {e}"
                return result

            # ---------- 4. ONNX 导出（模拟X2Paddle的ONNX导出阶段） ----------
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
                result["error"] = f"ONNX导出失败(X2Paddle核心路径): {e}"
                if onnx_path and os.path.exists(onnx_path):
                    os.remove(onnx_path)
                return result

            # ---------- 5. OnnxRuntime 推理（模拟X2Paddle转换后的推理） ----------
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
                result["error"] = f"OnnxRuntime推理失败: {e}"
            finally:
                if onnx_path and os.path.exists(onnx_path):
                    os.remove(onnx_path)

        return result

    def _prepare_inputs(self, test_case: Dict[str, Any],
                        is_class_api: bool) -> Tuple[List[torch.Tensor], Dict[str, Any], List[str], Dict[str, Any]]:
        """
        从测试用例中提取输入张量和初始化参数
        
        Returns:
            (input_tensors列表, init_kwargs字典, input_names列表, extra_kwargs字典)
        """
        input_tensors: List[torch.Tensor] = []
        init_kwargs: Dict[str, Any] = {}
        input_names: List[str] = []
        extra_kwargs: Dict[str, Any] = {}

        positional_tensor_params = ["condition", "input", "x", "y", "other"]

        # 可变参数处理
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

        # 按顺序收集位置参数作为输入张量
        for param_name in positional_tensor_params:
            if param_name in test_case:
                value = test_case[param_name]
                tensor = self._to_torch_tensor(value)
                if tensor is not None:
                    input_tensors.append(tensor)
                    input_names.append(param_name)

        # 其他参数
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
        """将各种格式的值转换为 PyTorch 张量"""
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
    """将普通函数包装为 nn.Module，用于 torch.onnx.export"""

    def __init__(self, func):
        super().__init__()
        self._func = func

    def forward(self, *args):
        return self._func(*args)


# ==================== LLM 方法（简化版） ====================

class LLMMethod:
    """
    基于 LLM 的测试用例转换方法（简化版，迭代1次）
    
    流程：从MongoDB提取用例 → 在PT和PD中执行 → LLM判断修复/跳过 → 执行LLM生成的用例
    复用了 llm_enhanced_compare.py 的核心逻辑，但精简了输出。
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

        # LLM 客户端
        api_key = self._load_api_key(key_path)
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # API 映射
        self.api_mapping = self._load_api_mapping()

        # 随机种子
        np.random.seed(42)
        torch.manual_seed(42)
        paddle.seed(42)

        # 会导致程序卡住或崩溃的算子列表（跳过这些算子的测试）
        self.problematic_apis = {
            "torch.nn.Embedding": "会导致程序卡住",
            "torch.nn.functional.embedding": "会导致程序卡住",
            "torch.nn.functional.max_unpool1d": "会导致程序卡住",
            "torch.nn.functional.max_unpool2d": "会导致程序卡住",
            "torch.nn.functional.max_unpool3d": "会导致程序卡住",
            "torch.nn.MaxUnpool1d": "会导致程序卡住",
            "torch.nn.MaxUnpool2d": "会导致程序卡住",
            "torch.nn.MaxUnpool3d": "会导致程序卡住",
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
            self._safe_print(f"❌ 加载映射表失败: {e}")
            return {}

    def convert_api_name(self, torch_api: str) -> Tuple[Optional[str], Optional[str], str]:
        if torch_api in self.api_mapping:
            pd_api = self.api_mapping[torch_api]["pd_api"]
            if pd_api in ("无对应实现", "NONE", ""):
                return torch_api, None, "无对应实现"
            return torch_api, pd_api, "映射表"
        return torch_api, None, "映射表中未找到"

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

    # -------- 数据生成 --------

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

    # -------- 参数准备 --------

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

    # -------- 执行测试用例 --------

    def execute_test_case(self, torch_api: str, paddle_api: str,
                          torch_test_case: Dict[str, Any],
                          paddle_test_case: Dict[str, Any]) -> Dict[str, Any]:
        """在PyTorch和PaddlePaddle中分别执行测试用例"""
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
                result["torch_error"] = f"算子 {torch_api} 未找到"
            else:
                args, kwargs = self.prepare_arguments_torch(torch_test_case)
                if is_class_api:
                    init_kwargs = {k: v for k, v in kwargs.items() if k != 'input'}
                    instance = torch_func(**init_kwargs)
                    input_data = kwargs.get('input', args[0] if args else None)
                    if input_data is None:
                        raise ValueError("类算子缺少input参数")
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
                result["paddle_error"] = f"算子 {paddle_api} 未找到"
            else:
                args, kwargs = self.prepare_arguments_paddle(paddle_test_case, paddle_api)
                if is_class_api:
                    init_kwargs = {k: v for k, v in kwargs.items() if k not in ['x', 'input']}
                    instance = paddle_func(**init_kwargs)
                    input_data = kwargs.get('x', kwargs.get('input', args[0] if args else None))
                    if input_data is None:
                        raise ValueError("类算子缺少input/x参数")
                    paddle_result = instance(input_data)
                else:
                    paddle_result = paddle_func(*args, **kwargs)
                result["paddle_success"] = True
        except Exception as e:
            result["paddle_error"] = str(e)

        # -- 比较 --
        if result["torch_success"] and result["paddle_success"]:
            try:
                torch_np = torch_result.detach().cpu().numpy() if hasattr(torch_result, 'detach') else np.array(torch_result)
                pd_np = paddle_result.numpy() if hasattr(paddle_result, 'numpy') else np.array(paddle_result)
                if torch_np.shape != pd_np.shape:
                    result["comparison_error"] = f"形状不匹配: {torch_np.shape} vs {pd_np.shape}"
                elif np.allclose(torch_np, pd_np, atol=1e-5, rtol=1e-5, equal_nan=True):
                    result["results_match"] = True
                else:
                    max_diff = np.max(np.abs(torch_np.astype(float) - pd_np.astype(float)))
                    result["comparison_error"] = f"最大差异: {max_diff}"
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

    # -------- 文档爬取 --------

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

    # -------- LLM 调用 --------

    def call_llm(self, execution_result: Dict, torch_tc: Dict, pd_tc: Dict,
                 torch_doc: str = "", pd_doc: str = "") -> Dict[str, Any]:
        """调用LLM进行修复/变异/跳过判断"""
        torch_api = execution_result.get("torch_api", "")
        paddle_api = execution_result.get("paddle_api", "")
        status = execution_result.get("status", "")
        torch_success = execution_result.get("torch_success", False)
        paddle_success = execution_result.get("paddle_success", False)
        results_match = execution_result.get("results_match", False)
        torch_error = execution_result.get("torch_error", "")
        paddle_error = execution_result.get("paddle_error", "")
        comparison_error = execution_result.get("comparison_error", "")

        # 简化PyTorch测试用例
        simplified_torch_test_case = {}
        for key, value in torch_tc.items():
            if isinstance(value, np.ndarray):
                simplified_torch_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_torch_test_case[key] = value

        # 简化PaddlePaddle测试用例
        simplified_paddle_test_case = {}
        for key, value in pd_tc.items():
            if isinstance(value, np.ndarray):
                simplified_paddle_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_paddle_test_case[key] = value

        # 构建PyTorch参数示例
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

        # 构建PaddlePaddle参数示例
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

        # 构建API文档部分
        doc_section = ""
        if torch_doc or pd_doc:
            doc_section = "\n## 官方API文档参考\n\n"
            if torch_doc:
                doc_section += f"### PyTorch {torch_api} 文档\n```\n{torch_doc}\n```\n\n"
            if pd_doc:
                doc_section += f"### PaddlePaddle {paddle_api} 文档\n```\n{pd_doc}\n```\n\n"

        prompt = f"""请分析以下算子测试用例在PyTorch和PaddlePaddle框架中的执行结果，并根据结果进行测试用例的修复或变异（fuzzing）。

## 测试信息
- **PyTorch API**: {torch_api}
- **PaddlePaddle API**: {paddle_api}
{doc_section}
## 执行结果
- **执行状态**: {status}
- **PyTorch执行成功**: {torch_success}
- **PaddlePaddle执行成功**: {paddle_success}
- **结果是否一致**: {results_match}

## 错误信息
- **PyTorch错误**: {torch_error if torch_error else "无"}
- **PaddlePaddle错误**: {paddle_error if paddle_error else "无"}
- **比较错误**: {comparison_error if comparison_error else "无"}

## 原始测试用例

### PyTorch测试用例
```json
{json.dumps(simplified_torch_test_case, indent=2, ensure_ascii=False)}
```

### PaddlePaddle测试用例
```json
{json.dumps(simplified_paddle_test_case, indent=2, ensure_ascii=False)}
```

## 任务要求
请根据以上信息（包括官方API文档），自主判断两框架的比较结果是**一致**、**不一致**还是**执行出错**，并执行以下操作：

1. **如果一致**：对用例进行**变异（fuzzing）**，例如修改输入张量的形状、修改参数值等（可以考虑一些极端值或边界值）
2. **如果执行出错**：根据报错原因和官方文档对用例进行**修复**（改变参数名称、数量、类型、取值范围等，不同框架可能不完全一样）或者**跳过**（当你认为这两个跨框架算子的功能不完全等价时）
3. **如果不一致**：判断是否为可容忍的精度误差（1e-3及以下）：（1）如果是可容忍精度误差则**变异**；（2）结合算子文档分析后，认为这两个跨框架算子的功能不完全等价时选择**跳过**；（3）如果既不是可容忍精度误差，两个算子功能也等价，那就是测试用例构造问题，请根据算子文档文档对用例进行**修复**。

## 输出格式要求
请严格按照以下JSON格式输出，不要包含任何其他文字、注释或markdown标记：

{{
  "operation": "mutation",
  "reason": "进行该操作的详细原因",
  "pytorch_test_case": {{
    "api": "{torch_api}",
{torch_param_example_str}
  }},
  "paddle_test_case": {{
    "api": "{paddle_api}",
{pd_param_example_str}
  }}
}}

**重要说明**：
1. operation的值必须是 "mutation"、"repair" 或 "skip" 之一
2. 张量参数必须使用 {{"shape": [...], "dtype": "..."}} 格式
3. 标量参数直接使用数值，例如 "y": 0
4. 构造两个框架的用例时必须保证输入相同（必要时进行张量形状的转换）、参数在语义上严格对应
5. PyTorch和PaddlePaddle的测试用例可以有参数名差异（如input vs x）、参数值差异或者参数数量的差异，只要保证理论上输出相同就行
6. 如果这个算子找不到官方文档，请判断是否是因为该算子不存在或者已经从PyTorch或者PaddlePaddle的当前版本移除了，如果是这样，请将 operation 设置为 "skip"，不需要尝试修复
7. 测试用例变异时可适当探索一些极端情况，例如：空张量（shape包含0）、单元素张量（shape=[1]或[]）、高维张量、超大张量、不同数据类型（int、float、bool）、边界值等
8. 请仔细阅读官方API文档，确保参数名称、参数类型、参数取值范围等与文档一致
"""

        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是深度学习框架测试专家，精通PyTorch和PaddlePaddle的API差异。返回严格JSON格式结果。"},
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
                return {"operation": "skip", "reason": "LLM返回格式错误"}
        except Exception as e:
            return {"operation": "skip", "reason": f"LLM调用失败: {e}"}

    # -------- 转换LLM测试用例 --------

    def convert_llm_test_cases(self, pt_tc: Dict, pd_tc: Dict) -> Tuple[Dict, Dict]:
        """将LLM返回的测试用例转换为可执行格式（共享张量数据）"""
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


# ==================== 主比较逻辑 ====================

class LLMvsX2PaddleComparator:
    """
    主比较类：统一管理 LLM 方法和 X2Paddle(ONNX) 方法的对比测试
    
    对每个算子的每个用例同时跑两种方法，最后汇总统计。
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

        # 结果目录
        self.result_dir = os.path.join(ROOT_DIR, "pt_pd_test")
        os.makedirs(self.result_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.realtime_file_path = os.path.join(self.result_dir, f"llm_vs_x2paddle_realtime_{timestamp}.jsonl")
        self.realtime_file = open(self.realtime_file_path, "a", encoding="utf-8")

    def _safe_print(self, msg: str, end: str = "\n"):
        safe_print(msg, self.print_lock, end)

    def _append_realtime_record(self, record: Dict[str, Any]):
        """实时写入单条统计记录（JSONL）"""
        with self.realtime_lock:
            self.realtime_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.realtime_file.flush()

    def run_comparison(self, operator_names: List[str],
                       num_cases: int = DEFAULT_NUM_CASES,
                       max_iterations: int = DEFAULT_MAX_ITERATIONS) -> Dict[str, Any]:
        """
        对一批算子运行 LLM vs X2Paddle 对比测试
        
        返回汇总结果字典
        """
        global_stats = {
            "total_operators": len(operator_names),
            "tested_operators": 0,
            "skipped_operators_llm": 0,
            "skipped_operators_no_pd": 0,
            "skipped_operators_problematic": 0,
            "skipped_operators_deprecated": 0,

            # LLM 方法统计
            "llm_total_cases": 0,
            "llm_pd_success": 0,

            # X2Paddle 方法统计
            "x2paddle_total_cases": 0,
            "x2paddle_export_success": 0,
            "x2paddle_run_success": 0,
        }

        operator_details = []

        for idx, op_name in enumerate(operator_names, 1):
            self._safe_print(f"\n{'='*70}")
            self._safe_print(f"[{idx}/{len(operator_names)}] 算子: {op_name}")
            self._safe_print(f"{'='*70}")

            # 0. 跳过会卡住的算子
            if op_name in self.llm_method.problematic_apis:
                reason = self.llm_method.problematic_apis.get(op_name, "会导致程序卡住")
                self._safe_print(f"  ⏭️ 跳过（{reason}）")
                global_stats["skipped_operators_problematic"] += 1
                operator_details.append({
                    "operator": op_name,
                    "status": "skipped_problematic",
                    "reason": reason
                })
                continue

            # 1. 查 API 映射
            torch_api, pd_api, mapping_method = self.llm_method.convert_api_name(op_name)
            if pd_api is None:
                self._safe_print(f"  ⏭️ 无PD对应实现（{mapping_method}），跳过")
                global_stats["skipped_operators_no_pd"] += 1
                operator_details.append({
                    "operator": op_name, "status": "skipped_no_pd",
                    "mapping_method": mapping_method
                })
                continue

            self._safe_print(f"  PT: {torch_api} → PD: {pd_api}")

            # 2. 从 MongoDB 获取文档
            document = self.llm_method.collection.find_one({"api": op_name})
            if document is None:
                self._safe_print(f"  ❌ 数据库中未找到")
                operator_details.append({"operator": op_name, "status": "not_found"})
                continue

            total_cases = self.llm_method.get_num_test_cases(document)
            actual_cases = min(num_cases, total_cases)

            # 3. 为每个用例同时测试 LLM 和 X2Paddle
            op_result = self._test_operator(
                op_name, torch_api, pd_api, document,
                actual_cases, max_iterations
            )

            operator_details.append(op_result)

            # 4. 汇总统计
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
        """对单个算子的多个用例同时测试 LLM 和 X2Paddle 方法"""
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

        # 准备所有测试用例
        cases = []
        for case_idx in range(num_cases):
            tc = self.llm_method.prepare_shared_numpy_data(document, case_index=case_idx)
            tc["api"] = torch_api
            cases.append((case_idx + 1, tc))

        # 预先爬取文档（只爬一次）
        self._safe_print(f"  📖 爬取API文档...")
        torch_doc, pd_doc = self.llm_method.fetch_api_docs(torch_api, pd_api)

        # 用线程池并发处理各用例
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

        # 检查是否因为版本淘汰而跳过
        any_deprecated = any(cr.get("deprecated_skip", False) for cr in case_results)
        if any_deprecated:
            op_result["status"] = "skipped_deprecated"
            op_result["case_details"] = case_results
            self._safe_print("  ⏭️ 该算子已被版本淘汰，跳过")
            return op_result

        # 检查是否所有用例都被 LLM skip
        all_skipped = all(cr.get("llm_skipped", False) for cr in case_results)
        if all_skipped and len(case_results) > 0:
            op_result["status"] = "skipped_by_llm"
            op_result["case_details"] = case_results
            self._safe_print(f"  ⏭️ LLM选择跳过该算子")
            return op_result

        # 汇总每个用例的结果
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
        处理单个用例：同时进行 LLM 方法和 X2Paddle 方法
        
        LLM 方法流程（迭代1次）：
          1. 用DB用例在PT和PD中执行
          2. 调用LLM获取修复/变异/跳过
          3. 如果LLM返回mutation/repair，执行LLM生成的用例
        
        X2Paddle 方法流程：
          1. 用同一个DB用例，包装成PyTorch Module
          2. torch.onnx.export + onnxruntime推理（模拟X2Paddle的ONNX核心路径）
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

        # ========== X2Paddle 方法 ==========
        self._safe_print(f"  [用例{case_number}] X2Paddle方法...", end="")
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

        # ========== LLM 方法 ==========
        # 步骤1：用DB原始用例在两个框架中执行
        torch_test_case = test_case
        pd_test_case = copy.deepcopy(test_case)
        pd_test_case["api"] = pd_api

        self._safe_print(f"  [用例{case_number}] LLM方法: 初始执行...", end="")
        try:
            with self.execution_lock:
                exec_result = self.llm_method._execute_sequential(
                    torch_api, pd_api, torch_test_case, pd_test_case
                )
            pt_s = "✓" if exec_result['torch_success'] else "✗"
            pd_s = "✓" if exec_result['paddle_success'] else "✗"
            self._safe_print(f" PT:{pt_s} PD:{pd_s}")
        except Exception as e:
            self._safe_print(f" ❌ 执行失败: {str(e)[:60]}")
            exec_result = {
                "torch_api": torch_api, "paddle_api": pd_api,
                "torch_success": False, "paddle_success": False,
                "results_match": False, "status": "fatal_error",
                "torch_error": str(e), "paddle_error": None,
                "comparison_error": None
            }

        # 如果PyTorch执行报错包含deprecated/removed，直接跳过该算子
        torch_error = str(exec_result.get("torch_error", ""))
        if not exec_result.get("torch_success", False) and torch_error:
            if re.search(r"deprecated|removed", torch_error, re.IGNORECASE):
                case_result["deprecated_skip"] = True
                case_result["llm_skipped"] = False
                case_result["llm_detail"] = {
                    "initial_exec": exec_result,
                    "llm_operation": "skip",
                    "llm_reason": "PyTorch算子已被版本淘汰"
                }
                self._safe_print("  版本淘汰检测到，跳过")
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

        # 步骤2：调用LLM
        self._safe_print(f"  [用例{case_number}] 调用LLM...", end="")
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

        # 步骤3：如果LLM返回了mutation/repair，执行LLM生成的用例
        llm_pt_tc = llm_result.get("pytorch_test_case", torch_test_case)
        llm_pd_tc = llm_result.get("paddle_test_case", pd_test_case)

        # 转换为可执行格式
        try:
            llm_pt_tc, llm_pd_tc = self.llm_method.convert_llm_test_cases(llm_pt_tc, llm_pd_tc)
        except Exception as e:
            case_result["llm_skipped"] = True
            case_result["llm_detail"] = {
                "initial_exec": exec_result,
                "llm_operation": "skip",
                "llm_reason": f"LLM返回的shape非法或无法生成张量: {e}"
            }
            self._safe_print(f"  [用例{case_number}] LLM用例转换失败，跳过: {str(e)[:60]}")
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

        self._safe_print(f"  [用例{case_number}] 执行LLM用例...", end="")
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
        """打印结果并保存到文件"""
        gs = comparison_result["global_stats"]

        print("\n" + "=" * 80)
        print("📊 LLM方法 vs X2Paddle方法 — 测试用例转换成功率对比")
        print("=" * 80)

        print(f"\n📌 算子总数: {gs['total_operators']}")
        print(f"   - 无PD映射跳过: {gs['skipped_operators_no_pd']}")
        print(f"   - 问题算子跳过: {gs['skipped_operators_problematic']}")
        print(f"   - LLM选择跳过: {gs['skipped_operators_llm']}")
        print(f"   - 已被版本淘汰跳过: {gs['skipped_operators_deprecated']}")
        print(f"   - 实际参与对比: {gs['tested_operators']}")

        print(f"\n{'─'*40}")
        print(f"🤖 LLM 方法（剔除跳过的算子）:")
        print(f"   LLM生成的PD测试用例总数: {gs['llm_total_cases']}")
        print(f"   PD执行成功数: {gs['llm_pd_success']}")
        if gs['llm_total_cases'] > 0:
            llm_rate = gs['llm_pd_success'] / gs['llm_total_cases'] * 100
            print(f"   ✅ PD执行成功率: {llm_rate:.2f}%")
        else:
            llm_rate = 0
            print(f"   ✅ PD执行成功率: N/A（无LLM生成的用例）")

        print(f"\n{'─'*40}")
        print(f"🔄 X2Paddle 方法（剔除LLM跳过的算子）:")
        print(f"   X2Paddle转换尝试总数: {gs['x2paddle_total_cases']}")
        print(f"   ONNX导出成功数: {gs['x2paddle_export_success']}")
        print(f"   ONNX推理成功数（=转换成功）: {gs['x2paddle_run_success']}")
        if gs['x2paddle_total_cases'] > 0:
            x2p_rate = gs['x2paddle_run_success'] / gs['x2paddle_total_cases'] * 100
            print(f"   ✅ X2Paddle转换成功率: {x2p_rate:.2f}%")
        else:
            x2p_rate = 0
            print(f"   ✅ X2Paddle转换成功率: N/A")

        print(f"\n{'─'*40}")
        print(f"📈 对比结论:")
        if gs['llm_total_cases'] > 0 and gs['x2paddle_total_cases'] > 0:
            diff = llm_rate - x2p_rate
            if diff > 0:
                print(f"   LLM方法 优于 X2Paddle方法 {diff:.2f} 个百分点")
            elif diff < 0:
                print(f"   X2Paddle方法 优于 LLM方法 {-diff:.2f} 个百分点")
            else:
                print(f"   两种方法成功率持平")
        else:
            print(f"   数据不足，无法比较")
        print("=" * 80)

        # -------- 保存到文件 --------
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

        print(f"\n💾 详细结果已保存到: {result_file}")

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """递归将对象转为可JSON序列化的格式"""
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


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="LLM方法 vs X2Paddle方法：PyTorch→PaddlePaddle 测试用例转换成功率对比"
    )
    parser.add_argument("--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES,
                        help=f"每个算子测试的用例数（默认{DEFAULT_NUM_CASES}）")
    parser.add_argument("--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS,
                        help=f"LLM方法每个用例的迭代次数（默认{DEFAULT_MAX_ITERATIONS}）")
    parser.add_argument("--start", type=int, default=1,
                        help="起始算子索引（从1开始）")
    parser.add_argument("--end", type=int, default=None,
                        help="结束算子索引（包含）")
    parser.add_argument("--operators", "-o", nargs="*",
                        help="指定算子名称列表")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS,
                        help=f"LLM并发线程数（默认{DEFAULT_WORKERS}）")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"LLM模型（默认{DEFAULT_MODEL}）")
    parser.add_argument("--key-path", "-k", default=DEFAULT_KEY_PATH,
                        help=f"API key路径（默认{DEFAULT_KEY_PATH}）")
    args = parser.parse_args()

    print("=" * 80)
    print("LLM方法 vs X2Paddle方法 — PyTorch→PaddlePaddle 测试用例转换成功率对比")
    print("=" * 80)
    print(f"📌 每个算子用例数: {args.num_cases}")
    print(f"📌 LLM迭代次数: {args.max_iterations}")
    print(f"📌 并发线程数: {args.workers}")
    print(f"📌 LLM模型: {args.model}")
    print("=" * 80)

    comparator = LLMvsX2PaddleComparator(
        key_path=args.key_path, model=args.model,
        num_workers=args.workers
    )

    start_time = time.time()

    try:
        # 获取算子列表
        all_docs = list(comparator.llm_method.collection.find({}, {"api": 1}))
        all_ops = [doc["api"] for doc in all_docs if "api" in doc]
        print(f"\n📋 数据库中共 {len(all_ops)} 个算子")

        if args.operators:
            operator_names = args.operators
        else:
            start_idx = max(1, args.start) - 1
            end_idx = args.end if args.end is not None else len(all_ops)
            end_idx = min(end_idx, len(all_ops))
            operator_names = all_ops[start_idx:end_idx]
            print(f"📌 测试范围: 第 {start_idx + 1} ~ {end_idx} 个算子")

        print(f"📋 将测试 {len(operator_names)} 个算子")
        print(f"📋 前10个: {', '.join(operator_names[:10])}{'...' if len(operator_names) > 10 else ''}\n")

        # 运行对比
        result = comparator.run_comparison(
            operator_names,
            num_cases=args.num_cases,
            max_iterations=args.max_iterations
        )

        # 输出并保存
        comparator.print_and_save_results(result)

        elapsed = time.time() - start_time
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
        print(f"\n⏱️ 总耗时: {h}h {m}m {s}s")

    finally:
        comparator.close()
        print("✅ 程序执行完成")


if __name__ == "__main__":
    main()
