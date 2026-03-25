#!/usr/bin/env python3
"""
PyTorch与MindSpore算子比较测试框架
比较前N个算子在两个框架中的执行结果是否一致
"""

import pymongo
import torch
import mindspore
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

class PyTorchMindSporeComparator:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "freefuzz-torch"):
        """
        初始化PyTorch和MindSpore比较器
        
        Args:
            mongo_uri: MongoDB连接URI
            db_name: 数据库名称
        """
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["argVS"]
        
        # 加载API映射表
        self.api_mapping = self.load_api_mapping()
        
        # 创建结果存储目录
        self.result_dir = os.path.abspath("pt_ms_log")
        os.makedirs(self.result_dir, exist_ok=True)
        print(f"📁 结果存储目录: {self.result_dir}")
        
        # 结果统计
        self.comparison_results = []
        
        # 已废弃的PyTorch算子列表
        self.deprecated_torch_apis = {
            "torch.symeig": "已在PyTorch 1.9版本中移除，请使用torch.linalg.eigh替代"
        }
        
        # 固定随机种子以确保可重复性
        self.random_seed = 42
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        mindspore.set_seed(self.random_seed)
        
    def load_api_mapping(self) -> Dict[str, Dict[str, str]]:
        """加载PyTorch到MindSpore的API映射表"""
        mapping_file = "api_mapping/pt_ms_mapping.csv"
        try:
            df = pd.read_csv(mapping_file)
            mapping = {}
            
            for _, row in df.iterrows():
                pt_api = str(row["PyTorch APIs"]).strip()
                ms_api = str(row["MindSpore APIs"]).strip()
                note = str(row.get("说明", "")).strip()
                mapping[pt_api] = {"ms_api": ms_api, "note": note}
            
            print(f"✅ 成功加载API映射表，共 {len(mapping)} 条映射")
            return mapping
        except Exception as e:
            print(f"❌ 加载API映射表失败: {e}")
            return {}
    
    def is_class_based_api(self, api_name: str) -> bool:
        """
        判断API是否是基于类的（通过检查是否包含大写字母）
        例如: torch.nn.Dropout2d, torch.nn.AvgPool2d
        """
        parts = api_name.split(".")
        if len(parts) >= 2:
            name = parts[-1]
            return any(c.isupper() for c in name)
        return False
    
    def convert_class_to_functional(self, torch_api: str) -> Tuple[Optional[str], Optional[str]]:
        """
        将类形式的API转换为函数形式
        例如: torch.nn.Dropout2d -> torch.nn.functional.dropout2d
        返回 (torch_functional_api, mindspore_functional_api)
        """
        if not self.is_class_based_api(torch_api):
            return None, None
        
        parts = torch_api.split(".")
        if len(parts) >= 3 and parts[1] == "nn":
            class_name = parts[-1]
            func_name = re.sub(r'(?<!^)(?<![0-9])([A-Z])', r'_\1', class_name).lower()
            torch_func_api = f"torch.nn.functional.{func_name}"
            ms_func_api = f"mindspore.ops.{func_name}"
            
            return torch_func_api, ms_func_api
        
        return None, None
    
    def convert_api_name(self, torch_api: str) -> Tuple[Optional[str], str]:
        """
        将PyTorch API转换为MindSpore API
        返回 (转换后的API名称, 使用的方法)
        """
        # 检查是否是类形式的API，如果是则转换为函数形式
        if self.is_class_based_api(torch_api):
            torch_func, ms_func = self.convert_class_to_functional(torch_api)
            if torch_func and ms_func:
                torch_func_obj = self.get_operator_function(torch_func, "torch")
                ms_func_obj = self.get_operator_function(ms_func, "mindspore")
                if torch_func_obj and ms_func_obj:
                    return ms_func, "类转函数"
        
        # 优先查映射表
        if torch_api in self.api_mapping:
            note = self.api_mapping[torch_api]["note"]
            ms_api = self.api_mapping[torch_api]["ms_api"]
            
            if "无对应实现" in note or "无" in note:
                return None, "无对应实现"
            elif "一致" in note:
                return ms_api, "映射表(功能一致)"
            else:
                return ms_api, "映射表(有差异)"
        
        # 默认转换规则
        api = torch_api.replace("torch", "mindspore.ops", 1)
        return api, "名称转换"
    
    def convert_dtype(self, torch_dtype_str: str) -> str:
        """将torch的dtype转换为mindspore的dtype"""
        if isinstance(torch_dtype_str, str):
            if torch_dtype_str.startswith("torch."):
                return torch_dtype_str.replace("torch.", "mindspore.")
        return torch_dtype_str
    
    def convert_key(self, key: str, ms_api: str = "") -> str:
        """转换参数名"""
        key_mapping = {
            "input": "x",
            "other": "y",
        }
        return key_mapping.get(key, key)
    
    def should_skip_param(self, key: str, ms_api: str) -> bool:
        """判断是否应该跳过某个参数（MindSpore不支持或版本不兼容）"""
        common_skip_params = ["layout", "requires_grad", "out", "memory_format"]
        
        skip_params = {
            "mindspore.ops.selu": ["inplace"],
        }
        
        if key in common_skip_params:
            return True
        
        if ms_api in skip_params:
            return key in skip_params[ms_api]
        
        return False
    
    def generate_numpy_data(self, data: Any) -> np.ndarray:
        """生成numpy数组作为共享数据源，确保PyTorch和MindSpore使用相同的输入"""
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
        """转换数据为PyTorch张量"""
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
        """转换数据为MindSpore张量"""
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
    
    def prepare_shared_numpy_data(self, document: Dict[str, Any], case_index: int) -> Dict[str, Any]:
        """
        准备共享的numpy数据，确保PyTorch和MindSpore使用相同的输入
        返回一个字典，键为参数名，值为numpy数组
        """
        shared_data = {}
        api_name = document.get("api", "")
        
        # 对于类形式的API，如果没有input参数，生成默认输入
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
        
        # 处理*size参数
        if "*size" in document:
            size_data = document["*size"]
            if isinstance(size_data, list) and len(size_data) > case_index:
                shared_data["*size"] = size_data[case_index]
        
        # 处理*tensors参数
        if "*tensors" in document:
            tensors_data = document["*tensors"]
            if isinstance(tensors_data, list) and len(tensors_data) > case_index:
                tensor_list = tensors_data[case_index]
                if isinstance(tensor_list, list):
                    shared_data["*tensors"] = [self.generate_numpy_data(t) for t in tensor_list]
                else:
                    shared_data["*tensors"] = [self.generate_numpy_data(tensor_list)]
        
        # 处理其他张量参数
        for param_name in ["condition", "x", "y", "input"]:
            if param_name in document:
                param_data = document[param_name]
                if isinstance(param_data, list) and len(param_data) > case_index:
                    shared_data[param_name] = self.generate_numpy_data(param_data[case_index])
        
        # 处理其他参数
        exclude_keys = ["_id", "api", "condition", "x", "y", "input", "*size", "*tensors", "tensors", "out", "eigenvectors", "upper"]
        for key, value in document.items():
            if key not in exclude_keys:
                if isinstance(value, list) and len(value) > 0:
                    idx = min(case_index, len(value) - 1)
                    param_value = value[idx]
                    if isinstance(param_value, dict):
                        shared_data[key] = self.generate_numpy_data(param_value)
                    else:
                        shared_data[key] = param_value
        
        return shared_data
    
    def prepare_arguments_torch(self, document: Dict[str, Any], case_index: int, shared_data: Dict[str, Any] = None) -> Tuple[List[Any], Dict[str, Any]]:
        """为PyTorch准备参数"""
        args = []
        kwargs = {}
        
        if shared_data is None:
            shared_data = self.prepare_shared_numpy_data(document, case_index)
        
        # 处理*size参数
        if "*size" in shared_data:
            size_value = shared_data["*size"]
            if isinstance(size_value, list):
                args.append(tuple(size_value) if size_value else ())
            elif isinstance(size_value, int):
                args.append((size_value,))
            else:
                args.append(size_value)
        
        # 处理*tensors参数
        if "*tensors" in shared_data:
            for numpy_tensor in shared_data["*tensors"]:
                args.append(self.convert_to_tensor_torch(None, numpy_tensor))
        
        # 处理其他张量参数
        for param_name in ["condition", "x", "y", "input"]:
            if param_name in shared_data:
                numpy_data = shared_data[param_name]
                tensor = self.convert_to_tensor_torch(None, numpy_data)
                args.append(tensor)
        
        # 处理其他参数
        for key, value in shared_data.items():
            if key not in ["*size", "*tensors", "condition", "x", "y", "input"]:
                if isinstance(value, np.ndarray):
                    kwargs[key] = self.convert_to_tensor_torch(None, value)
                else:
                    kwargs[key] = value
        
        # 转换dtype和device参数
        kwargs = self.convert_dtype_device_params_torch(kwargs)
        
        return args, kwargs
    
    def prepare_arguments_mindspore(self, document: Dict[str, Any], case_index: int, ms_api: str, shared_data: Dict[str, Any] = None) -> Tuple[List[Any], Dict[str, Any]]:
        """为MindSpore准备参数"""
        args = []
        kwargs = {}
        
        if shared_data is None:
            shared_data = self.prepare_shared_numpy_data(document, case_index)
        
        # 处理*size参数
        if "*size" in shared_data:
            size_value = shared_data["*size"]
            if isinstance(size_value, list):
                args.append(tuple(size_value) if size_value else ())
            elif isinstance(size_value, int):
                args.append((size_value,))
            else:
                args.append(size_value)
        
        # 处理*tensors参数
        if "*tensors" in shared_data:
            for numpy_tensor in shared_data["*tensors"]:
                args.append(self.convert_to_tensor_mindspore(None, numpy_tensor))
        
        # 处理其他张量参数
        for param_name in ["condition", "x", "y", "input"]:
            if param_name in shared_data:
                numpy_data = shared_data[param_name]
                tensor = self.convert_to_tensor_mindspore(None, numpy_data)
                args.append(tensor)
        
        # 处理其他参数
        for key, value in shared_data.items():
            if key not in ["*size", "*tensors", "condition", "x", "y", "input"]:
                if self.should_skip_param(key, ms_api):
                    continue
                
                new_key = self.convert_key(key, ms_api)
                if isinstance(value, np.ndarray):
                    kwargs[new_key] = self.convert_to_tensor_mindspore(None, value)
                else:
                    kwargs[new_key] = value
        
        # 转换dtype参数
        kwargs = self.convert_dtype_device_params_mindspore(kwargs)
        
        return args, kwargs
    
    def convert_dtype_device_params_torch(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """转换PyTorch的dtype和device参数"""
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
    
    def convert_dtype_device_params_mindspore(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """转换MindSpore的dtype参数"""
        if "dtype" in kwargs:
            dtype_value = kwargs["dtype"]
            if isinstance(dtype_value, str):
                if dtype_value == "torchdtype":
                    kwargs["dtype"] = mindspore.float32
                else:
                    torch_dtype = dtype_value
                    if torch_dtype.startswith("torch."):
                        ms_dtype = torch_dtype.replace("torch.", "mindspore.")
                        dtype_map = {
                            "mindspore.float32": mindspore.float32,
                            "mindspore.float64": mindspore.float64,
                            "mindspore.int32": mindspore.int32,
                            "mindspore.int64": mindspore.int64,
                            "mindspore.bool_": mindspore.bool_,
                            "mindspore.uint8": mindspore.uint8
                        }
                        kwargs["dtype"] = dtype_map.get(ms_dtype, mindspore.float32)
                    else:
                        kwargs["dtype"] = mindspore.float32
            elif isinstance(dtype_value, int):
                int_dtype_map = {
                    0: mindspore.float32, 1: mindspore.float64, 2: mindspore.int32,
                    3: mindspore.int64, 4: mindspore.bool_, 5: mindspore.uint8, 8: mindspore.float32,
                }
                kwargs["dtype"] = int_dtype_map.get(dtype_value, mindspore.float32)
        
        # MindSpore不使用device参数，移除它
        if "device" in kwargs:
            del kwargs["device"]
        
        return kwargs
    
    def get_operator_function(self, api_name: str, framework: str = "torch"):
        """获取算子函数"""
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
    
    def compare_tensors(self, torch_result, ms_result, tolerance: float = 1e-5) -> Tuple[bool, str]:
        """比较两个张量是否相等"""
        try:
            # 转换为numpy进行比较
            if hasattr(torch_result, 'detach'):
                torch_np = torch_result.detach().cpu().numpy()
            else:
                torch_np = np.array(torch_result)
            
            if hasattr(ms_result, 'asnumpy'):
                ms_np = ms_result.asnumpy()
            else:
                ms_np = np.array(ms_result)
            
            # 检查形状
            if torch_np.shape != ms_np.shape:
                return False, f"形状不匹配: PyTorch {torch_np.shape} vs MindSpore {ms_np.shape}"
            
            # 检查dtype是否为布尔类型或其他非数值类型
            if torch_np.dtype == np.bool_ or ms_np.dtype == np.bool_:
                if np.array_equal(torch_np, ms_np):
                    return True, "布尔值匹配"
                else:
                    diff_count = np.sum(torch_np != ms_np)
                    return False, f"布尔值不匹配，差异数量: {diff_count}"
            
            # 检查是否为对象类型或其他非数值类型
            if not np.issubdtype(torch_np.dtype, np.number) or not np.issubdtype(ms_np.dtype, np.number):
                if np.array_equal(torch_np, ms_np):
                    return True, "值匹配"
                else:
                    return False, f"值不匹配 (dtype: torch={torch_np.dtype}, ms={ms_np.dtype})"
            
            # 检查数值（仅对数值类型）
            if np.allclose(torch_np, ms_np, atol=tolerance, rtol=tolerance, equal_nan=True):
                return True, "数值匹配"
            else:
                max_diff = np.max(np.abs(torch_np - ms_np))
                return False, f"数值不匹配，最大差异: {max_diff}"
        
        except Exception as e:
            return False, f"比较过程出错: {str(e)}"
    
    def test_single_case(self, document: Dict[str, Any], case_index: int) -> Dict[str, Any]:
        """测试单个用例"""
        torch_api = document.get("api", "unknown")
        test_id = str(document.get("_id", "unknown"))
        
        # 检查是否是类形式的API，如果是则转换为函数形式
        original_torch_api = torch_api
        if self.is_class_based_api(torch_api):
            torch_func, _ = self.convert_class_to_functional(torch_api)
            if torch_func:
                torch_func_obj = self.get_operator_function(torch_func, "torch")
                if torch_func_obj:
                    torch_api = torch_func
                    print(f"    🔄 类转函数: {original_torch_api} -> {torch_api}")
        
        result = {
            "test_id": test_id,
            "torch_api": original_torch_api,
            "torch_api_used": torch_api,
            "case_index": case_index + 1,
            "status": "unknown",
            "mindspore_api": None,
            "mapping_method": None,
            "torch_success": False,
            "mindspore_success": False,
            "results_match": False,
            "torch_error": None,
            "mindspore_error": None,
            "comparison_error": None,
            "torch_shape": None,
            "mindspore_shape": None,
            "torch_dtype": None,
            "mindspore_dtype": None
        }
        
        # 检查是否为已废弃的算子
        if torch_api in self.deprecated_torch_apis:
            result["status"] = "deprecated"
            result["torch_error"] = self.deprecated_torch_apis[torch_api]
            result["mindspore_error"] = "对应PyTorch算子已废弃，跳过测试"
            return result
        
        # 获取对应的MindSpore API
        ms_api, mapping_method = self.convert_api_name(torch_api)
        result["mindspore_api"] = ms_api
        result["mapping_method"] = mapping_method
        
        if ms_api is None:
            result["status"] = "no_mindspore_equivalent"
            return result
        
        # 生成共享的numpy数据
        shared_data = self.prepare_shared_numpy_data(document, case_index)
        
        # 测试PyTorch
        torch_result = None
        try:
            torch_func = self.get_operator_function(torch_api, "torch")
            if torch_func is None:
                result["torch_error"] = f"PyTorch算子 {torch_api} 未找到"
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
            
            if "deprecated" in error_msg.lower() or "removed" in error_msg.lower():
                if torch_api not in self.deprecated_torch_apis:
                    self.deprecated_torch_apis[torch_api] = f"运行时发现已废弃: {error_msg[:100]}..."
                result["status"] = "deprecated"
                result["mindspore_error"] = "对应PyTorch算子已废弃，跳过测试"
                return result
        
        # 测试MindSpore
        ms_result = None
        try:
            ms_func = self.get_operator_function(ms_api, "mindspore")
            if ms_func is None:
                result["mindspore_error"] = f"MindSpore算子 {ms_api} 未找到"
            else:
                args, kwargs = self.prepare_arguments_mindspore(document, case_index, ms_api, shared_data)
                ms_result = ms_func(*args, **kwargs)
                result["mindspore_success"] = True
                result["mindspore_shape"] = list(ms_result.shape) if hasattr(ms_result, 'shape') else None
                result["mindspore_dtype"] = str(ms_result.dtype) if hasattr(ms_result, 'dtype') else None
        except Exception as e:
            result["mindspore_error"] = str(e)
        
        # 比较结果
        if result["torch_success"] and result["mindspore_success"]:
            try:
                is_match, comparison_msg = self.compare_tensors(torch_result, ms_result)
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
    
    def get_num_test_cases(self, document: Dict[str, Any]) -> int:
        """获取文档中的测试用例数量"""
        max_len = 0
        for key, value in document.items():
            if key not in ["_id", "api"] and isinstance(value, list):
                max_len = max(max_len, len(value))
        return max_len if max_len > 0 else 1
    
    def compare_first_n_operators(self, n: int = 20) -> List[Dict[str, Any]]:
        """比较前N个算子"""
        print(f"🚀 开始PyTorch与MindSpore算子比较测试 (前{n}个算子)...")
        print(f"📊 连接MongoDB: {self.client.address}")
        
        documents = list(self.collection.find().limit(n))
        
        print(f"📋 找到 {len(documents)} 个算子:")
        total_cases = 0
        for i, doc in enumerate(documents):
            api_name = doc.get("api", "unknown")
            num_cases = self.get_num_test_cases(doc)
            total_cases += num_cases
            print(f"  {i+1}. {api_name} ({num_cases} 个测试用例)")
        
        print(f"\n🎯 总计需要执行: {total_cases} 个测试用例")
        
        results = []
        current_case = 1
        
        for doc_idx, doc in enumerate(documents):
            api_name = doc.get("api", "unknown")
            num_cases = self.get_num_test_cases(doc)
            
            print(f"\n🔧 测试算子 {doc_idx+1}/{len(documents)}: {api_name} ({num_cases} 个用例)")
            
            for case_idx in range(num_cases):
                print(f"  用例 {current_case}/{total_cases} (算子用例: {case_idx+1}/{num_cases}): {api_name}")
                
                result = self.test_single_case(doc, case_idx)
                result["operator"] = api_name
                result["total_cases_for_operator"] = num_cases
                results.append(result)
                
                # 显示结果
                if result["status"] == "compared":
                    if result["results_match"]:
                        print(f"    ✅ 结果一致")
                    else:
                        print(f"    ❌ 结果不一致: {result['comparison_error']}")
                elif result["status"] == "deprecated":
                    print(f"    ⏭️ 已废弃算子，跳过测试")
                elif result["status"] == "no_mindspore_equivalent":
                    print(f"    ⚠️ MindSpore无对应实现")
                else:
                    print(f"    ❌ 测试失败: {result['status']}")
                
                current_case += 1
        
        self.print_summary(results)
        return results
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """打印测试结果摘要"""
        total = len(results)
        compared = len([r for r in results if r["status"] == "compared"])
        matched = len([r for r in results if r["results_match"]])
        deprecated = len([r for r in results if r["status"] == "deprecated"])
        no_ms = len([r for r in results if r["status"] == "no_mindspore_equivalent"])
        torch_failed = len([r for r in results if r["status"] in ["torch_failed", "both_failed"]])
        ms_failed = len([r for r in results if r["status"] in ["mindspore_failed", "both_failed"]])
        
        print(f"\n📊 测试结果摘要:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📈 总测试用例: {total}")
        print(f"🔍 成功比较: {compared}")
        print(f"✅ 结果一致: {matched}")
        print(f"❌ 结果不一致: {compared - matched}")
        print(f"⏭️ 已废弃算子: {deprecated}")
        print(f"⚠️ MindSpore无对应实现: {no_ms}")
        print(f"🔴 PyTorch执行失败: {torch_failed}")
        print(f"🟠 MindSpore执行失败: {ms_failed}")
        
        if compared > 0:
            match_rate = matched / compared * 100
            print(f"📊 一致性比率: {match_rate:.1f}%")
        
        if deprecated > 0:
            deprecated_apis = set([r["torch_api"] for r in results if r["status"] == "deprecated"])
            print(f"\n⏭️ 已废弃的算子 ({len(deprecated_apis)} 个):")
            for api in sorted(deprecated_apis):
                print(f"  - {api}")
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = None):
        """保存测试结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pt_ms_comparison_{timestamp}.json"
        
        filepath = os.path.join(self.result_dir, filename)
        
        total = len(results)
        compared = len([r for r in results if r["status"] == "compared"])
        matched = len([r for r in results if r["results_match"]])
        deprecated = len([r for r in results if r["status"] == "deprecated"])
        no_ms = len([r for r in results if r["status"] == "no_mindspore_equivalent"])
        torch_failed = len([r for r in results if r["status"] in ["torch_failed", "both_failed"]])
        ms_failed = len([r for r in results if r["status"] in ["mindspore_failed", "both_failed"]])
        
        output_data = {
            "summary": {
                "total_tests": total,
                "compared": compared,
                "matched": matched,
                "mismatch": compared - matched,
                "deprecated": deprecated,
                "no_mindspore_equivalent": no_ms,
                "torch_failed": torch_failed,
                "mindspore_failed": ms_failed,
                "match_rate": matched / compared * 100 if compared > 0 else 0,
                "timestamp": datetime.now().isoformat()
            },
            "results": results,
            "deprecated_apis": list(self.deprecated_torch_apis.keys())
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 结果已保存到: {filepath}")
        
        mismatched_cases = [r for r in results if r["status"] == "compared" and not r["results_match"]]
        if mismatched_cases:
            mismatch_filename = f"pt_ms_mismatches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            mismatch_filepath = os.path.join(self.result_dir, mismatch_filename)
            
            with open(mismatch_filepath, 'w', encoding='utf-8') as f:
                json.dump(mismatched_cases, f, indent=2, ensure_ascii=False)
            
            print(f"⚠️ 不一致用例已保存到: {mismatch_filepath}")
    
    def close(self):
        """关闭MongoDB连接"""
        self.client.close()

def main():
    """主函数"""
    comparator = PyTorchMindSporeComparator()
    
    try:
        results = comparator.compare_first_n_operators(10)
        comparator.save_results(results)
    finally:
        comparator.close()

if __name__ == "__main__":
    main()
