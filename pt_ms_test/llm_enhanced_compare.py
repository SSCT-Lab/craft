#!/usr/bin/env python3
"""
基于LLM的PyTorch与MindSpore算子比较测试框架
使用大模型进行测试用例修复和变异，提高用例可用性和覆盖率
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

# 添加项目根目录到路径，以便导入 component 模块
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from component.doc.doc_crawler_factory import get_doc_content

class LLMEnhancedComparator:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "freefuzz-torch"):
        """
        初始化基于LLM的PyTorch和MindSpore比较器
        
        Args:
            mongo_uri: MongoDB连接URI
            db_name: 数据库名称
        """
        # MongoDB连接
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["argVS"]
        
        # 初始化LLM客户端（阿里千问大模型）
        # 优先从项目根目录的 aliyun.key 文件读取密钥，否则使用环境变量
        api_key = self._load_api_key()
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 加载API映射表
        self.api_mapping = self.load_api_mapping()
        
        # 创建结果存储目录（在 pt_ms_test 目录下）
        self.result_dir = os.path.join(ROOT_DIR, "pt_ms_test", "pt_ms_log_1")
        os.makedirs(self.result_dir, exist_ok=True)
        print(f"📁 结果存储目录: {self.result_dir}")
        
        # 固定随机种子以确保可重复性
        self.random_seed = 42
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        mindspore.set_seed(self.random_seed)
        
        # 设置MindSpore为动态图模式
        mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
        
        # 已废弃的PyTorch算子列表
        self.deprecated_torch_apis = {
            "torch.symeig": "已在PyTorch 1.9版本中移除，请使用torch.linalg.eigh替代"
        }
        
        # 会导致程序卡住或崩溃的算子列表（跳过这些算子的测试）
        self.problematic_apis = {
            "torch.triu": "会导致程序卡住",
            "torch.as_tensor": "会导致 MindSpore 底层崩溃",
        }
    
    def _load_api_key(self) -> str:
        """
        加载阿里云 API 密钥
        
        优先从项目根目录的 aliyun.key 文件读取，如果文件不存在则使用环境变量 DASHSCOPE_API_KEY
        
        Returns:
            API 密钥字符串
        """
        key_file = os.path.join(ROOT_DIR, "aliyun.key")
        
        # 优先从文件读取
        if os.path.exists(key_file):
            try:
                with open(key_file, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
                if api_key:
                    print(f"✅ 从文件加载 API 密钥: {key_file}")
                    return api_key
            except Exception as e:
                print(f"⚠️ 读取密钥文件失败: {e}")
        
        # 回退到环境变量
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            print(f"✅ 从环境变量加载 API 密钥: DASHSCOPE_API_KEY")
            return api_key
        
        # 都没有找到
        print("❌ 未找到 API 密钥，请确保 aliyun.key 文件存在或设置 DASHSCOPE_API_KEY 环境变量")
        return ""
    
    def load_api_mapping(self) -> Dict[str, Dict[str, str]]:
        """加载PyTorch到MindSpore的API映射表"""
        # 使用新的映射表：ms_api_mappings_final.csv
        mapping_file = os.path.join(ROOT_DIR, "component", "data", "ms_api_mappings_final.csv")
        try:
            df = pd.read_csv(mapping_file)
            mapping = {}
            
            for _, row in df.iterrows():
                # 新映射表的列名：pytorch-api, mindspore-api
                pt_api = str(row["pytorch-api"]).strip()
                ms_api = str(row["mindspore-api"]).strip()
                
                # 保留所有映射（包括"无对应实现"），在 convert_api_name 中处理
                mapping[pt_api] = {"ms_api": ms_api, "note": ""}
            
            print(f"✅ 成功加载API映射表，共 {len(mapping)} 条映射")
            print(f"📄 映射文件: {mapping_file}")
            return mapping
        except Exception as e:
            print(f"❌ 加载API映射表失败: {e}")
            return {}
    
    def is_class_based_api(self, api_name: str) -> bool:
        """判断API是否是基于类的"""
        parts = api_name.split(".")
        if len(parts) >= 2:
            name = parts[-1]
            return any(c.isupper() for c in name)
        return False
    
    # def convert_class_to_functional(self, torch_api: str) -> Tuple[Optional[str], Optional[str]]:
    #     """将类形式的API转换为函数形式"""
    #     if not self.is_class_based_api(torch_api):
    #         return None, None
    #     
    #     parts = torch_api.split(".")
    #     if len(parts) >= 3 and parts[1] == "nn":
    #         class_name = parts[-1]
    #         
    #         # 改进的正则表达式：正确处理连续大写字母
    #         # 1. 先在"小写字母后跟大写字母"的位置插入下划线
    #         # 2. 再在"连续大写字母中，最后一个大写字母前"插入下划线（如果后面跟小写字母）
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
        将PyTorch API转换为MindSpore API
        
        Returns:
            (转换后的PyTorch API, 转换后的MindSpore API, 映射方法说明)
            对于类算子，直接映射到对应的MindSpore类算子
        """
        # # 注释掉类转函数的逻辑
        # # 检查是否是类形式的API
        # if self.is_class_based_api(torch_api):
        #     torch_func, paddle_func = self.convert_class_to_functional(torch_api)
        #     if torch_func and paddle_func:
        #         torch_func_obj = self.get_operator_function(torch_func, "torch")
        #         paddle_func_obj = self.get_operator_function(paddle_func, "paddle")
        #         if torch_func_obj and paddle_func_obj:
        #             return torch_func, paddle_func, "类转函数"
        
        # 查映射表
        if torch_api in self.api_mapping:
            ms_api = self.api_mapping[torch_api]["ms_api"]
            
            # 检查是否为"无对应实现"
            if ms_api == "无对应实现":
                return torch_api, None, "无对应实现"
            else:
                return torch_api, ms_api, "映射表"
        
        # 映射表中没有该 API，不再进行手动转换，直接返回 None
        return torch_api, None, "映射表中未找到"
    
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
    
    def convert_key(self, key: str, mindspore_api: str = "") -> str:
        """转换参数名"""
        key_mapping = {
            "input": "x",
            "other": "y",
        }
        return key_mapping.get(key, key)
    
    def should_skip_param(self, key: str, mindspore_api: str) -> bool:
        """判断是否应该跳过某个参数"""
        common_skip_params = ["layout", "requires_grad", "out"]
        skip_params = {
            # 可以根据需要添加特定API的跳过参数
        }
        
        if key in common_skip_params:
            return True
        
        if mindspore_api in skip_params:
            return key in skip_params[mindspore_api]
        
        return False
    
    def generate_numpy_data(self, data: Any) -> np.ndarray:
        """
        生成numpy数组作为共享数据源
        
        支持的dtype格式：
        - 带torch前缀：torch.float32, torch.bool, torch.int64等
        - 不带前缀：float32, bool, int64等
        - numpy格式：float32, bool_, int64等
        """
        if isinstance(data, dict):
            # 扩展的dtype映射表，支持多种格式
            dtype_map = {
                # torch格式（带前缀）
                "torch.float64": np.float64,
                "torch.float32": np.float32,
                "torch.int64": np.int64,
                "torch.int32": np.int32,
                "torch.bool": np.bool_,
                "torch.uint8": np.uint8,
                # 不带torch前缀的格式（LLM可能返回这种格式）
                "float64": np.float64,
                "float32": np.float32,
                "int64": np.int64,
                "int32": np.int32,
                "bool": np.bool_,
                "uint8": np.uint8,
                # numpy格式
                "bool_": np.bool_,
                "float": np.float32,
                "int": np.int64,
            }
            
            shape = data.get("shape", [])
            dtype_str = data.get("dtype", "torch.float32")
            dtype = dtype_map.get(dtype_str, np.float32)
            
            # 如果dtype_str不在映射表中，打印警告
            if dtype_str not in dtype_map:
                print(f"      ⚠️ 警告：未识别的dtype '{dtype_str}'，使用默认值 float32")
            else:
                print(f"      ✅ dtype映射：'{dtype_str}' -> {dtype}")
            
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
        """准备共享的numpy数据，确保PyTorch和MindSpore使用相同的输入"""
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
        
        # 处理文档中的其他参数
        exclude_keys = ["_id", "api"]
        for key, value in document.items():
            if key not in exclude_keys:
                # 对于可变参数（以*开头），直接保存原始值，不进行转换
                # 转换工作将在prepare_arguments_torch/paddle中完成
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
    
    def prepare_arguments_torch(self, test_case: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        为PyTorch准备参数
        
        注意：
        1. 对于torch.where等函数，参数需要按顺序作为位置参数传递：
           - torch.where(condition, x, y) 或 torch.where(condition, input, other)
        2. 对于以*开头的参数（如*tensors），表示可变参数，需要解包为位置参数
        """
        args = []
        kwargs = {}
        
        # 首先检查是否有可变参数（以*开头的参数）
        varargs_key = None
        for key in test_case.keys():
            if key.startswith('*'):
                varargs_key = key
                break
        
        # 如果有可变参数，将其解包为位置参数
        if varargs_key:
            varargs_value = test_case[varargs_key]
            if isinstance(varargs_value, list):
                for item in varargs_value:
                    if isinstance(item, dict) and "shape" in item:
                        # 这是一个张量描述，生成numpy数据并转换
                        numpy_data = self.generate_numpy_data(item)
                        args.append(self.convert_to_tensor_torch(None, numpy_data))
                    elif isinstance(item, list):
                        # 嵌套列表，递归处理
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
        
        # 按顺序处理位置参数：condition, x/input, y/other
        # 这些参数需要作为位置参数传递，而不是关键字参数
        positional_params = ["condition", "x", "y", "input", "other"]
        
        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, np.ndarray):
                    args.append(self.convert_to_tensor_torch(None, value))
                else:
                    # 标量值直接添加
                    args.append(value)
        
        # 处理其他参数（作为关键字参数）
        for key, value in test_case.items():
            if key not in positional_params + ["api"]:
                if isinstance(value, np.ndarray):
                    kwargs[key] = self.convert_to_tensor_torch(None, value)
                else:
                    kwargs[key] = value
        
        return args, kwargs
    
    def prepare_arguments_mindspore(self, test_case: Dict[str, Any], mindspore_api: str) -> Tuple[List[Any], Dict[str, Any]]:
        """
        为MindSpore准备参数
        
        注意：
        1. 对于mindspore.mint.where等函数，参数也需要按顺序作为位置参数传递
        2. 对于以*开头的参数（如*tensors），表示可变参数，需要解包为位置参数
        """
        args = []
        kwargs = {}
        
        # 首先检查是否有可变参数（以*开头的参数）
        varargs_key = None
        for key in test_case.keys():
            if key.startswith('*'):
                varargs_key = key
                break
        
        # 如果有可变参数，将其解包为位置参数
        if varargs_key:
            varargs_value = test_case[varargs_key]
            if isinstance(varargs_value, list):
                for item in varargs_value:
                    if isinstance(item, dict) and "shape" in item:
                        # 这是一个张量描述，生成numpy数据并转换
                        numpy_data = self.generate_numpy_data(item)
                        args.append(self.convert_to_tensor_mindspore(None, numpy_data))
                    elif isinstance(item, list):
                        # 嵌套列表，递归处理
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
        
        # 按顺序处理位置参数：condition, x/input, y/other
        positional_params = ["condition", "x", "y", "input", "other"]
        
        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, np.ndarray):
                    args.append(self.convert_to_tensor_mindspore(None, value))
                else:
                    # 标量值直接添加
                    args.append(value)
        
        # 处理其他参数（作为关键字参数）
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
        """比较两个张量是否相等"""
        try:
            # 转换为numpy进行比较
            if hasattr(torch_result, 'detach'):
                torch_np = torch_result.detach().cpu().numpy()
            else:
                torch_np = np.array(torch_result)
            
            if hasattr(mindspore_result, 'asnumpy'):
                mindspore_np = mindspore_result.asnumpy()
            else:
                mindspore_np = np.array(mindspore_result)
            
            # 检查形状
            if torch_np.shape != mindspore_np.shape:
                return False, f"形状不匹配: PyTorch {torch_np.shape} vs MindSpore {mindspore_np.shape}"
            
            # 检查dtype是否为布尔类型
            if torch_np.dtype == np.bool_ or mindspore_np.dtype == np.bool_:
                if np.array_equal(torch_np, mindspore_np):
                    return True, "布尔值匹配"
                else:
                    diff_count = np.sum(torch_np != mindspore_np)
                    return False, f"布尔值不匹配，差异数量: {diff_count}"
            
            # 检查数值
            if np.allclose(torch_np, mindspore_np, atol=tolerance, rtol=tolerance, equal_nan=True):
                return True, "数值匹配"
            else:
                max_diff = np.max(np.abs(torch_np - mindspore_np))
                return False, f"数值不匹配，最大差异: {max_diff}"
        
        except Exception as e:
            return False, f"比较过程出错: {str(e)}"
    
    def execute_test_case(self, torch_api: str, mindspore_api: str, torch_test_case: Dict[str, Any], mindspore_test_case: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行单个测试用例
        
        Args:
            torch_api: PyTorch API名称
            mindspore_api: MindSpore API名称
            torch_test_case: PyTorch测试用例（包含参数信息）
            mindspore_test_case: MindSpore测试用例（包含参数信息）
        
        Returns:
            执行结果字典
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
        
        # 如果没有提供mindspore_test_case，则使用torch_test_case（向后兼容）
        if mindspore_test_case is None:
            mindspore_test_case = torch_test_case
        
        # 判断是否是类算子
        is_class_api = self.is_class_based_api(torch_api)
        
        # 测试PyTorch
        torch_result = None
        try:
            torch_func = self.get_operator_function(torch_api, "torch")
            if torch_func is None:
                result["torch_error"] = f"PyTorch算子 {torch_api} 未找到"
            else:
                args, kwargs = self.prepare_arguments_torch(torch_test_case)
                
                if is_class_api:
                    # 对于类算子，需要先实例化，然后调用
                    # 从kwargs中提取初始化参数（非input参数）
                    init_kwargs = {k: v for k, v in kwargs.items() if k != 'input'}
                    # 实例化类
                    torch_instance = torch_func(**init_kwargs)
                    # 获取输入数据（可能在args中或kwargs中）
                    if 'input' in kwargs:
                        input_data = kwargs['input']
                    elif len(args) > 0:
                        input_data = args[0]
                    else:
                        # 如果没有input参数，尝试使用默认输入
                        raise ValueError("类算子缺少input参数")
                    
                    # 调用实例（前向传播）
                    with torch.no_grad():
                        torch_result = torch_instance(input_data)
                else:
                    # 对于函数算子，直接调用
                    with torch.no_grad():
                        torch_result = torch_func(*args, **kwargs)
                
                result["torch_success"] = True
                result["torch_shape"] = list(torch_result.shape) if hasattr(torch_result, 'shape') else None
                result["torch_dtype"] = str(torch_result.dtype) if hasattr(torch_result, 'dtype') else None
        except Exception as e:
            result["torch_error"] = str(e)
            result["torch_traceback"] = traceback.format_exc()
        
        # 测试MindSpore
        mindspore_result = None
        try:
            mindspore_func = self.get_operator_function(mindspore_api, "mindspore")
            if mindspore_func is None:
                result["mindspore_error"] = f"MindSpore算子 {mindspore_api} 未找到"
            else:
                args, kwargs = self.prepare_arguments_mindspore(mindspore_test_case, mindspore_api)
                
                if is_class_api:
                    # 对于类算子，需要先实例化，然后调用
                    # 从kwargs中提取初始化参数（非x/input参数）
                    init_kwargs = {k: v for k, v in kwargs.items() if k not in ['x', 'input']}
                    # 实例化类
                    mindspore_instance = mindspore_func(**init_kwargs)
                    # 获取输入数据（可能在args中或kwargs中）
                    if 'x' in kwargs:
                        input_data = kwargs['x']
                    elif 'input' in kwargs:
                        input_data = kwargs['input']
                    elif len(args) > 0:
                        input_data = args[0]
                    else:
                        # 如果没有input参数，尝试使用默认输入
                        raise ValueError("类算子缺少input/x参数")
                    
                    # 调用实例（前向传播）
                    mindspore_result = mindspore_instance(input_data)
                else:
                    # 对于函数算子，直接调用
                    mindspore_result = mindspore_func(*args, **kwargs)
                
                result["mindspore_success"] = True
                result["mindspore_shape"] = list(mindspore_result.shape) if hasattr(mindspore_result, 'shape') else None
                result["mindspore_dtype"] = str(mindspore_result.dtype) if hasattr(mindspore_result, 'dtype') else None
        except Exception as e:
            result["mindspore_error"] = str(e)
            result["mindspore_traceback"] = traceback.format_exc()
        
        # 比较结果
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
        爬取PyTorch和MindSpore的API文档
        
        Args:
            torch_api: PyTorch API名称
            mindspore_api: MindSpore API名称
        
        Returns:
            (PyTorch文档内容, MindSpore文档内容)
        """
        # 文档有效性判断的最小长度阈值
        MIN_DOC_LENGTH = 300
        
        torch_doc = ""
        mindspore_doc = ""
        
        try:
            print(f"    📖 正在爬取 PyTorch 文档: {torch_api}")
            torch_doc = get_doc_content(torch_api, "pytorch")
            # 判断文档是否有效：1. 内容不为空 2. 不包含错误提示 3. 长度超过阈值
            if (torch_doc 
                and "Unable" not in torch_doc 
                and "not supported" not in torch_doc
                and len(torch_doc.strip()) > MIN_DOC_LENGTH):
                # 截断过长的文档以节省token
                if len(torch_doc) > 3000:
                    torch_doc = torch_doc[:3000] + "\n... (doc truncated)"
                print(f"    ✅ PyTorch 文档爬取成功，长度: {len(torch_doc)}")
            else:
                doc_len = len(torch_doc.strip()) if torch_doc else 0
                torch_doc = f"Unable to fetch documentation for {torch_api} (length: {doc_len}, min required: {MIN_DOC_LENGTH})"
                print(f"    ⚠️ {torch_doc}")
        except Exception as e:
            torch_doc = f"Failed to fetch documentation: {str(e)}"
            print(f"    ❌ PyTorch 文档爬取失败: {e}")
        
        try:
            print(f"    📖 正在爬取 MindSpore 文档: {mindspore_api}")
            mindspore_doc = get_doc_content(mindspore_api, "mindspore")
            # 判断文档是否有效：1. 内容不为空 2. 不包含错误提示 3. 长度超过阈值
            if (mindspore_doc 
                and "Unable" not in mindspore_doc 
                and "not supported" not in mindspore_doc
                and len(mindspore_doc.strip()) > MIN_DOC_LENGTH):
                # 截断过长的文档以节省token
                if len(mindspore_doc) > 3000:
                    mindspore_doc = mindspore_doc[:3000] + "\n... (doc truncated)"
                print(f"    ✅ MindSpore 文档爬取成功，长度: {len(mindspore_doc)}")
            else:
                doc_len = len(mindspore_doc.strip()) if mindspore_doc else 0
                mindspore_doc = f"Unable to fetch documentation for {mindspore_api} (length: {doc_len}, min required: {MIN_DOC_LENGTH})"
                print(f"    ⚠️ {mindspore_doc}")
        except Exception as e:
            mindspore_doc = f"Failed to fetch documentation: {str(e)}"
            print(f"    ❌ MindSpore 文档爬取失败: {e}")
        
        return torch_doc, mindspore_doc
    
    def _build_llm_prompt(self, execution_result: Dict[str, Any], torch_test_case: Dict[str, Any], mindspore_test_case: Dict[str, Any], torch_doc: str = "", mindspore_doc: str = "") -> str:
        """构建LLM的提示词"""
        torch_api = execution_result.get("torch_api", "")
        mindspore_api = execution_result.get("mindspore_api", "")
        status = execution_result.get("status", "")
        torch_success = execution_result.get("torch_success", False)
        mindspore_success = execution_result.get("mindspore_success", False)
        results_match = execution_result.get("results_match", False)
        torch_error = execution_result.get("torch_error", "")
        mindspore_error = execution_result.get("mindspore_error", "")
        comparison_error = execution_result.get("comparison_error", "")
        
        # 简化PyTorch测试用例以减少token消耗
        simplified_torch_test_case = {}
        for key, value in torch_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_torch_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_torch_test_case[key] = value
        
        # 简化MindSpore测试用例以减少token消耗
        simplified_mindspore_test_case = {}
        for key, value in mindspore_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_mindspore_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_mindspore_test_case[key] = value
        
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
        
        # 构建MindSpore参数示例
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
        
        # 构建API文档部分
        doc_section = ""
        if torch_doc or mindspore_doc:
            doc_section = f"""
## API文档参考
### PyTorch文档
{torch_doc if torch_doc else "无法获取文档"}

### MindSpore文档
{mindspore_doc if mindspore_doc else "无法获取文档"}
"""
        
        prompt = f"""请分析以下算子测试用例在PyTorch和MindSpore框架中的执行结果，并根据结果进行测试用例的修复或变异（fuzzing）。

## 测试信息
- **PyTorch API**: {torch_api}
- **MindSpore API**: {mindspore_api}
{doc_section}
## 执行结果
- **执行状态**: {status}
- **PyTorch执行成功**: {torch_success}
- **MindSpore执行成功**: {mindspore_success}
- **结果是否一致**: {results_match}

## 错误信息
- **PyTorch错误**: {torch_error if torch_error else "无"}
- **MindSpore错误**: {mindspore_error if mindspore_error else "无"}
- **比较错误**: {comparison_error if comparison_error else "无"}

## 原始测试用例

### PyTorch测试用例
```json
{json.dumps(simplified_torch_test_case, indent=2, ensure_ascii=False)}
```

### MindSpore测试用例
```json
{json.dumps(simplified_mindspore_test_case, indent=2, ensure_ascii=False)}
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
  "mindspore_test_case": {{
    "api": "{mindspore_api}",
{mindspore_param_example_str}
  }}
}}

**重要说明**：
1. operation必须是 "mutation"、"repair" 或 "skip" 之一
2. 张量参数必须使用 {{"shape": [...], "dtype": "..."}} 格式
3. 标量参数直接使用数值，例如 "y": 0
4. 必须保证两个框架用例的输入相同、参数在语义上严格等价、理论上具有相同输出。
5. PyTorch和MindSpore的测试用例可以有参数名差异（如input vs x）、参数值差异或者参数数量的差异，只要保证理论上输出相同就行。
6. 如果这个算子找不到官方文档，请判断是否是因为该算子不存在或者已经从PyTorch或者MindSpore的当前版本移除了，如果是这样，请将 operation 设置为 "skip"，不需要尝试修复。
7. 测试用例变异时可适当探索一些极端情况，例如：空张量（shape包含0）、单元素张量（shape=[1]或[]）、高维张量、超大张量、不同数据类型（int、float、bool）、边界值等，以提高测试覆盖率和发现潜在bug
8. 请仔细阅读官方API文档，确保参数名称、参数类型、参数取值范围等与文档一致
"""
        return prompt
    
    def call_llm_for_repair_or_mutation(self, execution_result: Dict[str, Any], torch_test_case: Dict[str, Any], mindspore_test_case: Dict[str, Any], torch_doc: str = "", mindspore_doc: str = "") -> Dict[str, Any]:
        """调用LLM进行测试用例修复或变异"""
        prompt = self._build_llm_prompt(execution_result, torch_test_case, mindspore_test_case, torch_doc, mindspore_doc)
        
        # 打印传入LLM的简化测试用例
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
        
        # 简化打印：注释掉详细的测试用例信息（日志中会有记录）
        # print(f"\n{'='*40}")
        # print(f"📋 传入LLM的测试用例 (简化后):")
        # print(f"{'='*40}")
        # print(f"\n🅿️ PyTorch测试用例:")
        # print(json.dumps(simplified_torch_test_case, indent=2, ensure_ascii=False))
        # print(f"\n🅰️ MindSpore测试用例:")
        # print(json.dumps(simplified_mindspore_test_case, indent=2, ensure_ascii=False))
        # print(f"\n{'='*40}\n")
        
        try:
            print(f"    🤖 正在调用LLM进行分析...")
            completion = self.llm_client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个深度学习框架测试专家，精通PyTorch和MindSpore框架的API差异。你的任务是根据测试用例的执行结果，判断是否需要修复或变异测试用例，并返回严格的JSON格式结果。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
            )
            
            raw_response = completion.choices[0].message.content.strip()
            # 简化打印：注释掉LLM原始响应（日志中会有记录）
            # print(f"    🤖 LLM原始响应: {raw_response[:200]}...")
            # print(f"    🤖 LLM原始响应: {raw_response}")
            
            # 添加1秒时间间隔，避免API调用过于频繁
            time.sleep(1)
            
            # 尝试解析JSON
            try:
                llm_result = json.loads(raw_response)
                return llm_result
            except json.JSONDecodeError as e:
                print(f"    ⚠️ LLM返回的不是有效的JSON，尝试提取JSON内容...")
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    llm_result = json.loads(json_match.group())
                    return llm_result
                else:
                    return {
                        "operation": "skip",
                        "reason": f"LLM返回格式错误: {e}",
                        "pytorch_test_case": torch_test_case,
                        "mindspore_test_case": mindspore_test_case
                    }
        
        except Exception as e:
            print(f"    ❌ 调用LLM失败: {e}")
            return {
                "operation": "skip",
                "reason": f"LLM调用失败: {e}",
                "pytorch_test_case": torch_test_case,
                "mindspore_test_case": mindspore_test_case
            }
    
    def get_num_test_cases_from_document(self, document: Dict[str, Any]) -> int:
        """获取文档中的测试用例数量"""
        max_len = 0
        # 遍历文档中的所有字段，找出列表类型字段的最大长度
        for key, value in document.items():
            if key not in ["_id", "api"] and isinstance(value, list):
                max_len = max(max_len, len(value))
        return max_len if max_len > 0 else 1
    
    def llm_enhanced_test_operator(self, operator_name: str, max_iterations: int = 3, num_test_cases: int = None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        使用LLM增强的方式测试单个算子
        
        Args:
            operator_name: 算子名称，例如 "torch.where"
            max_iterations: 每个测试用例的最大迭代次数
            num_test_cases: 要测试的用例数量，None表示测试所有用例
        
        Returns:
            (所有测试用例的所有迭代结果列表, 统计信息字典)
        """
        print(f"\n{'='*80}")
        print(f"🎯 开始测试算子: {operator_name}")
        print(f"🔄 每个用例的最大迭代次数: {max_iterations}")
        print(f"{'='*80}\n")
        
        # 初始化统计计数器
        stats = {
            "llm_generated_cases": 0,      # LLM生成的测试用例总数
            "successful_cases": 0           # 两个框架都执行成功的测试用例数
        }
        
        # 检查是否是会导致程序卡住的算子
        if operator_name in self.problematic_apis:
            reason = self.problematic_apis[operator_name]
            print(f"⏭️ 跳过算子 {operator_name}：{reason}")
            return [], stats
        
        # 从MongoDB获取算子的测试用例
        document = self.collection.find_one({"api": operator_name})
        if document is None:
            print(f"❌ 未找到算子 {operator_name} 的测试用例")
            return [], stats
        
        # 获取测试用例总数
        total_cases = self.get_num_test_cases_from_document(document)
        print(f"📊 数据库中共有 {total_cases} 个测试用例")
        
        # 确定实际要测试的用例数量
        if num_test_cases is None:
            num_test_cases = total_cases
            print(f"📝 将测试所有 {num_test_cases} 个用例")
        else:
            num_test_cases = min(num_test_cases, total_cases)
            print(f"📝 将测试前 {num_test_cases} 个用例（共 {total_cases} 个）")
        
        # 获取转换后的PyTorch和MindSpore API
        torch_api, mindspore_api, mapping_method = self.convert_api_name(operator_name)
        if mindspore_api is None:
            print(f"❌ 算子 {operator_name} 无MindSpore对应实现")
            return [], stats
        
        # 显示API映射信息
        if torch_api != operator_name:
            print(f"✅ 原始 PyTorch API: {operator_name}")
            print(f"✅ 转换后 PyTorch API: {torch_api}")
        else:
            print(f"✅ PyTorch API: {torch_api}")
        print(f"✅ MindSpore API: {mindspore_api}")
        print(f"✅ 映射方法: {mapping_method}\n")
        
        # 存储所有测试用例的所有迭代结果
        all_results = []
        
        # 循环测试每个用例
        for case_idx in range(num_test_cases):
            print(f"\n{'#'*80}")
            print(f"📋 测试用例 {case_idx + 1}/{num_test_cases}")
            print(f"{'#'*80}")
            
            # 准备当前测试用例的初始数据
            print(f"  📦 准备测试用例 {case_idx + 1} 的数据...")
            initial_test_case = self.prepare_shared_numpy_data(document, case_index=case_idx)
            # 使用PyTorch API
            initial_test_case["api"] = torch_api
            
            # 打印测试用例的参数信息
            print(f"  📝 测试用例参数：")
            for key, value in initial_test_case.items():
                if key == "api":
                    continue
                if isinstance(value, np.ndarray):
                    print(f"    - {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"    - {key}: {value}")
            
            # 对当前测试用例进行迭代测试
            # 使用转换后的API进行测试
            case_results = self._test_single_case_with_iterations(
                torch_api, 
                mindspore_api, 
                initial_test_case, 
                max_iterations,
                case_idx + 1,
                stats
            )
            
            # 保存当前测试用例的结果
            all_results.extend(case_results)
        
        print(f"\n{'='*80}")
        print(f"✅ 所有测试完成")
        print(f"📊 共测试 {num_test_cases} 个用例，总计 {len(all_results)} 次迭代")
        print(f"📊 LLM生成的测试用例数: {stats['llm_generated_cases']}")
        print(f"📊 两个框架都执行成功的用例数: {stats['successful_cases']}")
        print(f"{'='*80}\n")
        
        return all_results, stats
    
    def _test_single_case_with_iterations(self, operator_name: str, mindspore_api: str, 
                                          initial_test_case: Dict[str, Any], 
                                          max_iterations: int,
                                          case_number: int,
                                          stats: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        对单个测试用例进行多轮迭代测试
        
        Args:
            operator_name: PyTorch算子名称
            mindspore_api: MindSpore算子名称
            initial_test_case: 初始测试用例
            max_iterations: 最大迭代次数
            case_number: 测试用例编号（用于显示）
            stats: 统计信息字典（用于记录LLM生成的用例数和成功执行的用例数）
        
        Returns:
            该测试用例的所有迭代结果
        """
        # 存储当前测试用例的所有迭代结果
        case_results = []
        
        # 当前测试用例
        # PyTorch 使用原始测试用例（api 已设置为 torch_api）
        # MindSpore 需要创建副本并设置正确的 api
        current_torch_test_case = initial_test_case
        current_mindspore_test_case = copy.deepcopy(initial_test_case)
        current_mindspore_test_case["api"] = mindspore_api  # 设置正确的 MindSpore API
        
        # 标记当前用例是否为LLM生成的（第一次迭代是数据库原始用例）
        is_llm_generated = False
        
        # 预先爬取API文档（只爬取一次，后续迭代复用）
        print(f"\n  📖 预先爬取API文档...")
        torch_doc, mindspore_doc = self._fetch_api_docs(operator_name, mindspore_api)
        
        # 开始迭代测试
        for iteration in range(max_iterations):
            print(f"\n{'─'*80}")
            print(f"🔄 迭代 {iteration + 1}/{max_iterations}")
            if is_llm_generated:
                print(f"   (LLM生成的用例)")
            else:
                print(f"   (数据库原始用例)")
            print(f"{'─'*80}")
            
            # 执行测试用例
            try:
                # print(f"  📝 执行测试用例...")
                execution_result = self.execute_test_case(operator_name, mindspore_api, current_torch_test_case, current_mindspore_test_case)
                
                # 简化打印：只显示关键结果状态
                status = execution_result['status']
                torch_ok = "✓" if execution_result['torch_success'] else "✗"
                mindspore_ok = "✓" if execution_result['mindspore_success'] else "✗"
                match_ok = "✓" if execution_result['results_match'] else "✗"
                print(f"  📊 执行结果: status={status}, PyTorch={torch_ok}, MindSpore={mindspore_ok}, match={match_ok}")
                
                # 注释掉详细的执行结果打印（日志中会有记录）
                # print(f"  📊 执行状态: {execution_result['status']}")
                # print(f"  🅿️ PyTorch执行结果: {execution_result['torch_success']}")
                # print(f"  🅰️ MindSpore执行结果: {execution_result['mindspore_success']}")
                # print(f"  ❓ 结果一致: {execution_result['results_match']}")
                
                # 只在有错误时打印错误信息
                if execution_result['torch_error']:
                    print(f"  ❌ PyTorch错误: {execution_result['torch_error'][:100]}..." if len(str(execution_result['torch_error'])) > 100 else f"  ❌ PyTorch错误: {execution_result['torch_error']}")
                if execution_result['mindspore_error']:
                    print(f"  ❌ MindSpore错误: {execution_result['mindspore_error'][:100]}..." if len(str(execution_result['mindspore_error'])) > 100 else f"  ❌ MindSpore错误: {execution_result['mindspore_error']}")
                # if execution_result['comparison_error']:
                #     print(f"  ⚠️ 比较错误: {execution_result['comparison_error']}")
                
                # 仅统计LLM生成的用例（不包括数据库原始用例）
                if is_llm_generated:
                    # 统计两个框架都执行成功的用例数
                    if execution_result['torch_success'] and execution_result['mindspore_success']:
                        stats["successful_cases"] += 1
                        # print(f"  📊 LLM生成的成功执行用例计数: {stats['successful_cases']}")
                
            except Exception as e:
                print(f"  ❌ 执行测试用例时发生严重错误: {str(e)[:100]}..." if len(str(e)) > 100 else f"  ❌ 执行测试用例时发生严重错误: {e}")
                # print(f"  ❌ 错误详情:")
                # traceback.print_exc()
                
                # 创建一个错误结果
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
            
            # 保存本次迭代结果
            iteration_result = {
                "iteration": iteration + 1,
                "torch_test_case": current_torch_test_case,
                "mindspore_test_case": current_mindspore_test_case,
                "execution_result": execution_result,
                "llm_operation": None,
                "is_llm_generated": is_llm_generated
            }
            
            # 调用LLM进行修复或变异（传入PyTorch和MindSpore测试用例及API文档）
            try:
                # print(f"\n  🤖 调用LLM进行测试用例分析...")
                llm_result = self.call_llm_for_repair_or_mutation(
                    execution_result, 
                    current_torch_test_case, 
                    current_mindspore_test_case,
                    torch_doc,
                    mindspore_doc
                )
            except Exception as e:
                print(f"  ❌ 调用LLM时发生错误: {str(e)[:100]}..." if len(str(e)) > 100 else f"  ❌ 调用LLM时发生错误: {e}")
                # print(f"  ❌ 错误详情:")
                # traceback.print_exc()
                print(f"  ⏭️ 跳过LLM分析，结束此测试用例的迭代")
                
                # 创建一个skip操作
                llm_result = {
                    "operation": "skip",
                    "reason": f"LLM调用失败: {str(e)}"
                }
                
                iteration_result["llm_operation"] = llm_result
                iteration_result["case_number"] = case_number
                case_results.append(iteration_result)
                break  # 结束迭代循环
            
            operation = llm_result.get("operation", "skip")
            reason = llm_result.get("reason", "")
            
            # 简化打印：只显示操作类型
            print(f"  🤖 LLM决策: {operation}")
            # print(f"  🤖 LLM操作: {operation}")
            # print(f"  🤖 LLM原因: {reason}")
            
            iteration_result["llm_operation"] = {
                "operation": operation,
                "reason": reason
            }
            
            # 添加测试用例编号信息
            iteration_result["case_number"] = case_number
            case_results.append(iteration_result)
            
            # 如果LLM建议跳过，则结束迭代
            if operation == "skip":
                # print(f"  ⏭️ LLM建议跳过，结束迭代")
                break
            
            # 准备下一次迭代的测试用例
            if operation == "mutation":
                # print(f"  🔀 使用LLM变异后的测试用例")
                next_pytorch_test_case = llm_result.get("pytorch_test_case", current_torch_test_case)
                next_mindspore_test_case = llm_result.get("mindspore_test_case", current_mindspore_test_case)
                # 统计LLM生成的测试用例数
                stats["llm_generated_cases"] += 1
                # print(f"  📊 LLM生成测试用例计数: {stats['llm_generated_cases']}")
                is_llm_generated = True
            elif operation == "repair":
                # print(f"  🔧 使用LLM修复后的测试用例")
                next_pytorch_test_case = llm_result.get("pytorch_test_case", current_torch_test_case)
                next_mindspore_test_case = llm_result.get("mindspore_test_case", current_mindspore_test_case)
                # 统计LLM生成的测试用例数
                stats["llm_generated_cases"] += 1
                # print(f"  📊 LLM生成测试用例计数: {stats['llm_generated_cases']}")
                is_llm_generated = True
            else:
                next_pytorch_test_case = current_torch_test_case
                next_mindspore_test_case = current_mindspore_test_case
            
            # 转换LLM返回的测试用例格式（分别转换PyTorch和MindSpore的测试用例，共享张量数据）
            current_torch_test_case, current_mindspore_test_case = self._convert_llm_test_cases(next_pytorch_test_case, next_mindspore_test_case)
        
        # 修复问题1：如果最后一次迭代LLM生成了新用例（mutation或repair），需要执行这个新用例
        if len(case_results) > 0:
            last_iteration = case_results[-1]
            last_operation = last_iteration["llm_operation"].get("operation", "skip")
            
            if last_operation in ["mutation", "repair"]:
                print(f"\n  🔄 执行最后一次LLM生成的用例...")
                
                try:
                    # 执行最后一次LLM生成的测试用例
                    # print(f"  📝 执行测试用例...")
                    execution_result = self.execute_test_case(operator_name, mindspore_api, current_torch_test_case, current_mindspore_test_case)
                    
                    # 简化打印：只显示关键结果状态
                    status = execution_result['status']
                    torch_ok = "✓" if execution_result['torch_success'] else "✗"
                    mindspore_ok = "✓" if execution_result['mindspore_success'] else "✗"
                    match_ok = "✓" if execution_result['results_match'] else "✗"
                    print(f"  📊 最终执行结果: status={status}, PyTorch={torch_ok}, MindSpore={mindspore_ok}, match={match_ok}")
                    
                    # 注释掉详细的执行结果打印（日志中会有记录）
                    # print(f"  📊 执行状态: {execution_result['status']}")
                    # print(f"  🅿️ PyTorch执行结果: {execution_result['torch_success']}")
                    # print(f"  🅰️ MindSpore执行结果: {execution_result['mindspore_success']}")
                    # print(f"  ❓ 结果一致: {execution_result['results_match']}")
                    
                    # 只在有错误时打印错误信息
                    if execution_result['torch_error']:
                        print(f"  ❌ PyTorch错误: {execution_result['torch_error'][:100]}..." if len(str(execution_result['torch_error'])) > 100 else f"  ❌ PyTorch错误: {execution_result['torch_error']}")
                    if execution_result['mindspore_error']:
                        print(f"  ❌ MindSpore错误: {execution_result['mindspore_error'][:100]}..." if len(str(execution_result['mindspore_error'])) > 100 else f"  ❌ MindSpore错误: {execution_result['mindspore_error']}")
                    # if execution_result['comparison_error']:
                    #     print(f"  ⚠️ 比较错误: {execution_result['comparison_error']}")
                    
                    # 统计两个框架都执行成功的用例数（这是LLM生成的用例）
                    if execution_result['torch_success'] and execution_result['mindspore_success']:
                        stats["successful_cases"] += 1
                        # print(f"  📊 LLM生成的成功执行用例计数: {stats['successful_cases']}")
                    
                    # 保存这次额外执行的结果
                    final_iteration_result = {
                        "iteration": len(case_results) + 1,
                        "torch_test_case": current_torch_test_case,
                        "mindspore_test_case": current_mindspore_test_case,
                        "execution_result": execution_result,
                        "llm_operation": {
                            "operation": "final_execution",
                            "reason": "执行最后一次LLM生成的用例"
                        },
                        "case_number": case_number,
                        "is_llm_generated": True
                    }
                    case_results.append(final_iteration_result)
                    
                except Exception as e:
                    print(f"  ❌ 执行最后一次LLM生成的用例时发生严重错误: {str(e)[:100]}..." if len(str(e)) > 100 else f"  ❌ 执行最后一次LLM生成的用例时发生严重错误: {e}")
                    # print(f"  ❌ 错误详情:")
                    # traceback.print_exc()
                    
                    # 即使出错也要记录这次尝试
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
                            "reason": "执行最后一次LLM生成的用例（发生严重错误）"
                        },
                        "case_number": case_number,
                        "is_llm_generated": True
                    }
                    case_results.append(final_iteration_result)
        
        print(f"\n  {'─'*76}")
        print(f"  ✅ 测试用例 {case_number} 完成，共执行 {len(case_results)} 次迭代")
        print(f"  {'─'*76}")
        
        return case_results
    
    def _convert_llm_test_cases(self, pytorch_test_case: Dict[str, Any], mindspore_test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        将LLM返回的PyTorch和MindSpore测试用例转换为可执行格式
        确保两个框架使用相同的张量数据，但允许其他参数不同
        
        Args:
            pytorch_test_case: LLM返回的PyTorch测试用例
            mindspore_test_case: LLM返回的MindSpore测试用例
        
        Returns:
            (转换后的PyTorch测试用例, 转换后的MindSpore测试用例)
        """
        # 简化打印：注释掉详细的转换过程信息（日志中会有记录）
        # print(f"    🔄 转换LLM返回的测试用例格式...")
        
        # 第一步：收集所有需要生成张量的参数名，并生成共享的numpy数组
        shared_tensors = {}  # 存储共享的numpy数组
        
        # 找出所有张量参数（在pytorch或mindspore测试用例中）
        all_keys = set(pytorch_test_case.keys()) | set(mindspore_test_case.keys())
        
        for key in all_keys:
            if key == "api":
                continue
            
            # 检查是否是张量描述
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
                # 生成共享的numpy数组
                # print(f"      - {key}: 张量描述 shape={tensor_desc.get('shape')}, dtype={tensor_desc.get('dtype')}")
                numpy_array = self.generate_numpy_data(tensor_desc)
                shared_tensors[key] = numpy_array
                # print(f"        生成共享numpy数组: shape={numpy_array.shape}, dtype={numpy_array.dtype}")
        
        # 第二步：分别构建PyTorch和MindSpore的测试用例
        converted_pytorch = {}
        converted_mindspore = {}
        
        # print(f"    📦 构建PyTorch测试用例:")
        for key, value in pytorch_test_case.items():
            if key in shared_tensors:
                converted_pytorch[key] = shared_tensors[key]
                # print(f"      - {key}: 使用共享张量")
            else:
                converted_pytorch[key] = value
                # print(f"      - {key}: 使用值={value}")
        
        # print(f"    📦 构建MindSpore测试用例:")
        for key, value in mindspore_test_case.items():
            if key in shared_tensors:
                converted_mindspore[key] = shared_tensors[key]
                # print(f"      - {key}: 使用共享张量")
            else:
                converted_mindspore[key] = value
                # print(f"      - {key}: 使用值={value}")
        
        return converted_pytorch, converted_mindspore
    
    def save_results(self, operator_name: str, results: List[Dict[str, Any]], stats: Dict[str, int] = None):
        """
        保存测试结果到JSON文件
        
        Args:
            operator_name: 算子名称
            results: 测试结果列表
            stats: 统计信息（LLM生成的用例数和成功执行的用例数）
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_enhanced_{operator_name.replace('.', '_')}_{timestamp}.json"
        filepath = os.path.join(self.result_dir, filename)
        
        # 准备输出数据（移除numpy数组以便JSON序列化）
        output_results = []
        for result in results:
            output_result = copy.deepcopy(result)
            
            # 简化测试用例中的numpy数组（处理旧格式：test_case）
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
            
            # 简化测试用例中的numpy数组（处理新格式：torch_test_case 和 mindspore_test_case）
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
        
        print(f"💾 结果已保存到: {filepath}")
    
    def close(self):
        """关闭MongoDB连接"""
        self.client.close()


def main():
    """
    主函数
    
    提供两种测试模式：
    1. 单个算子测试（注释掉批量测试代码，取消注释单个测试代码）
    2. 批量测试所有算子（当前模式）
    """
    # ==================== 测试参数配置 ====================
    max_iterations = 3  # 每个测试用例的最大迭代次数
    num_test_cases = 3  # 每个算子要测试的用例数量
    
    # 批量测试范围配置（仅用于模式2）
    # 设置为 None 表示测试所有算子
    # 设置为 (start, end) 表示测试第 start 到第 end 个算子（包含start和end，从1开始计数）
    # 示例:
    #   operator_range = None          # 测试所有算子
    #   operator_range = (1, 10)       # 测试第1到第10个算子
    #   operator_range = (10, 20)      # 测试第10到第20个算子
    #   operator_range = (50, 100)     # 测试第50到第100个算子
    operator_range = (251, 465)
    # ====================================================
    
    # ==================== 模式1: 单个算子测试 ====================
    # 如果想测试单个算子，请取消下面代码的三引号注释，并注释掉"模式2"的代码
    """
    operator_name = "torch.nn.Dropout2d"  # 待测试的算子
    
    print("="*80)
    print("基于LLM的PyTorch与MindSpore算子比较测试框架")
    print("="*80)
    print(f"📌 测试算子: {operator_name}")
    print(f"📌 每个用例的迭代次数: {max_iterations}")
    print(f"📌 测试用例数量: {num_test_cases}")
    print("="*80)
    
    # 初始化比较器
    comparator = LLMEnhancedComparator()
    
    try:
        # 使用LLM增强的方式测试算子
        results, stats = comparator.llm_enhanced_test_operator(
            operator_name, 
            max_iterations=max_iterations,
            num_test_cases=num_test_cases
        )
        
        # 保存结果
        comparator.save_results(operator_name, results, stats)
        
        # 打印详细摘要
        print("\n" + "="*80)
        print("📊 测试摘要")
        print("="*80)
        print(f"算子名称: {operator_name}")
        print(f"总迭代次数: {len(results)}")
        
        # 按测试用例分组统计
        case_groups = {}
        for result in results:
            case_num = result.get("case_number", 0)
            if case_num not in case_groups:
                case_groups[case_num] = []
            case_groups[case_num].append(result)
        
        print(f"\n共测试 {len(case_groups)} 个测试用例：")
        for case_num in sorted(case_groups.keys()):
            case_results = case_groups[case_num]
            print(f"\n测试用例 {case_num} ({len(case_results)} 次迭代):")
            for i, result in enumerate(case_results):
                exec_result = result["execution_result"]
                llm_op = result.get("llm_operation", {})
                print(f"  迭代 {i+1}:")
                print(f"    - 执行状态: {exec_result['status']}")
                print(f"    - PyTorch成功: {exec_result['torch_success']}")
                print(f"    - MindSpore成功: {exec_result['mindspore_success']}")
                print(f"    - 结果一致: {exec_result['results_match']}")
                print(f"    - LLM操作: {llm_op.get('operation', 'N/A')}")
        
    finally:
        # 关闭连接
        comparator.close()
        print("\n✅ 程序执行完成")
    """
    # ==================== 模式1结束 ====================
    
    # ==================== 模式2: 批量测试所有算子 ====================
    # 如果想批量测试所有算子，请取消下面代码的三引号注释，并注释掉"模式1"的代码
    
    print("="*80)
    print("基于LLM的PyTorch与MindSpore算子批量比较测试框架")
    print("="*80)
    print(f"📌 每个算子的迭代次数: {max_iterations}")
    print(f"📌 每个算子的测试用例数: {num_test_cases}")
    if operator_range is not None:
        print(f"📌 测试范围: 第 {operator_range[0]} 到第 {operator_range[1]} 个算子")
    else:
        print(f"📌 测试范围: 所有算子")
    print("="*80)
    
    # 初始化比较器
    comparator = LLMEnhancedComparator()
    
    # 记录开始时间
    import time
    start_time = time.time()
    start_datetime = datetime.now()
    
    try:
        # 获取数据库中所有的PyTorch算子
        print("\n🔍 正在获取数据库中的所有算子...")
        all_operators = list(comparator.collection.find({}, {"api": 1}))
        all_operator_names = [doc["api"] for doc in all_operators if "api" in doc]
        
        print(f"✅ 数据库中共有 {len(all_operator_names)} 个算子")
        
        # 根据 operator_range 过滤算子
        if operator_range is not None:
            start_idx, end_idx = operator_range
            # 转换为0-based索引
            start_idx = max(1, start_idx) - 1
            end_idx = min(len(all_operator_names), end_idx)
            operator_names = all_operator_names[start_idx:end_idx]
            print(f"📌 测试范围: 第 {start_idx + 1} 到第 {end_idx} 个算子")
            print(f"📋 将测试 {len(operator_names)} 个算子")
        else:
            operator_names = all_operator_names
            print(f"📋 将测试所有 {len(operator_names)} 个算子")
        
        # 过滤掉"无对应实现"的算子（在映射表中检查）
        print(f"\n🔍 过滤无 MindSpore 对应实现的算子...")
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
        
        print(f"✅ 过滤完成: 原有 {original_count} 个算子，跳过 {skipped_count} 个，剩余 {len(operator_names)} 个")
        if skipped_operators:
            print(f"⏭️ 跳过的算子（前10个）: {', '.join([f'{op}({reason})' for op, reason in skipped_operators[:10]])}{'...' if len(skipped_operators) > 10 else ''}")
        
        print(f"📋 算子列表: {', '.join(operator_names[:10])}{'...' if len(operator_names) > 10 else ''}\n")
        
        # 统计所有算子的测试结果
        all_operators_summary = []
        
        # 创建总日志文件
        batch_log_file = os.path.join(comparator.result_dir, f"batch_test_log_{start_datetime.strftime('%Y%m%d_%H%M%S')}.txt")
        log_file = open(batch_log_file, 'w', encoding='utf-8')
        
        # 写入日志头部
        log_file.write("="*80 + "\n")
        log_file.write("批量测试总日志\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"测试配置:\n")
        log_file.write(f"  - 每个算子的迭代次数: {max_iterations}\n")
        log_file.write(f"  - 每个算子的测试用例数: {num_test_cases}\n")
        log_file.write(f"  - 数据库总算子数: {len(all_operator_names)}\n")
        if operator_range is not None:
            log_file.write(f"  - 测试范围: 第 {operator_range[0]} 到第 {operator_range[1]} 个\n")
        log_file.write(f"  - 跳过的无对应实现算子数: {skipped_count}\n")
        log_file.write(f"  - 实际测试算子数: {len(operator_names)}\n")
        log_file.write("="*80 + "\n\n")
        log_file.flush()
        
        # 依次测试每个算子
        for idx, operator_name in enumerate(operator_names, 1):
            print("\n" + "🔷"*40)
            print(f"🎯 [{idx}/{len(operator_names)}] 开始测试算子: {operator_name}")
            print("🔷"*40)
            
            try:
                # 使用LLM增强的方式测试算子
                results, stats = comparator.llm_enhanced_test_operator(
                    operator_name, 
                    max_iterations=max_iterations,
                    num_test_cases=num_test_cases
                )
                
                # 保存结果
                if results:
                    comparator.save_results(operator_name, results, stats)
                    
                    # 记录摘要信息
                    all_operators_summary.append({
                        "operator": operator_name,
                        "total_iterations": len(results),
                        "llm_generated_cases": stats.get("llm_generated_cases", 0),
                        "successful_cases": stats.get("successful_cases", 0),
                        "status": "completed"
                    })
                    
                    print(f"\n✅ 算子 {operator_name} 测试完成")
                    print(f"   - 总迭代次数: {len(results)}")
                    print(f"   - LLM生成用例数: {stats.get('llm_generated_cases', 0)}")
                    print(f"   - 成功执行用例数: {stats.get('successful_cases', 0)}")
                    
                    # 写入日志
                    log_file.write(f"[{idx}/{len(operator_names)}] {operator_name}\n")
                    log_file.write(f"  状态: ✅ 完成\n")
                    log_file.write(f"  总迭代次数: {len(results)}\n")
                    log_file.write(f"  LLM生成用例数: {stats.get('llm_generated_cases', 0)}\n")
                    log_file.write(f"  成功执行用例数: {stats.get('successful_cases', 0)}\n")
                    if stats.get('llm_generated_cases', 0) > 0:
                        success_rate = (stats.get('successful_cases', 0) / stats.get('llm_generated_cases', 0)) * 100
                        log_file.write(f"  成功率: {success_rate:.2f}%\n")
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
                    print(f"\n⚠️ 算子 {operator_name} 无测试结果")
                    
                    # 写入日志
                    log_file.write(f"[{idx}/{len(operator_names)}] {operator_name}\n")
                    log_file.write(f"  状态: ⚠️ 无结果\n\n")
                    log_file.flush()
                    
            except Exception as e:
                print(f"\n❌ 算子 {operator_name} 测试失败: {e}")
                all_operators_summary.append({
                    "operator": operator_name,
                    "total_iterations": 0,
                    "llm_generated_cases": 0,
                    "successful_cases": 0,
                    "status": "failed",
                    "error": str(e)
                })
                
                # 写入日志
                log_file.write(f"[{idx}/{len(operator_names)}] {operator_name}\n")
                log_file.write(f"  状态: ❌ 失败\n")
                log_file.write(f"  错误: {str(e)}\n\n")
                log_file.flush()
                
                # 继续测试下一个算子
                continue
        
        # 计算运行时间
        end_time = time.time()
        end_datetime = datetime.now()
        total_duration = end_time - start_time
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)
        
        # 打印总体摘要
        print("\n" + "="*80)
        print("📊 批量测试总体摘要")
        print("="*80)
        print(f"总算子数: {len(operator_names)}")
        
        completed_count = sum(1 for s in all_operators_summary if s["status"] == "completed")
        failed_count = sum(1 for s in all_operators_summary if s["status"] == "failed")
        no_results_count = sum(1 for s in all_operators_summary if s["status"] == "no_results")
        
        print(f"✅ 成功完成: {completed_count}")
        print(f"❌ 测试失败: {failed_count}")
        print(f"⚠️ 无结果: {no_results_count}")
        
        total_llm_cases = sum(s["llm_generated_cases"] for s in all_operators_summary)
        total_successful_cases = sum(s["successful_cases"] for s in all_operators_summary)
        total_iterations = sum(s["total_iterations"] for s in all_operators_summary)
        
        print(f"\n📈 统计数据:")
        print(f"   - LLM生成的测试用例总数: {total_llm_cases}")
        print(f"   - 成功执行的用例总数: {total_successful_cases}")
        if total_llm_cases > 0:
            success_rate = (total_successful_cases / total_llm_cases) * 100
            print(f"   - 成功执行占比: {success_rate:.2f}%")
        print(f"   - 总迭代次数: {total_iterations}")
        print(f"\n⏱️ 运行时间: {hours}小时 {minutes}分钟 {seconds}秒")
        
        # 写入日志统计信息
        log_file.write("="*80 + "\n")
        log_file.write("总体统计\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"总运行时间: {hours}小时 {minutes}分钟 {seconds}秒 ({total_duration:.2f}秒)\n\n")
        
        log_file.write("算子测试结果:\n")
        log_file.write(f"  - 总算子数: {len(operator_names)}\n")
        log_file.write(f"  - 成功完成: {completed_count} ({completed_count/len(operator_names)*100:.2f}%)\n")
        log_file.write(f"  - 测试失败: {failed_count} ({failed_count/len(operator_names)*100:.2f}%)\n")
        log_file.write(f"  - 无结果: {no_results_count} ({no_results_count/len(operator_names)*100:.2f}%)\n\n")
        
        log_file.write("LLM生成用例统计:\n")
        log_file.write(f"  - LLM生成的测试用例总数: {total_llm_cases}\n")
        log_file.write(f"  - 两个框架都成功执行的用例数: {total_successful_cases}\n")
        if total_llm_cases > 0:
            success_rate = (total_successful_cases / total_llm_cases) * 100
            log_file.write(f"  - 成功执行占比: {success_rate:.2f}%\n")
        log_file.write(f"  - 总迭代次数: {total_iterations}\n")
        if completed_count > 0:
            avg_llm_cases = total_llm_cases / completed_count
            avg_successful = total_successful_cases / completed_count
            log_file.write(f"  - 平均每个算子LLM生成用例数: {avg_llm_cases:.2f}\n")
            log_file.write(f"  - 平均每个算子成功执行用例数: {avg_successful:.2f}\n")
        
        log_file.write("\n" + "="*80 + "\n")
        log_file.write("详细结果请查看各算子的单独日志文件\n")
        log_file.write("="*80 + "\n")
        log_file.close()
        
        print(f"\n💾 总日志已保存到: {batch_log_file}")
        
        # 保存总体摘要到JSON文件
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
        
        print(f"💾 JSON摘要已保存到: {summary_file}")
        
    finally:
        # 关闭连接
        comparator.close()
        print("\n✅ 批量测试程序执行完成")
    
    # ==================== 模式2结束 ====================


if __name__ == "__main__":
    main()
