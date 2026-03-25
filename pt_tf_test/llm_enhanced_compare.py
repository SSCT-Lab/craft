# ./pt_tf_test/llm_enhanced_compare.py
"""
基于LLM的PyTorch与TensorFlow算子比较测试框架
使用大模型进行测试用例修复和变异，提高用例可用性和覆盖率
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

# 添加项目根目录到路径，以便导入 component 模块
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from component.doc.doc_crawler_factory import get_doc_content

# ==================== 常量定义 ====================
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
        初始化基于LLM的PyTorch和TensorFlow比较器
        
        Args:
            mongo_uri: MongoDB连接URI
            db_name: 数据库名称
            key_path: API key文件路径
            model: LLM模型名称
            print_lock: 打印锁（用于并发时线程安全输出）
            llm_workers: LLM并发调用线程数
        """
        self.model = model
        self.print_lock = print_lock or Lock()
        self.llm_workers = max(1, int(llm_workers))
        self.execution_lock = RLock()
        self.stats_lock = Lock()

        # MongoDB连接
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["argVS"]
        
        # 初始化LLM客户端（阿里千问大模型）
        # 优先从指定key文件读取密钥，否则使用环境变量
        api_key = self._load_api_key(key_path)
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 加载API映射表
        self.api_mapping = self.load_api_mapping()
        
        # 创建结果存储目录（在 pt_tf_test 目录下）
        self.result_dir = os.path.join(ROOT_DIR, "pt_tf_test", "pt_tf_log_1")
        os.makedirs(self.result_dir, exist_ok=True)
        self._safe_print(f"📁 结果存储目录: {self.result_dir}")
        
        # 固定随机种子以确保可重复性
        self.random_seed = 42
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        
        # 已废弃的PyTorch算子列表
        self.deprecated_torch_apis = {
            "torch.symeig": "已在PyTorch 1.9版本中移除，请使用torch.linalg.eigh替代"
        }

    def _safe_print(self, msg: str, end: str = "\n"):
        """线程安全的打印"""
        with self.print_lock:
            print(msg, end=end, flush=True)
    
    def _load_api_key(self, key_path: str = DEFAULT_KEY_PATH) -> str:
        """
        加载阿里云 API 密钥
        
        优先从指定的文件读取，如果文件不存在则使用环境变量 DASHSCOPE_API_KEY
        
        Args:
            key_path: API key文件路径
        
        Returns:
            API 密钥字符串
        """
        if not os.path.isabs(key_path):
            key_file = os.path.join(ROOT_DIR, key_path)
        else:
            key_file = key_path
        
        # 优先从文件读取
        if os.path.exists(key_file):
            try:
                with open(key_file, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
                if api_key:
                    self._safe_print(f"✅ 从文件加载 API 密钥: {key_file}")
                    return api_key
            except Exception as e:
                self._safe_print(f"⚠️ 读取密钥文件失败: {e}")
        
        # 回退到环境变量
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            self._safe_print(f"✅ 从环境变量加载 API 密钥: DASHSCOPE_API_KEY")
            return api_key
        
        # 都没有找到
        self._safe_print("❌ 未找到 API 密钥，请确保 aliyun.key 文件存在或设置 DASHSCOPE_API_KEY 环境变量")
        return ""
    
    def load_api_mapping(self) -> Dict[str, Dict[str, str]]:
        """加载PyTorch到TensorFlow的API映射表"""
        # 使用更新后的映射文件
        mapping_file = os.path.join(ROOT_DIR, "component", "data", "api_mappings_final.csv")
        try:
            df = pd.read_csv(mapping_file)
            mapping = {}
            
            for _, row in df.iterrows():
                # 新映射文件的列名是 pytorch-api 和 tensorflow-api
                pt_api = str(row["pytorch-api"]).strip()
                tf_api = str(row["tensorflow-api"]).strip()
                mapping[pt_api] = {"tf_api": tf_api, "note": ""}
            
            self._safe_print(f"✅ 成功加载API映射表，共 {len(mapping)} 条映射")
            self._safe_print(f"📄 映射文件: {mapping_file}")
            return mapping
        except Exception as e:
            self._safe_print(f"❌ 加载API映射表失败: {e}")
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
    #         tf_func_api = f"tf.nn.functional.{func_name}"
    #         
    #         return torch_func_api, tf_func_api
    #     
    #     return None, None
    
    def convert_api_name(self, torch_api: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        将PyTorch API转换为TensorFlow API
        
        完全基于 api_mappings_final.csv 映射表查找，不再进行手动名称转换。
        
        Returns:
            (转换后的PyTorch API, 转换后的TensorFlow API, 映射方法说明)
            - 如果映射表中找到且有有效的 TensorFlow API → 返回映射后的 API
            - 如果映射表中找到但值为 "无对应实现" → 返回 (torch_api, None, "无对应实现")
            - 如果映射表中找不到该 API → 返回 (torch_api, None, "映射表中未找到")
        """
        # 查映射表
        if torch_api in self.api_mapping:
            tf_api = self.api_mapping[torch_api]["tf_api"]
            
            # 检查是否为 "无对应实现"
            if tf_api == "无对应实现" or tf_api == "NONE" or not tf_api:
                return torch_api, None, "无对应实现"
            else:
                return torch_api, tf_api, "映射表"
        
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
        """转换参数名"""
        key_mapping = {
            "input": "x",
            "other": "y",
        }
        return key_mapping.get(key, key)
    
    def should_skip_param(self, key: str, tensorflow_api: str) -> bool:
        """判断是否应该跳过某个参数"""
        common_skip_params = ["layout", "requires_grad", "out"]
        skip_params = {
            # 可以根据需要添加特定API的跳过参数
        }
        
        if key in common_skip_params:
            return True
        
        if tensorflow_api in skip_params:
            return key in skip_params[tensorflow_api]
        
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
            
            # 保持静默：未识别dtype时自动回退到默认值
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
        """准备共享的numpy数据，确保PyTorch和TensorFlow使用相同的输入"""
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
                # 转换工作将在prepare_arguments_torch/tf中完成
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
    
    def convert_to_tensor_tensorflow(self, data: Any, numpy_data: np.ndarray = None) -> tf.Tensor:
        """转换数据为TensorFlow张量"""
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
    
    def prepare_arguments_tensorflow(self, test_case: Dict[str, Any], tensorflow_api: str) -> Tuple[List[Any], Dict[str, Any]]:
        """
        为TensorFlow准备参数
        
        注意：
        1. 对于tf.where等函数，参数也需要按顺序作为位置参数传递
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
                        args.append(self.convert_to_tensor_tensorflow(None, numpy_data))
                    elif isinstance(item, list):
                        # 嵌套列表，递归处理
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
        
        # 按顺序处理位置参数：condition, x/input, y/other
        positional_params = ["condition", "x", "y", "input", "other"]
        
        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, np.ndarray):
                    args.append(self.convert_to_tensor_tensorflow(None, value))
                else:
                    # 标量值直接添加
                    args.append(value)
        
        # 处理其他参数（作为关键字参数）
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
        """比较两个张量是否相等"""
        try:
            # 转换为numpy进行比较
            if hasattr(torch_result, 'detach'):
                torch_np = torch_result.detach().cpu().numpy()
            else:
                torch_np = np.array(torch_result)
            
            if hasattr(tensorflow_result, 'numpy'):
                tensorflow_np = tensorflow_result.numpy()
            else:
                tensorflow_np = np.array(tensorflow_result)
            
            # 检查形状
            if torch_np.shape != tensorflow_np.shape:
                return False, f"形状不匹配: PyTorch {torch_np.shape} vs TensorFlow {tensorflow_np.shape}"
            
            # 检查dtype是否为布尔类型
            if torch_np.dtype == np.bool_ or tensorflow_np.dtype == np.bool_:
                if np.array_equal(torch_np, tensorflow_np):
                    return True, "布尔值匹配"
                else:
                    diff_count = np.sum(torch_np != tensorflow_np)
                    return False, f"布尔值不匹配，差异数量: {diff_count}"
            
            # 检查数值
            if np.allclose(torch_np, tensorflow_np, atol=tolerance, rtol=tolerance, equal_nan=True):
                return True, "数值匹配"
            else:
                max_diff = np.max(np.abs(torch_np - tensorflow_np))
                return False, f"数值不匹配，最大差异: {max_diff}"
        
        except Exception as e:
            return False, f"比较过程出错: {str(e)}"
    
    def execute_test_case(self, torch_api: str, tensorflow_api: str, torch_test_case: Dict[str, Any], tensorflow_test_case: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行单个测试用例
        
        Args:
            torch_api: PyTorch API名称
            tensorflow_api: TensorFlow API名称
            torch_test_case: PyTorch测试用例（包含参数信息）
            tensorflow_test_case: TensorFlow测试用例（包含参数信息）
        
        Returns:
            执行结果字典
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
        
        # 如果没有提供tensorflow_test_case，则使用torch_test_case（向后兼容）
        if tensorflow_test_case is None:
            tensorflow_test_case = torch_test_case
        
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
        
        # 测试TensorFlow
        tensorflow_result = None
        try:
            tensorflow_func = self.get_operator_function(tensorflow_api, "tensorflow")
            if tensorflow_func is None:
                result["tensorflow_error"] = f"TensorFlow算子 {tensorflow_api} 未找到"
            else:
                args, kwargs = self.prepare_arguments_tensorflow(tensorflow_test_case, tensorflow_api)
                
                if is_class_api:
                    # 对于类算子，需要先实例化，然后调用
                    # 从 kwargs中提取初始化参数（非x/input参数）
                    init_kwargs = {k: v for k, v in kwargs.items() if k not in ['x', 'input']}
                    # 实例化类
                    tensorflow_instance = tensorflow_func(**init_kwargs)
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
                    tensorflow_result = tensorflow_instance(input_data)
                else:
                    # 对于函数算子，直接调用
                    tensorflow_result = tensorflow_func(*args, **kwargs)
                
                result["tensorflow_success"] = True
                result["tensorflow_shape"] = list(tensorflow_result.shape) if hasattr(tensorflow_result, 'shape') else None
                result["tensorflow_dtype"] = str(tensorflow_result.dtype) if hasattr(tensorflow_result, 'dtype') else None
        except Exception as e:
            result["tensorflow_error"] = str(e)
            result["tensorflow_traceback"] = traceback.format_exc()
        
        # 比较结果
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
        """顺序执行算子（通过锁保证执行不并发）"""
        with self.execution_lock:
            return self.execute_test_case(torch_api, tensorflow_api, torch_test_case, tensorflow_test_case)
    
    def _fetch_api_docs(self, torch_api: str, tensorflow_api: str) -> Tuple[str, str]:
        """
        爬取PyTorch和TensorFlow的API文档
        
        Args:
            torch_api: PyTorch API名称
            tensorflow_api: TensorFlow API名称
        
        Returns:
            (PyTorch文档内容, TensorFlow文档内容)
        """
        # 文档有效性判断的最小长度阈值
        MIN_DOC_LENGTH = 300
        
        torch_doc = ""
        tensorflow_doc = ""
        
        try:
            self._safe_print(f"    📖 正在爬取 PyTorch 文档: {torch_api}")
            torch_doc = get_doc_content(torch_api, "pytorch")
            # 判断文档是否有效：1. 内容不为空 2. 不包含错误提示 3. 长度超过阈值
            if (torch_doc 
                and "Unable" not in torch_doc 
                and "not supported" not in torch_doc
                and len(torch_doc.strip()) > MIN_DOC_LENGTH):
                # 截断过长的文档以节省token
                if len(torch_doc) > 3000:
                    torch_doc = torch_doc[:3000] + "\n... (doc truncated)"
                self._safe_print(f"    ✅ PyTorch 文档获取成功 ({len(torch_doc)} 字符)")
            else:
                doc_len = len(torch_doc.strip()) if torch_doc else 0
                torch_doc = f"Unable to fetch documentation for {torch_api} (length: {doc_len}, min required: {MIN_DOC_LENGTH})"
                self._safe_print(f"    ⚠️ PyTorch 文档无效或过短")
        except Exception as e:
            torch_doc = f"Failed to fetch documentation: {str(e)}"
            self._safe_print(f"    ❌ PyTorch 文档爬取失败: {e}")
        
        try:
            self._safe_print(f"    📖 正在爬取 TensorFlow 文档: {tensorflow_api}")
            tensorflow_doc = get_doc_content(tensorflow_api, "tensorflow")
            # 判断文档是否有效：1. 内容不为空 2. 不包含错误提示 3. 长度超过阈值
            if (tensorflow_doc 
                and "Unable" not in tensorflow_doc 
                and "not supported" not in tensorflow_doc
                and len(tensorflow_doc.strip()) > MIN_DOC_LENGTH):
                # 截断过长的文档以节省token
                if len(tensorflow_doc) > 3000:
                    tensorflow_doc = tensorflow_doc[:3000] + "\n... (doc truncated)"
                self._safe_print(f"    ✅ TensorFlow 文档获取成功 ({len(tensorflow_doc)} 字符)")
            else:
                doc_len = len(tensorflow_doc.strip()) if tensorflow_doc else 0
                tensorflow_doc = f"Unable to fetch documentation for {tensorflow_api} (length: {doc_len}, min required: {MIN_DOC_LENGTH})"
                self._safe_print(f"    ⚠️ TensorFlow 文档无效或过短")
        except Exception as e:
            tensorflow_doc = f"Failed to fetch documentation: {str(e)}"
            self._safe_print(f"    ❌ TensorFlow 文档爬取失败: {e}")
        
        return torch_doc, tensorflow_doc
        
        return torch_doc, tensorflow_doc
    
    def _build_llm_prompt(self, execution_result: Dict[str, Any], torch_test_case: Dict[str, Any], tensorflow_test_case: Dict[str, Any], torch_doc: str = "", tensorflow_doc: str = "") -> str:
        """构建LLM的提示词"""
        torch_api = execution_result.get("torch_api", "")
        tensorflow_api = execution_result.get("tensorflow_api", "")
        status = execution_result.get("status", "")
        torch_success = execution_result.get("torch_success", False)
        tensorflow_success = execution_result.get("tensorflow_success", False)
        results_match = execution_result.get("results_match", False)
        torch_error = execution_result.get("torch_error", "")
        tensorflow_error = execution_result.get("tensorflow_error", "")
        comparison_error = execution_result.get("comparison_error", "")
        
        # 简化PyTorch测试用例以减少token消耗
        simplified_torch_test_case = {}
        for key, value in torch_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_torch_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_torch_test_case[key] = value
        
        # 简化TensorFlow测试用例以减少token消耗
        simplified_tensorflow_test_case = {}
        for key, value in tensorflow_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_tensorflow_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_tensorflow_test_case[key] = value
        
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
        
        # 构建TensorFlow参数示例
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
        
        # 构建API文档部分
        doc_section = ""
        if torch_doc or tensorflow_doc:
            doc_section = "\n## 官方API文档参考\n\n"
            if torch_doc:
                doc_section += f"### PyTorch {torch_api} 文档\n```\n{torch_doc}\n```\n\n"
            if tensorflow_doc:
                doc_section += f"### TensorFlow {tensorflow_api} 文档\n```\n{tensorflow_doc}\n```\n\n"
        
        prompt = f"""请分析以下算子测试用例在PyTorch和TensorFlow框架中的执行结果，并根据结果进行测试用例的修复或变异（fuzzing）。

## 测试信息
- **PyTorch API**: {torch_api}
- **TensorFlow API**: {tensorflow_api}
{doc_section}
## 执行结果
- **执行状态**: {status}
- **PyTorch执行成功**: {torch_success}
- **TensorFlow执行成功**: {tensorflow_success}
- **结果是否一致**: {results_match}

## 错误信息
- **PyTorch错误**: {torch_error if torch_error else "无"}
- **TensorFlow错误**: {tensorflow_error if tensorflow_error else "无"}
- **比较错误**: {comparison_error if comparison_error else "无"}

## 原始测试用例

### PyTorch测试用例
```json
{json.dumps(simplified_torch_test_case, indent=2, ensure_ascii=False)}
```

### TensorFlow测试用例
```json
{json.dumps(simplified_tensorflow_test_case, indent=2, ensure_ascii=False)}
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
  "tensorflow_test_case": {{
    "api": "{tensorflow_api}",
{tf_param_example_str}
  }}
}}

**重要说明**：
1. operation的值必须是 "mutation"、"repair" 或 "skip" 之一
2. 张量参数必须使用 {{"shape": [...], "dtype": "..."}} 格式
3. 标量参数直接使用数值，例如 "y": 0
4. 构造两个框架的用例时必须保证输入相同(必要时进行张量形状的转换，如NHWC与NCHW转换)、参数在语义上严格对应。比如“pad”参数，PyTorch中padding=1并不严格对应TensorFlow中的padding='SAME'，必须进行等价的pad操作。
5. PyTorch和TensorFlow的测试用例可以有参数名差异（如input vs x）、参数值差异或者参数数量的差异，只要保证理论上输出相同就行。
6. 如果这个算子找不到官方文档，请判断是否是因为该算子不存在或者已经从PyTorch或者TensorFlow的当前版本移除了，如果是这样，请将 operation 设置为 "skip"，不需要尝试修复。
7. 测试用例变异时可适当探索一些极端情况，例如：空张量（shape包含0）、单元素张量（shape=[1]或[]）、高维张量、超大张量、不同数据类型（int、float、bool）、边界值等，以提高测试覆盖率和发现潜在bug
8. 请仔细阅读官方API文档，确保参数名称、参数类型、参数取值范围等与文档一致
"""
        return prompt
    
    def call_llm_for_repair_or_mutation(self, execution_result: Dict[str, Any], torch_test_case: Dict[str, Any], tensorflow_test_case: Dict[str, Any], torch_doc: str = "", tensorflow_doc: str = "") -> Dict[str, Any]:
        """调用LLM进行测试用例修复或变异"""
        prompt = self._build_llm_prompt(execution_result, torch_test_case, tensorflow_test_case, torch_doc, tensorflow_doc)
        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个深度学习框架测试专家，精通PyTorch和TensorFlow框架的API差异。你的任务是根据测试用例的执行结果，判断是否需要修复或变异测试用例，并返回严格的JSON格式结果。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
            )
            
            raw_response = completion.choices[0].message.content.strip()
            
            # 添加1秒时间间隔，避免API调用过于频繁
            time.sleep(1)
            
            # 尝试解析JSON
            try:
                llm_result = json.loads(raw_response)
                return llm_result
            except json.JSONDecodeError as e:
                self._safe_print(f"    ⚠️ LLM返回的不是有效的JSON，尝试提取JSON内容...")
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    llm_result = json.loads(json_match.group())
                    return llm_result
                else:
                    return {
                        "operation": "skip",
                        "reason": f"LLM返回格式错误: {e}",
                        "pytorch_test_case": torch_test_case,
                        "tensorflow_test_case": tensorflow_test_case
                    }
        
        except Exception as e:
            self._safe_print(f"    ❌ 调用LLM失败: {e}")
            return {
                "operation": "skip",
                "reason": f"LLM调用失败: {e}",
                "pytorch_test_case": torch_test_case,
                "tensorflow_test_case": tensorflow_test_case
            }
    
    def get_num_test_cases_from_document(self, document: Dict[str, Any]) -> int:
        """获取文档中的测试用例数量"""
        max_len = 0
        # 遍历文档中的所有字段，找出列表类型字段的最大长度
        for key, value in document.items():
            if key not in ["_id", "api"] and isinstance(value, list):
                max_len = max(max_len, len(value))
        return max_len if max_len > 0 else 1
    
    def llm_enhanced_test_operator(self, operator_name: str, max_iterations: int = 3,
                                   num_test_cases: int = None,
                                   num_workers: int = DEFAULT_LLM_WORKERS) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        使用LLM增强的方式测试单个算子
        
        Args:
            operator_name: 算子名称，例如 "torch.where"
            max_iterations: 每个测试用例的最大迭代次数
            num_test_cases: 要测试的用例数量，None表示测试所有用例
            num_workers: LLM并发调用线程数（算子执行仍为顺序）
        
        Returns:
            (所有测试用例的所有迭代结果列表, 统计信息字典)
        """
        self._safe_print(f"\n{'='*80}")
        self._safe_print(f"🎯 开始测试算子: {operator_name}")
        self._safe_print(f"🔄 每个用例的最大迭代次数: {max_iterations}")
        self._safe_print(f"{'='*80}\n")
        
        # 初始化统计计数器
        stats = {
            "llm_generated_cases": 0,      # LLM生成的测试用例总数
            "successful_cases": 0           # 两个框架都执行成功的测试用例数
        }
        
        # 从MongoDB获取算子的测试用例
        document = self.collection.find_one({"api": operator_name})
        if document is None:
            self._safe_print(f"❌ 未找到算子 {operator_name} 的测试用例")
            return [], stats
        
        # 获取测试用例总数
        total_cases = self.get_num_test_cases_from_document(document)
        
        # 确定实际要测试的用例数量
        if num_test_cases is None:
            num_test_cases = total_cases
        else:
            num_test_cases = min(num_test_cases, total_cases)
        
        # 获取转换后的PyTorch和TensorFlow API
        torch_api, tensorflow_api, mapping_method = self.convert_api_name(operator_name)
        if tensorflow_api is None:
            self._safe_print(f"❌ 算子 {operator_name} 无TensorFlow对应实现")
            return [], stats
        
        # 显示API映射信息
        if torch_api != operator_name:
            self._safe_print(f"✅ 原始 PyTorch API: {operator_name}")
            self._safe_print(f"✅ 转换后 PyTorch API: {torch_api}")
        else:
            self._safe_print(f"✅ PyTorch API: {torch_api}")
        self._safe_print(f"✅ TensorFlow API: {tensorflow_api}")
        self._safe_print(f"✅ 映射方法: {mapping_method}")
        self._safe_print(f"📋 将测试 {num_test_cases} 个用例 (LLM并发={num_workers}, 执行顺序)")
        
        all_results = []
        initial_cases = []
        for case_idx in range(num_test_cases):
            initial_test_case = self.prepare_shared_numpy_data(document, case_index=case_idx)
            initial_test_case["api"] = torch_api
            initial_cases.append((case_idx + 1, initial_test_case))

        if num_workers <= 1:
            for case_number, initial_test_case in initial_cases:
                self._safe_print(f"\n📋 用例 {case_number}/{num_test_cases}")
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
        self._safe_print("✅ 所有测试完成")
        self._safe_print(f"📊 共测试 {num_test_cases} 个用例，总计 {len(all_results)} 次迭代")
        self._safe_print(f"📊 LLM生成的测试用例数: {stats['llm_generated_cases']}")
        self._safe_print(f"📊 两个框架都执行成功的用例数: {stats['successful_cases']}")
        self._safe_print(f"{'='*80}\n")
        
        return all_results, stats
    
    def _test_single_case_with_iterations(self, operator_name: str, tensorflow_api: str, 
                                          initial_test_case: Dict[str, Any], 
                                          max_iterations: int,
                                          case_number: int,
                                          stats: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        对单个测试用例进行多轮迭代测试
        
        Args:
            operator_name: PyTorch算子名称
            tensorflow_api: TensorFlow算子名称
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
        # TensorFlow 需要创建副本并设置正确的 api
        current_torch_test_case = initial_test_case
        current_tensorflow_test_case = copy.deepcopy(initial_test_case)
        current_tensorflow_test_case["api"] = tensorflow_api  # 设置正确的 TensorFlow API
        
        # 标记当前用例是否为LLM生成的（第一次迭代是数据库原始用例）
        is_llm_generated = False
        
        # 预先爬取API文档（只爬取一次，后续迭代复用）
        self._safe_print(f"  📖 预先爬取API文档...")
        torch_doc, tensorflow_doc = self._fetch_api_docs(operator_name, tensorflow_api)
        
        # 开始迭代测试
        for iteration in range(max_iterations):
            source_type = "LLM" if is_llm_generated else "DB"
            self._safe_print(f"  🔄 迭代 {iteration + 1}/{max_iterations} ({source_type})", end="")
            
            # 执行测试用例
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
                    self._safe_print(f"    ❌ PyTorch错误: {err_short}...")
                if execution_result['tensorflow_error'] and not execution_result['tensorflow_success']:
                    err_short = str(execution_result['tensorflow_error'])[:100]
                    self._safe_print(f"    ❌ TensorFlow错误: {err_short}...")
                if execution_result['comparison_error']:
                    err_short = str(execution_result['comparison_error'])[:100]
                    self._safe_print(f"    ⚠️ 比较错误: {err_short}...")

                # 仅统计LLM生成的用例（不包括数据库原始用例）
                if is_llm_generated:
                    if execution_result['torch_success'] and execution_result['tensorflow_success']:
                        with self.stats_lock:
                            stats["successful_cases"] += 1

            except Exception as e:
                self._safe_print(f" | ❌ 严重错误: {str(e)[:80]}...")
                
                # 创建一个错误结果
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
            
            # 保存本次迭代结果
            iteration_result = {
                "iteration": iteration + 1,
                "torch_test_case": current_torch_test_case,
                "tensorflow_test_case": current_tensorflow_test_case,
                "execution_result": execution_result,
                "llm_operation": None,
                "is_llm_generated": is_llm_generated
            }
            
            # 调用LLM进行修复或变异（传入PyTorch和TensorFlow测试用例及API文档）
            try:
                llm_result = self.call_llm_for_repair_or_mutation(
                    execution_result,
                    current_torch_test_case,
                    current_tensorflow_test_case,
                    torch_doc,
                    tensorflow_doc
                )
            except Exception as e:
                self._safe_print(f"    ❌ LLM调用失败: {str(e)[:80]}...")
                
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
            reason_short = reason[:80]
            self._safe_print(f"    🤖 LLM: {operation} - {reason_short}")
            
            iteration_result["llm_operation"] = {
                "operation": operation,
                "reason": reason
            }
            
            # 添加测试用例编号信息
            iteration_result["case_number"] = case_number
            case_results.append(iteration_result)
            
            # 如果LLM建议跳过，则结束迭代
            if operation == "skip":
                break
            
            # 准备下一次迭代的测试用例
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
            
            # 转换LLM返回的测试用例格式（分别转换PyTorch和TensorFlow的测试用例，共享张量数据）
            current_torch_test_case, current_tensorflow_test_case = self._convert_llm_test_cases(next_pytorch_test_case, next_tensorflow_test_case)
        
        # 修复问题1：如果最后一次迭代LLM生成了新用例（mutation或repair），需要执行这个新用例
        if len(case_results) > 0:
            last_iteration = case_results[-1]
            last_operation = last_iteration["llm_operation"].get("operation", "skip")
            
            if last_operation in ["mutation", "repair"]:
                self._safe_print(f"  🔄 执行最终LLM用例", end="")

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
                        self._safe_print(f"    ❌ PyTorch错误: {err_short}...")
                    if execution_result['tensorflow_error'] and not execution_result['tensorflow_success']:
                        err_short = str(execution_result['tensorflow_error'])[:100]
                        self._safe_print(f"    ❌ TensorFlow错误: {err_short}...")
                    if execution_result['comparison_error']:
                        err_short = str(execution_result['comparison_error'])[:100]
                        self._safe_print(f"    ⚠️ 比较错误: {err_short}...")

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
                            "reason": "执行最后一次LLM生成的用例"
                        },
                        "case_number": case_number,
                        "is_llm_generated": True
                    }
                    case_results.append(final_iteration_result)

                except Exception as e:
                    self._safe_print(f"  ❌ 最终用例执行失败: {str(e)[:80]}...")
                    
                    # 即使出错也要记录这次尝试
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
                            "reason": "执行最后一次LLM生成的用例（发生严重错误）"
                        },
                        "case_number": case_number,
                        "is_llm_generated": True
                    }
                    case_results.append(final_iteration_result)
        
        self._safe_print(f"  ✅ 用例 {case_number} 完成，共 {len(case_results)} 次迭代")
        
        return case_results
    
    def _convert_llm_test_cases(self, pytorch_test_case: Dict[str, Any], tensorflow_test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        将LLM返回的PyTorch和TensorFlow测试用例转换为可执行格式
        确保两个框架使用相同的张量数据，但允许其他参数不同
        
        Args:
            pytorch_test_case: LLM返回的PyTorch测试用例
            tensorflow_test_case: LLM返回的TensorFlow测试用例
        
        Returns:
            (转换后的PyTorch测试用例, 转换后的TensorFlow测试用例)
        """
        # 静默转换，减少输出
        
        # 第一步：收集所有需要生成张量的参数名，并生成共享的numpy数组
        shared_tensors = {}  # 存储共享的numpy数组
        
        # 找出所有张量参数（在pytorch或tensorflow测试用例中）
        all_keys = set(pytorch_test_case.keys()) | set(tensorflow_test_case.keys())
        
        for key in all_keys:
            if key == "api":
                continue
            
            # 检查是否是张量描述
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
                # 生成共享的numpy数组
                numpy_array = self.generate_numpy_data(tensor_desc)
                shared_tensors[key] = numpy_array
        
        # 第二步：分别构建PyTorch和TensorFlow的测试用例
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
            
            # 简化测试用例中的numpy数组（处理新格式：torch_test_case 和 tensorflow_test_case）
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
        
        self._safe_print(f"💾 结果已保存到: {filepath}")
    
    def close(self):
        """关闭MongoDB连接"""
        self.client.close()


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description="基于LLM的PyTorch与TensorFlow算子比较测试框架"
    )
    parser.add_argument("--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS,
                        help="每个测试用例的最大迭代次数（默认3）")
    parser.add_argument("--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES,
                        help="每个算子要测试的用例数量（默认3）")
    parser.add_argument("--start", type=int, default=1,
                        help="起始算子索引（从1开始，默认1）")
    parser.add_argument("--end", type=int, default=None,
                        help="结束算子索引（包含，默认全部）")
    parser.add_argument("--operators", "-o", nargs="*",
                        help="指定要测试的算子名称（PyTorch格式）")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS,
                        help="并发线程数（默认1，顺序执行更稳定）")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"LLM模型名称（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--key-path", "-k", default=DEFAULT_KEY_PATH,
                        help=f"API key文件路径（默认 {DEFAULT_KEY_PATH}）")

    args = parser.parse_args()

    max_iterations = args.max_iterations
    num_test_cases = args.num_cases
    num_workers = max(1, args.workers)

    print("="*80)
    print("基于LLM的PyTorch与TensorFlow算子批量比较测试框架")
    print("="*80)
    print(f"📌 每个算子的迭代次数: {max_iterations}")
    print(f"📌 每个算子的测试用例数: {num_test_cases}")
    print(f"📌 LLM并发线程数: {num_workers}")
    print(f"📌 LLM模型: {args.model}")
    print("="*80)

    comparator = LLMEnhancedComparator(
        key_path=args.key_path,
        model=args.model,
        llm_workers=num_workers
    )

    start_time = time.time()
    start_datetime = datetime.now()

    try:
        print("\n🔍 正在获取数据库中的所有算子...")
        all_operators = list(comparator.collection.find({}, {"api": 1}))
        all_operator_names = [doc["api"] for doc in all_operators if "api" in doc]
        print(f"✅ 数据库中共有 {len(all_operator_names)} 个算子")

        if args.operators:
            operator_names = args.operators
            print(f"📋 指定算子数: {len(operator_names)}")
        else:
            total_available = len(all_operator_names)
            start_idx = max(1, args.start) - 1
            end_idx = args.end if args.end is not None else total_available
            if end_idx > total_available:
                end_idx = total_available
            if start_idx >= end_idx:
                raise ValueError(f"起始索引 {args.start} 必须小于结束索引 {end_idx}")

            operator_names = all_operator_names[start_idx:end_idx]
            print(f"📌 测试范围: 第 {start_idx + 1} 到第 {end_idx} 个算子")
            print(f"📋 将测试 {len(operator_names)} 个算子")

        print("\n🔍 过滤无 TensorFlow 对应实现的算子...")
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

        print(f"✅ 过滤完成: 原有 {original_count} 个算子，跳过 {skipped_count} 个，剩余 {len(operator_names)} 个")
        if skipped_operators:
            print(f"⏭️ 跳过的算子（前10个）: {', '.join([f'{op}({reason})' for op, reason in skipped_operators[:10]])}{'...' if len(skipped_operators) > 10 else ''}")

        print(f"📋 算子列表: {', '.join(operator_names[:10])}{'...' if len(operator_names) > 10 else ''}\n")

        all_operators_summary = []

        batch_log_file = os.path.join(comparator.result_dir, f"batch_test_log_{start_datetime.strftime('%Y%m%d_%H%M%S')}.txt")
        log_file = open(batch_log_file, 'w', encoding='utf-8')

        log_file.write("="*80 + "\n")
        log_file.write("批量测试总日志\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("测试配置:\n")
        log_file.write(f"  - 每个算子的迭代次数: {max_iterations}\n")
        log_file.write(f"  - 每个算子的测试用例数: {num_test_cases}\n")
        log_file.write(f"  - LLM并发线程数: {num_workers}\n")
        log_file.write(f"  - 数据库总算子数: {len(all_operator_names)}\n")
        if not args.operators:
            log_file.write(f"  - 测试范围: 第 {args.start} 到第 {args.end if args.end is not None else len(all_operator_names)} 个\n")
        log_file.write(f"  - 跳过的无对应实现算子数: {skipped_count}\n")
        log_file.write(f"  - 实际测试算子数: {len(operator_names)}\n")
        log_file.write("="*80 + "\n\n")
        log_file.flush()

        for idx, operator_name in enumerate(operator_names, 1):
            print("\n" + "🔷"*40)
            print(f"🎯 [{idx}/{len(operator_names)}] 开始测试算子: {operator_name}")
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

                    print(f"\n✅ 算子 {operator_name} 测试完成")
                    print(f"   - 总迭代次数: {len(results)}")
                    print(f"   - LLM生成用例数: {stats.get('llm_generated_cases', 0)}")
                    print(f"   - 成功执行用例数: {stats.get('successful_cases', 0)}")

                    log_file.write(f"[{idx}/{len(operator_names)}] {operator_name}\n")
                    log_file.write("  状态: ✅ 完成\n")
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

                    log_file.write(f"[{idx}/{len(operator_names)}] {operator_name}\n")
                    log_file.write("  状态: ⚠️ 无结果\n\n")
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

                log_file.write(f"[{idx}/{len(operator_names)}] {operator_name}\n")
                log_file.write("  状态: ❌ 失败\n")
                log_file.write(f"  错误: {str(e)}\n\n")
                log_file.flush()
                continue

        end_time = time.time()
        end_datetime = datetime.now()
        total_duration = end_time - start_time
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)

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
        print(f"⏭️ 跳过（无对应实现）: {skipped_count}")

        total_llm_cases = sum(s["llm_generated_cases"] for s in all_operators_summary)
        total_successful_cases = sum(s["successful_cases"] for s in all_operators_summary)
        total_iterations = sum(s["total_iterations"] for s in all_operators_summary)

        print("\n📈 统计数据:")
        print(f"   - LLM生成的测试用例总数: {total_llm_cases}")
        print(f"   - 成功执行的用例总数: {total_successful_cases}")
        if total_llm_cases > 0:
            success_rate = (total_successful_cases / total_llm_cases) * 100
            print(f"   - 成功执行占比: {success_rate:.2f}%")
        print(f"   - 总迭代次数: {total_iterations}")
        print(f"\n⏱️ 运行时间: {hours}小时 {minutes}分钟 {seconds}秒")

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

        print(f"💾 JSON摘要已保存到: {summary_file}")

    finally:
        comparator.close()
        print("\n✅ 批量测试程序执行完成")


if __name__ == "__main__":
    main()
