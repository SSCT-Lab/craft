#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于LLM的TensorFlow与PaddlePaddle算子比较测试框架（并发版本）
使用大模型进行测试用例修复和变异，提高用例可用性和覆盖率

流程说明：
1. 从MongoDB读取PyTorch测试用例作为原始数据源
2. 将测试用例同时迁移到TensorFlow和PaddlePaddle
3. 执行TensorFlow和PaddlePaddle的测试
4. 比较两个框架的执行结果
5. 使用LLM进行修复/变异/跳过策略

并发特性：
- 支持多线程并发处理多个算子
- 每个算子内部的测试用例也可并发执行
- 使用线程锁保证日志输出的正确性

命令行参数：
  --max-iterations, -m : 每个测试用例的最大迭代次数（默认3）
  --num-cases, -n      : 每个算子要测试的用例数量（默认3）
  --start              : 起始算子索引（从1开始，默认1）
  --end                : 结束算子索引（包含，默认全部）
  --operators, -o      : 指定要测试的算子名称（PyTorch格式）
  --workers, -w        : 并发线程数（默认1，顺序执行更稳定）
  --model              : LLM模型名称（默认qwen-plus）
  --key-path, -k       : API key文件路径（默认aliyun.key）
"""

# ==================== 环境变量设置（必须在导入TensorFlow前设置）====================
# 防止 Intel MKL 在多线程并发时发生冲突崩溃
import os
# MKL 线程设置
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
# OpenMP 线程设置
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OMP_DYNAMIC'] = 'FALSE'
# TensorFlow 线程设置
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
# Intel 相关设置
os.environ['KMP_BLOCKTIME'] = '0'
os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'
# 禁用 TensorFlow 的 GPU（如果不需要）
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# 减少 TensorFlow 日志输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
import re
import sys
import copy
import time
import traceback
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, RLock

import numpy as np
import pandas as pd
import pymongo
import tensorflow as tf
import paddle
from openai import OpenAI

# 添加项目根目录到路径，以便导入 component 模块
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from component.doc.doc_crawler_factory import get_doc_content

# ==================== 常量定义 ====================
DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"
DEFAULT_MAX_ITERATIONS = 3      # 每个测试用例的默认最大迭代次数
DEFAULT_NUM_CASES = 3           # 每个算子默认测试的用例数量
DEFAULT_WORKERS = 8             # 默认8线程并发（通过锁保护BLAS操作防止MKL冲突）

# 全局 TensorFlow 执行锁（防止MKL并发冲突）
# 由于某些TF操作（如Conv3D）调用底层MKL的cblas_sgemm，并发时会冲突
# 注意：TensorFlow 和 PaddlePaddle 都可能使用 MKL，所以需要对两者的执行都加锁
_BLAS_EXECUTION_LOCK = RLock()


class LLMEnhancedComparator:
    """基于LLM的TensorFlow与PaddlePaddle算子比较测试器"""
    
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", 
                 db_name: str = "freefuzz-torch",
                 key_path: str = DEFAULT_KEY_PATH,
                 model: str = DEFAULT_MODEL,
                 print_lock: Lock = None):
        """
        初始化基于LLM的TensorFlow和PaddlePaddle比较器
        
        Args:
            mongo_uri: MongoDB连接URI
            db_name: 数据库名称（使用PyTorch测试用例作为原始数据集）
            key_path: API key文件路径
            model: LLM模型名称
            print_lock: 打印锁（用于并发时线程安全输出）
        """
        self.model = model
        self.print_lock = print_lock or Lock()
        
        # MongoDB连接
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["argVS"]
        
        # 初始化LLM客户端（阿里千问大模型）
        api_key = self._load_api_key(key_path)
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 加载API映射表
        self.api_mapping = self.load_api_mapping()
        
        # 创建结果存储目录
        self.result_dir = os.path.join(ROOT_DIR, "tf_pd_test", "tf_pd_log_1")
        os.makedirs(self.result_dir, exist_ok=True)
        self._safe_print(f"📁 结果存储目录: {self.result_dir}")
        
        # 固定随机种子以确保可重复性
        self.random_seed = 42
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        paddle.seed(self.random_seed)
        
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
            "torch.nn.Conv3d": "会导致MKL崩溃",
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
        # 如果是相对路径，转为绝对路径
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
        """
        加载统一的API映射表，获取PyTorch->TensorFlow和PyTorch->PaddlePaddle的映射
        
        Returns:
            映射字典，格式为 {pytorch_api: {"tf_api": ..., "paddle_api": ...}}
        """
        mapping_file = os.path.join(ROOT_DIR, "component", "data", "unified_api_mappings.csv")
        try:
            df = pd.read_csv(mapping_file)
            mapping = {}
            
            for _, row in df.iterrows():
                pt_api = str(row["pytorch-api"]).strip()
                tf_api = str(row["tensorflow-api"]).strip()
                paddle_api = str(row["paddle-api"]).strip()
                
                mapping[pt_api] = {
                    "tf_api": tf_api,
                    "paddle_api": paddle_api
                }
            
            # 统计有效映射数量
            tf_valid = sum(1 for v in mapping.values() if v["tf_api"] != "无对应实现")
            pd_valid = sum(1 for v in mapping.values() if v["paddle_api"] != "无对应实现")
            both_valid = sum(1 for v in mapping.values() 
                           if v["tf_api"] != "无对应实现" and v["paddle_api"] != "无对应实现")
            
            self._safe_print(f"✅ 成功加载API映射表，共 {len(mapping)} 条映射")
            self._safe_print(f"   - TensorFlow有效映射: {tf_valid} 条")
            self._safe_print(f"   - PaddlePaddle有效映射: {pd_valid} 条")
            self._safe_print(f"   - 两框架都有映射: {both_valid} 条")
            self._safe_print(f"📄 映射文件: {mapping_file}")
            return mapping
        except Exception as e:
            self._safe_print(f"❌ 加载API映射表失败: {e}")
            return {}
    
    def is_class_based_api(self, api_name: str) -> bool:
        """判断API是否是基于类的（如 torch.nn.Conv2d）"""
        parts = api_name.split(".")
        if len(parts) >= 2:
            name = parts[-1]
            # 类名通常以大写字母开头
            return name[0].isupper() if name else False
        return False
    
    def convert_api_name(self, pytorch_api: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        将PyTorch API转换为TensorFlow和PaddlePaddle API
        
        Args:
            pytorch_api: PyTorch API名称
        
        Returns:
            (TensorFlow API, PaddlePaddle API, 映射方法说明)
            - 如果映射表中找到且两个框架都有有效的API → 返回映射后的API
            - 如果任一框架为 "无对应实现" → 对应位置返回 None
            - 如果映射表中找不到该API → 返回 (None, None, "映射表中未找到")
        """
        if pytorch_api in self.api_mapping:
            tf_api = self.api_mapping[pytorch_api]["tf_api"]
            paddle_api = self.api_mapping[pytorch_api]["paddle_api"]
            
            # 检查是否为 "无对应实现"
            tf_api = None if tf_api == "无对应实现" else tf_api
            paddle_api = None if paddle_api == "无对应实现" else paddle_api
            
            if tf_api is None and paddle_api is None:
                return None, None, "两框架均无对应实现"
            elif tf_api is None:
                return None, paddle_api, "TensorFlow无对应实现"
            elif paddle_api is None:
                return tf_api, None, "PaddlePaddle无对应实现"
            else:
                return tf_api, paddle_api, "映射表"
        
        return None, None, "映射表中未找到"
    
    def get_operator_function(self, api_name: str, framework: str):
        """
        获取算子函数
        
        Args:
            api_name: API名称（如 tf.abs, paddle.abs）
            framework: 框架名称 ("tensorflow" 或 "paddle")
        
        Returns:
            算子函数对象，如果找不到则返回 None
        """
        try:
            parts = api_name.split(".")
            if len(parts) >= 2:
                if framework == "tensorflow" and parts[0] == "tf":
                    module = tf
                    for part in parts[1:]:
                        module = getattr(module, part)
                    return module
                elif framework == "paddle" and parts[0] == "paddle":
                    module = paddle
                    for part in parts[1:]:
                        module = getattr(module, part)
                    return module
            return None
        except AttributeError:
            return None
    
    def generate_numpy_data(self, data: Any) -> np.ndarray:
        """
        生成numpy数组作为共享数据源
        
        支持的dtype格式：
        - 带torch前缀：torch.float32, torch.bool, torch.int64等
        - 带tf前缀：tf.float32, tf.bool等
        - 不带前缀：float32, bool, int64等
        """
        if isinstance(data, dict):
            dtype_map = {
                # torch格式
                "torch.float64": np.float64,
                "torch.float32": np.float32,
                "torch.int64": np.int64,
                "torch.int32": np.int32,
                "torch.bool": np.bool_,
                "torch.uint8": np.uint8,
                # tensorflow格式
                "tf.float64": np.float64,
                "tf.float32": np.float32,
                "tf.int64": np.int64,
                "tf.int32": np.int32,
                "tf.bool": np.bool_,
                # 不带前缀
                "float64": np.float64,
                "float32": np.float32,
                "int64": np.int64,
                "int32": np.int32,
                "bool": np.bool_,
                "uint8": np.uint8,
                "bool_": np.bool_,
                "float": np.float32,
                "int": np.int64,
            }
            
            shape = data.get("shape", [])
            dtype_str = data.get("dtype", "float32")
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
    
    def prepare_shared_numpy_data(self, document: Dict[str, Any], case_index: int = 0) -> Dict[str, Any]:
        """
        从PyTorch测试用例准备共享的numpy数据
        确保TensorFlow和PaddlePaddle使用相同的输入
        
        重要：该方法会将所有dict格式（带shape和dtype）的参数转换为numpy.ndarray，
        以便后续两个框架使用完全相同的输入数据。
        """
        shared_data = {}
        api_name = document.get("api", "")
        
        # 对于类形式的API，如果没有input参数，生成默认输入
        if self.is_class_based_api(api_name) and "input" not in document:
            if "2d" in api_name.lower() or "2D" in api_name:
                default_shape = {"shape": [2, 3, 4, 4], "dtype": "float32"}
            elif "1d" in api_name.lower() or "1D" in api_name:
                default_shape = {"shape": [2, 3, 10], "dtype": "float32"}
            elif "3d" in api_name.lower() or "3D" in api_name:
                default_shape = {"shape": [2, 3, 4, 4, 4], "dtype": "float32"}
            else:
                default_shape = {"shape": [2, 3], "dtype": "float32"}
            
            shared_data["input"] = self.generate_numpy_data(default_shape)
        
        # 处理文档中的其他参数
        exclude_keys = ["_id", "api"]
        for key, value in document.items():
            if key not in exclude_keys:
                # 对于可变参数（以*开头）
                if key.startswith('*'):
                    if isinstance(value, list) and len(value) > 0:
                        idx = min(case_index, len(value) - 1)
                        vararg_value = value[idx]
                        # 如果可变参数是list，需要转换其中每个dict元素为numpy数组
                        if isinstance(vararg_value, list):
                            converted_list = []
                            for item in vararg_value:
                                if isinstance(item, dict) and "shape" in item:
                                    converted_list.append(self.generate_numpy_data(item))
                                else:
                                    converted_list.append(item)
                            shared_data[key] = converted_list
                        else:
                            shared_data[key] = vararg_value
                    else:
                        shared_data[key] = value
                elif isinstance(value, list):
                    if len(value) > 0:
                        idx = min(case_index, len(value) - 1)
                        param_value = value[idx]
                        # ✅ 将dict格式转换为numpy数组
                        if isinstance(param_value, dict) and "shape" in param_value:
                            shared_data[key] = self.generate_numpy_data(param_value)
                        else:
                            shared_data[key] = param_value
                else:
                    shared_data[key] = value
        
        return shared_data
    
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
    
    def convert_to_tensor_paddle(self, data: Any, numpy_data: np.ndarray = None) -> paddle.Tensor:
        """转换数据为PaddlePaddle张量"""
        if numpy_data is not None:
            return paddle.to_tensor(numpy_data.copy())
        
        if isinstance(data, dict):
            numpy_data = self.generate_numpy_data(data)
            return paddle.to_tensor(numpy_data.copy())
        elif isinstance(data, (int, float)):
            return paddle.to_tensor(data)
        elif isinstance(data, list):
            return paddle.to_tensor(data)
        else:
            return paddle.to_tensor(data)
    
    def should_skip_param(self, key: str, api_name: str) -> bool:
        """判断是否应该跳过某个参数"""
        # PyTorch特有的参数，不传给TF/Paddle
        common_skip_params = ["layout", "requires_grad", "out", "device", "memory_format"]
        return key in common_skip_params
    
    def prepare_arguments_tensorflow(self, test_case: Dict[str, Any], tensorflow_api: str) -> Tuple[List[Any], Dict[str, Any]]:
        """为TensorFlow准备参数
        
        重要：所有张量参数必须已经是numpy.ndarray格式，不应该是dict格式的描述。
        这样可以确保TensorFlow和PaddlePaddle使用完全相同的输入数据。
        """
        args = []
        kwargs = {}
        
        # 检查可变参数
        varargs_key = None
        for key in test_case.keys():
            if key.startswith('*'):
                varargs_key = key
                break
        
        if varargs_key:
            varargs_value = test_case[varargs_key]
            if isinstance(varargs_value, list):
                for idx, item in enumerate(varargs_value):
                    if isinstance(item, dict) and "shape" in item:
                        # ❌ 不应该出现dict格式！应该在测试用例准备阶段就转换为numpy数组
                        raise ValueError(
                            f"参数 '{varargs_key}[{idx}]' 仍然是dict格式 {item}，\n"
                            f"这会导致TensorFlow和PaddlePaddle使用不同的随机数据！\n"
                            f"请确保在测试用例准备阶段（prepare_shared_numpy_data或_convert_llm_test_cases）\n"
                            f"就将所有张量参数转换为numpy.ndarray。"
                        )
                    elif isinstance(item, np.ndarray):
                        args.append(self.convert_to_tensor_tensorflow(None, item))
                    else:
                        args.append(item)
            return args, kwargs
        
        # 位置参数
        positional_params = ["condition", "x", "y", "input", "other"]
        
        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, dict) and "shape" in value:
                    raise ValueError(
                        f"参数 '{param_name}' 仍然是dict格式 {value}，\n"
                        f"这会导致TensorFlow和PaddlePaddle使用不同的随机数据！\n"
                        f"请确保在测试用例准备阶段就转换为numpy.ndarray。"
                    )
                elif isinstance(value, np.ndarray):
                    args.append(self.convert_to_tensor_tensorflow(None, value))
                else:
                    args.append(value)
        
        # 其他参数作为关键字参数
        for key, value in test_case.items():
            if key not in positional_params + ["api"]:
                if self.should_skip_param(key, tensorflow_api):
                    continue
                if isinstance(value, dict) and "shape" in value:
                    raise ValueError(
                        f"参数 '{key}' 仍然是dict格式 {value}，\n"
                        f"这会导致TensorFlow和PaddlePaddle使用不同的随机数据！\n"
                        f"请确保在测试用例准备阶段就转换为numpy.ndarray。"
                    )
                elif isinstance(value, np.ndarray):
                    kwargs[key] = self.convert_to_tensor_tensorflow(None, value)
                else:
                    kwargs[key] = value
        
        return args, kwargs
    
    def prepare_arguments_paddle(self, test_case: Dict[str, Any], paddle_api: str) -> Tuple[List[Any], Dict[str, Any]]:
        """为PaddlePaddle准备参数
        
        重要：所有张量参数必须已经是numpy.ndarray格式，不应该是dict格式的描述。
        这样可以确保TensorFlow和PaddlePaddle使用完全相同的输入数据。
        """
        args = []
        kwargs = {}
        
        # 检查可变参数
        varargs_key = None
        for key in test_case.keys():
            if key.startswith('*'):
                varargs_key = key
                break
        
        if varargs_key:
            varargs_value = test_case[varargs_key]
            if isinstance(varargs_value, list):
                for idx, item in enumerate(varargs_value):
                    if isinstance(item, dict) and "shape" in item:
                        # ❌ 不应该出现dict格式！应该在测试用例准备阶段就转换为numpy数组
                        raise ValueError(
                            f"参数 '{varargs_key}[{idx}]' 仍然是dict格式 {item}，\n"
                            f"这会导致TensorFlow和PaddlePaddle使用不同的随机数据！\n"
                            f"请确保在测试用例准备阶段（prepare_shared_numpy_data或_convert_llm_test_cases）\n"
                            f"就将所有张量参数转换为numpy.ndarray。"
                        )
                    elif isinstance(item, np.ndarray):
                        args.append(self.convert_to_tensor_paddle(None, item))
                    else:
                        args.append(item)
            return args, kwargs
        
        # 位置参数
        positional_params = ["condition", "x", "y", "input", "other"]
        
        for param_name in positional_params:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, dict) and "shape" in value:
                    raise ValueError(
                        f"参数 '{param_name}' 仍然是dict格式 {value}，\n"
                        f"这会导致TensorFlow和PaddlePaddle使用不同的随机数据！\n"
                        f"请确保在测试用例准备阶段就转换为numpy.ndarray。"
                    )
                elif isinstance(value, np.ndarray):
                    args.append(self.convert_to_tensor_paddle(None, value))
                else:
                    args.append(value)
        
        # 其他参数作为关键字参数
        for key, value in test_case.items():
            if key not in positional_params + ["api"]:
                if self.should_skip_param(key, paddle_api):
                    continue
                if isinstance(value, dict) and "shape" in value:
                    raise ValueError(
                        f"参数 '{key}' 仍然是dict格式 {value}，\n"
                        f"这会导致TensorFlow和PaddlePaddle使用不同的随机数据！\n"
                        f"请确保在测试用例准备阶段就转换为numpy.ndarray。"
                    )
                elif isinstance(value, np.ndarray):
                    kwargs[key] = self.convert_to_tensor_paddle(None, value)
                else:
                    kwargs[key] = value
        
        return args, kwargs
    
    def compare_tensors(self, tf_result, paddle_result, tolerance: float = 1e-5) -> Tuple[bool, str]:
        """比较TensorFlow和PaddlePaddle的张量结果"""
        try:
            # 转换为numpy进行比较
            if hasattr(tf_result, 'numpy'):
                tf_np = tf_result.numpy()
            else:
                tf_np = np.array(tf_result)
            
            if hasattr(paddle_result, 'numpy'):
                paddle_np = paddle_result.numpy()
            else:
                paddle_np = np.array(paddle_result)
            
            # 检查形状
            if tf_np.shape != paddle_np.shape:
                return False, f"形状不匹配: TensorFlow {tf_np.shape} vs PaddlePaddle {paddle_np.shape}"
            
            # 检查dtype是否为布尔类型
            if tf_np.dtype == np.bool_ or paddle_np.dtype == np.bool_:
                if np.array_equal(tf_np, paddle_np):
                    return True, "布尔值匹配"
                else:
                    diff_count = np.sum(tf_np != paddle_np)
                    return False, f"布尔值不匹配，差异数量: {diff_count}"
            
            # 检查数值
            if np.allclose(tf_np, paddle_np, atol=tolerance, rtol=tolerance, equal_nan=True):
                return True, "数值匹配"
            else:
                max_diff = np.max(np.abs(tf_np - paddle_np))
                return False, f"数值不匹配，最大差异: {max_diff}"
        
        except Exception as e:
            return False, f"比较过程出错: {str(e)}"
    
    def _validate_test_case_params(self, test_case: Dict[str, Any], api_name: str) -> Tuple[bool, str]:
        """
        验证测试用例参数的合法性，防止TensorFlow C++层面崩溃
        
        Args:
            test_case: 测试用例
            api_name: API名称
        
        Returns:
            (是否合法, 错误信息)
        """
        # 检查perm/axis参数是否包含负数索引（某些TF API不支持）
        dangerous_apis_for_negative_perm = ["tf.transpose", "tf.gather"]
        
        for param_name in ["perm", "axis", "axes"]:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, (list, tuple)):
                    if any(isinstance(v, int) and v < 0 for v in value):
                        # 检查是否有对应的张量来确定维度
                        tensor_param = None
                        for key in ["x", "input", "a"]:
                            if key in test_case and isinstance(test_case[key], np.ndarray):
                                tensor_param = test_case[key]
                                break
                        
                        if tensor_param is not None:
                            ndim = len(tensor_param.shape)
                            # 尝试将负数索引转换为正数
                            try:
                                new_value = [v if v >= 0 else (ndim + v) for v in value]
                                if any(v < 0 or v >= ndim for v in new_value):
                                    return False, f"参数 {param_name}={value} 转换后超出范围 [0, {ndim})"
                                test_case[param_name] = new_value
                            except Exception:
                                return False, f"参数 {param_name}={value} 包含无效的负数索引"
                        else:
                            # 没有张量参数，直接拒绝负数索引
                            if api_name in dangerous_apis_for_negative_perm:
                                return False, f"参数 {param_name}={value} 包含负数索引，可能导致TF崩溃"
                elif isinstance(value, int) and value < 0:
                    # 单个负数索引
                    tensor_param = None
                    for key in ["x", "input", "a"]:
                        if key in test_case and isinstance(test_case[key], np.ndarray):
                            tensor_param = test_case[key]
                            break
                    
                    if tensor_param is not None:
                        ndim = len(tensor_param.shape)
                        new_value = ndim + value
                        if new_value < 0 or new_value >= ndim:
                            return False, f"参数 {param_name}={value} 转换后超出范围 [0, {ndim})"
                        test_case[param_name] = new_value
        
        # 检查shape参数是否包含0或负数维度
        for param_name in ["shape", "output_shape"]:
            if param_name in test_case:
                value = test_case[param_name]
                if isinstance(value, (list, tuple)):
                    if any(isinstance(v, int) and v < 0 for v in value):
                        return False, f"参数 {param_name}={value} 包含负数维度"
        
        return True, ""
    
    def execute_test_case(self, tensorflow_api: str, paddle_api: str, 
                          tensorflow_test_case: Dict[str, Any], 
                          paddle_test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单个测试用例，比较TensorFlow和PaddlePaddle的结果
        
        Args:
            tensorflow_api: TensorFlow API名称
            paddle_api: PaddlePaddle API名称
            tensorflow_test_case: TensorFlow测试用例
            paddle_test_case: PaddlePaddle测试用例
        
        Returns:
            执行结果字典
        """
        result = {
            "tensorflow_api": tensorflow_api,
            "paddle_api": paddle_api,
            "tensorflow_success": False,
            "paddle_success": False,
            "results_match": False,
            "tensorflow_error": None,
            "paddle_error": None,
            "comparison_error": None,
            "tensorflow_shape": None,
            "paddle_shape": None,
            "tensorflow_dtype": None,
            "paddle_dtype": None,
            "status": "unknown"
        }
        
        # 判断是否是类算子（基于原始PyTorch API）
        is_class_api = self.is_class_based_api(tensorflow_api)
        
        # 参数合法性检查（防止TensorFlow C++层崩溃）
        tf_valid, tf_validation_error = self._validate_test_case_params(tensorflow_test_case, tensorflow_api)
        if not tf_valid:
            result["tensorflow_error"] = f"参数验证失败: {tf_validation_error}"
            result["status"] = "validation_failed"
        
        pd_valid, pd_validation_error = self._validate_test_case_params(paddle_test_case, paddle_api)
        if not pd_valid:
            result["paddle_error"] = f"参数验证失败: {pd_validation_error}"
            if result["status"] == "validation_failed":
                result["status"] = "both_validation_failed"
            else:
                result["status"] = "paddle_validation_failed"
        
        # 如果验证都失败了，直接返回
        if not tf_valid and not pd_valid:
            return result
        
        # 测试TensorFlow（使用全局锁防止MKL并发冲突）
        tensorflow_result = None
        if tf_valid:
            try:
                tensorflow_func = self.get_operator_function(tensorflow_api, "tensorflow")
                if tensorflow_func is None:
                    result["tensorflow_error"] = f"TensorFlow算子 {tensorflow_api} 未找到"
                else:
                    args, kwargs = self.prepare_arguments_tensorflow(tensorflow_test_case, tensorflow_api)
                    
                    # 使用全局锁保护TensorFlow执行（防止MKL cblas_sgemm并发冲突）
                    with _BLAS_EXECUTION_LOCK:
                        if is_class_api:
                            # 类算子：先实例化再调用
                            init_kwargs = {k: v for k, v in kwargs.items() if k not in ['x', 'input']}
                            tensorflow_instance = tensorflow_func(**init_kwargs)
                            if 'x' in kwargs:
                                input_tensor = kwargs['x']
                            elif 'input' in kwargs:
                                input_tensor = kwargs['input']
                            elif len(args) > 0:
                                input_tensor = args[0]
                            else:
                                input_tensor = self.convert_to_tensor_tensorflow({"shape": [2, 3], "dtype": "float32"})
                            tensorflow_result = tensorflow_instance(input_tensor)
                        else:
                            tensorflow_result = tensorflow_func(*args, **kwargs)
                        
                        # 强制TensorFlow同步执行完成，确保BLAS操作结束后再释放锁
                        if hasattr(tensorflow_result, 'numpy'):
                            _ = tensorflow_result.numpy()
                    
                    result["tensorflow_success"] = True
                    result["tensorflow_shape"] = list(tensorflow_result.shape) if hasattr(tensorflow_result, 'shape') else None
                    result["tensorflow_dtype"] = str(tensorflow_result.dtype) if hasattr(tensorflow_result, 'dtype') else None
            except Exception as e:
                result["tensorflow_error"] = str(e)
                result["tensorflow_traceback"] = traceback.format_exc()
        
        # 测试PaddlePaddle（同样使用全局锁防止MKL并发冲突）
        paddle_result = None
        if pd_valid:
            try:
                paddle_func = self.get_operator_function(paddle_api, "paddle")
                if paddle_func is None:
                    result["paddle_error"] = f"PaddlePaddle算子 {paddle_api} 未找到"
                else:
                    args, kwargs = self.prepare_arguments_paddle(paddle_test_case, paddle_api)
                    
                    # 使用全局锁保护PaddlePaddle执行（防止MKL cblas_sgemm并发冲突）
                    with _BLAS_EXECUTION_LOCK:
                        if is_class_api:
                            # 类算子：先实例化再调用
                            init_kwargs = {k: v for k, v in kwargs.items() if k not in ['x', 'input']}
                            paddle_instance = paddle_func(**init_kwargs)
                            if 'x' in kwargs:
                                input_tensor = kwargs['x']
                            elif 'input' in kwargs:
                                input_tensor = kwargs['input']
                            elif len(args) > 0:
                                input_tensor = args[0]
                            else:
                                input_tensor = self.convert_to_tensor_paddle({"shape": [2, 3], "dtype": "float32"})
                            paddle_result = paddle_instance(input_tensor)
                        else:
                            paddle_result = paddle_func(*args, **kwargs)
                        
                        # 强制PaddlePaddle同步执行完成，确保BLAS操作结束后再释放锁
                        if hasattr(paddle_result, 'numpy'):
                            _ = paddle_result.numpy()
                    
                    result["paddle_success"] = True
                    result["paddle_shape"] = list(paddle_result.shape) if hasattr(paddle_result, 'shape') else None
                    result["paddle_dtype"] = str(paddle_result.dtype) if hasattr(paddle_result, 'dtype') else None
            except Exception as e:
                result["paddle_error"] = str(e)
                result["paddle_traceback"] = traceback.format_exc()
        
        # 比较结果
        if result["tensorflow_success"] and result["paddle_success"]:
            try:
                is_match, comparison_msg = self.compare_tensors(tensorflow_result, paddle_result)
                result["results_match"] = is_match
                result["comparison_error"] = comparison_msg if not is_match else None
                result["status"] = "compared"
            except Exception as e:
                result["comparison_error"] = str(e)
                result["status"] = "comparison_failed"
        elif result["tensorflow_success"] and not result["paddle_success"]:
            result["status"] = "paddle_failed"
        elif not result["tensorflow_success"] and result["paddle_success"]:
            result["status"] = "tensorflow_failed"
        else:
            result["status"] = "both_failed"
        
        return result
    
    def _fetch_api_docs(self, tensorflow_api: str, paddle_api: str) -> Tuple[str, str]:
        """爬取TensorFlow和PaddlePaddle的API文档"""
        MIN_DOC_LENGTH = 300
        
        tensorflow_doc = ""
        paddle_doc = ""
        
        try:
            self._safe_print(f"    📖 正在爬取 TensorFlow 文档: {tensorflow_api}")
            tensorflow_doc = get_doc_content(tensorflow_api, "tensorflow")
            if (tensorflow_doc 
                and "Unable" not in tensorflow_doc 
                and "not supported" not in tensorflow_doc
                and len(tensorflow_doc.strip()) > MIN_DOC_LENGTH):
                if len(tensorflow_doc) > 3000:
                    tensorflow_doc = tensorflow_doc[:3000] + "\n... (文档已截断)"
                self._safe_print(f"    ✅ TensorFlow 文档获取成功 ({len(tensorflow_doc)} 字符)")
            else:
                tensorflow_doc = ""
                self._safe_print(f"    ⚠️ TensorFlow 文档无效或过短")
        except Exception as e:
            tensorflow_doc = ""
            self._safe_print(f"    ❌ TensorFlow 文档爬取失败: {e}")
        
        try:
            self._safe_print(f"    📖 正在爬取 PaddlePaddle 文档: {paddle_api}")
            paddle_doc = get_doc_content(paddle_api, "paddle")
            if (paddle_doc 
                and "Unable" not in paddle_doc 
                and "not supported" not in paddle_doc
                and len(paddle_doc.strip()) > MIN_DOC_LENGTH):
                if len(paddle_doc) > 3000:
                    paddle_doc = paddle_doc[:3000] + "\n... (文档已截断)"
                self._safe_print(f"    ✅ PaddlePaddle 文档获取成功 ({len(paddle_doc)} 字符)")
            else:
                paddle_doc = ""
                self._safe_print(f"    ⚠️ PaddlePaddle 文档无效或过短")
        except Exception as e:
            paddle_doc = ""
            self._safe_print(f"    ❌ PaddlePaddle 文档爬取失败: {e}")
        
        return tensorflow_doc, paddle_doc
    
    def _build_llm_prompt(self, execution_result: Dict[str, Any], 
                          tensorflow_test_case: Dict[str, Any], 
                          paddle_test_case: Dict[str, Any],
                          tensorflow_doc: str = "", 
                          paddle_doc: str = "") -> str:
        """构建LLM的提示词"""
        tensorflow_api = execution_result.get("tensorflow_api", "")
        paddle_api = execution_result.get("paddle_api", "")
        status = execution_result.get("status", "")
        tensorflow_success = execution_result.get("tensorflow_success", False)
        paddle_success = execution_result.get("paddle_success", False)
        results_match = execution_result.get("results_match", False)
        tensorflow_error = execution_result.get("tensorflow_error", "")
        paddle_error = execution_result.get("paddle_error", "")
        comparison_error = execution_result.get("comparison_error", "")
        
        # 简化测试用例
        simplified_tensorflow_test_case = {}
        for key, value in tensorflow_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_tensorflow_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_tensorflow_test_case[key] = value
        
        simplified_paddle_test_case = {}
        for key, value in paddle_test_case.items():
            if isinstance(value, np.ndarray):
                simplified_paddle_test_case[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                simplified_paddle_test_case[key] = value
        
        # 构建参数示例
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
        if tensorflow_doc or paddle_doc:
            doc_section = "\n## 官方API文档参考\n\n"
            if tensorflow_doc:
                doc_section += f"### TensorFlow {tensorflow_api} 文档\n```\n{tensorflow_doc}\n```\n\n"
            if paddle_doc:
                doc_section += f"### PaddlePaddle {paddle_api} 文档\n```\n{paddle_doc}\n```\n\n"
        
        prompt = f"""请分析以下算子测试用例在TensorFlow和PaddlePaddle框架中的执行结果，并根据结果进行测试用例的修复或变异（fuzzing）。

## 测试信息
- **TensorFlow API**: {tensorflow_api}
- **PaddlePaddle API**: {paddle_api}
{doc_section}
## 执行结果
- **执行状态**: {status}
- **TensorFlow执行成功**: {tensorflow_success}
- **PaddlePaddle执行成功**: {paddle_success}
- **结果是否一致**: {results_match}

## 错误信息
- **TensorFlow错误**: {tensorflow_error if tensorflow_error else "无"}
- **PaddlePaddle错误**: {paddle_error if paddle_error else "无"}
- **比较错误**: {comparison_error if comparison_error else "无"}

## 原始测试用例

### TensorFlow测试用例
```json
{json.dumps(simplified_tensorflow_test_case, indent=2, ensure_ascii=False)}
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
  "operation": "mutation"或"repair"或"skip",
  "reason": "进行该操作的原因(不要太长，120字以内)",
  "tensorflow_test_case": {{
    "api": "{tensorflow_api}",
{tf_param_example_str}
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
4. 构造两个框架的用例时必须保证输入相同（必要时进行张量形状的转换，如NHWC与NCHW转换）、参数在语义上严格对应
5. TensorFlow和PaddlePaddle的测试用例可以有参数名差异、参数值差异或者参数数量的差异，只要保证理论上输出相同就行
6. 如果这个算子找不到官方文档，请判断是否是因为该算子不存在或者已经从当前版本移除了，如果是这样，请将 operation 设置为 "skip"
7. 测试用例变异时可适当探索一些极端情况，例如：空张量（shape包含0）、单元素张量（shape=[1]或[]）、高维张量、超大张量、不同数据类型（int、float、bool）、边界值等
8. 请仔细阅读官方API文档，确保参数名称、参数类型、参数取值范围等与文档一致
"""
        return prompt
    
    def call_llm_for_repair_or_mutation(self, execution_result: Dict[str, Any], 
                                         tensorflow_test_case: Dict[str, Any], 
                                         paddle_test_case: Dict[str, Any],
                                         tensorflow_doc: str = "", 
                                         paddle_doc: str = "") -> Dict[str, Any]:
        """调用LLM进行测试用例修复或变异"""
        prompt = self._build_llm_prompt(execution_result, tensorflow_test_case, paddle_test_case, tensorflow_doc, paddle_doc)
        
        try:
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个深度学习框架测试专家，精通TensorFlow和PaddlePaddle框架的API差异。你的任务是根据测试用例的执行结果，判断是否需要修复或变异测试用例，并返回严格的JSON格式结果。"
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
                        "tensorflow_test_case": tensorflow_test_case,
                        "paddle_test_case": paddle_test_case
                    }
        
        except Exception as e:
            self._safe_print(f"    ❌ 调用LLM失败: {e}")
            return {
                "operation": "skip",
                "reason": f"LLM调用失败: {e}",
                "tensorflow_test_case": tensorflow_test_case,
                "paddle_test_case": paddle_test_case
            }
    
    def get_num_test_cases_from_document(self, document: Dict[str, Any]) -> int:
        """获取文档中的测试用例数量"""
        max_len = 0
        for key, value in document.items():
            if key not in ["_id", "api"] and isinstance(value, list):
                max_len = max(max_len, len(value))
        return max_len if max_len > 0 else 1
    
    def migrate_pytorch_to_tf_pd(self, pytorch_test_case: Dict[str, Any], 
                                  tensorflow_api: str, 
                                  paddle_api: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        将PyTorch测试用例迁移到TensorFlow和PaddlePaddle格式
        
        Args:
            pytorch_test_case: PyTorch格式的测试用例
            tensorflow_api: TensorFlow API名称
            paddle_api: PaddlePaddle API名称
        
        Returns:
            (TensorFlow测试用例, PaddlePaddle测试用例)
        """
        tensorflow_test_case = copy.deepcopy(pytorch_test_case)
        paddle_test_case = copy.deepcopy(pytorch_test_case)
        
        # 设置正确的API名称
        tensorflow_test_case["api"] = tensorflow_api
        paddle_test_case["api"] = paddle_api
        
        # 参数名映射（PyTorch -> TensorFlow/PaddlePaddle 通用映射）
        param_mapping = {
            "input": "x",  # 很多TF/Paddle API使用x作为输入参数名
        }
        
        # 对TensorFlow测试用例进行参数名转换
        for old_key, new_key in param_mapping.items():
            if old_key in tensorflow_test_case and old_key != "api":
                tensorflow_test_case[new_key] = tensorflow_test_case.pop(old_key)
        
        # 对PaddlePaddle测试用例进行参数名转换（Paddle的参数名通常与PyTorch类似）
        # 注意：Paddle很多API的参数名与PyTorch相同，所以这里可能不需要太多转换
        
        return tensorflow_test_case, paddle_test_case
    
    def llm_enhanced_test_operator(self, pytorch_operator_name: str, 
                                    max_iterations: int = DEFAULT_MAX_ITERATIONS, 
                                    num_test_cases: int = None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        使用LLM增强的方式测试单个算子
        
        流程：
        1. 从MongoDB读取PyTorch测试用例
        2. 迁移到TensorFlow和PaddlePaddle
        3. 执行两个框架的测试
        4. 比较结果
        5. 使用LLM进行修复/变异
        
        Args:
            pytorch_operator_name: PyTorch算子名称（如 "torch.abs"）
            max_iterations: 每个测试用例的最大迭代次数
            num_test_cases: 要测试的用例数量，None表示测试所有用例
        
        Returns:
            (所有测试用例的所有迭代结果列表, 统计信息字典)
        """
        self._safe_print(f"\n{'='*80}")
        self._safe_print(f"🎯 开始测试算子: {pytorch_operator_name}")
        self._safe_print(f"🔄 每个用例的最大迭代次数: {max_iterations}")
        self._safe_print(f"{'='*80}\n")
        
        # 初始化统计计数器
        stats = {
            "llm_generated_cases": 0,      # LLM生成的测试用例总数
            "successful_cases": 0           # 两个框架都执行成功的测试用例数
        }
        
        # 检查是否是会导致程序卡住的算子
        if pytorch_operator_name in self.problematic_apis:
            reason = self.problematic_apis[pytorch_operator_name]
            self._safe_print(f"⏭️ 跳过算子 {pytorch_operator_name}：{reason}")
            return [], stats
        
        # 从MongoDB获取算子的测试用例
        document = self.collection.find_one({"api": pytorch_operator_name})
        if document is None:
            self._safe_print(f"❌ 未找到算子 {pytorch_operator_name} 的测试用例")
            return [], stats
        
        # 获取测试用例总数
        total_cases = self.get_num_test_cases_from_document(document)
        
        # 确定实际要测试的用例数量
        if num_test_cases is None:
            num_test_cases = total_cases
        else:
            num_test_cases = min(num_test_cases, total_cases)
        
        # 获取TensorFlow和PaddlePaddle API
        tensorflow_api, paddle_api, mapping_method = self.convert_api_name(pytorch_operator_name)
        
        if tensorflow_api is None or paddle_api is None:
            self._safe_print(f"❌ {pytorch_operator_name} 无法映射: TF={tensorflow_api}, PD={paddle_api}")
            return [], stats
        
        self._safe_print(f"📋 {pytorch_operator_name} -> TF:{tensorflow_api}, PD:{paddle_api} ({num_test_cases}个用例)")
        
        # 存储所有测试用例的所有迭代结果
        all_results = []
        
        # 循环测试每个用例
        for case_idx in range(num_test_cases):
            self._safe_print(f"\n📋 用例 {case_idx + 1}/{num_test_cases}")
            
            # 准备当前测试用例的初始数据（从PyTorch格式）
            initial_pytorch_test_case = self.prepare_shared_numpy_data(document, case_index=case_idx)
            initial_pytorch_test_case["api"] = pytorch_operator_name
            
            # 迁移到TensorFlow和PaddlePaddle
            initial_tensorflow_test_case, initial_paddle_test_case = self.migrate_pytorch_to_tf_pd(
                initial_pytorch_test_case, tensorflow_api, paddle_api
            )
            
            # 对当前测试用例进行迭代测试
            case_results = self._test_single_case_with_iterations(
                tensorflow_api, 
                paddle_api, 
                initial_tensorflow_test_case,
                initial_paddle_test_case,
                max_iterations,
                case_idx + 1,
                stats
            )
            
            # 保存当前测试用例的结果
            all_results.extend(case_results)
        
        self._safe_print(f"\n{'='*80}")
        self._safe_print(f"✅ 所有测试完成")
        self._safe_print(f"📊 共测试 {num_test_cases} 个用例，总计 {len(all_results)} 次迭代")
        self._safe_print(f"📊 LLM生成的测试用例数: {stats['llm_generated_cases']}")
        self._safe_print(f"📊 两个框架都执行成功的用例数: {stats['successful_cases']}")
        self._safe_print(f"{'='*80}\n")
        
        return all_results, stats
    
    def _test_single_case_with_iterations(self, tensorflow_api: str, paddle_api: str,
                                           initial_tensorflow_test_case: Dict[str, Any],
                                           initial_paddle_test_case: Dict[str, Any],
                                           max_iterations: int,
                                           case_number: int,
                                           stats: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        对单个测试用例进行多轮迭代测试
        
        Args:
            tensorflow_api: TensorFlow API名称
            paddle_api: PaddlePaddle API名称
            initial_tensorflow_test_case: 初始TensorFlow测试用例
            initial_paddle_test_case: 初始PaddlePaddle测试用例
            max_iterations: 最大迭代次数
            case_number: 测试用例编号
            stats: 统计信息字典
        
        Returns:
            该测试用例的所有迭代结果
        """
        case_results = []
        
        current_tensorflow_test_case = initial_tensorflow_test_case
        current_paddle_test_case = initial_paddle_test_case
        
        # 标记当前用例是否为LLM生成的
        is_llm_generated = False
        
        # 预先爬取API文档（静默执行）
        tensorflow_doc, paddle_doc = self._fetch_api_docs(tensorflow_api, paddle_api)
        
        # 开始迭代测试
        for iteration in range(max_iterations):
            source_type = "LLM" if is_llm_generated else "DB"
            self._safe_print(f"  🔄 迭代 {iteration + 1}/{max_iterations} ({source_type})", end="")
            
            # 执行测试用例
            try:
                execution_result = self.execute_test_case(
                    tensorflow_api, paddle_api, 
                    current_tensorflow_test_case, current_paddle_test_case
                )
                
                # 简化的执行结果显示
                tf_status = "✓" if execution_result['tensorflow_success'] else "✗"
                pd_status = "✓" if execution_result['paddle_success'] else "✗"
                match_status = "✓" if execution_result['results_match'] else "✗"
                self._safe_print(f" | TF:{tf_status} PD:{pd_status} Match:{match_status}")
                
                # 只在有错误时显示错误信息
                if execution_result['tensorflow_error'] and not execution_result['tensorflow_success']:
                    error_short = str(execution_result['tensorflow_error'])[:100]
                    self._safe_print(f"    ❌ TF错误: {error_short}...")
                if execution_result['paddle_error'] and not execution_result['paddle_success']:
                    error_short = str(execution_result['paddle_error'])[:100]
                    self._safe_print(f"    ❌ PD错误: {error_short}...")
                
                # 统计LLM生成的用例
                if is_llm_generated:
                    if execution_result['tensorflow_success'] and execution_result['paddle_success']:
                        stats["successful_cases"] += 1
                
            except Exception as e:
                self._safe_print(f" | ❌ 严重错误: {str(e)[:50]}...")
                
                execution_result = {
                    "status": "fatal_error",
                    "tensorflow_success": False,
                    "paddle_success": False,
                    "results_match": False,
                    "tensorflow_error": f"Fatal error: {str(e)}",
                    "paddle_error": None,
                    "comparison_error": None,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            
            # 保存本次迭代结果
            iteration_result = {
                "iteration": iteration + 1,
                "tensorflow_test_case": current_tensorflow_test_case,
                "paddle_test_case": current_paddle_test_case,
                "execution_result": execution_result,
                "llm_operation": None,
                "is_llm_generated": is_llm_generated
            }
            
            # 调用LLM进行修复或变异
            try:
                llm_result = self.call_llm_for_repair_or_mutation(
                    execution_result, 
                    current_tensorflow_test_case, 
                    current_paddle_test_case,
                    tensorflow_doc,
                    paddle_doc
                )
            except Exception as e:
                self._safe_print(f"    ❌ LLM调用失败: {str(e)[:50]}...")
                
                llm_result = {
                    "operation": "skip",
                    "reason": f"LLM调用失败: {str(e)}"
                }
                
                iteration_result["llm_operation"] = llm_result
                iteration_result["case_number"] = case_number
                case_results.append(iteration_result)
                break
            
            operation = llm_result.get("operation", "skip")
            reason = llm_result.get("reason", "")[:80]  # 截断原因
            
            self._safe_print(f"    🤖 LLM: {operation} - {reason}")
            
            iteration_result["llm_operation"] = {
                "operation": operation,
                "reason": llm_result.get("reason", "")
            }
            
            iteration_result["case_number"] = case_number
            case_results.append(iteration_result)
            
            # 如果LLM建议跳过，则结束迭代
            if operation == "skip":
                break
            
            # 准备下一次迭代的测试用例
            if operation in ["mutation", "repair"]:
                next_tensorflow_test_case = llm_result.get("tensorflow_test_case", current_tensorflow_test_case)
                next_paddle_test_case = llm_result.get("paddle_test_case", current_paddle_test_case)
                stats["llm_generated_cases"] += 1
                is_llm_generated = True
            else:
                next_tensorflow_test_case = current_tensorflow_test_case
                next_paddle_test_case = current_paddle_test_case
            
            # 转换LLM返回的测试用例格式
            current_tensorflow_test_case, current_paddle_test_case = self._convert_llm_test_cases(
                next_tensorflow_test_case, next_paddle_test_case
            )
        
        # 如果最后一次迭代LLM生成了新用例，需要执行这个新用例
        if len(case_results) > 0:
            last_iteration = case_results[-1]
            last_operation = last_iteration["llm_operation"].get("operation", "skip")
            
            if last_operation in ["mutation", "repair"]:
                self._safe_print(f"  🔄 执行最终LLM用例", end="")
                
                try:
                    execution_result = self.execute_test_case(
                        tensorflow_api, paddle_api,
                        current_tensorflow_test_case, current_paddle_test_case
                    )
                    
                    # 简化的执行结果显示
                    tf_status = "✓" if execution_result['tensorflow_success'] else "✗"
                    pd_status = "✓" if execution_result['paddle_success'] else "✗"
                    match_status = "✓" if execution_result['results_match'] else "✗"
                    self._safe_print(f" | TF:{tf_status} PD:{pd_status} Match:{match_status}")
                    
                    if execution_result['tensorflow_success'] and execution_result['paddle_success']:
                        stats["successful_cases"] += 1
                    
                    final_iteration_result = {
                        "iteration": len(case_results) + 1,
                        "tensorflow_test_case": current_tensorflow_test_case,
                        "paddle_test_case": current_paddle_test_case,
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
                    self._safe_print(f" | ❌ 严重错误: {str(e)[:50]}...")
                    
                    final_iteration_result = {
                        "iteration": len(case_results) + 1,
                        "tensorflow_test_case": current_tensorflow_test_case,
                        "paddle_test_case": current_paddle_test_case,
                        "execution_result": {
                            "status": "fatal_error",
                            "tensorflow_success": False,
                            "paddle_success": False,
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
        
        self._safe_print(f"  ✅ 用例 {case_number} 完成，{len(case_results)} 次迭代")
        
        return case_results
    
    def _convert_llm_test_cases(self, tensorflow_test_case: Dict[str, Any], 
                                 paddle_test_case: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        将LLM返回的测试用例转换为可执行格式
        确保两个框架使用相同的张量数据
        """
        # 定义等价参数名映射（这些参数名在不同框架中指代同一输入）
        equivalent_param_names = [
            {"x", "input"},  # 主输入参数
            {"y", "other"},  # 第二输入参数
        ]
        
        def find_equivalent_key(key: str) -> Optional[str]:
            """查找已经生成过的等价参数名"""
            for equiv_set in equivalent_param_names:
                if key in equiv_set:
                    for equiv_key in equiv_set:
                        if equiv_key in shared_tensors:
                            return equiv_key
            return None
        
        # 收集所有需要生成张量的参数，生成共享的numpy数组
        shared_tensors = {}
        
        all_keys = set(tensorflow_test_case.keys()) | set(paddle_test_case.keys())
        
        for key in all_keys:
            if key == "api":
                continue
            
            tensorflow_value = tensorflow_test_case.get(key)
            paddle_value = paddle_test_case.get(key)
            
            is_tensor = False
            tensor_desc = None
            
            if isinstance(tensorflow_value, dict) and "shape" in tensorflow_value:
                is_tensor = True
                tensor_desc = tensorflow_value
            elif isinstance(paddle_value, dict) and "shape" in paddle_value:
                is_tensor = True
                tensor_desc = paddle_value
            
            if is_tensor:
                # 检查是否已经有等价参数名生成过数据
                equiv_key = find_equivalent_key(key)
                if equiv_key is not None:
                    # 复用已有的共享张量
                    shared_tensors[key] = shared_tensors[equiv_key]
                else:
                    numpy_array = self.generate_numpy_data(tensor_desc)
                    shared_tensors[key] = numpy_array
        
        # 分别构建测试用例
        converted_tensorflow = {}
        converted_paddle = {}
        
        for key, value in tensorflow_test_case.items():
            if key in shared_tensors:
                converted_tensorflow[key] = shared_tensors[key]
            else:
                converted_tensorflow[key] = value
        
        for key, value in paddle_test_case.items():
            if key in shared_tensors:
                converted_paddle[key] = shared_tensors[key]
            else:
                converted_paddle[key] = value
        
        return converted_tensorflow, converted_paddle
    
    def save_results(self, operator_name: str, results: List[Dict[str, Any]], stats: Dict[str, int] = None):
        """保存测试结果到JSON文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_enhanced_{operator_name.replace('.', '_')}_{timestamp}.json"
        filepath = os.path.join(self.result_dir, filename)
        
        # 准备输出数据
        output_results = []
        for result in results:
            output_result = copy.deepcopy(result)
            
            # 简化测试用例中的numpy数组
            for test_case_key in ["tensorflow_test_case", "paddle_test_case"]:
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


def process_single_operator(comparator: LLMEnhancedComparator, 
                            pytorch_operator_name: str,
                            max_iterations: int,
                            num_test_cases: int,
                            idx: int,
                            total: int,
                            print_lock: Lock) -> Dict[str, Any]:
    """
    处理单个算子的测试（供并发调用）
    
    Args:
        comparator: LLMEnhancedComparator 实例
        pytorch_operator_name: PyTorch算子名称
        max_iterations: 最大迭代次数
        num_test_cases: 测试用例数量
        idx: 当前索引
        total: 总数
        print_lock: 打印锁
    
    Returns:
        测试结果摘要字典
    """
    with print_lock:
        print(f"\n🔷{'🔷'*39}")
        print(f"🎯 [{idx}/{total}] 开始测试算子: {pytorch_operator_name}")
        print(f"🔷{'🔷'*39}")
    
    try:
        # 使用LLM增强的方式测试算子
        results, stats = comparator.llm_enhanced_test_operator(
            pytorch_operator_name, 
            max_iterations=max_iterations,
            num_test_cases=num_test_cases
        )
        
        # 保存结果
        if results:
            comparator.save_results(pytorch_operator_name, results, stats)
            
            summary = {
                "operator": pytorch_operator_name,
                "total_iterations": len(results),
                "llm_generated_cases": stats.get("llm_generated_cases", 0),
                "successful_cases": stats.get("successful_cases", 0),
                "status": "completed"
            }
            
            with print_lock:
                print(f"\n✅ 算子 {pytorch_operator_name} 测试完成")
                print(f"   - 总迭代次数: {len(results)}")
                print(f"   - LLM生成用例数: {stats.get('llm_generated_cases', 0)}")
                print(f"   - 成功执行用例数: {stats.get('successful_cases', 0)}")
        else:
            summary = {
                "operator": pytorch_operator_name,
                "total_iterations": 0,
                "llm_generated_cases": 0,
                "successful_cases": 0,
                "status": "no_results"
            }
            with print_lock:
                print(f"\n⚠️ 算子 {pytorch_operator_name} 无测试结果")
        
        return summary
        
    except Exception as e:
        with print_lock:
            print(f"\n❌ 算子 {pytorch_operator_name} 测试失败: {e}")
        
        return {
            "operator": pytorch_operator_name,
            "total_iterations": 0,
            "llm_generated_cases": 0,
            "successful_cases": 0,
            "status": "failed",
            "error": str(e)
        }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="基于LLM的TensorFlow与PaddlePaddle算子比较测试框架（并发版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 测试所有算子，默认参数
  python llm_enhanced_compare.py

  # 指定迭代次数和用例数
  python llm_enhanced_compare.py -m 5 -n 5

  # 测试指定范围的算子（第10到第20个）
  python llm_enhanced_compare.py --start 10 --end 20

  # 测试指定的算子
  python llm_enhanced_compare.py -o torch.abs torch.add torch.mul

  # 使用8个线程并发测试
  python llm_enhanced_compare.py -w 8

  # 组合使用多个参数
  python llm_enhanced_compare.py -m 3 -n 3 --start 1 --end 50 -w 4
        """
    )
    
    parser.add_argument(
        "-m", "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"每个测试用例的最大迭代次数（默认: {DEFAULT_MAX_ITERATIONS}）"
    )
    
    parser.add_argument(
        "-n", "--num-cases",
        type=int,
        default=DEFAULT_NUM_CASES,
        help=f"每个算子要测试的用例数量（默认: {DEFAULT_NUM_CASES}）"
    )
    
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="起始算子索引，从1开始（默认: 1）"
    )
    
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="结束算子索引，包含该索引（默认: 全部）"
    )
    
    parser.add_argument(
        "-o", "--operators",
        nargs="+",
        default=None,
        help="指定要测试的算子名称（PyTorch格式），多个算子用空格分隔"
    )
    
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并发线程数（默认: {DEFAULT_WORKERS}）"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM模型名称（默认: {DEFAULT_MODEL}）"
    )
    
    parser.add_argument(
        "-k", "--key-path",
        type=str,
        default=DEFAULT_KEY_PATH,
        help=f"API key文件路径（默认: {DEFAULT_KEY_PATH}）"
    )
    
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="使用顺序执行模式（不使用并发）"
    )
    
    return parser.parse_args()


def main():
    """
    主函数
    
    支持命令行参数配置，提供并发和顺序两种执行模式。
    """
    args = parse_args()
    
    # 提取参数
    max_iterations = args.max_iterations
    num_test_cases = args.num_cases
    start_idx = args.start
    end_idx = args.end
    specified_operators = args.operators
    num_workers = args.workers
    model = args.model
    key_path = args.key_path
    use_sequential = args.sequential
    
    # 创建打印锁
    print_lock = Lock()
    
    print("="*80)
    print("基于LLM的TensorFlow与PaddlePaddle算子批量比较测试框架（并发版本）")
    print("="*80)
    print(f"📌 每个算子的迭代次数: {max_iterations}")
    print(f"📌 每个算子的测试用例数: {num_test_cases}")
    print(f"📌 并发线程数: {num_workers if not use_sequential else '顺序执行'}")
    print(f"📌 LLM模型: {model}")
    print(f"📌 API密钥文件: {key_path}")
    if specified_operators:
        print(f"📌 指定测试算子: {', '.join(specified_operators)}")
    elif end_idx is not None:
        print(f"📌 测试范围: 第 {start_idx} 到第 {end_idx} 个算子")
    else:
        print(f"📌 测试范围: 从第 {start_idx} 个开始的所有算子")
    print("="*80)
    
    # 初始化比较器
    comparator = LLMEnhancedComparator(
        key_path=key_path,
        model=model,
        print_lock=print_lock
    )
    
    # 记录开始时间
    start_time = time.time()
    start_datetime = datetime.now()
    
    try:
        # 确定要测试的算子列表
        if specified_operators:
            # 使用指定的算子
            operator_names = specified_operators
            print(f"\n📋 将测试指定的 {len(operator_names)} 个算子")
        else:
            # 获取数据库中所有的PyTorch算子
            print("\n🔍 正在获取数据库中的所有算子...")
            all_operators = list(comparator.collection.find({}, {"api": 1}))
            all_operator_names = [doc["api"] for doc in all_operators if "api" in doc]
            
            print(f"✅ 数据库中共有 {len(all_operator_names)} 个算子")
            
            # 根据范围过滤算子
            actual_start = max(1, start_idx) - 1
            actual_end = min(len(all_operator_names), end_idx) if end_idx else len(all_operator_names)
            operator_names = all_operator_names[actual_start:actual_end]
            print(f"📌 测试范围: 第 {actual_start + 1} 到第 {actual_end} 个算子")
            print(f"📋 将测试 {len(operator_names)} 个算子")
        
        # 过滤无对应实现的算子
        print(f"\n🔍 过滤无 TensorFlow 或 PaddlePaddle 对应实现的算子...")
        original_count = len(operator_names)
        filtered_operator_names = []
        skipped_operators = []
        
        for op_name in operator_names:
            tf_api, paddle_api, mapping_method = comparator.convert_api_name(op_name)
            if tf_api is not None and paddle_api is not None:
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
        log_file.write("TensorFlow与PaddlePaddle批量测试总日志\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"测试配置:\n")
        log_file.write(f"  - 每个算子的迭代次数: {max_iterations}\n")
        log_file.write(f"  - 每个算子的测试用例数: {num_test_cases}\n")
        log_file.write(f"  - 并发线程数: {num_workers if not use_sequential else '顺序执行'}\n")
        log_file.write(f"  - 跳过的无对应实现算子数: {skipped_count}\n")
        log_file.write(f"  - 实际测试算子数: {len(operator_names)}\n")
        log_file.write("="*80 + "\n\n")
        log_file.flush()
        
        total_operators = len(operator_names)
        
        if use_sequential:
            # 顺序执行模式
            for idx, pytorch_operator_name in enumerate(operator_names, 1):
                summary = process_single_operator(
                    comparator, pytorch_operator_name, max_iterations, num_test_cases,
                    idx, total_operators, print_lock
                )
                all_operators_summary.append(summary)
                
                # 写入日志
                log_file.write(f"[{idx}/{total_operators}] {pytorch_operator_name}\n")
                log_file.write(f"  状态: {summary['status']}\n")
                log_file.write(f"  总迭代次数: {summary['total_iterations']}\n")
                log_file.write(f"  LLM生成用例数: {summary['llm_generated_cases']}\n")
                log_file.write(f"  成功执行用例数: {summary['successful_cases']}\n")
                if summary.get('error'):
                    log_file.write(f"  错误: {summary['error']}\n")
                log_file.write("\n")
                log_file.flush()
        else:
            # 并发执行模式
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_op = {}
                for idx, pytorch_operator_name in enumerate(operator_names, 1):
                    future = executor.submit(
                        process_single_operator,
                        comparator, pytorch_operator_name, max_iterations, num_test_cases,
                        idx, total_operators, print_lock
                    )
                    future_to_op[future] = (idx, pytorch_operator_name)
                
                for future in as_completed(future_to_op):
                    idx, pytorch_operator_name = future_to_op[future]
                    try:
                        summary = future.result()
                        all_operators_summary.append(summary)
                        
                        # 写入日志
                        log_file.write(f"[{idx}/{total_operators}] {pytorch_operator_name}\n")
                        log_file.write(f"  状态: {summary['status']}\n")
                        log_file.write(f"  总迭代次数: {summary['total_iterations']}\n")
                        log_file.write(f"  LLM生成用例数: {summary['llm_generated_cases']}\n")
                        log_file.write(f"  成功执行用例数: {summary['successful_cases']}\n")
                        if summary.get('error'):
                            log_file.write(f"  错误: {summary['error']}\n")
                        log_file.write("\n")
                        log_file.flush()
                    except Exception as e:
                        with print_lock:
                            print(f"❌ 获取算子 {pytorch_operator_name} 的结果时出错: {e}")
                        all_operators_summary.append({
                            "operator": pytorch_operator_name,
                            "total_iterations": 0,
                            "llm_generated_cases": 0,
                            "successful_cases": 0,
                            "status": "failed",
                            "error": str(e)
                        })
        
        # 记录结束时间
        end_time = time.time()
        end_datetime = datetime.now()
        total_time = end_time - start_time
        
        # 计算总体统计
        total_iterations = sum(s["total_iterations"] for s in all_operators_summary)
        total_llm_cases = sum(s["llm_generated_cases"] for s in all_operators_summary)
        total_successful = sum(s["successful_cases"] for s in all_operators_summary)
        completed_count = sum(1 for s in all_operators_summary if s["status"] == "completed")
        failed_count = sum(1 for s in all_operators_summary if s["status"] == "failed")
        no_results_count = sum(1 for s in all_operators_summary if s["status"] == "no_results")
        
        # 打印总结
        print("\n" + "="*80)
        print("📊 批量测试总结")
        print("="*80)
        print(f"测试时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        print(f"测试算子数: {len(operator_names)}")
        print(f"  - 完成: {completed_count}")
        print(f"  - 失败: {failed_count}")
        print(f"  - 无结果: {no_results_count}")
        print(f"总迭代次数: {total_iterations}")
        print(f"LLM生成用例总数: {total_llm_cases}")
        print(f"成功执行用例总数: {total_successful}")
        if total_llm_cases > 0:
            print(f"总体成功率: {(total_successful/total_llm_cases)*100:.2f}%")
        print("="*80)
        
        # 写入日志总结
        log_file.write("="*80 + "\n")
        log_file.write("批量测试总结\n")
        log_file.write("="*80 + "\n")
        log_file.write(f"测试时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)\n")
        log_file.write(f"测试算子数: {len(operator_names)}\n")
        log_file.write(f"  - 完成: {completed_count}\n")
        log_file.write(f"  - 失败: {failed_count}\n")
        log_file.write(f"  - 无结果: {no_results_count}\n")
        log_file.write(f"总迭代次数: {total_iterations}\n")
        log_file.write(f"LLM生成用例总数: {total_llm_cases}\n")
        log_file.write(f"成功执行用例总数: {total_successful}\n")
        if total_llm_cases > 0:
            log_file.write(f"总体成功率: {(total_successful/total_llm_cases)*100:.2f}%\n")
        log_file.write("="*80 + "\n")
        log_file.close()
        
        # 保存批量测试摘要
        summary_file = os.path.join(comparator.result_dir, f"batch_test_summary_{start_datetime.strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "start_time": start_datetime.isoformat(),
                "end_time": end_datetime.isoformat(),
                "total_time_seconds": total_time,
                "total_operators": len(operator_names),
                "completed": completed_count,
                "failed": failed_count,
                "no_results": no_results_count,
                "total_iterations": total_iterations,
                "total_llm_cases": total_llm_cases,
                "total_successful": total_successful,
                "success_rate": (total_successful/total_llm_cases)*100 if total_llm_cases > 0 else 0,
                "operators": all_operators_summary
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 批量测试日志已保存到: {batch_log_file}")
        print(f"💾 批量测试摘要已保存到: {summary_file}")
        
    finally:
        comparator.close()
        print("\n✅ 程序执行完成")


if __name__ == "__main__":
    main()
