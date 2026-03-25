"""  
PyTorch-TensorFlow 基于 LLM 的 Fuzzing 差分测试工具（并发版本）

功能说明:
    1. 读取按算子分类的成功测试用例
    2. 爬取 PyTorch 和 TensorFlow 官方文档
    3. 使用 LLM 变异测试用例（复杂输入、极端值、边界值）
    4. 执行差分测试，检测框架不一致或潜在 bug
    5. 每个用例进行 3 轮 fuzzing
    6. 支持多线程并发处理（默认8个线程）
"""

import json
import sys
import argparse
import traceback
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_crawler_factory import get_doc_content
from component.migration.migrate_generate_tests import get_qwen_client

# ==================== 常量定义 ====================
DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"
FUZZING_ROUNDS = 3  # 每个用例的 fuzzing 轮数
DEFAULT_WORKERS = 8  # 默认并发线程数

# 成功用例目录
SUCCESS_CASES_DIR = Path(__file__).parent / "success_cases"
# Fuzzing 结果目录
RESULT_DIR = Path(__file__).parent / "result"

# 需要数据格式转换的算子（NCHW <-> NHWC）
CONV_OPS_NEED_TRANSPOSE = {
    # 卷积相关
    "torch.nn.Conv1d", "torch.nn.Conv2d", "torch.nn.Conv3d",
    "torch.nn.ConvTranspose1d", "torch.nn.ConvTranspose2d", "torch.nn.ConvTranspose3d",
    "torch.nn.functional.conv1d", "torch.nn.functional.conv2d", "torch.nn.functional.conv3d",
    "torch.nn.functional.conv_transpose1d", "torch.nn.functional.conv_transpose2d", "torch.nn.functional.conv_transpose3d",
    # 池化相关
    "torch.nn.MaxPool1d", "torch.nn.MaxPool2d", "torch.nn.MaxPool3d",
    "torch.nn.AvgPool1d", "torch.nn.AvgPool2d", "torch.nn.AvgPool3d",
    "torch.nn.AdaptiveMaxPool1d", "torch.nn.AdaptiveMaxPool2d", "torch.nn.AdaptiveMaxPool3d",
    "torch.nn.AdaptiveAvgPool1d", "torch.nn.AdaptiveAvgPool2d", "torch.nn.AdaptiveAvgPool3d",
    "torch.nn.functional.max_pool1d", "torch.nn.functional.max_pool2d", "torch.nn.functional.max_pool3d",
    "torch.nn.functional.avg_pool1d", "torch.nn.functional.avg_pool2d", "torch.nn.functional.avg_pool3d",
    # 归一化相关
    "torch.nn.BatchNorm1d", "torch.nn.BatchNorm2d", "torch.nn.BatchNorm3d",
    "torch.nn.InstanceNorm1d", "torch.nn.InstanceNorm2d", "torch.nn.InstanceNorm3d",
}


def needs_data_format_conversion(api_name: str) -> bool:
    """
    判断算子是否需要数据格式转换（NCHW <-> NHWC）
    """
    return api_name in CONV_OPS_NEED_TRANSPOSE


def convert_nchw_to_nhwc(tensor: np.ndarray) -> np.ndarray:
    """
    将 NCHW 格式转换为 NHWC 格式
    
    - 4D: (N, C, H, W) -> (N, H, W, C)
    - 3D: (N, C, L) -> (N, L, C)
    - 5D: (N, C, D, H, W) -> (N, D, H, W, C)
    """
    ndim = tensor.ndim
    if ndim == 4:  # 2D 卷积/池化
        return np.transpose(tensor, (0, 2, 3, 1))
    elif ndim == 3:  # 1D 卷积/池化
        return np.transpose(tensor, (0, 2, 1))
    elif ndim == 5:  # 3D 卷积/池化
        return np.transpose(tensor, (0, 2, 3, 4, 1))
    else:
        return tensor


def convert_nhwc_to_nchw(tensor: np.ndarray) -> np.ndarray:
    """
    将 NHWC 格式转换为 NCHW 格式
    
    - 4D: (N, H, W, C) -> (N, C, H, W)
    - 3D: (N, L, C) -> (N, C, L)
    - 5D: (N, D, H, W, C) -> (N, C, D, H, W)
    """
    ndim = tensor.ndim
    if ndim == 4:  # 2D 卷积/池化
        return np.transpose(tensor, (0, 3, 1, 2))
    elif ndim == 3:  # 1D 卷积/池化
        return np.transpose(tensor, (0, 2, 1))
    elif ndim == 5:  # 3D 卷积/池化
        return np.transpose(tensor, (0, 4, 1, 2, 3))
    else:
        return tensor


def build_fuzzing_prompt(
    torch_api: str,
    tf_api: str,
    original_case: Dict[str, Any],
    torch_doc: str,
    tf_doc: str,
    round_num: int
) -> str:
    """
    构建 LLM fuzzing 提示词
    
    参数:
        torch_api: PyTorch API 名称
        tf_api: TensorFlow API 名称
        original_case: 原始测试用例
        torch_doc: PyTorch 文档内容
        tf_doc: TensorFlow 文档内容
        round_num: 当前 fuzzing 轮次
    """
    torch_case = original_case.get("torch_test_case", {})
    tf_case = original_case.get("tensorflow_test_case", {})
    
    torch_case_json = json.dumps(torch_case, ensure_ascii=False, indent=2)
    tf_case_json = json.dumps(tf_case, ensure_ascii=False, indent=2)
    
    # 检测是否是逻辑运算算子
    is_logical_op = any(keyword in torch_api.lower() for keyword in ["logical_", "bitwise_"])
    
    # 不同轮次的变异策略
    mutation_strategies = {
        1: "极端数值变异：使用复杂的浮点数、极大值(1e38)、极小值(1e-38)、无穷大(inf)、负无穷(-inf)、NaN、零(0)、负零(-0.0)等特殊数值",
        2: "边界形状变异：使用空张量(shape含0)、标量(shape=[])、超高维张量(5维以上)、不规则形状、单元素张量等边界情况",
        3: "复杂类型变异：测试不同数据类型(float16/float32/float64/int32/int64/bool/complex64/complex128)、混合精度场景"
    }
    
    current_strategy = mutation_strategies.get(round_num, mutation_strategies[1])
    
    # 为逻辑运算添加特殊提示
    dtype_constraint = ""
    if is_logical_op:
        dtype_constraint = """
【重要：数据类型约束】
**此算子是逻辑运算，必须使用 bool 类型！**
- PyTorch 和 TensorFlow 的逻辑运算（如 logical_and, logical_or, logical_xor）都要求输入为 bool 类型
- 变异时必须保持 dtype 为 "bool"
- 不要使用 float/int 等其他类型，否则会导致执行失败
"""
    
    # 检测是否是需要注意数据格式的算子
    needs_format_attention = needs_data_format_conversion(torch_api)
    
    # 数据格式说明（仅对需要的算子添加）
    data_format_instruction = ""
    if needs_format_attention:
        data_format_instruction = """
【重要：数据格式差异】
**PyTorch 和 TensorFlow 对卷积/池化/归一化算子使用不同的数据格式：**
- PyTorch 使用 NCHW 格式（channels_first）：
  - 4D: (N, C, H, W) - batch, channels, height, width
  - 3D: (N, C, L) - batch, channels, length
  - 5D: (N, C, D, H, W) - batch, channels, depth, height, width
  
- TensorFlow 默认使用 NHWC 格式（channels_last）：
  - 4D: (N, H, W, C) - batch, height, width, channels
  - 3D: (N, L, C) - batch, length, channels  
  - 5D: (N, D, H, W, C) - batch, depth, height, width, channels

**生成测试用例时必须：**
1. PyTorch 测试用例的 input shape 使用 NCHW 格式
2. TensorFlow 测试用例的 input shape 使用 NHWC 格式
3. 确保两者的数据在数学上等价（只是排列顺序不同）
4. sample_values 的填充顺序要对应各自的格式
5. **TensorFlow Keras 层的 data_format 参数只接受以下值（严格区分大小写）：**
   - "channels_last" （对应 NHWC/NWC/NDHWC）
   - "channels_first" （对应 NCHW/NCW/NCDHW）
   - **不要使用 "NHWC", "NCHW", "NWC" 等，这些会导致报错！**

例如，对于 batch=1, channels=3, height=4, width=4 的输入：
- PyTorch shape: [1, 3, 4, 4]
- TensorFlow shape: [1, 4, 4, 3]，并设置 data_format: "channels_last"
"""
    
    prompt = f"""你是一个专业的深度学习框架测试专家，现在需要对 PyTorch 和 TensorFlow 的等价算子进行差分测试。

【测试目标】
我们要寻找的是**框架实现中的潜在 bug**，即：在数学上完全等价的操作，两框架却产生不同的输出。
- ✓ 要找的：相同数学计算下的框架实现差异（真正的 bug）
- ✗ 不要找的：因为测试用例本身不等价导致的差异（假阳性）
{data_format_instruction}
【当前测试的算子对】
- PyTorch API: {torch_api}
- TensorFlow API: {tf_api}

【原始成功测试用例】
PyTorch 测试用例:
```json
{torch_case_json}
```

TensorFlow 测试用例:
```json
{tf_case_json}
```

【PyTorch 官方文档】
{torch_doc if torch_doc else "文档获取失败"}

【TensorFlow 官方文档】
{tf_doc if tf_doc else "文档获取失败"}

【本轮变异策略 - 第{round_num}轮】
{current_strategy}

{dtype_constraint}

【核心变异要求：数学等价性（最重要！）】
**变异后的两个测试用例必须在数学上完全等价，理论输出必须相同。**

【关键要求：参数完整性】
**必须包含原始测试用例中的所有必需参数！**
- 如果原始用例有 `other` 参数，变异后的用例也必须有 `other` 参数
- 如果原始用例有 `dim`/`axis` 参数，变异后的用例也必须有对应参数
- 如果原始用例有其他配置参数（如 `kernel_size`, `stride` 等），变异后也必须保留
- **不要遗漏任何必需参数，否则会导致执行失败！**

【输出格式要求】
请严格按照以下 JSON 格式输出变异后的测试用例，不要输出任何其他内容：

**重要：特殊浮点值的 JSON 表示**
- 正无穷：使用字符串 "inf" 或 "Infinity"
- 负无穷：使用字符串 "-inf" 或 "-Infinity"  
- NaN：使用字符串 "nan" 或 "NaN"
- 负零：使用数值 -0.0
- 不要使用 Python 语法如 float('inf')，这在 JSON 中是非法的！

```json
{{
  "mutation_strategy": "简要描述本次变异策略（不超过50字）",
  "mutation_reason": "简要解释为什么这样变异可能发现问题（不超过100字）",
  "torch_test_case": {{
    "api": "{torch_api}",
    "input": {{
      "shape": [...],
      "dtype": "...",
      "sample_values": [1.0, -1.0, "inf", "-inf", "nan", -0.0, 1e-38, 1e38]
    }},
    "other": {{
      "shape": [...],
      "dtype": "...",
      "sample_values": [...]
    }},
    "dim": ...,
    "其他必需参数": ...
  }},
  "tensorflow_test_case": {{
    "api": "{tf_api}",
    "input": {{
      "shape": [...],
      "dtype": "...",
      "sample_values": [1.0, -1.0, "inf", "-inf", "nan", -0.0, 1e-38, 1e38]
    }},
    "other": {{
      "shape": [...],
      "dtype": "...",
      "sample_values": [...]
    }},
    "axis": ...,
    "其他必需参数": ...
  }}
}}
```

**注意**：
1. mutation_strategy 和 mutation_reason 要简洁，避免过长导致 token 超限
2. 特殊值必须用字符串表示（"inf", "-inf", "nan"）
3. 确保 JSON 格式完整，所有括号和引号都要闭合
4. **必须保留原始用例中的所有参数（如 other, dim, axis 等）**
5. **PyTorch 和 TensorFlow 测试用例的所有输入张量必须使用相同的 dtype（TensorFlow 不支持混合精度）**
"""
    return prompt


def parse_llm_response(response: str) -> Optional[Dict[str, Any]]:
    """
    解析 LLM 返回的 JSON 响应，支持处理截断和格式错误
    """
    try:
        # 尝试提取 JSON 块
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end == -1:
                json_str = response[start:].strip()
            else:
                json_str = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end == -1:
                json_str = response[start:].strip()
            else:
                json_str = response[start:end].strip()
        else:
            json_str = response.strip()
        
        # 预处理：替换 Python float() 语法为 JSON 字符串
        json_str = fix_python_float_syntax(json_str)
        
        # 尝试直接解析
        try:
            parsed = json.loads(json_str)
            # 验证必要字段
            if validate_parsed_json(parsed):
                return parsed
        except json.JSONDecodeError:
            pass
        
        # 尝试修复截断的 JSON
        repaired_json = try_repair_json(json_str)
        if repaired_json and validate_parsed_json(repaired_json):
            return repaired_json
        
        return None
    except Exception as e:
        print(f"[WARN] 解析 LLM 响应失败: {e}")
        return None


def fix_python_float_syntax(json_str: str) -> str:
    """
    将 Python float() 语法替换为 JSON 兼容的字符串格式
    
    例如:
        float('inf') -> "inf"
        float('-inf') -> "-inf"
        float('nan') -> "nan"
    """
    import re
    
    # 替换 float('inf')
    json_str = re.sub(r"float\s*\(\s*['\"]inf['\"]\s*\)", '"inf"', json_str, flags=re.IGNORECASE)
    
    # 替换 float('-inf')
    json_str = re.sub(r"float\s*\(\s*['\"]-inf['\"]\s*\)", '"-inf"', json_str, flags=re.IGNORECASE)
    
    # 替换 float('nan')
    json_str = re.sub(r"float\s*\(\s*['\"]nan['\"]\s*\)", '"nan"', json_str, flags=re.IGNORECASE)
    
    # 替换 float('+inf')
    json_str = re.sub(r"float\s*\(\s*['\"]\+inf['\"]\s*\)", '"inf"', json_str, flags=re.IGNORECASE)
    
    return json_str


def validate_parsed_json(parsed: Dict[str, Any]) -> bool:
    """
    验证解析后的 JSON 是否包含必要字段
    
    返回:
        True 如果包含必要字段，False 否则
    """
    if not isinstance(parsed, dict):
        return False
    
    # 检查必要字段
    required_fields = ["torch_test_case", "tensorflow_test_case"]
    for field in required_fields:
        if field not in parsed:
            return False
        if not isinstance(parsed[field], dict):
            return False
    
    return True


def try_repair_json(json_str: str) -> Optional[Dict[str, Any]]:
    """
    尝试修复不完整的 JSON 字符串
    
    处理常见的截断情况:
    1. 未闭合的数组和对象
    2. 截断的字符串值
    3. 截断的数值
    4. 截断的特殊值（如 float('inf')）
    """
    import re
    
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    
    repaired = json_str
    
    # 修复 Python float() 语法（如果还没修复）
    repaired = fix_python_float_syntax(repaired)
    
    # 尝试多种修复策略
    patterns_to_try = [
        # 移除截断的字符串值（未闭合的引号）
        (r',?\s*"[^"]*"\s*:\s*"[^"]*$', ''),
        # 移除截断的键值对（只有键没有值）
        (r',?\s*"[^"]*"\s*:\s*$', ''),
        # 移除尾部逗号
        (r',\s*$', ''),
        # 移除截断的数值
        (r',?\s*\d+\.?\d*e?[+-]?\d*\s*$', ''),
        # 移除截断的数组元素（包括特殊值）
        (r',?\s*("inf"|"-inf"|"nan"|float\([^)]*\))?\s*$', ''),
        # 移除截断的 sample_values 数组（更宽松的匹配）
        (r'"sample_values"\s*:\s*\[[^\]]*$', '"sample_values": []'),
        # 移除截断的 input 对象
        (r'"input"\s*:\s*\{[^}]*$', '"input": {"shape": [], "dtype": "float32", "sample_values": []}'),
    ]
    
    for pattern, replacement in patterns_to_try:
        test_str = re.sub(pattern, replacement, repaired)
        
        # 计算需要闭合的括号数量
        test_open_braces = test_str.count('{')
        test_close_braces = test_str.count('}')
        test_open_brackets = test_str.count('[')
        test_close_brackets = test_str.count(']')
        
        # 闭合所有未闭合的括号
        test_str += ']' * (test_open_brackets - test_close_brackets)
        test_str += '}' * (test_open_braces - test_close_braces)
        
        try:
            result = json.loads(test_str)
            if validate_parsed_json(result):
                return result
        except json.JSONDecodeError:
            continue
    
    # 最后尝试：直接闭合所有括号
    repaired += ']' * (open_brackets - close_brackets)
    repaired += '}' * (open_braces - close_braces)
    
    try:
        result = json.loads(repaired)
        if validate_parsed_json(result):
            return result
    except json.JSONDecodeError:
        pass
    
    return None


def parse_special_value(val: Any) -> float:
    """
    解析特殊值（如 "inf", "-inf", "nan" 字符串）
    """
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val_lower = val.lower().strip()
        if val_lower == "inf" or val_lower == "+inf":
            return np.inf
        elif val_lower == "-inf":
            return -np.inf
        elif val_lower == "nan":
            return np.nan
        else:
            try:
                return float(val)
            except ValueError:
                return 0.0
    return 0.0


def create_tensor_from_spec(spec: Dict[str, Any], framework: str) -> Any:
    """
    根据规格创建张量
    
    参数:
        spec: 包含 shape, dtype, sample_values 的字典
        framework: 'torch' 或 'tensorflow'
    """
    shape = spec.get("shape", [])
    dtype_str = spec.get("dtype", "float32")
    sample_values = spec.get("sample_values", [])
    
    # 预处理 sample_values，转换特殊值
    processed_values = [parse_special_value(v) for v in sample_values]
    
    # 计算张量大小
    size = 1
    for dim in shape:
        size *= dim
    
    # 生成数据
    if size == 0:
        data = np.array([]).reshape(shape)
    elif processed_values:
        if len(processed_values) >= size:
            data = np.array(processed_values[:size]).reshape(shape)
        else:
            # 使用 processed_values 作为种子扩展
            np.random.seed(42)  # 固定种子保证一致性
            if processed_values:
                # 过滤掉 NaN 和 Inf 用于计算统计量
                finite_values = [v for v in processed_values if np.isfinite(v)]
                if finite_values:
                    mean = np.mean(finite_values)
                    std = np.std(finite_values) if len(finite_values) > 1 else 1.0
                else:
                    mean, std = 0.0, 1.0
                
                # 生成基础数据
                base_data = np.random.normal(mean, std, size)
                
                # 将 sample_values 中的值按顺序填入
                for i, v in enumerate(processed_values):
                    if i < size:
                        base_data.flat[i] = v
                
                data = base_data.reshape(shape)
            else:
                data = np.random.randn(*shape)
    else:
        np.random.seed(42)
        data = np.random.randn(*shape)
    
    # 转换数据类型
    dtype_map = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
        "bool": np.bool_,
        "complex64": np.complex64,
        "complex128": np.complex128,
    }
    
    np_dtype = dtype_map.get(dtype_str, np.float32)
    
    # 处理特殊值
    if np_dtype in [np.float16, np.float32, np.float64]:
        data = data.astype(np_dtype)
    elif np_dtype == np.bool_:
        data = (data > 0).astype(np.bool_)
    elif np_dtype in [np.complex64, np.complex128]:
        data = (data + 1j * np.random.randn(*shape)).astype(np_dtype)
    else:
        data = data.astype(np_dtype)
    
    return data, dtype_str


def create_torch_tensor(spec: Any, torch_dtype_map: Dict) -> Any:
    """
    根据规格创建 PyTorch 张量，支持张量和标量
    """
    import torch
    
    if isinstance(spec, (int, float, bool)):
        return spec
    if isinstance(spec, dict) and "shape" in spec:
        data, dtype_str = create_tensor_from_spec(spec, "torch")
        torch_dtype = torch_dtype_map.get(dtype_str, torch.float32)
        return torch.tensor(data, dtype=torch_dtype)
    return spec


def execute_torch_test(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行 PyTorch 测试用例，支持多输入和额外参数
    """
    try:
        import torch
        
        api_name = test_case.get("api", "")
        
        # PyTorch dtype 映射
        torch_dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
            "complex64": torch.complex64,
            "complex128": torch.complex128,
        }
        
        # 保留的非张量参数名（这些是算子的配置参数，不是输入张量）
        non_tensor_params = {
            "kernel_size", "stride", "padding", "dilation", "groups",
            "ceil_mode", "count_include_pad", "divisor_override",
            "alpha", "dim", "keepdim", "dtype", "out", "axis",
            "p", "eps", "momentum", "affine", "track_running_stats",
            "num_features", "normalized_shape", "weight", "bias"
        }
        
        # 分离输入张量和参数
        args = []
        kwargs = {}
        
        # 处理主输入
        if "input" in test_case:
            input_tensor = create_torch_tensor(test_case["input"], torch_dtype_map)
            args.append(input_tensor)
        
        # 处理其他可能的输入张量（如 other, x, y 等）
        tensor_input_names = ["other", "x", "y", "tensor", "input1", "input2", "mat1", "mat2"]
        for name in tensor_input_names:
            if name in test_case:
                tensor = create_torch_tensor(test_case[name], torch_dtype_map)
                if name == "other":
                    kwargs["other"] = tensor
                else:
                    args.append(tensor)
        
        # 处理非张量参数
        for key, value in test_case.items():
            if key not in ["api", "input", "x", "y", "other", "tensor", "input1", "input2", "mat1", "mat2"]:
                if key in non_tensor_params or not isinstance(value, dict):
                    kwargs[key] = value
        
        # 获取 API 函数
        api_parts = api_name.split(".")
        func = torch
        for part in api_parts[1:]:  # 跳过 'torch'
            func = getattr(func, part)
        
        # 判断是否是类（如 nn.AvgPool1d）还是函数（如 torch.abs）
        if isinstance(func, type):
            # 类：先实例化再调用
            # 从 kwargs 中提取初始化参数
            init_params = {k: v for k, v in kwargs.items() if k in non_tensor_params}
            instance = func(**init_params)
            result = instance(args[0] if args else kwargs.get("input"))
        else:
            # 函数：直接调用
            result = func(*args, **kwargs)
        
        # 转换结果
        if isinstance(result, torch.Tensor):
            result_np = result.detach().cpu().numpy()
            return {
                "success": True,
                "result": result_np,
                "shape": list(result.shape),
                "dtype": str(result.dtype),
                "error": None
            }
        else:
            return {
                "success": True,
                "result": result,
                "shape": None,
                "dtype": str(type(result)),
                "error": None
            }
            
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "shape": None,
            "dtype": None,
            "error": f"{type(e).__name__}: {str(e)}"
        }


def create_tf_tensor(spec: Any, tf_dtype_map: Dict) -> Any:
    """
    根据规格创建 TensorFlow 张量，支持张量和标量
    """
    import tensorflow as tf
    
    if isinstance(spec, (int, float, bool)):
        return spec
    if isinstance(spec, dict) and "shape" in spec:
        data, dtype_str = create_tensor_from_spec(spec, "tensorflow")
        tf_dtype = tf_dtype_map.get(dtype_str, tf.float32)
        return tf.constant(data, dtype=tf_dtype)
    return spec


def execute_tensorflow_test(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行 TensorFlow 测试用例，支持多输入和额外参数
    """
    try:
        import tensorflow as tf
        
        api_name = test_case.get("api", "")
        
        # TensorFlow dtype 映射
        tf_dtype_map = {
            "float16": tf.float16,
            "float32": tf.float32,
            "float64": tf.float64,
            "int32": tf.int32,
            "int64": tf.int64,
            "bool": tf.bool,
            "complex64": tf.complex64,
            "complex128": tf.complex128,
        }
        
        # TensorFlow 的非张量参数名
        non_tensor_params = {
            "pool_size", "strides", "padding", "data_format",
            "axis", "keepdims", "name", "dtype",
            "epsilon", "center", "scale", "beta_initializer", "gamma_initializer",
            "filters", "kernel_size", "activation", "use_bias"
        }
        
        # 分离输入张量和参数
        args = []
        kwargs = {}
        
        # 处理主输入（LLM 已经生成了正确格式的数据，无需手动转换）
        if "input" in test_case:
            input_tensor = create_tf_tensor(test_case["input"], tf_dtype_map)
            args.append(input_tensor)
        
        # 处理其他可能的输入张量
        tensor_input_names = ["other", "x", "y", "tensor", "input1", "input2"]
        for name in tensor_input_names:
            if name in test_case:
                tensor = create_tf_tensor(test_case[name], tf_dtype_map)
                args.append(tensor)
        
        # 处理非张量参数（包括 LLM 生成的 data_format 参数）
        for key, value in test_case.items():
            if key not in ["api", "input", "x", "y", "other", "tensor", "input1", "input2"]:
                if key in non_tensor_params or not isinstance(value, dict):
                    kwargs[key] = value
        
        # 获取 API 函数/类
        api_parts = api_name.split(".")
        func = tf
        for part in api_parts[1:]:  # 跳过 'tf'
            func = getattr(func, part)
        
        # 判断是否是 Keras 层（如 tf.keras.layers.AveragePooling1D）
        is_keras_layer = "keras" in api_name and "layers" in api_name
        
        if is_keras_layer:
            # Keras 层：先实例化再调用
            init_params = {k: v for k, v in kwargs.items() if k in non_tensor_params}
            layer = func(**init_params)
            result = layer(args[0] if args else kwargs.get("input"))
        else:
            # 普通函数：直接调用
            if args and not kwargs:
                result = func(*args)
            elif args:
                result = func(*args, **kwargs)
            else:
                result = func(**kwargs)
        
        # 转换结果（LLM 已确保输入数学等价，输出可直接比较数值）
        if isinstance(result, tf.Tensor):
            result_np = result.numpy()
            return {
                "success": True,
                "result": result_np,
                "shape": list(result.shape),
                "dtype": str(result.dtype),
                "error": None
            }
        else:
            return {
                "success": True,
                "result": result,
                "shape": None,
                "dtype": str(type(result)),
                "error": None
            }
            
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "shape": None,
            "dtype": None,
            "error": f"{type(e).__name__}: {str(e)}"
        }


def compare_results(
    torch_result: Dict[str, Any], 
    tf_result: Dict[str, Any],
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> Dict[str, Any]:
    """
    比较两个框架的执行结果
    """
    comparison = {
        "torch_success": torch_result["success"],
        "tensorflow_success": tf_result["success"],
        "torch_error": torch_result["error"],
        "tensorflow_error": tf_result["error"],
        "results_match": False,
        "comparison_error": None,
        "torch_shape": torch_result["shape"],
        "tensorflow_shape": tf_result["shape"],
        "torch_dtype": torch_result["dtype"],
        "tensorflow_dtype": tf_result["dtype"],
    }
    
    # 如果有一方失败
    if not torch_result["success"] or not tf_result["success"]:
        if torch_result["success"] != tf_result["success"]:
            comparison["comparison_error"] = "执行状态不一致：一方成功一方失败"
        return comparison
    
    # 比较结果
    try:
        torch_res = torch_result["result"]
        tf_res = tf_result["result"]
        
        if torch_res is None and tf_res is None:
            comparison["results_match"] = True
            return comparison
        
        # 处理标量的情况（包括 NaN）- 扩展类型判断以支持 numpy 标量类型
        is_torch_scalar = isinstance(torch_res, (int, float, np.integer, np.floating))
        is_tf_scalar = isinstance(tf_res, (int, float, np.integer, np.floating))
        
        # 处理 0 维 numpy 数组（标量）
        if isinstance(torch_res, np.ndarray) and torch_res.ndim == 0:
            torch_res = torch_res.item()
            is_torch_scalar = True
        if isinstance(tf_res, np.ndarray) and tf_res.ndim == 0:
            tf_res = tf_res.item()
            is_tf_scalar = True
        
        if is_torch_scalar and is_tf_scalar:
            # 转换为 Python float 进行比较
            torch_val = float(torch_res)
            tf_val = float(tf_res)
            
            # 两个都是 NaN：认为一致
            if np.isnan(torch_val) and np.isnan(tf_val):
                comparison["results_match"] = True
                return comparison
            # 两个都是 Inf：检查符号
            elif np.isinf(torch_val) and np.isinf(tf_val):
                if np.sign(torch_val) == np.sign(tf_val):
                    comparison["results_match"] = True
                else:
                    comparison["comparison_error"] = f"Inf 符号不一致: torch={torch_val}, tf={tf_val}"
                return comparison
            # 一个是 NaN 另一个不是：不一致
            elif np.isnan(torch_val) or np.isnan(tf_val):
                comparison["comparison_error"] = f"NaN 不一致: torch={torch_val}, tf={tf_val}"
                return comparison
            # 都是普通数值：使用 allclose
            elif np.allclose(torch_val, tf_val, rtol=rtol, atol=atol):
                comparison["results_match"] = True
                return comparison
            else:
                diff = abs(torch_val - tf_val)
                comparison["comparison_error"] = f"数值不一致: torch={torch_val}, tf={tf_val}, diff={diff}"
                return comparison
        
        # 处理空容器的情况（tuple vs list）
        if isinstance(torch_res, (tuple, list)) and isinstance(tf_res, (tuple, list)):
            if len(torch_res) == 0 and len(tf_res) == 0:
                comparison["results_match"] = True
                return comparison
            
            # 递归比较容器中的每个元素
            if len(torch_res) != len(tf_res):
                comparison["comparison_error"] = f"容器长度不一致: torch={len(torch_res)}, tf={len(tf_res)}"
                return comparison
            
            all_match = True
            for i, (t_item, tf_item) in enumerate(zip(torch_res, tf_res)):
                # 如果元素是 numpy 数组，直接转换为合适的结果格式
                if isinstance(t_item, np.ndarray) and isinstance(tf_item, np.ndarray):
                    t_result = {"success": True, "result": t_item, "shape": list(t_item.shape), "dtype": str(t_item.dtype), "error": None}
                    tf_result = {"success": True, "result": tf_item, "shape": list(tf_item.shape), "dtype": str(tf_item.dtype), "error": None}
                else:
                    t_result = {"success": True, "result": t_item, "shape": None, "dtype": None, "error": None}
                    tf_result = {"success": True, "result": tf_item, "shape": None, "dtype": None, "error": None}
                
                # 递归调用比较函数
                item_comparison = compare_results(t_result, tf_result, rtol, atol)
                if not item_comparison["results_match"]:
                    all_match = False
                    comparison["comparison_error"] = f"容器第 {i} 个元素不一致: {item_comparison.get('comparison_error', '未知原因')}"
                    break
            
            if all_match:
                comparison["results_match"] = True
            return comparison
        
        if isinstance(torch_res, np.ndarray) and isinstance(tf_res, np.ndarray):
            # 形状检查
            if torch_res.shape != tf_res.shape:
                comparison["comparison_error"] = f"形状不一致: torch={torch_res.shape}, tf={tf_res.shape}"
                return comparison
            
            # 处理特殊值
            torch_nan = np.isnan(torch_res) if np.issubdtype(torch_res.dtype, np.floating) else np.zeros_like(torch_res, dtype=bool)
            tf_nan = np.isnan(tf_res) if np.issubdtype(tf_res.dtype, np.floating) else np.zeros_like(tf_res, dtype=bool)
            
            torch_inf = np.isinf(torch_res) if np.issubdtype(torch_res.dtype, np.floating) else np.zeros_like(torch_res, dtype=bool)
            tf_inf = np.isinf(tf_res) if np.issubdtype(tf_res.dtype, np.floating) else np.zeros_like(tf_res, dtype=bool)
            
            # NaN 位置必须一致
            if not np.array_equal(torch_nan, tf_nan):
                comparison["comparison_error"] = "NaN 位置不一致"
                return comparison
            
            # Inf 位置和符号必须一致
            if not np.array_equal(torch_inf, tf_inf):
                comparison["comparison_error"] = "Inf 位置不一致"
                return comparison
            
            if np.any(torch_inf):
                if not np.array_equal(np.sign(torch_res[torch_inf]), np.sign(tf_res[tf_inf])):
                    comparison["comparison_error"] = "Inf 符号不一致"
                    return comparison
            
            # 对于非特殊值进行数值比较
            mask = ~(torch_nan | torch_inf)
            if np.any(mask):
                if torch_res.size > 0:
                    if np.allclose(torch_res[mask], tf_res[mask], rtol=rtol, atol=atol, equal_nan=True):
                        comparison["results_match"] = True
                    else:
                        max_diff = np.max(np.abs(torch_res[mask] - tf_res[mask]))
                        comparison["comparison_error"] = f"数值不一致，最大差异: {max_diff}"
                else:
                    comparison["results_match"] = True
            else:
                comparison["results_match"] = True
        else:
            # 非张量结果
            if torch_res == tf_res:
                comparison["results_match"] = True
            else:
                comparison["comparison_error"] = f"结果不一致: torch={torch_res}, tf={tf_res}"
                
    except Exception as e:
        comparison["comparison_error"] = f"比较过程出错: {str(e)}"
    
    return comparison



# ==================== 并发处理函数 ====================

def process_single_fuzzing_round(
    client,
    original_case: Dict[str, Any],
    torch_doc: str,
    tf_doc: str,
    round_num: int,
    model: str,
    print_lock: Lock
) -> Dict[str, Any]:
    """
    处理单轮 fuzzing（用于并发执行）
    
    这个函数会被多个线程同时调用，每个线程处理一轮 fuzzing。
    
    参数:
        client: LLM 客户端（线程安全）
        original_case: 原始测试用例
        torch_doc: PyTorch 文档
        tf_doc: TensorFlow 文档
        round_num: 当前轮次（1-3）
        model: LLM 模型名称
        print_lock: 打印锁（线程安全输出）
    
    返回:
        fuzzing 结果字典
    """
    torch_case = original_case.get("torch_test_case", {})
    tf_case = original_case.get("tensorflow_test_case", {})
    torch_api = torch_case.get("api", "")
    tf_api = tf_case.get("api", "")
    
    with print_lock:
        print(f"    [Round {round_num}/{FUZZING_ROUNDS}] 生成变异用例...")
    
    # 构建提示词
    prompt = build_fuzzing_prompt(
        torch_api, tf_api, original_case, torch_doc, tf_doc, round_num
    )
    
    try:
        # 调用 LLM（带重试机制）
        max_retries = 2
        mutated_case = None
        llm_response = ""
        
        for retry in range(max_retries):
            try:
                # Round 1 使用更高的 token 限制，因为极端值变异的响应通常更长
                max_tokens = 6144 if round_num == 1 else 4096
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7 + retry * 0.1,
                    max_tokens=max_tokens,
                )
                llm_response = response.choices[0].message.content.strip()
                
                # 解析响应
                mutated_case = parse_llm_response(llm_response)
                
                if mutated_case is not None:
                    if "torch_test_case" in mutated_case and "tensorflow_test_case" in mutated_case:
                        break
                    else:
                        with print_lock:
                            print(f"[WARN] Round {round_num} 响应缺少必要字段，重试 ({retry + 1}/{max_retries})")
                            print(f"       已解析字段: {list(mutated_case.keys())}")
                        mutated_case = None
                else:
                    if retry < max_retries - 1:
                        with print_lock:
                            print(f"[WARN] Round {round_num} 解析失败，重试 ({retry + 1}/{max_retries})")
                            # 显示响应的前200个字符用于调试
                            preview = llm_response[:200].replace('\n', ' ')
                            print(f"       响应预览: {preview}...")
                            # 检查是否包含 Python float() 语法
                            if "float(" in llm_response:
                                print(f"       检测到 Python float() 语法，将尝试修复")
            except Exception as e:
                with print_lock:
                    print(f"[WARN] Round {round_num} LLM 调用异常: {e}，重试 ({retry + 1}/{max_retries})")
                if retry < max_retries - 1:
                    time.sleep(1)
        
        if mutated_case is None:
            return {
                "round": round_num,
                "success": False,
                "error": "LLM 响应解析失败",
                "llm_response": llm_response[:1000]
            }
        
        with print_lock:
            print(f"    [Round {round_num}] 执行差分测试...")
        
        # 执行测试
        torch_test = mutated_case.get("torch_test_case", {})
        tf_test = mutated_case.get("tensorflow_test_case", {})
        
        torch_result = execute_torch_test(torch_test)
        tf_result = execute_tensorflow_test(tf_test)
        
        # 比较结果
        comparison = compare_results(torch_result, tf_result)
        
        # 记录结果
        fuzzing_result = {
            "round": round_num,
            "success": True,
            "mutation_strategy": mutated_case.get("mutation_strategy", ""),
            "mutation_reason": mutated_case.get("mutation_reason", ""),
            "torch_test_case": torch_test,
            "tensorflow_test_case": tf_test,
            "execution_result": comparison,
            "is_bug_candidate": (
                comparison.get("comparison_error") is not None or
                (comparison["torch_success"] != comparison["tensorflow_success"])
            )
        }
        
        # 打印结果摘要
        with print_lock:
            if fuzzing_result["is_bug_candidate"]:
                print(f"    [Round {round_num}] ⚠️ 发现潜在问题: {comparison.get('comparison_error') or '执行状态不一致'}")
            else:
                print(f"    [Round {round_num}] ✓ 结果一致")
        
        return fuzzing_result
        
    except Exception as e:
        with print_lock:
            print(f"    [Round {round_num}] ✗ 错误: {e}")
        return {
            "round": round_num,
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc()
        }


def run_fuzzing_for_case(
    client,
    original_case: Dict[str, Any],
    torch_doc: str,
    tf_doc: str,
    model: str,
    print_lock: Lock,
    workers: int = DEFAULT_WORKERS
) -> List[Dict[str, Any]]:
    """
    对单个测试用例进行多轮 fuzzing（并发版本）
    
    使用线程池并发执行 3 轮 fuzzing，提高效率。
    """
    fuzzing_results = []
    
    # 使用线程池并发执行 3 轮 fuzzing
    with ThreadPoolExecutor(max_workers=min(workers, FUZZING_ROUNDS)) as executor:
        # 提交所有 fuzzing 轮次任务
        future_to_round = {}
        for round_num in range(1, FUZZING_ROUNDS + 1):
            future = executor.submit(
                process_single_fuzzing_round,
                client,
                original_case,
                torch_doc,
                tf_doc,
                round_num,
                model,
                print_lock
            )
            future_to_round[future] = round_num
        
        # 收集结果（按完成顺序）
        for future in as_completed(future_to_round):
            try:
                result = future.result()
                fuzzing_results.append(result)
            except Exception as e:
                round_num = future_to_round[future]
                with print_lock:
                    print(f"[ERROR] Round {round_num} 执行异常: {e}")
                fuzzing_results.append({
                    "round": round_num,
                    "success": False,
                    "error": f"执行异常: {str(e)}"
                })
    
    # 按轮次排序结果
    fuzzing_results.sort(key=lambda x: x.get("round", 0))
    
    return fuzzing_results


def process_single_case(
    client,
    case: Dict[str, Any],
    case_idx: int,
    total_cases: int,
    torch_doc: str,
    tf_doc: str,
    model: str,
    print_lock: Lock,
    workers: int
) -> Dict[str, Any]:
    """
    处理单个测试用例（用于并发执行）
    
    参数:
        client: LLM 客户端
        case: 测试用例
        case_idx: 用例索引（从1开始）
        total_cases: 总用例数
        torch_doc: PyTorch 文档
        tf_doc: TensorFlow 文档
        model: LLM 模型名称
        print_lock: 打印锁
        workers: 并发线程数
    
    返回:
        用例处理结果
    """
    with print_lock:
        print(f"\n  用例 {case_idx}/{total_cases} (iteration={case.get('iteration')}, case={case.get('case_number')})")
    
    # 对该用例进行 fuzzing（内部也是并发的）
    fuzzing_results = run_fuzzing_for_case(
        client, case, torch_doc, tf_doc, model, print_lock, workers
    )
    
    case_result = {
        "original_case_info": {
            "source_file": case.get("source_file"),
            "iteration": case.get("iteration"),
            "case_number": case.get("case_number"),
            "is_llm_generated": case.get("is_llm_generated")
        },
        "original_torch_test_case": case.get("torch_test_case"),
        "original_tensorflow_test_case": case.get("tensorflow_test_case"),
        "fuzzing_results": fuzzing_results
    }
    
    return case_result


def process_operator(
    operator_file: Path,
    client,
    model: str,
    max_cases: Optional[int],
    print_lock: Lock,
    workers: int = DEFAULT_WORKERS
) -> Dict[str, Any]:
    """
    处理单个算子的所有成功用例（并发版本）
    
    使用线程池并发处理多个测试用例，每个用例内部的 3 轮 fuzzing 也是并发的。
    """
    with open(operator_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    operator_name = data.get("operator", "unknown")
    success_cases = data.get("success_cases", [])
    
    if max_cases:
        success_cases = success_cases[:max_cases]
    
    with print_lock:
        print(f"\n处理算子: {operator_name} ({len(success_cases)} 个用例)")
    
    # 获取 API 名称
    if success_cases:
        torch_api = success_cases[0].get("torch_test_case", {}).get("api", "")
        tf_api = success_cases[0].get("tensorflow_test_case", {}).get("api", "")
    else:
        return {"operator": operator_name, "results": [], "error": "无成功用例"}
    
    # 爬取文档
    with print_lock:
        print(f"  获取 {torch_api} 文档...")
    torch_doc = get_doc_content(torch_api, "pytorch")
    with print_lock:
        print(f"  获取 {tf_api} 文档...")
    tf_doc = get_doc_content(tf_api, "tensorflow")
    
    # 并发处理所有用例
    all_results = []
    bug_candidates = 0
    
    # 使用线程池并发处理用例
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {}
            
            # 提交所有用例处理任务
            for idx, case in enumerate(success_cases, 1):
                future = executor.submit(
                    process_single_case,
                    client,
                    case,
                    idx,
                    len(success_cases),
                    torch_doc,
                    tf_doc,
                    model,
                    print_lock,
                    workers
                )
                future_to_idx[future] = idx
            
            # 收集结果
            results_dict = {}
            for future in as_completed(future_to_idx):
                try:
                    idx = future_to_idx[future]
                    case_result = future.result(timeout=300)  # 5分钟超时
                    results_dict[idx] = case_result
                    
                    # 统计潜在 bug
                    for fr in case_result.get("fuzzing_results", []):
                        if fr.get("is_bug_candidate"):
                            bug_candidates += 1
                except TimeoutError:
                    idx = future_to_idx[future]
                    with print_lock:
                        print(f"[ERROR] 則试用例 {idx} 超时")
                    results_dict[idx] = {
                        "error": "处理超时",
                        "fuzzing_results": []
                    }
                except Exception as e:
                    idx = future_to_idx[future]
                    with print_lock:
                        print(f"[ERROR] 处理用例 {idx} 时出错: {e}")
                        traceback.print_exc()
                    results_dict[idx] = {
                        "error": f"处理失败: {str(e)}",
                        "fuzzing_results": []
                    }
    except Exception as e:
        with print_lock:
            print(f"[ERROR] 线程池执行异常: {e}")
            traceback.print_exc()
        results_dict = {}
    
    # 按索引顺序排列结果
    for idx in sorted(results_dict.keys()):
        all_results.append(results_dict[idx])
    
    return {
        "operator": operator_name,
        "torch_api": torch_api,
        "tensorflow_api": tf_api,
        "total_cases": len(success_cases),
        "total_fuzzing_rounds": len(success_cases) * FUZZING_ROUNDS,
        "bug_candidates": bug_candidates,
        "timestamp": datetime.now().isoformat(),
        "results": all_results
    }


def save_operator_result(result: Dict[str, Any], output_dir: Path) -> None:
    """
    保存算子的 fuzzing 结果，文件名包含时间戳
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    operator_name = result.get("operator", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{operator_name}_fuzzing_result_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"  结果已保存: {output_file}")


def main():
    """
    主程序入口（并发版本）
    """
    import signal
    import sys
    
    # 设置信号处理，防止静默退出
    def signal_handler(signum, frame):
        print(f"\n[WARN] 收到信号 {signum}，程序即将退出...")
        sys.exit(1)
    
    # 注册信号处理器
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except (AttributeError, ValueError):
        pass  # Windows 可能不支持某些信号
    
    parser = argparse.ArgumentParser(
        description="PyTorch-TensorFlow 基于 LLM 的 Fuzzing 差分测试（并发版本）"
    )
    parser.add_argument(
        "--operators", "-o",
        nargs="*",
        help="指定要测试的算子名称（不指定则测试所有）"
    )
    parser.add_argument(
        "--max-cases", "-m",
        type=int,
        default=None,
        help="每个算子最多测试的用例数（默认全部）"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM 模型名称（默认 {DEFAULT_MODEL}）"
    )
    parser.add_argument(
        "--key-path", "-k",
        default=DEFAULT_KEY_PATH,
        help=f"API key 文件路径（默认 {DEFAULT_KEY_PATH}）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多处理的算子数量（用于测试）"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="起始算子索引（从1开始，默认为1）"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="结束算子索引（包含，默认到最后一个）"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并发线程数（默认 {DEFAULT_WORKERS}）"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PyTorch-TensorFlow 基于 LLM 的 Fuzzing 差分测试（并发版本）")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"每用例 Fuzzing 轮数: {FUZZING_ROUNDS}")
    print(f"并发线程数: {args.workers}")
    
    # 初始化 LLM 客户端
    try:
        client = get_qwen_client(args.key_path)
        print("LLM 客户端初始化成功")
    except Exception as e:
        print(f"[ERROR] 无法初始化 LLM 客户端: {e}")
        return
    
    # 获取要处理的算子文件
    if args.operators:
        operator_files = []
        for op in args.operators:
            op_file = SUCCESS_CASES_DIR / f"{op}_success_cases.json"
            if op_file.exists():
                operator_files.append(op_file)
            else:
                print(f"[WARN] 找不到算子文件: {op_file}")
    else:
        operator_files = sorted(SUCCESS_CASES_DIR.glob("*_success_cases.json"))
    
    if args.limit:
        operator_files = operator_files[:args.limit]
    
    # 应用范围筛选（--start 和 --end 参数）
    total_available = len(operator_files)
    if args.start > total_available:
        print(f"[ERROR] 起始索引 {args.start} 超出范围（共有 {total_available} 个算子）")
        return
    
    start_idx = args.start - 1  # 转换为0索引
    end_idx = args.end if args.end is not None else total_available  # 默认到结束
    
    if end_idx > total_available:
        print(f"[WARN] 结束索引 {end_idx} 超出范围，调整为 {total_available}")
        end_idx = total_available
    
    if start_idx >= end_idx:
        print(f"[ERROR] 起始索引 {args.start} 必须小于结束索引 {end_idx}")
        return
    
    operator_files = operator_files[start_idx:end_idx]
    
    if args.start > 1 or args.end is not None:
        range_info = f"第 {args.start}-{end_idx} 个算子（共 {total_available} 个可用）"
        print(f"测试范围: {range_info}")
    
    print(f"待处理算子数: {len(operator_files)}")
    print("=" * 70)
    
    # 统计
    total_operators = len(operator_files)
    total_bug_candidates = 0
    
    # 创建打印锁
    print_lock = Lock()
    
    # 记录开始时间
    start_time = time.time()
    
    # 处理每个算子（顺序处理算子，但每个算子内部的用例和 fuzzing 轮次是并发的）
    for idx, op_file in enumerate(operator_files, 1):
        print(f"\n[{idx}/{total_operators}] 处理: {op_file.stem}")
        
        try:
            result = process_operator(
                op_file, client, args.model, args.max_cases, print_lock, args.workers
            )
            
            # 保存结果
            save_operator_result(result, RESULT_DIR)
            
            total_bug_candidates += result.get("bug_candidates", 0)
            
            # 显示进度
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = total_operators - idx
            eta = avg_time * remaining
            print(f"\n[PROGRESS] 已完成 {idx}/{total_operators} 个算子，"
                  f"耗时 {elapsed:.1f}s，预计剩余 {eta:.1f}s")
            
        except KeyboardInterrupt:
            print(f"\n[INFO] 用户中断，正在保存已完成的结果...")
            break
        except MemoryError:
            print(f"[ERROR] 内存不足，跳过 {op_file.stem}")
            # 尝试释放内存
            import gc
            gc.collect()
            continue
        except Exception as e:
            print(f"[ERROR] 处理 {op_file.stem} 时出错: {e}")
            traceback.print_exc()
            # 继续处理下一个算子，而不是退出
            continue
    
    # 打印总结
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Fuzzing 测试完成！")
    print("=" * 70)
    print(f"处理算子数: {total_operators}")
    print(f"发现潜在问题数: {total_bug_candidates}")
    print(f"总耗时: {total_time:.1f}s")
    print(f"结果保存目录: {RESULT_DIR}")


if __name__ == "__main__":
    main()
