"""  
PyTorch-PaddlePaddle 基于 LLM 的 Fuzzing 差分测试工具（并发版本）

功能说明:
    1. 读取按算子分类的成功测试用例
    2. 爬取 PyTorch 和 PaddlePaddle 官方文档
    3. 使用 LLM 变异测试用例（复杂输入、极端值、边界值）
    4. 执行差分测试，检测框架不一致或潜在 bug
    5. 每个用例进行 3 轮 fuzzing
    6. 支持多线程并发处理（默认8个线程）
"""

import json
import sys
import re
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


def build_fuzzing_prompt(
    torch_api: str,
    paddle_api: str,
    original_case: Dict[str, Any],
    torch_doc: str,
    paddle_doc: str,
    round_num: int
) -> str:
    """
    构建 LLM fuzzing 提示词
    
    参数:
        torch_api: PyTorch API 名称
        paddle_api: PaddlePaddle API 名称
        original_case: 原始测试用例
        torch_doc: PyTorch 文档内容
        paddle_doc: PaddlePaddle 文档内容
        round_num: 当前 fuzzing 轮次
    """
    torch_case = original_case.get("torch_test_case", {})
    paddle_case = original_case.get("paddle_test_case", {})
    
    torch_case_json = json.dumps(torch_case, ensure_ascii=False, indent=2)
    paddle_case_json = json.dumps(paddle_case, ensure_ascii=False, indent=2)
    
    # 检测是否是逻辑运算算子
    is_logical_op = any(keyword in torch_api.lower() for keyword in ["logical_", "bitwise_"])
    
    # 检测是否是整数运算算子
    is_integer_op = any(keyword in torch_api.lower() for keyword in ["gcd", "lcm", "bitwise_", "remainder", "div", "mod"])
    
    # 不同轮次的变异策略
    mutation_strategies = {
        1: "极端数值变异：使用复杂的浮点数、极大值(1e38)、极小值(1e-38)、无穷大(inf)、负无穷(-inf)、NaN、零(0)、负零(-0.0)等特殊数值",
        2: "边界形状变异：使用空张量(shape含0)、标量(shape=[])、超高维张量(5维以上)、不规则形状、单元素张量等边界情况",
        3: "复杂类型变异：测试不同数据类型(float32/float64/int32/int64/bool)，注意：PaddlePaddle CPU 不支持 float16，请避免使用"
    }
    
    current_strategy = mutation_strategies.get(round_num, mutation_strategies[1])
    
    # 根据算子类型设置示例值
    if is_integer_op:
        sample_values_example = "[1, 2, -1, 100, -100, 1000]"
    elif is_logical_op:
        sample_values_example = "[true, false, true, false]"
    else:
        sample_values_example = '[1.0, -1.0, "inf", "-inf", "nan", -0.0, 1e-38, 1e38]'
    
    # 为逻辑运算和整数运算添加特殊提示
    dtype_constraint = ""
    if is_logical_op:
        dtype_constraint = """
【重要：数据类型约束】
**此算子是逻辑运算，必须使用 bool 类型！**
- PyTorch 和 PaddlePaddle 的逻辑运算（如 logical_and, logical_or, logical_xor）都要求输入为 bool 类型
- 变异时必须保持 dtype 为 "bool"
- 不要使用 float/int 等其他类型，否则会导致执行失败
"""
    elif is_integer_op:
        dtype_constraint = """
【重要：数据类型约束】
**此算子是整数运算，必须使用整数类型！**
- PyTorch 和 PaddlePaddle 的整数运算（如 gcd, lcm, bitwise 等）要求输入为整数类型
- 推荐使用 dtype："int32", "int64"
- **禁止使用 inf, -inf, nan 等特殊浮点值**，这些值对整数运算无意义且可能导致错误
- 使用有限的整数值，如：1, 2, -1, 100, -100, 1000 等
"""
    
    prompt = f"""你是一个专业的深度学习框架测试专家，现在需要对 PyTorch 和 PaddlePaddle 的等价算子进行差分测试。

【测试目标】
我们要寻找的是**框架实现中的潜在 bug**，即：在数学上完全等价的操作，两框架却产生不同的输出。
- ✓ 要找的：相同数学计算下的框架实现差异（真正的 bug）
- ✗ 不要找的：因为测试用例本身不等价导致的差异（假阳性）
- ✗ 不要找的：因为框架可支持数据类型的不同导致的差异（如 PaddlePaddle CPU 不支持 float16）
    
【当前测试的算子对】
- PyTorch API: {torch_api}
- PaddlePaddle API: {paddle_api}

【原始成功测试用例】
PyTorch 测试用例:
```json
{torch_case_json}
```

PaddlePaddle 测试用例:
```json
{paddle_case_json}
```

【PyTorch 官方文档】
{torch_doc if torch_doc else "文档获取失败"}

【PaddlePaddle 官方文档】
{paddle_doc if paddle_doc else "文档获取失败"}

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

【已知框架限制 - 必须避免！】
- **PaddlePaddle CPU 不支持 float16**：请勿使用 float16 dtype，会导致 "kernel not registered" 错误
- **PaddlePaddle 的 y 参数必须是 Tensor**：paddle.add(x, y) 的 y 不能是 Python 标量，必须是 Tensor
- 推荐使用的 dtype：float32、float64、int32、int64、bool
- PyTorch 和 PaddlePaddle 的数据格式通常相同（NCHW）

【输出格式要求】
请严格按照以下 JSON 格式输出变异后的测试用例，不要输出任何其他内容：

**重要：特殊浮点值的 JSON 表示**
- 正无穷：使用字符串 "inf" 或 "Infinity"
- 负无穷：使用字符串 "-inf" 或 "-Infinity"  
- NaN：使用字符串 "nan" 或 "NaN"
- 负零：使用数值 -0.0
- 不要使用 Python 语法如 float('inf')，这在 JSON 中是非法的！

**注意：对于整数算子（如 gcd, lcm 等），请勿使用 inf, -inf, nan 等特殊值！**

```json
{{
  "mutation_strategy": "简要描述本次变异策略（不超过50字）",
  "mutation_reason": "简要解释为什么这样变异可能发现问题（不超过100字）",
  "torch_test_case": {{
    "api": "{torch_api}",
    "input": {{
      "shape": [...],
      "dtype": "...",
      "sample_values": {sample_values_example}
    }},
    "other": {{
      "shape": [...],
      "dtype": "...",
      "sample_values": [...]
    }},
    "dim": ...,
    "其他必需参数": ...
  }},
  "paddle_test_case": {{
    "api": "{paddle_api}",
    "input": {{
      "shape": [...],
      "dtype": "...",
      "sample_values": {sample_values_example}
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
5. **PyTorch 和 PaddlePaddle 测试用例的所有输入张量必须使用相同的 dtype**
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
    """
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
    """
    if not isinstance(parsed, dict):
        return False
    
    required_fields = ["torch_test_case", "paddle_test_case"]
    for field in required_fields:
        if field not in parsed:
            return False
        if not isinstance(parsed[field], dict):
            return False
    
    return True


def try_repair_json(json_str: str) -> Optional[Dict[str, Any]]:
    """
    尝试修复不完整的 JSON 字符串
    """
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    
    repaired = json_str
    repaired = fix_python_float_syntax(repaired)
    
    patterns_to_try = [
        (r',?\s*"[^"]*"\s*:\s*"[^"]*$', ''),
        (r',?\s*"[^"]*"\s*:\s*$', ''),
        (r',\s*$', ''),
        (r',?\s*\d+\.?\d*e?[+-]?\d*\s*$', ''),
        (r',?\s*("inf"|"-inf"|"nan"|float\([^)]*\))?\s*$', ''),
        (r'"sample_values"\s*:\s*\[[^\]]*$', '"sample_values": []'),
        (r'"input"\s*:\s*\{[^}]*$', '"input": {"shape": [], "dtype": "float32", "sample_values": []}'),
    ]
    
    for pattern, replacement in patterns_to_try:
        test_str = re.sub(pattern, replacement, repaired)
        test_open_braces = test_str.count('{')
        test_close_braces = test_str.count('}')
        test_open_brackets = test_str.count('[')
        test_close_brackets = test_str.count(']')
        
        test_str += ']' * (test_open_brackets - test_close_brackets)
        test_str += '}' * (test_open_braces - test_close_braces)
        
        try:
            result = json.loads(test_str)
            if validate_parsed_json(result):
                return result
        except json.JSONDecodeError:
            continue
    
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
        if val_lower == "inf" or val_lower == "+inf" or val_lower == "infinity":
            return np.inf
        elif val_lower == "-inf" or val_lower == "-infinity":
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
    """
    shape = spec.get("shape", [])
    dtype_str = spec.get("dtype", "float32")
    sample_values = spec.get("sample_values", [])
    
    processed_values = [parse_special_value(v) for v in sample_values]
    
    size = 1
    for dim in shape:
        size *= dim
    
    if size == 0:
        data = np.array([]).reshape(shape)
    elif processed_values:
        if len(processed_values) >= size:
            data = np.array(processed_values[:size]).reshape(shape)
        else:
            np.random.seed(42)
            if processed_values:
                finite_values = [v for v in processed_values if np.isfinite(v)]
                if finite_values:
                    mean = np.mean(finite_values)
                    std = np.std(finite_values) if len(finite_values) > 1 else 1.0
                else:
                    mean, std = 0.0, 1.0
                
                base_data = np.random.normal(mean, std, size)
                for i, v in enumerate(processed_values):
                    if i < size:
                        base_data.flat[i] = v
                data = base_data.reshape(shape)
            else:
                data = np.random.randn(*shape)
    else:
        np.random.seed(42)
        data = np.random.randn(*shape)
    
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
    
    if np_dtype in [np.float16, np.float32, np.float64]:
        data = data.astype(np_dtype)
    elif np_dtype == np.bool_:
        data = (data > 0).astype(np.bool_)
    elif np_dtype in [np.complex64, np.complex128]:
        data = (data + 1j * np.random.randn(*shape)).astype(np_dtype)
    else:
        # 对于整数类型，需要处理 inf/nan 值
        if np_dtype in [np.int32, np.int64, np.int8, np.int16]:
            # 将 inf/nan 替换为有效的整数值
            data = np.where(np.isfinite(data), data, 0).astype(np_dtype)
        else:
            data = data.astype(np_dtype)
    
    return data, dtype_str


def create_torch_tensor(spec: Any, torch_dtype_map: Dict) -> Any:
    """
    根据规格创建 PyTorch 张量
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
    执行 PyTorch 测试用例
    """
    try:
        import torch
        
        api_name = test_case.get("api", "")
        
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
        
        non_tensor_params = {
            "kernel_size", "stride", "padding", "dilation", "groups",
            "ceil_mode", "count_include_pad", "divisor_override",
            "alpha", "dim", "keepdim", "dtype", "out", "axis",
            "p", "eps", "momentum", "affine", "track_running_stats",
            "num_features", "normalized_shape", "weight", "bias",
            "return_indices", "exclusive", "reverse"
        }
        
        args = []
        kwargs = {}
        
        if "input" in test_case:
            input_tensor = create_torch_tensor(test_case["input"], torch_dtype_map)
            args.append(input_tensor)
        
        tensor_input_names = ["other", "x", "y", "tensor", "input1", "input2", "mat1", "mat2"]
        for name in tensor_input_names:
            if name in test_case:
                tensor = create_torch_tensor(test_case[name], torch_dtype_map)
                if name == "other":
                    kwargs["other"] = tensor
                else:
                    args.append(tensor)
        
        for key, value in test_case.items():
            if key not in ["api", "input", "x", "y", "other", "tensor", "input1", "input2", "mat1", "mat2"]:
                if key in non_tensor_params or not isinstance(value, dict):
                    kwargs[key] = value
        
        api_parts = api_name.split(".")
        func = torch
        for part in api_parts[1:]:
            func = getattr(func, part)
        
        if isinstance(func, type):
            init_params = {k: v for k, v in kwargs.items() if k in non_tensor_params}
            instance = func(**init_params)
            result = instance(args[0] if args else kwargs.get("input"))
        else:
            result = func(*args, **kwargs)
        
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


def create_paddle_tensor(spec: Any, paddle_dtype_map: Dict) -> Any:
    """
    根据规格创建 PaddlePaddle 张量
    """
    import paddle
    
    if isinstance(spec, (int, float, bool)):
        return paddle.to_tensor(spec)
    if isinstance(spec, dict) and "shape" in spec:
        data, dtype_str = create_tensor_from_spec(spec, "paddle")
        paddle_dtype = paddle_dtype_map.get(dtype_str, "float32")
        return paddle.to_tensor(data, dtype=paddle_dtype)
    return spec


def execute_paddle_test(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行 PaddlePaddle 测试用例
    """
    try:
        import paddle
        
        # 确保 PaddlePaddle 处于动态图模式
        paddle.disable_static()
        
        api_name = test_case.get("api", "")
        
        paddle_dtype_map = {
            "float16": "float16",
            "float32": "float32",
            "float64": "float64",
            "int32": "int32",
            "int64": "int64",
            "bool": "bool",
            "complex64": "complex64",
            "complex128": "complex128",
        }
        
        non_tensor_params = {
            "kernel_size", "stride", "padding", "dilation", "groups",
            "ceil_mode", "count_include_pad", "divisor_override",
            "alpha", "dim", "keepdim", "dtype", "out", "axis",
            "p", "eps", "momentum", "affine", "track_running_stats",
            "num_features", "normalized_shape", "weight", "bias",
            "return_indices", "exclusive", "reverse", "data_format"
        }
        
        args = []
        kwargs = {}
        
        if "input" in test_case:
            input_tensor = create_paddle_tensor(test_case["input"], paddle_dtype_map)
            args.append(input_tensor)
        
        tensor_input_names = ["other", "x", "y", "tensor", "input1", "input2"]
        for name in tensor_input_names:
            if name in test_case:
                tensor = create_paddle_tensor(test_case[name], paddle_dtype_map)
                args.append(tensor)
        
        for key, value in test_case.items():
            if key not in ["api", "input", "x", "y", "other", "tensor", "input1", "input2"]:
                if key in non_tensor_params or not isinstance(value, dict):
                    kwargs[key] = value
        
        api_parts = api_name.split(".")
        func = paddle
        for part in api_parts[1:]:
            func = getattr(func, part)
        
        is_nn_layer = "nn" in api_name and api_name.count(".") >= 2
        
        if is_nn_layer and isinstance(func, type):
            init_params = {k: v for k, v in kwargs.items() if k in non_tensor_params}
            layer = func(**init_params)
            result = layer(args[0] if args else kwargs.get("input"))
        else:
            if args and not kwargs:
                result = func(*args)
            elif args:
                result = func(*args, **kwargs)
            else:
                result = func(**kwargs)
        
        if isinstance(result, paddle.Tensor):
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
    paddle_result: Dict[str, Any],
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> Dict[str, Any]:
    """
    比较两个框架的执行结果
    """
    comparison = {
        "torch_success": torch_result["success"],
        "paddle_success": paddle_result["success"],
        "torch_error": torch_result["error"],
        "paddle_error": paddle_result["error"],
        "results_match": False,
        "comparison_error": None,
        "torch_shape": torch_result["shape"],
        "paddle_shape": paddle_result["shape"],
        "torch_dtype": torch_result["dtype"],
        "paddle_dtype": paddle_result["dtype"],
    }
    
    if not torch_result["success"] or not paddle_result["success"]:
        if torch_result["success"] != paddle_result["success"]:
            comparison["comparison_error"] = "执行状态不一致：一方成功一方失败"
        return comparison
    
    try:
        torch_res = torch_result["result"]
        paddle_res = paddle_result["result"]
        
        if torch_res is None and paddle_res is None:
            comparison["results_match"] = True
            return comparison
        
        # 处理标量的情况 - 扩展类型判断
        is_torch_scalar = isinstance(torch_res, (int, float, np.integer, np.floating))
        is_paddle_scalar = isinstance(paddle_res, (int, float, np.integer, np.floating))
        
        # 处理 0 维 numpy 数组
        if isinstance(torch_res, np.ndarray) and torch_res.ndim == 0:
            torch_res = torch_res.item()
            is_torch_scalar = True
        if isinstance(paddle_res, np.ndarray) and paddle_res.ndim == 0:
            paddle_res = paddle_res.item()
            is_paddle_scalar = True
        
        if is_torch_scalar and is_paddle_scalar:
            torch_val = float(torch_res)
            paddle_val = float(paddle_res)
            
            if np.isnan(torch_val) and np.isnan(paddle_val):
                comparison["results_match"] = True
                return comparison
            elif np.isinf(torch_val) and np.isinf(paddle_val):
                if np.sign(torch_val) == np.sign(paddle_val):
                    comparison["results_match"] = True
                else:
                    comparison["comparison_error"] = f"Inf 符号不一致: torch={torch_val}, paddle={paddle_val}"
                return comparison
            elif np.isnan(torch_val) or np.isnan(paddle_val):
                comparison["comparison_error"] = f"NaN 不一致: torch={torch_val}, paddle={paddle_val}"
                return comparison
            elif np.allclose(torch_val, paddle_val, rtol=rtol, atol=atol):
                comparison["results_match"] = True
                return comparison
            else:
                diff = abs(torch_val - paddle_val)
                comparison["comparison_error"] = f"数值不一致: torch={torch_val}, paddle={paddle_val}, diff={diff}"
                return comparison
        
        # 处理容器情况
        if isinstance(torch_res, (tuple, list)) and isinstance(paddle_res, (tuple, list)):
            if len(torch_res) == 0 and len(paddle_res) == 0:
                comparison["results_match"] = True
                return comparison
            
            if len(torch_res) != len(paddle_res):
                comparison["comparison_error"] = f"容器长度不一致: torch={len(torch_res)}, paddle={len(paddle_res)}"
                return comparison
            
            all_match = True
            for i, (t_item, p_item) in enumerate(zip(torch_res, paddle_res)):
                if isinstance(t_item, np.ndarray) and isinstance(p_item, np.ndarray):
                    t_result = {"success": True, "result": t_item, "shape": list(t_item.shape), "dtype": str(t_item.dtype), "error": None}
                    p_result = {"success": True, "result": p_item, "shape": list(p_item.shape), "dtype": str(p_item.dtype), "error": None}
                else:
                    t_result = {"success": True, "result": t_item, "shape": None, "dtype": None, "error": None}
                    p_result = {"success": True, "result": p_item, "shape": None, "dtype": None, "error": None}
                
                item_comparison = compare_results(t_result, p_result, rtol, atol)
                if not item_comparison["results_match"]:
                    all_match = False
                    comparison["comparison_error"] = f"容器第 {i} 个元素不一致: {item_comparison.get('comparison_error', '未知原因')}"
                    break
            
            if all_match:
                comparison["results_match"] = True
            return comparison
        
        if isinstance(torch_res, np.ndarray) and isinstance(paddle_res, np.ndarray):
            if torch_res.shape != paddle_res.shape:
                comparison["comparison_error"] = f"形状不一致: torch={torch_res.shape}, paddle={paddle_res.shape}"
                return comparison
            
            torch_nan = np.isnan(torch_res) if np.issubdtype(torch_res.dtype, np.floating) else np.zeros_like(torch_res, dtype=bool)
            paddle_nan = np.isnan(paddle_res) if np.issubdtype(paddle_res.dtype, np.floating) else np.zeros_like(paddle_res, dtype=bool)
            
            torch_inf = np.isinf(torch_res) if np.issubdtype(torch_res.dtype, np.floating) else np.zeros_like(torch_res, dtype=bool)
            paddle_inf = np.isinf(paddle_res) if np.issubdtype(paddle_res.dtype, np.floating) else np.zeros_like(paddle_res, dtype=bool)
            
            if not np.array_equal(torch_nan, paddle_nan):
                comparison["comparison_error"] = "NaN 位置不一致"
                return comparison
            
            if not np.array_equal(torch_inf, paddle_inf):
                comparison["comparison_error"] = "Inf 位置不一致"
                return comparison
            
            if np.any(torch_inf):
                if not np.array_equal(np.sign(torch_res[torch_inf]), np.sign(paddle_res[paddle_inf])):
                    comparison["comparison_error"] = "Inf 符号不一致"
                    return comparison
            
            mask = ~(torch_nan | torch_inf)
            if np.any(mask):
                if torch_res.size > 0:
                    if np.allclose(torch_res[mask], paddle_res[mask], rtol=rtol, atol=atol, equal_nan=True):
                        comparison["results_match"] = True
                    else:
                        max_diff = np.max(np.abs(torch_res[mask] - paddle_res[mask]))
                        comparison["comparison_error"] = f"数值不一致，最大差异: {max_diff}"
                else:
                    comparison["results_match"] = True
            else:
                comparison["results_match"] = True
        else:
            if torch_res == paddle_res:
                comparison["results_match"] = True
            else:
                comparison["comparison_error"] = f"结果不一致: torch={torch_res}, paddle={paddle_res}"
                
    except Exception as e:
        comparison["comparison_error"] = f"比较过程出错: {str(e)}"
    
    return comparison


# ==================== 并发处理函数 ====================

def process_single_fuzzing_round(
    client,
    original_case: Dict[str, Any],
    torch_doc: str,
    paddle_doc: str,
    round_num: int,
    model: str,
    print_lock: Lock
) -> Dict[str, Any]:
    """
    处理单轮 fuzzing（用于并发执行）
    """
    torch_case = original_case.get("torch_test_case", {})
    paddle_case = original_case.get("paddle_test_case", {})
    torch_api = torch_case.get("api", "")
    paddle_api = paddle_case.get("api", "")
    
    with print_lock:
        print(f"    [Round {round_num}/{FUZZING_ROUNDS}] 生成变异用例...")
    
    prompt = build_fuzzing_prompt(
        torch_api, paddle_api, original_case, torch_doc, paddle_doc, round_num
    )
    
    try:
        max_retries = 2
        mutated_case = None
        llm_response = ""
        
        for retry in range(max_retries):
            try:
                max_tokens = 6144 if round_num == 1 else 4096
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7 + retry * 0.1,
                    max_tokens=max_tokens,
                )
                llm_response = response.choices[0].message.content.strip()
                
                mutated_case = parse_llm_response(llm_response)
                
                if mutated_case is not None:
                    if "torch_test_case" in mutated_case and "paddle_test_case" in mutated_case:
                        break
                    else:
                        with print_lock:
                            print(f"[WARN] Round {round_num} 响应缺少必要字段，重试 ({retry + 1}/{max_retries})")
                        mutated_case = None
                else:
                    if retry < max_retries - 1:
                        with print_lock:
                            print(f"[WARN] Round {round_num} 解析失败，重试 ({retry + 1}/{max_retries})")
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
        
        torch_test = mutated_case.get("torch_test_case", {})
        paddle_test = mutated_case.get("paddle_test_case", {})
        
        # 添加超时保护
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("测试执行超时")
        
        torch_result = None
        paddle_result = None
        
        try:
            # 为每个测试设置30秒超时
            if hasattr(signal, 'SIGALRM'):  # Unix系统
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
            
            torch_result = execute_torch_test(torch_test)
            paddle_result = execute_paddle_test(paddle_test)
            
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # 取消超时
                
        except TimeoutError:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            return {
                "round": round_num,
                "success": False,
                "error": "测试执行超时（30秒）",
                "timeout": True
            }
        except Exception as exec_e:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            return {
                "round": round_num,
                "success": False,
                "error": f"测试执行异常: {str(exec_e)}",
                "traceback": traceback.format_exc()
            }
        
        comparison = compare_results(torch_result, paddle_result)
        
        fuzzing_result = {
            "round": round_num,
            "success": True,
            "mutation_strategy": mutated_case.get("mutation_strategy", ""),
            "mutation_reason": mutated_case.get("mutation_reason", ""),
            "torch_test_case": torch_test,
            "paddle_test_case": paddle_test,
            "execution_result": comparison,
            "is_bug_candidate": (
                comparison.get("comparison_error") is not None or
                (comparison["torch_success"] != comparison["paddle_success"])
            )
        }
        
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
    paddle_doc: str,
    model: str,
    print_lock: Lock,
    workers: int = DEFAULT_WORKERS
) -> List[Dict[str, Any]]:
    """
    对单个测试用例进行多轮 fuzzing（并发版本）
    """
    fuzzing_results = []
    
    with ThreadPoolExecutor(max_workers=min(workers, FUZZING_ROUNDS)) as executor:
        future_to_round = {}
        for round_num in range(1, FUZZING_ROUNDS + 1):
            future = executor.submit(
                process_single_fuzzing_round,
                client,
                original_case,
                torch_doc,
                paddle_doc,
                round_num,
                model,
                print_lock
            )
            future_to_round[future] = round_num
        
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
    
    fuzzing_results.sort(key=lambda x: x.get("round", 0))
    return fuzzing_results


def process_single_case(
    client,
    case: Dict[str, Any],
    case_idx: int,
    total_cases: int,
    torch_doc: str,
    paddle_doc: str,
    model: str,
    print_lock: Lock,
    workers: int
) -> Dict[str, Any]:
    """
    处理单个测试用例（用于并发执行）
    """
    with print_lock:
        print(f"\n  用例 {case_idx}/{total_cases} (iteration={case.get('iteration')}, case={case.get('case_number')})")
    
    fuzzing_results = run_fuzzing_for_case(
        client, case, torch_doc, paddle_doc, model, print_lock, workers
    )
    
    case_result = {
        "original_case_info": {
            "source_file": case.get("source_file"),
            "iteration": case.get("iteration"),
            "case_number": case.get("case_number"),
            "is_llm_generated": case.get("is_llm_generated")
        },
        "original_torch_test_case": case.get("torch_test_case"),
        "original_paddle_test_case": case.get("paddle_test_case"),
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
    """
    with open(operator_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    operator_name = data.get("operator", "unknown")
    success_cases = data.get("success_cases", [])
    
    if max_cases:
        success_cases = success_cases[:max_cases]
    
    with print_lock:
        print(f"\n处理算子: {operator_name} ({len(success_cases)} 个用例)")
    
    if success_cases:
        torch_api = success_cases[0].get("torch_test_case", {}).get("api", "")
        paddle_api = success_cases[0].get("paddle_test_case", {}).get("api", "")
    else:
        return {"operator": operator_name, "results": [], "error": "无成功用例"}
    
    with print_lock:
        print(f"  获取 {torch_api} 文档...")
    torch_doc = get_doc_content(torch_api, "pytorch")
    with print_lock:
        print(f"  获取 {paddle_api} 文档...")
    paddle_doc = get_doc_content(paddle_api, "paddle")
    
    all_results = []
    bug_candidates = 0
    
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {}
            
            for idx, case in enumerate(success_cases, 1):
                future = executor.submit(
                    process_single_case,
                    client,
                    case,
                    idx,
                    len(success_cases),
                    torch_doc,
                    paddle_doc,
                    model,
                    print_lock,
                    workers
                )
                future_to_idx[future] = idx
            
            results_dict = {}
            for future in as_completed(future_to_idx):
                try:
                    idx = future_to_idx[future]
                    case_result = future.result(timeout=300)
                    results_dict[idx] = case_result
                    
                    for fr in case_result.get("fuzzing_results", []):
                        if fr.get("is_bug_candidate"):
                            bug_candidates += 1
                except TimeoutError:
                    idx = future_to_idx[future]
                    with print_lock:
                        print(f"[ERROR] 测试用例 {idx} 超时")
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
    
    for idx in sorted(results_dict.keys()):
        all_results.append(results_dict[idx])
    
    return {
        "operator": operator_name,
        "torch_api": torch_api,
        "paddle_api": paddle_api,
        "total_cases": len(success_cases),
        "total_fuzzing_rounds": len(success_cases) * FUZZING_ROUNDS,
        "bug_candidates": bug_candidates,
        "timestamp": datetime.now().isoformat(),
        "results": all_results
    }


def save_operator_result(result: Dict[str, Any], output_dir: Path) -> None:
    """
    保存算子的 fuzzing 结果
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
    
    def signal_handler(signum, frame):
        print(f"\n[WARN] 收到信号 {signum}，程序即将退出...")
        sys.exit(1)
    
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except (AttributeError, ValueError):
        pass
    
    parser = argparse.ArgumentParser(
        description="PyTorch-PaddlePaddle 基于 LLM 的 Fuzzing 差分测试（并发版本）"
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
    print("PyTorch-PaddlePaddle 基于 LLM 的 Fuzzing 差分测试（并发版本）")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"每用例 Fuzzing 轮数: {FUZZING_ROUNDS}")
    print(f"并发线程数: {args.workers}")
    
    try:
        client = get_qwen_client(args.key_path)
        print("LLM 客户端初始化成功")
    except Exception as e:
        print(f"[ERROR] 无法初始化 LLM 客户端: {e}")
        return
    
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
    
    total_available = len(operator_files)
    if args.start > total_available:
        print(f"[ERROR] 起始索引 {args.start} 超出范围（共有 {total_available} 个算子）")
        return
    
    start_idx = args.start - 1
    end_idx = args.end if args.end is not None else total_available
    
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
    
    total_operators = len(operator_files)
    total_bug_candidates = 0
    
    print_lock = Lock()
    start_time = time.time()
    
    for idx, op_file in enumerate(operator_files, 1):
        print(f"\n[{idx}/{total_operators}] 处理: {op_file.stem}")
        
        try:
            result = process_operator(
                op_file, client, args.model, args.max_cases, print_lock, args.workers
            )
            
            save_operator_result(result, RESULT_DIR)
            
            total_bug_candidates += result.get("bug_candidates", 0)
            
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
            import gc
            gc.collect()
            continue
        except Exception as e:
            print(f"[ERROR] 处理 {op_file.stem} 时出错: {e}")
            traceback.print_exc()
            continue
    
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
