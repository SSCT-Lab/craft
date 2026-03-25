"""  
PyTorch-PaddlePaddle 基于 LLM 的 Fuzzing 差分测试工具

功能说明:
    1. 读取按算子分类的成功测试用例
    2. 爬取 PyTorch 和 PaddlePaddle 官方文档
    3. 使用 LLM 变异测试用例（复杂输入、极端值、边界值）
    4. 执行差分测试，检测框架不一致或潜在 bug
    5. 每个用例进行 3 轮 fuzzing
"""

import json
import sys
import argparse
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    
    # 不同轮次的变异策略
    mutation_strategies = {
        1: "极端数值变异：使用复杂的浮点数、极大值(1e38)、极小值(1e-38)、无穷大(inf)、负无穷(-inf)、NaN、零(0)、负零(-0.0)等特殊数值",
        2: "边界形状变异：使用空张量(shape含0)、标量(shape=[])、超高维张量(5维以上)、不规则形状、单元素张量等边界情况",
        3: "复杂类型变异：测试不同数据类型(float32/float64/int32/int64/bool)，注意：PaddlePaddle CPU 不支持 float16，请避免使用"
    }
    
    current_strategy = mutation_strategies.get(round_num, mutation_strategies[1])
    
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

【核心变异要求：数学等价性（最重要！）】
**变异后的两个测试用例必须在数学上完全等价，理论输出必须相同。**

你可以自由地：
- 修改输入数据（shape、dtype、数值）
- 添加或修改参数
- 使用任何合法的参数组合

但你必须确保：
- 如果 PyTorch 使用了某个参数（如 `alpha=2.0`），PaddlePaddle 侧必须做等价处理
- 例如：`torch.add(x, y, alpha=2.0)` 等价于 `paddle.add(x, paddle.scale(y, scale=2.0))`
- 如果无法在 PaddlePaddle 侧实现等价逻辑，则**不要使用该参数**

【具体变异要求】
1. **输入必须完全相同**：两个框架的输入张量的 shape、dtype、数值必须完全一致
2. **参数语义等价**：如果 API 有额外参数，两边的参数值必须在语义上等价
3. **探索边界情况**：重点测试极端值、边界形状、特殊 dtype 等可能导致两框架行为不一致的边界场景
4. **保持可执行性**：变异后的用例必须是合法的，能够被框架正确执行

【重要提示】
- 根据文档确认两个 API 的参数映射关系
- 注意数据类型的兼容性（如某些 dtype 可能只有一个框架支持）
- 考虑数值稳定性问题（如除零、log(0)、sqrt(-1)等）
- shape 变异时注意算子对维度的要求
- PaddlePaddle 的数据格式通常与 PyTorch 相同（NCHW）

【已知框架限制 - 必须避免！】
- **PaddlePaddle CPU 不支持 float16**：请勿使用 float16 dtype，会导致 "kernel not registered" 错误
- **PaddlePaddle 的 y 参数必须是 Tensor**：paddle.add(x, y) 的 y 不能是 Python 标量，必须是 Tensor
- 推荐使用的 dtype：float32、float64、int32、int64、bool

【参数处理原则】
- 可以自由添加、修改、删除参数，只要保证数学等价
- 如果某参数只有一个框架支持，要么不用，要么在另一框架手动实现等价逻辑
- 参数名按框架文档正确映射（可能会有对应参数名称不同的情况）

【错误示例 - 不要这样做！】
❌ PyTorch 用了某参数，PaddlePaddle 却没有对应处理（以 torch.addcmul 的 value 为例）：
   torch.addcmul(input, tensor1, tensor2, value=2.0)  →  input + 2.0 * tensor1 * tensor2
   paddle.addcmul(input, tensor1, tensor2)            →  input + 1.0 * tensor1 * tensor2
   这两个在数学上不等价，会产生假阳性！

【正确示例】
✓ 方案1：两边都使用相同的参数值
   torch.addcmul(input, tensor1, tensor2, value=2.0)   →  input + 2*tensor1*tensor2
   paddle.addcmul(input, tensor1, tensor2, value=2.0)  →  input + 2*tensor1*tensor2

✓ 方案2：两边都不用可选参数（使用默认值）
   torch.addcmul(input, tensor1, tensor2)   →  input + tensor1*tensor2
   paddle.addcmul(input, tensor1, tensor2)  →  input + tensor1*tensor2

【输出格式要求】
**重要：保持输出简洁！** mutation_strategy 限制在 50 字以内，mutation_reason 限制在 150 字以内。
请严格按照以下 JSON 格式输出变异后的测试用例，不要输出任何其他内容：
```json
{{
  "mutation_strategy": "简要描述本次变异策略",
  "mutation_reason": "详细解释为什么这样变异可能发现问题",
  "torch_test_case": {{
    "api": "{torch_api}",
    "input": {{
      "shape": [...],
      "dtype": "...",
      "sample_values": [...]
    }},
    // 如果需要额外输入（如 torch.add 的 other），添加对应字段
    // 如："other": {{ "shape": [...], "dtype": "...", "sample_values": [...] }} 或标量值
    // 如果需要额外参数（如 AvgPool1d 的 kernel_size, stride），也添加对应字段
    // 如："kernel_size": 3, "stride": 2, "padding": 0 等
  }},
  "paddle_test_case": {{
    "api": "{paddle_api}",
    "input": {{
      "shape": [...],
      "dtype": "...",
      "sample_values": [...]
    }},
    // PaddlePaddle 对应的参数，注意参数名可能不同（但必须与 PyTorch 侧数学等价）
    // 如 pool_size 对应 kernel_size，strides 对应 stride
  }}
}}
```

【重要】关于多输入和参数的处理：
1. sample_values 应该包含足够的数值来填充整个张量，如果张量很大，可以只提供前几个值作为种子
2. **如果算子有多个输入**（如 torch.add 的 input 和 other），必须都包含在输出中
3. **参数名映射**：PyTorch 和 PaddlePaddle 的参数名、参数值类型和数量都可能不同，需要根据文档正确映射：
   - torch.nn.AvgPool1d(kernel_size, stride) <-> paddle.nn.AvgPool1D(kernel_size, stride)
   - torch.add(input, other) <-> paddle.add(x, y)
4. dtype 必须是两个框架都支持的类型
"""
    return prompt


def preprocess_json_string(json_str: str) -> str:
    """
    预处理 JSON 字符串，将 Python 语法的特殊值转换为 JSON 兼容格式
    
    LLM 有时会在 JSON 中使用 Python 语法（如 float('inf')），
    这不是合法 JSON，需要转换为字符串形式。
    
    注意：只替换作为 JSON 值出现的特殊值，不替换字符串内容中的文本。
    """
    import re
    
    # 只替换 float('...') 形式的 Python 语法
    # 这些明确是作为值出现的，不会在字符串内容中
    patterns_float = [
        # float('inf') 或 float("inf") 或 float('+inf')
        (r"float\s*\(\s*['\"]?\+?inf['\"]?\s*\)", '"Infinity"'),
        # float('-inf') 或 float("-inf")
        (r"float\s*\(\s*['\"]?-inf['\"]?\s*\)", '"-Infinity"'),
        # float('nan') 或 float("nan")
        (r"float\s*\(\s*['\"]?nan['\"]?\s*\)", '"NaN"'),
    ]
    
    result = json_str
    for pattern, replacement in patterns_float:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    # 对于裸的 inf/nan，只在数组上下文中替换（前后是逗号、方括号或空白）
    # 这样可以避免替换字符串内容中的文本
    # 匹配模式：在 [ 或 , 后面，或在 ] 或 , 前面的 inf/nan
    patterns_bare = [
        # 数组中的 inf (前面是 [ 或 , 或空白，后面是 ] 或 , 或空白)
        (r'(?<=[\[,])\s*inf\s*(?=[\],])', '"Infinity"'),
        (r'(?<=[\[,])\s*\+inf\s*(?=[\],])', '"Infinity"'),
        # 数组中的 -inf
        (r'(?<=[\[,])\s*-inf\s*(?=[\],])', '"-Infinity"'),
        # 数组中的 nan
        (r'(?<=[\[,])\s*nan\s*(?=[\],])', '"NaN"'),
    ]
    
    for pattern, replacement in patterns_bare:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result


def parse_llm_response(response: str) -> Optional[Dict[str, Any]]:
    """
    解析 LLM 返回的 JSON 响应，支持处理截断和格式错误
    """
    try:
        # 尝试提取 JSON 块
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            # 如果没有找到结束符（被截断），取剩余所有内容
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
            # 尝试直接解析
            json_str = response.strip()
        
        # 预处理：将 Python 语法的特殊值转换为 JSON 兼容格式
        json_str = preprocess_json_string(json_str)
        
        # 尝试直接解析
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # 尝试修复截断的 JSON
        repaired_json = try_repair_json(json_str)
        if repaired_json:
            return repaired_json
        
        return None
    except Exception as e:
        print(f"[WARN] 解析 LLM 响应失败: {e}")
        return None


def try_repair_json(json_str: str) -> Optional[Dict[str, Any]]:
    """
    尝试修复不完整的 JSON 字符串
    """
    import re
    
    # 方法1：检查并补全缺失的括号
    # 统计括号数量
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    
    # 尝试补全缺失的括号
    repaired = json_str
    
    # 移除末尾可能的不完整字符串（未闭合的引号内容）
    # 查找最后一个完整的键值对
    patterns_to_try = [
        # 情况1：截断在字符串值中间 "key": "value...
        (r',?\s*"[^"]*"\s*:\s*"[^"]*$', ''),
        # 情况2：截断在键名后 "key":
        (r',?\s*"[^"]*"\s*:\s*$', ''),
        # 情况3：截断在逗号后
        (r',\s*$', ''),
        # 情况4：截断在数组中间
        (r',?\s*\d+\.?\d*\s*$', ''),
    ]
    
    for pattern, replacement in patterns_to_try:
        test_str = re.sub(pattern, replacement, repaired)
        # 补全括号
        test_open_braces = test_str.count('{')
        test_close_braces = test_str.count('}')
        test_open_brackets = test_str.count('[')
        test_close_brackets = test_str.count(']')
        
        test_str += ']' * (test_open_brackets - test_close_brackets)
        test_str += '}' * (test_open_braces - test_close_braces)
        
        try:
            result = json.loads(test_str)
            # 验证必要字段存在
            if "torch_test_case" in result or "mutation_strategy" in result:
                print(f"[INFO] JSON 修复成功")
                return result
        except json.JSONDecodeError:
            continue
    
    # 方法2：简单地补全括号
    repaired += ']' * (open_brackets - close_brackets)
    repaired += '}' * (open_braces - close_braces)
    
    try:
        result = json.loads(repaired)
        if "torch_test_case" in result or "mutation_strategy" in result:
            print(f"[INFO] JSON 简单修复成功")
            return result
    except json.JSONDecodeError:
        pass
    
    # 方法3：尝试提取已有的完整部分
    # 找到最后一个完整的 } 或 ] 前的内容
    for i in range(len(json_str) - 1, -1, -1):
        if json_str[i] in '}]':
            try:
                test_str = json_str[:i+1]
                # 补全括号
                test_open_braces = test_str.count('{')
                test_close_braces = test_str.count('}')
                test_open_brackets = test_str.count('[')
                test_close_brackets = test_str.count(']')
                
                test_str += ']' * (test_open_brackets - test_close_brackets)
                test_str += '}' * (test_open_braces - test_close_braces)
                
                result = json.loads(test_str)
                if "torch_test_case" in result or "mutation_strategy" in result:
                    print(f"[INFO] JSON 部分提取成功")
                    return result
            except json.JSONDecodeError:
                continue
    
    print(f"[WARN] JSON 修复失败")
    return None


def parse_special_value(val: Any) -> float:
    """
    解析特殊值（如 "inf", "-inf", "nan", "Infinity", "-Infinity", "NaN" 字符串）
    支持 JSON 标准格式和 Python 风格格式
    """
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val_lower = val.lower().strip()
        # 支持多种格式：inf, +inf, infinity, Infinity
        if val_lower in ("inf", "+inf", "infinity"):
            return np.inf
        elif val_lower in ("-inf", "-infinity"):
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
        framework: 'torch' 或 'paddle'
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
            "num_features", "normalized_shape", "weight", "bias",
            "return_indices", "exclusive", "reverse"
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


def create_paddle_tensor(spec: Any, paddle_dtype_map: Dict) -> Any:
    """
    根据规格创建 PaddlePaddle 张量，支持张量和标量
    """
    import paddle
    
    if isinstance(spec, (int, float, bool)):
        # PaddlePaddle 的很多 API（如 paddle.add）要求输入必须是 Tensor
        # 所以需要将标量转换为 0 维 Tensor
        return paddle.to_tensor(spec)
    if isinstance(spec, dict) and "shape" in spec:
        data, dtype_str = create_tensor_from_spec(spec, "paddle")
        paddle_dtype = paddle_dtype_map.get(dtype_str, "float32")
        return paddle.to_tensor(data, dtype=paddle_dtype)
    return spec


def execute_paddle_test(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行 PaddlePaddle 测试用例，支持多输入和额外参数
    """
    try:
        import paddle
        
        api_name = test_case.get("api", "")
        
        # PaddlePaddle dtype 映射
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
        
        # PaddlePaddle 的非张量参数名
        non_tensor_params = {
            "kernel_size", "stride", "padding", "dilation", "groups",
            "ceil_mode", "count_include_pad", "divisor_override", "exclusive",
            "axis", "keepdim", "name", "dtype", "data_format",
            "epsilon", "weight_attr", "bias_attr", "return_indices",
            "p", "eps", "momentum", "use_input_stats", "reverse"
        }
        
        # 分离输入张量和参数
        args = []
        kwargs = {}
        
        # 处理主输入
        if "input" in test_case:
            input_tensor = create_paddle_tensor(test_case["input"], paddle_dtype_map)
            args.append(input_tensor)
        
        # 处理其他可能的输入张量
        tensor_input_names = ["other", "x", "y", "tensor", "input1", "input2"]
        for name in tensor_input_names:
            if name in test_case:
                tensor = create_paddle_tensor(test_case[name], paddle_dtype_map)
                args.append(tensor)
        
        # 处理非张量参数
        for key, value in test_case.items():
            if key not in ["api", "input", "x", "y", "other", "tensor", "input1", "input2"]:
                if key in non_tensor_params or not isinstance(value, dict):
                    kwargs[key] = value
        
        # 获取 API 函数/类
        api_parts = api_name.split(".")
        func = paddle
        for part in api_parts[1:]:  # 跳过 'paddle'
            func = getattr(func, part)
        
        # 判断是否是 nn 层（如 paddle.nn.AvgPool1D）
        is_nn_layer = "nn" in api_name and not "functional" in api_name
        
        if is_nn_layer and isinstance(func, type):
            # nn 层：先实例化再调用
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
        
        # 转换结果
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
    
    # 如果有一方失败
    if not torch_result["success"] or not paddle_result["success"]:
        if torch_result["success"] != paddle_result["success"]:
            comparison["comparison_error"] = "执行状态不一致：一方成功一方失败"
        return comparison
    
    # 比较结果
    try:
        torch_res = torch_result["result"]
        paddle_res = paddle_result["result"]
        
        if torch_res is None and paddle_res is None:
            comparison["results_match"] = True
            return comparison
        
        if isinstance(torch_res, np.ndarray) and isinstance(paddle_res, np.ndarray):
            # 形状检查
            if torch_res.shape != paddle_res.shape:
                comparison["comparison_error"] = f"形状不一致: torch={torch_res.shape}, paddle={paddle_res.shape}"
                return comparison
            
            # 处理特殊值
            torch_nan = np.isnan(torch_res) if np.issubdtype(torch_res.dtype, np.floating) else np.zeros_like(torch_res, dtype=bool)
            paddle_nan = np.isnan(paddle_res) if np.issubdtype(paddle_res.dtype, np.floating) else np.zeros_like(paddle_res, dtype=bool)
            
            torch_inf = np.isinf(torch_res) if np.issubdtype(torch_res.dtype, np.floating) else np.zeros_like(torch_res, dtype=bool)
            paddle_inf = np.isinf(paddle_res) if np.issubdtype(paddle_res.dtype, np.floating) else np.zeros_like(paddle_res, dtype=bool)
            
            # NaN 位置必须一致
            if not np.array_equal(torch_nan, paddle_nan):
                comparison["comparison_error"] = "NaN 位置不一致"
                return comparison
            
            # Inf 位置和符号必须一致
            if not np.array_equal(torch_inf, paddle_inf):
                comparison["comparison_error"] = "Inf 位置不一致"
                return comparison
            
            if np.any(torch_inf):
                if not np.array_equal(np.sign(torch_res[torch_inf]), np.sign(paddle_res[paddle_inf])):
                    comparison["comparison_error"] = "Inf 符号不一致"
                    return comparison
            
            # 对于非特殊值进行数值比较
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
            # 非张量结果
            if torch_res == paddle_res:
                comparison["results_match"] = True
            else:
                comparison["comparison_error"] = f"结果不一致: torch={torch_res}, paddle={paddle_res}"
                
    except Exception as e:
        comparison["comparison_error"] = f"比较过程出错: {str(e)}"
    
    return comparison


def run_fuzzing_for_case(
    client,
    original_case: Dict[str, Any],
    torch_doc: str,
    paddle_doc: str,
    model: str = DEFAULT_MODEL
) -> List[Dict[str, Any]]:
    """
    对单个测试用例进行多轮 fuzzing
    """
    torch_case = original_case.get("torch_test_case", {})
    paddle_case = original_case.get("paddle_test_case", {})
    torch_api = torch_case.get("api", "")
    paddle_api = paddle_case.get("api", "")
    
    fuzzing_results = []
    
    for round_num in range(1, FUZZING_ROUNDS + 1):
        print(f"    [Round {round_num}/{FUZZING_ROUNDS}] 生成变异用例...")
        
        # 构建提示词
        prompt = build_fuzzing_prompt(
            torch_api, paddle_api, original_case, torch_doc, paddle_doc, round_num
        )
        
        try:
            # 调用 LLM（带重试机制）
            max_retries = 2
            mutated_case = None
            llm_response = ""
            
            for retry in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7 + retry * 0.1,  # 重试时稍微增加随机性
                        max_tokens=8192,  # 增加 token 限制避免中文响应被截断
                    )
                    llm_response = response.choices[0].message.content.strip()
                    
                    # 解析响应
                    mutated_case = parse_llm_response(llm_response)
                    
                    if mutated_case is not None:
                        # 验证必要字段
                        if "torch_test_case" in mutated_case and "paddle_test_case" in mutated_case:
                            break
                        else:
                            print(f"[WARN] Round {round_num} 响应缺少必要字段，重试 ({retry + 1}/{max_retries})")
                            mutated_case = None
                    else:
                        if retry < max_retries - 1:
                            print(f"[WARN] Round {round_num} 解析失败，重试 ({retry + 1}/{max_retries})")
                except Exception as e:
                    print(f"[WARN] Round {round_num} LLM 调用异常: {e}，重试 ({retry + 1}/{max_retries})")
                    if retry < max_retries - 1:
                        import time
                        time.sleep(1)  # 等待 1 秒后重试
            
            if mutated_case is None:
                fuzzing_results.append({
                    "round": round_num,
                    "success": False,
                    "error": "LLM 响应解析失败",
                    "llm_response": llm_response[:1000]  # 保留更多响应用于调试
                })
                continue
            
            print(f"    [Round {round_num}] 执行差分测试...")
            
            # 执行测试
            torch_test = mutated_case.get("torch_test_case", {})
            paddle_test = mutated_case.get("paddle_test_case", {})
            
            torch_result = execute_torch_test(torch_test)
            paddle_result = execute_paddle_test(paddle_test)
            
            # 比较结果
            comparison = compare_results(torch_result, paddle_result)
            
            # 记录结果
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
            fuzzing_results.append(fuzzing_result)
            
            # 打印结果摘要
            if fuzzing_result["is_bug_candidate"]:
                print(f"    [Round {round_num}] ⚠️ 发现潜在问题: {comparison.get('comparison_error') or '执行状态不一致'}")
            else:
                print(f"    [Round {round_num}] ✓ 结果一致")
                
        except Exception as e:
            fuzzing_results.append({
                "round": round_num,
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "traceback": traceback.format_exc()
            })
            print(f"    [Round {round_num}] ✗ 错误: {e}")
    
    return fuzzing_results


def process_operator(
    operator_file: Path,
    client,
    model: str,
    max_cases: Optional[int] = None
) -> Dict[str, Any]:
    """
    处理单个算子的所有成功用例
    """
    with open(operator_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    operator_name = data.get("operator", "unknown")
    success_cases = data.get("success_cases", [])
    
    if max_cases:
        success_cases = success_cases[:max_cases]
    
    print(f"\n处理算子: {operator_name} ({len(success_cases)} 个用例)")
    
    # 获取 API 名称
    if success_cases:
        torch_api = success_cases[0].get("torch_test_case", {}).get("api", "")
        paddle_api = success_cases[0].get("paddle_test_case", {}).get("api", "")
    else:
        return {"operator": operator_name, "results": [], "error": "无成功用例"}
    
    # 爬取文档
    print(f"  获取 {torch_api} 文档...")
    torch_doc = get_doc_content(torch_api, "pytorch")
    print(f"  获取 {paddle_api} 文档...")
    paddle_doc = get_doc_content(paddle_api, "paddle")
    
    # 对每个用例进行 fuzzing
    all_results = []
    bug_candidates = 0
    
    for idx, case in enumerate(success_cases, 1):
        print(f"\n  用例 {idx}/{len(success_cases)} (iteration={case.get('iteration')}, case={case.get('case_number')})")
        
        fuzzing_results = run_fuzzing_for_case(
            client, case, torch_doc, paddle_doc, model
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
        all_results.append(case_result)
        
        # 统计潜在 bug
        for fr in fuzzing_results:
            if fr.get("is_bug_candidate"):
                bug_candidates += 1
    
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
    保存算子的 fuzzing 结果，文件名包含时间戳
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    operator_name = result.get("operator", "unknown")
    # 添加时间戳后缀 YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{operator_name}_fuzzing_result_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"  结果已保存: {output_file}")


def main():
    """
    主程序入口
    """
    parser = argparse.ArgumentParser(
        description="PyTorch-PaddlePaddle 基于 LLM 的 Fuzzing 差分测试"
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
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PyTorch-PaddlePaddle 基于 LLM 的 Fuzzing 差分测试")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"每用例 Fuzzing 轮数: {FUZZING_ROUNDS}")
    
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
    
    print(f"待处理算子数: {len(operator_files)}")
    print("=" * 70)
    
    # 统计
    total_operators = len(operator_files)
    total_bug_candidates = 0
    
    # 处理每个算子
    for idx, op_file in enumerate(operator_files, 1):
        print(f"\n[{idx}/{total_operators}] 处理: {op_file.stem}")
        
        try:
            result = process_operator(
                op_file, client, args.model, args.max_cases
            )
            
            # 保存结果
            save_operator_result(result, RESULT_DIR)
            
            total_bug_candidates += result.get("bug_candidates", 0)
            
        except Exception as e:
            print(f"[ERROR] 处理 {op_file.stem} 时出错: {e}")
            traceback.print_exc()
    
    # 打印总结
    print("\n" + "=" * 70)
    print("Fuzzing 测试完成！")
    print("=" * 70)
    print(f"处理算子数: {total_operators}")
    print(f"发现潜在问题数: {total_bug_candidates}")
    print(f"结果保存目录: {RESULT_DIR}")


if __name__ == "__main__":
    main()
