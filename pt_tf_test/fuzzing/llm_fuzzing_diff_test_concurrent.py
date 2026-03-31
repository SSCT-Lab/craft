"""  
PyTorch-TensorFlow Fuzzing differential testing tool based on LLM (concurrent version)  Function description:
    1. Read successful test cases classified by operator
    2. Crawl PyTorch and TensorFlow official documents
    3. Use LLM mutation test cases (complex inputs, extreme values, boundary values）
    4. Perform differential testing to detect framework inconsistencies or potential bug
    5. 3 rounds per use case fuzzing
    6. Supports multi-threaded concurrent processing (default 8 threads）
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

# Add project root directory to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_crawler_factory import get_doc_content
from component.migration.migrate_generate_tests import get_qwen_client

# ==================== constant definition ====================
DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"
FUZZING_ROUNDS = 3  # Number of fuzzing rounds per use case
DEFAULT_WORKERS = 8  # Default number of concurrent threads

# Successful use case catalog
SUCCESS_CASES_DIR = Path(__file__).parent / "success_cases"
# Fuzzing Results directory
RESULT_DIR = Path(__file__).parent / "result"

# Operators that require data format conversion（NCHW <-> NHWC）
CONV_OPS_NEED_TRANSPOSE = {
    # convolution correlation
    "torch.nn.Conv1d", "torch.nn.Conv2d", "torch.nn.Conv3d",
    "torch.nn.ConvTranspose1d", "torch.nn.ConvTranspose2d", "torch.nn.ConvTranspose3d",
    "torch.nn.functional.conv1d", "torch.nn.functional.conv2d", "torch.nn.functional.conv3d",
    "torch.nn.functional.conv_transpose1d", "torch.nn.functional.conv_transpose2d", "torch.nn.functional.conv_transpose3d",
    # Pooling related
    "torch.nn.MaxPool1d", "torch.nn.MaxPool2d", "torch.nn.MaxPool3d",
    "torch.nn.AvgPool1d", "torch.nn.AvgPool2d", "torch.nn.AvgPool3d",
    "torch.nn.AdaptiveMaxPool1d", "torch.nn.AdaptiveMaxPool2d", "torch.nn.AdaptiveMaxPool3d",
    "torch.nn.AdaptiveAvgPool1d", "torch.nn.AdaptiveAvgPool2d", "torch.nn.AdaptiveAvgPool3d",
    "torch.nn.functional.max_pool1d", "torch.nn.functional.max_pool2d", "torch.nn.functional.max_pool3d",
    "torch.nn.functional.avg_pool1d", "torch.nn.functional.avg_pool2d", "torch.nn.functional.avg_pool3d",
    # normalized correlation
    "torch.nn.BatchNorm1d", "torch.nn.BatchNorm2d", "torch.nn.BatchNorm3d",
    "torch.nn.InstanceNorm1d", "torch.nn.InstanceNorm2d", "torch.nn.InstanceNorm3d",
}


def needs_data_format_conversion(api_name: str) -> bool:
    """
    Determine whether the operator requires data format conversion（NCHW <-> NHWC）
    """
    return api_name in CONV_OPS_NEED_TRANSPOSE


def convert_nchw_to_nhwc(tensor: np.ndarray) -> np.ndarray:
    """
    Convert NCHW format to NHWC format
    
    - 4D: (N, C, H, W) -> (N, H, W, C)
    - 3D: (N, C, L) -> (N, L, C)
    - 5D: (N, C, D, H, W) -> (N, D, H, W, C)
    """
    ndim = tensor.ndim
    if ndim == 4:  # 2D convolution/Pooling
        return np.transpose(tensor, (0, 2, 3, 1))
    elif ndim == 3:  # 1D convolution/Pooling
        return np.transpose(tensor, (0, 2, 1))
    elif ndim == 5:  # 3D convolution/Pooling
        return np.transpose(tensor, (0, 2, 3, 4, 1))
    else:
        return tensor


def convert_nhwc_to_nchw(tensor: np.ndarray) -> np.ndarray:
    """
    Convert NHWC format to NCHW format
    
    - 4D: (N, H, W, C) -> (N, C, H, W)
    - 3D: (N, L, C) -> (N, C, L)
    - 5D: (N, D, H, W, C) -> (N, C, D, H, W)
    """
    ndim = tensor.ndim
    if ndim == 4:  # 2D convolution/Pooling
        return np.transpose(tensor, (0, 3, 1, 2))
    elif ndim == 3:  # 1D convolution/Pooling
        return np.transpose(tensor, (0, 2, 1))
    elif ndim == 5:  # 3D convolution/Pooling
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
    Building LLM fuzzing prompt words          parameters:
        torch_api: PyTorch API name
        tf_api: TensorFlow API name
        original_case: Original test case
        torch_doc: PyTorch Document content
        tf_doc: TensorFlow Document content
        round_num: Current fuzzing round
    """
    torch_case = original_case.get("torch_test_case", {})
    tf_case = original_case.get("tensorflow_test_case", {})
    
    torch_case_json = json.dumps(torch_case, ensure_ascii=False, indent=2)
    tf_case_json = json.dumps(tf_case, ensure_ascii=False, indent=2)
    
    # Check whether it is a logical operation operator
    is_logical_op = any(keyword in torch_api.lower() for keyword in ["logical_", "bitwise_"])
    
    # Mutation strategies for different rounds
    mutation_strategies = {
        1: "Extreme numerical variation: using complex floating point numbers, maximum value (1e38), minimum value (1e-38), infinity (inf), negative infinity (-inf), NaN, zero (0), negative zero(-0.0)and other special values",
        2: "Boundary shape mutation: use empty tensor (shape contains 0), scalar(shape=[])、Boundary situations such as ultra-high-dimensional tensors (more than 5 dimensions), irregular shapes, single-element tensors, etc.",
        3: "Complex type mutation: testing different data types(float16/float32/float64/int32/int64/bool/complex64/complex128)、mixed precision scene"
    }
    
    current_strategy = mutation_strategies.get(round_num, mutation_strategies[1])
    
    # Add special prompts for logical operations
    dtype_constraint = ""
    if is_logical_op:
        dtype_constraint = """
【Important: Data type constraints】
**This operator is a logical operation and must use the bool type！**
- PyTorch and TensorFlow's logical operations (such as logical_and, logical_or, logical_xor）Both require input to be of bool type - The dtype must be maintained when mutating. "bool"
- Do not use float/int Wait for other types, otherwise the execution will fail.
"""
    
    # Check whether it is an operator that requires attention to the data format.
    needs_format_attention = needs_data_format_conversion(torch_api)
    
    # Data format description (only add the required operators）
    data_format_instruction = ""
    if needs_format_attention:
        data_format_instruction = """
【Important: Data format differences】
**PyTorch Convolution with TensorFlow/Pooling/Normalization operators use different data formats：**
- PyTorch Use NCHW format（channels_first）：
  - 4D: (N, C, H, W) - batch, channels, height, width
  - 3D: (N, C, L) - batch, channels, length
  - 5D: (N, C, D, H, W) - batch, channels, depth, height, width
  
- TensorFlow Default uses NHWC format（channels_last）：
  - 4D: (N, H, W, C) - batch, height, width, channels
  - 3D: (N, L, C) - batch, length, channels  
  - 5D: (N, D, H, W, C) - batch, depth, height, width, channels

**Required when generating test cases：**
1. PyTorch The input shape of the test case uses NCHW format
2. TensorFlow The input shape of the test case uses NHWC format
3. Make sure the data from both are mathematically equivalent (just in different order)）
4. sample_values The order of filling should correspond to the respective formats.
5. **TensorFlow Keras The layer's data_format parameter only accepts the following values ​​(strictly case-sensitive）：**
   - "channels_last" （correspond NHWC/NWC/NDHWC）
   - "channels_first" （correspond NCHW/NCW/NCDHW）
   - **Do not use "NHWC", "NCHW", "NWC" Wait, these will cause errors！**

For example, for batch=1, channels=3, height=4, width=4 input：
- PyTorch shape: [1, 3, 4, 4]
- TensorFlow shape: [1, 4, 4, 3]，and set data_format: "channels_last"
"""
    
    prompt = f"""You are a professional deep learning framework testing expert, and now you need to perform differential testing on the equivalent operators of PyTorch and TensorFlow.  【Test goal】 What we are looking for is**potential in framework implementation bug**，That is: mathematically equivalent operations, but the two frameworks produce different outputs。
- ✓ What to look for: Framework implementation differences (real bug）
- ✗ What not to look for: Differences caused by non-equivalence of the test cases themselves (false positives）
{data_format_instruction}
【The currently tested operator pair】
- PyTorch API: {torch_api}
- TensorFlow API: {tf_api}

【Original successful test case] PyTorch test cases:
```json
{torch_case_json}
```

TensorFlow test case:
```json
{tf_case_json}
```

【PyTorch Official documentation】
{torch_doc if torch_doc else "Document retrieval failed"}

【TensorFlow Official documentation】
{tf_doc if tf_doc else "Document retrieval failed"}

【Mutation strategy for this round - Chapter{round_num}wheel】
{current_strategy}

{dtype_constraint}

【Core mutation requirements: Mathematical equivalence (most important！）】
**The two test cases after mutation must be completely equivalent mathematically, and the theoretical output must be the same。**

【Key requirement: Parameter integrity】
**Must contain all required parameters from the original test case！**
- If the original use case had `other` Parameters, the mutated use case must also have `other` parameters - If the original use case has `dim`/`axis` Parameters, the mutated use case must also have corresponding parameters - If the original use case has other configuration parameters (such as `kernel_size`, `stride` etc.), must also be retained after mutation
- **Do not omit any required parameters, otherwise the execution will fail！**

【Output format requirements] Please strictly follow the following JSON format to output the mutated test cases, and do not output any other content.：

**Important: JSON representation of special floating point values**
- Positive infinity: using strings "inf" or "Infinity"
- Negative infinity: using strings "-inf" or "-Infinity"  
- NaN：use string "nan" or "NaN"
- Negative zero: Use a numeric value -0.0
- Don't use Python syntax like float('inf')，This is illegal in JSON！

```json
{{
  "mutation_strategy": "Briefly describe this mutation strategy (no more than 50 words)）",
  "mutation_reason": "Briefly explain why this mutation may uncover the problem (no more than 100 words）",
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
    "Other required parameters": ...
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
    "Other required parameters": ...
  }}
}}
```

**Notice**：
1. mutation_strategy and mutation_reason should be concise to avoid being too long and causing the token to exceed the limit.
2. Special values ​​must be represented as strings（"inf", "-inf", "nan"）
3. Make sure the JSON is well-formed and all brackets and quotes are closed
4. **All parameters from the original use case must be preserved (e.g. other, dim, axis wait）**
5. **PyTorch All input tensors to the TensorFlow test case must use the same dtype (TensorFlow does not support mixed precision）**
"""
    return prompt


def parse_llm_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse the JSON response returned by LLM, supporting handling of truncation and format errors
    """
    try:
        # Try to extract JSON chunks
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
        
        # Preprocessing: Replace Python float() syntax with JSON string
        json_str = fix_python_float_syntax(json_str)
        
        # Try to parse directly
        try:
            parsed = json.loads(json_str)
            # Validate required fields
            if validate_parsed_json(parsed):
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Try to fix the truncated JSON
        repaired_json = try_repair_json(json_str)
        if repaired_json and validate_parsed_json(repaired_json):
            return repaired_json
        
        return None
    except Exception as e:
        print(f"[WARN] Failed to parse LLM response: {e}")
        return None


def fix_python_float_syntax(json_str: str) -> str:
    """
    Replace Python float() syntax with JSON-compatible string format          For example:
        float('inf') -> "inf"
        float('-inf') -> "-inf"
        float('nan') -> "nan"
    """
    import re
    
    # replace float('inf')
    json_str = re.sub(r"float\s*\(\s*['\"]inf['\"]\s*\)", '"inf"', json_str, flags=re.IGNORECASE)
    
    # replace float('-inf')
    json_str = re.sub(r"float\s*\(\s*['\"]-inf['\"]\s*\)", '"-inf"', json_str, flags=re.IGNORECASE)
    
    # replace float('nan')
    json_str = re.sub(r"float\s*\(\s*['\"]nan['\"]\s*\)", '"nan"', json_str, flags=re.IGNORECASE)
    
    # replace float('+inf')
    json_str = re.sub(r"float\s*\(\s*['\"]\+inf['\"]\s*\)", '"inf"', json_str, flags=re.IGNORECASE)
    
    return json_str


def validate_parsed_json(parsed: Dict[str, Any]) -> bool:
    """
    Verify that the parsed JSON contains necessary fields          Return:
        True False if required fields are included otherwise
    """
    if not isinstance(parsed, dict):
        return False
    
    # Check required fields
    required_fields = ["torch_test_case", "tensorflow_test_case"]
    for field in required_fields:
        if field not in parsed:
            return False
        if not isinstance(parsed[field], dict):
            return False
    
    return True


def try_repair_json(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Try to fix incomplete JSON string          Handle common truncation situations:
    1. Unclosed arrays and objects
    2. truncated string value
    3. truncated value
    4. truncated special values ​​(such as float('inf')）
    """
    import re
    
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    
    repaired = json_str
    
    # Fix Python float() syntax (if it's not already fixed）
    repaired = fix_python_float_syntax(repaired)
    
    # Try multiple repair strategies
    patterns_to_try = [
        # Remove truncated string values ​​(unclosed quotes）
        (r',?\s*"[^"]*"\s*:\s*"[^"]*$', ''),
        # Remove truncated key-value pairs (keys without values)）
        (r',?\s*"[^"]*"\s*:\s*$', ''),
        # Remove trailing comma
        (r',\s*$', ''),
        # Remove truncated values
        (r',?\s*\d+\.?\d*e?[+-]?\d*\s*$', ''),
        # Remove truncated array elements (including special values）
        (r',?\s*("inf"|"-inf"|"nan"|float\([^)]*\))?\s*$', ''),
        # Remove truncated sample_values ​​array (more relaxed matching）
        (r'"sample_values"\s*:\s*\[[^\]]*$', '"sample_values": []'),
        # Remove truncated input objects
        (r'"input"\s*:\s*\{[^}]*$', '"input": {"shape": [], "dtype": "float32", "sample_values": []}'),
    ]
    
    for pattern, replacement in patterns_to_try:
        test_str = re.sub(pattern, replacement, repaired)
        
        # Calculate the number of closing brackets required
        test_open_braces = test_str.count('{')
        test_close_braces = test_str.count('}')
        test_open_brackets = test_str.count('[')
        test_close_brackets = test_str.count(']')
        
        # Close all unclosed parentheses
        test_str += ']' * (test_open_brackets - test_close_brackets)
        test_str += '}' * (test_open_braces - test_close_braces)
        
        try:
            result = json.loads(test_str)
            if validate_parsed_json(result):
                return result
        except json.JSONDecodeError:
            continue
    
    # Last try: close all parentheses directly
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
    Parse special values ​​(such as "inf", "-inf", "nan" string）
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
    Create a tensor based on specifications          parameters:
        spec: Include shape, dtype, sample_values dictionary
        framework: 'torch' or 'tensorflow'
    """
    shape = spec.get("shape", [])
    dtype_str = spec.get("dtype", "float32")
    sample_values = spec.get("sample_values", [])
    
    # Preprocess sample_values, convert special values
    processed_values = [parse_special_value(v) for v in sample_values]
    
    # Calculate tensor size
    size = 1
    for dim in shape:
        size *= dim
    
    # Generate data
    if size == 0:
        data = np.array([]).reshape(shape)
    elif processed_values:
        if len(processed_values) >= size:
            data = np.array(processed_values[:size]).reshape(shape)
        else:
            # Use processed_values ​​as seed extension
            np.random.seed(42)  # Fixed seeds ensure consistency
            if processed_values:
                # Filter out NaN and Inf for calculating statistics
                finite_values = [v for v in processed_values if np.isfinite(v)]
                if finite_values:
                    mean = np.mean(finite_values)
                    std = np.std(finite_values) if len(finite_values) > 1 else 1.0
                else:
                    mean, std = 0.0, 1.0
                
                # Generate basic data
                base_data = np.random.normal(mean, std, size)
                
                # Fill in the values ​​in sample_values ​​in order
                for i, v in enumerate(processed_values):
                    if i < size:
                        base_data.flat[i] = v
                
                data = base_data.reshape(shape)
            else:
                data = np.random.randn(*shape)
    else:
        np.random.seed(42)
        data = np.random.randn(*shape)
    
    # Convert data type
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
    
    # Handle special values
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
    Create PyTorch tensors according to specifications, supporting tensors and scalars
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
    Execute PyTorch test cases, supporting multiple inputs and extra parameters
    """
    try:
        import torch
        
        api_name = test_case.get("api", "")
        
        # PyTorch dtype mapping
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
        
        # Reserved non-tensor parameter names (these are configuration parameters of the operator, not the input tensor）
        non_tensor_params = {
            "kernel_size", "stride", "padding", "dilation", "groups",
            "ceil_mode", "count_include_pad", "divisor_override",
            "alpha", "dim", "keepdim", "dtype", "out", "axis",
            "p", "eps", "momentum", "affine", "track_running_stats",
            "num_features", "normalized_shape", "weight", "bias"
        }
        
        # Separate input tensors and parameters
        args = []
        kwargs = {}
        
        # Handle main input
        if "input" in test_case:
            input_tensor = create_torch_tensor(test_case["input"], torch_dtype_map)
            args.append(input_tensor)
        
        # Handle other possible input tensors (such as other, x, y wait）
        tensor_input_names = ["other", "x", "y", "tensor", "input1", "input2", "mat1", "mat2"]
        for name in tensor_input_names:
            if name in test_case:
                tensor = create_torch_tensor(test_case[name], torch_dtype_map)
                if name == "other":
                    kwargs["other"] = tensor
                else:
                    args.append(tensor)
        
        # Handling non-tensor parameters
        for key, value in test_case.items():
            if key not in ["api", "input", "x", "y", "other", "tensor", "input1", "input2", "mat1", "mat2"]:
                if key in non_tensor_params or not isinstance(value, dict):
                    kwargs[key] = value
        
        # Get API function
        api_parts = api_name.split(".")
        func = torch
        for part in api_parts[1:]:  # jump over 'torch'
            func = getattr(func, part)
        
        # Determine whether it is a class (such as nn.AvgPool1d）Or a function (such as torch.abs）
        if isinstance(func, type):
            # Class: instantiate first and then call
            # Extract initialization parameters from kwargs
            init_params = {k: v for k, v in kwargs.items() if k in non_tensor_params}
            instance = func(**init_params)
            result = instance(args[0] if args else kwargs.get("input"))
        else:
            # Function: call directly
            result = func(*args, **kwargs)
        
        # Conversion result
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
    Create TensorFlow tensors according to specifications, supporting tensors and scalars
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
    Execute TensorFlow test cases, supporting multiple inputs and additional parameters
    """
    try:
        import tensorflow as tf
        
        api_name = test_case.get("api", "")
        
        # TensorFlow dtype mapping
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
        
        # TensorFlow The non-tensor parameter name of
        non_tensor_params = {
            "pool_size", "strides", "padding", "data_format",
            "axis", "keepdims", "name", "dtype",
            "epsilon", "center", "scale", "beta_initializer", "gamma_initializer",
            "filters", "kernel_size", "activation", "use_bias"
        }
        
        # Separate input tensors and parameters
        args = []
        kwargs = {}
        
        # Process the main input (LLM already generates the data in the correct format, no manual conversion is required）
        if "input" in test_case:
            input_tensor = create_tf_tensor(test_case["input"], tf_dtype_map)
            args.append(input_tensor)
        
        # Handle other possible input tensors
        tensor_input_names = ["other", "x", "y", "tensor", "input1", "input2"]
        for name in tensor_input_names:
            if name in test_case:
                tensor = create_tf_tensor(test_case[name], tf_dtype_map)
                args.append(tensor)
        
        # Handling non-tensor parameters (including LLM-generated data_format parameters）
        for key, value in test_case.items():
            if key not in ["api", "input", "x", "y", "other", "tensor", "input1", "input2"]:
                if key in non_tensor_params or not isinstance(value, dict):
                    kwargs[key] = value
        
        # Get API function/kind
        api_parts = api_name.split(".")
        func = tf
        for part in api_parts[1:]:  # jump over 'tf'
            func = getattr(func, part)
        
        # Determine whether it is a Keras layer (such as tf.keras.layers.AveragePooling1D）
        is_keras_layer = "keras" in api_name and "layers" in api_name
        
        if is_keras_layer:
            # Keras Layer: instantiate first and then call
            init_params = {k: v for k, v in kwargs.items() if k in non_tensor_params}
            layer = func(**init_params)
            result = layer(args[0] if args else kwargs.get("input"))
        else:
            # Ordinary function: call directly
            if args and not kwargs:
                result = func(*args)
            elif args:
                result = func(*args, **kwargs)
            else:
                result = func(**kwargs)
        
        # Conversion results (LLM has ensured that the inputs are mathematically equivalent and the outputs can be directly compared numerically）
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
    Compare the execution results of the two frameworks
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
    
    # If one party fails
    if not torch_result["success"] or not tf_result["success"]:
        if torch_result["success"] != tf_result["success"]:
            comparison["comparison_error"] = "Inconsistent execution status: one succeeds and the other fails"
        return comparison
    
    # Compare results
    try:
        torch_res = torch_result["result"]
        tf_res = tf_result["result"]
        
        if torch_res is None and tf_res is None:
            comparison["results_match"] = True
            return comparison
        
        # Handling scalar cases (including NaN) - Extended type judgment to support numpy scalar types
        is_torch_scalar = isinstance(torch_res, (int, float, np.integer, np.floating))
        is_tf_scalar = isinstance(tf_res, (int, float, np.integer, np.floating))
        
        # Handles 0-dimensional numpy arrays (scalar）
        if isinstance(torch_res, np.ndarray) and torch_res.ndim == 0:
            torch_res = torch_res.item()
            is_torch_scalar = True
        if isinstance(tf_res, np.ndarray) and tf_res.ndim == 0:
            tf_res = tf_res.item()
            is_tf_scalar = True
        
        if is_torch_scalar and is_tf_scalar:
            # Convert to Python float for comparison
            torch_val = float(torch_res)
            tf_val = float(tf_res)
            
            # Both are NaN: considered consistent
            if np.isnan(torch_val) and np.isnan(tf_val):
                comparison["results_match"] = True
                return comparison
            # Both are Inf: check symbols
            elif np.isinf(torch_val) and np.isinf(tf_val):
                if np.sign(torch_val) == np.sign(tf_val):
                    comparison["results_match"] = True
                else:
                    comparison["comparison_error"] = f"Inf Symbols are inconsistent: torch={torch_val}, tf={tf_val}"
                return comparison
            # One is NaN and the other is not: inconsistent
            elif np.isnan(torch_val) or np.isnan(tf_val):
                comparison["comparison_error"] = f"NaN inconsistent: torch={torch_val}, tf={tf_val}"
                return comparison
            # All are normal values: use allclose
            elif np.allclose(torch_val, tf_val, rtol=rtol, atol=atol):
                comparison["results_match"] = True
                return comparison
            else:
                diff = abs(torch_val - tf_val)
                comparison["comparison_error"] = f"Values ​​are inconsistent: torch={torch_val}, tf={tf_val}, diff={diff}"
                return comparison
        
        # Handling the case of empty containers（tuple vs list）
        if isinstance(torch_res, (tuple, list)) and isinstance(tf_res, (tuple, list)):
            if len(torch_res) == 0 and len(tf_res) == 0:
                comparison["results_match"] = True
                return comparison
            
            # Recursively compare each element in a container
            if len(torch_res) != len(tf_res):
                comparison["comparison_error"] = f"Container lengths are inconsistent: torch={len(torch_res)}, tf={len(tf_res)}"
                return comparison
            
            all_match = True
            for i, (t_item, tf_item) in enumerate(zip(torch_res, tf_res)):
                # If the element is a numpy array, convert directly to the appropriate result format
                if isinstance(t_item, np.ndarray) and isinstance(tf_item, np.ndarray):
                    t_result = {"success": True, "result": t_item, "shape": list(t_item.shape), "dtype": str(t_item.dtype), "error": None}
                    tf_result = {"success": True, "result": tf_item, "shape": list(tf_item.shape), "dtype": str(tf_item.dtype), "error": None}
                else:
                    t_result = {"success": True, "result": t_item, "shape": None, "dtype": None, "error": None}
                    tf_result = {"success": True, "result": tf_item, "shape": None, "dtype": None, "error": None}
                
                # Recursively call the comparison function
                item_comparison = compare_results(t_result, tf_result, rtol, atol)
                if not item_comparison["results_match"]:
                    all_match = False
                    comparison["comparison_error"] = f"Container No. {i} elements are inconsistent: {item_comparison.get('comparison_error', 'unknown reason')}"
                    break
            
            if all_match:
                comparison["results_match"] = True
            return comparison
        
        if isinstance(torch_res, np.ndarray) and isinstance(tf_res, np.ndarray):
            # Shape check
            if torch_res.shape != tf_res.shape:
                comparison["comparison_error"] = f"inconsistent shape: torch={torch_res.shape}, tf={tf_res.shape}"
                return comparison
            
            # Handle special values
            torch_nan = np.isnan(torch_res) if np.issubdtype(torch_res.dtype, np.floating) else np.zeros_like(torch_res, dtype=bool)
            tf_nan = np.isnan(tf_res) if np.issubdtype(tf_res.dtype, np.floating) else np.zeros_like(tf_res, dtype=bool)
            
            torch_inf = np.isinf(torch_res) if np.issubdtype(torch_res.dtype, np.floating) else np.zeros_like(torch_res, dtype=bool)
            tf_inf = np.isinf(tf_res) if np.issubdtype(tf_res.dtype, np.floating) else np.zeros_like(tf_res, dtype=bool)
            
            # NaN The positions must be consistent
            if not np.array_equal(torch_nan, tf_nan):
                comparison["comparison_error"] = "NaN Inconsistent location"
                return comparison
            
            # Inf Position and symbol must match
            if not np.array_equal(torch_inf, tf_inf):
                comparison["comparison_error"] = "Inf Inconsistent location"
                return comparison
            
            if np.any(torch_inf):
                if not np.array_equal(np.sign(torch_res[torch_inf]), np.sign(tf_res[tf_inf])):
                    comparison["comparison_error"] = "Inf Symbols are inconsistent"
                    return comparison
            
            # Numerical comparisons for non-special values
            mask = ~(torch_nan | torch_inf)
            if np.any(mask):
                if torch_res.size > 0:
                    if np.allclose(torch_res[mask], tf_res[mask], rtol=rtol, atol=atol, equal_nan=True):
                        comparison["results_match"] = True
                    else:
                        max_diff = np.max(np.abs(torch_res[mask] - tf_res[mask]))
                        comparison["comparison_error"] = f"Values ​​are inconsistent, maximum difference: {max_diff}"
                else:
                    comparison["results_match"] = True
            else:
                comparison["results_match"] = True
        else:
            # non-tensor results
            if torch_res == tf_res:
                comparison["results_match"] = True
            else:
                comparison["comparison_error"] = f"inconsistent results: torch={torch_res}, tf={tf_res}"
                
    except Exception as e:
        comparison["comparison_error"] = f"An error occurred during comparison: {str(e)}"
    
    return comparison



# ==================== Concurrent processing function ====================

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
    Handle single round of fuzzing (for concurrent execution)          This function will be called by multiple threads at the same time, and each thread will process a round of fuzzing.          parameter:
        client: LLM Client (thread safe）
        original_case: Original test case
        torch_doc: PyTorch document
        tf_doc: TensorFlow document
        round_num: current round（1-3）
        model: LLM Model name
        print_lock: Print lock (thread-safe output)          Return:
        fuzzing result dictionary
    """
    torch_case = original_case.get("torch_test_case", {})
    tf_case = original_case.get("tensorflow_test_case", {})
    torch_api = torch_case.get("api", "")
    tf_api = tf_case.get("api", "")
    
    with print_lock:
        print(f"    [Round {round_num}/{FUZZING_ROUNDS}] Generate variant use cases...")
    
    # Build prompt words
    prompt = build_fuzzing_prompt(
        torch_api, tf_api, original_case, torch_doc, tf_doc, round_num
    )
    
    try:
        # Call LLM (with retry mechanism）
        max_retries = 2
        mutated_case = None
        llm_response = ""
        
        for retry in range(max_retries):
            try:
                # Round 1 Use a higher token limit because responses to extreme value variations are often longer
                max_tokens = 6144 if round_num == 1 else 4096
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7 + retry * 0.1,
                    max_tokens=max_tokens,
                )
                llm_response = response.choices[0].message.content.strip()
                
                # Parse response
                mutated_case = parse_llm_response(llm_response)
                
                if mutated_case is not None:
                    if "torch_test_case" in mutated_case and "tensorflow_test_case" in mutated_case:
                        break
                    else:
                        with print_lock:
                            print(f"[WARN] Round {round_num} Response is missing required fields, try again ({retry + 1}/{max_retries})")
                            print(f"       Parsed field: {list(mutated_case.keys())}")
                        mutated_case = None
                else:
                    if retry < max_retries - 1:
                        with print_lock:
                            print(f"[WARN] Round {round_num} Parsing failed, try again ({retry + 1}/{max_retries})")
                            # Display the first 200 characters of the response for debugging
                            preview = llm_response[:200].replace('\n', ' ')
                            print(f"       Response preview: {preview}...")
                            # Check if Python float() syntax is included
                            if "float(" in llm_response:
                                print(f"       Python float() syntax detected, will try to fix")
            except Exception as e:
                with print_lock:
                    print(f"[WARN] Round {round_num} LLM Call exception: {e}，Try again ({retry + 1}/{max_retries})")
                if retry < max_retries - 1:
                    time.sleep(1)
        
        if mutated_case is None:
            return {
                "round": round_num,
                "success": False,
                "error": "LLM Response parsing failed",
                "llm_response": llm_response[:1000]
            }
        
        with print_lock:
            print(f"    [Round {round_num}] Perform differential testing...")
        
        # Execute tests
        torch_test = mutated_case.get("torch_test_case", {})
        tf_test = mutated_case.get("tensorflow_test_case", {})
        
        torch_result = execute_torch_test(torch_test)
        tf_result = execute_tensorflow_test(tf_test)
        
        # Compare results
        comparison = compare_results(torch_result, tf_result)
        
        # Record results
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
        
        # Print result summary
        with print_lock:
            if fuzzing_result["is_bug_candidate"]:
                print(f"    [Round {round_num}] ⚠️ Discover potential problems: {comparison.get('comparison_error') or 'Inconsistent execution status'}")
            else:
                print(f"    [Round {round_num}] ✓ The results are consistent")
        
        return fuzzing_result
        
    except Exception as e:
        with print_lock:
            print(f"    [Round {round_num}] ✗ mistake: {e}")
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
    Multiple rounds of fuzzing (concurrent versions) on a single test case          Use thread pool to execute 3 rounds of fuzzing concurrently to improve efficiency。
    """
    fuzzing_results = []
    
    # Execute 3 rounds concurrently using thread pool fuzzing
    with ThreadPoolExecutor(max_workers=min(workers, FUZZING_ROUNDS)) as executor:
        # Submit all fuzzing round tasks
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
        
        # Collect results (in order of completion）
        for future in as_completed(future_to_round):
            try:
                result = future.result()
                fuzzing_results.append(result)
            except Exception as e:
                round_num = future_to_round[future]
                with print_lock:
                    print(f"[ERROR] Round {round_num} Execution exception: {e}")
                fuzzing_results.append({
                    "round": round_num,
                    "success": False,
                    "error": f"Execution exception: {str(e)}"
                })
    
    # Sort results by round
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
    Process a single test case (for concurrent execution)          parameters:
        client: LLM client
        case: test case
        case_idx: Use case index (starts at 1）
        total_cases: Total number of use cases
        torch_doc: PyTorch document
        tf_doc: TensorFlow document
        model: LLM Model name
        print_lock: print lock
        workers: Number of concurrent threads          Return:
        Use case processing results
    """
    with print_lock:
        print(f"\n  use case {case_idx}/{total_cases} (iteration={case.get('iteration')}, case={case.get('case_number')})")
    
    # fuzzing the use case (also concurrent internally)）
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
    Handle all successful cases of a single operator (concurrent version)          Use a thread pool to process multiple test cases concurrently, and the 3 rounds of fuzzing inside each test case are also concurrent.。
    """
    with open(operator_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    operator_name = data.get("operator", "unknown")
    success_cases = data.get("success_cases", [])
    
    if max_cases:
        success_cases = success_cases[:max_cases]
    
    with print_lock:
        print(f"\nprocessing operator: {operator_name} ({len(success_cases)} use cases)")
    
    # Get API name
    if success_cases:
        torch_api = success_cases[0].get("torch_test_case", {}).get("api", "")
        tf_api = success_cases[0].get("tensorflow_test_case", {}).get("api", "")
    else:
        return {"operator": operator_name, "results": [], "error": "No successful use case"}
    
    # Crawl documents
    with print_lock:
        print(f"  get {torch_api} document...")
    torch_doc = get_doc_content(torch_api, "pytorch")
    with print_lock:
        print(f"  get {tf_api} document...")
    tf_doc = get_doc_content(tf_api, "tensorflow")
    
    # Handle all use cases concurrently
    all_results = []
    bug_candidates = 0
    
    # Using thread pools to handle use cases concurrently
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {}
            
            # Submit all use case processing tasks
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
            
            # Collect results
            results_dict = {}
            for future in as_completed(future_to_idx):
                try:
                    idx = future_to_idx[future]
                    case_result = future.result(timeout=300)  # 5minutes timeout
                    results_dict[idx] = case_result
                    
                    # statistical potential bug
                    for fr in case_result.get("fuzzing_results", []):
                        if fr.get("is_bug_candidate"):
                            bug_candidates += 1
                except TimeoutError:
                    idx = future_to_idx[future]
                    with print_lock:
                        print(f"[ERROR] then trial case {idx} time out")
                    results_dict[idx] = {
                        "error": "Processing timeout",
                        "fuzzing_results": []
                    }
                except Exception as e:
                    idx = future_to_idx[future]
                    with print_lock:
                        print(f"[ERROR] Handle use cases {idx} error: {e}")
                        traceback.print_exc()
                    results_dict[idx] = {
                        "error": f"Processing failed: {str(e)}",
                        "fuzzing_results": []
                    }
    except Exception as e:
        with print_lock:
            print(f"[ERROR] Thread pool execution exception: {e}")
            traceback.print_exc()
        results_dict = {}
    
    # Arrange results in index order
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
    Save the fuzzing results of the operator, the file name contains the timestamp
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    operator_name = result.get("operator", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{operator_name}_fuzzing_result_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"  Results saved: {output_file}")


def main():
    """
    Main program entry (concurrent version）
    """
    import signal
    import sys
    
    # Set up signal processing to prevent silent exit
    def signal_handler(signum, frame):
        print(f"\n[WARN] signal received {signum}，The program is about to exit...")
        sys.exit(1)
    
    # Register signal handler
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except (AttributeError, ValueError):
        pass  # Windows Some signals may not be supported
    
    parser = argparse.ArgumentParser(
        description="PyTorch-TensorFlow Fuzzing differential testing based on LLM (concurrent version）"
    )
    parser.add_argument(
        "--operators", "-o",
        nargs="*",
        help="Specify the name of the operator to be tested (if not specified, all will be tested)）"
    )
    parser.add_argument(
        "--max-cases", "-m",
        type=int,
        default=None,
        help="The maximum number of test cases for each operator (default all）"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model name (default {DEFAULT_MODEL}）"
    )
    parser.add_argument(
        "--key-path", "-k",
        default=DEFAULT_KEY_PATH,
        help=f"API key File path (default {DEFAULT_KEY_PATH}）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of operators processed (for testing）"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting operator index (starts from 1, defaults to1）"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End operator index (inclusive, default to the last）"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of concurrent threads (default {DEFAULT_WORKERS}）"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PyTorch-TensorFlow Fuzzing differential testing based on LLM (concurrent version）")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Number of fuzzing rounds per use case: {FUZZING_ROUNDS}")
    print(f"Number of concurrent threads: {args.workers}")
    
    # Initialize LLM client
    try:
        client = get_qwen_client(args.key_path)
        print("LLM Client initialization successful")
    except Exception as e:
        print(f"[ERROR] Unable to initialize LLM client: {e}")
        return
    
    # Get the operator file to be processed
    if args.operators:
        operator_files = []
        for op in args.operators:
            op_file = SUCCESS_CASES_DIR / f"{op}_success_cases.json"
            if op_file.exists():
                operator_files.append(op_file)
            else:
                print(f"[WARN] Operator file not found: {op_file}")
    else:
        operator_files = sorted(SUCCESS_CASES_DIR.glob("*_success_cases.json"))
    
    if args.limit:
        operator_files = operator_files[:args.limit]
    
    # Apply scope filtering (--start and --end parameters）
    total_available = len(operator_files)
    if args.start > total_available:
        print(f"[ERROR] starting index {args.start} out of range (shared {total_available} an operator）")
        return
    
    start_idx = args.start - 1  # Convert to 0 index
    end_idx = args.end if args.end is not None else total_available  # Default to end
    
    if end_idx > total_available:
        print(f"[WARN] end index {end_idx} Out of range, adjust to {total_available}")
        end_idx = total_available
    
    if start_idx >= end_idx:
        print(f"[ERROR] starting index {args.start} Must be less than end index {end_idx}")
        return
    
    operator_files = operator_files[start_idx:end_idx]
    
    if args.start > 1 or args.end is not None:
        range_info = f"No. {args.start}-{end_idx} operators (total {total_available} available）"
        print(f"Test range: {range_info}")
    
    print(f"Number of operators to be processed: {len(operator_files)}")
    print("=" * 70)
    
    # statistics
    total_operators = len(operator_files)
    total_bug_candidates = 0
    
    # Create a print lock
    print_lock = Lock()
    
    # Recording start time
    start_time = time.time()
    
    # Process each operator (operators are processed sequentially, but use cases and fuzzing rounds within each operator are concurrent）
    for idx, op_file in enumerate(operator_files, 1):
        print(f"\n[{idx}/{total_operators}] deal with: {op_file.stem}")
        
        try:
            result = process_operator(
                op_file, client, args.model, args.max_cases, print_lock, args.workers
            )
            
            # Save results
            save_operator_result(result, RESULT_DIR)
            
            total_bug_candidates += result.get("bug_candidates", 0)
            
            # show progress
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = total_operators - idx
            eta = avg_time * remaining
            print(f"\n[PROGRESS] Completed {idx}/{total_operators} an operator，"
                  f"time consuming {elapsed:.1f}s，Estimated remaining {eta:.1f}s")
            
        except KeyboardInterrupt:
            print(f"\n[INFO] User interrupted, saving completed results...")
            break
        except MemoryError:
            print(f"[ERROR] Insufficient memory, skip {op_file.stem}")
            # Try to free memory
            import gc
            gc.collect()
            continue
        except Exception as e:
            print(f"[ERROR] deal with {op_file.stem} error: {e}")
            traceback.print_exc()
            # Continue processing the next operator instead of exiting
            continue
    
    # Print summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Fuzzing Test completed！")
    print("=" * 70)
    print(f"Number of processing operators: {total_operators}")
    print(f"Number of potential problems found: {total_bug_candidates}")
    print(f"Total time spent: {total_time:.1f}s")
    print(f"Result saving directory: {RESULT_DIR}")


if __name__ == "__main__":
    main()
