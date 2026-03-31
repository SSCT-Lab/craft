"""  
PyTorch-TensorFlow Fuzzing differential testing tool based on LLM  Function description:
    1. Read successful test cases classified by operator
    2. Crawl PyTorch and TensorFlow official documents
    3. Use LLM mutation test cases (complex inputs, extreme values, boundary values）
    4. Perform differential testing to detect framework inconsistencies or potential bug
    5. 3 rounds per use case fuzzing
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

# Successful use case catalog
SUCCESS_CASES_DIR = Path(__file__).parent / "success_cases"
# Fuzzing Results directory
RESULT_DIR = Path(__file__).parent / "result"


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
    
    # Mutation strategies for different rounds
    mutation_strategies = {
        1: "Extreme numerical variation: using complex floating point numbers, maximum value (1e38), minimum value (1e-38), infinity (inf), negative infinity (-inf), NaN, zero (0), negative zero(-0.0)and other special values",
        2: "Boundary shape mutation: use empty tensor (shape contains 0), scalar(shape=[])、Boundary situations such as ultra-high-dimensional tensors (more than 5 dimensions), irregular shapes, single-element tensors, etc.",
        3: "Complex type mutation: testing different data types(float16/float32/float64/int32/int64/bool/complex64/complex128)、mixed precision scene"
    }
    
    current_strategy = mutation_strategies.get(round_num, mutation_strategies[1])
    
    prompt = f"""You are a professional deep learning framework testing expert, and now you need to perform differential testing on the equivalent operators of PyTorch and TensorFlow.  【Test goal】 What we are looking for is**potential in framework implementation bug**，That is: mathematically equivalent operations, but the two frameworks produce different outputs。
- ✓ What to look for: Framework implementation differences (real bug）
- ✗ What not to look for: Differences (false positives) caused by inequivalence of the test cases themselves      [Currently tested operator pair】
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

【Core mutation requirements: Mathematical equivalence (most important！）】
**The two test cases after mutation must be completely equivalent mathematically, and the theoretical output must be the same。**

You are free to: - Modify input data (shape, dtype, value) - Add or modify parameters - Use any legal parameter combination  But you must make sure: - If PyTorch uses a parameter (such as `alpha=2.0`），TensorFlow must be treated equivalently - e.g.：`torch.add(x, y, alpha=2.0)` Equivalent to `tf.add(x, tf.multiply(y, 2.0))`
- If equivalent logic cannot be implemented on the TensorFlow side, then**Do not use this parameter**

【Specific variation requirements】
1. **Inputs must be exactly the same**：The shape, dtype, and value of the input tensors of the two frameworks must be completely consistent (format conversion can be performed if necessary, such as NCHW toNHWC）
2. **Parameter semantic equivalence**：If the API has additional parameters, the parameter values ​​on both sides must be semantically equivalent
3. **Explore edge cases**：Focus on testing extreme values, boundary shapes, special dtypes, and other boundary scenarios that may cause inconsistent behavior between the two frameworks
4. **Keep it enforceable**：The mutated use case must be legal and can be executed correctly by the framework  【Important Tips】 - Confirm the parameter mapping relationship between the two APIs according to the documentation - Pay attention to data type compatibility (for example, some dtypes may only be supported by one framework) - Consider numerical stability issues (such as division by zero, log(0), sqrt(-1), etc.) - Pay attention to the operator's requirements for dimensions when mutating shape  [Principles of parameter processing] - You can freely add, modify, and delete parameters as long as they are mathematically equivalent - If a parameter is supported by only one framework, either do not use it or manually implement equivalent logic in another framework. - Parameter names are correctly mapped according to the framework documentation (e.g. kernel_size <-> pool_size）

【Bad example - don't do this！】
❌ PyTorch use `alpha=2.0`，TensorFlow but no corresponding treatment：
   torch.add(x, y, alpha=2.0)  →  x + 2*y
   tf.add(x, y)                →  x + y
   These two are not mathematically equivalent and will produce false positives!  [Correct example】
✓ Option 1: Don’t use alpha on both sides (recommended, simpler）
   torch.add(x, y)  →  x + y
   tf.add(x, y)     →  x + y

✓ Option 2: Manually implement equivalent logic on TensorFlow side
   torch.add(x, y, alpha=2.0)       →  x + 2*y
   tf.add(x, tf.multiply(y, 2.0))   →  x + 2*y

【Output format requirements] Please strictly follow the following JSON format to output the mutated test cases, and do not output any other content.：
```json
{{
  "mutation_strategy": "Briefly describe this mutation strategy",
  "mutation_reason": "Explain in detail why this mutation may lead to problems",
  "torch_test_case": {{
    "api": "{torch_api}",
    "input": {{
      "shape": [...],
      "dtype": "...",
      "sample_values": [...]
    }},
    // If additional input is required (e.g. torch.add of other), add corresponding fields
    // like："other": {{ "shape": [...], "dtype": "...", "sample_values": [...] }} or scalar value
    // If additional parameters are required (such as AvgPool1d's kernel_size, stride），Also add corresponding fields
    // like："kernel_size": 3, "stride": 2, "padding": 0 wait
  }},
  "tensorflow_test_case": {{
    "api": "{tf_api}",
    "input": {{
      "shape": [...],
      "dtype": "...",
      "sample_values": [...]
    }},
    // TensorFlow Corresponding parameters, note that the parameter names may be different (but must be mathematically equivalent to the PyTorch side）
    // For example, pool_size corresponds to kernel_size, and strides corresponds to stride
  }}
}}
```

【Important】About the processing of multiple inputs and parameters：
1. sample_values Should contain enough values ​​to fill the entire tensor, if the tensor is large you can provide only the first few values ​​as a seed
2. **If the operator has multiple inputs**（like torch.add of input and other), must both be included in the output
3. **Parameter name mapping**：PyTorch The parameter names, parameter value types and quantities of TensorFlow may be different and need to be mapped correctly according to the documentation.：
   - torch.nn.AvgPool1d(kernel_size, stride) <-> tf.keras.layers.AveragePooling1D(pool_size, strides)
   - torch.add(input, other) <-> tf.add(x, y)
4. dtype Must be a type supported by both frameworks
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
            # If the terminator is not found (truncated), take all remaining content
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
            # Try to parse directly
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
    Try to fix incomplete JSON string
    """
    import re
    
    # Method 1: Check and complete missing brackets
    # Count the number of brackets
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    
    # Try to complete missing brackets
    repaired = json_str
    
    # Remove possible incomplete strings at the end (unclosed quote content）
    # Find the last complete key-value pair
    patterns_to_try = [
        # Case 1: Truncated in the middle of string value "key": "value...
        (r',?\s*"[^"]*"\s*:\s*"[^"]*$', ''),
        # Case 2: Truncated after key name "key":
        (r',?\s*"[^"]*"\s*:\s*$', ''),
        # Case 3: Truncated after comma
        (r',\s*$', ''),
        # Case 4: Truncated in the middle of the array
        (r',?\s*\d+\.?\d*\s*$', ''),
    ]
    
    for pattern, replacement in patterns_to_try:
        test_str = re.sub(pattern, replacement, repaired)
        # Completion brackets
        test_open_braces = test_str.count('{')
        test_close_braces = test_str.count('}')
        test_open_brackets = test_str.count('[')
        test_close_brackets = test_str.count(']')
        
        test_str += ']' * (test_open_brackets - test_close_brackets)
        test_str += '}' * (test_open_braces - test_close_braces)
        
        try:
            result = json.loads(test_str)
            # Verify required fields exist
            if "torch_test_case" in result or "mutation_strategy" in result:
                print(f"[INFO] JSON Repair successful")
                return result
        except json.JSONDecodeError:
            continue
    
    # Method 2: Simply complete the brackets
    repaired += ']' * (open_brackets - close_brackets)
    repaired += '}' * (open_braces - close_braces)
    
    try:
        result = json.loads(repaired)
        if "torch_test_case" in result or "mutation_strategy" in result:
            print(f"[INFO] JSON Simple repair successful")
            return result
    except json.JSONDecodeError:
        pass
    
    # Method 3: Try to extract the existing complete part
    # Find the last complete } or content before ]
    for i in range(len(json_str) - 1, -1, -1):
        if json_str[i] in '}]':
            try:
                test_str = json_str[:i+1]
                # Completion brackets
                test_open_braces = test_str.count('{')
                test_close_braces = test_str.count('}')
                test_open_brackets = test_str.count('[')
                test_close_brackets = test_str.count(']')
                
                test_str += ']' * (test_open_brackets - test_close_brackets)
                test_str += '}' * (test_open_braces - test_close_braces)
                
                result = json.loads(test_str)
                if "torch_test_case" in result or "mutation_strategy" in result:
                    print(f"[INFO] JSON Partial extraction successful")
                    return result
            except json.JSONDecodeError:
                continue
    
    print(f"[WARN] JSON Repair failed")
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
        
        # Handle main input
        if "input" in test_case:
            input_tensor = create_tf_tensor(test_case["input"], tf_dtype_map)
            args.append(input_tensor)
        
        # Handle other possible input tensors
        tensor_input_names = ["other", "x", "y", "tensor", "input1", "input2"]
        for name in tensor_input_names:
            if name in test_case:
                tensor = create_tf_tensor(test_case[name], tf_dtype_map)
                args.append(tensor)
        
        # Handling non-tensor parameters
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
        
        # Conversion result
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


def run_fuzzing_for_case(
    client,
    original_case: Dict[str, Any],
    torch_doc: str,
    tf_doc: str,
    model: str = DEFAULT_MODEL
) -> List[Dict[str, Any]]:
    """
    Perform multiple rounds on a single test case fuzzing
    """
    torch_case = original_case.get("torch_test_case", {})
    tf_case = original_case.get("tensorflow_test_case", {})
    torch_api = torch_case.get("api", "")
    tf_api = tf_case.get("api", "")
    
    fuzzing_results = []
    
    for round_num in range(1, FUZZING_ROUNDS + 1):
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
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7 + retry * 0.1,  # Add slightly more randomness when retrying
                        max_tokens=4096,  # Increase token limit to avoid truncation
                    )
                    llm_response = response.choices[0].message.content.strip()
                    
                    # Parse response
                    mutated_case = parse_llm_response(llm_response)
                    
                    if mutated_case is not None:
                        # Validate required fields
                        if "torch_test_case" in mutated_case and "tensorflow_test_case" in mutated_case:
                            break
                        else:
                            print(f"[WARN] Round {round_num} Response is missing required fields, try again ({retry + 1}/{max_retries})")
                            mutated_case = None
                    else:
                        if retry < max_retries - 1:
                            print(f"[WARN] Round {round_num} Parsing failed, try again ({retry + 1}/{max_retries})")
                except Exception as e:
                    print(f"[WARN] Round {round_num} LLM Call exception: {e}，Try again ({retry + 1}/{max_retries})")
                    if retry < max_retries - 1:
                        import time
                        time.sleep(1)  # Wait 1 second and try again
            
            if mutated_case is None:
                fuzzing_results.append({
                    "round": round_num,
                    "success": False,
                    "error": "LLM Response parsing failed",
                    "llm_response": llm_response[:1000]  # Keep more responses for debugging
                })
                continue
            
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
            fuzzing_results.append(fuzzing_result)
            
            # Print result summary
            if fuzzing_result["is_bug_candidate"]:
                print(f"    [Round {round_num}] ⚠️ Discover potential problems: {comparison.get('comparison_error') or 'Inconsistent execution status'}")
            else:
                print(f"    [Round {round_num}] ✓ The results are consistent")
                
        except Exception as e:
            fuzzing_results.append({
                "round": round_num,
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "traceback": traceback.format_exc()
            })
            print(f"    [Round {round_num}] ✗ mistake: {e}")
    
    return fuzzing_results


def process_operator(
    operator_file: Path,
    client,
    model: str,
    max_cases: Optional[int] = None
) -> Dict[str, Any]:
    """
    Handle all successful use cases of a single operator
    """
    with open(operator_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    operator_name = data.get("operator", "unknown")
    success_cases = data.get("success_cases", [])
    
    if max_cases:
        success_cases = success_cases[:max_cases]
    
    print(f"\nprocessing operator: {operator_name} ({len(success_cases)} use cases)")
    
    # Get API name
    if success_cases:
        torch_api = success_cases[0].get("torch_test_case", {}).get("api", "")
        tf_api = success_cases[0].get("tensorflow_test_case", {}).get("api", "")
    else:
        return {"operator": operator_name, "results": [], "error": "No successful use case"}
    
    # Crawl documents
    print(f"  get {torch_api} document...")
    torch_doc = get_doc_content(torch_api, "pytorch")
    print(f"  get {tf_api} document...")
    tf_doc = get_doc_content(tf_api, "tensorflow")
    
    # for each use case fuzzing
    all_results = []
    bug_candidates = 0
    
    for idx, case in enumerate(success_cases, 1):
        print(f"\n  use case {idx}/{len(success_cases)} (iteration={case.get('iteration')}, case={case.get('case_number')})")
        
        fuzzing_results = run_fuzzing_for_case(
            client, case, torch_doc, tf_doc, model
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
        all_results.append(case_result)
        
        # statistical potential bug
        for fr in fuzzing_results:
            if fr.get("is_bug_candidate"):
                bug_candidates += 1
    
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
    # Add timestamp suffix YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{operator_name}_fuzzing_result_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"  Results saved: {output_file}")


def main():
    """
    Main program entrance
    """
    parser = argparse.ArgumentParser(
        description="PyTorch-TensorFlow Fuzzing differential testing based on LLM"
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
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PyTorch-TensorFlow Fuzzing differential testing based on LLM")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Number of fuzzing rounds per use case: {FUZZING_ROUNDS}")
    
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
    
    print(f"Number of operators to be processed: {len(operator_files)}")
    print("=" * 70)
    
    # statistics
    total_operators = len(operator_files)
    total_bug_candidates = 0
    
    # Process each operator
    for idx, op_file in enumerate(operator_files, 1):
        print(f"\n[{idx}/{total_operators}] deal with: {op_file.stem}")
        
        try:
            result = process_operator(
                op_file, client, args.model, args.max_cases
            )
            
            # Save results
            save_operator_result(result, RESULT_DIR)
            
            total_bug_candidates += result.get("bug_candidates", 0)
            
        except Exception as e:
            print(f"[ERROR] deal with {op_file.stem} error: {e}")
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 70)
    print("Fuzzing Test completed！")
    print("=" * 70)
    print(f"Number of processing operators: {total_operators}")
    print(f"Number of potential problems found: {total_bug_candidates}")
    print(f"Result saving directory: {RESULT_DIR}")


if __name__ == "__main__":
    main()
