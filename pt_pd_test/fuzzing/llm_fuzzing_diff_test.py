"""  
PyTorch-PaddlePaddle LLM-based fuzzing differential test tool.

Features:
    1. Read success cases grouped by operator
    2. Fetch official docs for PyTorch and PaddlePaddle
    3. Use LLM to mutate test cases (complex inputs, extremes, boundary values)
    4. Run differential tests to detect framework inconsistencies or potential bugs
    5. Run 3 fuzzing rounds per case
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

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_crawler_factory import get_doc_content
from component.migration.migrate_generate_tests import get_qwen_client

# ==================== Constants ====================
DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"
FUZZING_ROUNDS = 3  # Fuzzing rounds per case

# Success case directory
SUCCESS_CASES_DIR = Path(__file__).parent / "success_cases"
# Fuzzing result directory
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
    Build LLM fuzzing prompt.
    
    Args:
        torch_api: PyTorch API name
        paddle_api: PaddlePaddle API name
        original_case: Original test case
        torch_doc: PyTorch doc content
        paddle_doc: PaddlePaddle doc content
        round_num: Current fuzzing round
    """
    torch_case = original_case.get("torch_test_case", {})
    paddle_case = original_case.get("paddle_test_case", {})
    
    torch_case_json = json.dumps(torch_case, ensure_ascii=False, indent=2)
    paddle_case_json = json.dumps(paddle_case, ensure_ascii=False, indent=2)
    
    # Mutation strategies by round
    mutation_strategies = {
        1: "Extreme value mutation: use complex floats, huge values (1e38), tiny values (1e-38), inf, -inf, NaN, zero (0), negative zero (-0.0)",
        2: "Boundary shape mutation: empty tensors (shape contains 0), scalars (shape=[]), high-rank tensors (>=5D), irregular shapes, single-element tensors",
        3: "Type mutation: test dtypes (float32/float64/int32/int64/bool). Note: PaddlePaddle CPU does not support float16; avoid it."
    }
    
    current_strategy = mutation_strategies.get(round_num, mutation_strategies[1])
    
        prompt = f"""You are a deep learning framework testing expert. We need to run differential tests on equivalent PyTorch and PaddlePaddle operators.

[Test Goals]
We want to find **potential bugs in framework implementations**: mathematically equivalent operations that produce different outputs.
- ✓ Look for: implementation differences under identical math (real bugs)
- ✗ Do not look for: differences caused by non-equivalent test cases (false positives)
- ✗ Do not look for: differences due to unsupported dtypes (e.g., PaddlePaddle CPU does not support float16)
    
[Current Operator Pair]
- PyTorch API: {torch_api}
- PaddlePaddle API: {paddle_api}

[Original Success Cases]
PyTorch test case:
```json
{torch_case_json}
```

PaddlePaddle test case:
```json
{paddle_case_json}
```

[PyTorch Official Docs]
{torch_doc if torch_doc else "Failed to fetch docs"}

[PaddlePaddle Official Docs]
{paddle_doc if paddle_doc else "Failed to fetch docs"}

[Mutation Strategy - Round {round_num}]
{current_strategy}

[Core Requirement: Mathematical Equivalence (Most Important!)]
**The mutated test cases must be mathematically equivalent, with identical theoretical outputs.**

You may:
- Modify inputs (shape, dtype, values)
- Add or modify parameters
- Use any valid parameter combinations

But you must ensure:
- If PyTorch uses a parameter (e.g., `alpha=2.0`), PaddlePaddle must apply an equivalent transformation
- Example: `torch.add(x, y, alpha=2.0)` is equivalent to `paddle.add(x, paddle.scale(y, scale=2.0))`
- If equivalent logic cannot be implemented in PaddlePaddle, **do not use that parameter**

[Specific Mutation Requirements]
1. **Inputs must match exactly**: shapes, dtypes, and values must be identical across frameworks
2. **Parameter semantics must match**: if the API has extra params, values must be semantically equivalent
3. **Explore edge cases**: focus on extreme values, boundary shapes, special dtypes likely to cause differences
4. **Keep executability**: mutated cases must be valid and executable

[Important Notes]
- Confirm parameter mapping between APIs from docs
- Check dtype compatibility (some dtypes may only be supported by one framework)
- Consider numerical stability issues (divide by zero, log(0), sqrt(-1), etc.)
- When mutating shapes, respect operator dimension requirements
- PaddlePaddle data format usually matches PyTorch (NCHW)

[Known Framework Limitations - Must Avoid]
- **PaddlePaddle CPU does not support float16**: do not use float16 dtype ("kernel not registered" error)
- **PaddlePaddle y must be Tensor**: for paddle.add(x, y), y cannot be a Python scalar
- Recommended dtypes: float32, float64, int32, int64, bool

[Parameter Handling Principles]
- You may add/modify/remove params as long as mathematical equivalence is preserved
- If a param is only supported by one framework, either omit it or implement an equivalent logic manually
- Map parameter names per framework docs (names may differ)

[Incorrect Example - Do NOT do this]
❌ PyTorch uses a param that PaddlePaddle does not handle (e.g., value in torch.addcmul):
     torch.addcmul(input, tensor1, tensor2, value=2.0)  →  input + 2.0 * tensor1 * tensor2
     paddle.addcmul(input, tensor1, tensor2)            →  input + 1.0 * tensor1 * tensor2
     These are not mathematically equivalent and produce false positives.

[Correct Examples]
✓ Option 1: use the same param values on both sides
     torch.addcmul(input, tensor1, tensor2, value=2.0)   →  input + 2*tensor1*tensor2
     paddle.addcmul(input, tensor1, tensor2, value=2.0)  →  input + 2*tensor1*tensor2

✓ Option 2: omit optional params (use defaults)
     torch.addcmul(input, tensor1, tensor2)   →  input + tensor1*tensor2
     paddle.addcmul(input, tensor1, tensor2)  →  input + tensor1*tensor2

[Output Format]
**Important: Keep output concise!** mutation_strategy <= 50 chars, mutation_reason <= 150 chars.
Return strictly the following JSON only (no extra content):
```json
{{
    "mutation_strategy": "Briefly describe the mutation strategy",
    "mutation_reason": "Explain why this mutation may reveal issues",
    "torch_test_case": {{
        "api": "{torch_api}",
        "input": {{
            "shape": [...],
            "dtype": "...",
            "sample_values": [...]
        }},
        // If extra inputs are needed (e.g., torch.add other), add fields
        // e.g., "other": {{ "shape": [...], "dtype": "...", "sample_values": [...] }} or scalar value
        // If extra params are needed (e.g., AvgPool1d kernel_size/stride), add fields
        // e.g., "kernel_size": 3, "stride": 2, "padding": 0
    }},
    "paddle_test_case": {{
        "api": "{paddle_api}",
        "input": {{
            "shape": [...],
            "dtype": "...",
            "sample_values": [...]
        }},
        // PaddlePaddle params, note name differences (must be math-equivalent)
        // e.g., pool_size corresponds to kernel_size, strides to stride
    }}
}}
```

[Important] Multi-input and parameter handling:
1. sample_values should be enough to fill the tensor; for large tensors, provide a few values as seeds
2. **If the operator has multiple inputs** (e.g., torch.add input and other), include all of them
3. **Parameter mapping**: PyTorch and PaddlePaddle may differ in param names/types/counts; map per docs:
     - torch.nn.AvgPool1d(kernel_size, stride) <-> paddle.nn.AvgPool1D(kernel_size, stride)
     - torch.add(input, other) <-> paddle.add(x, y)
4. dtype must be supported by both frameworks
"""
    return prompt


def preprocess_json_string(json_str: str) -> str:
    """
    Preprocess JSON string: convert Python-style special values to JSON-friendly format.
    
    LLM may output Python syntax in JSON (e.g., float('inf')).
    This is not valid JSON, so we convert to string literals.
    
    Note: only replace special values used as JSON values, not within text strings.
    """
    import re
    
    # Replace Python float('...') syntax used as values (not inside strings)
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
    
    # For bare inf/nan, only replace in array contexts (adjacent to [ ] or ,)
    # to avoid replacing text inside strings
    # Pattern: inf/nan near [ , ] boundaries
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
    Parse JSON response from LLM; handle truncation and format errors.
    """
    try:
        # Try extracting JSON block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            # If missing closing fence (truncated), take the rest
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
            # Try direct parse
            json_str = response.strip()
        
        # Preprocess Python-style special values
        json_str = preprocess_json_string(json_str)
        
        # Try direct parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Try repairing truncated JSON
        repaired_json = try_repair_json(json_str)
        if repaired_json:
            return repaired_json
        
        return None
    except Exception as e:
        print(f"[WARN] Failed to parse LLM response: {e}")
        return None


def try_repair_json(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Try to repair incomplete JSON strings.
    """
    import re
    
    # Method 1: check and补全 missing brackets
    # Count brackets
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    
    # Try to complete missing brackets
    repaired = json_str
    
    # Remove trailing incomplete string (unterminated quote)
    # Find last complete key-value pair
    patterns_to_try = [
        # Case 1: truncation in string value "key": "value...
        (r',?\s*"[^"]*"\s*:\s*"[^"]*$', ''),
        # Case 2: truncation after key "key":
        (r',?\s*"[^"]*"\s*:\s*$', ''),
        # Case 3: truncation after comma
        (r',\s*$', ''),
        # Case 4: truncation in array
        (r',?\s*\d+\.?\d*\s*$', ''),
    ]
    
    for pattern, replacement in patterns_to_try:
        test_str = re.sub(pattern, replacement, repaired)
        # Close brackets
        test_open_braces = test_str.count('{')
        test_close_braces = test_str.count('}')
        test_open_brackets = test_str.count('[')
        test_close_brackets = test_str.count(']')
        
        test_str += ']' * (test_open_brackets - test_close_brackets)
        test_str += '}' * (test_open_braces - test_close_braces)
        
        try:
            result = json.loads(test_str)
            # Validate required fields
            if "torch_test_case" in result or "mutation_strategy" in result:
                print(f"[INFO] JSON repair succeeded")
                return result
        except json.JSONDecodeError:
            continue
    
    # Method 2: simple bracket completion
    repaired += ']' * (open_brackets - close_brackets)
    repaired += '}' * (open_braces - close_braces)
    
    try:
        result = json.loads(repaired)
        if "torch_test_case" in result or "mutation_strategy" in result:
            print(f"[INFO] JSON simple repair succeeded")
            return result
    except json.JSONDecodeError:
        pass
    
    # Method 3: try extracting last complete part
    # Find content up to last complete } or ]
    for i in range(len(json_str) - 1, -1, -1):
        if json_str[i] in '}]':
            try:
                test_str = json_str[:i+1]
                # Close brackets
                test_open_braces = test_str.count('{')
                test_close_braces = test_str.count('}')
                test_open_brackets = test_str.count('[')
                test_close_brackets = test_str.count(']')
                
                test_str += ']' * (test_open_brackets - test_close_brackets)
                test_str += '}' * (test_open_braces - test_close_braces)
                
                result = json.loads(test_str)
                if "torch_test_case" in result or "mutation_strategy" in result:
                    print(f"[INFO] JSON partial extraction succeeded")
                    return result
            except json.JSONDecodeError:
                continue
    
    print(f"[WARN] JSON repair failed")
    return None


def parse_special_value(val: Any) -> float:
    """
    Parse special values (e.g., "inf", "-inf", "nan", "Infinity", "-Infinity", "NaN").
    Supports JSON standard and Python-style formats.
    """
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val_lower = val.lower().strip()
        # Support multiple formats: inf, +inf, infinity
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
    Create tensor from spec.
    
    Args:
        spec: Dict with shape, dtype, sample_values
        framework: 'torch' or 'paddle'
    """
    shape = spec.get("shape", [])
    dtype_str = spec.get("dtype", "float32")
    sample_values = spec.get("sample_values", [])
    
    # Preprocess sample_values for special values
    processed_values = [parse_special_value(v) for v in sample_values]
    
    # Compute tensor size
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
            # Expand using processed_values as seed
            np.random.seed(42)  # Fixed seed for consistency
            if processed_values:
                # Filter NaN/Inf for statistics
                finite_values = [v for v in processed_values if np.isfinite(v)]
                if finite_values:
                    mean = np.mean(finite_values)
                    std = np.std(finite_values) if len(finite_values) > 1 else 1.0
                else:
                    mean, std = 0.0, 1.0
                
                # Generate base data
                base_data = np.random.normal(mean, std, size)
                
                # Fill sample_values in order
                for i, v in enumerate(processed_values):
                    if i < size:
                        base_data.flat[i] = v
                
                data = base_data.reshape(shape)
            else:
                data = np.random.randn(*shape)
    else:
        np.random.seed(42)
        data = np.random.randn(*shape)
    
    # Convert dtype
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
    Create PyTorch tensor from spec (supports tensor and scalar).
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
    Execute PyTorch test case (supports multiple inputs and extra params).
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
        
        # Non-tensor params (operator config, not input tensors)
        non_tensor_params = {
            "kernel_size", "stride", "padding", "dilation", "groups",
            "ceil_mode", "count_include_pad", "divisor_override",
            "alpha", "dim", "keepdim", "dtype", "out", "axis",
            "p", "eps", "momentum", "affine", "track_running_stats",
            "num_features", "normalized_shape", "weight", "bias",
            "return_indices", "exclusive", "reverse"
        }
        
        # Separate input tensors and params
        args = []
        kwargs = {}
        
        # Handle main input
        if "input" in test_case:
            input_tensor = create_torch_tensor(test_case["input"], torch_dtype_map)
            args.append(input_tensor)
        
        # Handle other possible tensor inputs (other, x, y, etc.)
        tensor_input_names = ["other", "x", "y", "tensor", "input1", "input2", "mat1", "mat2"]
        for name in tensor_input_names:
            if name in test_case:
                tensor = create_torch_tensor(test_case[name], torch_dtype_map)
                if name == "other":
                    kwargs["other"] = tensor
                else:
                    args.append(tensor)
        
        # Handle non-tensor params
        for key, value in test_case.items():
            if key not in ["api", "input", "x", "y", "other", "tensor", "input1", "input2", "mat1", "mat2"]:
                if key in non_tensor_params or not isinstance(value, dict):
                    kwargs[key] = value
        
        # Resolve API function
        api_parts = api_name.split(".")
        func = torch
        for part in api_parts[1:]:  # Skip 'torch'
            func = getattr(func, part)
        
        # Determine class (e.g., nn.AvgPool1d) vs function (e.g., torch.abs)
        if isinstance(func, type):
            # Class: instantiate then call
            # Extract init params from kwargs
            init_params = {k: v for k, v in kwargs.items() if k in non_tensor_params}
            instance = func(**init_params)
            result = instance(args[0] if args else kwargs.get("input"))
        else:
            # Function: call directly
            result = func(*args, **kwargs)
        
        # Convert result
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
    Create PaddlePaddle tensor from spec (supports tensor and scalar).
    """
    import paddle
    
    if isinstance(spec, (int, float, bool)):
        # Many PaddlePaddle APIs (e.g., paddle.add) require Tensor inputs
        # Convert scalar to 0-D tensor
        return paddle.to_tensor(spec)
    if isinstance(spec, dict) and "shape" in spec:
        data, dtype_str = create_tensor_from_spec(spec, "paddle")
        paddle_dtype = paddle_dtype_map.get(dtype_str, "float32")
        return paddle.to_tensor(data, dtype=paddle_dtype)
    return spec


def execute_paddle_test(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute PaddlePaddle test case (supports multiple inputs and extra params).
    """
    try:
        import paddle
        
        api_name = test_case.get("api", "")
        
        # PaddlePaddle dtype mapping
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
        
        # PaddlePaddle non-tensor params
        non_tensor_params = {
            "kernel_size", "stride", "padding", "dilation", "groups",
            "ceil_mode", "count_include_pad", "divisor_override", "exclusive",
            "axis", "keepdim", "name", "dtype", "data_format",
            "epsilon", "weight_attr", "bias_attr", "return_indices",
            "p", "eps", "momentum", "use_input_stats", "reverse"
        }
        
        # Separate input tensors and params
        args = []
        kwargs = {}
        
        # Handle main input
        if "input" in test_case:
            input_tensor = create_paddle_tensor(test_case["input"], paddle_dtype_map)
            args.append(input_tensor)
        
        # Handle other tensor inputs
        tensor_input_names = ["other", "x", "y", "tensor", "input1", "input2"]
        for name in tensor_input_names:
            if name in test_case:
                tensor = create_paddle_tensor(test_case[name], paddle_dtype_map)
                args.append(tensor)
        
        # Handle non-tensor params
        for key, value in test_case.items():
            if key not in ["api", "input", "x", "y", "other", "tensor", "input1", "input2"]:
                if key in non_tensor_params or not isinstance(value, dict):
                    kwargs[key] = value
        
        # Resolve API function/class
        api_parts = api_name.split(".")
        func = paddle
        for part in api_parts[1:]:  # Skip 'paddle'
            func = getattr(func, part)
        
        # Determine nn layer (e.g., paddle.nn.AvgPool1D)
        is_nn_layer = "nn" in api_name and not "functional" in api_name
        
        if is_nn_layer and isinstance(func, type):
            # nn layer: instantiate then call
            init_params = {k: v for k, v in kwargs.items() if k in non_tensor_params}
            layer = func(**init_params)
            result = layer(args[0] if args else kwargs.get("input"))
        else:
            # Regular function: call directly
            if args and not kwargs:
                result = func(*args)
            elif args:
                result = func(*args, **kwargs)
            else:
                result = func(**kwargs)
        
        # Convert result
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
    Compare execution results between two frameworks.
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
    
    # If either side failed
    if not torch_result["success"] or not paddle_result["success"]:
        if torch_result["success"] != paddle_result["success"]:
            comparison["comparison_error"] = "Execution mismatch: one success, one failure"
        return comparison
    
    # Compare results
    try:
        torch_res = torch_result["result"]
        paddle_res = paddle_result["result"]
        
        if torch_res is None and paddle_res is None:
            comparison["results_match"] = True
            return comparison
        
        if isinstance(torch_res, np.ndarray) and isinstance(paddle_res, np.ndarray):
            # Shape check
            if torch_res.shape != paddle_res.shape:
                comparison["comparison_error"] = f"Shape mismatch: torch={torch_res.shape}, paddle={paddle_res.shape}"
                return comparison
            
            # Handle special values
            torch_nan = np.isnan(torch_res) if np.issubdtype(torch_res.dtype, np.floating) else np.zeros_like(torch_res, dtype=bool)
            paddle_nan = np.isnan(paddle_res) if np.issubdtype(paddle_res.dtype, np.floating) else np.zeros_like(paddle_res, dtype=bool)
            
            torch_inf = np.isinf(torch_res) if np.issubdtype(torch_res.dtype, np.floating) else np.zeros_like(torch_res, dtype=bool)
            paddle_inf = np.isinf(paddle_res) if np.issubdtype(paddle_res.dtype, np.floating) else np.zeros_like(paddle_res, dtype=bool)
            
            # NaN positions must match
            if not np.array_equal(torch_nan, paddle_nan):
                comparison["comparison_error"] = "NaN positions mismatch"
                return comparison
            
            # Inf positions and signs must match
            if not np.array_equal(torch_inf, paddle_inf):
                comparison["comparison_error"] = "Inf positions mismatch"
                return comparison
            
            if np.any(torch_inf):
                if not np.array_equal(np.sign(torch_res[torch_inf]), np.sign(paddle_res[paddle_inf])):
                    comparison["comparison_error"] = "Inf sign mismatch"
                    return comparison
            
            # Compare numeric values
            mask = ~(torch_nan | torch_inf)
            if np.any(mask):
                if torch_res.size > 0:
                    if np.allclose(torch_res[mask], paddle_res[mask], rtol=rtol, atol=atol, equal_nan=True):
                        comparison["results_match"] = True
                    else:
                        max_diff = np.max(np.abs(torch_res[mask] - paddle_res[mask]))
                        comparison["comparison_error"] = f"Numeric mismatch, max diff: {max_diff}"
                else:
                    comparison["results_match"] = True
            else:
                comparison["results_match"] = True
        else:
            # Non-tensor result
            if torch_res == paddle_res:
                comparison["results_match"] = True
            else:
                comparison["comparison_error"] = f"Result mismatch: torch={torch_res}, paddle={paddle_res}"
                
    except Exception as e:
        comparison["comparison_error"] = f"Comparison error: {str(e)}"
    
    return comparison


def run_fuzzing_for_case(
    client,
    original_case: Dict[str, Any],
    torch_doc: str,
    paddle_doc: str,
    model: str = DEFAULT_MODEL
) -> List[Dict[str, Any]]:
    """
    Run multiple fuzzing rounds for a single test case.
    """
    torch_case = original_case.get("torch_test_case", {})
    paddle_case = original_case.get("paddle_test_case", {})
    torch_api = torch_case.get("api", "")
    paddle_api = paddle_case.get("api", "")
    
    fuzzing_results = []
    
    for round_num in range(1, FUZZING_ROUNDS + 1):
        print(f"    [Round {round_num}/{FUZZING_ROUNDS}] Generating mutated case...")
        
        # Build prompt
        prompt = build_fuzzing_prompt(
            torch_api, paddle_api, original_case, torch_doc, paddle_doc, round_num
        )
        
        try:
            # Call LLM (with retries)
            max_retries = 2
            mutated_case = None
            llm_response = ""
            
            for retry in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7 + retry * 0.1,  # Slightly increase randomness on retry
                        max_tokens=8192,  # Higher limit to avoid truncation
                    )
                    llm_response = response.choices[0].message.content.strip()
                    
                    # Parse response
                    mutated_case = parse_llm_response(llm_response)
                    
                    if mutated_case is not None:
                        # Validate required fields
                        if "torch_test_case" in mutated_case and "paddle_test_case" in mutated_case:
                            break
                        else:
                            print(f"[WARN] Round {round_num} missing required fields, retry ({retry + 1}/{max_retries})")
                            mutated_case = None
                    else:
                        if retry < max_retries - 1:
                            print(f"[WARN] Round {round_num} parse failed, retry ({retry + 1}/{max_retries})")
                except Exception as e:
                    print(f"[WARN] Round {round_num} LLM call error: {e}, retry ({retry + 1}/{max_retries})")
                    if retry < max_retries - 1:
                        import time
                        time.sleep(1)  # Wait 1 second then retry
            
            if mutated_case is None:
                fuzzing_results.append({
                    "round": round_num,
                    "success": False,
                    "error": "Failed to parse LLM response",
                    "llm_response": llm_response[:1000]  # Keep more for debugging
                })
                continue
            
            print(f"    [Round {round_num}] Running differential test...")
            
            # Execute tests
            torch_test = mutated_case.get("torch_test_case", {})
            paddle_test = mutated_case.get("paddle_test_case", {})
            
            torch_result = execute_torch_test(torch_test)
            paddle_result = execute_paddle_test(paddle_test)
            
            # Compare results
            comparison = compare_results(torch_result, paddle_result)
            
            # Record result
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
            
            # Print summary
            if fuzzing_result["is_bug_candidate"]:
                print(f"    [Round {round_num}] ⚠️ Potential issue: {comparison.get('comparison_error') or 'Execution mismatch'}")
            else:
                print(f"    [Round {round_num}] ✓ Results match")
                
        except Exception as e:
            fuzzing_results.append({
                "round": round_num,
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "traceback": traceback.format_exc()
            })
            print(f"    [Round {round_num}] ✗ Error: {e}")
    
    return fuzzing_results


def process_operator(
    operator_file: Path,
    client,
    model: str,
    max_cases: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process all success cases for a single operator.
    """
    with open(operator_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    operator_name = data.get("operator", "unknown")
    success_cases = data.get("success_cases", [])
    
    if max_cases:
        success_cases = success_cases[:max_cases]
    
    print(f"\nProcessing operator: {operator_name} ({len(success_cases)} cases)")
    
    # Get API names
    if success_cases:
        torch_api = success_cases[0].get("torch_test_case", {}).get("api", "")
        paddle_api = success_cases[0].get("paddle_test_case", {}).get("api", "")
    else:
        return {"operator": operator_name, "results": [], "error": "No success cases"}
    
    # Fetch docs
    print(f"  Fetching {torch_api} docs...")
    torch_doc = get_doc_content(torch_api, "pytorch")
    print(f"  Fetching {paddle_api} docs...")
    paddle_doc = get_doc_content(paddle_api, "paddle")
    
    # Fuzz each case
    all_results = []
    bug_candidates = 0
    
    for idx, case in enumerate(success_cases, 1):
        print(f"\n  Case {idx}/{len(success_cases)} (iteration={case.get('iteration')}, case={case.get('case_number')})")
        
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
        
        # Count potential bugs
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
    Save operator fuzzing results with timestamped filename.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    operator_name = result.get("operator", "unknown")
    # Add timestamp suffix YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{operator_name}_fuzzing_result_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"  Result saved: {output_file}")


def main():
    """
    Program entry point.
    """
    parser = argparse.ArgumentParser(
        description="PyTorch-PaddlePaddle LLM-based fuzzing differential test"
    )
    parser.add_argument(
        "--operators", "-o",
        nargs="*",
        help="Operators to test (if omitted, test all)"
    )
    parser.add_argument(
        "--max-cases", "-m",
        type=int,
        default=None,
        help="Max cases per operator (default: all)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model name (default {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--key-path", "-k",
        default=DEFAULT_KEY_PATH,
        help=f"API key file path (default {DEFAULT_KEY_PATH})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max operators to process (for testing)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PyTorch-PaddlePaddle LLM-based fuzzing differential test")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Fuzzing rounds per case: {FUZZING_ROUNDS}")
    
    # Initialize LLM client
    try:
        client = get_qwen_client(args.key_path)
        print("LLM client initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM client: {e}")
        return
    
    # Get operator files to process
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
    
    print(f"Operators to process: {len(operator_files)}")
    print("=" * 70)
    
    # Stats
    total_operators = len(operator_files)
    total_bug_candidates = 0
    
    # Process each operator
    for idx, op_file in enumerate(operator_files, 1):
        print(f"\n[{idx}/{total_operators}] Processing: {op_file.stem}")
        
        try:
            result = process_operator(
                op_file, client, args.model, args.max_cases
            )
            
            # Save result
            save_operator_result(result, RESULT_DIR)
            
            total_bug_candidates += result.get("bug_candidates", 0)
            
        except Exception as e:
            print(f"[ERROR] Error processing {op_file.stem}: {e}")
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 70)
    print("Fuzzing test complete!")
    print("=" * 70)
    print(f"Operators processed: {total_operators}")
    print(f"Potential issues found: {total_bug_candidates}")
    print(f"Results directory: {RESULT_DIR}")


if __name__ == "__main__":
    main()
