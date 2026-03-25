# LLM Fuzzing Differential Testing Tool - Bug Fixes Design

## Architecture Overview

This design addresses three categories of false positives in the differential testing tool:
1. **Dtype Constraint Violations** - Framework-specific type requirements
2. **NaN Comparison Gaps** - Incomplete special value handling
3. **Shape Format Mismatches** - NCHW vs NHWC layout differences

## Design Principles

1. **Fail Fast**: Detect incompatibilities before execution when possible
2. **Clear Attribution**: Distinguish between true bugs and expected differences
3. **Minimal Overhead**: Keep validation and conversion costs low
4. **Maintainability**: Centralize configuration and make logic reusable

## Component Design

### 1. Dtype Constraint System

#### 1.1 API Metadata Registry

Create a centralized registry of API-specific dtype constraints:

```python
# File: pt_tf_test/fuzzing/api_constraints.py

API_DTYPE_CONSTRAINTS = {
    # Logical operators require bool in TensorFlow
    "tf.logical_and": {"input_dtypes": ["bool"], "strict": True},
    "tf.logical_or": {"input_dtypes": ["bool"], "strict": True},
    "tf.logical_xor": {"input_dtypes": ["bool"], "strict": True},
    "tf.logical_not": {"input_dtypes": ["bool"], "strict": True},
    
    # Bitwise operators require integer types
    "tf.bitwise.bitwise_and": {"input_dtypes": ["int32", "int64"], "strict": True},
    "tf.bitwise.bitwise_or": {"input_dtypes": ["int32", "int64"], "strict": True},
    
    # Add more as needed
}

def get_dtype_constraints(api_name: str) -> Optional[Dict[str, Any]]:
    """Get dtype constraints for an API, if any."""
    return API_DTYPE_CONSTRAINTS.get(api_name)

def validate_dtype_compatibility(
    torch_api: str,
    tf_api: str,
    torch_case: Dict[str, Any],
    tf_case: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Validate dtype compatibility between test cases.
    
    Returns:
        (is_compatible, error_message)
    """
    tf_constraints = get_dtype_constraints(tf_api)
    if not tf_constraints:
        return True, None
    
    # Check TensorFlow input dtype
    tf_input = tf_case.get("input", {})
    tf_dtype = tf_input.get("dtype", "float32")
    
    allowed_dtypes = tf_constraints.get("input_dtypes", [])
    if tf_dtype not in allowed_dtypes:
        return False, f"TensorFlow API {tf_api} requires dtype in {allowed_dtypes}, got {tf_dtype}"
    
    # Check 'other' parameter if exists
    if "other" in tf_case:
        other_dtype = tf_case["other"].get("dtype", "float32")
        if other_dtype not in allowed_dtypes:
            return False, f"TensorFlow API {tf_api} requires 'other' dtype in {allowed_dtypes}, got {other_dtype}"
    
    return True, None
```

#### 1.2 Enhanced LLM Prompt

Update `build_fuzzing_prompt()` to include dtype constraints:

```python
def build_fuzzing_prompt(...) -> str:
    # ... existing code ...
    
    # Add dtype constraints section
    tf_constraints = get_dtype_constraints(tf_api)
    dtype_guidance = ""
    if tf_constraints:
        allowed = tf_constraints.get("input_dtypes", [])
        dtype_guidance = f"""
【重要：数据类型约束】
TensorFlow API {tf_api} 要求输入数据类型必须是: {', '.join(allowed)}
- 请确保 tensorflow_test_case 中的 input.dtype 和 other.dtype 都符合此要求
- PyTorch API {torch_api} 可能接受更多类型，但为了数学等价性，两边应使用相同的合法类型
"""
    
    prompt = f"""你是一个专业的深度学习框架测试专家...
    
{dtype_guidance}

【核心变异要求：数学等价性（最重要！）】
...
"""
    return prompt
```

#### 1.3 Pre-Execution Validation

Add validation before executing tests:

```python
def process_single_fuzzing_round(...) -> Dict[str, Any]:
    # ... existing code to get mutated_case ...
    
    # Validate dtype compatibility
    torch_test = mutated_case.get("torch_test_case", {})
    tf_test = mutated_case.get("tensorflow_test_case", {})
    
    is_compatible, error_msg = validate_dtype_compatibility(
        torch_api, tf_api, torch_test, tf_test
    )
    
    if not is_compatible:
        return {
            "round": round_num,
            "success": True,  # Not a failure, just incompatible
            "mutation_strategy": mutated_case.get("mutation_strategy", ""),
            "mutation_reason": mutated_case.get("mutation_reason", ""),
            "torch_test_case": torch_test,
            "tensorflow_test_case": tf_test,
            "execution_result": {
                "torch_success": False,
                "tensorflow_success": False,
                "torch_error": None,
                "tensorflow_error": None,
                "results_match": False,
                "comparison_error": f"Dtype incompatibility (skipped): {error_msg}",
                "torch_shape": None,
                "tensorflow_shape": None,
                "torch_dtype": None,
                "tensorflow_dtype": None,
            },
            "is_bug_candidate": False,  # Not a bug, just incompatible
            "skip_reason": "dtype_incompatibility"
        }
    
    # ... continue with execution ...
```

### 2. Complete NaN Comparison Support

#### 2.1 Enhanced Scalar Comparison

Update `compare_results()` to handle all scalar types:

```python
def is_scalar_nan(value: Any) -> bool:
    """Check if a value is a scalar NaN."""
    if isinstance(value, (float, np.floating)):
        return np.isnan(value)
    if isinstance(value, np.ndarray) and value.shape == ():
        return np.isnan(value.item())
    return False

def is_scalar_inf(value: Any) -> Tuple[bool, Optional[int]]:
    """
    Check if a value is a scalar Inf.
    
    Returns:
        (is_inf, sign) where sign is 1 for +inf, -1 for -inf, None otherwise
    """
    val = None
    if isinstance(value, (float, np.floating)):
        val = value
    elif isinstance(value, np.ndarray) and value.shape == ():
        val = value.item()
    
    if val is not None and np.isinf(val):
        return True, np.sign(val)
    return False, None

def compare_scalars(val1: Any, val2: Any) -> Tuple[bool, Optional[str]]:
    """
    Compare two scalar values, handling NaN and Inf.
    
    Returns:
        (matches, error_message)
    """
    # Both NaN
    if is_scalar_nan(val1) and is_scalar_nan(val2):
        return True, None
    
    # One NaN, one not
    if is_scalar_nan(val1) or is_scalar_nan(val2):
        return False, f"NaN mismatch: val1={val1}, val2={val2}"
    
    # Both Inf
    is_inf1, sign1 = is_scalar_inf(val1)
    is_inf2, sign2 = is_scalar_inf(val2)
    
    if is_inf1 and is_inf2:
        if sign1 == sign2:
            return True, None
        else:
            return False, f"Inf sign mismatch: val1={val1}, val2={val2}"
    
    # One Inf, one not
    if is_inf1 or is_inf2:
        return False, f"Inf mismatch: val1={val1}, val2={val2}"
    
    # Regular comparison
    try:
        if np.allclose(val1, val2, rtol=1e-5, atol=1e-8):
            return True, None
        else:
            return False, f"Value mismatch: val1={val1}, val2={val2}"
    except:
        if val1 == val2:
            return True, None
        else:
            return False, f"Value mismatch: val1={val1}, val2={val2}"
```

#### 2.2 Recursive Container Comparison

Update container comparison to use scalar comparison:

```python
def compare_results(...) -> Dict[str, Any]:
    # ... existing setup code ...
    
    # Handle scalar results (including 0-d arrays)
    if not isinstance(torch_res, (tuple, list, np.ndarray)) or \
       (isinstance(torch_res, np.ndarray) and torch_res.shape == ()):
        matches, error = compare_scalars(torch_res, tf_res)
        comparison["results_match"] = matches
        if error:
            comparison["comparison_error"] = error
        return comparison
    
    # Handle empty containers
    if isinstance(torch_res, (tuple, list)) and isinstance(tf_res, (tuple, list)):
        if len(torch_res) == 0 and len(tf_res) == 0:
            comparison["results_match"] = True
            return comparison
        
        # Recursive comparison for non-empty containers
        if len(torch_res) != len(tf_res):
            comparison["comparison_error"] = f"Container length mismatch: torch={len(torch_res)}, tf={len(tf_res)}"
            return comparison
        
        all_match = True
        for i, (t_item, tf_item) in enumerate(zip(torch_res, tf_res)):
            # Recursively compare each element
            item_comparison = compare_results(
                {"success": True, "result": t_item, "shape": None, "dtype": None, "error": None},
                {"success": True, "result": tf_item, "shape": None, "dtype": None, "error": None},
                rtol, atol
            )
            if not item_comparison["results_match"]:
                all_match = False
                comparison["comparison_error"] = f"Container element {i} mismatch: {item_comparison.get('comparison_error', 'unknown')}"
                break
        
        comparison["results_match"] = all_match
        return comparison
    
    # Handle numpy arrays (existing code with improvements)
    if isinstance(torch_res, np.ndarray) and isinstance(tf_res, np.ndarray):
        # ... existing array comparison code ...
        pass
    
    # Fallback for other types
    matches, error = compare_scalars(torch_res, tf_res)
    comparison["results_match"] = matches
    if error:
        comparison["comparison_error"] = error
    return comparison
```

### 3. Shape Format Adaptation

#### 3.1 API Shape Format Registry

Create registry of APIs requiring shape conversion:

```python
# File: pt_tf_test/fuzzing/shape_format.py

# APIs that use NCHW in PyTorch and NHWC in TensorFlow
NCHW_APIS = {
    # Convolution layers
    "torch.nn.Conv2d": "tf.keras.layers.Conv2D",
    "torch.nn.functional.conv2d": "tf.nn.conv2d",
    "torch.nn.ConvTranspose2d": "tf.keras.layers.Conv2DTranspose",
    
    # Pooling layers
    "torch.nn.AvgPool2d": "tf.keras.layers.AveragePooling2D",
    "torch.nn.MaxPool2d": "tf.keras.layers.MaxPooling2D",
    "torch.nn.functional.avg_pool2d": "tf.nn.avg_pool2d",
    "torch.nn.functional.max_pool2d": "tf.nn.max_pool2d",
    
    # Normalization layers
    "torch.nn.BatchNorm2d": "tf.keras.layers.BatchNormalization",
    
    # Add more as needed
}

def requires_shape_conversion(torch_api: str, tf_api: str) -> bool:
    """Check if APIs require NCHW/NHWC conversion."""
    return torch_api in NCHW_APIS and NCHW_APIS[torch_api] == tf_api

def nchw_to_nhwc(tensor: np.ndarray) -> np.ndarray:
    """
    Convert tensor from NCHW to NHWC format.
    
    Args:
        tensor: 4D array with shape (N, C, H, W)
    
    Returns:
        4D array with shape (N, H, W, C)
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tensor.shape}")
    
    # Transpose: (N, C, H, W) -> (N, H, W, C)
    return np.transpose(tensor, (0, 2, 3, 1))

def nhwc_to_nchw(tensor: np.ndarray) -> np.ndarray:
    """
    Convert tensor from NHWC to NCHW format.
    
    Args:
        tensor: 4D array with shape (N, H, W, C)
    
    Returns:
        4D array with shape (N, C, H, W)
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tensor.shape}")
    
    # Transpose: (N, H, W, C) -> (N, C, H, W)
    return np.transpose(tensor, (0, 3, 1, 2))
```

#### 3.2 Tensor Creation with Format Conversion

Update tensor creation functions:

```python
def create_tf_tensor_with_format(
    spec: Any,
    tf_dtype_map: Dict,
    torch_api: str,
    tf_api: str
) -> Any:
    """
    Create TensorFlow tensor with shape format conversion if needed.
    """
    import tensorflow as tf
    
    if isinstance(spec, (int, float, bool)):
        return spec
    
    if isinstance(spec, dict) and "shape" in spec:
        data, dtype_str = create_tensor_from_spec(spec, "tensorflow")
        
        # Apply shape conversion if needed
        if requires_shape_conversion(torch_api, tf_api) and len(data.shape) == 4:
            data = nchw_to_nhwc(data)
        
        tf_dtype = tf_dtype_map.get(dtype_str, tf.float32)
        return tf.constant(data, dtype=tf_dtype)
    
    return spec

def execute_tensorflow_test_with_format(
    test_case: Dict[str, Any],
    torch_api: str
) -> Dict[str, Any]:
    """
    Execute TensorFlow test with shape format handling.
    """
    # ... existing setup code ...
    
    tf_api = test_case.get("api", "")
    needs_conversion = requires_shape_conversion(torch_api, tf_api)
    
    # Create tensors with format conversion
    if "input" in test_case:
        input_tensor = create_tf_tensor_with_format(
            test_case["input"], tf_dtype_map, torch_api, tf_api
        )
        args.append(input_tensor)
    
    # ... execute API ...
    
    # Convert result back if needed
    if isinstance(result, tf.Tensor):
        result_np = result.numpy()
        
        # Convert output from NHWC back to NCHW for comparison
        if needs_conversion and result_np.ndim == 4:
            result_np = nhwc_to_nchw(result_np)
        
        return {
            "success": True,
            "result": result_np,
            "shape": list(result_np.shape),
            "dtype": str(result.dtype),
            "error": None
        }
    
    # ... rest of function ...
```

#### 3.3 Integration Points

Update main execution functions:

```python
def process_single_fuzzing_round(...) -> Dict[str, Any]:
    # ... existing code ...
    
    # Execute tests with format awareness
    torch_result = execute_torch_test(torch_test)
    
    # Pass torch_api to TensorFlow execution for format conversion
    tf_result = execute_tensorflow_test_with_format(
        tf_test,
        torch_api=torch_api
    )
    
    # ... rest of function ...
```

## Testing Strategy

### Unit Tests

Create comprehensive test file: `pt_tf_test/fuzzing/test_bug_fixes.py`

```python
def test_dtype_validation():
    """Test dtype constraint validation."""
    # Test logical_or with float32 (should be incompatible)
    # Test logical_or with bool (should be compatible)
    pass

def test_nan_scalar_comparison():
    """Test NaN comparison for scalars."""
    # Test float NaN vs float NaN
    # Test np.float32 NaN vs np.float32 NaN
    # Test 0-d array NaN vs 0-d array NaN
    pass

def test_nan_container_comparison():
    """Test NaN comparison in containers."""
    # Test tuple of NaN scalars
    # Test list of NaN scalars
    # Test tuple of arrays containing NaN
    pass

def test_shape_conversion():
    """Test NCHW/NHWC conversion."""
    # Test nchw_to_nhwc
    # Test nhwc_to_nchw
    # Test round-trip conversion
    pass

def test_shape_format_detection():
    """Test API shape format detection."""
    # Test Conv2d requires conversion
    # Test torch.abs does not require conversion
    pass
```

### Integration Tests

Test with actual problematic cases:

```python
def test_logical_or_dtype_fix():
    """Test that logical_or dtype issue is fixed."""
    # Load torch_logical_or_fuzzing_result_20260131_234827.json
    # Verify dtype incompatibility is detected and skipped
    # Verify not marked as bug candidate
    pass

def test_max_nan_fix():
    """Test that torch.max NaN comparison is fixed."""
    # Load torch_max_fuzzing_result_20260131_234929.json
    # Verify NaN results are correctly compared
    # Verify not marked as bug candidate
    pass
```

## Implementation Plan

### Phase 1: Dtype Constraints (Priority: High)
1. Create `api_constraints.py` with dtype registry
2. Implement `validate_dtype_compatibility()`
3. Update `build_fuzzing_prompt()` with dtype guidance
4. Add pre-execution validation in `process_single_fuzzing_round()`
5. Write unit tests for dtype validation
6. Test with `torch_logical_or` cases

### Phase 2: NaN Comparison (Priority: High)
1. Implement `is_scalar_nan()` and `is_scalar_inf()`
2. Implement `compare_scalars()`
3. Update `compare_results()` to use scalar comparison
4. Enhance recursive container comparison
5. Write comprehensive unit tests
6. Test with `torch_max` cases

### Phase 3: Shape Format (Priority: Medium)
1. Create `shape_format.py` with API registry
2. Implement `nchw_to_nhwc()` and `nhwc_to_nchw()`
3. Update `create_tf_tensor()` with format conversion
4. Update `execute_tensorflow_test()` with format conversion
5. Write unit tests for conversion functions
6. Test with Conv2d and AvgPool2d cases

### Phase 4: Integration & Documentation (Priority: Medium)
1. Run full test suite to ensure no regressions
2. Update documentation with new features
3. Add examples of each fix in action
4. Create migration guide for existing results

## Rollout Strategy

1. **Development**: Implement fixes in feature branch
2. **Testing**: Run against known problematic cases
3. **Validation**: Verify no false positives in test suite
4. **Deployment**: Merge to main branch
5. **Monitoring**: Track false positive rate in production

## Success Criteria

- ✅ Zero false positives from dtype mismatches
- ✅ Zero false positives from NaN comparisons
- ✅ Successful testing of Conv2d and AvgPool2d operations
- ✅ All existing tests pass
- ✅ Test coverage > 90% for new code
