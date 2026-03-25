# LLM Fuzzing Differential Testing Tool - Bug Fixes

## Overview
This spec addresses three remaining issues in the LLM-based fuzzing differential testing tool for PyTorch-TensorFlow API comparison.

## Background
The tool generates mutated test cases using LLM and performs differential testing between PyTorch and TensorFlow. Previous fixes addressed:
- ✅ Missing required parameters in LLM-generated cases
- ✅ False positive comparisons for empty containers and basic NaN values

Three issues remain:
1. **Dtype Mismatch**: TensorFlow's logical operators require bool tensors, but LLM generates float32 inputs
2. **NaN Comparison Failures**: Some NaN comparisons still report mismatch despite fixes
3. **Shape Format Differences**: PyTorch uses NCHW, TensorFlow uses NHWC - no adaptation logic exists

## User Stories

### 1. Dtype Constraint Validation
**As a** differential testing user  
**I want** the tool to validate dtype constraints before execution  
**So that** I don't get false positives from incompatible dtype combinations

**Acceptance Criteria:**
1.1. Tool identifies APIs with dtype constraints (e.g., logical operators require bool)
1.2. LLM prompt includes dtype requirements for specific API categories
1.3. Pre-execution validation skips incompatible dtype combinations
1.4. Execution errors due to dtype mismatches are clearly labeled as "dtype incompatibility" not "bug candidate"
1.5. Test results distinguish between true bugs and dtype constraint violations

### 2. Complete NaN Comparison Support
**As a** differential testing user  
**I want** all NaN values to be compared correctly regardless of container type  
**So that** I don't get false positives from mathematically equivalent NaN results

**Acceptance Criteria:**
2.1. Scalar NaN values (float, np.floating) are correctly identified as matching
2.2. NaN values in numpy arrays are correctly compared by position
2.3. NaN values in tuples/lists of scalars are correctly compared
2.4. NaN values in nested containers (tuple of arrays, list of scalars) are correctly compared
2.5. Test file demonstrates all NaN comparison scenarios pass

### 3. Framework Shape Format Adaptation
**As a** differential testing user  
**I want** the tool to handle NCHW/NHWC format differences automatically  
**So that** I can test convolution and pooling operations without false positives

**Acceptance Criteria:**
3.1. Tool identifies APIs that require shape format conversion (Conv2D, AvgPool2D, etc.)
3.2. Input tensors are converted from NCHW to NHWC when calling TensorFlow APIs
3.3. Output tensors are converted from NHWC back to NCHW for comparison
3.4. Shape conversion is only applied to relevant APIs (not all operations)
3.5. Conversion logic handles 4D tensors (batch, channels, height, width)
3.6. Test results show correct comparisons for convolution/pooling operations

## Non-Functional Requirements

### Performance
- Dtype validation should add < 1ms overhead per test case
- Shape conversion should add < 5ms overhead per test case

### Maintainability
- Dtype constraints should be defined in a centralized configuration
- Shape format conversion should be modular and reusable
- All fixes should include comprehensive test coverage

### Compatibility
- Must work with existing test case format
- Must not break existing passing tests
- Must maintain backward compatibility with saved results

## Out of Scope
- Automatic dtype conversion (e.g., float32 → bool) - we only validate and skip
- Support for 3D convolutions (NCDHW/NDHWC) - only 2D for now
- Custom shape format specifications - only standard NCHW/NHWC

## Technical Notes

### Issue 1: Dtype Mismatch Details
From `torch_logical_or_fuzzing_result_20260131_234827.json`:
- PyTorch's `torch.logical_or` accepts float32 and auto-converts to bool
- TensorFlow's `tf.logical_or` requires bool tensors, rejects float32
- Error: "InvalidArgumentError: cannot compute LogicalOr as input #0(zero-based) was expected to be a bool tensor but is a float tensor"

### Issue 2: NaN Comparison Details
From `torch_max_fuzzing_result_20260131_234929.json`:
- Both frameworks return scalar NaN: `torch=nan, tf=nan`
- Comparison reports: `"comparison_error": "结果不一致: torch=nan, tf=nan"`
- Current fix handles scalar NaN but may not handle all result types

### Issue 3: Shape Format Details
- PyTorch: NCHW (batch, channels, height, width)
- TensorFlow: NHWC (batch, height, width, channels)
- Affects: Conv2D, AvgPool2D, MaxPool2D, BatchNorm2D, etc.
- Conversion: `(N, C, H, W) → (N, H, W, C)` for input, reverse for output

## Dependencies
- Existing codebase: `pt_tf_test/fuzzing/llm_fuzzing_diff_test_concurrent.py`
- Test framework: `pt_tf_test/fuzzing/test_comparison_fixes.py`
- NumPy for array operations
- PyTorch and TensorFlow for tensor operations

## Success Metrics
1. Zero false positives from dtype mismatches in logical operators
2. Zero false positives from NaN comparisons in any result type
3. Successful differential testing of at least 3 convolution/pooling operations
4. All existing tests continue to pass
5. New test coverage for all three fixes
