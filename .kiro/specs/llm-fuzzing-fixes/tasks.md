# LLM Fuzzing Differential Testing Tool - Bug Fixes Tasks

## Phase 1: Dtype Constraint System

### 1.1 Create API Constraints Module
- [ ] 1.1.1 Create `pt_tf_test/fuzzing/api_constraints.py` file
- [ ] 1.1.2 Define `API_DTYPE_CONSTRAINTS` dictionary with logical operators
- [ ] 1.1.3 Implement `get_dtype_constraints(api_name)` function
- [ ] 1.1.4 Implement `validate_dtype_compatibility()` function
- [ ] 1.1.5 Add docstrings and type hints

### 1.2 Update LLM Prompt Generation
- [ ] 1.2.1 Import `get_dtype_constraints` in main file
- [ ] 1.2.2 Update `build_fuzzing_prompt()` to check for dtype constraints
- [ ] 1.2.3 Add dtype guidance section to prompt template
- [ ] 1.2.4 Test prompt generation with logical operators

### 1.3 Add Pre-Execution Validation
- [ ] 1.3.1 Import validation function in `process_single_fuzzing_round()`
- [ ] 1.3.2 Add dtype validation before test execution
- [ ] 1.3.3 Return appropriate result structure for incompatible dtypes
- [ ] 1.3.4 Ensure `is_bug_candidate` is False for dtype incompatibilities
- [ ] 1.3.5 Add `skip_reason` field to result

### 1.4 Test Dtype Constraint System
- [ ] 1.4.1 Create test cases for `validate_dtype_compatibility()`
- [ ] 1.4.2 Test with logical_or + float32 (should be incompatible)
- [ ] 1.4.3 Test with logical_or + bool (should be compatible)
- [ ] 1.4.4 Test with torch.abs + float32 (should be compatible - no constraints)
- [ ] 1.4.5 Run against `torch_logical_or_fuzzing_result_20260131_234827.json`
- [ ] 1.4.6 Verify no false positives from dtype mismatches

## Phase 2: Complete NaN Comparison Support

### 2.1 Implement Scalar Comparison Utilities
- [ ] 2.1.1 Implement `is_scalar_nan(value)` function
- [ ] 2.1.2 Implement `is_scalar_inf(value)` function
- [ ] 2.1.3 Implement `compare_scalars(val1, val2)` function
- [ ] 2.1.4 Add comprehensive docstrings
- [ ] 2.1.5 Add unit tests for each utility function

### 2.2 Update compare_results() Function
- [ ] 2.2.1 Add scalar comparison branch at start of function
- [ ] 2.2.2 Handle 0-dimensional numpy arrays as scalars
- [ ] 2.2.3 Update container comparison to use `compare_scalars()`
- [ ] 2.2.4 Ensure recursive comparison works for nested containers
- [ ] 2.2.5 Preserve existing array comparison logic

### 2.3 Test NaN Comparison
- [ ] 2.3.1 Test scalar float NaN comparison
- [ ] 2.3.2 Test scalar np.floating NaN comparison
- [ ] 2.3.3 Test 0-d array NaN comparison
- [ ] 2.3.4 Test tuple of NaN scalars
- [ ] 2.3.5 Test list of NaN scalars
- [ ] 2.3.6 Test tuple containing arrays with NaN
- [ ] 2.3.7 Run against `torch_max_fuzzing_result_20260131_234929.json`
- [ ] 2.3.8 Verify all NaN comparisons pass correctly

## Phase 3: Shape Format Adaptation

### 3.1 Create Shape Format Module
- [ ] 3.1.1 Create `pt_tf_test/fuzzing/shape_format.py` file
- [ ] 3.1.2 Define `NCHW_APIS` dictionary with Conv2d, AvgPool2d, etc.
- [ ] 3.1.3 Implement `requires_shape_conversion()` function
- [ ] 3.1.4 Implement `nchw_to_nhwc()` conversion function
- [ ] 3.1.5 Implement `nhwc_to_nchw()` conversion function
- [ ] 3.1.6 Add validation for 4D tensor requirement
- [ ] 3.1.7 Add docstrings and type hints

### 3.2 Update Tensor Creation
- [ ] 3.2.1 Create `create_tf_tensor_with_format()` function
- [ ] 3.2.2 Add shape conversion logic for 4D tensors
- [ ] 3.2.3 Preserve existing behavior for non-4D tensors
- [ ] 3.2.4 Update `execute_tensorflow_test()` to accept torch_api parameter
- [ ] 3.2.5 Apply format conversion to input tensors
- [ ] 3.2.6 Apply reverse conversion to output tensors

### 3.3 Integrate Shape Format System
- [ ] 3.3.1 Import shape format functions in main file
- [ ] 3.3.2 Update `process_single_fuzzing_round()` to pass torch_api
- [ ] 3.3.3 Update all calls to `execute_tensorflow_test()`
- [ ] 3.3.4 Ensure backward compatibility with existing code

### 3.4 Test Shape Format Conversion
- [ ] 3.4.1 Test `nchw_to_nhwc()` with sample 4D tensor
- [ ] 3.4.2 Test `nhwc_to_nchw()` with sample 4D tensor
- [ ] 3.4.3 Test round-trip conversion (NCHW → NHWC → NCHW)
- [ ] 3.4.4 Test `requires_shape_conversion()` with Conv2d
- [ ] 3.4.5 Test `requires_shape_conversion()` with torch.abs (should be False)
- [ ] 3.4.6 Create integration test with Conv2d operation
- [ ] 3.4.7 Create integration test with AvgPool2d operation
- [ ] 3.4.8 Verify no false positives for convolution/pooling ops

## Phase 4: Integration & Testing

### 4.1 Create Comprehensive Test Suite
- [ ] 4.1.1 Create `pt_tf_test/fuzzing/test_bug_fixes.py`
- [ ] 4.1.2 Add all unit tests from previous phases
- [ ] 4.1.3 Add integration tests for each fix
- [ ] 4.1.4 Add regression tests for existing functionality
- [ ] 4.1.5 Ensure test coverage > 90%

### 4.2 Run Full Test Suite
- [ ] 4.2.1 Run all unit tests
- [ ] 4.2.2 Run all integration tests
- [ ] 4.2.3 Run existing test suite to check for regressions
- [ ] 4.2.4 Fix any failing tests
- [ ] 4.2.5 Verify all tests pass

### 4.3 Validate with Real Cases
- [ ] 4.3.1 Re-run fuzzing on torch_logical_or
- [ ] 4.3.2 Verify dtype incompatibilities are properly handled
- [ ] 4.3.3 Re-run fuzzing on torch_max
- [ ] 4.3.4 Verify NaN comparisons work correctly
- [ ] 4.3.5 Run fuzzing on Conv2d or AvgPool2d
- [ ] 4.3.6 Verify shape format conversion works correctly

### 4.4 Documentation
- [ ] 4.4.1 Update README with new features
- [ ] 4.4.2 Document dtype constraint system
- [ ] 4.4.3 Document NaN comparison improvements
- [ ] 4.4.4 Document shape format adaptation
- [ ] 4.4.5 Add examples for each fix
- [ ] 4.4.6 Create troubleshooting guide

### 4.5 Code Review & Cleanup
- [ ] 4.5.1 Review all new code for quality
- [ ] 4.5.2 Ensure consistent code style
- [ ] 4.5.3 Remove any debug print statements
- [ ] 4.5.4 Optimize performance if needed
- [ ] 4.5.5 Add TODO comments for future improvements

## Phase 5: Deployment

### 5.1 Prepare for Merge
- [ ] 5.1.1 Ensure all tests pass
- [ ] 5.1.2 Update CHANGELOG with new features
- [ ] 5.1.3 Create pull request with detailed description
- [ ] 5.1.4 Address code review feedback

### 5.2 Monitor Production
- [ ] 5.2.1 Deploy to production
- [ ] 5.2.2 Monitor false positive rate
- [ ] 5.2.3 Collect user feedback
- [ ] 5.2.4 Address any issues that arise

## Notes

- **Priority**: Phase 1 and Phase 2 are high priority (address immediate false positives)
- **Priority**: Phase 3 is medium priority (enables new test scenarios)
- **Dependencies**: Phase 4 depends on completion of Phases 1-3
- **Testing**: Each phase should be tested independently before moving to next phase
- **Backward Compatibility**: All changes must maintain backward compatibility with existing test results
