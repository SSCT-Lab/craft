Add a title*:
mindspore.ops.slogdet rejects 1x1 matrix while paddle.linalg.slogdet succeeds

Add a description:
`mindspore.ops.slogdet` errors on a valid 1x1 float32 matrix with `row size must be greater than or equal to 2`, while `paddle.linalg.slogdet` succeeds. This appears to be an unnecessary shape restriction.

<!--  Thanks for sending an issue!  Here are some tips for you:

If this is your first time, please read our contributor guidelines: https://github.com/mindspore-ai/mindspore/blob/master/CONTRIBUTING.md
-->

## Environment
### Hardware Environment(`Ascend`/`GPU`/`CPU`):
/device cpu

### Software Environment:
- **MindSpore version (source or binary)**: 2.8.0 (binary)
- **Python version (e.g., Python 3.7.5)**: 3.10.18
- **OS platform and distribution (e.g., Linux Ubuntu 16.04)**: Windows 10 (10.0.26200)
- **GCC/Compiler version (if compiled from source)**: N/A

## Describe the current behavior
For input shape `(1, 1)`, `mindspore.ops.slogdet` raises:
`ValueError: For primitive[LogMatrixDeterminant], the row size must be greater than or equal to 2, but got 1.`
Paddle reference succeeds.

## Describe the expected behavior
`slogdet` should support 1x1 square matrices (standard linear algebra behavior), matching mainstream framework behavior.

## Steps to reproduce the issue
1. Run `pd_ms_test_1/simple_tests/ms_error_only_candidate_209_linalg_slogdet.py`.
2. Observe `[PD] success` and `[OTHER] error`.
3. Verify MindSpore throws row-size restriction error for 1x1 matrix.

## Related log / screenshot
```text
ValueError: For primitive[LogMatrixDeterminant], the row size must be greater than or equal to 2, but got 1.
- C++ Call Stack:
mindspore/ops/infer/ops_func_impl//log_matrix_determinant.cc:50
```

## Special notes for this issue
- Repro script: `pd_ms_test_1/simple_tests/ms_error_only_candidate_209_linalg_slogdet.py`
- Paddle version in repro: 3.2.0
