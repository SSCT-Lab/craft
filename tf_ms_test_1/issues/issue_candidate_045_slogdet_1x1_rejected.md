Add a title*:
mindspore.ops.slogdet rejects 1x1 matrix while tf.linalg.slogdet succeeds (candidate 045)

Add a description:
On a 1x1 float32 matrix input (`[[3.0]]`), `tf.linalg.slogdet` succeeds but `mindspore.ops.slogdet` raises a ValueError requiring row size >= 2. This appears inconsistent with expected linear algebra behavior.

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
`ops.slogdet(ms.Tensor([[3.0]], ms.float32))` throws:
`ValueError: For primitive[LogMatrixDeterminant], the row size must be greater than or equal to 2, but got 1.`
TensorFlow `tf.linalg.slogdet` succeeds.

## Describe the expected behavior
`ops.slogdet` should support 1x1 matrices and return valid sign/logabsdet.

## Steps to reproduce the issue
1. Set environment variable before run: `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.
2. Run `tf_ms_test_1/simple_tests/ms_error_only_candidate_045_slogdet_1x1.py`.
3. Observe `[TF] success` and `[MS] error` with row-size restriction.

## Related log / screenshot
```text
ValueError: For primitive[LogMatrixDeterminant], the row size must be greater than or equal to 2, but got 1.
- C++ Call Stack:
mindspore/ops/infer/ops_func_impl//log_matrix_determinant.cc:50
```

## Special notes for this issue
- Repro script: `tf_ms_test_1/simple_tests/ms_error_only_candidate_045_slogdet_1x1.py`
- TensorFlow version in repro: 2.20.0
