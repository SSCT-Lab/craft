Add a title*:
mindspore.ops.isreal fails on empty complex tensor while paddle.isreal succeeds (CPU)

Add a description:
For empty complex input (shape `(1, 0, 4)`), `paddle.isreal` succeeds but `mindspore.ops.isreal` fails. Internally MindSpore reports `Imag` kernel init failure and CreateKernel RuntimeError.

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
`mindspore.ops.isreal` fails on `complex64` input shape `(1, 0, 4)`. Logs indicate underlying `Imag` CPU kernel rejects empty input/output.

## Describe the expected behavior
`isreal` should work for empty complex tensors and return empty boolean tensor, or fail with clear API-level validation message instead of internal kernel creation failure.

## Steps to reproduce the issue
1. Run `pd_ms_test_1/simple_tests/ms_error_only_candidate_179_isreal.py`.
2. Observe `[PD] success` and `[OTHER] error`.
3. Error output contains `For 'Imag', it got empty inputs or outputs, which is invalid.`

## Related log / screenshot
```text
[ERROR] ... For 'Imag', it got empty inputs or outputs, which is invalid.
RuntimeError:
- C++ Call Stack:
mindspore\ccsrc\plugin\cpu\cpu_device_context.cc:543 mindspore::device::cpu::CPUKernelExecutor::CreateKernel
```

## Special notes for this issue
- Repro script: `pd_ms_test_1/simple_tests/ms_error_only_candidate_179_isreal.py`
- Paddle version in repro: 3.2.0
