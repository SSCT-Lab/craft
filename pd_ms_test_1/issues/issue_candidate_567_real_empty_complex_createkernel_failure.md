Add a title*:
mindspore.ops.real fails on empty complex vector while paddle.real succeeds (CPU)

Add a description:
For empty complex input with shape `(0,)`, `paddle.real` succeeds but `mindspore.ops.real` fails with CPU kernel CreateKernel RuntimeError.

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
`mindspore.ops.real` fails for `complex64` input shape `(0,)` with RuntimeError and CreateKernel call stack; Paddle succeeds.

## Describe the expected behavior
`real` should support empty complex vectors and return empty real outputs, or raise a clear user-facing error rather than internal kernel init failure.

## Steps to reproduce the issue
1. Run `pd_ms_test_1/simple_tests/ms_error_only_candidate_567_real.py`.
2. Observe `[PD] success` and `[OTHER] error`.
3. Check runtime log showing unary CPU kernel init failure.

## Related log / screenshot
```text
[ERROR] ... For 'Real', it got empty inputs or outputs, which is invalid.
RuntimeError:
- C++ Call Stack:
mindspore\ccsrc\plugin\cpu\cpu_device_context.cc:543 mindspore::device::cpu::CPUKernelExecutor::CreateKernel
```

## Special notes for this issue
- Repro script: `pd_ms_test_1/simple_tests/ms_error_only_candidate_567_real.py`
- Paddle version in repro: 3.2.0
