Add a title*:
mindspore.ops.Real fails on empty complex tensors with CPU CreateKernel RuntimeError

Add a description:
`mindspore.ops.Real` fails on empty complex tensors in MindSpore 2.8.0 (CPU, Windows). Runtime reports kernel init failure (`it got empty inputs or outputs, which is invalid`) and CreateKernel RuntimeError. Equivalent PyTorch `torch.real` succeeds and returns empty output.

<!--  Thanks for sending an issue!  Here are some tips for you:

If this is your first time, please read our contributor guidelines: https://github.com/mindspore-ai/mindspore/blob/master/CONTRIBUTING.md
-->

## Environment
### Hardware Environment(`Ascend`/`GPU`/`CPU`):
/device cpu

### Software Environment:
- **MindSpore version (source or binary)**: 2.8.0 (binary, pip in conda env)
- **Python version (e.g., Python 3.7.5)**: 3.10.18
- **OS platform and distribution (e.g., Linux Ubuntu 16.04)**: Windows 10 (10.0.26200)
- **GCC/Compiler version (if compiled from source)**: N/A (not compiled from source)

## Describe the current behavior
`ops.Real()(x)` with `x.shape=(2, 0, 3, 1), dtype=complex128` fails in MindSpore and throws RuntimeError with CreateKernel call stack.

## Describe the expected behavior
- Return empty output tensor with correct shape/dtype, or
- Raise clear user-facing validation error (if empty tensors are unsupported), without internal CreateKernel failure.

## Steps to reproduce the issue
1. Save and run:
```python
import numpy as np
import torch
import mindspore as ms
from mindspore import ops

print("mindspore:", ms.__version__)
print("torch:", torch.__version__)

x_ms = ms.Tensor(np.empty((2, 0, 3, 1), dtype=np.complex128))
try:
    y_ms = ops.Real()(x_ms)
    print("MindSpore success:", y_ms.shape, y_ms.dtype)
except Exception as e:
    print("MindSpore failed:", type(e).__name__)
    print(e)

x_t = torch.complex(
    torch.empty((2, 0, 3, 1), dtype=torch.float64),
    torch.empty((2, 0, 3, 1), dtype=torch.float64),
)
y_t = torch.real(x_t)
print("PyTorch success:", y_t.shape, y_t.dtype)
```
2. Observe MindSpore RuntimeError and CreateKernel stack.
3. Observe PyTorch succeeds.

## Related log / screenshot
Representative log:
```text
[ERROR] ... UnaryOpCpuKernelMod::Init] For 'Real', it got empty inputs or outputs, which is invalid.
RuntimeError:
- C++ Call Stack:
mindspore\ccsrc\plugin\cpu\cpu_device_context.cc:543 mindspore::device::cpu::CPUKernelExecutor::CreateKernel
```

## Special notes for this issue
- Repro script:
  - ms_pt_test/simple_tests/ms_error_only_candidate_633_real_class.py
