Add a title*:
mindspore.ops.imag fails on empty complex tensors with CPU CreateKernel RuntimeError

Add a description:
`mindspore.ops.Imag` and `mindspore.ops.imag` fail on empty complex tensors in MindSpore 2.8.0 (CPU, Windows). Runtime reports kernel init failure (`it got empty inputs or outputs, which is invalid`) and CreateKernel RuntimeError. PyTorch succeeds on equivalent inputs.

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
Both APIs fail when input has zero-sized dimension:
- `ops.Imag()(x)` with `x.shape=(2, 0, 3, 4), dtype=complex64`
- `ops.imag(x)` with `x.shape=(2, 0, 4), dtype=complex128`

MindSpore raises RuntimeError with CreateKernel call stack.

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

# Imag class API
x1 = ms.Tensor(np.empty((2, 0, 3, 4), dtype=np.complex64))
try:
    y1 = ops.Imag()(x1)
    print("MindSpore Imag class success:", y1.shape, y1.dtype)
except Exception as e:
    print("MindSpore Imag class failed:", type(e).__name__)
    print(e)

# imag function API
x2 = ms.Tensor(np.empty((2, 0, 4), dtype=np.complex128))
try:
    y2 = ops.imag(x2)
    print("MindSpore imag func success:", y2.shape, y2.dtype)
except Exception as e:
    print("MindSpore imag func failed:", type(e).__name__)
    print(e)

# PyTorch reference
t1 = torch.complex(torch.empty((2, 0, 3, 4), dtype=torch.float32), torch.empty((2, 0, 3, 4), dtype=torch.float32))
t2 = torch.complex(torch.empty((2, 0, 4), dtype=torch.float64), torch.empty((2, 0, 4), dtype=torch.float64))
print("PyTorch torch.imag #1:", torch.imag(t1).shape, torch.imag(t1).dtype)
print("PyTorch torch.imag #2:", torch.imag(t2).shape, torch.imag(t2).dtype)
```
2. Observe MindSpore RuntimeError and CreateKernel stack.
3. Observe PyTorch succeeds.

## Related log / screenshot
Representative log:
```text
[ERROR] ... UnaryOpCpuKernelMod::Init] For 'Imag', it got empty inputs or outputs, which is invalid.
RuntimeError:
- C++ Call Stack:
mindspore\ccsrc\plugin\cpu\cpu_device_context.cc:543 mindspore::device::cpu::CPUKernelExecutor::CreateKernel
```

## Special notes for this issue
- Repro scripts:
  - ms_pt_test/simple_tests/ms_error_only_candidate_380_imag_class.py
  - ms_pt_test/simple_tests/ms_error_only_candidate_865_imag_func.py
