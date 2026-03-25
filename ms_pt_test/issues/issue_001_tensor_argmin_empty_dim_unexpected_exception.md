Add a title*:
mindspore.Tensor.argmin on empty dimension triggers Framework Unexpected Exception on CPU

Add a description:
When calling `mindspore.Tensor.argmin` on an input that contains a zero-sized dimension, MindSpore 2.8.0 (CPU, Windows) throws a Framework Unexpected Exception with kernel build failure (`Default/Argmin-op0 resize failed`). The same case succeeds in PyTorch and returns an empty-shaped index tensor. Even if this input should be rejected, the current behavior looks like an internal failure rather than a stable user-facing validation error.

## Environment
### Hardware Environment(`Ascend`/`GPU`/`CPU`):
/device cpu

### Software Environment:
- **MindSpore version (source or binary)**: 2.8.0 (binary, pip in conda env)
- **Python version (e.g., Python 3.7.5)**: 3.10.18
- **OS platform and distribution (e.g., Linux Ubuntu 16.04)**: Windows 10 (10.0.26200)
- **GCC/Compiler version (if compiled from source)**: N/A (not compiled from source)

## Describe the current behavior
`x.argmin(axis=0, keepdims=True)` fails for `x.shape == (1, 0, 3)` with RuntimeError:
- Framework Unexpected Exception Raised
- CPU kernel op [Default/Argmin-op0] resize failed

## Describe the expected behavior
One of the following stable behaviors is expected:
1. Return a valid empty output tensor (compatible with common framework behavior, such as PyTorch), or
2. Raise a clear and deterministic user-facing `ValueError` without internal kernel build failure / unexpected-exception message.

## Steps to reproduce the issue
1. Save and run the following script:
```python
import numpy as np
import torch
import mindspore as ms

print("mindspore:", ms.__version__)
print("torch:", torch.__version__)

x_ms = ms.Tensor(np.empty((1, 0, 3), dtype=np.float32))
try:
    y_ms = x_ms.argmin(axis=0, keepdims=True)
    print("MindSpore success:", y_ms.shape)
except Exception as e:
    print("MindSpore failed:", type(e).__name__)
    print(e)

x_torch = torch.empty((1, 0, 3), dtype=torch.float32)
y_torch = torch.argmin(x_torch, dim=0, keepdim=True)
print("PyTorch success:", tuple(y_torch.shape), y_torch.dtype)
```
2. Observe MindSpore error: `Framework Unexpected Exception Raised` and `Default/Argmin-op0 resize failed`.
3. Observe PyTorch succeeds and returns shape `(1, 0, 3)`.

## Related log / screenshot
Key error text:
```text
RuntimeError:
- Framework Unexpected Exception Raised:
This exception is caused by framework's unexpected error.
- Kernel build failed:
CPU kernel op [Default/Argmin-op0] resize failed.
- C++ Call Stack:
mindspore\ccsrc\plugin\cpu\cpu_device_context.cc:547
```

## Special notes for this issue
- Reproduced with script: `ms_pt_test/simple_tests/ms_error_only_candidate_001_tensor_argmin.py`
- This looks like an internal robustness issue for zero-sized dimensions on CPU kernel path.
