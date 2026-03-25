Add a title*:
mindspore.ops.cdist with broadcastable batch dimensions throws Framework Unexpected Exception on CPU

Add a description:
`mindspore.ops.cdist` fails when input batch dimensions are broadcastable but not exactly equal (example: `(2, 5, 8)` vs `(1, 7, 8)`). The runtime reports a Framework Unexpected Exception and kernel resize failure on CPU. In PyTorch, this case succeeds and returns output shape `(2, 5, 7)`. If broadcasting is unsupported by design, the error should still be a clear user-facing validation message instead of an internal unexpected exception.

## Environment
### Hardware Environment(`Ascend`/`GPU`/`CPU`):
/device cpu

### Software Environment:
- **MindSpore version (source or binary)**: 2.8.0 (binary, pip in conda env)
- **Python version (e.g., Python 3.7.5)**: 3.10.18
- **OS platform and distribution (e.g., Linux Ubuntu 16.04)**: Windows 10 (10.0.26200)
- **GCC/Compiler version (if compiled from source)**: N/A (not compiled from source)

## Describe the current behavior
`ops.cdist(x1, x2, p=0.0)` where:
- `x1.shape = (2, 5, 8)`
- `x2.shape = (1, 7, 8)`
throws RuntimeError with:
- Framework Unexpected Exception Raised
- CPU kernel op [Default/Cdist-op0] resize failed

## Describe the expected behavior
Expected behavior should be one of:
1. Support broadcastable batch dimensions and return output (similar to PyTorch behavior), or
2. Raise a stable and explicit user-facing `ValueError` stating unsupported shape rules.

In either case, internal "Framework Unexpected Exception" should be avoided.

## Steps to reproduce the issue
1. Run:
```python
import numpy as np
import torch
import mindspore as ms
from mindspore import ops

print("mindspore:", ms.__version__)
print("torch:", torch.__version__)

x1_ms = ms.Tensor(np.ones((2, 5, 8), dtype=np.float32))
x2_ms = ms.Tensor(np.ones((1, 7, 8), dtype=np.float32))
try:
    y_ms = ops.cdist(x1_ms, x2_ms, p=0.0)
    print("MindSpore success:", y_ms.shape)
except Exception as e:
    print("MindSpore failed:", type(e).__name__)
    print(e)

x1_t = torch.ones((2, 5, 8), dtype=torch.float32)
x2_t = torch.ones((1, 7, 8), dtype=torch.float32)
y_t = torch.cdist(x1_t, x2_t, p=0.0, compute_mode="donot_use_mm_for_euclid_dist")
print("PyTorch success:", tuple(y_t.shape), y_t.dtype)
```
2. Observe MindSpore reports kernel resize failure and unexpected exception.
3. Observe PyTorch succeeds with shape `(2, 5, 7)`.

## Related log / screenshot
Key logs:
```text
[ERROR] ... CdistCpuKernelMod::Resize] invalid input shape, the batch shape of input0 must be the same as the shape of input1
RuntimeError:
- Framework Unexpected Exception Raised:
- Kernel build failed:
CPU kernel op [Default/Cdist-op0] resize failed.
- C++ Call Stack:
mindspore\ccsrc\plugin\cpu\cpu_device_context.cc:547
```

## Special notes for this issue
- Reproduced with script: `ms_pt_test/simple_tests/ms_error_only_candidate_851_cdist.py`
- This issue is marked as worth reporting mainly because of internal unexpected-exception behavior.
