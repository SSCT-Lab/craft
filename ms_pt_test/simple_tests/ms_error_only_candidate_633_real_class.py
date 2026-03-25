"""Minimal repro for ms_error_only candidate index 633.

API: mindspore.ops.Real
"""

import traceback

import numpy as np
import torch
import mindspore as ms
from mindspore import ops


def run_mindspore() -> None:
    print("=== MindSpore Repro: mindspore.ops.Real ===")
    x = ms.Tensor(np.empty((2, 0, 3, 1), dtype=np.complex128))
    print("input shape:", x.shape, "dtype:", x.dtype)
    try:
        out = ops.Real()(x)
        print("unexpected success, out shape:", out.shape)
        print(out.asnumpy())
    except Exception as error:
        print("caught exception type:", type(error).__name__)
        print(error)
        traceback.print_exc()


def run_pytorch() -> None:
    print("=== PyTorch Reference: torch.real ===")
    real = torch.empty((2, 0, 3, 1), dtype=torch.float64)
    imag = torch.empty((2, 0, 3, 1), dtype=torch.float64)
    x = torch.complex(real, imag)
    out = torch.real(x)
    print("success, out shape:", tuple(out.shape), "dtype:", out.dtype)


if __name__ == "__main__":
    print("mindspore:", ms.__version__)
    print("torch:", torch.__version__)
    run_mindspore()
    run_pytorch()
