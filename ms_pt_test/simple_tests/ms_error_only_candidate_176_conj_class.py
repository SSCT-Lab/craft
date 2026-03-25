"""Minimal repro for ms_error_only candidate index 176.

API: mindspore.ops.Conj
"""

import traceback

import numpy as np
import torch
import mindspore as ms
from mindspore import ops


def run_mindspore() -> None:
    print("=== MindSpore Repro: mindspore.ops.Conj ===")
    x_np = np.empty((0, 2, 3), dtype=np.complex64)
    x = ms.Tensor(x_np)
    print("input shape:", x.shape, "dtype:", x.dtype)
    try:
        out = ops.Conj()(x)
        print("unexpected success, out shape:", out.shape)
        print(out.asnumpy())
    except Exception as error:
        print("caught exception type:", type(error).__name__)
        print(error)
        traceback.print_exc()


def run_pytorch() -> None:
    print("=== PyTorch Reference: torch.conj ===")
    real = torch.empty((0, 2, 3), dtype=torch.float32)
    imag = torch.empty((0, 2, 3), dtype=torch.float32)
    x = torch.complex(real, imag)
    out = torch.conj(x)
    print("success, out shape:", tuple(out.shape), "dtype:", out.dtype)


if __name__ == "__main__":
    print("mindspore:", ms.__version__)
    print("torch:", torch.__version__)
    run_mindspore()
    run_pytorch()
