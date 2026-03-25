"""Minimal repro for ms_error_only candidate index 1.

API: mindspore.Tensor.argmin
"""

import traceback

import numpy as np
import torch
import mindspore as ms


def run_mindspore() -> None:
    print("=== MindSpore Repro: mindspore.Tensor.argmin ===")
    x = ms.Tensor(np.empty((1, 0, 3), dtype=np.float32))
    print("input shape:", x.shape, "dtype:", x.dtype)
    try:
        out = x.argmin(axis=0, keepdims=True)
        print("unexpected success, out shape:", out.shape)
        print(out.asnumpy())
    except Exception as error:
        print("caught exception type:", type(error).__name__)
        print(error)
        traceback.print_exc()


def run_pytorch() -> None:
    print("=== PyTorch Reference: torch.Tensor.argmin ===")
    x = torch.empty((1, 0, 3), dtype=torch.float32)
    out = torch.argmin(x, dim=0, keepdim=True)
    print("success, out shape:", tuple(out.shape), "dtype:", out.dtype)


if __name__ == "__main__":
    print("mindspore:", ms.__version__)
    print("torch:", torch.__version__)
    run_mindspore()
    run_pytorch()
