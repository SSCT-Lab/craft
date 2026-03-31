"""Minimal repro for ms_error_only candidate index 125.

API: mindspore.ops.BitwiseAnd
"""

# Repro did not succeed
import traceback

import numpy as np
import torch
import mindspore as ms
from mindspore import ops


def run_mindspore() -> None:
    print("=== MindSpore Repro: mindspore.ops.BitwiseAnd ===")
    x = ms.Tensor(np.arange(20, dtype=np.uint8).reshape(4, 5))
    y = ms.Tensor((np.arange(20, dtype=np.uint8).reshape(4, 5) * 3) % 255)
    print("x shape:", x.shape, "dtype:", x.dtype)
    print("y shape:", y.shape, "dtype:", y.dtype)
    try:
        out = ops.BitwiseAnd()(x, y)
        print("unexpected success, out shape:", out.shape)
        print(out.asnumpy())
    except Exception as error:
        print("caught exception type:", type(error).__name__)
        print(error)
        traceback.print_exc()


def run_pytorch() -> None:
    print("=== PyTorch Reference: torch.bitwise_and ===")
    x = torch.arange(20, dtype=torch.uint8).reshape(4, 5)
    y = (torch.arange(20, dtype=torch.uint8).reshape(4, 5) * 3) % 255
    out = torch.bitwise_and(x, y)
    print("success, out shape:", tuple(out.shape), "dtype:", out.dtype)


if __name__ == "__main__":
    print("mindspore:", ms.__version__)
    print("torch:", torch.__version__)
    run_mindspore()
    run_pytorch()
