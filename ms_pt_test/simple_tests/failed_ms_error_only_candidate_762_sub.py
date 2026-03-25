"""Minimal repro for ms_error_only candidate index 762.

API: mindspore.ops.Sub
"""
# 未复现成功
import traceback

import numpy as np
import torch
import mindspore as ms
from mindspore import ops


def run_mindspore() -> None:
    print("=== MindSpore Repro: mindspore.ops.Sub ===")
    x = ms.Tensor(np.arange(8, dtype=np.int32))
    y = ms.Tensor(np.array(3, dtype=np.int32))
    print("x shape:", x.shape, "y shape:", y.shape)
    try:
        out = ops.Sub()(x, y)
        print("unexpected success, out shape:", out.shape)
        print(out.asnumpy())
    except Exception as error:
        print("caught exception type:", type(error).__name__)
        print(error)
        traceback.print_exc()


def run_pytorch() -> None:
    print("=== PyTorch Reference: torch.sub ===")
    x = torch.arange(8, dtype=torch.int32)
    y = torch.tensor(3, dtype=torch.int32)
    out = torch.sub(x, y)
    print("success, out shape:", tuple(out.shape), "dtype:", out.dtype)


if __name__ == "__main__":
    print("mindspore:", ms.__version__)
    print("torch:", torch.__version__)
    run_mindspore()
    run_pytorch()
