"""Minimal repro for ms_error_only candidate index 851.

API: mindspore.ops.cdist
"""

import traceback

import numpy as np
import torch
import mindspore as ms
from mindspore import ops


def run_mindspore() -> None:
    print("=== MindSpore Repro: mindspore.ops.cdist ===")
    x1 = ms.Tensor(np.ones((2, 5, 8), dtype=np.float32))
    x2 = ms.Tensor(np.ones((1, 7, 8), dtype=np.float32))
    try:
        out = ops.cdist(x1, x2, p=0.0)
        out_np = out.asnumpy()
        print("unexpected success, out shape:", out_np.shape)
        print(out_np)
    except Exception as error:
        print("caught exception type:", type(error).__name__)
        print(error)
        traceback.print_exc()


def run_pytorch() -> None:
    print("=== PyTorch Reference: torch.cdist ===")
    x1 = torch.ones((2, 5, 8), dtype=torch.float32)
    x2 = torch.ones((1, 7, 8), dtype=torch.float32)
    out = torch.cdist(x1, x2, p=0.0, compute_mode="donot_use_mm_for_euclid_dist")
    print("success, out shape:", tuple(out.shape), "dtype:", out.dtype)


if __name__ == "__main__":
    print("mindspore:", ms.__version__)
    print("torch:", torch.__version__)
    run_mindspore()
    run_pytorch()
