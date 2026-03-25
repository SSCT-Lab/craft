"""Minimal repro for ms_error_only candidate index 269.

API: mindspore.ops.Elu
"""

import traceback

import numpy as np
import torch
import mindspore as ms
from mindspore import ops


def run_mindspore() -> None:
    print("=== MindSpore Repro: mindspore.ops.Elu ===")
    x = ms.Tensor(np.linspace(-1.0, 1.0, num=24, dtype=np.float32).reshape(1, 2, 3, 4))
    try:
        out = ops.Elu(alpha=0.001)(x)
        print("unexpected success, out shape:", out.shape)
        print(out.asnumpy())
    except Exception as error:
        print("caught exception type:", type(error).__name__)
        print(error)
        traceback.print_exc()


def run_pytorch() -> None:
    print("=== PyTorch Reference: torch.nn.ELU ===")
    x = torch.linspace(-1.0, 1.0, steps=24, dtype=torch.float32).reshape(1, 2, 3, 4)
    out = torch.nn.ELU(alpha=0.001)(x)
    print("success, out shape:", tuple(out.shape), "dtype:", out.dtype)


if __name__ == "__main__":
    print("mindspore:", ms.__version__)
    print("torch:", torch.__version__)
    run_mindspore()
    run_pytorch()
