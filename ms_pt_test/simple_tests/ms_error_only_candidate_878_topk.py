"""Minimal repro for ms_error_only candidate index 878.

API: mindspore.ops.topk
"""

import traceback

import numpy as np
import torch
import mindspore as ms
from mindspore import ops


def run_mindspore() -> None:
    print("=== MindSpore Repro: mindspore.ops.topk ===")
    x = ms.Tensor(np.empty((1, 1, 1, 0, 5), dtype=np.float32))
    print("input shape:", x.shape, "dtype:", x.dtype)
    try:
        values, indices = ops.topk(x, k=0, dim=-2, largest=False, sorted=False)
        print("unexpected success")
        print("values shape:", values.shape, "indices shape:", indices.shape)
        print(values.asnumpy(), indices.asnumpy())
    except Exception as error:
        print("caught exception type:", type(error).__name__)
        print(error)
        traceback.print_exc()


def run_pytorch() -> None:
    print("=== PyTorch Reference: torch.topk ===")
    x = torch.empty((1, 1, 1, 0, 5), dtype=torch.float32)
    values, indices = torch.topk(x, k=0, dim=-2, largest=False, sorted=False)
    print("success, values shape:", tuple(values.shape), "indices shape:", tuple(indices.shape))


if __name__ == "__main__":
    print("mindspore:", ms.__version__)
    print("torch:", torch.__version__)
    run_mindspore()
    run_pytorch()
