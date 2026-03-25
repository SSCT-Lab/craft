"""Minimal repro for pytorch_error_only candidate index 267.

API: mindspore.ops.cholesky vs torch.linalg.cholesky
Goal: reproduce divergence where MindSpore succeeds but PyTorch reports non-positive-definite input.
"""

import traceback

import numpy as np
import torch
import mindspore as ms
from mindspore import ops


def build_non_positive_definite_matrix() -> np.ndarray:
    # A symmetric matrix that is not positive-definite.
    return np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def run_mindspore() -> None:
    print("=== MindSpore Repro: mindspore.ops.cholesky ===")
    x_np = build_non_positive_definite_matrix()
    x = ms.Tensor(x_np)
    print("input:\n", x_np)
    print("input shape:", x.shape, "dtype:", x.dtype)
    try:
        out = ops.cholesky(x)
        out_np = out.asnumpy()
        print("MindSpore success, output shape:", out_np.shape, "dtype:", out_np.dtype)
        print(out_np)
    except Exception as error:
        print("caught exception type:", type(error).__name__)
        print(error)
        traceback.print_exc()


def run_pytorch() -> None:
    print("=== PyTorch Reference: torch.linalg.cholesky ===")
    x = torch.tensor(build_non_positive_definite_matrix(), dtype=torch.float32)
    print("input:\n", x)
    try:
        out = torch.linalg.cholesky(x)
        print("unexpected success, output shape:", tuple(out.shape), "dtype:", out.dtype)
        print(out)
    except Exception as error:
        print("caught exception type:", type(error).__name__)
        print(error)


if __name__ == "__main__":
    print("mindspore:", ms.__version__)
    print("torch:", torch.__version__)
    run_mindspore()
    run_pytorch()
