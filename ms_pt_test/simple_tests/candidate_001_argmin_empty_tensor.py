"""Minimal repro for candidate index 1.

From analysis issue_candidates_both_error_samples_20260221_171310.json:
- ms_api: mindspore.Tensor.argmin
- input shape: [2, 0], dtype: float32
- kwargs: axis=None, keepdims=True
"""

import traceback

import numpy as np

import torch
import mindspore as ms


def run_mindspore() -> None:
    print("=== MindSpore Repro ===")
    print("mindspore version:", ms.__version__)
    x_np = np.empty((2, 0), dtype=np.float32)
    x_ms = ms.Tensor(x_np)
    print("input shape:", x_ms.shape, "dtype:", x_ms.dtype)
    try:
        # Candidate sample: axis=None, keepdims=True
        out = x_ms.argmin(axis=None, keepdims=True)
        print("unexpected success, output:", out)
    except Exception as error:
        print("caught exception type:", type(error).__name__)
        print("caught exception message:")
        print(error)
        print("traceback:")
        traceback.print_exc()


def run_pytorch() -> None:
    print("=== PyTorch Reference ===")
    print("torch version:", torch.__version__)
    x_torch = torch.empty((2, 0), dtype=torch.float32)
    print("input shape:", tuple(x_torch.shape), "dtype:", x_torch.dtype)
    try:
        # Candidate sample: dim=None, keepdim=True
        out = torch.argmin(x_torch, dim=None, keepdim=True)
        print("unexpected success, output:", out)
    except Exception as error:
        print("caught exception type:", type(error).__name__)
        print("caught exception message:")
        print(error)


if __name__ == "__main__":
    run_mindspore()
    print()
    run_pytorch()
