"""Minimal repro for ms_error_only candidate index 180.

API: mindspore.ops.Conv3DTranspose
"""
# 未复现成功
import traceback

import numpy as np
import torch
import mindspore as ms
from mindspore import ops


def run_mindspore() -> None:
    print("=== MindSpore Repro: mindspore.ops.Conv3DTranspose ===")
    x = ms.Tensor(np.ones((1, 2, 3, 3, 3), dtype=np.float32))
    w = ms.Tensor(np.ones((2, 2, 2, 2, 2), dtype=np.float32))
    print("x shape:", x.shape, "w shape:", w.shape)
    try:
        conv_t = ops.Conv3DTranspose(
            in_channel=2,
            out_channel=2,
            kernel_size=2,
            pad_mode="pad",
            pad=1,
            stride=1,
            dilation=1,
            group=1,
        )
        out = conv_t(x, w)
        print("unexpected success, out shape:", out.shape)
        print(out.asnumpy())
    except Exception as error:
        print("caught exception type:", type(error).__name__)
        print(error)
        traceback.print_exc()


def run_pytorch() -> None:
    print("=== PyTorch Reference: torch.nn.ConvTranspose3d ===")
    layer = torch.nn.ConvTranspose3d(
        in_channels=2,
        out_channels=2,
        kernel_size=2,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.ones_like(layer.weight))
    x = torch.ones((1, 2, 3, 3, 3), dtype=torch.float32)
    out = layer(x)
    print("success, out shape:", tuple(out.shape), "dtype:", out.dtype)


if __name__ == "__main__":
    print("mindspore:", ms.__version__)
    print("torch:", torch.__version__)
    run_mindspore()
    run_pytorch()
