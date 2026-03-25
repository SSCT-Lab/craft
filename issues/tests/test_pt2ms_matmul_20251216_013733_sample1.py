import numpy as np
import torch
import mindspore

input_np = np.array(
    [
        -2.3755927,
        -0.8214182,
        -0.48699182,
        1.4049829,
        1.5832627,
        0.91923374,
        -0.8661216,
        -0.8675243,
        -1.1785852,
        -0.6409078,
    ],
    dtype=np.float32,
).reshape(2, 5)

other_np = np.array(
    [
        0.9517129,
        0.25526756,
        -1.2263191,
        -0.58238614,
        0.8957569,
        0.13603328,
        -2.2835088,
        -0.6570328,
        2.260117,
        0.6589172,
    ],
    dtype=np.float32,
).reshape(5, 2)

# PyTorch
out_pt = torch.matmul(torch.tensor(input_np), torch.tensor(other_np)).detach().numpy()

# MindSpore
mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
input_ms = mindspore.Tensor(input_np)
other_ms = mindspore.Tensor(other_np)
if hasattr(mindspore, "mint") and hasattr(mindspore.mint, "matmul"):
    out_ms = mindspore.mint.matmul(input_ms, other_ms).asnumpy()
else:
    out_ms = mindspore.ops.matmul(input_ms, other_ms).asnumpy()

max_diff = np.max(np.abs(out_pt - out_ms))
print(f"PyTorch output shape: {out_pt.shape}")
print(f"MindSpore output shape: {out_ms.shape}")
print(f"Maximum difference: {max_diff}")
