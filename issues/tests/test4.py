import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(20, 16, 10, 50, 100).astype(np.float32)
weight_np = np.random.randn(33, 16, 3, 5, 2).astype(np.float32)
bias_np = np.random.randn(33).astype(np.float32)

out_pt = torch.nn.functional.conv3d(
    torch.tensor(input_np), torch.tensor(weight_np), torch.tensor(bias_np),
    stride=[2, 1, 1], padding=[4, 2, 0], dilation=[1, 1, 1], groups=1
)
out_pd = paddle.nn.functional.conv3d(
    paddle.to_tensor(input_np), paddle.to_tensor(weight_np), paddle.to_tensor(bias_np),
    stride=[2, 1, 1], padding=[4, 2, 0], dilation=[1, 1, 1], groups=1
)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")