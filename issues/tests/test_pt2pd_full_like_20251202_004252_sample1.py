import numpy as np
import torch
import paddle

# 来自样例: llm_enhanced_torch_full_like_20251202_004252.json_sample1.txt
input_np = np.array(
    [-2, -10, 3, -9, -4, 5, 4, -3, 6, 8],
    dtype=np.int64,
).reshape(2, 1, 1, 1, 5)

fill_value = 9223372036854775807

# PyTorch
out_pt = torch.full_like(torch.tensor(input_np), fill_value=fill_value).numpy()

# Paddle
out_pd = paddle.full_like(paddle.to_tensor(input_np), fill_value=fill_value).numpy()

max_diff = np.max(np.abs(out_pt.astype(np.int64) - out_pd.astype(np.int64)))
print(f"PyTorch output dtype: {out_pt.dtype}")
print(f"Paddle output dtype: {out_pd.dtype}")
print(f"PyTorch first value: {out_pt.flatten()[0]}")
print(f"Paddle first value: {out_pd.flatten()[0]}")
print(f"Maximum difference: {max_diff}")
