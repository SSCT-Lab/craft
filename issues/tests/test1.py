import numpy as np
import torch
import mindspore

# 使用随机种子生成与原始测试一致的输入
np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 5).astype(np.float64)

# PyTorch 执行
input_pt = torch.tensor(input_np)
out_pt = torch.erfinv(input_pt)

# MindSpore 执行
input_ms = mindspore.Tensor(input_np)
out_ms = mindspore.mint.erfinv(input_ms)

# 比较结果
pt_np = out_pt.numpy()
ms_np = out_ms.asnumpy()
oob_count = np.sum(np.abs(input_np) >= 1)
max_diff = np.max(np.abs(pt_np - ms_np))
print(f"Input shape: {input_np.shape}")
print(f"Number of out-of-domain values (|x| >= 1): {oob_count}")
print(f"Maximum difference: {max_diff}")