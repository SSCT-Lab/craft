import numpy as np
import torch
import mindspore
import mindspore.nn as ms_nn

# 来自样例: llm_enhanced_torch_nn_LPPool1d_20251215_193350.json_sample1.txt
input_np = np.array(
    [
        0.63573098,
        0.81810886,
        -0.96552032,
        -0.99548149,
        0.22021013,
        -0.02396944,
    ],
    dtype=np.float32,
).reshape(2, 1, 3)

# PyTorch
pt_layer = torch.nn.LPPool1d(norm_type=1.5, kernel_size=3, stride=1)
out_pt = pt_layer(torch.tensor(input_np)).detach().numpy()

# MindSpore
mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
ms_layer = ms_nn.LPPool1d(norm_type=1.5, kernel_size=3, stride=1)
out_ms = ms_layer(mindspore.Tensor(input_np)).asnumpy()

abs_diff = np.abs(out_pt - out_ms)
max_diff = np.nan if np.isnan(abs_diff).all() else np.nanmax(abs_diff)
print(f"PyTorch output: {out_pt}")
print(f"MindSpore output: {out_ms}")
print(f"NaN count (PT, MS): {np.isnan(out_pt).sum()}, {np.isnan(out_ms).sum()}")
print(f"Maximum difference: {max_diff}")
