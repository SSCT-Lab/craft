import numpy as np
import torch
import mindspore


def build_input(shape, dtype, sample_values):
    total_size = int(np.prod(shape))
    flat_array = np.zeros(total_size, dtype=dtype)
    prefix = np.asarray(sample_values, dtype=dtype)
    copy_length = min(prefix.size, total_size)
    flat_array[:copy_length] = prefix[:copy_length]
    return flat_array.reshape(shape)


# 来自样例: llm_enhanced_torch_tanh_20251216_010955.json_sample1.txt
input_np = build_input(
    shape=(64, 3, 28, 28),
    dtype=np.float32,
    sample_values=[
        -0.015008660033345222,
        0.3706440329551697,
        -1.0460474491119385,
        0.34985223412513733,
        0.5233860611915588,
        0.32309994101524353,
        -1.4455026388168335,
        -1.6050543785095215,
        0.13055920600891113,
        0.29976707696914673,
    ],
)

# PyTorch
out_pt = torch.tanh(torch.tensor(input_np)).detach().numpy().astype(np.float64)

# MindSpore
mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
out_ms = mindspore.ops.tanh(mindspore.Tensor(input_np)).asnumpy().astype(np.float64)

abs_diff = np.abs(out_pt - out_ms)
max_diff = np.nan if np.isnan(abs_diff).all() else np.nanmax(abs_diff)
print(f"Input shape: {input_np.shape}, dtype: {input_np.dtype}")
print(f"PyTorch output shape: {out_pt.shape}")
print(f"MindSpore output shape: {out_ms.shape}")
print(f"Maximum difference: {max_diff}")
