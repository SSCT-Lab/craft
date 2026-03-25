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


# 来自样例: llm_enhanced_torch_erfinv_20251215_184626.json_sample1.txt
input_np = build_input(
    shape=(2, 3, 4, 5),
    dtype=np.float64,
    sample_values=[
        -0.6813052710748467,
        -0.8495579314494206,
        -1.2201734282525922,
        0.0029796246678243966,
        0.9159602157960965,
        -0.6011707561855925,
        0.8630660602520888,
        -0.6426682542886082,
        -1.7247004652368227,
        -0.961864163197681,
    ],
)

# PyTorch
out_pt = torch.erfinv(torch.tensor(input_np)).detach().numpy().astype(np.float64)

# MindSpore
input_ms = mindspore.Tensor(input_np)
if hasattr(mindspore, "mint") and hasattr(mindspore.mint, "erfinv"):
    out_ms = mindspore.mint.erfinv(input_ms).asnumpy().astype(np.float64)
else:
    out_ms = mindspore.ops.erfinv(input_ms).asnumpy().astype(np.float64)

abs_diff = np.abs(out_pt - out_ms)
max_diff = np.nan if np.isnan(abs_diff).all() else np.nanmax(abs_diff)
print(f"Input shape: {input_np.shape}, dtype: {input_np.dtype}")
print(f"Out-of-domain count (|x|>=1): {np.sum(np.abs(input_np) >= 1)}")
print(f"NaN count (PT, MS): {np.isnan(out_pt).sum()}, {np.isnan(out_ms).sum()}")
print(f"Maximum difference: {max_diff}")
