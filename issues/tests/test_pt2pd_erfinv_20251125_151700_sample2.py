import numpy as np
import torch
import paddle


def build_input(shape, dtype, sample_values):
    total_size = int(np.prod(shape))
    flat_array = np.zeros(total_size, dtype=dtype)
    prefix = np.asarray(sample_values, dtype=dtype)
    copy_length = min(prefix.size, total_size)
    flat_array[:copy_length] = prefix[:copy_length]
    return flat_array.reshape(shape)


# 来自样例: llm_enhanced_torch_erfinv_20251125_151700.json_sample2.txt
input_np = build_input(
    shape=(5, 2),
    dtype=np.float32,
    sample_values=[
        -1.2022143602371216,
        -1.300358533859253,
        -0.4211108386516571,
        -0.054816946387290955,
        0.029255155473947525,
        0.2552543878555298,
        -0.10607155412435532,
        0.7177766561508179,
        -0.9769144058227539,
        -0.8542170524597168,
    ],
)

# PyTorch
out_pt = torch.erfinv(torch.tensor(input_np)).detach().numpy().astype(np.float64)

# Paddle
out_pd = paddle.erfinv(paddle.to_tensor(input_np)).numpy().astype(np.float64)

abs_diff = np.abs(out_pt - out_pd)
max_diff = np.nan if np.isnan(abs_diff).all() else np.nanmax(abs_diff)
print(f"Input shape: {input_np.shape}, dtype: {input_np.dtype}")
print(f"Out-of-domain count (|x|>=1): {np.sum(np.abs(input_np) >= 1)}")
print(f"NaN count (PT, PD): {np.isnan(out_pt).sum()}, {np.isnan(out_pd).sum()}")
print(f"Maximum difference: {max_diff}")
