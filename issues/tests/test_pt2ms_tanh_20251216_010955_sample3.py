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


# 来自样例: llm_enhanced_torch_tanh_20251216_010955.json_sample3.txt
input_np = build_input(
    shape=(4, 768),
    dtype=np.float32,
    sample_values=[
        1.1228810548782349,
        -0.10368899255990982,
        0.4753471314907074,
        -1.2070391178131104,
        0.1820811629295349,
        0.15206992626190186,
        -0.40252137184143066,
        -2.916238307952881,
        -1.167243242263794,
        -1.4618971347808838,
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
