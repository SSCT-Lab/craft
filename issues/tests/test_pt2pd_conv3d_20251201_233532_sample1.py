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


# 来自样例: llm_enhanced_torch_nn_functional_conv3d_20251201_233532.json_sample1.txt
input_np = build_input(
    shape=(20, 16, 10, 50, 100),
    dtype=np.float32,
    sample_values=[
        0.7752637267112732,
        0.8845512866973877,
        -0.017568811774253845,
        -0.7523747086524963,
        -0.981509268283844,
        0.2940584719181061,
        0.03562941402196884,
        -0.7505152225494385,
        -0.4497143030166626,
        -0.49824902415275574,
    ],
)

weight_np = build_input(
    shape=(33, 16, 3, 5, 2),
    dtype=np.float32,
    sample_values=[
        -0.09101136028766632,
        -0.6142028570175171,
        -1.0077792406082153,
        0.03650888055562973,
        -0.33617982268333435,
        0.5204266905784607,
        -0.8811998963356018,
        1.0933336019515991,
        -0.1312147080898285,
        0.8963345885276794,
    ],
)

bias_np = build_input(
    shape=(33,),
    dtype=np.float32,
    sample_values=[
        0.07692712545394897,
        4.147119998931885,
        0.5593327879905701,
        -0.17988839745521545,
        -0.8006935715675354,
        -1.1780678033828735,
        0.4137747883796692,
        -0.6542391777038574,
        0.21862509846687317,
        0.5949686765670776,
    ],
)

# PyTorch
out_pt = torch.nn.functional.conv3d(
    torch.tensor(input_np),
    torch.tensor(weight_np),
    torch.tensor(bias_np),
    stride=[2, 1, 1],
    padding=[4, 2, 0],
    dilation=[1, 1, 1],
    groups=1,
).detach().numpy().astype(np.float64)

# Paddle
out_pd = paddle.nn.functional.conv3d(
    paddle.to_tensor(input_np),
    paddle.to_tensor(weight_np),
    paddle.to_tensor(bias_np),
    stride=[2, 1, 1],
    padding=[4, 2, 0],
    dilation=[1, 1, 1],
    groups=1,
).numpy().astype(np.float64)

abs_diff = np.abs(out_pt - out_pd)
max_diff = np.nan if np.isnan(abs_diff).all() else np.nanmax(abs_diff)
print(f"Input shape: {input_np.shape}, dtype: {input_np.dtype}")
print(f"Weight shape: {weight_np.shape}, Bias shape: {bias_np.shape}")
print(f"PyTorch output shape: {out_pt.shape}")
print(f"Paddle output shape: {out_pd.shape}")
print(f"Maximum difference: {max_diff}")
