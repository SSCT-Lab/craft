"""
torch.nn.ReplicationPad2d vs paddle replicate pad 最小复现
样例来源: llm_enhanced_torch_nn_ReplicationPad2d_20251202_004340.json (样例1)
"""
import numpy as np

def create_input(shape, sample_values, dtype=np.float32):
    total = int(np.prod(shape))
    data = [sample_values[i % len(sample_values)] for i in range(total)]
    return np.array(data, dtype=dtype).reshape(shape)

shape = (2,3,4,4)
sample_values = [-0.47856003, -1.88550234, -0.64987302, -0.24165535, -1.62113285,
                 3.17547703, 0.6390152, 1.88449156, -0.53552455, 0.47458777]
arr = create_input(shape, sample_values)
print("输入形状:", arr.shape)

# PyTorch
try:
    import torch
    import torch.nn as nn
    pt_in = torch.tensor(arr)
    pad = 2
    module = nn.ReplicationPad2d(pad)
    pt_out = module(pt_in)
    pt_vals = pt_out.numpy()
    print("PyTorch 输出形状:", pt_vals.shape)
    print("PyTorch 输出样本:", pt_vals.flatten()[:10])
except Exception as e:
    print("PyTorch 调用错误:", e)
    pt_vals = None

# Paddle: 尝试使用 paddle.nn.Pad2D 或 paddle.nn.functional.pad
try:
    import paddle
    import paddle.nn.functional as F
    pd_in = paddle.to_tensor(arr)
    try:
        pd_module = paddle.nn.Pad2D(padding=pad, mode='replicate')
        pd_out = pd_module(pd_in)
    except Exception:
        # fallback to functional.pad with pad sequence [pad_left, pad_right, pad_top, pad_bottom]
        pad_seq = [pad, pad, pad, pad]
        pd_out = F.pad(pd_in, pad_seq, mode='replicate')
    pd_vals = pd_out.numpy()
    print("Paddle 输出形状:", pd_vals.shape)
    print("Paddle 输出样本:", pd_vals.flatten()[:10])
except Exception as e:
    print("Paddle 调用错误:", type(e).__name__, e)
    pd_vals = None

# 比较
print() 
print("比较:")
if pt_vals is None or pd_vals is None:
    print("无法比较：至少一方失败")
else:
    print("PyTorch shape", pt_vals.shape, "Paddle shape", pd_vals.shape)
    if pt_vals.shape == pd_vals.shape:
        diff = np.max(np.abs(pt_vals - pd_vals))
        print("最大绝对差:", diff)
    else:
        print("形状不一致，PyTorch:\n", pt_vals)
        print("Paddle:\n", pd_vals)
