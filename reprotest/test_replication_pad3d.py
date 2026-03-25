"""
torch.nn.ReplicationPad3d vs paddle replicate pad3d 最小复现
样例来源: llm_enhanced_torch_nn_ReplicationPad3d_20251202_132259.json (样例5)
"""
import numpy as np

def create_input(shape, sample_values, dtype=np.float32):
    total = int(np.prod(shape))
    data = [sample_values[i % len(sample_values)] for i in range(total)]
    return np.array(data, dtype=dtype).reshape(shape)

shape = (2,3,4,4,4)
sample_values = [-1.98311651, 0.07937394, -1.84384274, -0.01793666, -0.02141475,
                 -0.07281157, 2.46551728, -0.56019425, -0.24777344, -0.67732817]
arr = create_input(shape, sample_values)
print("输入形状:", arr.shape)

# PyTorch
try:
    import torch
    import torch.nn as nn
    pt_in = torch.tensor(arr)
    padding = [3,3,6,6,1,1]
    module = nn.ReplicationPad3d(padding)
    pt_out = module(pt_in)
    pt_vals = pt_out.numpy()
    print("PyTorch 输出形状:", pt_vals.shape)
    print("PyTorch 输出样本:", pt_vals.flatten()[:10])
except Exception as e:
    print("PyTorch 调用错误:", e)
    pt_vals = None

# Paddle
try:
    import paddle
    import paddle.nn.functional as F
    pd_in = paddle.to_tensor(arr)
    # paddle does not have ReplicationPad3d class in some versions; try functional.pad with mode='replicate'
    # Align pad sequence with PyTorch ReplicationPad3d ordering.
    # PyTorch padding used: [wL,wR,hL,hR,dL,dR] = [3,3,6,6,1,1]
    # Use the same ordering for paddle F.pad so outputs align.
    pad_seq = [3,3,6,6,1,1]
    try:
        pd_out = F.pad(pd_in, pad_seq, mode='replicate')
    except Exception:
        # fallback: use numpy replicate pad for verification
        pd_out = None
    if pd_out is not None:
        pd_vals = pd_out.numpy()
        print("Paddle 输出形状:", pd_vals.shape)
        print("Paddle 输出样本:", pd_vals.flatten()[:10])
    else:
        print("Paddle pad3d 不可用，跳过")
        pd_vals = None
except Exception as e:
    print("Paddle 调用错误:", type(e).__name__, e)
    pd_vals = None

# 比较
print() 
print("比较:")
if pt_vals is None or pd_vals is None:
    print("无法比较：至少一方失败或不可用")
else:
    if pt_vals.shape == pd_vals.shape:
        diff = np.max(np.abs(pt_vals - pd_vals))
        print("最大绝对差:", diff)
    else:
        print("形状不一致")
