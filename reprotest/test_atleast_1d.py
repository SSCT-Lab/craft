"""
torch.atleast_1d vs paddle.atleast_1d 最小复现
样例来源: llm_enhanced_torch_atleast_1d_20251202_132406.json (样例3)
"""
import numpy as np

# create tensors: float32, int64, bool
import torch

a = np.array([1.0], dtype=np.float32)
b = np.array([1], dtype=np.int64)
c = np.array([True], dtype=bool)

print('输入: a', a.shape, 'b', b.shape, 'c', c.shape)

# PyTorch
try:
    pt_a = torch.atleast_1d(torch.tensor(a))
    pt_b = torch.atleast_1d(torch.tensor(b))
    pt_c = torch.atleast_1d(torch.tensor(c))
    print('PyTorch outputs shapes:', pt_a.shape, pt_b.shape, pt_c.shape)
except Exception as e:
    print('PyTorch 错误:', e)

# Paddle
try:
    import paddle
    try:
        pd_a = paddle.atleast_1d(paddle.to_tensor(a))
        pd_b = paddle.atleast_1d(paddle.to_tensor(b))
        pd_c = paddle.atleast_1d(paddle.to_tensor(c))
        print('Paddle outputs shapes:', pd_a.shape, pd_b.shape, pd_c.shape)
    except Exception:
        # fallback: ensure numpy like behavior
        pd_a = paddle.to_tensor(a).unsqueeze(0) if paddle.to_tensor(a).ndim==0 else paddle.to_tensor(a)
        pd_b = paddle.to_tensor(b).unsqueeze(0) if paddle.to_tensor(b).ndim==0 else paddle.to_tensor(b)
        pd_c = paddle.to_tensor(c).unsqueeze(0) if paddle.to_tensor(c).ndim==0 else paddle.to_tensor(c)
        print('Paddle fallback outputs shapes:', pd_a.shape, pd_b.shape, pd_c.shape)
except Exception as e:
    print('Paddle 错误:', type(e).__name__, e)

print('\n比较:')
# convert to numpy and compare shapes
try:
    import numpy as _np
    pt_shapes = (pt_a.numpy().shape, pt_b.numpy().shape, pt_c.numpy().shape)
    pd_shapes = (pd_a.numpy().shape, pd_b.numpy().shape, pd_c.numpy().shape)
    print('PyTorch shapes', pt_shapes)
    print('Paddle shapes', pd_shapes)
except Exception as e:
    print('比较失败:', e)
