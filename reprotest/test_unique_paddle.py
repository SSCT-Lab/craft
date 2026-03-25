"""
torch.unique vs paddle.unique 最小复现
样例来源: llm_enhanced_torch_unique_20251202_135202.json (样例4)
"""
import numpy as np

shape = (5,5)
# fill with sample pattern
arr = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
print('输入形状:', arr.shape)

# PyTorch
try:
    import torch
    pt_in = torch.tensor(arr)
    pt_out = torch.unique(pt_in)
    pt_vals = pt_out.numpy()
    print('PyTorch unique 输出:', pt_vals)
except Exception as e:
    print('PyTorch 错误:', e)
    pt_vals = None

# Paddle
try:
    import paddle
    pd_in = paddle.to_tensor(arr)
    pd_out = paddle.unique(pd_in)
    pd_vals = pd_out.numpy()
    print('Paddle unique 输出:', pd_vals)
except Exception as e:
    print('Paddle 错误:', type(e).__name__, e)
    pd_vals = None

print('\n比较:')
if pt_vals is None or pd_vals is None:
    print('无法比较')
else:
    if pt_vals.shape == pd_vals.shape:
        print('最大绝对差:', np.max(np.abs(pt_vals - pd_vals)))
    else:
        print('输出形状不一致', pt_vals.shape, pd_vals.shape)
