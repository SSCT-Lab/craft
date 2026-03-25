"""
torch.nn.functional.adaptive_max_pool2d vs paddle.nn.functional.adaptive_max_pool2d
样例来源: llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json (样例5)
"""
import numpy as np

shape = (1,32,8,5)
sample_values = list(np.linspace(-1.0,1.0,10))
arr = np.tile(np.array(sample_values, dtype=np.float64), int(np.prod(shape)/10)+1)[:int(np.prod(shape))].reshape(shape)
print('输入形状:', arr.shape)

# PyTorch
try:
    import torch
    import torch.nn.functional as F
    pt_in = torch.tensor(arr, dtype=torch.float64)
    out_size = (6,4)
    pt_out = F.adaptive_max_pool2d(pt_in, out_size)
    pt_vals = pt_out.numpy()
    print('PyTorch 输出形状:', pt_vals.shape)
    print('PyTorch 输出样本:', pt_vals.flatten()[:6])
except Exception as e:
    print('PyTorch 错误:', e)
    pt_vals = None

# Paddle
try:
    import paddle
    import paddle.nn.functional as Fpd
    pd_in = paddle.to_tensor(arr)
    pd_out = Fpd.adaptive_max_pool2d(pd_in, output_size=[6,4])
    pd_vals = pd_out.numpy()
    print('Paddle 输出形状:', pd_vals.shape)
    print('Paddle 输出样本:', pd_vals.flatten()[:6])
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
        print('形状不一致', pt_vals.shape, pd_vals.shape)
