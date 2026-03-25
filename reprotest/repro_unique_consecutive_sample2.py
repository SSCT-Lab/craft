"""
复现样例 2：torch.unique_consecutive vs mindspore.mint.unique_consecutive
使用输入 shape=[4], dtype=float64，包含值与原始差异提示相关的数值。
"""

import numpy as np

# 输入（与报告中差异值对应）
arr = np.array([0.0, -2.780798721451495, -2.780798721451495, 5.0], dtype=np.float64)

print("输入:", arr)

# PyTorch
try:
    import torch
    pt_in = torch.tensor(arr, dtype=torch.float64)
    print(f"PyTorch 版本: {torch.__version__}")
    pt_out = torch.unique_consecutive(pt_in)
    pt_vals = pt_out.numpy()
    print("PyTorch unique_consecutive 输出:", pt_vals)
except Exception as e:
    print("PyTorch 调用错误:", type(e).__name__, e)
    pt_vals = None

# MindSpore (mint)
try:
    import mindspore as ms
    import mindspore.mint as mint
    from mindspore import Tensor
    ms.set_context(mode=ms.PYNATIVE_MODE)
    print(f"MindSpore 版本: {ms.__version__}")
    ms_in = Tensor(arr)
    ms_out = mint.unique_consecutive(ms_in)
    # 如果返回 tuple，取第一个
    if isinstance(ms_out, (tuple, list)):
        first = ms_out[0]
    else:
        first = ms_out
    ms_vals = first.asnumpy()
    print("MindSpore mint.unique_consecutive 输出:", ms_vals)
except Exception as e:
    print("MindSpore 调用错误:", type(e).__name__, e)
    ms_vals = None

# 比较
print()
print("比较结果:")
if pt_vals is None or ms_vals is None:
    print("无法比较：至少一方执行失败")
else:
    print("PyTorch 输出形状:", pt_vals.shape, "MindSpore 输出形状:", ms_vals.shape)
    if pt_vals.shape == ms_vals.shape:
        diff = np.max(np.abs(pt_vals - ms_vals))
        print("最大绝对差:", diff)
    else:
        print("形状不一致，PyTorch:", pt_vals, "MindSpore:", ms_vals)
