"""
torch.unique_consecutive vs paddle.unique_consecutive 最小复现
样例来源: llm_enhanced_torch_unique_consecutive_20251125_142753.json (样例3)
"""
import numpy as np

arr = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float64)
print("输入:", arr)

# PyTorch
try:
    import torch
    pt_in = torch.tensor(arr, dtype=torch.float64)
    pt_out = torch.unique_consecutive(pt_in)
    pt_vals = pt_out.numpy()
    print("PyTorch 输出:", pt_vals)
except Exception as e:
    print("PyTorch 调用错误:", e)
    pt_vals = None

# Paddle
try:
    import paddle
    # try paddle.unique_consecutive if exists
    try:
        pd_out = paddle.unique_consecutive(paddle.to_tensor(arr))
    except Exception:
        # fallback to paddle.unique (not consecutive)
        pd_out = paddle.unique(paddle.to_tensor(arr))
    pd_vals = pd_out.numpy() if hasattr(pd_out, 'numpy') else pd_out
    print("Paddle 输出:", pd_vals)
except Exception as e:
    print("Paddle 调用错误:", type(e).__name__, e)
    pd_vals = None

# 比较
print() 
print("比较:")
if pt_vals is None or pd_vals is None:
    print("无法比较：至少一方失败")
else:
    print("pt shape", pt_vals.shape, "pd shape", pd_vals.shape)
    if pt_vals.shape == pd_vals.shape:
        print("最大绝对差:", np.max(np.abs(pt_vals - pd_vals)))
    else:
        print("输出不一致\nPyTorch:", pt_vals, "\nPaddle:", pd_vals)
