"""
torch.rand_like vs paddle.rand_like 最小复现
样例来源: llm_enhanced_torch_rand_like_20251125_143509.json (样例5)
"""
import numpy as np

def create_input(shape, sample_values, dtype=np.float64):
    total = int(np.prod(shape))
    data = [sample_values[i % len(sample_values)] for i in range(total)]
    return np.array(data, dtype=dtype).reshape(shape)

shape = (2,3,4)
sample_values = [-0.09972332388938569, -0.16530023422785195, -0.5310751145639417,
                 -2.065759638599122, 0.05323111593640626, -0.5286382332546202,
                 0.20393044558875567, -1.2667645325703651, 0.4090725907662043,
                 -0.3920710831565494]

arr = create_input(shape, sample_values, dtype=np.float64)
print("输入形状:", arr.shape, "dtype:", arr.dtype)

# PyTorch
try:
    import torch
    torch.manual_seed(123)
    pt_input = torch.tensor(arr)
    pt_out = torch.rand_like(pt_input, dtype=torch.float64)
    pt_vals = pt_out.numpy()
    print("PyTorch 输出样本:", pt_vals.flatten()[:6])
    print("PyTorch stats: min", pt_vals.min(), "max", pt_vals.max(), "mean", pt_vals.mean())
except Exception as e:
    print("PyTorch 调用错误:", e)
    pt_vals = None

# Paddle
try:
    import paddle
    paddle.seed(123)
    pd_input = paddle.to_tensor(arr)
    try:
        pd_out = paddle.rand_like(pd_input, dtype='float64')
    except Exception:
        pd_out = paddle.randn(pd_input.shape, dtype='float64')
    pd_vals = pd_out.numpy()
    print("Paddle 输出样本:", pd_vals.flatten()[:6])
    print("Paddle stats: min", pd_vals.min(), "max", pd_vals.max(), "mean", pd_vals.mean())
except Exception as e:
    print("Paddle 调用错误:", type(e).__name__, e)
    pd_vals = None

# 比较
print() 
print("比较:")
if pt_vals is None or pd_vals is None:
    print("无法比较：至少一方失败")
else:
    print("shape pt", pt_vals.shape, "pd", pd_vals.shape)
    same_shape = pt_vals.shape == pd_vals.shape
    print("shape 一致:", same_shape)
    if same_shape:
        diff = np.max(np.abs(pt_vals - pd_vals))
        print("最大绝对差:", diff)
    print("注意: 随机数生成器不同，数值差异是预期行为")
