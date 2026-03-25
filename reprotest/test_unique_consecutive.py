"""
最小可复现代码：torch.unique_consecutive vs mindspore.mint.unique_consecutive
来源文件: llm_enhanced_torch_unique_consecutive_20251215_183635.json
样例编号: 2

比较两框架在相同输入下 unique_consecutive 的输出和值差异
"""

import numpy as np


def run_case(arr, dtype=np.float64):
    print("="*60)
    print(f"输入 (dtype={dtype}): {arr}")
    print("="*60)

    # PyTorch
    try:
        import torch
        pt_in = torch.tensor(arr, dtype=getattr(torch, 'double' if dtype==np.float64 else 'float'))
        print(f"PyTorch 版本: {torch.__version__}")
        try:
            pt_out = torch.unique_consecutive(pt_in)
        except TypeError:
            pt_out = torch.unique_consecutive(pt_in, return_inverse=False, return_counts=False)
        print(f"PyTorch 输出 type: {type(pt_out)}, value: {pt_out}")
        pt_vals = pt_out.numpy() if hasattr(pt_out, 'numpy') else np.array(pt_out)
        print(f"PyTorch 输出 ndarray: {pt_vals}, dtype: {pt_vals.dtype}")
    except Exception as e:
        print(f"PyTorch 调用错误: {type(e).__name__}: {e}")
        pt_vals = None

    # MindSpore (mint)
    try:
        import mindspore.mint as mint
        import mindspore as ms
        from mindspore import Tensor
        ms.set_context(mode=ms.PYNATIVE_MODE)
        print(f"MindSpore 版本: {ms.__version__}")
        # mint.unique_consecutive API may return a Tensor or tuple, try invoke
        ms_in = Tensor(np.array(arr, dtype=dtype))
        try:
            ms_out = mint.unique_consecutive(ms_in)
        except Exception as e:
            # try with explicit kwargs
            ms_out = mint.unique_consecutive(ms_in, return_counts=False)
        print(f"MindSpore (mint) 输出 type: {type(ms_out)}, value: {ms_out}")
        # normalize to numpy array
        if hasattr(ms_out, 'asnumpy'):
            ms_vals = ms_out.asnumpy()
        elif isinstance(ms_out, tuple) or isinstance(ms_out, list):
            # take first element
            first = ms_out[0]
            ms_vals = first.asnumpy() if hasattr(first, 'asnumpy') else np.array(first)
        else:
            ms_vals = np.array(ms_out)
        print(f"MindSpore 输出 ndarray: {ms_vals}, dtype: {ms_vals.dtype}")
    except Exception as e:
        print(f"MindSpore 调用错误: {type(e).__name__}: {e}")
        ms_vals = None

    # 比较
    print('\n比较:')
    if pt_vals is None or ms_vals is None:
        print('无法比较：至少有一方失败')
        return

    # 对齐形状
    print(f'PyTorch shape: {pt_vals.shape}, MindSpore shape: {ms_vals.shape}')
    if pt_vals.shape == ms_vals.shape:
        diff = np.max(np.abs(pt_vals - ms_vals))
        print(f'最大绝对差: {diff}')
    else:
        print('形状不一致，无法直接计算逐元素差异')
        # 展示各自输出
        print(f'PyTorch 输出: {pt_vals}')
        print(f'MindSpore 输出: {ms_vals}')


if __name__ == '__main__':
    # 用简单的示例验证行为
    case1 = [1.0, 1.0, 2.0, 2.0]
    run_case(case1, dtype=np.float64)

    # 含浮点接近值，检验数值精度影响
    case2 = [1.0, 1.0 + 1e-12, 2.0, 2.0 + 1e-12]
    run_case(case2, dtype=np.float64)

    # 含负值和较大偏差的例子
    case3 = [0.0, -2.780798721451495, -2.780798721451495, 5.0]
    run_case(case3, dtype=np.float64)
