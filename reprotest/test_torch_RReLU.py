"""
最小可复现代码：torch.nn.RReLU vs mindspore.nn.RReLU
来源文件: llm_enhanced_torch_nn_RReLU_20251216_010240.json
样例编号: 1, 2

说明：保持输入和参数完全一致，打印两框架输出与比较结果。
"""

import numpy as np


def create_input(shape, sample_values, dtype=np.float32):
    total = int(np.prod(shape))
    data = [sample_values[i % len(sample_values)] for i in range(total)]
    return np.array(data, dtype=dtype).reshape(shape)


def run_test(name, shape, sample_values, lower, upper, seed=42):
    print("=" * 60)
    print(f"测试: {name}")
    print("=" * 60)
    print(f"shape={shape}, lower={lower}, upper={upper}, seed={seed}")

    input_data = create_input(shape, sample_values)
    print(f"输入前10个值: {input_data.flatten()[:10]}")

    # PyTorch
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        torch.manual_seed(seed)
        pt_module = torch.nn.RReLU(lower=lower, upper=upper)
        pt_module.train()
        pt_input = torch.tensor(input_data, dtype=torch.float32)
        pt_out = pt_module(pt_input)
        pt_res = pt_out.detach().numpy()
        print(f"PyTorch 输出前10个值: {pt_res.flatten()[:10]}")
    except Exception as e:
        print(f"PyTorch 执行错误: {e}")
        pt_res = None

    # MindSpore
    try:
        import mindspore as ms
        from mindspore import Tensor
        print(f"MindSpore 版本: {ms.__version__}")
        ms.set_context(mode=ms.PYNATIVE_MODE)
        ms.set_seed(seed)
        # instantiate; parameter names expected lower, upper
        ms_module = ms.nn.RReLU(lower=lower, upper=upper)
        ms_module.set_train(True)
        ms_input = Tensor(input_data, dtype=ms.float32)
        ms_out = ms_module(ms_input)
        ms_res = ms_out.asnumpy()
        print(f"MindSpore 输出前10个值: {ms_res.flatten()[:10]}")
    except Exception as e:
        print(f"MindSpore 执行错误: {type(e).__name__}: {e}")
        ms_res = None

    # 比较
    print()
    print("比较结果:")
    if pt_res is None or ms_res is None:
        print("无法比较：至少有一方执行失败")
        return

    # 基本统计
    def stats(arr):
        return {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
        }

    print(f"PyTorch stats: {stats(pt_res)}")
    print(f"MindSpore stats: {stats(ms_res)}")

    # 位置一致性：正值/负值掩码
    inp_mask_pos = input_data > 0
    inp_mask_neg = input_data <= 0

    # 对于正输入，RReLU 应保持原值
    pos_match = np.allclose(pt_res[inp_mask_pos], ms_res[inp_mask_pos], equal_nan=True)
    print(f"正输入位置输出是否一致: {pos_match}")

    # 对于负输入，两者为随机缩放，无法期望数值相同，但可以检查是否都在 [lower*x, upper*x]
    neg_inputs = input_data[inp_mask_neg]
    if neg_inputs.size > 0:
        pt_neg = pt_res[inp_mask_neg]
        ms_neg = ms_res[inp_mask_neg]
        # 检查范围
        pt_in_range = np.all((pt_neg >= lower * neg_inputs - 1e-6) & (pt_neg <= upper * neg_inputs + 1e-6))
        ms_in_range = np.all((ms_neg >= lower * neg_inputs - 1e-6) & (ms_neg <= upper * neg_inputs + 1e-6))
        print(f"PyTorch 负输入缩放是否在 [{lower}, {upper}] 范围: {pt_in_range}")
        print(f"MindSpore 负输入缩放是否在 [{lower}, {upper}] 范围: {ms_in_range}")
    else:
        print("无负输入，跳过负值范围检查")

    # 数值差异（最大绝对差）
    try:
        max_abs_diff = float(np.max(np.abs(pt_res - ms_res)))
    except Exception:
        max_abs_diff = float('nan')
    print(f"两者最大绝对差: {max_abs_diff}")

    # 结论提示
    if pos_match and (neg_inputs.size == 0 or (pt_in_range and ms_in_range)):
        print("结论: 两者语义一致（随机性允许的情况下）")
    else:
        print("结论: 存在差异（可能由于随机采样或实现细节）")


if __name__ == '__main__':
    # 样例1
    run_test(
        name='样例1: shape=[2,3], lower=0.1, upper=0.3',
        shape=[2, 3],
        sample_values=[0.4273471236228943, -0.10162176936864853, -0.7239707708358765,
                       -0.18958866596221924, -0.4222275912761688, 0.005457823630422354],
        lower=0.1,
        upper=0.3,
        seed=42
    )

    # 样例2
    run_test(
        name='样例2: shape=[2,3,4,5], lower=0.01, upper=0.99',
        shape=[2, 3, 4, 5],
        sample_values=[-1.958209753036499, -0.5572744011878967, 0.39172637462615967,
                       0.9652032256126404, -0.34185442328453064, -1.026029348373413,
                       -0.5314424633979797, 0.7895099520683289, 0.8332284688949585,
                       0.11467783898115158],
        lower=0.01,
        upper=0.99,
        seed=42
    )
