"""
最小可复现代码：torch.nn.Dropout2d vs mindspore.mint.nn.Dropout2d
来源文件: llm_enhanced_torch_nn_Dropout2d_20251214_170822.json
样例编号: 2, 5

Dropout2d: 随机将整个通道置零（而不是单个元素）
"""

import numpy as np

def create_input(shape, sample_values, dtype=np.float32):
    total = np.prod(shape)
    data = [sample_values[i % len(sample_values)] for i in range(total)]
    return np.array(data, dtype=dtype).reshape(shape)

def test_dropout2d(test_name, shape, sample_values, p):
    print("=" * 60)
    print(f"测试: {test_name}")
    print("=" * 60)
    print(f"形状: {shape} (N, C, H, W), p: {p}")
    
    input_data = create_input(shape, sample_values)
    
    # PyTorch
    import torch
    torch.manual_seed(42)
    pt_dropout2d = torch.nn.Dropout2d(p=p)
    pt_dropout2d.train()
    pt_input = torch.tensor(input_data)
    pt_output = pt_dropout2d(pt_input)
    pt_result = pt_output.numpy()
    
    print(f"\nPyTorch 输出前10个值: {pt_result.flatten()[:10]}")
    print(f"PyTorch 零值比例: {(pt_result == 0).sum() / pt_result.size:.2%}")
    
    # 检查是否是整个通道被置零
    N, C = shape[0], shape[1]
    pt_channel_zeros = 0
    for n in range(N):
        for c in range(C):
            if np.all(pt_result[n, c] == 0):
                pt_channel_zeros += 1
    print(f"PyTorch 被置零的通道数: {pt_channel_zeros}/{N*C}")
    
    # MindSpore (使用 mindspore.nn.Dropout2d)
    import mindspore as ms
    from mindspore import Tensor
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_seed(42)
    
    # mindspore.nn.Dropout2d 使用 p (丢弃概率)
    ms_dropout2d = ms.nn.Dropout2d(p=p)
    ms_dropout2d.set_train(True)
    ms_input = Tensor(input_data, dtype=ms.float32)
    ms_output = ms_dropout2d(ms_input)
    ms_result = ms_output.asnumpy()
    
    print(f"\nMindSpore 输出前10个值: {ms_result.flatten()[:10]}")
    print(f"MindSpore 零值比例: {(ms_result == 0).sum() / ms_result.size:.2%}")
    
    # 检查是否是整个通道被置零
    ms_channel_zeros = 0
    for n in range(N):
        for c in range(C):
            if np.all(ms_result[n, c] == 0):
                ms_channel_zeros += 1
    print(f"MindSpore 被置零的通道数: {ms_channel_zeros}/{N*C}")
    
    # 比较
    diff = np.abs(pt_result - ms_result).max()
    print(f"\n最大差异: {diff:.4f}")
    print(f"数值一致: {np.allclose(pt_result, ms_result)}")
    
    # eval 模式测试
    print("\n--- eval 模式测试 ---")
    pt_dropout2d.eval()
    pt_output_eval = pt_dropout2d(pt_input)
    
    ms_dropout2d.set_train(False)
    ms_output_eval = ms_dropout2d(ms_input)
    
    eval_diff = np.abs(pt_output_eval.numpy() - ms_output_eval.asnumpy()).max()
    print(f"eval 模式最大差异: {eval_diff}")
    print(f"eval 模式两者一致: {np.allclose(pt_output_eval.numpy(), ms_output_eval.asnumpy())}")
    print()

if __name__ == "__main__":
    import torch
    import mindspore as ms
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"MindSpore 版本: {ms.__version__}")
    print()
    
    # 样例2: [2,3,4,4] p=0.5
    test_dropout2d(
        "样例2: [2,3,4,4] p=0.5",
        shape=[2, 3, 4, 4],
        sample_values=[0.154, -0.159, -1.187, -0.370, 0.023, -0.650, 0.937, -0.162, -1.341, -1.426],
        p=0.5
    )
    
    # 样例5: [2,3,4,5] p=0.5
    test_dropout2d(
        "样例5: [2,3,4,5] p=0.5",
        shape=[2, 3, 4, 5],
        sample_values=[1.515, 2.036, 0.028, 0.418, -1.767, -1.451, -0.838, -0.444, -1.096, -0.036],
        p=0.5
    )
    
    print("=" * 60)
    print("结论")
    print("=" * 60)
    print("""
Dropout2d 是随机操作，随机将整个通道置零。
- train 模式：随机丢弃的通道不同，数值必然不同（预期行为）
- eval 模式：两者都不丢弃，输出等于输入，结果一致

原始测试的"数值不匹配"是误报，随机操作不应直接比较数值。
""")
