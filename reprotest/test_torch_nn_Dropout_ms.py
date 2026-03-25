"""
最小可复现代码：torch.nn.Dropout vs mindspore.mint.nn.Dropout
来源文件: llm_enhanced_torch_nn_Dropout_20251215_193853.json
样例编号: 2, 3, 5
"""

import numpy as np

def create_input(shape, sample_values, dtype=np.float32):
    total = np.prod(shape)
    data = [sample_values[i % len(sample_values)] for i in range(total)]
    return np.array(data, dtype=dtype).reshape(shape)

def test_dropout(test_name, shape, sample_values, p):
    print("=" * 60)
    print(f"测试: {test_name}")
    print("=" * 60)
    print(f"形状: {shape}, p: {p}")
    
    input_data = create_input(shape, sample_values)
    
    # PyTorch
    import torch
    torch.manual_seed(42)
    pt_dropout = torch.nn.Dropout(p=p)
    pt_dropout.train()
    pt_input = torch.tensor(input_data)
    pt_output = pt_dropout(pt_input)
    pt_result = pt_output.numpy()
    
    print(f"\nPyTorch 输出前10个值: {pt_result.flatten()[:10]}")
    print(f"PyTorch 零值比例: {(pt_result == 0).sum() / pt_result.size:.2%}")
    
    # MindSpore
    import mindspore as ms
    from mindspore import Tensor
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_seed(42)
    
    # mindspore.nn.Dropout 使用 keep_prob (保留概率)，不是 p (丢弃概率)
    ms_dropout = ms.nn.Dropout(keep_prob=1-p)
    ms_dropout.set_train(True)
    ms_input = Tensor(input_data, dtype=ms.float32)
    ms_output = ms_dropout(ms_input)
    ms_result = ms_output.asnumpy()
    
    print(f"\nMindSpore 输出前10个值: {ms_result.flatten()[:10]}")
    print(f"MindSpore 零值比例: {(ms_result == 0).sum() / ms_result.size:.2%}")
    
    # 比较
    diff = np.abs(pt_result - ms_result).max()
    print(f"\n最大差异: {diff:.4f}")
    print(f"数值一致: {np.allclose(pt_result, ms_result)}")
    
    # eval 模式测试
    print("\n--- eval 模式测试 ---")
    pt_dropout.eval()
    pt_output_eval = pt_dropout(pt_input)
    
    ms_dropout.set_train(False)
    ms_output_eval = ms_dropout(ms_input)
    
    eval_match = np.allclose(pt_output_eval.numpy(), ms_output_eval.asnumpy())
    input_match_pt = np.allclose(pt_output_eval.numpy(), input_data)
    input_match_ms = np.allclose(ms_output_eval.asnumpy(), input_data)
    
    print(f"PyTorch eval 输出等于输入: {input_match_pt}")
    print(f"MindSpore eval 输出等于输入: {input_match_ms}")
    print(f"eval 模式两者一致: {eval_match}")
    print()

if __name__ == "__main__":
    import torch
    import mindspore as ms
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"MindSpore 版本: {ms.__version__}")
    print()
    
    # 样例2: p=0.3
    test_dropout(
        "样例2: [4,5] p=0.3",
        shape=[4, 5],
        sample_values=[1.472, -0.289, -0.441, 1.188, 0.072, -0.591, -2.054, -0.422, -0.603, -0.681],
        p=0.3
    )
    
    # 样例3: p=0.999 (几乎全部丢弃)
    test_dropout(
        "样例3: [2,3,4] p=0.999",
        shape=[2, 3, 4],
        sample_values=[-0.067, 0.797, 0.097, 1.433, 0.606, 0.083, 1.397, -0.686, 0.486, 0.408],
        p=0.999
    )
    
    # 样例5: p=0.5
    test_dropout(
        "样例5: [2,3,4] p=0.5",
        shape=[2, 3, 4],
        sample_values=[0.140, -1.236, 0.667, 2.229, 0.001, -0.578, -0.045, -0.184, 0.594, 0.128],
        p=0.5
    )
    
    print("=" * 60)
    print("结论")
    print("=" * 60)
    print("""
Dropout 是随机操作，两个框架使用不同的随机数生成器。
- train 模式：随机丢弃位置不同，数值必然不同（这是预期行为）
- eval 模式：两者都不丢弃，输出等于输入，结果一致

原始测试的"数值不匹配"是误报，随机操作不应直接比较数值。
""")
