"""
最小可复现代码：torch.erfinv vs paddle.erfinv
测试场景：逆误差函数，输入包含超出定义域的值

来源文件: llm_enhanced_torch_erfinv_20251125_145454.json
样例编号: 2
问题描述: 数值不匹配，最大差异: nan
"""

import numpy as np

# ==================== 测试配置 ====================
TEST_SHAPE = [5]
TEST_DTYPE = np.float32
SAMPLE_VALUES = [
    0.7778396010398865,
    -0.9116541147232056,
    0.43586117029190063,
    -1.4654308557510376,   # 注意：超出 (-1, 1) 定义域！
    -0.7387789487838745
]

print("=" * 60)
print("测试配置")
print("=" * 60)
print(f"输入形状: {TEST_SHAPE}")
print(f"数据类型: {TEST_DTYPE}")
print(f"样例值: {SAMPLE_VALUES}")
print()

# 分析定义域
print("=" * 60)
print("输入值分析 (erfinv 定义域为 (-1, 1))")
print("=" * 60)
out_of_domain = [v for v in SAMPLE_VALUES if v <= -1 or v >= 1]
in_domain = [v for v in SAMPLE_VALUES if -1 < v < 1]
print(f"在定义域内的值: {in_domain}")
print(f"超出定义域的值: {out_of_domain}")
print(f"超出定义域的值会导致 erfinv 返回 NaN")
print()

# 创建输入数据
input_data = np.array(SAMPLE_VALUES, dtype=TEST_DTYPE)
print(f"输入数据: {input_data}")
print()

# ==================== PyTorch 测试 ====================
print("=" * 60)
print("PyTorch 测试: torch.erfinv")
print("=" * 60)

try:
    import torch
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 创建张量
    pt_input = torch.tensor(input_data)
    print(f"PyTorch 输入: {pt_input}")
    
    # 计算 erfinv
    pt_output = torch.erfinv(pt_input)
    print(f"PyTorch 输出: {pt_output}")
    
    # 转换为 numpy
    pt_result = pt_output.numpy()
    
    # 统计
    nan_count = np.isnan(pt_result).sum()
    inf_count = np.isinf(pt_result).sum()
    print(f"PyTorch NaN 数量: {nan_count}")
    print(f"PyTorch Inf 数量: {inf_count}")
    
    # 逐个值分析
    print("\n逐值分析:")
    for i, (inp, out) in enumerate(zip(input_data, pt_result)):
        status = "有效" if np.isfinite(out) else ("NaN" if np.isnan(out) else "Inf")
        domain_status = "在定义域内" if -1 < inp < 1 else "超出定义域"
        print(f"  [{i}] 输入: {inp:+.6f} ({domain_status}) -> 输出: {out} ({status})")
    
except Exception as e:
    print(f"PyTorch 执行错误: {type(e).__name__}: {e}")
    pt_result = None

print()

# ==================== PaddlePaddle 测试 ====================
print("=" * 60)
print("PaddlePaddle 测试: paddle.erfinv")
print("=" * 60)

try:
    import paddle
    print(f"PaddlePaddle 版本: {paddle.__version__}")
    
    # 创建张量
    pd_input = paddle.to_tensor(input_data)
    print(f"Paddle 输入: {pd_input}")
    
    # 计算 erfinv
    pd_output = paddle.erfinv(pd_input)
    print(f"Paddle 输出: {pd_output}")
    
    # 转换为 numpy
    pd_result = pd_output.numpy()
    
    # 统计
    nan_count = np.isnan(pd_result).sum()
    inf_count = np.isinf(pd_result).sum()
    print(f"Paddle NaN 数量: {nan_count}")
    print(f"Paddle Inf 数量: {inf_count}")
    
    # 逐个值分析
    print("\n逐值分析:")
    for i, (inp, out) in enumerate(zip(input_data, pd_result)):
        status = "有效" if np.isfinite(out) else ("NaN" if np.isnan(out) else "Inf")
        domain_status = "在定义域内" if -1 < inp < 1 else "超出定义域"
        print(f"  [{i}] 输入: {inp:+.6f} ({domain_status}) -> 输出: {out} ({status})")
    
except Exception as e:
    print(f"PaddlePaddle 执行错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    pd_result = None

print()

# ==================== 结果比较 ====================
print("=" * 60)
print("结果比较")
print("=" * 60)

if pt_result is not None and pd_result is not None:
    print(f"PyTorch 结果:  {pt_result}")
    print(f"Paddle 结果:   {pd_result}")
    
    # 检查 NaN/Inf 位置
    pt_valid = np.isfinite(pt_result)
    pd_valid = np.isfinite(pd_result)
    
    nan_inf_match = np.array_equal(pt_valid, pd_valid)
    print(f"\nNaN/Inf 位置是否一致: {nan_inf_match}")
    
    # 统计
    both_valid = pt_valid & pd_valid
    both_invalid = ~pt_valid & ~pd_valid
    pt_invalid_pd_valid = ~pt_valid & pd_valid
    pt_valid_pd_invalid = pt_valid & ~pd_valid
    
    print(f"两者都有效的位置数: {both_valid.sum()}")
    print(f"两者都是 NaN/Inf 的位置数: {both_invalid.sum()}")
    print(f"PyTorch NaN/Inf 但 Paddle 有效: {pt_invalid_pd_valid.sum()}")
    print(f"PyTorch 有效但 Paddle NaN/Inf: {pt_valid_pd_invalid.sum()}")
    
    # 计算有效值差异
    if both_valid.sum() > 0:
        valid_diff = np.abs(pt_result[both_valid] - pd_result[both_valid])
        print(f"\n有效值最大差异: {np.max(valid_diff)}")
        print(f"有效值平均差异: {np.mean(valid_diff)}")
    
    # 原始比较
    raw_diff = np.abs(pt_result - pd_result)
    print(f"\n原始比较最大差异: {np.max(raw_diff)}")
    
    # 最终判定
    print("\n" + "=" * 60)
    print("最终判定")
    print("=" * 60)
    if nan_inf_match:
        if both_valid.sum() > 0:
            max_valid_diff = np.max(np.abs(pt_result[both_valid] - pd_result[both_valid]))
            if max_valid_diff < 1e-5:
                print("✓ 两个框架行为一致（NaN位置相同，有效值差异在容差内）")
            else:
                print(f"✗ 有效值存在显著差异: {max_valid_diff}")
        else:
            print("✓ 两个框架行为一致（所有输出都是 NaN/Inf，位置相同）")
    else:
        print("✗ NaN/Inf 位置不一致，行为不同")
        
else:
    print("无法比较：至少有一个框架执行失败")

print()

# ==================== 分析说明 ====================
print("=" * 60)
print("分析说明")
print("=" * 60)
print("""
erfinv (逆误差函数) 的数学性质：
- 定义域: (-1, 1)
- erfinv(-1) = -∞, erfinv(1) = +∞
- 当输入 < -1 或 > 1 时，结果为 NaN（数学上无定义）

本测试样例中：
- 输入值 -1.4654308557510376 超出定义域
- 该值会导致 erfinv 返回 NaN
- 原始比较使用 np.max(np.abs(a - b))，NaN 参与运算导致结果为 nan

这是比较逻辑的问题，而非框架实际行为不一致。
""")
