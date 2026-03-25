"""
最小可复现代码：torch.erfinv vs mindspore.mint.erfinv
测试场景：逆误差函数，输入包含超出定义域 (-1, 1) 的值

来源文件: llm_enhanced_torch_erfinv_20251215_184626.json
样例编号: 1
问题描述: 数值不匹配，最大差异: nan
"""

import numpy as np

# ==================== 测试配置 ====================
TEST_SHAPE = [2, 3, 4, 5]
TEST_DTYPE = np.float64
SAMPLE_VALUES = [
    -0.6813052710748467,
    -0.8495579314494206,
    -1.2201734282525922,      # 注意：超出 (-1, 1) 范围！
    0.0029796246678243966,
    0.9159602157960965,
    -0.6011707561855925,
    0.8630660602520888,
    -0.6426682542886082,
    -1.7247004652368227,      # 注意：超出 (-1, 1) 范围！
    -0.961864163197681
]

print("=" * 60)
print("测试配置")
print("=" * 60)
print(f"输入形状: {TEST_SHAPE}")
print(f"数据类型: {TEST_DTYPE}")
print(f"张量元素总数: {np.prod(TEST_SHAPE)}")
print(f"样例值数量: {len(SAMPLE_VALUES)}")
print()

# 分析样例值中超出定义域的值
print("=" * 60)
print("输入值分析 (erfinv 定义域为 (-1, 1))")
print("=" * 60)
out_of_domain = [v for v in SAMPLE_VALUES if v <= -1 or v >= 1]
print(f"超出定义域的值: {out_of_domain}")
print(f"这些值会导致 erfinv 返回 NaN 或 Inf")
print()

# 创建完整的输入数据
def create_input_data(shape, sample_values, dtype):
    """使用样例值填充完整张量"""
    total_elements = np.prod(shape)
    # 循环使用样例值填充
    full_data = []
    for i in range(total_elements):
        full_data.append(sample_values[i % len(sample_values)])
    return np.array(full_data, dtype=dtype).reshape(shape)

input_data = create_input_data(TEST_SHAPE, SAMPLE_VALUES, TEST_DTYPE)
print(f"输入数据形状: {input_data.shape}")
print(f"输入数据前10个值: {input_data.flatten()[:10]}")
print()

# ==================== PyTorch 测试 ====================
print("=" * 60)
print("PyTorch 测试: torch.erfinv")
print("=" * 60)

try:
    import torch
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 创建张量
    pt_input = torch.tensor(input_data, dtype=torch.float64)
    print(f"PyTorch 输入张量形状: {pt_input.shape}")
    
    # 计算 erfinv
    pt_output = torch.erfinv(pt_input)
    print(f"PyTorch 输出形状: {pt_output.shape}")
    
    # 转换为 numpy
    pt_result = pt_output.numpy()
    
    # 统计 NaN 和 Inf
    nan_count = np.isnan(pt_result).sum()
    inf_count = np.isinf(pt_result).sum()
    print(f"PyTorch 输出中 NaN 数量: {nan_count}")
    print(f"PyTorch 输出中 Inf 数量: {inf_count}")
    print(f"PyTorch 输出前10个值: {pt_result.flatten()[:10]}")
    
except Exception as e:
    print(f"PyTorch 执行错误: {type(e).__name__}: {e}")
    pt_result = None

print()

# ==================== MindSpore 测试 ====================
print("=" * 60)
print("MindSpore 测试: mindspore.mint.erfinv")
print("=" * 60)

try:
    import mindspore as ms
    from mindspore import Tensor
    import mindspore.mint as mint
    
    print(f"MindSpore 版本: {ms.__version__}")
    
    # 设置运行模式
    ms.set_context(mode=ms.PYNATIVE_MODE)
    
    # 创建张量
    ms_input = Tensor(input_data, dtype=ms.float64)
    print(f"MindSpore 输入张量形状: {ms_input.shape}")
    
    # 计算 erfinv
    ms_output = mint.erfinv(ms_input)
    print(f"MindSpore 输出形状: {ms_output.shape}")
    
    # 转换为 numpy
    ms_result = ms_output.asnumpy()
    
    # 统计 NaN 和 Inf
    nan_count = np.isnan(ms_result).sum()
    inf_count = np.isinf(ms_result).sum()
    print(f"MindSpore 输出中 NaN 数量: {nan_count}")
    print(f"MindSpore 输出中 Inf 数量: {inf_count}")
    print(f"MindSpore 输出前10个值: {ms_result.flatten()[:10]}")
    
except Exception as e:
    print(f"MindSpore 执行错误: {type(e).__name__}: {e}")
    ms_result = None

print()

# ==================== 结果比较 ====================
print("=" * 60)
print("结果比较")
print("=" * 60)

if pt_result is not None and ms_result is not None:
    # 获取有效值的掩码（非 NaN 且非 Inf）
    pt_valid = np.isfinite(pt_result)
    ms_valid = np.isfinite(ms_result)
    
    # 检查 NaN/Inf 位置是否一致
    nan_inf_match = np.array_equal(pt_valid, ms_valid)
    print(f"NaN/Inf 位置是否一致: {nan_inf_match}")
    
    # 分别统计
    both_valid = pt_valid & ms_valid
    pt_nan_ms_valid = ~pt_valid & ms_valid
    pt_valid_ms_nan = pt_valid & ~ms_valid
    both_invalid = ~pt_valid & ~ms_valid
    
    print(f"两者都是有效值的位置数: {both_valid.sum()}")
    print(f"两者都是 NaN/Inf 的位置数: {both_invalid.sum()}")
    print(f"PyTorch NaN/Inf 但 MindSpore 有效的位置数: {pt_nan_ms_valid.sum()}")
    print(f"PyTorch 有效但 MindSpore NaN/Inf 的位置数: {pt_valid_ms_nan.sum()}")
    
    # 计算有效值的差异
    if both_valid.sum() > 0:
        valid_diff = np.abs(pt_result[both_valid] - ms_result[both_valid])
        max_diff = np.max(valid_diff)
        mean_diff = np.mean(valid_diff)
        print(f"\n有效值的最大差异: {max_diff}")
        print(f"有效值的平均差异: {mean_diff}")
    
    # 原始比较方式（会产生 nan）
    raw_diff = np.abs(pt_result - ms_result)
    raw_max_diff = np.nanmax(raw_diff) if not np.all(np.isnan(raw_diff)) else float('nan')
    print(f"\n原始比较方式的最大差异: {np.max(raw_diff)}")
    print(f"原始比较方式的最大差异(忽略nan): {raw_max_diff}")
    
    # 找出差异最大的位置
    print("\n差异详情（前5个不一致的位置）:")
    diff_indices = np.where(raw_diff > 1e-10)
    count = 0
    for idx in zip(*diff_indices):
        if count >= 5:
            break
        pt_val = pt_result[idx]
        ms_val = ms_result[idx]
        input_val = input_data[idx]
        print(f"  位置 {idx}: 输入={input_val:.6f}, PyTorch={pt_val}, MindSpore={ms_val}")
        count += 1
        
else:
    print("无法比较：至少有一个框架执行失败")

print()

# ==================== 分析说明 ====================
print("=" * 60)
print("分析说明")
print("=" * 60)
print("""
erfinv (逆误差函数) 的数学性质：
- 定义域: (-1, 1)，即输入必须严格在 -1 和 1 之间
- 当输入 = -1 或 1 时，输出为 -Inf 或 +Inf
- 当输入 < -1 或 > 1 时，输出为 NaN（数学上无定义）

本测试样例中的问题：
1. 样例值包含超出定义域的值：-1.2201734282525922, -1.7247004652368227
2. 这些值会导致 erfinv 返回 NaN
3. NaN 与 NaN 比较会得到 NaN，导致"最大差异: nan"

建议的比较策略：
1. 先检查 NaN/Inf 的位置是否一致
2. 对有效值部分单独计算差异
3. 如果 NaN/Inf 位置一致且有效值差异在容差内，则认为行为一致
""")
