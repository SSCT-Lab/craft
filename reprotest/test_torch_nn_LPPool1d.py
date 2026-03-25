"""
最小可复现代码：torch.nn.LPPool1d vs mindspore.nn.LPPool1d
测试场景：LP范数池化，norm_type=1.5

来源文件: llm_enhanced_torch_nn_LPPool1d_20251215_193350.json
样例编号: 1
问题描述: 数值不匹配，最大差异: nan
"""

import numpy as np

# ==================== 测试配置 ====================
TEST_SHAPE = [2, 1, 3]  # [batch, channels, length]
TEST_DTYPE = np.float32
SAMPLE_VALUES = [
    0.6357309818267822,
    0.818108856678009,
    -0.9655203223228455,
    -0.9954814910888672,
    0.22021013498306274,
    -0.023969437927007675
]
NORM_TYPE = 1.5
KERNEL_SIZE = 3
STRIDE = 1

print("=" * 60)
print("测试配置")
print("=" * 60)
print(f"输入形状: {TEST_SHAPE} (batch, channels, length)")
print(f"数据类型: {TEST_DTYPE}")
print(f"norm_type: {NORM_TYPE}")
print(f"kernel_size: {KERNEL_SIZE}")
print(f"stride: {STRIDE}")
print()

# 创建输入数据
input_data = np.array(SAMPLE_VALUES, dtype=TEST_DTYPE).reshape(TEST_SHAPE)
print(f"输入数据:\n{input_data}")
print()

# 分析：LPPool1d 对负数的处理
# print("=" * 60)
# print("输入值分析 (LPPool 对负数的处理)")
# print("=" * 60)
# print(f"输入包含负数: {np.any(input_data < 0)}")
# print(f"负数值: {input_data[input_data < 0]}")
# print("""
# LPPool1d 公式: output = (sum(|x|^p))^(1/p)
# 当 p 不是整数（如 1.5）且输入包含负数时：
# - |x|^1.5 = |x| * |x|^0.5 = |x| * sqrt(|x|)
# - 这在数学上是定义良好的
# 但某些实现可能直接计算 x^p，对负数会产生 NaN
# """)
# print()

# ==================== PyTorch 测试 ====================
print("=" * 60)
print("PyTorch 测试: torch.nn.LPPool1d")
print("=" * 60)

try:
    import torch
    import torch.nn as nn
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 创建张量
    pt_input = torch.tensor(input_data)
    print(f"PyTorch 输入形状: {pt_input.shape}")
    print(f"PyTorch 输入:\n{pt_input}")
    
    # 创建 LPPool1d 层
    pt_pool = nn.LPPool1d(norm_type=NORM_TYPE, kernel_size=KERNEL_SIZE, stride=STRIDE)
    
    # 前向传播
    pt_output = pt_pool(pt_input)
    print(f"PyTorch 输出形状: {pt_output.shape}")
    print(f"PyTorch 输出:\n{pt_output}")
    
    # 转换为 numpy
    pt_result = pt_output.detach().numpy()
    
    # 检查 NaN
    nan_count = np.isnan(pt_result).sum()
    print(f"PyTorch 输出中 NaN 数量: {nan_count}")
    
except Exception as e:
    print(f"PyTorch 执行错误: {type(e).__name__}: {e}")
    pt_result = None

print()

# ==================== MindSpore 测试 ====================
print("=" * 60)
print("MindSpore 测试: mindspore.nn.LPPool1d")
print("=" * 60)

try:
    import mindspore as ms
    from mindspore import Tensor
    import mindspore.nn as nn_ms
    
    print(f"MindSpore 版本: {ms.__version__}")
    
    # 设置运行模式
    ms.set_context(mode=ms.PYNATIVE_MODE)
    
    # 创建张量
    ms_input = Tensor(input_data, dtype=ms.float32)
    print(f"MindSpore 输入形状: {ms_input.shape}")
    print(f"MindSpore 输入:\n{ms_input}")
    
    # 创建 LPPool1d 层
    ms_pool = nn_ms.LPPool1d(norm_type=NORM_TYPE, kernel_size=KERNEL_SIZE, stride=STRIDE)
    
    # 前向传播
    ms_output = ms_pool(ms_input)
    print(f"MindSpore 输出形状: {ms_output.shape}")
    print(f"MindSpore 输出:\n{ms_output}")
    
    # 转换为 numpy
    ms_result = ms_output.asnumpy()
    
    # 检查 NaN
    nan_count = np.isnan(ms_result).sum()
    print(f"MindSpore 输出中 NaN 数量: {nan_count}")
    
except Exception as e:
    print(f"MindSpore 执行错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    ms_result = None

print()

# ==================== 结果比较 ====================
print("=" * 60)
print("结果比较")
print("=" * 60)

if pt_result is not None and ms_result is not None:
    print(f"PyTorch 结果:\n{pt_result}")
    print(f"MindSpore 结果:\n{ms_result}")
    
    # 检查 NaN 情况
    pt_nan = np.isnan(pt_result)
    ms_nan = np.isnan(ms_result)
    
    print(f"\nPyTorch NaN 位置: {np.where(pt_nan)}")
    print(f"MindSpore NaN 位置: {np.where(ms_nan)}")
    
    # NaN 位置是否一致
    nan_match = np.array_equal(pt_nan, ms_nan)
    print(f"NaN 位置是否一致: {nan_match}")
    
    # 计算有效值的差异
    valid_mask = ~pt_nan & ~ms_nan
    if valid_mask.sum() > 0:
        valid_diff = np.abs(pt_result[valid_mask] - ms_result[valid_mask])
        print(f"\n有效值数量: {valid_mask.sum()}")
        print(f"有效值最大差异: {np.max(valid_diff)}")
        print(f"有效值平均差异: {np.mean(valid_diff)}")
    
    # 原始比较
    raw_diff = np.abs(pt_result - ms_result)
    print(f"\n原始比较最大差异: {np.max(raw_diff)}")
    
else:
    print("无法比较：至少有一个框架执行失败")

print()

# ==================== 分析说明 ====================
# print("=" * 60)
# print("分析说明")
# print("=" * 60)
# print("""
# LPPool1d (Lp范数池化) 的计算公式：
#   output = (sum(|x_i|^p))^(1/p)
#   其中 p = norm_type

# 潜在的 NaN 来源：
# 1. 当 norm_type 不是整数（如 1.5）时，某些实现可能直接计算 x^p
#    而不是 |x|^p，导致负数的非整数次幂产生 NaN
# 2. 当所有输入都是 0 时，0^(1/p) = 0，不会产生 NaN
# 3. 两个框架可能使用不同的内部实现策略

# 建议：检查两个框架对负数输入的处理方式是否一致
# """)
