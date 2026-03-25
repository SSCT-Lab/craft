"""
最小可复现代码：torch.mean vs mindspore.mint.mean
测试场景：包含空维度的张量 (shape 中有 0)

来源文件: llm_enhanced_torch_mean_20251215_184512.json
样例编号: 1
问题描述: 数值不匹配，最大差异: nan
"""

import numpy as np

# ==================== 测试配置 ====================
TEST_SHAPE = [2, 0, 4, 5, 6]  # 注意：第二个维度为 0，这是一个空张量
TEST_DTYPE = np.float64

print("=" * 60)
print("测试配置")
print("=" * 60)
print(f"输入形状: {TEST_SHAPE}")
print(f"数据类型: {TEST_DTYPE}")
print(f"张量元素总数: {np.prod(TEST_SHAPE)}")  # 应该是 0
print()

# ==================== PyTorch 测试 ====================
print("=" * 60)
print("PyTorch 测试: torch.mean")
print("=" * 60)

try:
    import torch
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 创建空张量
    pt_input = torch.empty(TEST_SHAPE, dtype=torch.float64)
    print(f"PyTorch 输入张量形状: {pt_input.shape}")
    print(f"PyTorch 输入张量元素数: {pt_input.numel()}")
    
    # 计算 mean
    pt_output = torch.mean(pt_input)
    print(f"PyTorch 输出值: {pt_output}")
    print(f"PyTorch 输出类型: {type(pt_output)}")
    print(f"PyTorch 输出 dtype: {pt_output.dtype}")
    
    # 转换为 numpy 用于后续比较
    pt_result = pt_output.numpy()
    print(f"PyTorch 结果 (numpy): {pt_result}")
    
except Exception as e:
    print(f"PyTorch 执行错误: {type(e).__name__}: {e}")
    pt_result = None

print()

# ==================== MindSpore 测试 ====================
print("=" * 60)
print("MindSpore 测试: mindspore.mint.mean")
print("=" * 60)

try:
    import mindspore as ms
    from mindspore import Tensor
    import mindspore.mint as mint
    
    print(f"MindSpore 版本: {ms.__version__}")
    
    # 设置运行模式
    ms.set_context(mode=ms.PYNATIVE_MODE)
    
    # 创建空张量
    ms_input = Tensor(np.empty(TEST_SHAPE, dtype=TEST_DTYPE))
    print(f"MindSpore 输入张量形状: {ms_input.shape}")
    print(f"MindSpore 输入张量元素数: {ms_input.size}")
    
    # 计算 mean
    ms_output = mint.mean(ms_input)
    print(f"MindSpore 输出值: {ms_output}")
    print(f"MindSpore 输出类型: {type(ms_output)}")
    print(f"MindSpore 输出 dtype: {ms_output.dtype}")
    
    # 转换为 numpy 用于后续比较
    ms_result = ms_output.asnumpy()
    print(f"MindSpore 结果 (numpy): {ms_result}")
    
except Exception as e:
    print(f"MindSpore 执行错误: {type(e).__name__}: {e}")
    ms_result = None

print()

# ==================== 结果比较 ====================
print("=" * 60)
print("结果比较")
print("=" * 60)

if pt_result is not None and ms_result is not None:
    print(f"PyTorch 结果: {pt_result}")
    print(f"MindSpore 结果: {ms_result}")
    
    # 检查是否都是 nan
    pt_is_nan = np.isnan(pt_result)
    ms_is_nan = np.isnan(ms_result)
    
    print(f"PyTorch 是否为 NaN: {pt_is_nan}")
    print(f"MindSpore 是否为 NaN: {ms_is_nan}")
    
    # 计算差异
    if pt_is_nan and ms_is_nan:
        print("两者都是 NaN，行为一致")
        diff = 0.0
    elif pt_is_nan or ms_is_nan:
        print("警告：只有一方是 NaN，行为不一致！")
        diff = float('nan')
    else:
        diff = np.abs(pt_result - ms_result)
    
    print(f"数值差异: {diff}")
    
    # 判断是否匹配
    if np.isnan(diff):
        match_status = "不匹配 (存在 NaN 差异)"
    elif diff < 1e-6:
        match_status = "匹配"
    else:
        match_status = f"不匹配 (差异: {diff})"
    
    print(f"比较结果: {match_status}")
else:
    print("无法比较：至少有一个框架执行失败")

print()

# ==================== 分析说明 ====================
print("=" * 60)
print("分析说明")
print("=" * 60)
print("""
当对空张量（元素数为 0）计算均值时：
- 数学上，0/0 是未定义的，通常返回 NaN
- PyTorch 和 MindSpore 对空张量的 mean 操作可能有不同的处理方式

可能的差异原因：
1. 两个框架对空张量 mean 的返回值不同
2. 一方返回 NaN，另一方可能抛出异常或返回其他值
3. NaN 与任何值（包括另一个 NaN）比较都会得到 NaN 差异
""")
