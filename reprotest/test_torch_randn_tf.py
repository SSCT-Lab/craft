"""
最小可复现代码：torch.randn vs tf.random.normal
测试场景：正态分布随机数生成函数的语义一致性测试

来源文件: llm_enhanced_torch_randn_20251215_231154.json
样例编号: 1, 4

结论：随机数生成函数不应比较具体数值，而应比较语义一致性
"""

import numpy as np

def test_randn_comparison(test_name, shape, pt_dtype, tf_dtype, seed=None):
    """测试 torch.randn 和 tf.random.normal 的语义一致性"""
    print("=" * 60)
    print(f"测试: {test_name}")
    print("=" * 60)
    print(f"形状: {shape}")
    print(f"PyTorch dtype: {pt_dtype}, TensorFlow dtype: {tf_dtype}")
    print(f"Seed: {seed}")
    print()
    
    # ==================== PyTorch ====================
    import torch
    
    if seed is not None:
        torch.manual_seed(seed)
    
    pt_output = torch.randn(*shape, dtype=getattr(torch, pt_dtype.replace('torch.', '')))
    pt_result = pt_output.detach().numpy()
    
    print(f"PyTorch 输出形状: {pt_result.shape}")
    print(f"PyTorch 输出范围: [{pt_result.min():.4f}, {pt_result.max():.4f}]")
    print(f"PyTorch 输出均值: {pt_result.mean():.6f} (期望: 0.0)")
    print(f"PyTorch 输出标准差: {pt_result.std():.6f} (期望: 1.0)")
    print(f"PyTorch 前5个值: {pt_result.flatten()[:5]}")
    print()
    
    # ==================== TensorFlow ====================
    import tensorflow as tf
    
    if seed is not None:
        tf.random.set_seed(seed)
    
    tf_output = tf.random.normal(shape=shape, mean=0.0, stddev=1.0, dtype=tf_dtype)
    tf_result = tf_output.numpy()
    
    print(f"TensorFlow 输出形状: {tf_result.shape}")
    print(f"TensorFlow 输出范围: [{tf_result.min():.4f}, {tf_result.max():.4f}]")
    print(f"TensorFlow 输出均值: {tf_result.mean():.6f} (期望: 0.0)")
    print(f"TensorFlow 输出标准差: {tf_result.std():.6f} (期望: 1.0)")
    print(f"TensorFlow 前5个值: {tf_result.flatten()[:5]}")
    print()
    
    # ==================== 语义一致性比较 ====================
    print("-" * 40)
    print("语义一致性比较:")
    
    # 1. 形状一致性
    shape_match = pt_result.shape == tf_result.shape
    print(f"  形状一致: {shape_match}")
    
    # 2. 数据类型一致性
    dtype_match = pt_result.dtype == tf_result.dtype
    print(f"  dtype 一致: {dtype_match} (PT: {pt_result.dtype}, TF: {tf_result.dtype})")
    
    # 3. 统计特性（标准正态分布: mean≈0, std≈1）
    pt_mean_ok = abs(pt_result.mean()) < 0.1  # 均值接近0
    tf_mean_ok = abs(tf_result.mean()) < 0.1
    pt_std_ok = 0.8 < pt_result.std() < 1.2   # 标准差接近1
    tf_std_ok = 0.8 < tf_result.std() < 1.2
    
    print(f"  PyTorch 均值合理 (≈0): {pt_mean_ok} ({pt_result.mean():.4f})")
    print(f"  TensorFlow 均值合理 (≈0): {tf_mean_ok} ({tf_result.mean():.4f})")
    print(f"  PyTorch 标准差合理 (≈1): {pt_std_ok} ({pt_result.std():.4f})")
    print(f"  TensorFlow 标准差合理 (≈1): {tf_std_ok} ({tf_result.std():.4f})")
    
    # 4. 数值差异
    value_diff = np.abs(pt_result - tf_result).max()
    print(f"  数值最大差异: {value_diff:.4f} (预期存在差异)")
    
    # 总结
    semantic_match = shape_match and dtype_match
    print()
    if semantic_match:
        print("✓ 语义一致：两个API产生相同形状、类型的标准正态分布随机数")
        print("  (数值不同是正常的，因为PRNG算法不同)")
    else:
        print("✗ 语义不一致")
    
    print()
    return semantic_match


if __name__ == "__main__":
    import torch
    import tensorflow as tf
    
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"TensorFlow 版本: {tf.__version__}")
    print()
    
    # 样例1: [512], float32, 无seed
    test_randn_comparison(
        test_name="样例1: [512] float32 无seed",
        shape=[512],
        pt_dtype="float32",
        tf_dtype=tf.float32,
        seed=None
    )
    
    # 样例4: [2, 1024, 3, 3], float64, seed=12345
    test_randn_comparison(
        test_name="样例4: [2,1024,3,3] float64 seed=12345",
        shape=[2, 1024, 3, 3],
        pt_dtype="float64",
        tf_dtype=tf.float64,
        seed=12345
    )
    
    print("=" * 60)
    print("结论")
    print("=" * 60)
    print("""
torch.randn 和 tf.random.normal 都是生成标准正态分布 N(0,1) 的随机数。

数值差异是 **算子本身必然导致的**，原因：
1. PyTorch 和 TensorFlow 使用不同的伪随机数生成算法 (PRNG)
2. 即使设置相同的 seed，生成的随机序列也完全不同
3. 这是设计如此，不是 bug

正确的语义一致性验证应检查：
- 输出形状相同 ✓
- 数据类型相同 ✓  
- 统计特性相似（均值≈0，标准差≈1）✓

原始测试报告的"数值不匹配"是 **误报**。
""")
