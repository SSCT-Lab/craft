"""
最小可复现代码：torch.rand vs tf.random.uniform
测试场景：随机数生成函数的语义一致性测试

来源文件: llm_enhanced_torch_rand_20251215_231651.json
样例编号: 1, 2, 3, 4, 6

关键点：随机数生成函数不应比较具体数值，而应比较：
1. 输出形状是否一致
2. 输出范围是否一致 [0, 1)
3. 数据类型是否一致
4. 统计特性是否一致（均值约0.5，均匀分布）
"""

import numpy as np

def test_rand_comparison(test_name, pt_size, tf_shape, pt_dtype, tf_dtype, seed=None):
    """测试 torch.rand 和 tf.random.uniform 的语义一致性"""
    print("=" * 60)
    print(f"测试: {test_name}")
    print("=" * 60)
    print(f"形状: {pt_size}")
    print(f"PyTorch dtype: {pt_dtype}, TensorFlow dtype: {tf_dtype}")
    print(f"Seed: {seed}")
    print()
    
    # ==================== PyTorch ====================
    import torch
    
    if seed is not None:
        torch.manual_seed(seed)
    
    pt_output = torch.rand(*pt_size, dtype=getattr(torch, pt_dtype.replace('torch.', '')))
    pt_result = pt_output.numpy()
    
    print(f"PyTorch 输出形状: {pt_result.shape}")
    print(f"PyTorch 输出范围: [{pt_result.min():.6f}, {pt_result.max():.6f}]")
    print(f"PyTorch 输出均值: {pt_result.mean():.6f}")
    print(f"PyTorch 输出标准差: {pt_result.std():.6f}")
    print(f"PyTorch 前5个值: {pt_result.flatten()[:5]}")
    print()
    
    # ==================== TensorFlow ====================
    import tensorflow as tf
    
    if seed is not None:
        tf.random.set_seed(seed)
    
    tf_output = tf.random.uniform(shape=tf_shape, minval=0.0, maxval=1.0, dtype=tf_dtype)
    tf_result = tf_output.numpy()
    
    print(f"TensorFlow 输出形状: {tf_result.shape}")
    print(f"TensorFlow 输出范围: [{tf_result.min():.6f}, {tf_result.max():.6f}]")
    print(f"TensorFlow 输出均值: {tf_result.mean():.6f}")
    print(f"TensorFlow 输出标准差: {tf_result.std():.6f}")
    print(f"TensorFlow 前5个值: {tf_result.flatten()[:5]}")
    print()
    
    # ==================== 语义一致性比较 ====================
    print("-" * 40)
    print("语义一致性比较:")
    
    # 1. 形状一致性
    shape_match = pt_result.shape == tf_result.shape
    print(f"  形状一致: {shape_match} (PT: {pt_result.shape}, TF: {tf_result.shape})")
    
    # 2. 范围一致性 [0, 1)
    pt_in_range = (pt_result >= 0).all() and (pt_result < 1).all()
    tf_in_range = (tf_result >= 0).all() and (tf_result < 1).all()
    print(f"  PyTorch 范围 [0,1): {pt_in_range}")
    print(f"  TensorFlow 范围 [0,1): {tf_in_range}")
    
    # 3. 数据类型一致性
    dtype_match = pt_result.dtype == tf_result.dtype
    print(f"  dtype 一致: {dtype_match} (PT: {pt_result.dtype}, TF: {tf_result.dtype})")
    
    # 4. 统计特性（均匀分布的均值应接近 0.5）
    pt_mean_ok = 0.3 < pt_result.mean() < 0.7  # 宽松检查
    tf_mean_ok = 0.3 < tf_result.mean() < 0.7
    print(f"  PyTorch 均值合理: {pt_mean_ok} ({pt_result.mean():.4f})")
    print(f"  TensorFlow 均值合理: {tf_mean_ok} ({tf_result.mean():.4f})")
    
    # 5. 数值差异（随机数生成器不同，预期会有差异）
    value_diff = np.abs(pt_result - tf_result).max()
    print(f"  数值最大差异: {value_diff:.6f} (预期存在差异，因为随机算法不同)")
    
    # 总结
    semantic_match = shape_match and pt_in_range and tf_in_range and dtype_match
    print()
    if semantic_match:
        print("✓ 语义一致：两个API产生相同形状、范围、类型的随机数")
        print("  (数值不同是正常的，因为随机数生成算法不同)")
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
    
    # 样例1: [64, 62], float32, 无seed
    test_rand_comparison(
        test_name="样例1: [64,62] float32 无seed",
        pt_size=[64, 62],
        tf_shape=[64, 62],
        pt_dtype="float32",
        tf_dtype=tf.float32,
        seed=None
    )
    
    # 样例2: [64, 62], float32, seed=1234
    test_rand_comparison(
        test_name="样例2: [64,62] float32 seed=1234",
        pt_size=[64, 62],
        tf_shape=[64, 62],
        pt_dtype="float32",
        tf_dtype=tf.float32,
        seed=1234
    )
    
    # 样例3: [1], float32, 无seed
    test_rand_comparison(
        test_name="样例3: [1] float32 无seed",
        pt_size=[1],
        tf_shape=[1],
        pt_dtype="float32",
        tf_dtype=tf.float32,
        seed=None
    )
    
    # 样例4: [2, 3, 4], float32, 无seed
    test_rand_comparison(
        test_name="样例4: [2,3,4] float32 无seed",
        pt_size=[2, 3, 4],
        tf_shape=[2, 3, 4],
        pt_dtype="float32",
        tf_dtype=tf.float32,
        seed=None
    )
    
    # 样例6: [2, 3, 4, 5], float64, seed=1234
    test_rand_comparison(
        test_name="样例6: [2,3,4,5] float64 seed=1234",
        pt_size=[2, 3, 4, 5],
        tf_shape=[2, 3, 4, 5],
        pt_dtype="float64",
        tf_dtype=tf.float64,
        seed=1234
    )
    
    print("=" * 60)
    print("总结")
    print("=" * 60)
    print("""
对于随机数生成函数 (torch.rand vs tf.random.uniform):
- 数值差异是 **预期行为**，因为两个框架使用不同的随机数生成算法
- 即使设置相同的 seed，生成的序列也不同
- 正确的比较方式是验证 **语义一致性**：
  1. 输出形状相同
  2. 输出范围相同 [0, 1)
  3. 数据类型相同
  4. 统计特性相似（均匀分布）

原始测试报告的"数值不匹配"是 **误报**，这类随机数生成API不应直接比较数值。
""")
