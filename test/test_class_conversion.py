#!/usr/bin/env python3
"""
测试类转函数功能
"""
import torch
import paddle
import numpy as np

# 测试Dropout2d
print("=" * 60)
print("测试 torch.nn.Dropout2d vs torch.nn.functional.dropout2d")
print("=" * 60)

# 创建测试数据
x_np = np.random.randn(2, 3, 4, 4).astype(np.float32)
x_torch = torch.from_numpy(x_np.copy())
x_paddle = paddle.to_tensor(x_np.copy())

# 测试类形式 (需要实例化)
print("\n1. 类形式 (torch.nn.Dropout2d):")
try:
    dropout_class = torch.nn.Dropout2d(p=0.5)
    with torch.no_grad():
        result_class = dropout_class(x_torch)
    print(f"   ✅ 成功执行")
    print(f"   输出形状: {result_class.shape}")
    print(f"   输出dtype: {result_class.dtype}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试函数形式
print("\n2. 函数形式 (torch.nn.functional.dropout2d):")
try:
    with torch.no_grad():
        result_func = torch.nn.functional.dropout2d(x_torch, p=0.5)
    print(f"   ✅ 成功执行")
    print(f"   输出形状: {result_func.shape}")
    print(f"   输出dtype: {result_func.dtype}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试PaddlePaddle函数形式
print("\n3. PaddlePaddle函数形式 (paddle.nn.functional.dropout2d):")
try:
    result_paddle = paddle.nn.functional.dropout2d(x_paddle, p=0.5)
    print(f"   ✅ 成功执行")
    print(f"   输出形状: {result_paddle.shape}")
    print(f"   输出dtype: {result_paddle.dtype}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试AvgPool2d
print("\n" + "=" * 60)
print("测试 torch.nn.AvgPool2d vs torch.nn.functional.avg_pool2d")
print("=" * 60)

# 测试类形式
print("\n1. 类形式 (torch.nn.AvgPool2d):")
try:
    pool_class = torch.nn.AvgPool2d(kernel_size=2)
    with torch.no_grad():
        result_class = pool_class(x_torch)
    print(f"   ✅ 成功执行")
    print(f"   输出形状: {result_class.shape}")
    print(f"   输出dtype: {result_class.dtype}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试函数形式
print("\n2. 函数形式 (torch.nn.functional.avg_pool2d):")
try:
    with torch.no_grad():
        result_func = torch.nn.functional.avg_pool2d(x_torch, kernel_size=2)
    print(f"   ✅ 成功执行")
    print(f"   输出形状: {result_func.shape}")
    print(f"   输出dtype: {result_func.dtype}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试PaddlePaddle函数形式
print("\n3. PaddlePaddle函数形式 (paddle.nn.functional.avg_pool2d):")
try:
    result_paddle = paddle.nn.functional.avg_pool2d(x_paddle, kernel_size=2)
    print(f"   ✅ 成功执行")
    print(f"   输出形状: {result_paddle.shape}")
    print(f"   输出dtype: {result_paddle.dtype}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
