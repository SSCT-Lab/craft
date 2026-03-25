#!/usr/bin/env python3
"""
测试 paddle.eye 的参数别名支持 (n, m)
"""
import paddle
import numpy as np

def test_paddle_eye_alias():
    """测试 paddle.eye 是否支持 n 和 m 作为参数别名"""
    print("="*70)
    print("测试 paddle.eye 参数别名支持")
    print("="*70)
    
    # 测试1: 使用标准参数名
    print("\n测试1: 使用标准参数名 (num_rows, num_columns)")
    print("-"*70)
    try:
        result1 = paddle.eye(num_rows=5, num_columns=3)
        print(f"✅ 成功: paddle.eye(num_rows=5, num_columns=3)")
        print(f"   形状: {result1.shape}")
        print(f"   矩阵:\n{result1.numpy()}")
    except Exception as e:
        print(f"❌ 失败: {e}")
    
    # 测试2: 使用别名 n 和 m
    print("\n测试2: 使用别名 (n, m)")
    print("-"*70)
    try:
        result2 = paddle.eye(n=5, m=3)
        print(f"✅ 成功: paddle.eye(n=5, m=3)")
        print(f"   形状: {result2.shape}")
        print(f"   矩阵:\n{result2.numpy()}")
    except Exception as e:
        print(f"❌ 失败: {e}")
        print(f"   说明: 当前 PaddlePaddle 版本可能不支持 n/m 别名")
        return False
    
    # 测试3: 比较两种方式的结果
    print("\n测试3: 比较两种参数方式的结果")
    print("-"*70)
    try:
        result_standard = paddle.eye(num_rows=5, num_columns=3)
        result_alias = paddle.eye(n=5, m=3)
        
        if np.array_equal(result_standard.numpy(), result_alias.numpy()):
            print(f"✅ 两种方式结果完全一致")
            print(f"   标准方式: paddle.eye(num_rows=5, num_columns=3)")
            print(f"   别名方式: paddle.eye(n=5, m=3)")
            return True
        else:
            print(f"❌ 两种方式结果不一致")
            return False
    except Exception as e:
        print(f"❌ 比较失败: {e}")
        return False

def test_paddle_version():
    """显示 PaddlePaddle 版本信息"""
    print("\n"+"="*70)
    print("PaddlePaddle 版本信息")
    print("="*70)
    print(f"版本: {paddle.__version__}")
    print(f"位置: {paddle.__file__}")

if __name__ == "__main__":
    test_paddle_version()
    result = test_paddle_eye_alias()
    
    print("\n"+"="*70)
    if result:
        print("✅ paddle.eye 支持 n 和 m 作为参数别名")
    else:
        print("⚠️ paddle.eye 可能不支持 n 和 m 别名，或版本不兼容")
    print("="*70)
