#!/usr/bin/env python3
"""
简单测试 torch.eye 和 paddle.eye 的输出一致性
"""
import torch
import paddle
import numpy as np

def test_eye_case(n, m=None, dtype_torch=None, dtype_paddle=None, case_name=""):
    """测试单个用例"""
    print(f"\n{'='*70}")
    print(f"测试用例: {case_name}")
    print(f"{'='*70}")
    print(f"参数: n={n}, m={m}")
    
    # PyTorch
    try:
        if dtype_torch:
            torch_result = torch.eye(n, m, dtype=dtype_torch)
        else:
            torch_result = torch.eye(n, m) if m else torch.eye(n)
        
        torch_np = torch_result.numpy()
        print(f"\n✅ PyTorch 执行成功")
        print(f"   形状: {torch_result.shape}")
        print(f"   dtype: {torch_result.dtype}")
        print(f"   前3x3元素:\n{torch_np[:3, :3]}")
    except Exception as e:
        print(f"\n❌ PyTorch 执行失败: {e}")
        return False
    
    # PaddlePaddle
    try:
        if dtype_paddle:
            paddle_result = paddle.eye(n, m, dtype=dtype_paddle)
        else:
            paddle_result = paddle.eye(n, m) if m else paddle.eye(n)
        
        paddle_np = paddle_result.numpy()
        print(f"\n✅ PaddlePaddle 执行成功")
        print(f"   形状: {paddle_result.shape}")
        print(f"   dtype: {paddle_result.dtype}")
        print(f"   前3x3元素:\n{paddle_np[:3, :3]}")
    except Exception as e:
        print(f"\n❌ PaddlePaddle 执行失败: {e}")
        return False
    
    # 比较结果
    print(f"\n📊 结果比较:")
    
    # 比较形状
    if torch_np.shape == paddle_np.shape:
        print(f"   ✅ 形状一致: {torch_np.shape}")
    else:
        print(f"   ❌ 形状不一致: PyTorch {torch_np.shape} vs Paddle {paddle_np.shape}")
        return False
    
    # 比较数值
    if np.allclose(torch_np, paddle_np):
        print(f"   ✅ 数值完全一致")
        return True
    else:
        diff = np.abs(torch_np - paddle_np)
        max_diff = np.max(diff)
        print(f"   ❌ 数值不一致，最大差异: {max_diff}")
        print(f"   差异位置数量: {np.sum(diff > 1e-6)}")
        return False

def main():
    print("="*70)
    print("torch.eye vs paddle.eye 一致性测试")
    print("="*70)
    
    results = []
    
    # 测试用例1: 方阵 (n=5)
    result1 = test_eye_case(
        n=5,
        m=None,
        case_name="测试1: 方阵 5x5"
    )
    results.append(("测试1: 方阵 5x5", result1))
    
    # 测试用例2: 矩形矩阵 (n=10, m=5)
    result2 = test_eye_case(
        n=10,
        m=5,
        case_name="测试2: 矩形矩阵 10x5"
    )
    results.append(("测试2: 矩形矩阵 10x5", result2))
    
    # 测试用例3: 带dtype的方阵
    result3 = test_eye_case(
        n=8,
        m=None,
        dtype_torch=torch.float32,
        dtype_paddle=paddle.float32,
        case_name="测试3: 方阵 8x8 (float32)"
    )
    results.append(("测试3: 方阵 8x8 (float32)", result3))
    
    # 测试用例4: 带dtype的矩形矩阵
    result4 = test_eye_case(
        n=6,
        m=10,
        dtype_torch=torch.float64,
        dtype_paddle=paddle.float64,
        case_name="测试4: 矩形矩阵 6x10 (float64)"
    )
    results.append(("测试4: 矩形矩阵 6x10 (float64)", result4))
    
    # 测试用例5: 大矩阵
    result5 = test_eye_case(
        n=100,
        m=50,
        case_name="测试5: 大矩阵 100x50"
    )
    results.append(("测试5: 大矩阵 100x50", result5))
    
    # 总结
    print(f"\n{'='*70}")
    print("测试总结")
    print(f"{'='*70}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print(f"\n🎉 所有测试通过！torch.eye 和 paddle.eye 输出完全一致")
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
