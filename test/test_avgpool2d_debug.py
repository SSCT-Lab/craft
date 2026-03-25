#!/usr/bin/env python3
"""
调试 torch.nn.functional.avg_pool2d 和 paddle.nn.functional.avg_pool2d 的差异
"""

import torch
import paddle
import numpy as np

def test_avgpool2d_differences():
    """测试 avg_pool2d 的差异"""
    print("=" * 80)
    print("调试 torch.nn.functional.avg_pool2d vs paddle.nn.functional.avg_pool2d")
    print("=" * 80)
    
    # 固定随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    paddle.seed(42)
    
    # 创建测试输入 - 使用简单的已知数据
    input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)
    
    print(f"输入形状: {input_np.shape}")
    print(f"输入数据范围: [{input_np.min():.4f}, {input_np.max():.4f}]")
    
    # 测试不同的参数组合
    test_cases = [
        {"kernel_size": 3, "stride": 2, "padding": 1},
        {"kernel_size": [3, 2], "stride": [2, 1], "padding": 1},
        {"kernel_size": 2, "stride": 1, "padding": 0},
        {"kernel_size": 3, "stride": 2, "padding": 1, "ceil_mode": True},
        {"kernel_size": 3, "stride": 2, "padding": 1, "count_include_pad": False},
    ]
    
    for i, params in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"测试用例 {i+1}: {params}")
        print(f"{'='*60}")
        
        try:
            # PyTorch
            torch_input = torch.from_numpy(input_np.copy())
            torch_result = torch.nn.functional.avg_pool2d(torch_input, **params)
            torch_output = torch_result.numpy()
            
            print(f"PyTorch输出形状: {torch_output.shape}")
            print(f"PyTorch输出范围: [{torch_output.min():.6f}, {torch_output.max():.6f}]")
            print(f"PyTorch输出均值: {torch_output.mean():.6f}")
            
        except Exception as e:
            print(f"PyTorch执行失败: {e}")
            continue
        
        try:
            # PaddlePaddle
            paddle_input = paddle.to_tensor(input_np.copy())
            paddle_result = paddle.nn.functional.avg_pool2d(paddle_input, **params)
            paddle_output = paddle_result.numpy()
            
            print(f"Paddle输出形状: {paddle_output.shape}")
            print(f"Paddle输出范围: [{paddle_output.min():.6f}, {paddle_output.max():.6f}]")
            print(f"Paddle输出均值: {paddle_output.mean():.6f}")
            
        except Exception as e:
            print(f"Paddle执行失败: {e}")
            continue
        
        # 比较结果
        if torch_output.shape == paddle_output.shape:
            max_diff = np.abs(torch_output - paddle_output).max()
            mean_diff = np.abs(torch_output - paddle_output).mean()
            is_close = np.allclose(torch_output, paddle_output, atol=1e-5, rtol=1e-5)
            
            print(f"\n比较结果:")
            print(f"  形状匹配: ✅")
            print(f"  最大差异: {max_diff:.8f}")
            print(f"  平均差异: {mean_diff:.8f}")
            print(f"  数值接近: {'✅' if is_close else '❌'}")
            
            if not is_close:
                print(f"\n详细差异分析:")
                diff = torch_output - paddle_output
                print(f"  差异统计: min={diff.min():.6f}, max={diff.max():.6f}, std={diff.std():.6f}")
                
                # 显示前几个元素的对比
                print(f"\n前5个元素对比:")
                flat_torch = torch_output.flatten()[:5]
                flat_paddle = paddle_output.flatten()[:5]
                flat_diff = (flat_torch - flat_paddle)
                for j in range(5):
                    print(f"    [{j}] PyTorch: {flat_torch[j]:.8f}, Paddle: {flat_paddle[j]:.8f}, 差异: {flat_diff[j]:.8f}")
        else:
            print(f"\n比较结果:")
            print(f"  形状不匹配: PyTorch {torch_output.shape} vs Paddle {paddle_output.shape}")

def test_specific_case():
    """测试具体的失败案例"""
    print(f"\n{'='*80}")
    print("测试具体的失败案例")
    print(f"{'='*80}")
    
    # 使用固定的输入数据
    input_data = np.array([[[[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 10.0, 11.0, 12.0],
                            [13.0, 14.0, 15.0, 16.0]]]], dtype=np.float32)
    
    print(f"输入数据形状: {input_data.shape}")
    print(f"输入数据:\n{input_data[0, 0]}")
    
    # 测试参数：kernel_size=3, stride=2, padding=1
    params = {"kernel_size": 3, "stride": 2, "padding": 1}
    
    # PyTorch
    torch_input = torch.from_numpy(input_data.copy())
    torch_result = torch.nn.functional.avg_pool2d(torch_input, **params)
    torch_output = torch_result.numpy()
    
    # PaddlePaddle
    paddle_input = paddle.to_tensor(input_data.copy())
    paddle_result = paddle.nn.functional.avg_pool2d(paddle_input, **params)
    paddle_output = paddle_result.numpy()
    
    print(f"\nPyTorch输出形状: {torch_output.shape}")
    print(f"PyTorch输出:\n{torch_output[0, 0]}")
    
    print(f"\nPaddle输出形状: {paddle_output.shape}")
    print(f"Paddle输出:\n{paddle_output[0, 0]}")
    
    print(f"\n差异:\n{torch_output[0, 0] - paddle_output[0, 0]}")
    
    max_diff = np.abs(torch_output - paddle_output).max()
    print(f"\n最大差异: {max_diff:.8f}")

if __name__ == "__main__":
    test_avgpool2d_differences()
    test_specific_case()
