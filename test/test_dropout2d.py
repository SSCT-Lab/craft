import torch
import paddle
import numpy as np

def test_dropout2d_functional():
    print("=" * 70)
    print("torch.nn.functional.dropout2d vs paddle.nn.functional.dropout2d")
    print("=" * 70)
    
    # 创建测试输入
    np.random.seed(42)
    input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)
    
    print("\n输入数据:")
    print(f"Shape: {input_np.shape}")
    print(f"Mean: {input_np.mean():.6f}, Std: {input_np.std():.6f}")
    print(f"Min: {input_np.min():.6f}, Max: {input_np.max():.6f}")
    
    # 测试1: training=False (评估模式)
    print("\n" + "=" * 70)
    print("测试1: training=False (评估模式)")
    print("=" * 70)
    
    torch_input = torch.from_numpy(input_np.copy())
    paddle_input = paddle.to_tensor(input_np.copy())
    
    torch_output = torch.nn.functional.dropout2d(torch_input, p=0.5, training=False)
    paddle_output = paddle.nn.functional.dropout2d(paddle_input, p=0.5, training=False)
    
    torch_out_np = torch_output.numpy()
    paddle_out_np = paddle_output.numpy()
    
    print(f"PyTorch输出 - Mean: {torch_out_np.mean():.6f}, Std: {torch_out_np.std():.6f}")
    print(f"Paddle输出  - Mean: {paddle_out_np.mean():.6f}, Std: {paddle_out_np.std():.6f}")
    print(f"最大差异: {np.abs(torch_out_np - paddle_out_np).max():.6f}")
    print(f"是否相同: {np.allclose(torch_out_np, paddle_out_np, atol=1e-6)}")
    print(f"输出是否等于输入: {np.allclose(torch_out_np, input_np, atol=1e-6)}")
    
    # 测试2: training=True (训练模式) - 多次运行
    print("\n" + "=" * 70)
    print("测试2: training=True (训练模式) - 多次运行")
    print("=" * 70)
    
    for i in range(5):
        input_np_i = np.random.randn(2, 3, 4, 4).astype(np.float32)
        torch_input_i = torch.from_numpy(input_np_i.copy())
        paddle_input_i = paddle.to_tensor(input_np_i.copy())
        
        torch_out = torch.nn.functional.dropout2d(torch_input_i, p=0.5, training=True)
        paddle_out = paddle.nn.functional.dropout2d(paddle_input_i, p=0.5, training=True)
        
        torch_np = torch_out.numpy()
        paddle_np = paddle_out.numpy()
        
        diff = np.abs(torch_np - paddle_np).max()
        
        # 统计被dropout的通道数
        torch_zeros = np.sum(np.all(torch_np == 0, axis=(2, 3)))
        paddle_zeros = np.sum(np.all(paddle_np == 0, axis=(2, 3)))
        
        print(f"Run {i+1}:")
        print(f"  最大差异: {diff:.6f}")
        print(f"  PyTorch被置零的通道数: {torch_zeros}/6")
        print(f"  Paddle被置零的通道数: {paddle_zeros}/6")
    
    # 测试3: 不同的p值
    print("\n" + "=" * 70)
    print("测试3: 不同的dropout概率p (training=False)")
    print("=" * 70)
    
    for p in [0.0, 0.3, 0.5, 0.7, 1.0]:
        torch_input = torch.from_numpy(input_np.copy())
        paddle_input = paddle.to_tensor(input_np.copy())
        
        torch_out = torch.nn.functional.dropout2d(torch_input, p=p, training=False)
        paddle_out = paddle.nn.functional.dropout2d(paddle_input, p=p, training=False)
        
        diff = np.abs(torch_out.numpy() - paddle_out.numpy()).max()
        print(f"p={p}: 最大差异 = {diff:.6f}, 是否相同 = {np.allclose(torch_out.numpy(), paddle_out.numpy())}")
    
    # 测试4: 观察训练模式下的缩放行为
    print("\n" + "=" * 70)
    print("测试4: 训练模式下的缩放行为 (p=0.5)")
    print("=" * 70)
    
    input_simple = np.ones((1, 2, 3, 3), dtype=np.float32) * 2.0
    torch_input = torch.from_numpy(input_simple.copy())
    paddle_input = paddle.to_tensor(input_simple.copy())
    
    torch.manual_seed(123)
    paddle.seed(123)
    
    torch_out = torch.nn.functional.dropout2d(torch_input, p=0.5, training=True)
    paddle_out = paddle.nn.functional.dropout2d(paddle_input, p=0.5, training=True)
    
    print(f"输入值: 全部为 2.0")
    print(f"PyTorch输出的非零值: {torch_out.numpy()[torch_out.numpy() != 0][:5]}")
    print(f"Paddle输出的非零值: {paddle_out.numpy()[paddle_out.numpy() != 0][:5]}")
    print(f"期望的缩放值 (2.0 / (1-0.5)): {2.0 / 0.5}")
    
    print("\n" + "=" * 70)
    print("结论:")
    print("=" * 70)
    print("1. training=False时，dropout不起作用，输出应该等于输入")
    print("2. training=True时，dropout随机丢弃整个通道，每次结果不同")
    print("3. 这是正常行为，因为dropout使用随机mask")
    print("4. 保留的值会被缩放 1/(1-p) 以保持期望不变")
    print("=" * 70)

if __name__ == "__main__":
    test_dropout2d_functional()
