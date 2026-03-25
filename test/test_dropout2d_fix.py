#!/usr/bin/env python3
"""
测试Dropout2d修复
"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_pytorch_paddle import PyTorchPaddleComparator

def test_dropout2d():
    """测试Dropout2d算子"""
    print("=" * 70)
    print("测试 Dropout2d 修复")
    print("=" * 70)
    
    # 创建比较器
    comparator = PyTorchPaddleComparator()
    
    # 模拟数据库中的Dropout2d文档
    document = {
        "_id": "61b97c1ef3e4313e34852d61",
        "api": "torch.nn.Dropout2d",
        "p": [0.4, 0.4, 0.4, 0.2, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        "inplace": []
    }
    
    print(f"\n📋 测试文档:")
    print(f"  API: {document['api']}")
    print(f"  参数 p: {document['p'][:3]}... (共{len(document['p'])}个)")
    print(f"  参数 inplace: {document['inplace']}")
    
    # 测试第一个用例
    print(f"\n🧪 测试用例 1:")
    result = comparator.test_single_case(document, 0)
    
    print(f"\n📊 测试结果:")
    print(f"  状态: {result['status']}")
    print(f"  PyTorch API: {result['torch_api']}")
    print(f"  实际使用: {result.get('torch_api_used', 'N/A')}")
    print(f"  Paddle API: {result['paddle_api']}")
    print(f"  映射方法: {result['mapping_method']}")
    print(f"  PyTorch 成功: {result['torch_success']}")
    print(f"  Paddle 成功: {result['paddle_success']}")
    print(f"  结果匹配: {result['results_match']}")
    
    if result['torch_error']:
        print(f"  ❌ PyTorch 错误: {result['torch_error']}")
    else:
        print(f"  ✅ PyTorch 执行成功")
        print(f"     形状: {result['torch_shape']}")
        print(f"     dtype: {result['torch_dtype']}")
    
    if result['paddle_error']:
        print(f"  ❌ Paddle 错误: {result['paddle_error']}")
    else:
        print(f"  ✅ Paddle 执行成功")
        print(f"     形状: {result['paddle_shape']}")
        print(f"     dtype: {result['paddle_dtype']}")
    
    if result['comparison_error']:
        print(f"  ⚠️ 比较错误: {result['comparison_error']}")
    
    # 关闭连接
    comparator.close()
    
    print("\n" + "=" * 70)
    if result['torch_success'] and result['paddle_success']:
        print("✅ 测试通过：两个框架都成功执行")
    else:
        print("❌ 测试失败：存在执行错误")
    print("=" * 70)

if __name__ == "__main__":
    test_dropout2d()
