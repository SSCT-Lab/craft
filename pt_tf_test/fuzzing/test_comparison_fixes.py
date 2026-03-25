"""
测试比较函数的修复
验证空结果和 NaN 的特殊情况处理
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 导入比较函数
from pt_tf_test.fuzzing.llm_fuzzing_diff_test_concurrent import compare_results


def test_empty_containers():
    """测试空容器比较（tuple vs list）"""
    print("测试 1: 空容器比较")
    
    torch_result = {
        "success": True,
        "result": (),  # 空 tuple
        "shape": None,
        "dtype": "<class 'tuple'>",
        "error": None
    }
    
    tf_result = {
        "success": True,
        "result": [],  # 空 list
        "shape": None,
        "dtype": "<class 'list'>",
        "error": None
    }
    
    comparison = compare_results(torch_result, tf_result)
    
    print(f"  torch_result: {torch_result['result']}")
    print(f"  tf_result: {tf_result['result']}")
    print(f"  results_match: {comparison['results_match']}")
    print(f"  comparison_error: {comparison['comparison_error']}")
    
    assert comparison['results_match'] == True, "空容器应该判定为一致"
    print("  ✓ 测试通过\n")


def test_scalar_nan():
    """测试标量 NaN 比较"""
    print("测试 2: 标量 NaN 比较")
    
    torch_result = {
        "success": True,
        "result": np.nan,
        "shape": [],
        "dtype": "torch.float32",
        "error": None
    }
    
    tf_result = {
        "success": True,
        "result": np.nan,
        "shape": [],
        "dtype": "<dtype: 'float32'>",
        "error": None
    }
    
    comparison = compare_results(torch_result, tf_result)
    
    print(f"  torch_result: {torch_result['result']}")
    print(f"  tf_result: {tf_result['result']}")
    print(f"  results_match: {comparison['results_match']}")
    print(f"  comparison_error: {comparison['comparison_error']}")
    
    assert comparison['results_match'] == True, "两个 NaN 应该判定为一致"
    print("  ✓ 测试通过\n")


def test_scalar_inf():
    """测试标量 Inf 比较"""
    print("测试 3: 标量 Inf 比较")
    
    # 测试正无穷
    torch_result = {
        "success": True,
        "result": np.inf,
        "shape": [],
        "dtype": "torch.float32",
        "error": None
    }
    
    tf_result = {
        "success": True,
        "result": np.inf,
        "shape": [],
        "dtype": "<dtype: 'float32'>",
        "error": None
    }
    
    comparison = compare_results(torch_result, tf_result)
    
    print(f"  torch_result: {torch_result['result']}")
    print(f"  tf_result: {tf_result['result']}")
    print(f"  results_match: {comparison['results_match']}")
    print(f"  comparison_error: {comparison['comparison_error']}")
    
    assert comparison['results_match'] == True, "两个正无穷应该判定为一致"
    print("  ✓ 测试通过\n")


def test_container_with_tensors():
    """测试包含张量的容器比较"""
    print("测试 4: 包含张量的容器比较")
    
    # 创建包含 NaN 的张量
    torch_tensor = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    tf_tensor = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    
    torch_result = {
        "success": True,
        "result": (torch_tensor,),  # tuple
        "shape": None,
        "dtype": "<class 'tuple'>",
        "error": None
    }
    
    tf_result = {
        "success": True,
        "result": [tf_tensor],  # list
        "shape": None,
        "dtype": "<class 'list'>",
        "error": None
    }
    
    comparison = compare_results(torch_result, tf_result)
    
    print(f"  torch_result: tuple with tensor shape {torch_tensor.shape}")
    print(f"  tf_result: list with tensor shape {tf_tensor.shape}")
    print(f"  results_match: {comparison['results_match']}")
    print(f"  comparison_error: {comparison['comparison_error']}")
    
    assert comparison['results_match'] == True, "包含相同张量的容器应该判定为一致"
    print("  ✓ 测试通过\n")


def test_array_with_nan():
    """测试包含 NaN 的数组比较"""
    print("测试 5: 包含 NaN 的数组比较")
    
    torch_array = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    tf_array = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    
    torch_result = {
        "success": True,
        "result": torch_array,
        "shape": [3],
        "dtype": "torch.float32",
        "error": None
    }
    
    tf_result = {
        "success": True,
        "result": tf_array,
        "shape": [3],
        "dtype": "<dtype: 'float32'>",
        "error": None
    }
    
    comparison = compare_results(torch_result, tf_result)
    
    print(f"  torch_result: {torch_result['result']}")
    print(f"  tf_result: {tf_result['result']}")
    print(f"  results_match: {comparison['results_match']}")
    print(f"  comparison_error: {comparison['comparison_error']}")
    
    assert comparison['results_match'] == True, "包含相同 NaN 位置的数组应该判定为一致"
    print("  ✓ 测试通过\n")


def test_different_nan_positions():
    """测试不同 NaN 位置的数组比较"""
    print("测试 6: 不同 NaN 位置的数组比较")
    
    torch_array = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    tf_array = np.array([1.0, 2.0, np.nan], dtype=np.float32)
    
    torch_result = {
        "success": True,
        "result": torch_array,
        "shape": [3],
        "dtype": "torch.float32",
        "error": None
    }
    
    tf_result = {
        "success": True,
        "result": tf_array,
        "shape": [3],
        "dtype": "<dtype: 'float32'>",
        "error": None
    }
    
    comparison = compare_results(torch_result, tf_result)
    
    print(f"  torch_result: {torch_result['result']}")
    print(f"  tf_result: {tf_result['result']}")
    print(f"  results_match: {comparison['results_match']}")
    print(f"  comparison_error: {comparison['comparison_error']}")
    
    assert comparison['results_match'] == False, "不同 NaN 位置应该判定为不一致"
    assert "NaN 位置不一致" in comparison['comparison_error'], "应该报告 NaN 位置不一致"
    print("  ✓ 测试通过\n")


def main():
    """运行所有测试"""
    print("=" * 70)
    print("测试比较函数的修复")
    print("=" * 70 + "\n")
    
    try:
        test_empty_containers()
        test_scalar_nan()
        test_scalar_inf()
        test_container_with_tensors()
        test_array_with_nan()
        test_different_nan_positions()
        
        print("=" * 70)
        print("所有测试通过！✓")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
