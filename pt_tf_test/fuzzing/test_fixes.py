"""
测试 fuzzing 代码的修复
"""
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pt_tf_test.fuzzing.llm_fuzzing_diff_test_concurrent import (
    compare_results,
    needs_data_format_conversion,
    convert_nchw_to_nhwc,
    convert_nhwc_to_nchw,
)


def test_nan_comparison():
    """测试 NaN 比较逻辑"""
    print("测试 NaN 比较...")
    
    # 测试1: 两个标量 NaN 应该被认为一致
    torch_result = {
        "success": True,
        "result": np.nan,
        "shape": None,
        "dtype": "float32",
        "error": None
    }
    tf_result = {
        "success": True,
        "result": np.nan,
        "shape": None,
        "dtype": "float32",
        "error": None
    }
    
    comparison = compare_results(torch_result, tf_result)
    assert comparison["results_match"], "两个 NaN 应该被认为一致"
    print("  ✓ 标量 NaN 比较正确")
    
    # 测试2: 一个 NaN 一个普通值应该不一致
    tf_result["result"] = 1.0
    comparison = compare_results(torch_result, tf_result)
    assert not comparison["results_match"], "NaN 和普通值应该不一致"
    assert "NaN 不一致" in comparison["comparison_error"]
    print("  ✓ NaN 与普通值比较正确")
    
    # 测试3: 数组中的 NaN
    torch_result = {
        "success": True,
        "result": np.array([1.0, np.nan, 3.0]),
        "shape": [3],
        "dtype": "float32",
        "error": None
    }
    tf_result = {
        "success": True,
        "result": np.array([1.0, np.nan, 3.0]),
        "shape": [3],
        "dtype": "float32",
        "error": None
    }
    
    comparison = compare_results(torch_result, tf_result)
    assert comparison["results_match"], "包含 NaN 的数组应该一致"
    print("  ✓ 数组 NaN 比较正确")


def test_inf_comparison():
    """测试 Inf 比较逻辑"""
    print("\n测试 Inf 比较...")
    
    # 测试1: 两个正无穷应该一致
    torch_result = {
        "success": True,
        "result": np.inf,
        "shape": None,
        "dtype": "float32",
        "error": None
    }
    tf_result = {
        "success": True,
        "result": np.inf,
        "shape": None,
        "dtype": "float32",
        "error": None
    }
    
    comparison = compare_results(torch_result, tf_result)
    assert comparison["results_match"], "两个正无穷应该一致"
    print("  ✓ 正无穷比较正确")
    
    # 测试2: 正负无穷应该不一致
    tf_result["result"] = -np.inf
    comparison = compare_results(torch_result, tf_result)
    assert not comparison["results_match"], "正负无穷应该不一致"
    assert "Inf 符号不一致" in comparison["comparison_error"]
    print("  ✓ 正负无穷比较正确")


def test_data_format_conversion():
    """测试数据格式转换"""
    print("\n测试数据格式转换...")
    
    # 测试1: 检测需要转换的算子
    assert needs_data_format_conversion("torch.nn.Conv2d")
    assert needs_data_format_conversion("torch.nn.AvgPool2d")
    assert not needs_data_format_conversion("torch.abs")
    print("  ✓ 算子检测正确")
    
    # 测试2: NCHW -> NHWC (4D)
    nchw = np.random.randn(2, 3, 4, 5)  # (N, C, H, W)
    nhwc = convert_nchw_to_nhwc(nchw)
    assert nhwc.shape == (2, 4, 5, 3), f"期望 (2, 4, 5, 3)，得到 {nhwc.shape}"
    print("  ✓ NCHW -> NHWC 转换正确")
    
    # 测试3: NHWC -> NCHW (4D)
    nchw_back = convert_nhwc_to_nchw(nhwc)
    assert nchw_back.shape == nchw.shape
    assert np.allclose(nchw_back, nchw)
    print("  ✓ NHWC -> NCHW 转换正确")
    
    # 测试4: 3D 转换
    ncl = np.random.randn(2, 3, 10)  # (N, C, L)
    nlc = convert_nchw_to_nhwc(ncl)
    assert nlc.shape == (2, 10, 3), f"期望 (2, 10, 3)，得到 {nlc.shape}"
    print("  ✓ 3D 转换正确")
    
    # 测试5: 5D 转换
    ncdhw = np.random.randn(2, 3, 4, 5, 6)  # (N, C, D, H, W)
    ndhwc = convert_nchw_to_nhwc(ncdhw)
    assert ndhwc.shape == (2, 4, 5, 6, 3), f"期望 (2, 4, 5, 6, 3)，得到 {ndhwc.shape}"
    print("  ✓ 5D 转换正确")


def test_normal_value_comparison():
    """测试普通数值比较"""
    print("\n测试普通数值比较...")
    
    # 测试1: 相同的值
    torch_result = {
        "success": True,
        "result": 1.5,
        "shape": None,
        "dtype": "float32",
        "error": None
    }
    tf_result = {
        "success": True,
        "result": 1.5,
        "shape": None,
        "dtype": "float32",
        "error": None
    }
    
    comparison = compare_results(torch_result, tf_result)
    assert comparison["results_match"], "相同的值应该一致"
    print("  ✓ 相同值比较正确")
    
    # 测试2: 在容差范围内的值
    tf_result["result"] = 1.5 + 1e-6
    comparison = compare_results(torch_result, tf_result)
    assert comparison["results_match"], "容差范围内的值应该一致"
    print("  ✓ 容差范围内比较正确")
    
    # 测试3: 超出容差的值
    tf_result["result"] = 2.0
    comparison = compare_results(torch_result, tf_result)
    assert not comparison["results_match"], "超出容差的值应该不一致"
    assert "数值不一致" in comparison["comparison_error"]
    print("  ✓ 超出容差比较正确")


def main():
    """运行所有测试"""
    print("=" * 70)
    print("运行 Fuzzing 修复测试")
    print("=" * 70)
    
    try:
        test_nan_comparison()
        test_inf_comparison()
        test_data_format_conversion()
        test_normal_value_comparison()
        
        print("\n" + "=" * 70)
        print("✓ 所有测试通过！")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
