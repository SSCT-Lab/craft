"""
Test fixes for fuzzing code
"""
import numpy as np
import sys
from pathlib import Path

# Add project root directory to path
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
    """Test NaN comparison logic"""
    print("Test for NaN comparison...")
    
    # test1: Two scalars NaN should be considered identical
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
    assert comparison["results_match"], "Two NaNs should be considered identical"
    print("  ✓ Scalar NaN is more correct")
    
    # test2: A NaN and a normal value should be inconsistent
    tf_result["result"] = 1.0
    comparison = compare_results(torch_result, tf_result)
    assert not comparison["results_match"], "NaN It should be inconsistent with the normal value"
    assert "NaN inconsistent" in comparison["comparison_error"]
    print("  ✓ NaN Correct compared to normal values")
    
    # test3: in array NaN
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
    assert comparison["results_match"], "Arrays containing NaN should be consistent"
    print("  ✓ Array NaN comparison is correct")


def test_inf_comparison():
    """Testing Inf comparison logic"""
    print("\nTest Inf Comparison...")
    
    # test1: Two positive infinities should be consistent
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
    assert comparison["results_match"], "Two positive infinities should be consistent"
    print("  ✓ Positive infinity is more correct")
    
    # test2: Positive and negative infinity should be inconsistent
    tf_result["result"] = -np.inf
    comparison = compare_results(torch_result, tf_result)
    assert not comparison["results_match"], "Positive and negative infinity should be inconsistent"
    assert "Inf Symbols are inconsistent" in comparison["comparison_error"]
    print("  ✓ Plus or minus infinity is more correct")


def test_data_format_conversion():
    """Test data format conversion"""
    print("\nTest data format conversion...")
    
    # test1: Detect operators that require conversion
    assert needs_data_format_conversion("torch.nn.Conv2d")
    assert needs_data_format_conversion("torch.nn.AvgPool2d")
    assert not needs_data_format_conversion("torch.abs")
    print("  ✓ Operator detection is correct")
    
    # test2: NCHW -> NHWC (4D)
    nchw = np.random.randn(2, 3, 4, 5)  # (N, C, H, W)
    nhwc = convert_nchw_to_nhwc(nchw)
    assert nhwc.shape == (2, 4, 5, 3), f"expect (2, 4, 5, 3)，get {nhwc.shape}"
    print("  ✓ NCHW -> NHWC Conversion is correct")
    
    # test3: NHWC -> NCHW (4D)
    nchw_back = convert_nhwc_to_nchw(nhwc)
    assert nchw_back.shape == nchw.shape
    assert np.allclose(nchw_back, nchw)
    print("  ✓ NHWC -> NCHW Conversion is correct")
    
    # test4: 3D Convert
    ncl = np.random.randn(2, 3, 10)  # (N, C, L)
    nlc = convert_nchw_to_nhwc(ncl)
    assert nlc.shape == (2, 10, 3), f"expect (2, 10, 3)，get {nlc.shape}"
    print("  ✓ 3D Conversion is correct")
    
    # test5: 5D Convert
    ncdhw = np.random.randn(2, 3, 4, 5, 6)  # (N, C, D, H, W)
    ndhwc = convert_nchw_to_nhwc(ncdhw)
    assert ndhwc.shape == (2, 4, 5, 6, 3), f"expect (2, 4, 5, 6, 3)，get {ndhwc.shape}"
    print("  ✓ 5D Conversion is correct")


def test_normal_value_comparison():
    """Test common numerical comparisons"""
    print("\nTest common numerical comparisons...")
    
    # test1: same value
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
    assert comparison["results_match"], "Identical values ​​should be consistent"
    print("  ✓ The same value is more correct")
    
    # test2: Value within tolerance
    tf_result["result"] = 1.5 + 1e-6
    comparison = compare_results(torch_result, tf_result)
    assert comparison["results_match"], "Values ​​within tolerance should be consistent"
    print("  ✓ Correct within tolerance")
    
    # test3: Value outside tolerance
    tf_result["result"] = 2.0
    comparison = compare_results(torch_result, tf_result)
    assert not comparison["results_match"], "Values ​​outside the tolerance should be inconsistent"
    assert "Values ​​are inconsistent" in comparison["comparison_error"]
    print("  ✓ More accurate than tolerance")


def main():
    """Run all tests"""
    print("=" * 70)
    print("Run the fuzzing fix test")
    print("=" * 70)
    
    try:
        test_nan_comparison()
        test_inf_comparison()
        test_data_format_conversion()
        test_normal_value_comparison()
        
        print("\n" + "=" * 70)
        print("✓ All tests passed！")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Test error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
