"""
Fixes for test comparison functions Validate special case handling of empty results and NaN
"""

import numpy as np
import sys
from pathlib import Path

# Add project root directory to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import comparison function
from pt_tf_test.fuzzing.llm_fuzzing_diff_test_concurrent import compare_results


def test_empty_containers():
    """Test empty container comparison（tuple vs list）"""
    print("test 1: Empty container comparison")
    
    torch_result = {
        "success": True,
        "result": (),  # null tuple
        "shape": None,
        "dtype": "<class 'tuple'>",
        "error": None
    }
    
    tf_result = {
        "success": True,
        "result": [],  # null list
        "shape": None,
        "dtype": "<class 'list'>",
        "error": None
    }
    
    comparison = compare_results(torch_result, tf_result)
    
    print(f"  torch_result: {torch_result['result']}")
    print(f"  tf_result: {tf_result['result']}")
    print(f"  results_match: {comparison['results_match']}")
    print(f"  comparison_error: {comparison['comparison_error']}")
    
    assert comparison['results_match'] == True, "Empty containers should be considered consistent"
    print("  ✓ Test passed\n")


def test_scalar_nan():
    """Test scalar NaN comparison"""
    print("test 2: Scalar NaN comparison")
    
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
    
    assert comparison['results_match'] == True, "Two NaN should be considered consistent"
    print("  ✓ Test passed\n")


def test_scalar_inf():
    """Test scalar Inf comparison"""
    print("test 3: Scalar Inf comparison")
    
    # Test positive infinity
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
    
    assert comparison['results_match'] == True, "Two positive infinities should be judged to be consistent"
    print("  ✓ Test passed\n")


def test_container_with_tensors():
    """Test comparison of containers containing tensors"""
    print("test 4: Comparison of containers containing tensors")
    
    # Create a tensor containing NaN
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
    
    assert comparison['results_match'] == True, "Containers containing the same tensor should be considered consistent"
    print("  ✓ Test passed\n")


def test_array_with_nan():
    """Testing array comparisons containing NaN"""
    print("test 5: Array comparison containing NaN")
    
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
    
    assert comparison['results_match'] == True, "Arrays containing the same NaN positions should be considered consistent"
    print("  ✓ Test passed\n")


def test_different_nan_positions():
    """Test array comparison for different NaN positions"""
    print("test 6: Array comparison of different NaN positions")
    
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
    
    assert comparison['results_match'] == False, "Different NaN positions should be judged as inconsistent"
    assert "NaN Inconsistent location" in comparison['comparison_error'], "NaN positional inconsistencies should be reported"
    print("  ✓ Test passed\n")


def main():
    """Run all tests"""
    print("=" * 70)
    print("Fixes for test comparison functions")
    print("=" * 70 + "\n")
    
    try:
        test_empty_containers()
        test_scalar_nan()
        test_scalar_inf()
        test_container_with_tensors()
        test_array_with_nan()
        test_different_nan_positions()
        
        print("=" * 70)
        print("All tests passed！✓")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Test error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
