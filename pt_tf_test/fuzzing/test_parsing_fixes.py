"""
Test LLM response parsing fix  This script tests the newly implemented parsing repair functionality, including:
1. Python float() syntax substitution
2. JSON verify
3. Truncation fix
"""

import json
import sys
from pathlib import Path

# Add project root directory to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import repair function
from pt_tf_test.fuzzing.llm_fuzzing_diff_test_concurrent import (
    fix_python_float_syntax,
    validate_parsed_json,
    parse_llm_response,
    try_repair_json
)


def test_fix_python_float_syntax():
    """Testing Python float() syntax replacement"""
    print("\n" + "="*70)
    print("test 1: Python float() syntax substitution")
    print("="*70)
    
    test_cases = [
        {
            "input": '"sample_values": [0.0, float(\'inf\'), float(\'-inf\'), float(\'nan\')]',
            "expected_contains": ['"inf"', '"-inf"', '"nan"'],
            "should_not_contain": ["float("]
        },
        {
            "input": '[float("inf"), float("-inf"), float("nan"), float("+inf")]',
            "expected_contains": ['"inf"', '"-inf"', '"nan"'],
            "should_not_contain": ["float("]
        },
        {
            "input": 'float( \'inf\' ), float( \'-inf\' )',
            "expected_contains": ['"inf"', '"-inf"'],
            "should_not_contain": ["float("]
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        result = fix_python_float_syntax(test["input"])
        
        # Check what is expected
        all_present = all(exp in result for exp in test["expected_contains"])
        # Check what should not be included
        none_present = all(bad not in result for bad in test["should_not_contain"])
        
        if all_present and none_present:
            print(f"  ✓ test case {i} pass")
            passed += 1
        else:
            print(f"  ✗ test case {i} fail")
            print(f"    enter: {test['input'][:80]}...")
            print(f"    output: {result[:80]}...")
            failed += 1
    
    print(f"\nresult: {passed} pass, {failed} fail")
    return failed == 0


def test_validate_parsed_json():
    """Test JSON validation"""
    print("\n" + "="*70)
    print("test 2: JSON verify")
    print("="*70)
    
    test_cases = [
        {
            "input": {
                "torch_test_case": {"api": "torch.abs"},
                "tensorflow_test_case": {"api": "tf.abs"}
            },
            "expected": True,
            "description": "Complete and valid JSON"
        },
        {
            "input": {
                "torch_test_case": {"api": "torch.abs"}
            },
            "expected": False,
            "description": "Lack tensorflow_test_case"
        },
        {
            "input": {
                "torch_test_case": "not a dict",
                "tensorflow_test_case": {"api": "tf.abs"}
            },
            "expected": False,
            "description": "torch_test_case not a dictionary"
        },
        {
            "input": "not a dict",
            "expected": False,
            "description": "Not a dictionary at all"
        },
        {
            "input": {},
            "expected": False,
            "description": "empty dictionary"
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        result = validate_parsed_json(test["input"])
        
        if result == test["expected"]:
            print(f"  ✓ test case {i} pass: {test['description']}")
            passed += 1
        else:
            print(f"  ✗ test case {i} fail: {test['description']}")
            print(f"    expect: {test['expected']}, actual: {result}")
            failed += 1
    
    print(f"\nresult: {passed} pass, {failed} fail")
    return failed == 0


def test_parse_llm_response():
    """Test full LLM response parsing"""
    print("\n" + "="*70)
    print("test 3: Full LLM response parsing")
    print("="*70)
    
    test_cases = [
        {
            "name": "Response containing Python float() syntax",
            "input": '''```json
{
  "mutation_strategy": "test",
  "torch_test_case": {
    "api": "torch.abs",
    "input": {
      "shape": [10],
      "dtype": "float32",
      "sample_values": [0.0, float('inf'), float('-inf'), float('nan')]
    }
  },
  "tensorflow_test_case": {
    "api": "tf.abs",
    "input": {
      "shape": [10],
      "dtype": "float32",
      "sample_values": [0.0, float('inf'), float('-inf'), float('nan')]
    }
  }
}
```''',
            "should_parse": True
        },
        {
            "name": "Truncated JSON (missing closing bracket)",
            "input": '''```json
{
  "mutation_strategy": "test",
  "torch_test_case": {
    "api": "torch.abs",
    "input": {
      "shape": [10],
      "dtype": "float32",
      "sample_values": [0.0, "inf", "-inf"
```''',
            "should_parse": True
        },
        {
            "name": "completely valid JSON",
            "input": '''```json
{
  "mutation_strategy": "test",
  "torch_test_case": {
    "api": "torch.abs",
    "input": {"shape": [10], "dtype": "float32", "sample_values": [0.0]}
  },
  "tensorflow_test_case": {
    "api": "tf.abs",
    "input": {"shape": [10], "dtype": "float32", "sample_values": [0.0]}
  }
}
```''',
            "should_parse": True
        },
        {
            "name": "Invalid JSON (missing required fields)",
            "input": '''```json
{
  "mutation_strategy": "test",
  "torch_test_case": {
    "api": "torch.abs"
  }
}
```''',
            "should_parse": False
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        result = parse_llm_response(test["input"])
        
        parsed_successfully = result is not None
        
        if parsed_successfully == test["should_parse"]:
            print(f"  ✓ test case {i} pass: {test['name']}")
            if result:
                print(f"    The parsed result contains fields: {list(result.keys())}")
            passed += 1
        else:
            print(f"  ✗ test case {i} fail: {test['name']}")
            print(f"    expect parsing: {test['should_parse']}, actual: {parsed_successfully}")
            failed += 1
    
    print(f"\nresult: {passed} pass, {failed} fail")
    return failed == 0


def test_real_failure_case():
    """Test real failure cases"""
    print("\n" + "="*70)
    print("test 4: Real failure cases")
    print("="*70)
    
    # from torch_abs_fuzzing_result_20260131_215615.json The real failure response extracted from
    real_failure = '''```json
{
  "mutation_strategy": "Extreme numerical variation: While maintaining mathematical equivalence, replace sample_values ​​with a set covering floating point boundaries and special values, including positive and negative zero, extremely small positive numbers (1e-38), extremely large positive numbers (1e38), and positive and negative infinity、NaN",
  "mutation_reason": "abs Function mathematically for all real and extended real numbers（±0, ±inf, NaN）clearly defined：abs(±0)=0, abs(±inf)=inf, abs(NaN)=NaN。But different frameworks may have problems with subnormal numbers, signed zero propagation, NaN value types (quiet vs signaling), or inf/NaN There are differences in bit-level processing of IEEE 754. Both PyTorch and TensorFlow are based on LLVM/LLVM-like or -0.0 Some backends may incorrectly preserve signs (violating IEEE 754 abs definitions), or trigger non-standard rounding on subnormal input. This mutation can expose compliance introduced by the underlying numerical library or compiler optimizations bug。",
  "torch_test_case": {
    "api": "torch.abs",
    "input": {
      "shape": [10],
      "dtype": "float32",
      "sample_values": [0.0, -0.0, 1e-38, -1e-38, 1e38, -1e38, 0.0, float('inf'), float('-inf'), float('nan')]
    }
  },
  "tensorflow_test_case": {
    "api": "tf.abs",
    "input": {
      "shape": [10],
      "dtype": "float32",
      "sample_values": [0.0, -0.0, 1e-38, -1e-38, 1e38, -1e38, 0.0, float('inf'), float('-inf'), float('nan')]
    }
  }
}
```'''
    
    print("  Test real failure cases...")
    result = parse_llm_response(real_failure)
    
    if result is not None:
        print("  ✓ Successful analysis of real failure cases")
        print(f"    The parsed result contains fields: {list(result.keys())}")
        
        # Check if sample_values ​​are converted correctly
        torch_values = result.get("torch_test_case", {}).get("input", {}).get("sample_values", [])
        print(f"    torch sample_values: {torch_values[:5]}...")
        
        # Check if there is also float() syntax
        json_str = json.dumps(result)
        if "float(" in json_str:
            print("  ⚠️ warn: The parsing result still contains float() syntax")
            return False
        else:
            print("  ✓ Successfully removed all float() syntax")
            return True
    else:
        print("  ✗ Failed to parse real failure cases")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("LLM Response parsing fix function test")
    print("="*70)
    
    results = []
    
    results.append(("Python float() syntax substitution", test_fix_python_float_syntax()))
    results.append(("JSON verify", test_validate_parsed_json()))
    results.append(("Full LLM response parsing", test_parse_llm_response()))
    results.append(("Real failure cases", test_real_failure_case()))
    
    print("\n" + "="*70)
    print("Test summary")
    print("="*70)
    
    for name, passed in results:
        status = "✓ pass" if passed else "✗ fail"
        print(f"  {status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Repair function works fine。")
        return 0
    else:
        print("\n⚠️ Some tests failed and require further debugging。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
