"""
测试 LLM 响应解析修复功能

这个脚本测试新实现的解析修复功能，包括:
1. Python float() 语法替换
2. JSON 验证
3. 截断修复
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 导入修复函数
from pt_tf_test.fuzzing.llm_fuzzing_diff_test_concurrent import (
    fix_python_float_syntax,
    validate_parsed_json,
    parse_llm_response,
    try_repair_json
)


def test_fix_python_float_syntax():
    """测试 Python float() 语法替换"""
    print("\n" + "="*70)
    print("测试 1: Python float() 语法替换")
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
        
        # 检查期望包含的内容
        all_present = all(exp in result for exp in test["expected_contains"])
        # 检查不应包含的内容
        none_present = all(bad not in result for bad in test["should_not_contain"])
        
        if all_present and none_present:
            print(f"  ✓ 测试用例 {i} 通过")
            passed += 1
        else:
            print(f"  ✗ 测试用例 {i} 失败")
            print(f"    输入: {test['input'][:80]}...")
            print(f"    输出: {result[:80]}...")
            failed += 1
    
    print(f"\n结果: {passed} 通过, {failed} 失败")
    return failed == 0


def test_validate_parsed_json():
    """测试 JSON 验证"""
    print("\n" + "="*70)
    print("测试 2: JSON 验证")
    print("="*70)
    
    test_cases = [
        {
            "input": {
                "torch_test_case": {"api": "torch.abs"},
                "tensorflow_test_case": {"api": "tf.abs"}
            },
            "expected": True,
            "description": "完整的有效 JSON"
        },
        {
            "input": {
                "torch_test_case": {"api": "torch.abs"}
            },
            "expected": False,
            "description": "缺少 tensorflow_test_case"
        },
        {
            "input": {
                "torch_test_case": "not a dict",
                "tensorflow_test_case": {"api": "tf.abs"}
            },
            "expected": False,
            "description": "torch_test_case 不是字典"
        },
        {
            "input": "not a dict",
            "expected": False,
            "description": "根本不是字典"
        },
        {
            "input": {},
            "expected": False,
            "description": "空字典"
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        result = validate_parsed_json(test["input"])
        
        if result == test["expected"]:
            print(f"  ✓ 测试用例 {i} 通过: {test['description']}")
            passed += 1
        else:
            print(f"  ✗ 测试用例 {i} 失败: {test['description']}")
            print(f"    期望: {test['expected']}, 实际: {result}")
            failed += 1
    
    print(f"\n结果: {passed} 通过, {failed} 失败")
    return failed == 0


def test_parse_llm_response():
    """测试完整的 LLM 响应解析"""
    print("\n" + "="*70)
    print("测试 3: 完整 LLM 响应解析")
    print("="*70)
    
    test_cases = [
        {
            "name": "包含 Python float() 语法的响应",
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
            "name": "截断的 JSON (缺少闭合括号)",
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
            "name": "完全有效的 JSON",
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
            "name": "无效的 JSON (缺少必要字段)",
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
            print(f"  ✓ 测试用例 {i} 通过: {test['name']}")
            if result:
                print(f"    解析结果包含字段: {list(result.keys())}")
            passed += 1
        else:
            print(f"  ✗ 测试用例 {i} 失败: {test['name']}")
            print(f"    期望解析: {test['should_parse']}, 实际: {parsed_successfully}")
            failed += 1
    
    print(f"\n结果: {passed} 通过, {failed} 失败")
    return failed == 0


def test_real_failure_case():
    """测试真实的失败案例"""
    print("\n" + "="*70)
    print("测试 4: 真实失败案例")
    print("="*70)
    
    # 从 torch_abs_fuzzing_result_20260131_215615.json 中提取的真实失败响应
    real_failure = '''```json
{
  "mutation_strategy": "极端数值变异：在保持数学等价前提下，将 sample_values 替换为涵盖浮点边界与特殊值的集合，包括正负零、极小正数(1e-38)、极大正数(1e38)、正负无穷、NaN",
  "mutation_reason": "abs 函数在数学上对所有实数和扩展实数（±0, ±inf, NaN）有明确定义：abs(±0)=0, abs(±inf)=inf, abs(NaN)=NaN。但不同框架在底层实现中可能对 subnormal 数、符号零传播、NaN 值类型（quiet vs signaling）、或 inf/NaN 的 IEEE 754 位级处理存在差异。PyTorch 和 TensorFlow 均基于 LLVM/LLVM-like 或 XLA 后端，但数值库（如 MKL、Eigen、CUDNN）对特殊值的处理路径可能不一致；尤其当输入含 -0.0 时，某些后端可能错误保留符号（违反 IEEE 754 abs 定义），或对 subnormal 输入触发非标准舍入。此变异能暴露底层数值库或编译器优化引入的合规性 bug。",
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
    
    print("  测试真实失败案例...")
    result = parse_llm_response(real_failure)
    
    if result is not None:
        print("  ✓ 成功解析真实失败案例")
        print(f"    解析结果包含字段: {list(result.keys())}")
        
        # 检查 sample_values 是否正确转换
        torch_values = result.get("torch_test_case", {}).get("input", {}).get("sample_values", [])
        print(f"    torch sample_values: {torch_values[:5]}...")
        
        # 检查是否还有 float() 语法
        json_str = json.dumps(result)
        if "float(" in json_str:
            print("  ⚠️ 警告: 解析结果中仍包含 float() 语法")
            return False
        else:
            print("  ✓ 已成功移除所有 float() 语法")
            return True
    else:
        print("  ✗ 解析真实失败案例失败")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("LLM 响应解析修复功能测试")
    print("="*70)
    
    results = []
    
    results.append(("Python float() 语法替换", test_fix_python_float_syntax()))
    results.append(("JSON 验证", test_validate_parsed_json()))
    results.append(("完整 LLM 响应解析", test_parse_llm_response()))
    results.append(("真实失败案例", test_real_failure_case()))
    
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！修复功能正常工作。")
        return 0
    else:
        print("\n⚠️ 部分测试失败，需要进一步调试。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
