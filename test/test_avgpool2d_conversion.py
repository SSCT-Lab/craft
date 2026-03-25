#!/usr/bin/env python3
"""
测试AvgPool2d转换逻辑
"""
import re
from typing import Optional, Tuple

def is_class_based_api(api_name: str) -> bool:
    """判断API是否是基于类的"""
    parts = api_name.split(".")
    if len(parts) >= 2:
        name = parts[-1]
        return any(c.isupper() for c in name)
    return False

def convert_class_to_functional(torch_api: str) -> Tuple[Optional[str], Optional[str]]:
    """将类形式的API转换为函数形式"""
    if not is_class_based_api(torch_api):
        return None, None
    
    parts = torch_api.split(".")
    if len(parts) >= 3 and parts[1] == "nn":
        # 获取类名并转换为snake_case
        class_name = parts[-1]
        
        # 将驼峰命名转换为下划线命名
        # 在大写字母前插入下划线（不在开头和数字后）
        func_name = re.sub(r'(?<!^)(?<![0-9])([A-Z])', r'_\1', class_name).lower()
        
        # 构建torch functional API
        torch_func_api = f"torch.nn.functional.{func_name}"
        
        # 构建paddle functional API
        paddle_func_api = f"paddle.nn.functional.{func_name}"
        
        return torch_func_api, paddle_func_api
    
    return None, None

# 测试用例
test_cases = [
    ("torch.nn.Dropout2d", "torch.nn.functional.dropout2d", "paddle.nn.functional.dropout2d"),
    ("torch.nn.AvgPool2d", "torch.nn.functional.avg_pool2d", "paddle.nn.functional.avg_pool2d"),
    ("torch.nn.MaxPool2d", "torch.nn.functional.max_pool2d", "paddle.nn.functional.max_pool2d"),
    ("torch.nn.Conv2d", "torch.nn.functional.conv2d", "paddle.nn.functional.conv2d"),
    ("torch.nn.BatchNorm2d", "torch.nn.functional.batch_norm", "paddle.nn.functional.batch_norm"),
]

print("=" * 80)
print("AvgPool2d 转换测试")
print("=" * 80)

all_passed = True

for original, expected_torch, expected_paddle in test_cases:
    print(f"\n测试: {original}")
    torch_func, paddle_func = convert_class_to_functional(original)
    
    torch_match = torch_func == expected_torch
    paddle_match = paddle_func == expected_paddle
    
    if torch_match and paddle_match:
        print(f"  ✅ 通过")
        print(f"     PyTorch:  {torch_func}")
        print(f"     Paddle:   {paddle_func}")
    else:
        all_passed = False
        print(f"  ❌ 失败")
        if not torch_match:
            print(f"     PyTorch 期望: {expected_torch}")
            print(f"     PyTorch 实际: {torch_func}")
        if not paddle_match:
            print(f"     Paddle 期望:  {expected_paddle}")
            print(f"     Paddle 实际:  {paddle_func}")

print("\n" + "=" * 80)
if all_passed:
    print("✅ 所有测试通过")
else:
    print("❌ 部分测试失败")
print("=" * 80)

# 详细分析 AvgPool2d
print("\n" + "=" * 80)
print("AvgPool2d 详细转换过程")
print("=" * 80)

class_name = "AvgPool2d"
print(f"\n1. 原始类名: {class_name}")

func_name = re.sub(r'(?<!^)(?<![0-9])([A-Z])', r'_\1', class_name)
print(f"2. 在大写字母前加下划线（不在开头和数字后）: {func_name}")

func_name = func_name.lower()
print(f"3. 全部转小写: {func_name}")

print(f"\n最终结果: torch.nn.functional.{func_name}")
print(f"期望结果: torch.nn.functional.avg_pool2d")
print(f"匹配: {func_name == 'avg_pool2d'}")
