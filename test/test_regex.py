#!/usr/bin/env python3
"""
测试正则表达式转换
"""
import re

def test_conversion(class_name, expected):
    """测试单个转换"""
    func_name = re.sub(r'(?<!^)(?<![0-9])([A-Z])', r'_\1', class_name).lower()
    match = func_name == expected
    status = "✅" if match else "❌"
    print(f"{status} {class_name:20s} -> {func_name:25s} (期望: {expected})")
    return match

print("=" * 80)
print("驼峰命名到下划线命名转换测试")
print("=" * 80)
print()

test_cases = [
    ("AvgPool2d", "avg_pool2d"),
    ("Dropout2d", "dropout2d"),
    ("MaxPool2d", "max_pool2d"),
    ("Conv2d", "conv2d"),
    ("BatchNorm2d", "batch_norm2d"),
    ("InstanceNorm2d", "instance_norm2d"),
    ("AdaptiveAvgPool2d", "adaptive_avg_pool2d"),
    ("ReLU", "re_lu"),  # 这个可能有问题
    ("LSTM", "l_s_t_m"),  # 这个可能有问题
]

all_passed = True
for class_name, expected in test_cases:
    if not test_conversion(class_name, expected):
        all_passed = False

print()
print("=" * 80)
if all_passed:
    print("✅ 所有测试通过")
else:
    print("❌ 部分测试失败")
print("=" * 80)
