#!/usr/bin/env python3
"""
测试参数映射和过滤功能
"""

def convert_key(key: str, paddle_api: str = "") -> str:
    """转换参数名"""
    # 通用参数映射
    key_mapping = {
        "input": "x",
        "other": "y",
        "n": "num_rows",  # torch.eye
        "m": "num_columns"  # torch.eye
    }
    return key_mapping.get(key, key)

def should_skip_param(key: str, paddle_api: str) -> bool:
    """判断是否应该跳过某个参数（Paddle不支持）"""
    # API特定的不支持参数
    skip_params = {
        "paddle.nn.functional.selu": ["inplace"],
    }
    
    # 检查是否在跳过列表中
    if paddle_api in skip_params:
        return key in skip_params[paddle_api]
    
    return False

print("=" * 80)
print("参数映射和过滤测试")
print("=" * 80)

# 测试1: torch.eye 参数映射
print("\n测试1: torch.eye 参数映射")
print("-" * 80)
params = {"n": 10, "m": 5, "dtype": "torch.float32"}
paddle_api = "paddle.eye"

print(f"原始参数: {params}")
print(f"Paddle API: {paddle_api}")
print("\n转换后:")
for key, value in params.items():
    new_key = convert_key(key, paddle_api)
    skip = should_skip_param(key, paddle_api)
    if skip:
        print(f"  {key} -> [跳过] (Paddle不支持)")
    else:
        print(f"  {key} -> {new_key}: {value}")

# 测试2: torch.nn.functional.selu 参数过滤
print("\n测试2: torch.nn.functional.selu 参数过滤")
print("-" * 80)
params = {"input": "tensor", "inplace": False}
paddle_api = "paddle.nn.functional.selu"

print(f"原始参数: {params}")
print(f"Paddle API: {paddle_api}")
print("\n转换后:")
for key, value in params.items():
    skip = should_skip_param(key, paddle_api)
    if skip:
        print(f"  {key} -> [跳过] (Paddle不支持)")
    else:
        new_key = convert_key(key, paddle_api)
        print(f"  {key} -> {new_key}: {value}")

# 测试3: 验证预期结果
print("\n测试3: 验证预期结果")
print("-" * 80)

test_cases = [
    ("torch.eye", "n", "num_rows", False),
    ("torch.eye", "m", "num_columns", False),
    ("paddle.nn.functional.selu", "inplace", None, True),
    ("paddle.nn.functional.selu", "input", "x", False),
]

all_passed = True
for api, key, expected_key, expected_skip in test_cases:
    new_key = convert_key(key, api)
    skip = should_skip_param(key, api)
    
    if expected_skip:
        if skip:
            print(f"✅ {api} - {key} 正确跳过")
        else:
            print(f"❌ {api} - {key} 应该跳过但没有")
            all_passed = False
    else:
        if new_key == expected_key:
            print(f"✅ {api} - {key} -> {new_key}")
        else:
            print(f"❌ {api} - {key} -> {new_key} (期望: {expected_key})")
            all_passed = False

print("\n" + "=" * 80)
if all_passed:
    print("✅ 所有测试通过")
else:
    print("❌ 部分测试失败")
print("=" * 80)
