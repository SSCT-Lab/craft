#!/usr/bin/env python3
"""
测试API转换逻辑
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
        # 获取类名并转换为小写
        class_name = parts[-1]
        func_name = class_name[0].lower() + class_name[1:]
        # 将大写字母转换为小写
        func_name = re.sub(r'([A-Z])', lambda m: m.group(1).lower(), func_name)
        
        # 构建torch functional API
        torch_func_api = f"torch.nn.functional.{func_name}"
        
        # 构建paddle functional API（保持全小写，不做大小写转换）
        paddle_func_api = f"paddle.nn.functional.{func_name}"
        
        return torch_func_api, paddle_func_api
    
    return None, None

# 测试用例
test_cases = [
    "torch.nn.Dropout2d",
    "torch.nn.AvgPool2d",
    "torch.nn.MaxPool2d",
    "torch.nn.Conv2d",
    "torch.nn.BatchNorm2d",
]

print("=" * 80)
print("API 转换测试")
print("=" * 80)

for api in test_cases:
    print(f"\n原始 API: {api}")
    print(f"  是类形式: {is_class_based_api(api)}")
    
    torch_func, paddle_func = convert_class_to_functional(api)
    if torch_func and paddle_func:
        print(f"  ✅ 转换成功:")
        print(f"     PyTorch:  {torch_func}")
        print(f"     Paddle:   {paddle_func}")
    else:
        print(f"  ❌ 转换失败")

print("\n" + "=" * 80)
print("关键验证:")
print("=" * 80)

# 验证 Dropout2d
torch_func, paddle_func = convert_class_to_functional("torch.nn.Dropout2d")
print(f"\nDropout2d 转换:")
print(f"  PyTorch:  {torch_func}")
print(f"  Paddle:   {paddle_func}")
print(f"  Paddle 是否包含大写 D: {'D' in paddle_func}")
print(f"  预期: paddle.nn.functional.dropout2d")
print(f"  匹配: {paddle_func == 'paddle.nn.functional.dropout2d'}")

# 验证 AvgPool2d
torch_func, paddle_func = convert_class_to_functional("torch.nn.AvgPool2d")
print(f"\nAvgPool2d 转换:")
print(f"  PyTorch:  {torch_func}")
print(f"  Paddle:   {paddle_func}")
print(f"  Paddle 是否包含大写: {any(c.isupper() for c in paddle_func.split('.')[-1])}")
print(f"  预期: paddle.nn.functional.avgpool2d")
print(f"  匹配: {paddle_func == 'paddle.nn.functional.avgpool2d'}")
