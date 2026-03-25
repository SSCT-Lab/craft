"""
分析 paddle_a 目录中的文件，提取 API 名称，
并检查这些 API 是否在 torch_error_samples_report.txt 中出现过

功能：
1. 遍历 pt_pd_test/analysis/paddle_a 目录中的所有文件
2. 从文件名中提取 torch_xxx 部分（在 llm_enhanced_ 之后，_日期 之前）
3. 检查这些 API 名称是否在 torch_error_samples_report.txt 文件中出现过
4. 打印出所有未出现过的 API 名称
"""

import os
import re
from typing import Set


def extract_api_name_from_filename(filename: str) -> str:
    """
    从文件名中提取 API 名称
    
    文件名格式: llm_enhanced_torch_xxx_日期.json_sampleN.txt
    需要提取: torch_xxx 部分
    
    Args:
        filename: 文件名
    
    Returns:
        提取的 API 名称（使用下划线分隔）
    """
    # 匹配模式: llm_enhanced_(torch_xxx)_日期
    # 日期格式: 8位数字（如 20251202）
    pattern = r'llm_enhanced_(torch_[\w_]+?)_(\d{8}_\d{6})'
    match = re.search(pattern, filename)
    
    if match:
        return match.group(1)
    
    return ""


def extract_apis_from_report(report_path: str) -> Set[str]:
    """
    从 torch_error_samples_report.txt 中提取所有出现的 API 名称
    
    报告中的格式: 文件: llm_enhanced_torch_xxx_日期.json
    
    Args:
        report_path: 报告文件路径
    
    Returns:
        报告中出现的 API 名称集合
    """
    apis = set()
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配报告中的文件名行
    pattern = r'文件: llm_enhanced_(torch_[\w_]+?)_(\d{8}_\d{6})\.json'
    matches = re.findall(pattern, content)
    
    for match in matches:
        api_name = match[0]
        apis.add(api_name)
    
    return apis


def main():
    """主函数"""
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # paddle_a 目录路径
    paddle_a_dir = os.path.join(script_dir, 'analysis', 'paddle_a')
    
    # torch_error_samples_report.txt 路径
    report_path = os.path.join(script_dir, 'analysis', 'torch_error_samples_report.txt')
    
    print("=" * 60)
    print("📊 Paddle_a 目录 API 分析工具")
    print("=" * 60)
    
    # 1. 从 paddle_a 目录提取所有 API 名称
    paddle_a_apis = set()
    
    if not os.path.exists(paddle_a_dir):
        print(f"❌ 目录不存在: {paddle_a_dir}")
        return
    
    files = os.listdir(paddle_a_dir)
    print(f"\n📁 paddle_a 目录文件数: {len(files)}")
    
    for filename in files:
        api_name = extract_api_name_from_filename(filename)
        if api_name:
            paddle_a_apis.add(api_name)
    
    print(f"✅ 提取到 {len(paddle_a_apis)} 个唯一 API 名称")
    
    # 2. 从 torch_error_samples_report.txt 提取 API 名称
    if not os.path.exists(report_path):
        print(f"❌ 报告文件不存在: {report_path}")
        return
    
    report_apis = extract_apis_from_report(report_path)
    print(f"✅ torch_error_report 中包含 {len(report_apis)} 个唯一 API 名称")
    
    # 3. 找出未在报告中出现的 API
    apis_not_in_report = paddle_a_apis - report_apis
    
    print("\n" + "=" * 60)
    print(f"📋 未在 torch_error_report 中出现的 API ({len(apis_not_in_report)} 个):")
    print("=" * 60)
    
    # 按字母顺序排序后打印
    for api_name in sorted(apis_not_in_report):
        print(f"  - {api_name}")
    
    # 额外信息：打印在报告中出现的 API（用于验证）
    apis_in_report = paddle_a_apis & report_apis
    print("\n" + "-" * 60)
    print(f"📋 在 torch_error_report 中出现的 API ({len(apis_in_report)} 个):")
    print("-" * 60)
    for api_name in sorted(apis_in_report):
        print(f"  - {api_name}")
    
    print("\n" + "=" * 60)
    print("🎉 分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
