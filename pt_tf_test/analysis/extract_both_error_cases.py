"""
both_error 样例提取工具

功能说明:
    1. 从错误分析报告中识别包含 both_error 的文件
    2. 从这些 JSON 文件中提取所有同时存在 torch_error 和 tensorflow_error 的测试样例
    3. 格式化输出样例详情（包括测试代码、错误信息等）
    4. 生成独立的样例报告文件，便于深入分析两个框架都失败的原因

使用场景:
    当测试显示 PyTorch 和 TensorFlow 都执行失败时，
    使用此工具提取具体的测试用例代码和错误信息，便于人工审查和调试。

作者: [Your Name]
日期: 2026-01-24
"""

import re
import json
from pathlib import Path

def parse_report(report_path):
    """
    解析错误分析报告，提取包含 both_error 的 JSON 文件名列表
    
    参数:
        report_path (str): 错误分析报告文件的路径（由 analyze_errors.py 生成）
    
    返回:
        list: 包含 both_error 的 JSON 文件名列表
    
    解析逻辑:
        - 查找包含 "文件名: xxx.json" 的行
        - 检查该文件部分是否包含 "两个框架都报错的样例数" 字段
        - 只有同时满足两个条件才认为该文件包含 both_error
    """
    files = []               # 存储包含 both_error 的文件名
    current_file = None      # 当前正在处理的文件名
    has_field = False        # 标记当前文件是否包含 both_error 字段
    
    # 逐行读取报告文件
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            # 使用正则表达式匹配文件名行（格式："文件名: xxx.json"）
            m = re.search(r"文件名:\s+([^\s]+\.json)", line)
            if m:
                # 如果找到新文件名，先检查上一个文件是否满足条件
                if current_file and has_field:
                    files.append(current_file)  # 上一个文件满足条件，添加到结果列表
                
                # 更新为新的文件名，重置标记
                current_file = m.group(1).strip()
                has_field = False
                continue
            
            # 检查当前行是否包含 both_error 字段
            if "两个框架都报错的样例数" in line:
                has_field = True  # 标记当前文件包含 both_error
        
        # 处理最后一个文件（循环结束后需要单独检查）
        if current_file and has_field:
            files.append(current_file)
    
    return files

def collect_samples(json_path):
    """
    从 JSON 测试结果文件中收集所有 both_error（torch_error 和 tensorflow_error 都非 null）的样例
    
    参数:
        json_path (str): JSON 测试结果文件的路径
    
    返回:
        list: 包含 both_error 的样例信息字典列表，每个字典包含：
            - iteration: 迭代次数
            - torch_error: PyTorch 错误信息
            - tensorflow_error: TensorFlow 错误信息
            - torch_test_case/tensorflow_test_case: 分别的测试代码（如果存在）
            - test_case: 统一的测试代码（如果存在）
            - raw_item: 原始数据（如果以上都不存在）
    
    判断逻辑:
        只有当 torch_error 和 tensorflow_error 都非 null 时，才认为是 both_error
    """
    results = []  # 存储所有包含 both_error 的样例
    
    # 读取 JSON 文件
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 遍历所有测试结果
    for item in data.get("results", []):
        # 获取执行结果字段（使用 or {} 避免 None 值）
        exec_res = item.get("execution_result") or {}
        
        # 提取两个框架的错误字段
        torch_err = exec_res.get("torch_error")
        tensorflow_err = exec_res.get("tensorflow_error")
        
        # 只处理两个框架都报错的样例（both_error）
        if torch_err is not None and tensorflow_err is not None:
            # 构建基础信息
            entry = {
                "iteration": item.get("iteration"),     # 迭代次数（用于追溯）
                "torch_error": torch_err,               # PyTorch 错误信息
                "tensorflow_error": tensorflow_err,     # TensorFlow 错误信息
            }
            
            # 根据数据格式提取测试用例代码（优先级递减）
            if "torch_test_case" in item or "tensorflow_test_case" in item:
                # 格式1：分离的 PyTorch 和 TensorFlow 测试代码
                entry["torch_test_case"] = item.get("torch_test_case")
                entry["tensorflow_test_case"] = item.get("tensorflow_test_case")
            elif "test_case" in item:
                # 格式2：统一的测试代码
                entry["test_case"] = item.get("test_case")
            else:
                # 格式3：保存完整原始数据（兜底方案）
                entry["raw_item"] = item
            
            results.append(entry)
    
    return results

def format_section(filename, samples):
    """
    格式化输出单个文件的所有 both_error 样例
    
    参数:
        filename (str): JSON 文件名
        samples (list): collect_samples() 返回的样例列表
    
    返回:
        str: 格式化后的文本，包含文件名、样例编号、错误信息和测试代码
    
    输出格式:
        ================================================================================
        文件: xxx.json
        --------------------------------------------------------------------------------
        样例 1:
        torch_error: <PyTorch 错误信息>
        tensorflow_error: <TensorFlow 错误信息>
        torch_test_case: <PyTorch 测试代码 JSON>
        tensorflow_test_case: <TensorFlow 测试代码 JSON>
        ...
    """
    lines = []  # 存储格式化后的文本行
    
    # 添加文件头部（使用 = 作为分隔符）
    lines.append("=" * 80)
    lines.append(f"文件: {filename}")
    lines.append("-" * 80)
    
    # 遍历该文件的所有样例
    for idx, s in enumerate(samples, 1):
        lines.append(f"样例 {idx}:")
        lines.append(f"torch_error: {s.get('torch_error')}")
        lines.append(f"tensorflow_error: {s.get('tensorflow_error')}")
        
        # 根据数据格式输出测试用例代码（与 collect_samples 的逻辑对应）
        if "torch_test_case" in s or "tensorflow_test_case" in s:
            # 格式1：分离的测试代码
            if s.get("torch_test_case") is not None:
                lines.append("torch_test_case:")
                # 使用 ensure_ascii=False 保留中文，indent=2 增强可读性
                lines.append(json.dumps(s["torch_test_case"], ensure_ascii=False, indent=2))
            if s.get("tensorflow_test_case") is not None:
                lines.append("tensorflow_test_case:")
                lines.append(json.dumps(s["tensorflow_test_case"], ensure_ascii=False, indent=2))
        elif "test_case" in s:
            # 格式2：统一的测试代码
            lines.append("test_case:")
            lines.append(json.dumps(s["test_case"], ensure_ascii=False, indent=2))
        else:
            # 格式3：输出完整原始数据
            lines.append("raw_item:")
            lines.append(json.dumps(s.get("raw_item"), ensure_ascii=False, indent=2))
        
        lines.append("")  # 样例之间添加空行，增强可读性
    
    return "\n".join(lines)

def main():
    """
    主程序入口：提取并格式化输出所有 both_error 样例
    
    执行流程:
        1. 配置输入输出路径
        2. 解析错误分析报告，获取包含 both_error 的文件列表
        3. 逐个读取这些文件，提取错误样例
        4. 格式化输出所有样例到报告文件
        5. 打印输出文件路径
    
    输出文件:
        analysis/new_both_error_samples_report.txt
        包含所有 both_error 样例的详细信息，供人工审查
    """
    # 配置路径
    base_dir = Path(r"d:\graduate\DFrameworkTest\pt_tf_test")         # 项目根目录
    report_path = base_dir / "pt_tf_log_1" / "error_analysis_report_new.txt"  # 输入：错误分析报告
    log_dir = base_dir / "pt_tf_log_1"                                # JSON 日志文件目录
    output_dir = base_dir / "analysis"                                # 输出目录
    output_dir.mkdir(parents=True, exist_ok=True)                     # 确保输出目录存在
    output_path = output_dir / "new_both_error_samples_report.txt"    # 输出文件路径
    
    # 步骤1：解析报告，获取包含 both_error 的文件名列表
    target_files = parse_report(report_path)
    
    # 步骤2：逐个处理这些文件，收集样例并格式化
    sections = []  # 存储每个文件的格式化输出
    for fname in target_files:
        jpath = log_dir / fname
        
        # 检查文件是否存在（容错处理）
        if not jpath.exists():
            continue
        
        # 从 JSON 文件中提取 both_error 样例
        samples = collect_samples(jpath)
        
        # 如果该文件确实包含样例，格式化后添加到结果中
        if samples:
            sections.append(format_section(fname, samples))
    
    # 步骤3：生成最终报告内容
    content = "\n".join(sections) if sections else "无两个框架都报错的样例"
    
    # 步骤4：写入输出文件
    with open(output_path, "w", encoding="utf-8") as wf:
        wf.write(content)
    
    # 打印输出文件路径（便于用户快速定位）
    print(f"✅ 报告已生成: {output_path}")
    print(f"📊 共处理 {len(target_files)} 个包含 both_error 的文件")
    print(f"📝 共生成 {len(sections)} 个文件的样例报告")

if __name__ == "__main__":
    main()
