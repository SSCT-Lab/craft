"""  
PyTorch-MindSpore 成功测试用例提取工具（按算子分类版本）

功能说明:
    1. 扫描指定目录下的所有测试结果 JSON 文件
    2. 提取所有没有任何错误的测试用例
    3. 按算子名称分类，每个算子生成一个独立的 JSON 文件
    4. 每个样例包含完整的 torch_test_case、mindspore_test_case 和 execution_result
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


def extract_operator_name(filename: str) -> str:
    """
    从文件名中提取算子名称
    
    示例: llm_enhanced_torch_abs_20260126_115321.json -> torch_abs
    
    参数:
        filename (str): JSON 文件名
    
    返回:
        str: 算子名称
    """
    # 匹配 llm_enhanced_ 后面到 _日期 前面的部分
    # 日期格式为 8位数字（YYYYMMDD）
    pattern = r'llm_enhanced_(.+?)_(\d{8})_'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    
    # 备用方案：去掉前缀和后缀
    name = filename.replace('llm_enhanced_', '').replace('.json', '')
    # 尝试去掉日期时间后缀
    parts = name.rsplit('_', 2)
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        return '_'.join(parts[:-2])
    return name


def analyze_and_extract_success_cases(log_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    分析指定目录下的所有 JSON 测试结果文件，按算子提取成功用例
    
    参数:
        log_dir (str): 日志文件所在目录的路径
    
    返回:
        Dict[str, List[Dict]]: 算子名称 -> 成功用例列表的映射
    """
    # 按算子分类的成功用例字典
    operator_success_cases: Dict[str, List[Dict[str, Any]]] = {}
    
    # 统计信息
    total_files = 0
    total_cases = 0
    total_success = 0
    
    # 获取日志目录路径并查找所有匹配的 JSON 文件
    log_path = Path(log_dir)
    json_files = sorted(log_path.glob("llm_enhanced_torch*.json"))
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    # 遍历所有 JSON 文件进行分析
    for json_file in json_files:
        total_files += 1
        try:
            # 读取 JSON 文件内容
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 提取算子名称
            operator_name = extract_operator_name(json_file.name)
            
            # 如果该算子还没有记录，初始化列表
            if operator_name not in operator_success_cases:
                operator_success_cases[operator_name] = []
            
            # 检查 JSON 数据中是否包含 results 字段
            if "results" not in data:
                continue
                
            # 遍历每个测试结果
            for result in data["results"]:
                total_cases += 1
                
                # 检查是否包含执行结果字段
                if "execution_result" not in result:
                    continue
                    
                exec_result = result["execution_result"]
                
                # 检查是否完全成功（三种错误都为 null 且结果匹配）
                torch_error = exec_result.get("torch_error")
                mindspore_error = exec_result.get("mindspore_error")
                comparison_error = exec_result.get("comparison_error")
                results_match = exec_result.get("results_match", False)
                
                # 完全成功的条件：无任何错误且结果匹配
                if (torch_error is None and 
                    mindspore_error is None and 
                    comparison_error is None and
                    results_match):
                    
                    total_success += 1
                    
                    # 构建完整的成功用例信息
                    success_case = {
                        "source_file": json_file.name,
                        "operator": data.get("operator", operator_name),
                        "iteration": result.get("iteration", "N/A"),
                        "case_number": result.get("case_number", "N/A"),
                        "is_llm_generated": result.get("is_llm_generated", False),
                        "torch_test_case": result.get("torch_test_case", {}),
                        "mindspore_test_case": result.get("mindspore_test_case", {}),
                        "execution_result": exec_result
                    }
                    
                    # 如果有 llm_operation 信息，也保留
                    if "llm_operation" in result:
                        success_case["llm_operation"] = result["llm_operation"]
                    
                    operator_success_cases[operator_name].append(success_case)
                    
        except Exception as e:
            print(f"处理文件 {json_file.name} 时出错: {e}")
    
    # 打印统计信息
    print(f"\n统计信息:")
    print(f"  - 处理文件数: {total_files}")
    print(f"  - 总用例数: {total_cases}")
    print(f"  - 成功用例数: {total_success}")
    print(f"  - 成功率: {total_success / total_cases * 100:.2f}%" if total_cases > 0 else "  - 成功率: N/A")
    print(f"  - 涉及算子数: {len(operator_success_cases)}")
    
    return operator_success_cases


def save_operator_json_files(
    operator_cases: Dict[str, List[Dict[str, Any]]], 
    output_dir: str
) -> None:
    """
    将按算子分类的成功用例保存为独立的 JSON 文件
    
    参数:
        operator_cases (Dict): 算子名称 -> 成功用例列表的映射
        output_dir (str): 输出目录路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for operator_name, cases in operator_cases.items():
        if not cases:
            continue
            
        # 构建输出文件路径
        output_file = output_path / f"{operator_name}_success_cases.json"
        
        # 构建输出数据结构
        output_data = {
            "operator": operator_name,
            "export_time": datetime.now().isoformat(),
            "total_success_cases": len(cases),
            "success_cases": cases
        }
        
        # 写入 JSON 文件
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        saved_count += 1
    
    print(f"\n已保存 {saved_count} 个算子的成功用例文件到: {output_dir}")


def generate_summary_report(
    operator_cases: Dict[str, List[Dict[str, Any]]], 
    output_file: str
) -> None:
    """
    生成汇总报告
    
    参数:
        operator_cases (Dict): 算子名称 -> 成功用例列表的映射
        output_file (str): 输出报告文件路径
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 按成功用例数量排序
    sorted_operators = sorted(
        operator_cases.items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    )
    
    total_success = sum(len(cases) for cases in operator_cases.values())
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch-MindSpore 成功测试用例提取报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"【汇总统计】\n")
        f.write(f"  - 涉及算子数: {len(operator_cases)}\n")
        f.write(f"  - 总成功用例数: {total_success}\n")
        f.write(f"  - 平均每算子用例数: {total_success / len(operator_cases):.2f}\n" if operator_cases else "")
        f.write("\n")
        
        f.write("【各算子成功用例数】\n")
        f.write("-" * 60 + "\n")
        for idx, (op_name, cases) in enumerate(sorted_operators, 1):
            f.write(f"{idx:3d}. {op_name}: {len(cases)} 个用例\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 80 + "\n")
    
    print(f"报告已保存到: {output_file}")


def save_all_success_cases(
    operator_cases: Dict[str, List[Dict[str, Any]]], 
    output_file: str
) -> None:
    """
    将所有成功用例保存为单一 JSON 文件
    
    参数:
        operator_cases (Dict): 算子名称 -> 成功用例列表的映射
        output_file (str): 输出文件路径
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_success = sum(len(cases) for cases in operator_cases.values())
    
    output_data = {
        "export_time": datetime.now().isoformat(),
        "total_operators": len(operator_cases),
        "total_success_cases": total_success,
        "operators": operator_cases
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"所有成功用例已保存到: {output_file}")


def generate_summary_txt(
    operator_cases: Dict[str, List[Dict[str, Any]]], 
    output_file: str,
    total_cases: int = 0,
    total_success: int = 0
) -> None:
    """
    生成简洁的统计摘要
    
    参数:
        operator_cases (Dict): 算子名称 -> 成功用例列表的映射
        output_file (str): 输出文件路径
        total_cases (int): 总用例数
        total_success (int): 成功用例数
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_success_count = sum(len(cases) for cases in operator_cases.values())
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("PyTorch-MindSpore 成功用例提取统计\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 40 + "\n")
        f.write(f"涉及算子数: {len(operator_cases)}\n")
        f.write(f"成功用例数: {total_success_count}\n")
        if total_cases > 0:
            f.write(f"成功率: {total_success_count / total_cases * 100:.2f}%\n")
    
    print(f"统计摘要已保存到: {output_file}")


def main():
    """
    主程序入口
    """
    # 设置路径
    script_dir = Path(__file__).parent
    log_dir = script_dir.parent / "pt_ms_log_1"
    success_cases_dir = script_dir / "success_cases"
    
    print("=" * 60)
    print("PyTorch-MindSpore 成功测试用例提取工具")
    print("=" * 60)
    print(f"日志目录: {log_dir}")
    print(f"输出目录: {success_cases_dir}")
    print()
    
    # 检查日志目录是否存在
    if not log_dir.exists():
        print(f"[ERROR] 日志目录不存在: {log_dir}")
        return
    
    # 分析并提取成功用例
    operator_cases = analyze_and_extract_success_cases(str(log_dir))
    
    if not operator_cases:
        print("\n[WARN] 未提取到任何成功用例")
        return
    
    # 保存按算子分类的 JSON 文件
    save_operator_json_files(operator_cases, str(success_cases_dir))
    
    # 保存所有成功用例到单一文件
    save_all_success_cases(
        operator_cases, 
        str(script_dir / "success_cases_data.json")
    )
    
    # 生成报告
    generate_summary_report(
        operator_cases, 
        str(script_dir / "success_cases_report.txt")
    )
    
    # 生成简洁摘要
    total_success = sum(len(cases) for cases in operator_cases.values())
    generate_summary_txt(
        operator_cases,
        str(script_dir / "success_cases_summary.txt"),
        total_cases=0,  # 会在 analyze_and_extract_success_cases 中计算
        total_success=total_success
    )
    
    print("\n" + "=" * 60)
    print("提取完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
