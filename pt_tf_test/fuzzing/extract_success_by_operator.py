"""  
PyTorch-TensorFlow 成功测试用例提取工具（按算子分类版本）

功能说明:
    1. 扫描指定目录下的所有测试结果 JSON 文件
    2. 提取所有没有任何错误的测试用例
    3. 按算子名称分类，每个算子生成一个独立的 JSON 文件
    4. 每个样例包含完整的 torch_test_case、tensorflow_test_case 和 execution_result
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


def extract_operator_name(filename: str) -> str:
    """
    从文件名中提取算子名称
    
    示例: llm_enhanced_torch_abs_20260123_191052.json -> torch_abs
    
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
                tensorflow_error = exec_result.get("tensorflow_error")
                comparison_error = exec_result.get("comparison_error")
                results_match = exec_result.get("results_match", False)
                
                # 完全成功的条件：无任何错误且结果匹配
                if (torch_error is None and 
                    tensorflow_error is None and 
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
                        "tensorflow_test_case": result.get("tensorflow_test_case", {}),
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
    
    total_cases = sum(len(cases) for cases in operator_cases.values())
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch-TensorFlow 成功测试用例汇总报告（按算子分类）\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【总体统计】\n")
        f.write("-" * 40 + "\n")
        f.write(f"涉及算子数: {len(operator_cases)}\n")
        f.write(f"成功用例总数: {total_cases}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        f.write("【各算子成功用例统计】\n")
        f.write("-" * 40 + "\n\n")
        
        for idx, (operator_name, cases) in enumerate(sorted_operators, 1):
            if not cases:
                continue
            
            # 统计 LLM 生成的用例数量
            llm_generated = sum(1 for c in cases if c.get("is_llm_generated", False))
            original = len(cases) - llm_generated
            
            f.write(f"{idx}. {operator_name}\n")
            f.write(f"   成功用例数: {len(cases)} (原始: {original}, LLM生成: {llm_generated})\n")
            
            # 统计不同的 shape 和 dtype
            shapes = set()
            dtypes = set()
            for case in cases:
                input_info = case.get("torch_test_case", {}).get("input", {})
                if isinstance(input_info, dict):
                    shape = input_info.get("shape", [])
                    dtype = input_info.get("dtype", "")
                    if shape:
                        shapes.add(str(shape))
                    if dtype:
                        dtypes.add(dtype)
            
            if shapes:
                f.write(f"   测试 shapes: {', '.join(list(shapes)[:5])}" + 
                       (" ..." if len(shapes) > 5 else "") + "\n")
            if dtypes:
                f.write(f"   测试 dtypes: {', '.join(dtypes)}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 80 + "\n")


def main():
    """
    主程序入口
    """
    # 配置路径
    log_dir = r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1"
    output_dir = r"d:\graduate\DFrameworkTest\pt_tf_test\fuzzing\success_cases"
    report_file = r"d:\graduate\DFrameworkTest\pt_tf_test\fuzzing\success_cases_summary.txt"
    
    print("=" * 60)
    print("开始提取成功测试用例并按算子分类...")
    print("=" * 60)
    
    # 分析并提取成功用例
    operator_cases = analyze_and_extract_success_cases(log_dir)
    
    # 保存各算子的 JSON 文件
    save_operator_json_files(operator_cases, output_dir)
    
    # 生成汇总报告
    generate_summary_report(operator_cases, report_file)
    print(f"汇总报告已生成: {report_file}")
    
    print("\n" + "=" * 60)
    print("提取完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
