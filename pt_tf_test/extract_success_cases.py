"""  
PyTorch-TensorFlow 成功测试用例提取工具

功能说明:
    1. 扫描指定目录下的所有测试结果 JSON 文件
    2. 提取所有没有任何错误的测试用例（torch_error、tensorflow_error、comparison_error 均为 null）
    3. 按文件汇总成功用例数量和对应的迭代次数
    4. 生成详细的成功用例分析报告
"""

import json
from pathlib import Path
from datetime import datetime


def analyze_success_cases(log_dir: str) -> tuple:
    """
    分析指定目录下的所有 JSON 测试结果文件，提取成功用例统计信息
    
    参数:
        log_dir (str): 日志文件所在目录的路径
    
    返回:
        tuple: 包含以下元素的元组
            - total_success_cases (int): 完全成功的测试用例总数
            - total_cases (int): 所有测试用例的总数
            - files_all_success (list): 所有用例都成功的文件列表
            - files_with_success (list): 包含成功用例的文件详细信息列表
            - all_success_details (list): 所有成功用例的详细信息
    
    异常处理:
        - 遇到文件解析错误时会打印错误信息但不中断整体分析
    """
    # 初始化全局计数器
    total_success_cases = 0      # 完全成功的用例总数
    total_cases = 0              # 所有用例总数
    files_all_success = []       # 所有用例都成功的文件
    files_with_success = []      # 包含成功用例的文件详细信息
    all_success_details = []     # 所有成功用例的详细信息
    
    # 获取日志目录路径并查找所有匹配的 JSON 文件
    log_path = Path(log_dir)
    json_files = sorted(log_path.glob("llm_enhanced_torch*.json"))  # 按文件名排序
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    # 遍历所有 JSON 文件进行分析
    for json_file in json_files:
        try:
            # 读取 JSON 文件内容
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 初始化当前文件的计数器
            file_success_cases = 0       # 当前文件成功用例数
            file_total_cases = 0         # 当前文件总用例数
            success_iterations = []      # 成功用例的迭代次数列表
            success_case_details = []    # 成功用例的详细信息
            
            # 获取算子名称
            operator_name = data.get("operator", "unknown")
            
            # 检查 JSON 数据中是否包含 results 字段
            if "results" in data:
                # 遍历每个测试结果
                for result in data["results"]:
                    file_total_cases += 1
                    total_cases += 1
                    
                    # 检查是否包含执行结果字段
                    if "execution_result" in result:
                        exec_result = result["execution_result"]
                        iteration = result.get("iteration", "N/A")
                        case_number = result.get("case_number", "N/A")
                        is_llm_generated = result.get("is_llm_generated", False)
                        
                        # 检查是否完全成功（三种错误都为 null）
                        torch_error = exec_result.get("torch_error")
                        tensorflow_error = exec_result.get("tensorflow_error")
                        comparison_error = exec_result.get("comparison_error")
                        results_match = exec_result.get("results_match", False)
                        
                        # 完全成功的条件：无任何错误且结果匹配
                        if (torch_error is None and 
                            tensorflow_error is None and 
                            comparison_error is None and
                            results_match):
                            
                            file_success_cases += 1
                            total_success_cases += 1
                            success_iterations.append(iteration)
                            
                            # 记录成功用例的详细信息
                            case_detail = {
                                "filename": json_file.name,
                                "operator": operator_name,
                                "iteration": iteration,
                                "case_number": case_number,
                                "is_llm_generated": is_llm_generated,
                                "torch_shape": exec_result.get("torch_shape", []),
                                "torch_dtype": exec_result.get("torch_dtype", ""),
                                "tensorflow_shape": exec_result.get("tensorflow_shape", []),
                                "tensorflow_dtype": exec_result.get("tensorflow_dtype", ""),
                                "input_info": result.get("torch_test_case", {}).get("input", {})
                            }
                            success_case_details.append(case_detail)
                            all_success_details.append(case_detail)
            
            # 记录当前文件的统计信息
            if file_success_cases > 0:
                file_info = {
                    "filename": json_file.name,
                    "operator": operator_name,
                    "success_cases": file_success_cases,
                    "total_cases": file_total_cases,
                    "success_rate": f"{file_success_cases / file_total_cases * 100:.1f}%" if file_total_cases > 0 else "0%",
                    "success_iterations": success_iterations,
                    "success_details": success_case_details
                }
                files_with_success.append(file_info)
                
                # 如果所有用例都成功，加入完全成功文件列表
                if file_success_cases == file_total_cases:
                    files_all_success.append(file_info)
                    
        except Exception as e:
            # 异常处理：打印错误但不中断整体分析流程
            print(f"处理文件 {json_file.name} 时出错: {e}")
    
    return total_success_cases, total_cases, files_all_success, files_with_success, all_success_details


def generate_report(output_file: str, 
                    total_success: int, 
                    total_cases: int,
                    files_all_success: list,
                    files_with_success: list,
                    all_success_details: list) -> None:
    """
    生成格式化的成功用例分析报告并写入文本文件
    
    参数:
        output_file (str): 输出报告文件的路径
        total_success (int): 成功用例总数
        total_cases (int): 总用例数
        files_all_success (list): 所有用例都成功的文件列表
        files_with_success (list): 包含成功用例的文件详细信息列表
        all_success_details (list): 所有成功用例的详细信息
    """
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        # 写入报告头部
        f.write("=" * 80 + "\n")
        f.write("PyTorch-TensorFlow 成功测试用例分析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # 写入总体统计信息
        f.write("【总体统计】\n")
        f.write("-" * 40 + "\n")
        f.write(f"成功用例总数: {total_success}\n")
        f.write(f"总用例数: {total_cases}\n")
        success_rate = total_success / total_cases * 100 if total_cases > 0 else 0
        f.write(f"总成功率: {success_rate:.2f}%\n")
        f.write(f"包含成功用例的文件数: {len(files_with_success)}\n")
        f.write(f"所有用例都成功的文件数: {len(files_all_success)}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # 写入所有用例都成功的文件列表
        f.write("【所有用例都成功的文件】\n")
        f.write("-" * 40 + "\n")
        if files_all_success:
            for idx, file_info in enumerate(files_all_success, 1):
                f.write(f"{idx}. {file_info['filename']}\n")
                f.write(f"   算子: {file_info['operator']}\n")
                f.write(f"   成功用例数: {file_info['success_cases']}/{file_info['total_cases']}\n")
                f.write(f"   迭代次数: {', '.join(map(str, file_info['success_iterations']))}\n")
                f.write("\n")
        else:
            f.write("没有找到所有用例都成功的文件。\n\n")
        
        f.write("=" * 80 + "\n\n")
        
        # 写入包含成功用例的文件详细信息
        f.write("【包含成功用例的文件详细信息】\n")
        f.write("-" * 40 + "\n\n")
        if files_with_success:
            for idx, file_info in enumerate(files_with_success, 1):
                f.write(f"{idx}. 文件名: {file_info['filename']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"   算子: {file_info['operator']}\n")
                f.write(f"   成功用例数: {file_info['success_cases']}/{file_info['total_cases']} ({file_info['success_rate']})\n")
                f.write(f"   成功用例的 iteration 值: {', '.join(map(str, file_info['success_iterations']))}\n")
                
                # 输出每个成功用例的简要信息
                f.write("   成功用例详情:\n")
                for detail in file_info['success_details']:
                    input_info = detail.get('input_info', {})
                    # 处理 input_info 可能是列表或字典的情况
                    if isinstance(input_info, dict):
                        shape = input_info.get('shape', [])
                        dtype = input_info.get('dtype', 'unknown')
                    elif isinstance(input_info, list) and len(input_info) > 0:
                        # 如果是列表，取第一个元素
                        first_input = input_info[0] if isinstance(input_info[0], dict) else {}
                        shape = first_input.get('shape', [])
                        dtype = first_input.get('dtype', 'unknown')
                    else:
                        shape = []
                        dtype = 'unknown'
                    llm_tag = "[LLM生成]" if detail.get('is_llm_generated') else "[原始用例]"
                    f.write(f"      - iteration {detail['iteration']}, case {detail['case_number']}: "
                            f"shape={shape}, dtype={dtype} {llm_tag}\n")
                f.write("\n")
        else:
            f.write("没有找到包含成功用例的文件。\n\n")
        
        # 写入报告尾部
        f.write("=" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 80 + "\n")


def export_success_cases_json(output_file: str, all_success_details: list) -> None:
    """
    将所有成功用例的详细信息导出为 JSON 文件，便于后续处理
    
    参数:
        output_file (str): 输出 JSON 文件的路径
        all_success_details (list): 所有成功用例的详细信息
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_data = {
        "export_time": datetime.now().isoformat(),
        "total_success_cases": len(all_success_details),
        "success_cases": all_success_details
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)


def main():
    """
    主程序入口：执行成功用例分析并生成报告
    
    执行流程:
        1. 指定日志目录和输出文件路径
        2. 调用 analyze_success_cases() 分析所有 JSON 文件
        3. 在控制台打印统计摘要
        4. 调用 generate_report() 生成详细报告文件
        5. 调用 export_success_cases_json() 导出成功用例 JSON
    """
    # 配置输入输出路径
    log_dir = r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1"
    output_report = r"d:\graduate\DFrameworkTest\pt_tf_test\fuzzing\success_cases_report.txt"
    output_json = r"d:\graduate\DFrameworkTest\pt_tf_test\fuzzing\success_cases_data.json"
    
    # 开始分析
    print("开始分析 JSON 文件，提取成功用例...")
    (total_success, total_cases, files_all_success, 
     files_with_success, all_success_details) = analyze_success_cases(log_dir)
    
    # 在控制台打印统计摘要
    print("\n" + "=" * 50)
    print("分析完成！")
    print("=" * 50)
    print(f"成功用例统计:")
    print(f"  - 成功用例总数: {total_success}")
    print(f"  - 总用例数: {total_cases}")
    success_rate = total_success / total_cases * 100 if total_cases > 0 else 0
    print(f"  - 总成功率: {success_rate:.2f}%")
    print(f"  - 包含成功用例的文件数: {len(files_with_success)}")
    print(f"  - 所有用例都成功的文件数: {len(files_all_success)}")
    
    # 生成详细报告文件
    generate_report(output_report, total_success, total_cases, 
                    files_all_success, files_with_success, all_success_details)
    print(f"\n文本报告已生成: {output_report}")
    
    # 导出成功用例 JSON
    export_success_cases_json(output_json, all_success_details)
    print(f"JSON 数据已导出: {output_json}")


if __name__ == "__main__":
    main()
