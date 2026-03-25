"""  
PyTorch-TensorFlow 测试错误分析工具

功能说明:
    1. 扫描指定目录下的所有测试结果 JSON 文件
    2. 统计三类错误：torch_error、tensorflow_error、comparison_error
    3. 按文件汇总错误数量和对应的迭代次数
    4. 生成详细的错误分析报告
"""

import json
from pathlib import Path

def analyze_json_files(log_dir):
    """
    分析指定目录下的所有 JSON 测试结果文件，提取错误统计信息
    
    参数:
        log_dir (str): 日志文件所在目录的路径
    
    返回:
        tuple: 包含以下四个元素的元组
            - total_torch_errors (int): torch_error 非 null 值的总数
            - total_tensorflow_errors (int): tensorflow_error 非 null 值的总数
            - total_comparison_errors (int): comparison_error 非 null 值的总数
            - files_with_errors (list): 包含错误的文件详细信息列表
    
    异常处理:
        - 遇到文件解析错误时会打印错误信息但不中断整体分析
    """
    """
    分析指定目录下的所有 JSON 测试结果文件，提取错误统计信息
    
    参数:
        log_dir (str): 日志文件所在目录的路径
    
    返回:
        tuple: 包含以下五个元素的元组
            - total_torch_errors (int): 仅 PyTorch 报错的样例总数
            - total_tensorflow_errors (int): 仅 TensorFlow 报错的样例总数
            - total_both_errors (int): 两个框架都报错的样例总数
            - total_comparison_errors (int): comparison_error 非 null 值的总数
            - files_with_errors (list): 包含错误的文件详细信息列表
    
    异常处理:
        - 遇到文件解析错误时会打印错误信息但不中断整体分析
    """
    # 初始化全局错误计数器
    total_torch_errors = 0           # 仅 PyTorch 执行错误总数
    total_tensorflow_errors = 0      # 仅 TensorFlow 执行错误总数
    total_both_errors = 0            # 两个框架都报错的总数
    total_comparison_errors = 0      # 结果比较错误总数
    files_with_errors = []           # 存储包含错误的文件详细信息
    
    # 获取日志目录路径并查找所有匹配的 JSON 文件
    log_path = Path(log_dir)
    json_files = sorted(log_path.glob("llm_enhanced_torch*.json"))  # 按文件名排序
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 遍历所有 JSON 文件进行分析
    for json_file in json_files:
        try:
            # 读取 JSON 文件内容
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 初始化当前文件的错误计数器
            file_torch_errors = 0            # 仅 PyTorch 错误数
            file_tensorflow_errors = 0       # 仅 TensorFlow 错误数
            file_both_errors = 0             # 两个框架都错误数
            file_comparison_errors = 0       # 比较错误数
            
            # 记录每种错误类型对应的迭代次数（用于追踪具体是哪个测试用例出错）
            error_iterations = {
                "torch_error": [],           # 仅 PyTorch 错误的迭代
                "tensorflow_error": [],      # 仅 TensorFlow 错误的迭代
                "both_error": [],            # 两个框架都错误的迭代
                "comparison_error": []       # 比较错误的迭代
            }
            
            # 检查 JSON 数据中是否包含 results 字段（测试结果数组）
            if "results" in data:
                # 遍历每个测试结果
                for result in data["results"]:
                    # 检查是否包含执行结果字段
                    if "execution_result" in result:
                        exec_result = result["execution_result"]
                        iteration = result.get("iteration", "N/A")  # 获取迭代次数，默认值为 "N/A"
                        
                        # 检查执行错误类型
                        has_torch_error = exec_result.get("torch_error") is not None
                        has_tensorflow_error = exec_result.get("tensorflow_error") is not None
                        
                        # 区分三种情况：仅 PyTorch 错误、仅 TensorFlow 错误、两者都错误
                        if has_torch_error and has_tensorflow_error:
                            # 两个框架都报错
                            file_both_errors += 1
                            total_both_errors += 1
                            error_iterations["both_error"].append(iteration)
                        elif has_torch_error:
                            # 仅 PyTorch 报错
                            file_torch_errors += 1
                            total_torch_errors += 1
                            error_iterations["torch_error"].append(iteration)
                        elif has_tensorflow_error:
                            # 仅 TensorFlow 报错
                            file_tensorflow_errors += 1
                            total_tensorflow_errors += 1
                            error_iterations["tensorflow_error"].append(iteration)
                        
                        # 检查结果比较错误（输出不一致）
                        if exec_result.get("comparison_error") is not None:
                            file_comparison_errors += 1
                            total_comparison_errors += 1
                            error_iterations["comparison_error"].append(iteration)
            
            # 如果当前文件包含任何类型的错误，记录详细信息
            if file_torch_errors > 0 or file_tensorflow_errors > 0 or file_both_errors > 0 or file_comparison_errors > 0:
                files_with_errors.append(
                    {
                        "filename": json_file.name,                          # 文件名
                        "torch_errors": file_torch_errors,                   # 仅 PyTorch 错误数
                        "tensorflow_errors": file_tensorflow_errors,         # 仅 TensorFlow 错误数
                        "both_errors": file_both_errors,                     # 两个框架都错误数
                        "comparison_errors": file_comparison_errors,         # 比较错误数
                        "torch_error_iterations": error_iterations["torch_error"],           # 仅 PyTorch 错误的迭代次数列表
                        "tensorflow_error_iterations": error_iterations["tensorflow_error"],  # 仅 TensorFlow 错误的迭代次数列表
                        "both_error_iterations": error_iterations["both_error"],             # 两个框架都错误的迭代次数列表
                        "comparison_error_iterations": error_iterations["comparison_error"],  # 比较错误的迭代次数列表
                    }
                )
        except Exception as e:
            # 异常处理：打印错误但不中断整体分析流程
            print(f"处理文件 {json_file.name} 时出错: {e}")
    
    # 返回统计结果
    return total_torch_errors, total_tensorflow_errors, total_both_errors, total_comparison_errors, files_with_errors

def generate_report(output_file, total_torch, total_tensorflow, total_both, total_comparison, files_with_errors):
    """
    生成格式化的错误分析报告并写入文本文件
    
    参数:
        output_file (str): 输出报告文件的路径
        total_torch (int): 仅 PyTorch 错误总数
        total_tensorflow (int): 仅 TensorFlow 错误总数
        total_both (int): 两个框架都报错的总数
        total_comparison (int): comparison_error 总数
        files_with_errors (list): 包含错误的文件详细信息列表
    
    输出格式:
        - 总体错误统计（四类错误的总数）
        - 详细错误文件信息（每个文件的错误数量和对应迭代次数）
    """
    # 打开输出文件，以 UTF-8 编码写入
    with open(output_file, "w", encoding="utf-8") as f:
        # 写入报告头部
        f.write("=" * 80 + "\n")
        f.write("PyTorch-TensorFlow 测试错误分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 写入总体错误统计信息
        f.write("【总体错误统计】\n")
        f.write(f"仅 PyTorch 报错的样例数: {total_torch}\n")
        f.write(f"仅 TensorFlow 报错的样例数: {total_tensorflow}\n")
        f.write(f"两个框架都报错的样例数: {total_both}\n")
        f.write(f"comparison_error 非null值总数: {total_comparison}\n")
        f.write(f"包含错误的文件总数: {len(files_with_errors)}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # 写入详细错误文件信息
        if files_with_errors:
            f.write("【详细错误文件信息】\n\n")
            # 遍历所有包含错误的文件
            for idx, file_info in enumerate(files_with_errors, 1):
                f.write(f"{idx}. 文件名: {file_info['filename']}\n")
                f.write("-" * 80 + "\n")
                
                # 如果存在仅 PyTorch 错误，写入详细信息
                if file_info["torch_errors"] > 0:
                    f.write(f"   仅 PyTorch 报错的样例数: {file_info['torch_errors']}\n")
                    f.write(f"   对应用例的iteration值: {', '.join(map(str, file_info['torch_error_iterations']))}\n")
                
                # 如果存在仅 TensorFlow 错误，写入详细信息
                if file_info["tensorflow_errors"] > 0:
                    f.write(f"   仅 TensorFlow 报错的样例数: {file_info['tensorflow_errors']}\n")
                    f.write(f"   对应用例的iteration值: {', '.join(map(str, file_info['tensorflow_error_iterations']))}\n")
                
                # 如果存在两个框架都错误，写入详细信息
                if file_info["both_errors"] > 0:
                    f.write(f"   两个框架都报错的样例数: {file_info['both_errors']}\n")
                    f.write(f"   对应用例的iteration值: {', '.join(map(str, file_info['both_error_iterations']))}\n")
                
                # 如果存在比较错误，写入详细信息
                if file_info["comparison_errors"] > 0:
                    f.write(f"   comparison_error 非null值个数: {file_info['comparison_errors']}\n")
                    f.write(f"   对应用例的iteration值: {', '.join(map(str, file_info['comparison_error_iterations']))}\n")
                
                f.write("\n")
        else:
            # 没有发现任何错误
            f.write("【详细错误文件信息】\n\n")
            f.write("未发现任何包含错误的文件。\n\n")
        
        # 写入报告尾部
        f.write("=" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 80 + "\n")

def main():
    """
    主程序入口：执行错误分析并生成报告
    
    执行流程:
        1. 指定日志目录和输出文件路径
        2. 调用 analyze_json_files() 分析所有 JSON 文件
        3. 在控制台打印统计摘要
        4. 调用 generate_report() 生成详细报告文件
    """
    """
    主程序入口：执行错误分析并生成报告
    
    执行流程:
        1. 指定日志目录和输出文件路径
        2. 调用 analyze_json_files() 分析所有 JSON 文件
        3. 在控制台打印统计摘要
        4. 调用 generate_report() 生成详细报告文件
    """
    # 配置输入输出路径
    # log_dir = r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log"          # JSON 日志文件目录
    log_dir = r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1" 
    output_file = r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1\error_analysis_report_new.txt"  # 输出报告文件
    
    # 开始分析
    print("开始分析JSON文件...")
    total_torch, total_tensorflow, total_both, total_comparison, files_with_errors = analyze_json_files(log_dir)
    
    # 在控制台打印统计摘要
    print("\n分析完成！")
    print("总错误统计:")
    print(f"  - 仅 PyTorch 报错: {total_torch}")
    print(f"  - 仅 TensorFlow 报错: {total_tensorflow}")
    print(f"  - 两个框架都报错: {total_both}")
    print(f"  - comparison_error: {total_comparison}")
    print(f"  - 包含错误的文件数: {len(files_with_errors)}")
    
    # 生成详细报告文件
    generate_report(output_file, total_torch, total_tensorflow, total_both, total_comparison, files_with_errors)
    print(f"\n报告已生成: {output_file}")

if __name__ == "__main__":
    main()
