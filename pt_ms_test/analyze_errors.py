import json
from pathlib import Path

def analyze_json_files(log_dir):
    total_torch_errors = 0
    total_mindspore_errors = 0
    total_comparison_errors = 0
    files_with_errors = []
    log_path = Path(log_dir)
    json_files = sorted(log_path.glob("llm_enhanced_torch*.json"))
    print(f"找到 {len(json_files)} 个JSON文件")
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            file_torch_errors = 0
            file_mindspore_errors = 0
            file_comparison_errors = 0
            error_iterations = {"torch_error": [], "mindspore_error": [], "comparison_error": []}
            if "results" in data:
                for result in data["results"]:
                    if "execution_result" in result:
                        exec_result = result["execution_result"]
                        iteration = result.get("iteration", "N/A")
                        if exec_result.get("torch_error") is not None:
                            file_torch_errors += 1
                            total_torch_errors += 1
                            error_iterations["torch_error"].append(iteration)
                        if exec_result.get("mindspore_error") is not None:
                            file_mindspore_errors += 1
                            total_mindspore_errors += 1
                            error_iterations["mindspore_error"].append(iteration)
                        if exec_result.get("comparison_error") is not None:
                            file_comparison_errors += 1
                            total_comparison_errors += 1
                            error_iterations["comparison_error"].append(iteration)
            if file_torch_errors > 0 or file_mindspore_errors > 0 or file_comparison_errors > 0:
                files_with_errors.append(
                    {
                        "filename": json_file.name,
                        "torch_errors": file_torch_errors,
                        "mindspore_errors": file_mindspore_errors,
                        "comparison_errors": file_comparison_errors,
                        "torch_error_iterations": error_iterations["torch_error"],
                        "mindspore_error_iterations": error_iterations["mindspore_error"],
                        "comparison_error_iterations": error_iterations["comparison_error"],
                    }
                )
        except Exception as e:
            print(f"处理文件 {json_file.name} 时出错: {e}")
    return total_torch_errors, total_mindspore_errors, total_comparison_errors, files_with_errors

def generate_report(output_file, total_torch, total_mindspore, total_comparison, files_with_errors):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch-MindSpore 测试错误分析报告\n")
        f.write("=" * 80 + "\n\n")
        f.write("【总体错误统计】\n")
        f.write(f"torch_error 非null值总数: {total_torch}\n")
        f.write(f"mindspore_error 非null值总数: {total_mindspore}\n")
        f.write(f"comparison_error 非null值总数: {total_comparison}\n")
        f.write(f"包含错误的文件总数: {len(files_with_errors)}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        if files_with_errors:
            f.write("【详细错误文件信息】\n\n")
            for idx, file_info in enumerate(files_with_errors, 1):
                f.write(f"{idx}. 文件名: {file_info['filename']}\n")
                f.write("-" * 80 + "\n")
                if file_info["torch_errors"] > 0:
                    f.write(f"   torch_error 非null值个数: {file_info['torch_errors']}\n")
                    f.write(f"   对应用例的iteration值: {', '.join(map(str, file_info['torch_error_iterations']))}\n")
                if file_info["mindspore_errors"] > 0:
                    f.write(f"   mindspore_error 非null值个数: {file_info['mindspore_errors']}\n")
                    f.write(f"   对应用例的iteration值: {', '.join(map(str, file_info['mindspore_error_iterations']))}\n")
                if file_info["comparison_errors"] > 0:
                    f.write(f"   comparison_error 非null值个数: {file_info['comparison_errors']}\n")
                    f.write(f"   对应用例的iteration值: {', '.join(map(str, file_info['comparison_error_iterations']))}\n")
                f.write("\n")
        else:
            f.write("【详细错误文件信息】\n\n")
            f.write("未发现任何包含错误的文件。\n\n")
        f.write("=" * 80 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 80 + "\n")

def main():
    log_dir = r"d:\graduate\DFrameworkTest\pt_ms_test\pt_ms_log"
    output_file = r"d:\graduate\DFrameworkTest\pt_ms_test\error_analysis_report.txt"
    print("开始分析JSON文件...")
    total_torch, total_mindspore, total_comparison, files_with_errors = analyze_json_files(log_dir)
    print("\n分析完成！")
    print("总错误统计:")
    print(f"  - torch_error: {total_torch}")
    print(f"  - mindspore_error: {total_mindspore}")
    print(f"  - comparison_error: {total_comparison}")
    print(f"  - 包含错误的文件数: {len(files_with_errors)}")
    generate_report(output_file, total_torch, total_mindspore, total_comparison, files_with_errors)
    print(f"\n报告已生成: {output_file}")

if __name__ == "__main__":
    main()
