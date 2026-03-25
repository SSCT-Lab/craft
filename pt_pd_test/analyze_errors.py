import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_json_files(log_dir):
    """分析pt_pd_log文件夹中的llm_enhanced_torch开头的JSON文件"""
    
    # 统计总错误数
    total_torch_errors = 0
    total_paddle_errors = 0
    total_comparison_errors = 0
    
    # 存储有错误的文件信息
    files_with_errors = []
    
    # 获取所有符合条件的JSON文件
    log_path = Path(log_dir)
    json_files = sorted(log_path.glob("llm_enhanced_torch*.json"))
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 统计当前文件的错误
            file_torch_errors = 0
            file_paddle_errors = 0
            file_comparison_errors = 0
            error_iterations = {
                'torch_error': [],
                'paddle_error': [],
                'comparison_error': []
            }
            
            # 遍历所有测试结果
            if 'results' in data:
                for result in data['results']:
                    if 'execution_result' in result:
                        exec_result = result['execution_result']
                        iteration = result.get('iteration', 'N/A')
                        
                        # 检查torch_error
                        if exec_result.get('torch_error') is not None:
                            file_torch_errors += 1
                            total_torch_errors += 1
                            error_iterations['torch_error'].append(iteration)
                        
                        # 检查paddle_error
                        if exec_result.get('paddle_error') is not None:
                            file_paddle_errors += 1
                            total_paddle_errors += 1
                            error_iterations['paddle_error'].append(iteration)
                        
                        # 检查comparison_error
                        if exec_result.get('comparison_error') is not None:
                            file_comparison_errors += 1
                            total_comparison_errors += 1
                            error_iterations['comparison_error'].append(iteration)
            
            # 如果有任何错误，记录该文件
            if file_torch_errors > 0 or file_paddle_errors > 0 or file_comparison_errors > 0:
                files_with_errors.append({
                    'filename': json_file.name,
                    'torch_errors': file_torch_errors,
                    'paddle_errors': file_paddle_errors,
                    'comparison_errors': file_comparison_errors,
                    'torch_error_iterations': error_iterations['torch_error'],
                    'paddle_error_iterations': error_iterations['paddle_error'],
                    'comparison_error_iterations': error_iterations['comparison_error']
                })
        
        except Exception as e:
            print(f"处理文件 {json_file.name} 时出错: {e}")
    
    return total_torch_errors, total_paddle_errors, total_comparison_errors, files_with_errors

def generate_report(output_file, total_torch, total_paddle, total_comparison, files_with_errors):
    """生成错误报告文本文件"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch-PaddlePaddle 测试错误分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 总体统计
        f.write("【总体错误统计】\n")
        f.write(f"torch_error 非null值总数: {total_torch}\n")
        f.write(f"paddle_error 非null值总数: {total_paddle}\n")
        f.write(f"comparison_error 非null值总数: {total_comparison}\n")
        f.write(f"包含错误的文件总数: {len(files_with_errors)}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # 详细文件错误信息
        if files_with_errors:
            f.write("【详细错误文件信息】\n\n")
            
            for idx, file_info in enumerate(files_with_errors, 1):
                f.write(f"{idx}. 文件名: {file_info['filename']}\n")
                f.write("-" * 80 + "\n")
                
                # torch_error信息
                if file_info['torch_errors'] > 0:
                    f.write(f"   torch_error 非null值个数: {file_info['torch_errors']}\n")
                    f.write(f"   对应用例的iteration值: {', '.join(map(str, file_info['torch_error_iterations']))}\n")
                
                # paddle_error信息
                if file_info['paddle_errors'] > 0:
                    f.write(f"   paddle_error 非null值个数: {file_info['paddle_errors']}\n")
                    f.write(f"   对应用例的iteration值: {', '.join(map(str, file_info['paddle_error_iterations']))}\n")
                
                # comparison_error信息
                if file_info['comparison_errors'] > 0:
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
    # 设置路径
    log_dir = r"d:\graduate\DFrameworkTest\pt_pd_test\pt_pd_log"
    output_file = r"d:\graduate\DFrameworkTest\pt_pd_test\error_analysis_report.txt"
    
    print("开始分析JSON文件...")
    
    # 分析文件
    total_torch, total_paddle, total_comparison, files_with_errors = analyze_json_files(log_dir)
    
    # 生成报告
    print(f"\n分析完成！")
    print(f"总错误统计:")
    print(f"  - torch_error: {total_torch}")
    print(f"  - paddle_error: {total_paddle}")
    print(f"  - comparison_error: {total_comparison}")
    print(f"  - 包含错误的文件数: {len(files_with_errors)}")
    
    generate_report(output_file, total_torch, total_paddle, total_comparison, files_with_errors)
    
    print(f"\n报告已生成: {output_file}")

if __name__ == "__main__":
    main()
