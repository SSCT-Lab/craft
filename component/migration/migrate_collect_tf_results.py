# ./component/migrate_collect_tf_results.py
# 批量运行 TensorFlow 测试并保存结果到静态文件
import json
import ast
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import re

def load_jsonl(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []

def extract_test_function_code(file_path, test_name):
    """从 TensorFlow 测试文件中提取测试函数代码"""
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        
        # 如果测试函数名是 unknown_test_*，尝试从文件中查找实际的测试函数
        actual_test_name = test_name
        if test_name.startswith("unknown_test_"):
            test_funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test")]
            if test_funcs:
                actual_test_name = test_funcs[0]
        
        # 查找测试函数
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == actual_test_name:
                lines = source.split('\n')
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else len(lines)
                func_lines = lines[start_line:end_line]
                
                # 移除基础缩进
                if func_lines:
                    base_indent = len(func_lines[0]) - len(func_lines[0].lstrip())
                    if base_indent > 0:
                        func_lines = [line[base_indent:] if len(line) > base_indent else line for line in func_lines]
                
                return '\n'.join(func_lines), actual_test_name
    except Exception as e:
        pass
    return None, test_name

def find_test_class(file_path, test_name):
    """查找包含测试函数的类名"""
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == test_name:
                        return node.name
    except:
        pass
    return None

def run_tf_test_with_pytest(tf_file_path, test_name, tf_root, extracted_test_file=None):
    """运行 TensorFlow 测试 - 优先使用提取的独立测试文件"""
    # 如果提供了提取的测试文件，直接运行它
    if extracted_test_file and Path(extracted_test_file).exists():
        env = os.environ.copy()
        env['TF_CPP_MIN_LOG_LEVEL'] = '2'
        env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        try:
            result = subprocess.run(
                ["python", str(extracted_test_file)],
                capture_output=True,
                text=True,
                timeout=60,
                env=env
            )
            
            status = "pass" if "PASS" in result.stdout or result.returncode == 0 else "fail"
            
            return {
                "status": status,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "actual_test_name": test_name,
                "source": "extracted"  # 标记来源
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "执行超时", "source": "extracted"}
        except Exception as e:
            return {"status": "error", "error": str(e), "source": "extracted"}
    
    # 否则使用原始方法（pytest）
    # 处理文件路径
    if tf_file_path.startswith("framework/tensorflow-master/"):
        full_path = Path(tf_file_path)
    elif Path(tf_file_path).exists():
        full_path = Path(tf_file_path)
    else:
        full_path = tf_root / tf_file_path
    
    if not full_path.exists():
        return {"status": "not_found", "error": f"文件不存在: {full_path}"}
    
    # 提取测试函数代码以获取实际函数名
    test_code, actual_test_name = extract_test_function_code(full_path, test_name)
    
    if not test_code:
        return {"status": "not_found", "error": f"无法找到测试函数 {test_name}"}
    
    # 检查是否是类方法
    is_class_method = 'self' in test_code and 'def ' in test_code
    
    # 设置环境变量
    env = os.environ.copy()
    env['TF_CPP_MIN_LOG_LEVEL'] = '2'
    env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    try:
        if is_class_method:
            # 查找类名
            class_name = find_test_class(full_path, actual_test_name)
            if class_name:
                pytest_target = f"{full_path.absolute()}::{class_name}::{actual_test_name}"
            else:
                pytest_target = str(full_path.absolute())
                actual_test_name = f"-k={actual_test_name}"
        else:
            pytest_target = str(full_path.absolute())
            actual_test_name = f"-k={actual_test_name}"
        
        # 运行 pytest
        cmd = ["python", "-m", "pytest", pytest_target]
        if not is_class_method or not class_name:
            cmd.append(actual_test_name)
        cmd.extend(["-v", "--tb=short", "-q"])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(full_path.parent.absolute()),
            env=env
        )
        
        status = "pass" if result.returncode == 0 else "fail"
        
        return {
            "status": status,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "actual_test_name": actual_test_name,
            "source": "original"  # 标记来源
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": "执行超时", "source": "original"}
    except Exception as e:
        return {"status": "error", "error": str(e), "source": "original"}

def main():
    parser = argparse.ArgumentParser(description="批量运行 TensorFlow 测试并保存结果到静态文件")
    Path("data/results").mkdir(parents=True, exist_ok=True)
    parser.add_argument("--input", default="data/migration/migration_candidates_fuzzy.jsonl", 
                       help="输入文件（候选测试列表或 tests_tf.mapped.jsonl）")
    parser.add_argument("--tf-root", default="framework/tensorflow-master", 
                       help="TensorFlow 源码根目录")
    parser.add_argument("--output", default="data/results/tf_test_results.jsonl", 
                       help="输出结果文件")
    parser.add_argument("--limit", type=int, default=-1, 
                       help="限制测试数量，-1 表示全部")
    parser.add_argument("--tests-tf-mapped", default="data/mapping/tests_tf.mapped.jsonl",
                       help="测试元数据文件（用于查找真实测试函数名）")
    parser.add_argument("--from-tests-tf", action="store_true",
                       help="直接从 tests_tf.mapped.jsonl 读取所有测试（而不是从 candidates）")
    args = parser.parse_args()
    
    # 加载测试列表
    if args.from_tests_tf and Path(args.input).exists():
        # 直接从 tests_tf.mapped.jsonl 读取所有测试
        all_tests = load_jsonl(args.input)
        print(f"[LOAD] 从 {args.input} 加载了 {len(all_tests)} 个测试")
        # 转换为统一的格式
        candidates = []
        for test in all_tests:
            candidates.append({
                "file": test.get("file", ""),
                "name": test.get("name", ""),
                "apis_used": test.get("apis", [])
            })
    else:
        # 从候选文件读取
        candidates = load_jsonl(args.input)
        print(f"[LOAD] 从 {args.input} 加载了 {len(candidates)} 个候选测试")
    
    # 加载测试元数据（用于查找真实测试函数名）
    tests_tf_mapped = None
    if Path(args.tests_tf_mapped).exists():
        tests_tf_mapped = load_jsonl(args.tests_tf_mapped)
        print(f"[LOAD] 加载了 {len(tests_tf_mapped)} 条测试元数据")
    
    # 限制数量
    if args.limit > 0:
        candidates = candidates[:args.limit]
        print(f"[INFO] 限制测试数量: {args.limit}")
    
    # 创建输出文件
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    tf_root = Path(args.tf_root)
    results = []
    
    # 提取源文件信息（类似 migrate_compare.py 的逻辑）
    def extract_source_info(file_path, test_name, tests_tf_mapped):
        """从文件路径和测试名中提取信息"""
        if tests_tf_mapped:
            file_key = file_path.replace("framework/tensorflow-master/", "")
            for test_data in tests_tf_mapped:
                if file_key in test_data.get("file", "") or test_data.get("file", "").endswith(Path(file_path).name):
                    real_name = test_data.get("name", test_name)
                    if real_name.startswith("test"):
                        return file_path, real_name
        return file_path, test_name
    
    # 检查是否有提取的测试文件目录
    extracted_dir = Path(args.extracted_dir)
    extracted_tests_map = {}
    if extracted_dir.exists():
        print(f"[INFO] 发现提取的测试文件目录: {extracted_dir}")
        # 构建映射：从原始文件路径和测试名到提取的文件
        for extracted_file in extracted_dir.glob("*.py"):
            # 从文件内容中提取原始信息
            try:
                content = extracted_file.read_text()
                # 查找注释中的原始信息
                file_match = re.search(r'# Extracted from: (.+?):(.+)', content)
                if file_match:
                    orig_file = file_match.group(1)
                    orig_test = file_match.group(2)
                    extracted_tests_map[(orig_file, orig_test)] = extracted_file
            except:
                pass
        print(f"[INFO] 映射了 {len(extracted_tests_map)} 个提取的测试文件")
    
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in tqdm(candidates, desc="Running TF tests"):
            file_path = item.get("file", "")
            test_name = item.get("name", "")
            
            # 提取真实测试函数名
            tf_file_path, actual_test_name = extract_source_info(file_path, test_name, tests_tf_mapped)
            
            # 检查是否有提取的测试文件
            extracted_test_file = extracted_tests_map.get((file_path, test_name)) or \
                                 extracted_tests_map.get((tf_file_path, actual_test_name))
            
            # 运行测试（优先使用提取的文件）
            tf_result = run_tf_test_with_pytest(tf_file_path, actual_test_name, tf_root, 
                                               extracted_test_file=str(extracted_test_file) if extracted_test_file else None)
            
            # 保存结果（使用统一的键格式，方便后续查表）
            result = {
                "file": file_path,  # 原始文件路径（用于匹配）
                "test_name": test_name,  # 原始测试名（用于匹配）
                "actual_test_name": actual_test_name,  # 实际执行的测试函数名
                "tf_file": tf_file_path,  # TensorFlow 文件路径
                "tf_result": tf_result,  # 执行结果（包含 status, stdout, stderr, returncode）
                "apis_used": item.get("apis_used", []),  # 使用的 API（如果有）
                "timestamp": str(Path(__file__).stat().st_mtime)  # 收集时间戳
            }
            
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(result)
    
    # 统计
    total = len(results)
    passed = sum(1 for r in results if r.get("tf_result", {}).get("status") == "pass")
    failed = sum(1 for r in results if r.get("tf_result", {}).get("status") == "fail")
    error = sum(1 for r in results if r.get("tf_result", {}).get("status") in ["error", "timeout", "not_found"])
    
    print("\n==== COLLECTION SUMMARY ====")
    print(f"总测试数: {total}")
    print(f"通过: {passed} ({passed/total*100:.1f}%)")
    print(f"失败: {failed} ({failed/total*100:.1f}%)")
    print(f"错误/超时: {error} ({error/total*100:.1f}%)")
    print(f"\n结果已保存到: {output_path}")

if __name__ == "__main__":
    main()

