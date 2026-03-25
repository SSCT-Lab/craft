# ./component/migrate_compare.py
# 执行迁移的测试并对比 TensorFlow 和 PyTorch 的结果
import json
import ast
import subprocess
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import re

def load_jsonl(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []

def extract_source_info(migrated_file, tests_tf_mapped=None):
    """从迁移后的文件中提取源文件信息"""
    try:
        content = Path(migrated_file).read_text()
        # 查找 source 注释
        match = re.search(r'# source: (.+?):(.+)', content)
        if match:
            file_path = match.group(1)
            test_name = match.group(2)
            
            # 如果测试名是 unknown_test_*，尝试从 tests_tf.mapped.jsonl 中查找真实名称
            if test_name.startswith("unknown_test_") and tests_tf_mapped:
                # 从文件名中提取可能的匹配
                file_key = file_path.replace("framework/tensorflow-master/", "")
                for test_data in tests_tf_mapped:
                    if file_key in test_data.get("file", "") or test_data.get("file", "").endswith(Path(file_path).name):
                        # 找到匹配的文件，使用真实的测试函数名
                        real_name = test_data.get("name", test_name)
                        if real_name.startswith("test"):
                            return file_path, real_name
            
            return file_path, test_name
    except:
        pass
    return None, None

def extract_test_function_code(file_path, test_name):
    """从 TensorFlow 测试文件中提取测试函数代码"""
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        
        # 如果测试函数名是 unknown_test_*，尝试从文件中查找实际的测试函数
        actual_test_name = test_name
        if test_name.startswith("unknown_test_"):
            # 查找所有以 test 开头的函数
            test_funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test")]
            if test_funcs:
                # 使用第一个找到的测试函数
                actual_test_name = test_funcs[0]
        
        # 查找测试函数
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == actual_test_name:
                lines = source.split('\n')
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else len(lines)
                func_lines = lines[start_line:end_line]
                
                # 移除基础缩进（假设第一行是函数定义）
                if func_lines:
                    base_indent = len(func_lines[0]) - len(func_lines[0].lstrip())
                    if base_indent > 0:
                        func_lines = [line[base_indent:] if len(line) > base_indent else line for line in func_lines]
                
                return '\n'.join(func_lines), actual_test_name
    except Exception as e:
        pass
    return None, test_name

def run_tf_test(tf_file_path, test_name, tf_root):
    """运行 TensorFlow 测试 - 直接运行原始测试文件"""
    try:
        # 处理文件路径
        if tf_file_path.startswith("framework/tensorflow-master/"):
            full_path = Path(tf_file_path)
        elif Path(tf_file_path).exists():
            full_path = Path(tf_file_path)
        else:
            full_path = tf_root / tf_file_path
        
        if not full_path.exists():
            return {"status": "not_found", "error": f"文件不存在: {full_path}"}
        
        # 提取测试函数代码（返回代码和实际函数名）
        test_code, actual_test_name = extract_test_function_code(full_path, test_name)
        
        if not test_code:
            return {"status": "not_found", "error": f"无法找到测试函数 {test_name} (尝试查找 {actual_test_name})"}
        
        # 检查测试代码是否是类方法（包含 self 参数）
        is_class_method = 'self' in test_code and 'def ' in test_code
        
        # 设置环境变量
        env = os.environ.copy()
        env['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少 TensorFlow 日志
        env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        if is_class_method:
            # 如果是类方法，使用 pytest 直接运行原始测试文件
            # pytest 格式: file_path::ClassName::test_method_name
            # 需要找到类名
            try:
                source = full_path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source)
                
                # 查找包含测试函数的类
                class_name = None
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for child in node.body:
                            if isinstance(child, ast.FunctionDef) and child.name == actual_test_name:
                                class_name = node.name
                                break
                        if class_name:
                            break
                
                if class_name:
                    # 使用 pytest 运行: file::ClassName::test_method
                    # 使用绝对路径避免路径问题
                    pytest_target = f"{full_path.absolute()}::{class_name}::{actual_test_name}"
                    result = subprocess.run(
                        ["python", "-m", "pytest", pytest_target, "-v", "--tb=short", "-q"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=str(full_path.parent.absolute()),
                        env=env
                    )
                else:
                    # 如果找不到类名，尝试直接运行文件并过滤测试
                    result = subprocess.run(
                        ["python", "-m", "pytest", str(full_path.absolute()), 
                         f"-k={actual_test_name}", "-v", "--tb=short", "-q"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=str(full_path.parent.absolute()),
                        env=env
                    )
            except Exception as e:
                # 如果 pytest 失败，尝试使用 tf.test.main
                try:
                    # 创建一个临时脚本来运行测试
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                        tmp_script = f"""import sys
sys.path.insert(0, '{str(full_path.parent.absolute())}')
import tensorflow as tf
from {full_path.stem} import {class_name}

# 创建测试实例并运行
test_instance = {class_name}()
test_method = getattr(test_instance, '{actual_test_name}')
try:
    test_method()
    print("PASS")
except Exception as e:
    print(f"FAIL: {{e}}")
    import traceback
    traceback.print_exc()
"""
                        tmp_file.write(tmp_script)
                        tmp_path = tmp_file.name
                    
                    result = subprocess.run(
                        ["python", tmp_path],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=str(full_path.parent.absolute()),
                        env=env
                    )
                    Path(tmp_path).unlink()
                except Exception as e2:
                    return {"status": "error", "error": f"pytest 和 tf.test 都失败: {str(e)}, {str(e2)}"}
        else:
            # 如果是独立函数，使用 pytest 的 -k 参数过滤
            result = subprocess.run(
                ["python", "-m", "pytest", str(full_path.absolute()), 
                 f"-k={actual_test_name}", "-v", "--tb=short", "-q"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(full_path.parent.absolute()),
                env=env
            )
        
        # 判断测试是否通过
        # pytest 返回码: 0=通过, 非0=失败
        status = "pass" if result.returncode == 0 else "fail"
        
        return {
            "status": status,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "actual_test_name": actual_test_name
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": "执行超时"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def run_pt_test(pt_file_path):
    """运行 PyTorch 测试"""
    try:
        result = subprocess.run(
            ["pytest", str(pt_file_path), "-q", "--disable-warnings", "-v"],
            capture_output=True,
            text=True,
            timeout=30
        )
        return {
            "status": "pass" if result.returncode == 0 else "fail",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": "执行超时"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def compare_results(tf_result, pt_result):
    """对比 TensorFlow 和 PyTorch 的测试结果"""
    comparison = {
        "tf_status": tf_result.get("status", "unknown"),
        "pt_status": pt_result.get("status", "unknown"),
        "match": False,
        "notes": []
    }
    
    # 如果两者都通过，则认为匹配
    if tf_result.get("status") == "pass" and pt_result.get("status") == "pass":
        comparison["match"] = True
        comparison["notes"].append("两者都通过")
    # 如果两者都失败，需要进一步检查
    elif tf_result.get("status") == "fail" and pt_result.get("status") == "fail":
        comparison["match"] = True  # 都失败也算匹配（可能是测试本身有问题）
        comparison["notes"].append("两者都失败")
    else:
        comparison["match"] = False
        comparison["notes"].append(f"TF: {tf_result.get('status')}, PT: {pt_result.get('status')}")
    
    return comparison

def main():
    parser = argparse.ArgumentParser(description="执行并对比 TensorFlow 和 PyTorch 测试")
    parser.add_argument("--migrated-dir", default="migrated_tests", help="迁移后的测试目录")
    parser.add_argument("--tf-root", default="framework/tensorflow-master", help="TensorFlow 源码根目录")
    Path("data/results").mkdir(parents=True, exist_ok=True)
    parser.add_argument("--out", default="data/results/migrate_comparison.jsonl", help="对比结果输出文件")
    parser.add_argument("--limit", type=int, default=-1, help="限制测试数量，-1 表示全部")
    parser.add_argument("--skip-tf", action="store_true", help="跳过 TensorFlow 测试执行（只运行 PyTorch）")
    parser.add_argument("--tf-results", default="data/results/tf_test_results.jsonl", 
                       help="TensorFlow 测试结果静态文件（如果存在，直接读取而不执行）")
    args = parser.parse_args()
    
    migrated_dir = Path(args.migrated_dir)
    tf_root = Path(args.tf_root)
    out_file = Path(args.out)
    out_file.parent.mkdir(exist_ok=True, parents=True)
    
    # 获取所有迁移的测试文件
    test_files = sorted(migrated_dir.glob("*.py"))
    
    if args.limit > 0:
        test_files = test_files[:args.limit]
        print(f"[INFO] 限制测试数量: {args.limit}")
    
    print(f"[INFO] 找到 {len(test_files)} 个迁移的测试文件")
    
    # 加载 tests_tf.mapped.jsonl 用于查找真实的测试函数名
    tests_tf_mapped_path = Path("data/mapping/tests_tf.mapped.jsonl")
    tests_tf_mapped = None
    if tests_tf_mapped_path.exists():
        tests_tf_mapped = load_jsonl(tests_tf_mapped_path)
        print(f"[INFO] 加载了 {len(tests_tf_mapped)} 条测试元数据")
    
    # 加载 TensorFlow 测试结果静态文件（如果存在）
    tf_results_cache = {}
    tf_results_path = Path(args.tf_results)
    if tf_results_path.exists() and not args.skip_tf:
        print(f"[INFO] 加载 TensorFlow 测试结果缓存: {tf_results_path}")
        tf_results_data = load_jsonl(tf_results_path)
        # 构建缓存：使用多种键格式，方便匹配
        for item in tf_results_data:
            file_path = item.get("file", "")
            test_name = item.get("test_name", "")
            tf_result = item.get("tf_result")
            
            # 键1: 完整路径
            key1 = (file_path, test_name)
            tf_results_cache[key1] = tf_result
            
            # 键2: 去掉 framework/tensorflow-master/ 前缀
            if file_path.startswith("framework/tensorflow-master/"):
                key2 = (file_path.replace("framework/tensorflow-master/", ""), test_name)
                tf_results_cache[key2] = tf_result
            
            # 键3: 只使用文件名
            key3 = (Path(file_path).name, test_name)
            if key3 not in tf_results_cache:  # 避免覆盖
                tf_results_cache[key3] = tf_result
        
        print(f"[INFO] 加载了 {len(tf_results_data)} 条 TensorFlow 测试结果（构建了 {len(tf_results_cache)} 个缓存键）")
    
    results = []
    
    with open(out_file, "w") as fout:
        for pt_file in tqdm(test_files, desc="Comparing tests"):
            # 提取源文件信息（传入 tests_tf_mapped 用于查找真实函数名）
            tf_file_path, test_name = extract_source_info(pt_file, tests_tf_mapped)
            
            if not tf_file_path:
                result = {
                    "pt_file": str(pt_file),
                    "status": "error",
                    "error": "无法提取源文件信息"
                }
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                continue
            
            # 运行 PyTorch 测试
            pt_result = run_pt_test(pt_file)
            
            # 获取 TensorFlow 测试结果（优先使用静态文件查表）
            tf_result = None
            if not args.skip_tf:
                # 尝试多种键格式匹配
                cache_key1 = (tf_file_path, test_name)  # 完整路径
                cache_key2 = (tf_file_path.replace("framework/tensorflow-master/", ""), test_name)  # 去掉前缀
                cache_key3 = (Path(tf_file_path).name, test_name)  # 只文件名
                
                if cache_key1 in tf_results_cache:
                    tf_result = tf_results_cache[cache_key1]
                elif cache_key2 in tf_results_cache:
                    tf_result = tf_results_cache[cache_key2]
                elif cache_key3 in tf_results_cache:
                    tf_result = tf_results_cache[cache_key3]
                
                if tf_result:
                    # 使用静态文件中的结果（查表成功）
                    pass
                else:
                    # 静态文件中没有，执行测试（但建议先收集）
                    if tf_results_cache:
                        # 如果已经有缓存但未命中，给出提示
                        print(f"[WARN] 静态文件中未找到: {tf_file_path}:{test_name}")
                        print(f"[INFO] 建议先运行: python3 component/migrate_collect_tf_results.py")
                    tf_result = run_tf_test(tf_file_path, test_name, tf_root)
            
            # 对比结果
            comparison = None
            if tf_result:
                comparison = compare_results(tf_result, pt_result)
            
            # 保存结果
            result = {
                "pt_file": str(pt_file),
                "tf_file": tf_file_path,
                "test_name": test_name,
                "pt_result": pt_result,
                "tf_result": tf_result,
                "comparison": comparison
            }
            
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(result)
    
    # 生成摘要
    print("\n==== COMPARISON SUMMARY ====")
    total = len(results)
    pt_pass = sum(1 for r in results if r.get("pt_result", {}).get("status") == "pass")
    pt_fail = sum(1 for r in results if r.get("pt_result", {}).get("status") == "fail")
    
    print(f"总测试数: {total}")
    print(f"PyTorch 通过: {pt_pass}")
    print(f"PyTorch 失败: {pt_fail}")
    print(f"PyTorch 通过率: {pt_pass/total*100:.1f}%")
    
    if not args.skip_tf:
        tf_pass = sum(1 for r in results if r.get("tf_result", {}).get("status") == "pass")
        tf_fail = sum(1 for r in results if r.get("tf_result", {}).get("status") == "fail")
        matches = sum(1 for r in results if r.get("comparison", {}).get("match", False))
        
        print(f"TensorFlow 通过: {tf_pass}")
        print(f"TensorFlow 失败: {tf_fail}")
        print(f"结果匹配: {matches}/{total} ({matches/total*100:.1f}%)")
    
    print(f"\n详细结果已保存到: {out_file}")

if __name__ == "__main__":
    main()

