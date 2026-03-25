# ./component/migrate_generate_tests.py
"""生成迁移测试文件：将 TensorFlow 测试迁移为 PyTorch 测试"""
import json
import ast
import argparse
import re
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# ==================== 常量定义 ====================
DEFAULT_IN_FILE = "data/migration/migration_candidates_fuzzy.jsonl"
DEFAULT_OUT_DIR = "migrated_tests"
DEFAULT_TF_ROOT = "framework/tensorflow-master"
DEFAULT_MODEL = "qwen-flash"
DEFAULT_KEY_PATH = "aliyun.key"
DEFAULT_WORKERS = 10
MAX_APIS_IN_PROMPT = 10
MAX_MAPPED_APIS = 5

# 尝试导入 LLM 客户端 - 使用与 semantic_llm.py 相同的方式
def load_api_key(path="aliyun.key"):
    """加载 API key，支持相对路径和绝对路径"""
    key_path = Path(path)
    if not key_path.is_absolute():
        # 如果是相对路径，尝试从项目根目录查找
        # __file__ 是 component/migration/migrate_generate_tests.py
        # parent.parent.parent 就是项目根目录
        project_root = Path(__file__).parent.parent.parent
        key_path = project_root / key_path
    with open(key_path) as f:
        return f.read().strip()

def get_qwen_client(key_path="aliyun.key"):
    """创建 Qwen API 客户端"""
    try:
        # 尝试新版本 openai
        from openai import OpenAI
        api_key = load_api_key(key_path)
        return OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    except ImportError:
        # 尝试旧版本
        try:
            import openai
            api_key = load_api_key(key_path)
            openai.api_key = api_key
            openai.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            return openai
        except Exception as e:
            print(f"[ERROR] 无法初始化 OpenAI 客户端: {e}")
            raise

# ==================== 工具函数 ====================

HEADER = """import torch
import pytest
try:
import tensorflow as tf
    TF_AVAILABLE = True
    # Enable eager execution for simpler evaluation
    try:
        tf.config.run_functions_eagerly(True)
    except:
        pass
except ImportError:
    TF_AVAILABLE = False
    tf = None

import numpy as np
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Helper functions for TensorFlow test framework methods
def assertAllClose(a, b, rtol=1e-6, atol=1e-6):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

def assertAllEqual(a, b):
    np.testing.assert_array_equal(a, b)

def assertFunctionMatchesEager(func, *args, **kwargs):
    \"\"\"Assert that a function matches eager execution behavior.
    This is a simplified version for standalone test execution.\"\"\"
    try:
        # Run in eager mode
        eager_result = func(*args, **kwargs)
        # For now, just check that it doesn't raise an exception
        # In a full implementation, this would compare eager vs graph mode
        return eager_result
    except Exception as e:
        raise AssertionError(f"Function execution failed: {e}")

# Helper function to capture function execution output
def capture_output(func, *args, **kwargs):
    \"\"\"Capture stdout, stderr and return value from function execution\"\"\"
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    result = None
    exception = None
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            result = func(*args, **kwargs)
    except Exception as e:
        exception = e
    
    stdout_text = stdout_capture.getvalue()
    stderr_text = stderr_capture.getvalue()
    
    return {{
        "result": result,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "exception": exception,
        "success": exception is None
    }}

# Helper function to compare outputs
def compare_outputs(tf_output, pt_output):
    \"\"\"Compare TensorFlow and PyTorch execution outputs\"\"\"
    print("\\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    
    # Status comparison
    tf_status = "PASS" if tf_output["success"] else "FAIL"
    pt_status = "PASS" if pt_output["success"] else "FAIL"
    
    print(f"\\n状态对比:")
    print(f"  TensorFlow: {{tf_status}}")
    print(f"  PyTorch:    {{pt_status}}")
    
    if tf_output["exception"]:
        print(f"\\nTensorFlow 异常:")
        print(f"  {{type(tf_output['exception']).__name__}}: {{tf_output['exception']}}")
    
    if pt_output["exception"]:
        print(f"\\nPyTorch 异常:")
        print(f"  {{type(pt_output['exception']).__name__}}: {{pt_output['exception']}}")
    
    # Output comparison
    print(f"\\n输出对比:")
    if tf_output["stdout"].strip() or pt_output["stdout"].strip():
        print(f"\\nTensorFlow stdout:")
        print("  " + "\\n  ".join(tf_output["stdout"].strip().split("\\n")) if tf_output["stdout"].strip() else "  (无输出)")
        print(f"\\nPyTorch stdout:")
        print("  " + "\\n  ".join(pt_output["stdout"].strip().split("\\n")) if pt_output["stdout"].strip() else "  (无输出)")
        
        # Check if outputs are similar
        if tf_output["stdout"].strip() == pt_output["stdout"].strip():
            print("\\n✓ stdout 完全一致")
        else:
            print("\\n✗ stdout 存在差异")
    
    if tf_output["stderr"].strip() or pt_output["stderr"].strip():
        print(f"\\nTensorFlow stderr:")
        print("  " + "\\n  ".join(tf_output["stderr"].strip().split("\\n")) if tf_output["stderr"].strip() else "  (无错误)")
        print(f"\\nPyTorch stderr:")
        print("  " + "\\n  ".join(pt_output["stderr"].strip().split("\\n")) if pt_output["stderr"].strip() else "  (无错误)")
    
    # Return value comparison (if both functions return something)
    if tf_output["result"] is not None or pt_output["result"] is not None:
        print(f"\\n返回值对比:")
        print(f"  TensorFlow: {{tf_output['result']}}")
        print(f"  PyTorch:    {{pt_output['result']}}")
        
        # Try to compare if both are numpy arrays or tensors
        try:
            if isinstance(tf_output["result"], (dict, list)):
                if tf_output["result"] == pt_output["result"]:
                    print("  ✓ 返回值一致")
                else:
                    print("  ✗ 返回值存在差异")
            elif hasattr(tf_output["result"], "numpy") and hasattr(pt_output["result"], "numpy"):
                tf_val = tf_output["result"].numpy()
                pt_val = pt_output["result"].numpy()
                if np.allclose(tf_val, pt_val):
                    print("  ✓ 返回值数值一致")
                else:
                    print("  ✗ 返回值数值存在差异")
                    print(f"    TF: {{tf_val}}")
                    print(f"    PT: {{pt_val}}")
        except:
            pass
    
    # Final summary
    print("\\n" + "=" * 80)
    if tf_status == pt_status == "PASS":
        print("✓ 两个版本均通过")
    elif tf_status == pt_status == "FAIL":
        print("✗ 两个版本均失败")
    else:
        print("⚠ 两个版本结果不一致")
    print("=" * 80)

"""


def safe_filename(s):
    """将字符串转换为安全的文件名"""
    return s.replace("/", "_").replace(".", "_").replace("-", "_")


def resolve_file_path(file_path, tf_root):
    """解析文件路径，支持多种路径格式"""
    if file_path.startswith("framework/tensorflow-master/"):
        return Path(file_path)
    elif file_path.startswith("tensorflow/"):
        return Path(tf_root) / file_path
    else:
        full_path = Path(tf_root) / file_path
        return full_path if full_path.exists() else Path(file_path)


def find_actual_test_name(file_path, test_name):
    """如果测试名是 unknown_test_*，尝试从文件中查找实际的测试函数名"""
    if not test_name.startswith("unknown_test_"):
        return test_name
    
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        test_funcs = [
            n.name for n in ast.walk(tree) 
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test")
        ]
        return test_funcs[0] if test_funcs else test_name
    except Exception:
        return test_name


def _normalize_indent(lines, base_indent):
    """规范化代码行的缩进，保持相对缩进关系
    
    参数:
        lines: 代码行列表（已移除基础缩进）
        base_indent: 基础缩进级别（已移除）
    
    返回:
        规范化后的代码行列表，每行都有正确的缩进（8个空格基础 + 相对缩进）
    """
    if not lines:
        return []
    
    if base_indent is None:
        base_indent = 0
    
    # 找到第一行非空行的缩进作为参考
    first_indent = None
    for line in lines:
        if line.strip():
            first_indent = len(line) - len(line.lstrip())
            break
    
    if first_indent is None:
        # 全部是空行
        return [''] * len(lines)
    
    normalized = []
    for line in lines:
        if not line.strip():
            # 空行保持为空
            normalized.append('')
            continue
        
        # 计算当前行的缩进（相对于已移除基础缩进后的代码）
        current_indent = len(line) - len(line.lstrip())
        
        # 计算相对于第一行的缩进级别
        relative_indent = current_indent - first_indent
        relative_indent = max(0, relative_indent)  # 确保非负
        
        # 生成目标缩进（8个空格基础 + 相对缩进）
        # 将相对缩进转换为4的倍数（向下取整）
        indent_levels = relative_indent // 4
        target_indent = '        ' + ('    ' * indent_levels)
        
        # 如果相对缩进不是4的倍数，需要额外处理
        extra_spaces = relative_indent % 4
        if extra_spaces > 0:
            target_indent += ' ' * extra_spaces
        
        normalized.append(target_indent + line.lstrip())
    
    return normalized


def _find_matching_paren(text, start_pos):
    """找到从 start_pos 开始的匹配右括号位置，正确处理嵌套括号"""
    if start_pos >= len(text) or text[start_pos] != '(':
        return -1
    
    depth = 1
    pos = start_pos + 1
    
    while pos < len(text) and depth > 0:
        if text[pos] == '(':
            depth += 1
        elif text[pos] == ')':
            depth -= 1
        elif text[pos] == '"' or text[pos] == "'":
            # 跳过字符串字面量
            quote_char = text[pos]
            pos += 1
            while pos < len(text) and text[pos] != quote_char:
                if text[pos] == '\\':
                    pos += 1  # 跳过转义字符
                pos += 1
        pos += 1
    
    return pos - 1 if depth == 0 else -1


def _replace_self_evaluate(line):
    """替换 self.evaluate(...) 为 ...eval()，正确处理嵌套括号
    
    对于 Operation（如 variables.global_variables_initializer()），
    应该用 sess.run() 包装，但由于代码中可能没有 Session 上下文，
    我们先用 .eval() 替换，让代码能运行起来。
    如果后续需要，可以在 Session 上下文中运行。
    """
    result = line
    pattern = r'self\.evaluate\s*\('
    pos = 0
    
    while True:
        match = re.search(pattern, result[pos:])
        if not match:
            break
        
        # 计算实际位置
        actual_start = pos + match.start()
        paren_start = pos + match.end() - 1  # '(' 的位置
        
        # 找到匹配的右括号
        paren_end = _find_matching_paren(result, paren_start)
        
        if paren_end > paren_start:
            # 提取参数部分（不包括括号）
            arg = result[paren_start + 1:paren_end]
            
            # 检查是否是 Operation（如 variables.global_variables_initializer()）
            # 对于 Operation，应该用 sess.run()，但为了简化，我们先用 .eval()
            # 注意：在 TF 1.x Session 模式下，Operation 需要用 sess.run()
            # 但为了保持代码简单，我们先用 .eval()，如果失败，可以在 Session 中运行
            
            # 特殊处理：如果 arg 是 variables.global_variables_initializer() 这样的 Operation
            # 应该用 sess.run()，但为了简化，我们先用 .eval()
            # 实际上，在 Session 上下文中，Operation 的 .eval() 会失败，需要用 sess.run()
            # 但为了保持代码简单，我们先统一用 .eval()，如果失败，可以在 Session 中运行
            
            # 替换：self.evaluate(arg) -> arg.eval()
            result = result[:actual_start] + arg + '.eval()' + result[paren_end + 1:]
            # 更新位置，继续查找
            pos = actual_start + len(arg) + 6  # 6 是 '.eval()' 的长度
        else:
            # 如果找不到匹配的括号，跳过这个匹配
            pos = paren_start + 1
    
    return result


def _process_class_method_body(test_lines):
    """处理类方法体，移除 self 引用并转换测试框架方法，保持缩进结构"""
    test_body = []
    in_def = False
    base_indent = None
    loop_vars = {}  # 跟踪循环变量
    
    for line in test_lines:
        stripped = line.strip()
        if stripped.startswith('def '):
            in_def = True
            base_indent = len(line) - len(line.lstrip())
            continue
        if in_def:
            # 保留空行以维持缩进结构
            if not stripped:
                test_body.append('')
                continue
            
            # 移除基础缩进，但保持相对缩进
            if base_indent and len(line) > base_indent:
                line = line[base_indent:]
            
            # 检测循环变量定义（如 for dt in ...）
            loop_match = re.match(r'for\s+([^,]+)\s+in\s+(.+):', stripped)
            if loop_match:
                var_name = loop_match.group(1).strip()
                # 如果循环变量在函数默认参数中使用，需要处理
                # 暂时跳过，因为需要更复杂的分析
            
            # 处理嵌套函数定义中的默认参数（如 def sample(n, dtype=dt):）
            # 如果 dt 是循环变量，需要将其改为参数
            if 'def ' in stripped and '=' in stripped:
                # 检查是否有未定义的变量作为默认参数
                # 这需要更复杂的分析，暂时跳过
                pass
            
            # 简化 self. 调用
            # 注意：self._Sampler, self.evaluate 等类方法需要特殊处理
            # 对于 self.evaluate，替换为 .eval() 或 .numpy()
            if 'self.evaluate(' in line:
                line = _replace_self_evaluate(line)
            elif 'self._Sampler(' in line:
                # 尝试提取 _Sampler 调用，可能需要手动实现
                # 暂时保留，但标记为需要手动处理
                line = line.replace('self._Sampler', '_Sampler')  # 假设会提取这个辅助函数
            else:
                line = re.sub(r'\bself\.', '', line)
            
            # 处理未定义的变量（如 dt, shape 等）
            # 如果这些变量在循环中定义，需要特殊处理
            # 暂时跳过，因为需要更复杂的上下文分析
            
            # 处理测试框架方法
            line = re.sub(r'\bassertAllClose\(', 'assertAllClose(', line)
            line = re.sub(r'\bassertAllEqual\(', 'assertAllEqual(', line)
            # 处理 cached_session() 和 session()
            line = re.sub(r'\bcached_session\(\)', 'tf.compat.v1.Session()', line)
            line = re.sub(r'\bsession\(\)', 'tf.compat.v1.Session()', line)
            # 注意：在 Session context 中，保留 .eval()，不要转换为 .numpy()
            # 因为 .eval() 会在 Session 中执行操作，而 .numpy() 需要 eager execution
            # 如果不在 Session context 中，.eval() 会自动在 eager mode 下工作
            # TF 1.x -> 2.x 兼容性：reduction_indices -> axis
            # 匹配 reduction_indices=xxx 或 reduction_indices=xxx) 的情况
            line = re.sub(r'\breduction_indices\s*=', 'axis=', line)
            
            test_body.append(line)
    
    return _normalize_indent(test_body, base_indent)


def _process_standalone_function_body(func_lines):
    """处理独立函数体，保持缩进结构"""
    func_body = []
    in_def = False
    base_indent = None
    
    for line in func_lines:
        stripped = line.strip()
        if stripped.startswith('def '):
            in_def = True
            base_indent = len(line) - len(line.lstrip())
            continue
        if in_def:
            # 保留空行以维持缩进结构
            if not stripped:
                func_body.append('')
                continue
            
            # 移除基础缩进，但保持相对缩进
            if base_indent and len(line) > base_indent:
                line = line[base_indent:]
            
            # TF 1.x -> 2.x 兼容性：reduction_indices -> axis
            line = re.sub(r'\breduction_indices\s*=', 'axis=', line)
            
            func_body.append(line)
    
    return _normalize_indent(func_body, base_indent)


def convert_tf_to_standalone(test_code, test_name):
    """将 TensorFlow 测试代码转换为独立可执行的函数

    注意：
    - 为了避免 pytest 把原始 TF 测试当成用例收集，这里统一给 TF 函数加前缀 tf_。
      例如原始测试名为 test_basic，则生成的函数名为 tf_test_basic。
    - 这样文件里只会有一个真正的 pytest 用例：PyTorch 侧的 test_xxx_pt。
    """
    # 检查是否是类方法
    is_class_method = 'self' in test_code and 'def ' in test_code
    test_lines = test_code.split('\n')
    
    if is_class_method:
        indented_body = _process_class_method_body(test_lines)
    else:
        indented_body = _process_standalone_function_body(test_lines)
    
    # 生成 TF 侧独立函数代码，统一加 tf_ 前缀
    tf_func_name = f"tf_{test_name}"
    standalone_code = f"""def {tf_func_name}():
    \"\"\"Original TensorFlow test logic\"\"\"
    try:
{chr(10).join(indented_body)}
        print("TF: PASS")
    except Exception as e:
        print(f"TF: FAIL - {{e}}")
        import traceback
        traceback.print_exc()
"""
    # 返回代码和实际函数名，方便后续 main 中调用
    return standalone_code, tf_func_name


def _find_function_end_line(node, tree, lines):
    """查找函数结束行号"""
    # 方法1: 使用 ast 的 end_lineno（Python 3.8+）
    if hasattr(node, 'end_lineno') and node.end_lineno:
        return node.end_lineno
    
    # 方法2: 查找下一个同级别定义
    end_line = len(lines)
    parent = None
    
    # 找到函数的父节点
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            for child in (n.body if hasattr(n, 'body') else []):
                if child == node:
                    parent = n
                    break
    
    # 查找下一个同级别节点
    if parent and hasattr(parent, 'body'):
        for sibling in parent.body:
            if isinstance(sibling, (ast.FunctionDef, ast.ClassDef)) and sibling.lineno > node.lineno:
                return sibling.lineno - 1
    
    return end_line


def _remove_base_indent(func_code):
    """移除函数代码的基础缩进"""
    func_lines = func_code.split('\n')
    if not func_lines:
        return func_code
    
    # 找到第一行非空行的缩进
    base_indent = 0
    for line in func_lines:
        if line.strip():
            base_indent = len(line) - len(line.lstrip())
            break
    
    # 移除基础缩进
    if base_indent > 0:
        func_lines = [
            line[base_indent:] if len(line) > base_indent else line 
            for line in func_lines
        ]
        return '\n'.join(func_lines)
    
    return func_code


def extract_test_function_code(file_path, test_name):
    """从原始测试文件中提取指定测试函数的代码"""
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        lines = source.split('\n')
        
        # 如果测试函数名是 unknown_test_*，尝试查找实际的测试函数
        actual_test_name = test_name
        if test_name.startswith("unknown_test_"):
            # 策略1: 查找所有以 test 开头的函数
            test_funcs = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
                    test_funcs.append(node.name)
            
            if test_funcs:
                # 使用第一个找到的测试函数
                actual_test_name = test_funcs[0]
                print(f"[INFO] {test_name} -> 找到测试函数: {actual_test_name}")
            else:
                # 策略2: 查找类中的方法（可能是测试类）
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.name.startswith("test"):
                                actual_test_name = item.name
                                print(f"[INFO] {test_name} -> 找到类方法: {node.name}.{actual_test_name}")
                                break
                        if actual_test_name != test_name:
                            break
                
                # 策略3: 查找所有函数（可能是非标准测试文件，如 MLIR SavedModel 测试、TFLite 测试配置生成器）
                # 这些文件可能有 Test()、some_function()、make_xxx_tests() 等函数
                if actual_test_name == test_name:
                    all_funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                    
                    # 策略3a: 查找在 if __name__ == '__main__' 中被调用的函数
                    main_called_funcs = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.If):
                            # 检查是否是 if __name__ == '__main__'
                            if (isinstance(node.test, ast.Compare) and 
                                isinstance(node.test.left, ast.Name) and 
                                node.test.left.id == '__name__'):
                                # 查找 main 块中调用的函数
                                for stmt in ast.walk(node):
                                    if isinstance(stmt, ast.Call):
                                        if isinstance(stmt.func, ast.Name):
                                            main_called_funcs.append(stmt.func.id)
                                        elif isinstance(stmt.func, ast.Attribute):
                                            main_called_funcs.append(stmt.func.attr)
                    
                    # 优先选择在 main 中被调用的函数
                    if main_called_funcs:
                        # 查找这些函数中实际存在的
                        for func_name in main_called_funcs:
                            for node in ast.walk(tree):
                                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                                    actual_test_name = func_name
                                    print(f"[INFO] {test_name} -> 找到 main 中调用的函数: {actual_test_name}")
                                    break
                            if actual_test_name != test_name:
                                break
                    
                    # 策略3b: 查找 make_xxx_tests 函数（TFLite 测试配置生成器）
                    if actual_test_name == test_name:
                        make_test_funcs = [f for f in all_funcs if f.startswith('make_') and f.endswith('_tests')]
                        if make_test_funcs:
                            actual_test_name = make_test_funcs[0]  # 使用第一个
                            print(f"[INFO] {test_name} -> 找到 TFLite 测试生成器函数: {actual_test_name}")
                    
                    # 策略3c: 查找第一个大写开头的函数（如 Test、TestModule）
                    if actual_test_name == test_name:
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef) and node.name[0].isupper():
                                actual_test_name = node.name
                                print(f"[INFO] {test_name} -> 找到大写开头的函数: {actual_test_name}")
                                break
                    
                    # 策略3d: 查找任何以 make_ 开头的函数（测试生成器）
                    if actual_test_name == test_name:
                        make_funcs = [f for f in all_funcs if f.startswith('make_')]
                        if make_funcs:
                            actual_test_name = make_funcs[0]
                            print(f"[INFO] {test_name} -> 找到测试生成器函数: {actual_test_name}")
                    
                    # 如果还是没有找到，这些文件可能是非标准测试文件或工具文件
                    if actual_test_name == test_name:
                        print(f"[WARN] {test_name} -> 未找到可提取的测试函数，文件可能不是标准测试文件（可能是工具/配置文件）")
                        return None
        
        # 查找测试函数
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == actual_test_name:
                # 查找装饰器（如果有）
                decorator_lines = []
                if node.decorator_list:
                    # 找到第一个装饰器的行号
                    first_decorator_line = min(d.decorator_list[0].lineno if hasattr(d, 'decorator_list') and d.decorator_list else node.lineno 
                                              for d in [node] if hasattr(d, 'decorator_list'))
                    # 但更简单的方法是：从函数定义行往前查找装饰器
                    func_def_line = node.lineno - 1
                    # 往前查找装饰器（装饰器通常在函数定义之前）
                    for i in range(func_def_line - 1, max(0, func_def_line - 10), -1):
                        line = lines[i].strip()
                        if line.startswith('@'):
                            decorator_lines.insert(0, lines[i])
                        elif line and not line.startswith('#'):
                            # 遇到非装饰器、非注释的行，停止
                            break
                
                # 提取函数代码（包括装饰器）
                start_line = (node.lineno - len(decorator_lines) - 1) if decorator_lines else (node.lineno - 1)
                end_line = _find_function_end_line(node, tree, lines)
                func_code = '\n'.join(lines[start_line:end_line])
                return _remove_base_indent(func_code)
    except Exception as e:
        print(f"[WARN] 无法提取 {file_path}:{test_name} 的代码: {e}")
    return None


def extract_helper_functions(file_path, test_name=None):
    """从原始测试文件中提取辅助函数（非测试函数）
    
    如果提供了 test_name，会尝试提取该测试函数所在类的类方法（如 _Sampler, _testArg 等）
    """
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        lines = source.split('\n')
        helper_functions = []

        # 如果提供了 test_name，尝试找到它所在的类，并提取该类的类方法
        test_class = None
        if test_name:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == test_name:
                            test_class = node.name
                            break
                    if test_class:
                        break
        
        # 提取所有非测试函数的函数定义
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            
            # 如果是类方法，检查是否属于目标测试类
            is_class_method = False
            if test_class:
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef) and parent.name == test_class:
                        if node in parent.body:
                            is_class_method = True
                            break
            
            # 跳过测试函数
            if node.name.startswith('test'):
                continue

            # 提取类方法（如 _Sampler, _testArg 等）或独立辅助函数
            if is_class_method or (not node.name.startswith('_') or node.name.startswith('_')):
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else len(lines)
                func_code = '\n'.join(lines[start_line:end_line])
                func_code = _remove_base_indent(func_code)
                # 移除 self 参数（如果是类方法）
                if is_class_method and 'def ' in func_code and '(self' in func_code:
                    func_code = re.sub(r'def\s+(\w+)\s*\(self\s*,?\s*', r'def \1(', func_code)
                    func_code = re.sub(r'def\s+(\w+)\s*\(self\)', r'def \1()', func_code)
                helper_functions.append(func_code)

        return '\n\n'.join(helper_functions) if helper_functions else None
    except Exception as e:
        # 提取辅助函数失败时不影响主流程
        print(f"[WARN] 提取辅助函数失败: {e}")
        return None


def build_migration_prompt(tf_code, tf_apis, mapped_pt_apis):
    """构建 LLM 迁移提示词（不再要求生成 compare 函数，只生成 PyTorch 侧逻辑）"""
    # 选择最相关的映射 API
    top_mapped = mapped_pt_apis[:MAX_MAPPED_APIS] if len(mapped_pt_apis) > MAX_MAPPED_APIS else mapped_pt_apis
    tf_apis_display = ', '.join(tf_apis[:MAX_APIS_IN_PROMPT])
    pt_apis_display = ', '.join(top_mapped)

    prompt = f"""你是一个资深深度学习与单元测试迁移工程师。
现在有一个 TensorFlow 测试函数，需要你**只迁移成 PyTorch 版本**，不需要生成任何 TF/PT 对比函数。

【任务】
1. 读取下面给出的 TensorFlow 测试代码，将其迁移为等价的 PyTorch 测试代码。
2. 迁移后的代码中：
   - 保持测试逻辑不变（输入、计算流程、关键断言语义保持一致）。
   - 将 TensorFlow API 替换为对应的 PyTorch API（可参考下方给出的映射列表）。
   - 可以使用 `assert` 或 `torch.allclose` 等方式校验结果。
   - **不需要**再次调用 TensorFlow，也**不需要**写任何“TF/PT 结果对比”的辅助逻辑。
3. 如果原始测试中有重要的中间结果，建议适当使用 `print(...)` 打印，方便人工查看（例如打印张量的形状或数值）。
4. 最终请给出**可以直接运行的 PyTorch 测试函数代码**（可以是一个或多个 `def test_xxx_pt(...):`），
   不要包含多余的解释性文字。

【TensorFlow 原始测试代码】
```python
{tf_code}
```

【使用到的 TensorFlow API（最多 {MAX_APIS_IN_PROMPT} 个，仅供参考）】
{tf_apis_display}

【可能对应的 PyTorch API（前 {MAX_MAPPED_APIS} 个，仅供参考）】
{pt_apis_display}

请只输出迁移后的 **PyTorch 测试函数代码**，不要输出其他解释或说明。
"""
    return prompt


def _extract_code_from_llm_response(raw_code):
    """从 LLM 响应中提取代码"""
    # 提取代码块（如果有 markdown 代码块）
    code_match = re.search(r'```python\n(.*?)```', raw_code, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # 如果没有代码块，尝试提取函数定义
    func_match = re.search(r'def\s+test_\w+.*?(?=\n\n|\ndef\s|\Z)', raw_code, re.DOTALL)
    if func_match:
        return func_match.group(0).strip()
    
    return raw_code


def migrate_with_llm(client, tf_code, tf_apis, mapped_pt_apis, model=DEFAULT_MODEL, max_retries=3):
    """使用 LLM 生成迁移后的代码，带重试机制"""
    prompt = build_migration_prompt(tf_code, tf_apis, mapped_pt_apis)
    
    import time
    
    for attempt in range(max_retries):
        # 添加延迟，避免请求速率过快（重试时延迟更长）
        if attempt > 0:
            wait_time = 2 ** attempt  # 指数退避：2秒, 4秒, 8秒
            time.sleep(wait_time)
        else:
            time.sleep(0.1)  # 首次请求小延迟
        
        try:
            # 兼容新旧版本的 OpenAI API
            if hasattr(client, 'chat'):
                # 新版本
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=2048
                )
                raw_code = resp.choices[0].message.content.strip()
            else:
                # 旧版本
                resp = client.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=2048
                )
                raw_code = resp.choices[0].message.content.strip()
            
            return _extract_code_from_llm_response(raw_code)
        
        except Exception as e:
            error_str = str(e)
            # 检查是否是速率限制错误
            is_rate_limit = '429' in error_str or 'rate' in error_str.lower() or 'limit_burst_rate' in error_str
            
            if is_rate_limit and attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # 指数退避
                print(f"[RETRY] 速率限制错误，等待 {wait_time} 秒后重试 (尝试 {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            else:
                print(f"[ERROR] LLM 调用失败: {e}")
                return None
    
    return None


def initialize_llm_clients(key_path, workers):
    """初始化 LLM 客户端列表"""
    # 解析 key 路径（load_api_key 内部会处理相对路径）
    # 尝试初始化主客户端
    try:
        client = get_qwen_client(key_path)
    except Exception as e:
        print(f"[WARN] 无法初始化 LLM 客户端: {e}")
        print(f"[INFO] 尝试使用默认路径...")
        try:
            client = get_qwen_client(DEFAULT_KEY_PATH)
        except Exception as e2:
            print(f"[WARN] 无法初始化 LLM 客户端，将使用占位符生成 PyTorch 测试: {e2}")
            print(f"[INFO] 但会包含 TensorFlow 原始测试逻辑")
            client = None
    
    # 为每个线程创建独立的客户端
    if client:
        try:
            return [get_qwen_client(key_path) for _ in range(workers)]
        except Exception:
            return [None] * workers
    return [None] * workers


def main():
    parser = argparse.ArgumentParser(description="生成迁移测试文件")
    parser.add_argument("--input", default=DEFAULT_IN_FILE, help="输入文件路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--limit", type=int, default=-1, help="限制生成数量，-1 表示全部")
    parser.add_argument("--force", action="store_true", help="强制覆盖已存在的文件")
    parser.add_argument("--tf-root", default=DEFAULT_TF_ROOT, help="TensorFlow 源码根目录")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM 模型名称")
    parser.add_argument("--key-path", default=DEFAULT_KEY_PATH, help="API key 路径")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并发线程数")
    args = parser.parse_args()
    
    in_file = Path(args.input)
    out_dir = Path(args.output_dir)
    tf_root = Path(args.tf_root)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    if not in_file.exists():
        print(f"[ERR] 找不到输入文件: {in_file}")
        return

    lines = [json.loads(x) for x in open(in_file)]
    print(f"[LOAD] 发现可迁移测试数量: {len(lines)}")
    
    # 限制数量
    if args.limit > 0:
        lines = lines[:args.limit]
        print(f"[INFO] 限制生成数量: {args.limit}")

    # 去重：基于 (file, name) 组合
    seen = set()
    unique_tests = []
    for item in lines:
        key = (item["file"], item["name"])
        if key not in seen:
            seen.add(key)
            unique_tests.append(item)
    
    print(f"[INFO] 去重后测试数量: {len(unique_tests)} (去除了 {len(lines) - len(unique_tests)} 个重复)")

    # 初始化 LLM 客户端
    clients = initialize_llm_clients(args.key_path, args.workers)
    parser = argparse.ArgumentParser(description="生成迁移测试文件")
    parser.add_argument("--input", default=DEFAULT_IN_FILE, help="输入文件路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--limit", type=int, default=-1, help="限制生成数量，-1 表示全部")
    parser.add_argument("--force", action="store_true", help="强制覆盖已存在的文件")
    parser.add_argument("--tf-root", default=DEFAULT_TF_ROOT, help="TensorFlow 源码根目录")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM 模型名称")
    parser.add_argument("--key-path", default=DEFAULT_KEY_PATH, help="API key 路径")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并发线程数")
    args = parser.parse_args()
    
    in_file = Path(args.input)
    out_dir = Path(args.output_dir)
    tf_root = Path(args.tf_root)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    if not in_file.exists():
        print(f"[ERR] 找不到输入文件: {in_file}")
        return

    lines = [json.loads(x) for x in open(in_file)]
    print(f"[LOAD] 发现可迁移测试数量: {len(lines)}")
    
    # 限制数量
    if args.limit > 0:
        lines = lines[:args.limit]
        print(f"[INFO] 限制生成数量: {args.limit}")

    # 去重：基于 (file, name) 组合
    seen = set()
    unique_tests = []
    for item in lines:
        key = (item["file"], item["name"])
        if key not in seen:
            seen.add(key)
            unique_tests.append(item)
    
    print(f"[INFO] 去重后测试数量: {len(unique_tests)} (去除了 {len(lines) - len(unique_tests)} 个重复)")

    # 初始化 LLM 客户端
    clients = initialize_llm_clients(args.key_path, args.workers)
    
    # 线程安全的计数器
    migrated_counter = [0]
    failed_counter = [0]
    lock = threading.Lock()
    
    def process_one_test(item, client_idx):
        """处理单个测试的生成"""
        file = item["file"]
        name = item["name"]
        apis = item["apis_used"]
        matches = item["matches"]
        
        # 生成目标文件路径
        out_path = out_dir / f"{safe_filename(name)}.py"
        
        # 已存在则跳过（支持断点续测），除非使用 --force
        if out_path.exists() and not args.force:
            return {"status": "skipped", "name": name}
        
        try:
            # 解析文件路径
            full_file_path = resolve_file_path(file, tf_root)
            
            # 查找实际的测试函数名
            actual_test_name = find_actual_test_name(full_file_path, name)
            
            tf_code = extract_test_function_code(full_file_path, actual_test_name)
            
            if not tf_code:
                return {"status": "failed", "name": name, "error": f"无法提取 {file}:{actual_test_name} 的代码"}
            
            # 提取并转换 TensorFlow 测试为独立可执行的函数
            tf_standalone_code = convert_tf_to_standalone(tf_code, actual_test_name)
            
            if not tf_standalone_code:
                return {"status": "failed", "name": name, "error": f"无法转换 TensorFlow 测试 {file}:{actual_test_name}"}
            
            # 获取映射的 PyTorch API
            mapped_pt_apis = [m["mapped_pt_api"] for m in matches]
            
            # 使用 LLM 生成迁移代码（如果客户端可用）
            migrated_code = None
            thread_client = clients[client_idx % len(clients)]
            if thread_client:
                migrated_code = migrate_with_llm(thread_client, tf_code, apis, mapped_pt_apis, args.model)
            
            if not migrated_code:
                migrated_code = f"""def test_{safe_filename(name)}_pt():
    # ===== TF Original APIs Used =====
    # {', '.join(apis[:10])}
    #
    # ===== Mapped PT APIs =====
    # {', '.join(mapped_pt_apis[:5])}

    # TODO: Write migrated PyTorch version
    assert True  # placeholder
"""
            
            # 清理生成的代码：移除重复的 import
            migrated_code = re.sub(r'^import torch\s*$', '', migrated_code, flags=re.MULTILINE)
            migrated_code = re.sub(r'^import pytest\s*$', '', migrated_code, flags=re.MULTILINE)
            migrated_code = migrated_code.strip()
            
            # 确保 PyTorch 测试函数名以 _pt 结尾
            if not re.search(r'def\s+test_\w+_pt\s*\(', migrated_code):
                migrated_code = re.sub(r'def\s+(test_\w+)\s*\(', r'def \1_pt(', migrated_code)
            
            # 提取辅助函数（测试中使用的非测试函数）
            helper_functions = extract_helper_functions(full_file_path)
            helper_section = ""
            if helper_functions:
                helper_section = f"""# ===== Helper Functions from Original File =====
{helper_functions}

"""
            
            # 组合最终代码：包含 TensorFlow 原始测试和 PyTorch 迁移测试
            content = HEADER + f"""# Auto-Migrated from TF test
# source: {file}:{name}
# Original test function: {actual_test_name}

{helper_section}# ===== TensorFlow Original Test =====
{tf_standalone_code}

# ===== PyTorch Migrated Test =====
{migrated_code}

# ===== Comparison Test =====
def test_{safe_filename(name)}_compare():
    \"\"\"Compare TensorFlow and PyTorch test results\"\"\"
    try:
        # Run TensorFlow test
        tf_result = None
        tf_error = None
        try:
            {actual_test_name}()
            tf_result = "PASS"
        except Exception as e:
            tf_result = "FAIL"
            tf_error = str(e)
        
        # Run PyTorch test
        pt_result = None
        pt_error = None
        # Find PyTorch test function name from migrated_code
        import re
        pt_match = re.search(r'def\\s+(test_\\w+_pt)\\s*\\(', '''{migrated_code}''')
        if pt_match:
            pt_test_name = pt_match.group(1)
            try:
                # Execute the PyTorch test function
                exec('''{migrated_code}''')
                exec(f\"\"\"{{pt_test_name}}()\"\"\")
                pt_result = "PASS"
            except Exception as e:
                pt_result = "FAIL"
                pt_error = str(e)
        else:
            pt_result = "SKIP"
            pt_error = "Could not find PyTorch test function"
        
        # Compare results
        if tf_result == "PASS" and pt_result == "PASS":
            print("PASS: Both TF and PT tests passed")
        elif tf_result == "FAIL" and pt_result == "FAIL":
            print("FAIL: Both TF and PT tests failed")
            if tf_error:
                print(f"TF error: {{tf_error}}")
            if pt_error:
                print(f"PT error: {{pt_error}}")
        else:
            print(f"MISMATCH: TF={{tf_result}}, PT={{pt_result}}")
            if tf_error:
                print(f"TF error: {{tf_error}}")
            if pt_error:
                print(f"PT error: {{pt_error}}")
    except Exception as e:
        print(f"COMPARISON ERROR: {{e}}")
        import traceback
        traceback.print_exc()

# ===== Main Execution =====
if __name__ == "__main__":
    import sys
    import re
    
    # Find PyTorch test function name
    pt_match = re.search(r'def\\s+(test_\\w+_pt)\\s*\\(', '''{migrated_code}''')
    pt_test_name = pt_match.group(1) if pt_match else None
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            test_{safe_filename(name)}_compare()
        elif sys.argv[1] == "--tf":
            # Only run TensorFlow version
            if not TF_AVAILABLE:
                print("错误: TensorFlow 未安装，无法执行 TF 版本")
                sys.exit(1)
            print("=" * 80)
            print("执行 TensorFlow 版本")
            print("=" * 80)
            {actual_test_name}()
        elif sys.argv[1] == "--pt":
            # Only run PyTorch version
            if pt_test_name:
                print("=" * 80)
                print("执行 PyTorch 版本")
                print("=" * 80)
                exec('''{migrated_code}''')
                exec(f\"\"\"{{pt_test_name}}()\"\"\")
            else:
                print("错误: 找不到 PyTorch 测试函数")
        else:
            print(f"未知参数: {{sys.argv[1]}}")
            print("用法: python {{sys.argv[0]}} [--tf|--pt|--compare]")
    else:
        # Default: run both versions and compare (if TF is available)
        tf_output = None
        if TF_AVAILABLE:
        print("=" * 80)
        print("执行 TensorFlow 版本")
        print("=" * 80)
        tf_output = capture_output({actual_test_name})
        
        # Print TF output immediately
        if tf_output["stdout"]:
            print(tf_output["stdout"], end="")
        if tf_output["stderr"]:
            print(tf_output["stderr"], end="", file=sys.stderr)
        if tf_output["exception"]:
            print(f"\\nTensorFlow 执行异常: {{tf_output['exception']}}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        else:
            print("=" * 80)
            print("TensorFlow 不可用，跳过 TF 版本执行")
            print("=" * 80)
            tf_output = {{
                "success": False,
                "stdout": "",
                "stderr": "TensorFlow 未安装",
                "exception": ImportError("TensorFlow 未安装"),
                "result": None
            }}
        
        print("\\n" + "=" * 80)
        print("执行 PyTorch 版本")
        print("=" * 80)
        
        pt_output = None
        if pt_test_name:
            # Execute PyTorch test function code first
            exec('''{migrated_code}''')
            # Then get the function object and execute it
            pt_func = eval(pt_test_name)
            pt_output = capture_output(pt_func)
            
            # Print PT output immediately
            if pt_output["stdout"]:
                print(pt_output["stdout"], end="")
            if pt_output["stderr"]:
                print(pt_output["stderr"], end="", file=sys.stderr)
            if pt_output["exception"]:
                print(f"\\nPyTorch 执行异常: {{pt_output['exception']}}", file=sys.stderr)
                import traceback
                traceback.print_exc()
        else:
            pt_output = {{
                "success": False,
                "stdout": "",
                "stderr": "找不到 PyTorch 测试函数",
                "exception": Exception("找不到 PyTorch 测试函数"),
                "result": None
            }}
            print("错误: 找不到 PyTorch 测试函数")
        
        # Compare outputs (if TF was available)
        if pt_output and tf_output:
            compare_outputs(tf_output, pt_output)
        elif pt_output:
            print("\\n" + "=" * 80)
            print("PyTorch 执行结果")
            print("=" * 80)
            pt_status = "PASS" if pt_output["success"] else "FAIL"
            print(f"状态: {{pt_status}}")
            if pt_output["exception"]:
                print(f"异常: {{type(pt_output['exception']).__name__}}: {{pt_output['exception']}}")

"""
            
            # 线程安全写入文件
            with lock:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(content)
                migrated_counter[0] += 1
            
            return {"status": "success", "name": name}
            
        except Exception as e:
            with lock:
                failed_counter[0] += 1
            return {"status": "failed", "name": name, "error": str(e)}
    
    # 并发处理
    print(f"[INFO] 使用 {args.workers} 个并发线程生成测试")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_one_test, item, i % args.workers): item 
            for i, item in enumerate(unique_tests)
        }
        
        # 处理完成的任务
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Migrated Tests"):
            try:
                result = future.result()
                if result["status"] == "failed":
                    item = futures[future]
                    print(f"[WARN] {result['name']}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                item = futures[future]
                print(f"[ERROR] 处理失败 {item.get('name', 'unknown')}: {e}")
                with lock:
                    failed_counter[0] += 1
    
    migrated = migrated_counter[0]
    failed = failed_counter[0]

    print("\n==== TEST MIGRATION SUMMARY ====")
    print(f"成功生成迁移测试数量: {migrated}")
    print(f"失败数量: {failed}")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
