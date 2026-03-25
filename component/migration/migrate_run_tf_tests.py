# ./component/migration/migrate_run_tf_tests.py
"""运行迁移测试文件中的 TensorFlow 版本（在 tf2pt-dev-tf 环境中）"""
import subprocess
import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm

DEFAULT_MIGRATED_DIR = Path("migrated_tests")
DEFAULT_OUT_FILE = Path("dev/results/tf_exec_dev.jsonl")
DEFAULT_LOG_DIR = Path("dev/logs")
DEFAULT_CONDA_ENV = "tf2pt-dev-tf"


def find_tf_function_name(file_path):
    """从测试文件中找到 TF 函数名（格式：tf_xxx 或 tf_testXxx）"""
    try:
        content = Path(file_path).read_text(encoding='utf-8')
        # 查找 def tf_xxx(): 或 def tf_testXxx(): 格式的函数
        match = re.search(r'def\s+(tf_\w+)\s*\(', content)
        if match:
            return match.group(1)
        # 如果没有找到，尝试查找 main 函数（TF core 文件可能有 main）
        if 'def main()' in content:
            # 对于有 main 函数的文件，直接运行 main
            return 'main'
    except Exception as e:
        pass
    return None


def get_conda_python(conda_env):
    """获取 conda 环境的 Python 可执行文件路径"""
    import os
    conda_base = os.environ.get('CONDA_PREFIX', '/opt/homebrew/Caskroom/miniconda/base')
    env_path = f"{conda_base}/envs/{conda_env}/bin/python"
    if Path(env_path).exists():
        return env_path
    # 备用方案：尝试从 conda info 获取
    try:
        result = subprocess.run(
            ["conda", "info", "--envs"],
            capture_output=True,
            text=True,
            timeout=5
        )
        for line in result.stdout.split('\n'):
            if conda_env in line and '*' not in line:
                # 解析路径
                parts = line.split()
                if parts:
                    env_dir = parts[-1]
                    python_path = Path(env_dir) / "bin" / "python"
                    if python_path.exists():
                        return str(python_path)
    except:
        pass
    return None


def run_tf_test(file_path, conda_env=DEFAULT_CONDA_ENV):
    """在指定 Conda 环境中运行 TF 测试"""
    try:
        # 获取 conda 环境的 Python 路径
        python_exe = get_conda_python(conda_env)
        if not python_exe:
            return {
                "status": "error",
                "stdout": "",
                "stderr": f"无法找到 conda 环境 {conda_env} 的 Python 可执行文件",
                "returncode": -1
            }
        
        # 确保文件路径是绝对路径
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = file_path.absolute()
        
        # 检查文件是否存在
        if not file_path.exists():
            return {
                "status": "error",
                "stdout": "",
                "stderr": f"文件不存在: {file_path}",
                "returncode": -1
            }
        
        # 检查文件是否有 main 函数或可以直接运行
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        has_main = 'if __name__' in content or 'def main()' in content
        
        # 如果有 main 函数或 __name__ == '__main__'，直接运行文件
        # 否则尝试查找 TF 函数并调用
        if has_main:
            # 直接运行文件（TF core 文件通常有 main 函数）
            # 使用绝对路径，避免路径重复
            result = subprocess.run(
                [python_exe, str(file_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                text=True,
                cwd=str(file_path.parent)
            )
            
            # 判断是否通过（必须同时满足：有 "TF: PASS" 且没有 "TF: FAIL"，或者 returncode == 0 且没有 "TF: FAIL"）
            has_pass = "TF: PASS" in result.stdout
            has_fail = "TF: FAIL" in result.stdout
            if has_pass and not has_fail:
                status = "pass"
            elif result.returncode == 0 and not has_fail:
                status = "pass"
            else:
                status = "fail"
            
            return {
                "status": status,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        else:
            # 尝试查找并调用 TF 函数
            tf_func_name = find_tf_function_name(file_path)
            if not tf_func_name:
                return {
                    "status": "skip",
                    "stdout": "",
                    "stderr": f"未找到 TF 测试函数或 main 函数",
                    "returncode": -1
                }
            
            # 创建临时脚本调用函数
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                module_path = str(file_path)  # file_path 已经是绝对路径
                module_name = file_path.stem
                
                exec_script = f"""import sys
import importlib.util

# 临时禁用 torch 导入（TF 环境中不需要）
class TorchMock:
    pass
sys.modules['torch'] = TorchMock()

try:
    spec = importlib.util.spec_from_file_location("{module_name}", "{module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["{module_name}"] = module
    spec.loader.exec_module(module)
    
    tf_func = getattr(module, "{tf_func_name}")
    tf_func()
    print("\\nTF_TEST_PASSED")
except Exception as e:
    print(f"\\nTF_TEST_FAILED: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
                tmp_file.write(exec_script)
                tmp_path = tmp_file.name
            
            try:
                result = subprocess.run(
                    [python_exe, tmp_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                    text=True,
                    cwd=str(file_path.parent)  # file_path 已经是绝对路径
                )
                
                # 判断是否通过（必须同时满足：有 "TF_TEST_PASSED" 且没有 "TF_TEST_FAILED"，或者 returncode == 0 且没有 "TF_TEST_FAILED"）
                has_passed = "TF_TEST_PASSED" in result.stdout
                has_failed = "TF_TEST_FAILED" in result.stdout
                if has_passed and not has_failed:
                    status = "pass"
                elif result.returncode == 0 and not has_failed:
                    status = "pass"
                else:
                    status = "fail"
                
                return {
                    "status": status,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
    except subprocess.TimeoutExpired as e:
        return {
            "status": "timeout",
            "stdout": "",
            "stderr": str(e),
            "returncode": -1
        }
    except Exception as e:
        return {
            "status": "error",
            "stdout": "",
            "stderr": str(e),
            "returncode": -2
        }


def main():
    parser = argparse.ArgumentParser(description="运行迁移测试文件中的 TensorFlow 版本")
    parser.add_argument("--migrated-dir", default=str(DEFAULT_MIGRATED_DIR), help="迁移测试目录")
    parser.add_argument("--out", default=str(DEFAULT_OUT_FILE), help="输出结果文件")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="日志目录")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV, help="Conda 环境名称")
    parser.add_argument("--limit", type=int, default=-1, help="限制测试数量，-1 表示全部")
    args = parser.parse_args()
    
    migrated_dir = Path(args.migrated_dir)
    out_file = Path(args.out)
    log_dir = Path(args.log_dir)
    
    Path(out_file.parent).mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    files = sorted(migrated_dir.glob("*.py"))
    if args.limit > 0:
        files = files[:args.limit]
    
    print(f"[RUN] found {len(files)} migrated tests")
    print(f"[ENV] using conda environment: {args.conda_env}")

    fout = open(out_file, "w")

    for f in tqdm(files, desc="Running TF tests"):
        result = run_tf_test(f, args.conda_env)

        # 保存单个文件的详细日志
        log_path = log_dir / f"{f.stem}_tf.log"
        with open(log_path, "w") as lf:
            lf.write("=== STDOUT ===\n")
            lf.write(result["stdout"])
            lf.write("\n\n=== STDERR ===\n")
            lf.write(result["stderr"])

        # 简要记录 JSON
        rec = {
            "tf_file": str(f),
            "tf_result": {
                "status": result["status"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "returncode": result["returncode"]
            }
        }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

    fout.close()

    print(f"[DONE] execution results saved to {out_file}")
    print(f"[LOG] detailed logs stored in {log_dir}")


if __name__ == "__main__":
    main()

