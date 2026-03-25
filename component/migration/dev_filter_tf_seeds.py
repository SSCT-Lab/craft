"""运行 TF core 测试并筛选合规的 seed

从 dev/tf_core/ 目录运行所有 TF core 测试，筛选出能正常运行的作为 seed。
"""
import json
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DEFAULT_TF_CORE_DIR = Path("dev/tf_core")
DEFAULT_SEEDS_OUT = Path("dev/tf_seeds.jsonl")
DEFAULT_CONDA_ENV = "tf2pt-dev-tf"


def get_conda_python(conda_env):
    """获取 conda 环境的 Python 可执行文件路径"""
    import os
    
    # 方法1: 从 CONDA_PREFIX 推断（如果当前在某个 conda 环境中）
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        # 如果当前环境就是目标环境
        if Path(conda_prefix).name == conda_env:
            python_path = Path(conda_prefix) / "bin" / "python"
            if python_path.exists():
                return str(python_path)
        # 否则从 base 目录查找
        conda_base = Path(conda_prefix).parent.parent
        env_path = conda_base / "envs" / conda_env / "bin" / "python"
        if env_path.exists():
            return str(env_path)
    
    # 方法2: 尝试常见的 conda 安装路径
    common_bases = [
        Path("/opt/homebrew/Caskroom/miniconda/base"),
        Path("/opt/homebrew/anaconda3"),
        Path("/usr/local/anaconda3"),
        Path(os.path.expanduser("~/anaconda3")),
        Path(os.path.expanduser("~/miniconda3")),
        Path(os.path.expanduser("~/miniconda")),
    ]
    
    for conda_base in common_bases:
        if conda_base.exists():
            env_path = conda_base / "envs" / conda_env / "bin" / "python"
            if env_path.exists():
                return str(env_path)
    
    # 方法3: 从 CONDA_EXE 推断
    conda_exe = os.environ.get('CONDA_EXE')
    if conda_exe:
        conda_base = Path(conda_exe).parent.parent
        env_path = conda_base / "envs" / conda_env / "bin" / "python"
        if env_path.exists():
            return str(env_path)
    
    # 方法4: 尝试使用 conda run（如果可用）
    try:
        # 先检查 conda 命令是否存在
        result = subprocess.run(
            ["which", "conda"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            # 使用 conda run 的方式（返回 None，让调用者使用 conda run）
            return None
    except:
        pass
    
    return None


def run_tf_core_test_with_python(core_file: Path, python_exe: str):
    """使用指定的 Python 可执行文件运行 TF core 测试"""
    try:
        result = subprocess.run(
            [python_exe, str(core_file.absolute())],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            text=True,
            cwd=str(core_file.parent.absolute())
        )
        
        # 判断是否通过
        has_pass = "TF: PASS" in result.stdout
        has_fail = "TF: FAIL" in result.stdout
        has_import_error = "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr or "No module named 'tensorflow'" in result.stderr
        
        if has_import_error:
            status = "error"
            error_msg = "TensorFlow 导入失败（可能使用了错误的 Python 环境）"
        elif has_pass and not has_fail:
            status = "pass"
            error_msg = None
        elif result.returncode == 0 and not has_fail:
            status = "pass"
            error_msg = None
        else:
            status = "fail"
            error_msg = result.stderr[:200] if result.stderr else (result.stdout[:200] if result.stdout else "未知错误")
        
        return {
            "status": status,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "error": error_msg
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": "执行超时",
            "stdout": "",
            "stderr": "",
            "returncode": -1
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "stdout": "",
            "stderr": "",
            "returncode": -2
        }


def run_tf_core_test(core_file: Path, conda_env: str):
    """在指定 Conda 环境中运行 TF core 测试"""
    python_exe = get_conda_python(conda_env)
    
    try:
        if python_exe:
            # 直接使用 Python 路径
            cmd = [python_exe, str(core_file.absolute())]
        else:
            # 使用 conda run
            cmd = ["conda", "run", "-n", conda_env, "python", str(core_file.absolute())]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            text=True,
            cwd=str(core_file.parent.absolute())
        )
        
        # 判断是否通过（与 migrate_run_tf_tests.py 相同的逻辑）
        # 优先级：TF: FAIL > TF: PASS > returncode
        has_pass = "TF: PASS" in result.stdout
        has_fail = "TF: FAIL" in result.stdout
        has_import_error = "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr or "No module named 'tensorflow'" in result.stderr
        
        if has_import_error:
            status = "error"
            error_msg = "TensorFlow 导入失败（可能使用了错误的 Python 环境）"
        elif has_fail:
            # 如果输出中有 "TF: FAIL"，无论 returncode 是什么，都判定为 fail
            status = "fail"
            error_msg = result.stderr[:200] if result.stderr else (result.stdout[:200] if result.stdout else "未知错误")
        elif has_pass and not has_fail:
            status = "pass"
            error_msg = None
        elif result.returncode == 0 and not has_fail:
            status = "pass"
            error_msg = None
        else:
            status = "fail"
            error_msg = result.stderr[:200] if result.stderr else (result.stdout[:200] if result.stdout else "未知错误")
        
        return {
            "status": status,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "error": error_msg
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": "执行超时",
            "stdout": "",
            "stderr": "",
            "returncode": -1
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "stdout": "",
            "stderr": "",
            "returncode": -2
        }


def main():
    parser = argparse.ArgumentParser(description="运行 TF core 测试并筛选合规的 seed")
    parser.add_argument("--tf-core-dir", default=str(DEFAULT_TF_CORE_DIR), help="TF core 测试目录")
    parser.add_argument("--seeds-out", default=str(DEFAULT_SEEDS_OUT), help="输出 seed 列表文件")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV, help="Conda 环境名称")
    parser.add_argument("--python-path", default=None, help="手动指定 Python 可执行文件路径（可选）")
    parser.add_argument("--index-file", default="dev/tf_core/tf_core_index.jsonl", help="TF core 索引文件")
    parser.add_argument("--limit", type=int, default=-1, help="限制测试数量，-1 表示全部")
    args = parser.parse_args()
    
    tf_core_dir = Path(args.tf_core_dir)
    seeds_out = Path(args.seeds_out)
    index_file = Path(args.index_file)
    
    if not tf_core_dir.exists():
        print(f"[ERROR] TF core 目录不存在: {tf_core_dir}")
        return
    
    # 加载索引文件（如果有）
    index_map = {}
    if index_file.exists():
        for line in index_file.open():
            item = json.loads(line)
            if item.get("status") == "ok":
                core_file = item.get("core_file")
                if core_file:
                    index_map[Path(core_file).name] = item
        print(f"[LOAD] 从索引文件加载了 {len(index_map)} 条记录")
    
    # 获取所有 TF core 测试文件
    core_files = sorted(tf_core_dir.glob("tf_core_*.py"))
    
    if args.limit > 0:
        core_files = core_files[:args.limit]
    
    print(f"[RUN] 找到 {len(core_files)} 个 TF core 测试文件")
    print(f"[ENV] 使用 conda 环境: {args.conda_env}")
    
    seeds_out.parent.mkdir(exist_ok=True, parents=True)
    
    seeds = []
    failed = []
    
    # 如果指定了 Python 路径，使用它；否则尝试自动查找
    python_exe = args.python_path
    if not python_exe:
        python_exe = get_conda_python(args.conda_env)
    
    # 验证 Python 环境是否有 TensorFlow
    if python_exe:
        print(f"[CHECK] 使用 Python: {python_exe}")
        try:
            check_result = subprocess.run(
                [python_exe, "-c", "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if check_result.returncode == 0:
                print(f"[CHECK] ✓ TensorFlow 可用: {check_result.stdout.strip()}")
            else:
                print(f"[WARN] ⚠ TensorFlow 检查失败: {check_result.stderr[:200]}")
                print(f"[WARN] 建议使用 --python-path 指定正确的 Python 路径")
        except Exception as e:
            print(f"[WARN] ⚠ 无法验证 TensorFlow: {e}")
    else:
        print(f"[INFO] 将使用 conda run -n {args.conda_env} python")
    
    with open(seeds_out, "w") as fout:
        for core_file in tqdm(core_files, desc="Running TF core tests"):
            # 从索引中获取元信息
            meta = index_map.get(core_file.name, {})
            
            # 运行测试
            if python_exe:
                result = run_tf_core_test_with_python(core_file, python_exe)
            else:
                result = run_tf_core_test(core_file, args.conda_env)
            
            if result["status"] == "pass":
                # 这是一个合规的 seed
                seed_record = {
                    "core_file": str(core_file),
                    "core_name": core_file.stem,
                    "status": "pass",
                    "source_file": meta.get("file", ""),
                    "test_name": meta.get("test_name", ""),
                    "tf_file": meta.get("tf_file", ""),
                    "tf_func_name": meta.get("tf_func_name", ""),
                    "execution_result": {
                        "stdout": result.get("stdout", "")[:500],  # 只保存前500字符
                        "stderr": result.get("stderr", "")[:500],
                        "returncode": result.get("returncode", 0)
                    }
                }
                seeds.append(seed_record)
                fout.write(json.dumps(seed_record, ensure_ascii=False) + "\n")
                fout.flush()
            else:
                failed.append({
                    "core_file": str(core_file),
                    "status": result["status"],
                    "error": result.get("error", result.get("stderr", "")[:200]),
                    "stderr": result.get("stderr", "")[:200],
                    "stdout": result.get("stdout", "")[:200]
                })
    
    print(f"\n[SUMMARY]")
    print(f"  总测试数: {len(core_files)}")
    print(f"  合规 seed: {len(seeds)}")
    print(f"  失败: {len(failed)}")
    print(f"\n[OUTPUT] Seed 列表已保存到: {seeds_out}")
    
    if failed:
        print(f"\n[FAILED] 失败的测试（前5个）:")
        for f in failed[:5]:
            error_info = f.get('error', '') or f.get('stderr', '')[:100] or '未知错误'
            print(f"  {Path(f['core_file']).name}: {f['status']}")
            print(f"    错误: {error_info[:150]}")
            if f.get('stdout'):
                print(f"    stdout: {f['stdout'][:100]}")


if __name__ == "__main__":
    main()

