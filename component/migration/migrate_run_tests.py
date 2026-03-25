# ./component/migrate_run_tests.py
import subprocess
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import tempfile
import os

DEFAULT_MIGRATED_DIR = Path("migrated_tests")
DEFAULT_OUT_FILE = Path("data/results/migrate_exec.jsonl")
DEFAULT_LOG_DIR = Path("data/logs")


def run_pytest(file_path):
    """单独执行一个 PyTorch 测试文件，确保不会互相污染环境。"""
    try:
        # 建立独立进程执行 pytest
        result = subprocess.run(
            ["pytest", str(file_path), "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,  # 超时保护
            text=True
        )
        return {
            "status": "pass" if result.returncode == 0 else "fail",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
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
    parser = argparse.ArgumentParser(description="运行迁移的测试文件")
    parser.add_argument("--migrated-dir", default=str(DEFAULT_MIGRATED_DIR), help="迁移测试目录")
    parser.add_argument("--out", default=str(DEFAULT_OUT_FILE), help="输出结果文件")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="日志目录")
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

    fout = open(out_file, "w")

    for f in tqdm(files, desc="Running migrated tests"):
        result = run_pytest(f)

        # 保存单个文件的详细日志
        log_path = log_dir / f"{f.stem}.log"
        with open(log_path, "w") as lf:
            lf.write("=== STDOUT ===\n")
            lf.write(result["stdout"])
            lf.write("\n\n=== STDERR ===\n")
            lf.write(result["stderr"])

        # 简要记录 JSON
        rec = {
            "pt_file": str(f),
            "pt_result": {
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
