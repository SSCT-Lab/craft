import json
from pathlib import Path
from typing import Dict, Tuple, List

def read_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def analyze_records(records: List[Dict]) -> Tuple[int, int, int]:
    completed = [r for r in records if r.get("status") == "completed"]
    total_completed = len(completed)
    llm_success = sum(1 for r in completed if r.get("llm_tf_success") is True)
    onnx_success = sum(1 for r in completed if r.get("onnx_run_success") is True)
    return total_completed, llm_success, onnx_success

def rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100) if denominator > 0 else 0.0

def print_summary(name: str, total: int, llm_ok: int, onnx_ok: int):
    llm_rate = rate(llm_ok, total)
    onnx_rate = rate(onnx_ok, total)

    print(f"\n=== {name} ===")
    print(f"completed 用例数: {total}")
    print(f"LLM 执行成功: {llm_ok} ({llm_rate:.2f}%)")
    print(f"ONNX 执行成功: {onnx_ok} ({onnx_rate:.2f}%)")

    if total == 0:
        print("对比结论：无 completed 用例，无法比较")
    else:
        diff = llm_rate - onnx_rate
        if diff > 0:
            print(f"对比结论：LLM 优于 ONNX {diff:.2f} 个百分点")
        elif diff < 0:
            print(f"对比结论：ONNX 优于 LLM {-diff:.2f} 个百分点")
        else:
            print("对比结论：两者持平")

def main():
    # 修改为你的日志路径
    files = [
        r"D:\graduate\DFrameworkTest\pt_tf_test\llm_vs_onnx_realtime_20260206_180018.jsonl",
        r"D:\graduate\DFrameworkTest\pt_tf_test\llm_vs_onnx_realtime_20260206_181045.jsonl",
        r"D:\graduate\DFrameworkTest\pt_tf_test\llm_vs_onnx_realtime_20260206_182151.jsonl",
        r"D:\graduate\DFrameworkTest\pt_tf_test\llm_vs_onnx_realtime_20260206_215822.jsonl"
    ]

    all_records: List[Dict] = []
    for file_path in files:
        path = Path(file_path)
        all_records.extend(read_jsonl(path))

    total, llm_ok, onnx_ok = analyze_records(all_records)
    print_summary("ALL_FILES", total, llm_ok, onnx_ok)

if __name__ == "__main__":
    main()