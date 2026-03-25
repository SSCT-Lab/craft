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
    llm_success = sum(1 for r in completed if r.get("llm_pd_success") is True)
    x2paddle_success = sum(1 for r in completed if r.get("x2paddle_run_success") is True)
    return total_completed, llm_success, x2paddle_success


def rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100) if denominator > 0 else 0.0


def print_summary(name: str, total: int, llm_ok: int, x2p_ok: int):
    llm_rate = rate(llm_ok, total)
    x2p_rate = rate(x2p_ok, total)

    print(f"\n=== {name} ===")
    print(f"completed 用例数: {total}")
    print(f"LLM 执行成功: {llm_ok} ({llm_rate:.2f}%)")
    print(f"X2Paddle 执行成功: {x2p_ok} ({x2p_rate:.2f}%)")

    if total == 0:
        print("对比结论：无 completed 用例，无法比较")
    else:
        diff = llm_rate - x2p_rate
        if diff > 0:
            print(f"对比结论：LLM 优于 X2Paddle {diff:.2f} 个百分点")
        elif diff < 0:
            print(f"对比结论：X2Paddle 优于 LLM {-diff:.2f} 个百分点")
        else:
            print("对比结论：两者持平")


def main():
    # 修改为你的四个日志路径
    files = [
        r"D:\graduate\DFrameworkTest\pt_pd_test\llm_vs_x2paddle_realtime_20260206_200933.jsonl",
        r"D:\graduate\DFrameworkTest\pt_pd_test\llm_vs_x2paddle_realtime_20260206_202914.jsonl",
        r"D:\graduate\DFrameworkTest\pt_pd_test\llm_vs_x2paddle_realtime_20260206_204447.jsonl",
        r"D:\graduate\DFrameworkTest\pt_pd_test\llm_vs_x2paddle_realtime_20260206_205130.jsonl",
        r"D:\graduate\DFrameworkTest\pt_pd_test\llm_vs_x2paddle_realtime_20260206_223503.jsonl",
        r"D:\graduate\DFrameworkTest\pt_pd_test\llm_vs_x2paddle_realtime_20260206_224210.jsonl",
    ]

    all_records: List[Dict] = []
    for file_path in files:
        path = Path(file_path)
        all_records.extend(read_jsonl(path))

    total, llm_ok, x2p_ok = analyze_records(all_records)
    print_summary("ALL_FILES", total, llm_ok, x2p_ok)


if __name__ == "__main__":
    main()
