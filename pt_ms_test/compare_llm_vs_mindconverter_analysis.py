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
    llm_success = sum(1 for r in completed if r.get("llm_ms_success") is True)
    mindconverter_success = sum(1 for r in completed if r.get("mindconverter_run_success") is True)
    return total_completed, llm_success, mindconverter_success


def rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100) if denominator > 0 else 0.0


def print_summary(name: str, total: int, llm_ok: int, mc_ok: int):
    llm_rate = rate(llm_ok, total)
    mc_rate = rate(mc_ok, total)

    print(f"\n=== {name} ===")
    print(f"Completed cases: {total}")
    print(f"LLM execution success: {llm_ok} ({llm_rate:.2f}%)")
    print(f"MindConverter execution success: {mc_ok} ({mc_rate:.2f}%)")

    if total == 0:
        print("Comparison conclusion: no completed cases, unable to compare")
    else:
        diff = llm_rate - mc_rate
        if diff > 0:
            print(f"Comparison conclusion: LLM outperforms MindConverter by {diff:.2f} percentage points")
        elif diff < 0:
            print(f"Comparison conclusion: MindConverter outperforms LLM by {-diff:.2f} percentage points")
        else:
            print("Comparison conclusion: tie")


def main():
    # Update to your log paths
    files = [
        r"D:\graduate\DFrameworkTest\pt_ms_test\llm_vs_mindconverter_realtime_20260206_211957.jsonl",
        r"D:\graduate\DFrameworkTest\pt_ms_test\llm_vs_mindconverter_realtime_20260206_213203.jsonl",
        r"D:\graduate\DFrameworkTest\pt_ms_test\llm_vs_mindconverter_realtime_20260206_213647.jsonl",
        r"D:\graduate\DFrameworkTest\pt_ms_test\llm_vs_mindconverter_realtime_20260206_231638.jsonl",
    ]

    all_records: List[Dict] = []
    for file_path in files:
        path = Path(file_path)
        all_records.extend(read_jsonl(path))

    total, llm_ok, mc_ok = analyze_records(all_records)
    print_summary("ALL_FILES", total, llm_ok, mc_ok)


if __name__ == "__main__":
    main()
