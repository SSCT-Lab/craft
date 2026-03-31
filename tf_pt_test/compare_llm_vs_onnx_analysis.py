import json
from pathlib import Path
from typing import Dict, List, Tuple


def read_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze_records(records: List[Dict]) -> Tuple[int, int, int]:
    completed = [record for record in records if record.get("status") == "completed"]
    total_completed = len(completed)

    # In TF->PT experiments, the realtime log field is usually llm_pt_success; keep compatibility with llm_tf_success.
    llm_success = sum(
        1
        for record in completed
        if record.get("llm_pt_success") is True or record.get("llm_tf_success") is True
    )
    onnx_success = sum(1 for record in completed if record.get("onnx_run_success") is True)
    return total_completed, llm_success, onnx_success


def rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100) if denominator > 0 else 0.0


def print_summary(name: str, total: int, llm_ok: int, onnx_ok: int):
    llm_rate = rate(llm_ok, total)
    onnx_rate = rate(onnx_ok, total)

    print(f"\n=== {name} ===")
    print(f"completed cases: {total}")
    print(f"LLM success: {llm_ok} ({llm_rate:.2f}%)")
    print(f"ONNX success: {onnx_ok} ({onnx_rate:.2f}%)")

    if total == 0:
        print("Conclusion: no completed cases, cannot compare")
        return

    diff = llm_rate - onnx_rate
    if diff > 0:
        print(f"Conclusion: LLM outperforms ONNX by {diff:.2f} percentage points")
    elif diff < 0:
        print(f"Conclusion: ONNX outperforms LLM by {-diff:.2f} percentage points")
    else:
        print("Conclusion: both are tied")


def main():
    files = [
        r"D:\graduate\DFrameworkTest\tf_pt_test\llm_vs_onnx_realtime_20260318_102126.jsonl",
    ]

    all_records: List[Dict] = []
    for file_path in files:
        path = Path(file_path)
        all_records.extend(read_jsonl(path))

    total, llm_ok, onnx_ok = analyze_records(all_records)
    print_summary("TF_PT_FILE", total, llm_ok, onnx_ok)


if __name__ == "__main__":
    main()
