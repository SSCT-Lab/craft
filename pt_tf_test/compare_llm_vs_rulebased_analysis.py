import json
from pathlib import Path
from typing import Dict, List, Tuple


def read_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []

    if not path.exists():
        print(f"[WARN] File does not exist and has been skipped: {path}")
        return records

    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                records.append(json.loads(stripped_line))
            except json.JSONDecodeError as error:
                print(
                    f"[WARN] JSON Parsing failed, skipped {path.name}:{line_number}，reason: {error}"
                )

    return records


def analyze_records(records: List[Dict]) -> Tuple[int, int, int, int, int, int]:
    completed = [record for record in records if record.get("status") == "completed"]

    total_completed = len(completed)
    llm_success = 0
    rule_success = 0
    both_success = 0
    llm_only_success = 0
    rule_only_success = 0

    for record in completed:
        llm_ok = record.get("llm_tf_success") is True
        rule_ok = record.get("rule_tf_success") is True

        if llm_ok:
            llm_success += 1
        if rule_ok:
            rule_success += 1

        if llm_ok and rule_ok:
            both_success += 1
        elif llm_ok and not rule_ok:
            llm_only_success += 1
        elif not llm_ok and rule_ok:
            rule_only_success += 1

    both_fail = total_completed - both_success - llm_only_success - rule_only_success
    return (
        total_completed,
        llm_success,
        rule_success,
        both_success,
        llm_only_success,
        rule_only_success,
    )


def rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100) if denominator > 0 else 0.0


def print_summary(
    name: str,
    total: int,
    llm_ok: int,
    rule_ok: int,
    both_ok: int,
    llm_only_ok: int,
    rule_only_ok: int,
) -> None:
    llm_rate = rate(llm_ok, total)
    rule_rate = rate(rule_ok, total)
    both_fail = total - both_ok - llm_only_ok - rule_only_ok

    print(f"\n=== {name} ===")
    print(f"completed Number of use cases: {total}")
    print(f"LLM Executed successfully: {llm_ok} ({llm_rate:.2f}%)")
    print(f"Rule-based Executed successfully: {rule_ok} ({rule_rate:.2f}%)")
    print(f"double success: {both_ok}")
    print(f"Only LLM successful: {llm_only_ok}")
    print(f"Only Rule-based succeeds: {rule_only_ok}")
    print(f"double fail: {both_fail}")

    if total == 0:
        print("Comparison conclusion: no completed use case, cannot be compared")
        return

    diff = llm_rate - rule_rate
    if diff > 0:
        print(f"Comparison conclusion: LLM is better than Rule-based {diff:.2f} percentage points")
    elif diff < 0:
        print(f"Comparison conclusion: Rule-based is better than LLM {-diff:.2f} percentage points")
    else:
        print("Comparison conclusion: Both are equal")


def main() -> None:
    files = [
        Path(r"D:\graduate\DFrameworkTest\pt_tf_test\llm_vs_rulebased_realtime_20260318_140538.jsonl"),
        Path(r"D:\graduate\DFrameworkTest\pt_tf_test\llm_vs_rulebased_realtime_20260318_144140.jsonl"),
        Path(r"D:\graduate\DFrameworkTest\pt_tf_test\llm_vs_rulebased_realtime_20260318_230728.jsonl"),
    ]

    all_records: List[Dict] = []
    for file_path in files:
        all_records.extend(read_jsonl(file_path))

    (
        total,
        llm_ok,
        rule_ok,
        both_ok,
        llm_only_ok,
        rule_only_ok,
    ) = analyze_records(all_records)
    print_summary(
        "ALL_FILES",
        total,
        llm_ok,
        rule_ok,
        both_ok,
        llm_only_ok,
        rule_only_ok,
    )


if __name__ == "__main__":
    main()
