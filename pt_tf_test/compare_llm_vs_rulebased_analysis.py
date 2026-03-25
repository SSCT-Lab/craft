import json
from pathlib import Path
from typing import Dict, List, Tuple


def read_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []

    if not path.exists():
        print(f"[WARN] 文件不存在，已跳过: {path}")
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
                    f"[WARN] JSON 解析失败，已跳过 {path.name}:{line_number}，原因: {error}"
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
    print(f"completed 用例数: {total}")
    print(f"LLM 执行成功: {llm_ok} ({llm_rate:.2f}%)")
    print(f"Rule-based 执行成功: {rule_ok} ({rule_rate:.2f}%)")
    print(f"双成功: {both_ok}")
    print(f"仅 LLM 成功: {llm_only_ok}")
    print(f"仅 Rule-based 成功: {rule_only_ok}")
    print(f"双失败: {both_fail}")

    if total == 0:
        print("对比结论：无 completed 用例，无法比较")
        return

    diff = llm_rate - rule_rate
    if diff > 0:
        print(f"对比结论：LLM 优于 Rule-based {diff:.2f} 个百分点")
    elif diff < 0:
        print(f"对比结论：Rule-based 优于 LLM {-diff:.2f} 个百分点")
    else:
        print("对比结论：两者持平")


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
