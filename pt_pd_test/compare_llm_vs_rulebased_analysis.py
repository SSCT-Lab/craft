#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_FILES: List[Path] = []


def read_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []

    if not path.exists():
        print(f"[WARN] File does not exist, skipped: {path}")
        return records

    with path.open("r", encoding="utf-8-sig") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                records.append(json.loads(stripped_line))
            except json.JSONDecodeError as error:
                print(f"[WARN] JSON parse failed, skipped {path.name}:{line_number}, reason: {error}")

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
        llm_ok = record.get("llm_pd_success") is True
        rule_ok = record.get("rule_pd_success") is True

        if llm_ok:
            llm_success += 1
        if rule_ok:
            rule_success += 1

        if llm_ok and rule_ok:
            both_success += 1
        elif llm_ok and (not rule_ok):
            llm_only_success += 1
        elif (not llm_ok) and rule_ok:
            rule_only_success += 1

    return (total_completed, llm_success, rule_success, both_success, llm_only_success, rule_only_success)


def rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100.0) if denominator > 0 else 0.0


def print_summary(name: str, total: int, llm_ok: int, rule_ok: int, both_ok: int, llm_only_ok: int, rule_only_ok: int) -> None:
    llm_rate = rate(llm_ok, total)
    rule_rate = rate(rule_ok, total)
    both_fail = total - both_ok - llm_only_ok - rule_only_ok

    print(f"\n=== {name} ===")
    print(f"completed cases: {total}")
    print(f"LLM success: {llm_ok} ({llm_rate:.2f}%)")
    print(f"Rule-based success: {rule_ok} ({rule_rate:.2f}%)")
    print(f"Both success: {both_ok}")
    print(f"LLM only success: {llm_only_ok}")
    print(f"Rule-based only success: {rule_only_ok}")
    print(f"Both failed: {both_fail}")

    if total == 0:
        print("Conclusion: no completed cases, cannot compare")
        return

    diff = llm_rate - rule_rate
    if diff > 0:
        print(f"Conclusion: LLM outperforms Rule-based by {diff:.2f} percentage points")
    elif diff < 0:
        print(f"Conclusion: Rule-based outperforms LLM by {-diff:.2f} percentage points")
    else:
        print("Conclusion: both are tied")


def collect_default_files(input_dir: Path, latest: int) -> List[Path]:
    candidates = sorted(input_dir.glob("llm_vs_rulebased_realtime_*.jsonl"), key=lambda path: path.name)
    if latest > 0:
        return candidates[-latest:]
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze LLM vs Rule-based realtime JSONL results under pt_pd_test")
    parser.add_argument("--files", nargs="*", default=None, help="JSONL file paths to analyze; if omitted, read latest files in the directory")
    parser.add_argument("--input-dir", default="pt_pd_test", help="Realtime JSONL directory (default pt_pd_test)")
    parser.add_argument("--latest", type=int, default=3, help="When --files is not set, read latest N files (default 3, <=0 means all)")
    args = parser.parse_args()

    if args.files:
        files = [Path(path) for path in args.files]
    else:
        files = [path for path in DEFAULT_FILES if path.exists()]
        if not files:
            files = collect_default_files(Path(args.input_dir), args.latest)

    if not files:
        print("[WARN] No JSONL files found to analyze")
        return

    print("Will analyze the following files:")
    for file_path in files:
        print(f"- {file_path}")

    all_records: List[Dict] = []
    for file_path in files:
        all_records.extend(read_jsonl(file_path))

    total, llm_ok, rule_ok, both_ok, llm_only_ok, rule_only_ok = analyze_records(all_records)
    print_summary("ALL_FILES", total, llm_ok, rule_ok, both_ok, llm_only_ok, rule_only_ok)


if __name__ == "__main__":
    main()
