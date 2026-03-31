#!/usr/bin/env python3
"""
Infer effectiveness stats from llm_enhanced log files under pt_ms_log_1,
and output a summary file aligned with llm_effectiveness_validation.py.
"""

import argparse
import glob
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT_DIR, "pt_ms_test", "pt_ms_log_1")


def parse_filename_timestamp(file_path: str) -> Optional[datetime]:
    name = os.path.basename(file_path)
    match = re.search(r"_(\d{8}_\d{6})\.json$", name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def parse_payload_timestamp(payload: Dict[str, Any]) -> Optional[datetime]:
    raw = payload.get("timestamp")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def load_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def both_success(record: Dict[str, Any]) -> bool:
    execution = record.get("execution_result") or {}
    return bool(execution.get("torch_success") and execution.get("mindspore_success"))


def split_cases(results: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    cases: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for item in results:
        iteration = item.get("iteration")
        if iteration == 1 and current:
            cases.append(current)
            current = [item]
        else:
            current.append(item)

    if current:
        cases.append(current)

    return cases


def get_record_by_iteration(case_records: List[Dict[str, Any]], iteration: int) -> Optional[Dict[str, Any]]:
    for record in case_records:
        if record.get("iteration") == iteration:
            return record
    return None


def is_skipped_failed_case(case_records: List[Dict[str, Any]]) -> bool:
    for iteration in (2, 3, 4):
        record = get_record_by_iteration(case_records, iteration)
        if record is None:
            continue
        op = (record.get("llm_operation") or {}).get("operation")
        if op == "skip":
            return True
        return False

    return True


def build_repair_output(stats: Dict[str, Any]) -> Dict[str, Any]:
    effective_total = stats["effective_failed_total"]
    repaired_total = stats["repaired_total"]

    repaired_at_ratio: Dict[str, float] = {}
    for k, v in stats["repaired_at"].items():
        repaired_at_ratio[k] = (v / effective_total) if effective_total > 0 else 0.0

    mutation_ratio: Dict[str, float] = {}
    for k, v in stats["post_repair_mutation_success"].items():
        mutation_ratio[k] = (v / repaired_total) if repaired_total > 0 else 0.0

    return {
        "updated_at": datetime.now().isoformat(),
        "repair_stats": {
            **stats,
            "repaired_at_ratio": repaired_at_ratio,
            "post_repair_mutation_success_ratio": mutation_ratio,
        },
    }


def build_mutate_output(stats: Dict[str, Any]) -> Dict[str, Any]:
    total = stats["initial_success_total"]
    mutate_ratio: Dict[str, float] = {}
    for k, v in stats["mutate_success_at"].items():
        mutate_ratio[k] = (v / total) if total > 0 else 0.0

    return {
        "updated_at": datetime.now().isoformat(),
        "mutate_stats": {
            **stats,
            "mutate_success_at_ratio": mutate_ratio,
        },
    }


def select_latest_file_per_operator(log_files: List[str]) -> Tuple[Dict[str, str], int]:
    latest: Dict[str, Tuple[datetime, str]] = {}
    invalid_count = 0

    for file_path in log_files:
        try:
            payload = load_json(file_path)
        except Exception:
            invalid_count += 1
            continue

        operator = payload.get("operator")
        if not isinstance(operator, str) or not operator.strip():
            invalid_count += 1
            continue

        op = operator.strip()
        ts = parse_payload_timestamp(payload) or parse_filename_timestamp(file_path)
        if ts is None:
            invalid_count += 1
            continue

        prev = latest.get(op)
        if prev is None or ts > prev[0]:
            latest[op] = (ts, file_path)

    return {op: path for op, (_, path) in latest.items()}, invalid_count


def summarize(log_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, int]]:
    pattern = os.path.join(log_dir, "llm_enhanced_*.json")
    all_logs = glob.glob(pattern)

    latest_per_operator, invalid_count = select_latest_file_per_operator(all_logs)

    repair_stats: Dict[str, Any] = {
        "initial_failed_total": 0,
        "effective_failed_total": 0,
        "skipped_cases": 0,
        "repaired_total": 0,
        "repaired_at": {"1": 0, "2": 0, "3": 0},
        "post_repair_mutation_success": {"1": 0, "2": 0, "3": 0},
    }

    mutate_stats: Dict[str, Any] = {
        "initial_success_total": 0,
        "mutate_success_at": {"1": 0, "2": 0, "3": 0},
    }

    total_cases = 0

    for _, file_path in sorted(latest_per_operator.items(), key=lambda x: x[0]):
        payload = load_json(file_path)
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            continue

        cases = split_cases(results)
        for case_records in cases:
            init_record = get_record_by_iteration(case_records, 1)
            if init_record is None:
                continue

            total_cases += 1
            init_ok = both_success(init_record)

            if init_ok:
                mutate_stats["initial_success_total"] += 1
                for round_idx in (1, 2, 3):
                    iter_no = round_idx + 1
                    rec = get_record_by_iteration(case_records, iter_no)
                    if rec is not None and both_success(rec):
                        mutate_stats["mutate_success_at"][str(round_idx)] += 1
                continue

            repair_stats["initial_failed_total"] += 1
            if is_skipped_failed_case(case_records):
                repair_stats["skipped_cases"] += 1
                continue

            repair_stats["effective_failed_total"] += 1

            repaired_round: Optional[int] = None
            for round_idx in (1, 2, 3):
                iter_no = round_idx + 1
                rec = get_record_by_iteration(case_records, iter_no)
                if rec is not None and both_success(rec):
                    repaired_round = round_idx
                    break

            if repaired_round is None:
                continue

            repair_stats["repaired_total"] += 1
            repair_stats["repaired_at"][str(repaired_round)] += 1

            for mut_round in (1, 2, 3):
                iter_no = repaired_round + 1 + mut_round
                rec = get_record_by_iteration(case_records, iter_no)
                if rec is not None and both_success(rec):
                    repair_stats["post_repair_mutation_success"][str(mut_round)] += 1

    diagnostics = {
        "all_log_files": len(all_logs),
        "latest_operator_files": len(latest_per_operator),
        "invalid_or_skipped_files": invalid_count,
        "total_cases": total_cases,
    }

    return build_repair_output(repair_stats), build_mutate_output(mutate_stats), diagnostics


def atomic_write_json(file_path: str, payload: Dict[str, Any]) -> None:
    temp = file_path + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(temp, file_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize llm_enhanced logs into effectiveness summary files (PT-MS)")
    parser.add_argument("--log-dir", default=LOG_DIR)
    parser.add_argument("--output-dir", default=None, help="Optional explicit output directory")
    args = parser.parse_args()

    log_dir = os.path.abspath(args.log_dir)
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = args.output_dir or os.path.join(log_dir, f"effectiveness_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    repair_output, mutate_output, diagnostics = summarize(log_dir)

    repair_path = os.path.join(result_dir, "repair_stats_summary.json")
    mutate_path = os.path.join(result_dir, "mutate_stats_summary.json")

    atomic_write_json(repair_path, repair_output)
    atomic_write_json(mutate_path, mutate_output)

    print("=" * 80)
    print("LLM Enhanced Log Effectiveness Summary (PT-MS)")
    print("=" * 80)
    print(f"log_dir={log_dir}")
    print(f"result_dir={result_dir}")
    print(f"all_log_files={diagnostics['all_log_files']}")
    print(f"latest_operator_files={diagnostics['latest_operator_files']}")
    print(f"invalid_or_skipped_files={diagnostics['invalid_or_skipped_files']}")
    print(f"total_cases={diagnostics['total_cases']}")
    print(f"repair_summary={repair_path}")
    print(f"mutate_summary={mutate_path}")


if __name__ == "__main__":
    main()
