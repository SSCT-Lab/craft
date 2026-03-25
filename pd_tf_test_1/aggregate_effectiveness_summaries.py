#!/usr/bin/env python3
"""
聚合多个 effectiveness 目录中的 repair/mutate summary，
输出两个字段结构保持不变的新 summary 文件。
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG_DIR = os.path.join(ROOT_DIR, "pd_tf_test_1", "pd_tf_log_1")


def _read_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_json_atomic(file_path: str, payload: Dict[str, Any]) -> None:
    temp_path = file_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, file_path)


def _merge_repair(source_dirs: List[str]) -> Dict[str, Any]:
    merged = {
        "initial_failed_total": 0,
        "effective_failed_total": 0,
        "skipped_cases": 0,
        "repaired_total": 0,
        "repaired_at": {"1": 0, "2": 0, "3": 0},
        "post_repair_mutation_success": {"1": 0, "2": 0, "3": 0},
    }

    for folder in source_dirs:
        file_path = os.path.join(folder, "repair_stats_summary.json")
        payload = _read_json(file_path)
        stats = payload.get("repair_stats", {})

        merged["initial_failed_total"] += int(stats.get("initial_failed_total", 0))
        merged["effective_failed_total"] += int(stats.get("effective_failed_total", 0))
        merged["skipped_cases"] += int(stats.get("skipped_cases", 0))
        merged["repaired_total"] += int(stats.get("repaired_total", 0))

        for key in ("1", "2", "3"):
            merged["repaired_at"][key] += int((stats.get("repaired_at") or {}).get(key, 0))
            merged["post_repair_mutation_success"][key] += int(
                (stats.get("post_repair_mutation_success") or {}).get(key, 0)
            )

    effective_total = merged["effective_failed_total"]
    repaired_total = merged["repaired_total"]

    repaired_at_ratio = {
        k: (v / effective_total) if effective_total > 0 else 0.0
        for k, v in merged["repaired_at"].items()
    }
    post_repair_mutation_success_ratio = {
        k: (v / repaired_total) if repaired_total > 0 else 0.0
        for k, v in merged["post_repair_mutation_success"].items()
    }

    return {
        "updated_at": datetime.now().isoformat(),
        "repair_stats": {
            **merged,
            "repaired_at_ratio": repaired_at_ratio,
            "post_repair_mutation_success_ratio": post_repair_mutation_success_ratio,
        },
    }


def _merge_mutate(source_dirs: List[str]) -> Dict[str, Any]:
    merged = {
        "initial_success_total": 0,
        "mutate_success_at": {"1": 0, "2": 0, "3": 0},
    }

    for folder in source_dirs:
        file_path = os.path.join(folder, "mutate_stats_summary.json")
        payload = _read_json(file_path)
        stats = payload.get("mutate_stats", {})

        merged["initial_success_total"] += int(stats.get("initial_success_total", 0))
        for key in ("1", "2", "3"):
            merged["mutate_success_at"][key] += int((stats.get("mutate_success_at") or {}).get(key, 0))

    total = merged["initial_success_total"]
    mutate_success_at_ratio = {
        k: (v / total) if total > 0 else 0.0
        for k, v in merged["mutate_success_at"].items()
    }

    return {
        "updated_at": datetime.now().isoformat(),
        "mutate_stats": {
            **merged,
            "mutate_success_at_ratio": mutate_success_at_ratio,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge repair/mutate summary files from multiple effectiveness folders")
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        required=True,
        help="要聚合的 effectiveness 目录列表",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录，默认在 pd_tf_log_1 下新建 effectiveness_时间戳",
    )
    args = parser.parse_args()

    source_dirs = [os.path.abspath(path) for path in args.source_dirs]
    for folder in source_dirs:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Source folder not found: {folder}")

        repair_path = os.path.join(folder, "repair_stats_summary.json")
        mutate_path = os.path.join(folder, "mutate_stats_summary.json")
        if not os.path.isfile(repair_path):
            raise FileNotFoundError(f"Missing repair summary: {repair_path}")
        if not os.path.isfile(mutate_path):
            raise FileNotFoundError(f"Missing mutate summary: {mutate_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(DEFAULT_LOG_DIR, f"effectiveness_{timestamp}")
    )
    os.makedirs(output_dir, exist_ok=True)

    repair_summary = _merge_repair(source_dirs)
    mutate_summary = _merge_mutate(source_dirs)

    repair_out = os.path.join(output_dir, "repair_stats_summary.json")
    mutate_out = os.path.join(output_dir, "mutate_stats_summary.json")

    _write_json_atomic(repair_out, repair_summary)
    _write_json_atomic(mutate_out, mutate_summary)

    print("=" * 80)
    print("Merged Effectiveness Summary (PD-TF)")
    print("=" * 80)
    print(f"source_dirs={len(source_dirs)}")
    for idx, folder in enumerate(source_dirs, 1):
        print(f"  [{idx}] {folder}")
    print(f"output_dir={output_dir}")
    print(f"repair_summary={repair_out}")
    print(f"mutate_summary={mutate_out}")


if __name__ == "__main__":
    main()
