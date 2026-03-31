#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract full samples from inconsistent_success_samples JSON where comparison_error is
"结果不一致，最大差异: xxx" and xxx < 1.

Default input:
    tf_pd_test_1/analysis/inconsistent_success_samples_*.json (latest)
Default output:
    tf_pd_test_1/analysis/inconsistent_success_samples_lt1_*.json

Usage:
    conda activate tf_env
    python tf_pd_test_1/analysisCode/extract_low_diff_inconsistent_samples.py
    python tf_pd_test_1/analysisCode/extract_low_diff_inconsistent_samples.py \
        --input tf_pd_test_1/analysis/inconsistent_success_samples_20260306_151707.json \
        --output tf_pd_test_1/analysis/inconsistent_success_samples_lt1_20260306_151707.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


ERROR_PATTERN = re.compile(r"^结果不一致，最大差异:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")


def extract_max_diff(error_text: Any) -> Optional[float]:
    """Parse max diff value from comparison_error; return None on failure."""
    if not isinstance(error_text, str):
        return None

    match = ERROR_PATTERN.match(error_text.strip())
    if not match:
        return None

    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def build_default_output_path(input_path: Path) -> Path:
    """Build output filename based on input name."""
    stem = input_path.stem
    if stem.startswith("inconsistent_success_samples_"):
        suffix = stem.replace("inconsistent_success_samples_", "", 1)
        output_name = f"inconsistent_success_samples_lt1_{suffix}.json"
    else:
        output_name = f"{stem}_lt1.json"
    return input_path.parent / output_name


def find_latest_inconsistent_file(analysis_dir: Path) -> Path:
    """Auto-select the latest inconsistent_success_samples file in analysis."""
    candidates = sorted(analysis_dir.glob("inconsistent_success_samples_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No file found: {analysis_dir / 'inconsistent_success_samples_*.json'}")
    return candidates[-1]


def filter_samples(input_path: Path, output_path: Path, threshold: float = 1.0) -> Dict[str, Any]:
    """Filter full samples where max comparison_error diff < threshold and write output."""
    with input_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    samples = raw_data.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("Input JSON 'samples' is not a list")

    selected_samples: List[Dict[str, Any]] = []

    for sample in samples:
        if not isinstance(sample, dict):
            continue

        execution_result = sample.get("execution_result", {})
        if not isinstance(execution_result, dict):
            continue

        comparison_error = execution_result.get("comparison_error")
        max_diff = extract_max_diff(comparison_error)
        if max_diff is None:
            continue

        if max_diff < threshold:
            selected_samples.append(sample)

    output_data = {
        "source_file": str(input_path),
        "filter_rule": "comparison_error matches '结果不一致，最大差异: xxx' and xxx < threshold",
        "threshold": threshold,
        "total_samples": len(samples),
        "matched_samples": len(selected_samples),
        "samples": selected_samples,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, ensure_ascii=False, indent=2)

    return output_data


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[2]
    analysis_dir = root_dir / "tf_pd_test_1" / "analysis"
    default_input = find_latest_inconsistent_file(analysis_dir)
    default_output = build_default_output_path(default_input)

    parser = argparse.ArgumentParser(
        description="Extract full samples where max comparison_error diff < threshold"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Input JSON file path (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output JSON file path (default: {default_output})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Max diff threshold (default: 1.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = filter_samples(input_path=input_path, output_path=output_path, threshold=args.threshold)

    print("=" * 80)
    print("Filtering complete")
    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Total samples: {result['total_samples']}")
    print(f"Matched samples: {result['matched_samples']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
