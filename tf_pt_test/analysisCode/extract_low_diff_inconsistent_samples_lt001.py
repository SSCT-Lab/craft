#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract full samples from inconsistent_success_samples JSON where comparison_error is
"Results inconsistent, max diff: xxx" and xxx < 0.01.

Default input:
    tf_pt_test/analysis/inconsistent_success_samples_*.json (auto-pick latest)
Default output:
    tf_pt_test/analysis/inconsistent_success_samples_lt001_*.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_THRESHOLD = 0.01
ERROR_PATTERN = re.compile(r"^Results inconsistent, max diff:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")
SOURCE_FILE_PATTERN = re.compile(r"^inconsistent_success_samples_\d{8}_\d{6}\.json$")


def extract_max_diff(error_text: Any) -> Optional[float]:
    """Parse max-diff from comparison_error; return None on failure."""
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
    """Build output file name from input file name."""
    stem = input_path.stem
    if stem.startswith("inconsistent_success_samples_"):
        suffix = stem.replace("inconsistent_success_samples_", "", 1)
        output_name = f"inconsistent_success_samples_lt001_{suffix}.json"
    else:
        output_name = f"{stem}_lt001.json"
    return input_path.parent / output_name


def find_latest_inconsistent_file(analysis_dir: Path) -> Path:
    """Select the latest inconsistent_success_samples file under analysis dir."""
    candidates = sorted(
        path for path in analysis_dir.glob("inconsistent_success_samples_*.json")
        if SOURCE_FILE_PATTERN.match(path.name)
    )
    if not candidates:
        raise FileNotFoundError(f"No file found: {analysis_dir / 'inconsistent_success_samples_*.json'}")
    return candidates[-1]


def filter_samples(input_path: Path, output_path: Path, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """Filter samples where comparison_error max diff < threshold and write output."""
    with input_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    samples = raw_data.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("Input JSON samples field is not a list")

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
        "filter_rule": "comparison_error matches 'Results inconsistent, max diff: xxx' and xxx < 0.01",
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
    analysis_dir = root_dir / "tf_pt_test" / "analysis"
    default_input = find_latest_inconsistent_file(analysis_dir)
    default_output = build_default_output_path(default_input)

    parser = argparse.ArgumentParser(
        description="Extract samples with comparison_error max diff < 0.01"
    )
    parser.add_argument("--input", type=Path, default=default_input, help=f"Input JSON (default: {default_input})")
    parser.add_argument("--output", type=Path, default=default_output, help=f"Output JSON (default: {default_output})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Max diff threshold (default: 0.01)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

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