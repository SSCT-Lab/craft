#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Use Qwen qwen3-max to filter error samples for whether they are worth filing as issues.

Input files (any can be specified via --input) typically include:
- tf_pd_test_1/analysis/tf_error_only_samples_*.json
- tf_pd_test_1/analysis/pd_error_only_samples_*.json
- tf_pd_test_1/analysis/both_error_samples_*.json

Output file:
- Defaults to the same directory as the input file, named issue_candidates_xxx.json (prefix)

Filtering goal:
- Let the LLM judge whether each sample is more likely a framework issue rather than a migration/parameter mismatch.
- Save all decisions: keep full samples when is_issue=true, keep minimal discriminative info when is_issue=false.
- Save with the LLM decision and a short English reason.

Usage:
    conda activate tf_env
    python tf_pd_test_1/analysisCode/filter_issue_candidates_with_llm.py
    python tf_pd_test_1/analysisCode/filter_issue_candidates_with_llm.py `
        --input tf_pd_test_1/analysis/tf_error_only_samples_20260306_151707.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "qwen3-max"
DEFAULT_KEY_PATH = "aliyun.key"
DEFAULT_WORKERS = 6


def load_api_key(key_path: str = DEFAULT_KEY_PATH) -> str:
    """Load the Aliyun API key."""
    candidate = Path(key_path)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / key_path

    if candidate.exists():
        api_key = candidate.read_text(encoding="utf-8").strip()
        if api_key:
            return api_key

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if api_key:
        return api_key

    print("API key not found. Check aliyun.key or DASHSCOPE_API_KEY")
    return ""


def build_default_output_path(input_path: Path) -> Path:
    """Build the default output path."""
    return input_path.with_name(f"issue_candidates_{input_path.stem}.json")


def detect_counterpart(sample: Dict[str, Any]) -> Tuple[str, str]:
    """Infer counterpart framework name and API field from the sample."""
    execution_result = sample.get("execution_result", {}) if isinstance(sample, dict) else {}
    if not isinstance(execution_result, dict):
        return "counterpart", "counterpart_api"

    known_pairs = [
        ("mindspore", "mindspore_api"),
        ("paddle", "paddle_api"),
        ("pytorch", "pytorch_api"),
    ]
    for name, key in known_pairs:
        if key in execution_result:
            return name, key

    for key in execution_result.keys():
        if key.endswith("_api") and key != "tf_api":
            return key.replace("_api", ""), key
    return "counterpart", "counterpart_api"


def simplify_sample_for_prompt(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Compact the sample to reduce token usage while keeping key fields."""
    execution_result = sample.get("execution_result", {}) if isinstance(sample, dict) else {}
    if not isinstance(execution_result, dict):
        execution_result = {}

    counterpart_name, counterpart_api_key = detect_counterpart(sample)
    counterpart_success_key = f"{counterpart_name}_success"
    counterpart_error_key = f"{counterpart_name}_error"
    counterpart_shape_key = f"{counterpart_name}_shape"
    counterpart_dtype_key = f"{counterpart_name}_dtype"
    counterpart_case_key = f"{counterpart_name}_test_case"

    tf_case = sample.get("tf_test_case", {}) if isinstance(sample, dict) else {}
    counterpart_case = sample.get(counterpart_case_key, {}) if isinstance(sample, dict) else {}
    llm_operation = sample.get("llm_operation", {}) if isinstance(sample, dict) else {}

    return {
        "iteration": sample.get("iteration") if isinstance(sample, dict) else None,
        "case_number": sample.get("case_number") if isinstance(sample, dict) else None,
        "is_llm_generated": sample.get("is_llm_generated") if isinstance(sample, dict) else None,
        "tf_test_case": tf_case,
        counterpart_case_key: counterpart_case,
        "execution_result": {
            "tf_api": execution_result.get("tf_api"),
            counterpart_api_key: execution_result.get(counterpart_api_key),
            "tf_success": execution_result.get("tf_success"),
            counterpart_success_key: execution_result.get(counterpart_success_key),
            "tf_error": execution_result.get("tf_error"),
            counterpart_error_key: execution_result.get(counterpart_error_key),
            "comparison_error": execution_result.get("comparison_error"),
            "tf_shape": execution_result.get("tf_shape"),
            counterpart_shape_key: execution_result.get(counterpart_shape_key),
            "tf_dtype": execution_result.get("tf_dtype"),
            counterpart_dtype_key: execution_result.get(counterpart_dtype_key),
            "status": execution_result.get("status"),
        },
        "llm_operation": {
            "operation": llm_operation.get("operation"),
            "reason": llm_operation.get("reason"),
        },
    }


def build_minimal_discriminative_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Build minimal discriminative info for non-issue samples."""
    execution_result = sample.get("execution_result", {}) if isinstance(sample, dict) else {}
    if not isinstance(execution_result, dict):
        execution_result = {}

    _, counterpart_api_key = detect_counterpart(sample)
    return {
        "iteration": sample.get("iteration") if isinstance(sample, dict) else None,
        "case_number": sample.get("case_number") if isinstance(sample, dict) else None,
        "is_llm_generated": sample.get("is_llm_generated") if isinstance(sample, dict) else None,
        "execution_result": {
            "tf_api": execution_result.get("tf_api"),
            counterpart_api_key: execution_result.get(counterpart_api_key),
            "status": execution_result.get("status"),
        },
    }


def build_prompt(sample: Dict[str, Any]) -> str:
    """Build the prompt for issue filtering."""
    compact = simplify_sample_for_prompt(sample)
    sample_json = json.dumps(compact, ensure_ascii=False, indent=2)

        return f"""You are an expert in differential testing for deep learning frameworks. Determine whether the sample is worth filing as an official framework issue.

[Decision Goal]
Return true only when the issue is more likely a framework bug. If it is more likely caused by test construction/parameter mapping/migration differences, return false.

[High-value issue characteristics]
1. Severe crashes: segfaults, core dumps, process crashes rather than ordinary Python exceptions.
2. Significant numerical errors: abnormal results far beyond floating-point error under valid inputs (e.g., NaN/Inf, huge deviations).
3. Documentation violations: inputs conform to docs but raise unexpected errors.
4. Silent errors: returns success but shape/dtype/semantics are clearly wrong.
5. Stability issues: hangs, deadlocks, obvious resource leaks.

[Cases to exclude (usually return false)]
1. Migration issues such as parameter name mismatches, missing parameters, dtype string vs object mismatches.
2. Invalid or semantically mismatched inputs (e.g., passing dict as tensor).
3. Forced comparisons between non-equivalent APIs.
4. Obvious test harness assembly errors rather than operator implementation bugs.

[Sample]
```json
{sample_json}
```

[Output]
Output JSON only, no extra text:
{
    "is_issue": true or false,
    "reason": "brief reason in English (1-2 sentences, <= 60 words)"
}
"""


def parse_llm_json(text: str) -> Tuple[bool, str]:
    """Parse LLM output with a conservative fallback on failure."""
    if not isinstance(text, str) or not text.strip():
        return False, "LLM returned empty output"

    raw = text.strip()
    try:
        data = json.loads(raw)
        return normalize_llm_result(data)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            data = json.loads(match.group(0))
            return normalize_llm_result(data)
        except json.JSONDecodeError:
            pass

    return False, "LLM output parse failed"


def normalize_llm_result(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Normalize is_issue/reason fields."""
    issue_val = data.get("is_issue", False)
    if isinstance(issue_val, bool):
        is_issue = issue_val
    elif isinstance(issue_val, str):
        is_issue = issue_val.strip().lower() in {"true", "1", "yes", "y"}
    else:
        is_issue = False

    reason = data.get("reason", "")
    if not isinstance(reason, str) or not reason.strip():
        reason = "No valid reason provided"

    return is_issue, reason.strip()[:120]


def evaluate_one_sample(
    client: OpenAI,
    sample: Dict[str, Any],
    model: str,
    temperature: float,
    max_retries: int,
    print_lock: Lock,
) -> Tuple[bool, str]:
    """Call the LLM to judge one sample."""
    prompt = build_prompt(sample)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a rigorous deep learning framework triage expert. Output valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=256,
            )

            content = response.choices[0].message.content
            is_issue, reason = parse_llm_json(content)
            return is_issue, reason

        except Exception as exc:
            with print_lock:
                print(f"  LLM call failed, retry {attempt + 1}/{max_retries}: {str(exc)[:120]}")
            time.sleep(2 ** attempt)

    return False, "LLM call failed (retries exhausted)"


def load_input_samples(input_path: Path) -> Dict[str, Any]:
    """Load the input JSON."""
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data.get("samples"), list):
        raise ValueError("Input JSON missing 'samples' list")

    return data


def save_output(output_path: Path, payload: Dict[str, Any]) -> None:
    """Save the output JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def find_latest_input_file(analysis_dir: Path) -> Path:
    """Auto-select the latest tf_error_only_samples file in analysis."""
    candidates = sorted(analysis_dir.glob("tf_error_only_samples_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No file found: {analysis_dir / 'tf_error_only_samples_*.json'}")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    analysis_dir = ROOT_DIR / "tf_pd_test_1" / "analysis"
    default_input = find_latest_input_file(analysis_dir)

    parser = argparse.ArgumentParser(
        description="Use qwen3-max to filter differential test error samples likely worth filing as issues"
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
        default=None,
        help="Output JSON path (default: issue_candidates_xxx.json in input directory)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Worker threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process only the first N samples (for small tests)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default 0.0, more stable)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per sample (default 3)",
    )
    parser.add_argument(
        "--key-path",
        type=str,
        default=DEFAULT_KEY_PATH,
        help=f"API key file path (default: {DEFAULT_KEY_PATH})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay seconds after each successful request (default 0.2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else build_default_output_path(input_path)
    workers = max(1, int(args.workers))

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    api_key = load_api_key(args.key_path)
    if not api_key:
        raise RuntimeError("No usable API key")

    raw_data = load_input_samples(input_path)
    all_samples: List[Dict[str, Any]] = raw_data.get("samples", [])

    if args.max_samples is not None and args.max_samples > 0:
        samples = all_samples[: args.max_samples]
    else:
        samples = all_samples

    print("=" * 80)
    print("Issue candidate filtering (qwen3-max)")
    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Total samples: {len(all_samples)}")
    print(f"Processing: {len(samples)}")
    print(f"Workers: {workers}")
    print(f"Model: {args.model}")
    print("=" * 80)

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    print_lock = Lock()
    selected_lock = Lock()
    all_results_lock = Lock()
    selected_samples: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []
    started_at = time.time()

    def process(idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        is_issue, reason = evaluate_one_sample(
            client=client,
            sample=sample,
            model=args.model,
            temperature=float(args.temperature),
            max_retries=int(args.max_retries),
            print_lock=print_lock,
        )

        result = {
            "index": idx,
            "is_issue": is_issue,
            "reason": reason,
            "sample": sample if is_issue else build_minimal_discriminative_sample(sample),
        }

        with print_lock:
            print(f"  Sample#{idx}: is_issue={is_issue} | {reason[:40]}")

        if args.delay > 0:
            time.sleep(args.delay)
        return result

    if workers <= 1:
        for idx, sample in enumerate(samples, start=1):
            outcome = process(idx, sample)
            with all_results_lock:
                all_results.append(outcome)
            if outcome["is_issue"]:
                with selected_lock:
                    selected_samples.append(outcome)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(process, idx, sample): idx
                for idx, sample in enumerate(samples, start=1)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    outcome = future.result()
                except Exception as exc:
                    with print_lock:
                        print(f"  Sample#{idx} processing error: {str(exc)[:120]}")
                    continue

                with all_results_lock:
                    all_results.append(outcome)
                if outcome["is_issue"]:
                    with selected_lock:
                        selected_samples.append(outcome)

    all_results.sort(key=lambda x: x["index"])
    selected_samples.sort(key=lambda x: x["index"])

    payload = {
        "generated_at": datetime.now().isoformat(),
        "source_file": str(input_path),
        "model": args.model,
        "criteria": "Filter samples more likely to be framework issues rather than migration/parameter mapping issues",
        "total_samples": len(all_samples),
        "processed_samples": len(samples),
        "non_issue_count": len(samples) - len(selected_samples),
        "issue_candidate_count": len(selected_samples),
        "results": all_results,
        "candidates": selected_samples,
    }
    save_output(output_path, payload)

    elapsed = time.time() - started_at
    print("=" * 80)
    print("Filtering complete")
    print("=" * 80)
    print(f"Candidates: {len(selected_samples)} / {len(samples)}")
    print(f"Elapsed: {elapsed:.1f} seconds")
    print(f"Result file: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
