#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Use Qwen qwen3-max to filter error samples for "worth filing an issue".

Default input:
    ms_tf_test_1/analysis/ms_error_only_samples_*.json (latest automatically)
Default output:
    issue_candidates_{input_stem}.json (in the same directory as input)
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
from typing import Any, Dict, List, Tuple

from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "qwen3-max"
DEFAULT_KEY_PATH = "aliyun.key"
DEFAULT_WORKERS = 6


def load_api_key(key_path: str = DEFAULT_KEY_PATH) -> str:
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

    print("API key not found. Check aliyun.key or DASHSCOPE_API_KEY.")
    return ""


def build_default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"issue_candidates_{input_path.stem}.json")


def find_latest_input_file(analysis_dir: Path) -> Path:
    candidates = sorted(analysis_dir.glob("ms_error_only_samples_*.json"))
    if not candidates:
        raise FileNotFoundError(f"File not found: {analysis_dir / 'ms_error_only_samples_*.json'}")
    return candidates[-1]


def detect_api_roles(sample: Dict[str, Any]) -> Tuple[str, str]:
    execution_result = sample.get("execution_result", {}) if isinstance(sample, dict) else {}
    if not isinstance(execution_result, dict):
        return "ms_api", "counterpart_api"

    api_keys = [key for key in execution_result.keys() if key.endswith("_api")]
    if not api_keys:
        return "ms_api", "counterpart_api"

    primary_api_key = "ms_api" if "ms_api" in api_keys else api_keys[0]
    counterpart_api_key = next((k for k in api_keys if k != primary_api_key), "counterpart_api")
    return primary_api_key, counterpart_api_key


def simplify_sample_for_prompt(sample: Dict[str, Any]) -> Dict[str, Any]:
    execution_result = sample.get("execution_result", {}) if isinstance(sample, dict) else {}
    if not isinstance(execution_result, dict):
        execution_result = {}

    primary_api_key, counterpart_api_key = detect_api_roles(sample)
    primary_prefix = primary_api_key[:-4]
    counterpart_prefix = counterpart_api_key[:-4]

    primary_case_key = f"{primary_prefix}_test_case"
    counterpart_case_key = f"{counterpart_prefix}_test_case"

    primary_success_key = f"{primary_prefix}_success"
    counterpart_success_key = f"{counterpart_prefix}_success"
    primary_error_key = f"{primary_prefix}_error"
    counterpart_error_key = f"{counterpart_prefix}_error"
    primary_shape_key = f"{primary_prefix}_shape"
    counterpart_shape_key = f"{counterpart_prefix}_shape"
    primary_dtype_key = f"{primary_prefix}_dtype"
    counterpart_dtype_key = f"{counterpart_prefix}_dtype"

    llm_operation = sample.get("llm_operation", {}) if isinstance(sample, dict) else {}

    return {
        "iteration": sample.get("iteration") if isinstance(sample, dict) else None,
        "case_number": sample.get("case_number") if isinstance(sample, dict) else None,
        "is_llm_generated": sample.get("is_llm_generated") if isinstance(sample, dict) else None,
        primary_case_key: sample.get(primary_case_key, {}) if isinstance(sample, dict) else {},
        counterpart_case_key: sample.get(counterpart_case_key, {}) if isinstance(sample, dict) else {},
        "execution_result": {
            primary_api_key: execution_result.get(primary_api_key),
            counterpart_api_key: execution_result.get(counterpart_api_key),
            primary_success_key: execution_result.get(primary_success_key),
            counterpart_success_key: execution_result.get(counterpart_success_key),
            primary_error_key: execution_result.get(primary_error_key),
            counterpart_error_key: execution_result.get(counterpart_error_key),
            "comparison_error": execution_result.get("comparison_error"),
            primary_shape_key: execution_result.get(primary_shape_key),
            counterpart_shape_key: execution_result.get(counterpart_shape_key),
            primary_dtype_key: execution_result.get(primary_dtype_key),
            counterpart_dtype_key: execution_result.get(counterpart_dtype_key),
            "status": execution_result.get("status"),
        },
        "llm_operation": {
            "operation": llm_operation.get("operation"),
            "reason": llm_operation.get("reason"),
        },
    }


def build_minimal_discriminative_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    execution_result = sample.get("execution_result", {}) if isinstance(sample, dict) else {}
    if not isinstance(execution_result, dict):
        execution_result = {}

    primary_api_key, counterpart_api_key = detect_api_roles(sample)
    return {
        "iteration": sample.get("iteration") if isinstance(sample, dict) else None,
        "case_number": sample.get("case_number") if isinstance(sample, dict) else None,
        "is_llm_generated": sample.get("is_llm_generated") if isinstance(sample, dict) else None,
        "execution_result": {
            primary_api_key: execution_result.get(primary_api_key),
            counterpart_api_key: execution_result.get(counterpart_api_key),
            "status": execution_result.get("status"),
        },
    }


def build_prompt(sample: Dict[str, Any]) -> str:
    compact = simplify_sample_for_prompt(sample)
    sample_json = json.dumps(compact, ensure_ascii=False, indent=2)

        return f"""You are an expert in differential testing of deep learning frameworks. Determine whether the sample below is "worth filing an official framework issue".

Decision goal:
Return true only when it is more likely to be a framework issue. If it is more likely caused by test construction/parameter mapping/migration differences, return false.

High-value issue signals:
1. Severe crash: segfault, core dump, process crash (not a normal Python exception).
2. Significant numeric error: abnormal results far beyond floating-point tolerance for valid inputs.
3. Doc violation: inputs conform to official docs but unexpected exceptions occur.
4. Silent error: returns success but shape/dtype/semantics are clearly wrong.
5. Stability issues: hang, deadlock, obvious resource leak.

Cases to exclude (usually return false):
1. Migration issues like parameter name mismatch, missing params, dtype string vs object mismatch.
2. Invalid inputs or semantic mismatch (e.g., passing a dict as a tensor).
3. Forcing comparisons between non-equivalent APIs.
4. Test harness assembly error rather than operator implementation error.

Sample:
```json
{sample_json}
```

Output requirements:
Output JSON only, no extra text:
{{
    "is_issue": true or false,
    "reason": "Short reason in English (1-2 sentences, <= 60 words)"
}}
"""


def normalize_llm_result(data: Dict[str, Any]) -> Tuple[bool, str]:
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


def parse_llm_json(text: str) -> Tuple[bool, str]:
    if not isinstance(text, str) or not text.strip():
        return False, "LLM returned empty"

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

    return False, "Failed to parse LLM response"


def evaluate_one_sample(
    client: OpenAI,
    sample: Dict[str, Any],
    model: str,
    temperature: float,
    max_retries: int,
    print_lock: Lock,
) -> Tuple[bool, str]:
    prompt = build_prompt(sample)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a rigorous triage expert for deep learning framework defects. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=256,
            )
            content = response.choices[0].message.content
            return parse_llm_json(content)
        except Exception as exc:
            with print_lock:
                print(f"  LLM call failed, retry {attempt + 1}/{max_retries}: {str(exc)[:120]}")
            time.sleep(2 ** attempt)

    return False, "LLM call failed (retries exhausted)"


def load_input_samples(input_path: Path) -> Dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data.get("samples"), list):
        raise ValueError("Input JSON missing 'samples' list")
    return data


def save_output(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    analysis_dir = ROOT_DIR / "ms_tf_test_1" / "analysis"
    default_input = find_latest_input_file(analysis_dir)

    parser = argparse.ArgumentParser(description="Use qwen3-max to filter error samples likely worth filing issues")
    parser.add_argument("--input", type=Path, default=default_input, help=f"Input JSON (default: {default_input})")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON (auto-generated by default)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Worker threads (default: {DEFAULT_WORKERS})")
    parser.add_argument("--max-samples", type=int, default=None, help="Process only the first N samples")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per sample (default: 3)")
    parser.add_argument("--key-path", type=str, default=DEFAULT_KEY_PATH, help=f"API key file (default: {DEFAULT_KEY_PATH})")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay seconds after each successful request (default: 0.2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else build_default_output_path(input_path)
    workers = max(1, int(args.workers))

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    api_key = load_api_key(args.key_path)
    if not api_key:
        raise RuntimeError("No usable API key available")

    raw_data = load_input_samples(input_path)
    all_samples: List[Dict[str, Any]] = raw_data.get("samples", [])
    samples = all_samples[: args.max_samples] if args.max_samples and args.max_samples > 0 else all_samples

    print("=" * 80)
    print("Issue candidate filtering (qwen3-max)")
    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Total samples: {len(all_samples)}")
    print(f"Processing this run: {len(samples)}")
    print(f"Worker threads: {workers}")
    print(f"Model: {args.model}")
    print("=" * 80)

    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

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
            print(f"  Sample #{idx}: is_issue={is_issue} | {reason[:40]}")
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
            future_to_idx = {executor.submit(process, idx, sample): idx for idx, sample in enumerate(samples, start=1)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    outcome = future.result()
                except Exception as exc:
                    with print_lock:
                        print(f"  Sample #{idx} processing error: {str(exc)[:120]}")
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
    print(f"Output file: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
