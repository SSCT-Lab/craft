#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3: Generate MindSpore -> PaddlePaddle API mapping with an LLM

Functionality:
- Read the MS API list from Step 1/1.5 outputs
- For each MS API, call the LLM to find a functionally equivalent PaddlePaddle API
- Support concurrent LLM calls and resume from existing results
- Output a CSV mapping table

PaddlePaddle notes:
- Paddle uses NCHW by default (same as MindSpore), no format conversion needed
- Paddle dense layer is paddle.nn.Linear (MindSpore uses mindspore.nn.Dense)
- Paddle functional APIs live in paddle.nn.functional and the top-level namespace
- Paddle convolution class uses uppercase D: paddle.nn.Conv2D

Usage:
    conda activate tf_env
    python ms_pd_test_1/extract_ms_pd_mapping.py [--input data/ms_apis_existing.json] [--output data/ms_pd_mapping.csv] [--workers 6]

Output: ms_pd_test_1/data/ms_pd_mapping.csv
"""

import os
import sys

import csv
import json
import time
import re
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== Constants ====================
DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"
DEFAULT_WORKERS = 6


def load_api_key(key_path: str = DEFAULT_KEY_PATH) -> str:
    """Load the Aliyun API key."""
    if not os.path.isabs(key_path):
        key_file = os.path.join(ROOT_DIR, key_path)
    else:
        key_file = key_path

    if os.path.exists(key_file):
        with open(key_file, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()
        if api_key:
            return api_key

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        return api_key

    print("❌ API key not found")
    return ""


def determine_api_level(api_name: str) -> str:
    """Determine MS API level: function/class/tensor method."""
    parts = api_name.split(".")
    last_part = parts[-1] if parts else ""

    if "Tensor" in api_name:
        return "tensor_method"
    if ".nn." in api_name and last_part and last_part[0].isupper():
        return "class"
    if ".ops." in api_name and last_part and last_part[0].isupper():
        return "class"
    return "function"


def build_prompt_for_api(ms_api: str, api_level: str) -> str:
    """Build the LLM prompt for a single MS API (MS->PD mapping)."""
    level_desc = {"function": "function", "class": "class", "tensor_method": "tensor method"}.get(api_level, "function")

    prompt = f"""You are an expert in MindSpore and PaddlePaddle.

[Task]
Find the functionally equivalent PaddlePaddle API for the following MindSpore API.

[MindSpore API]
{ms_api}

[API Level]
This API is a **{level_desc}-level** API.

[Common MS->PD mapping references]
- mindspore.ops.Abs / mindspore.ops.abs -> paddle.abs
- mindspore.ops.Add / mindspore.ops.add -> paddle.add
- mindspore.ops.MatMul / mindspore.ops.matmul -> paddle.matmul
- mindspore.ops.Conv2D -> paddle.nn.functional.conv2d
- mindspore.nn.Conv2d -> paddle.nn.Conv2D (note the uppercase D in Paddle)
- mindspore.nn.BatchNorm2d -> paddle.nn.BatchNorm2D (note the uppercase D in Paddle)
- mindspore.nn.Dense -> paddle.nn.Linear
- mindspore.nn.ReLU -> paddle.nn.ReLU
- mindspore.ops.Softmax / mindspore.ops.softmax -> paddle.nn.functional.softmax
- mindspore.Tensor.add -> paddle.Tensor.add

[Requirements]
1. The returned PaddlePaddle API must be functionally equivalent or very close.
2. Prefer the API with the closest functionality and parameters.
3. If no equivalent API exists in PaddlePaddle, return "no_matching_impl".
4. Return exactly one best API, not multiple candidates.
5. Distinguish these cases:
   - MindSpore Primitive ops (e.g., mindspore.ops.Abs) usually map to Paddle top-level functions (e.g., paddle.abs)
   - MindSpore NN layers (e.g., mindspore.nn.Conv2d) usually map to Paddle NN layers (e.g., paddle.nn.Conv2D)
   - MindSpore functional APIs (e.g., mindspore.ops.relu) usually map to paddle.nn.functional.xxx
   - MindSpore tensor methods map to Paddle tensor methods

[PaddlePaddle API namespace reference]
- Base functions: paddle.xxx (e.g., paddle.abs, paddle.add, paddle.matmul, paddle.reshape)
- Neural network layers (classes): paddle.nn.XXX (e.g., paddle.nn.Conv2D, paddle.nn.ReLU, paddle.nn.Linear)
  Note: Paddle uses uppercase suffixes like Conv2D, BatchNorm2D (different from MindSpore's Conv2d, BatchNorm2d)
- Neural network functions: paddle.nn.functional.xxx (e.g., paddle.nn.functional.relu, paddle.nn.functional.softmax)
- Linear algebra: paddle.linalg.xxx (e.g., paddle.linalg.det, paddle.linalg.inv)
- Signal processing: paddle.fft.xxx (e.g., paddle.fft.fft, paddle.fft.ifft)
- Random: paddle.xxx (e.g., paddle.randn, paddle.rand) or paddle.distribution.xxx
- Tensor methods: paddle.Tensor.xxx (e.g., paddle.Tensor.add, paddle.Tensor.reshape)

[Data format]
MindSpore and PaddlePaddle both use NCHW by default, no conversion needed.

[Output format]
Return strictly in the following JSON format with no extra text:

```json
{{
    "mindspore_api": "{ms_api}",
    "paddle_api": "<PaddlePaddle API name or 'no_matching_impl'>",
    "confidence": "<high/medium/low>",
    "reason": "<brief reason for mapping or why no match>"
}}
```

Notes:
- The paddle_api field must be the full API name (e.g., paddle.abs or paddle.nn.Conv2D), or "no_matching_impl".
- The paddle_api value must be a real PaddlePaddle API, not fabricated.
- confidence: your confidence (high >= 85%, medium 40%-85%, low < 40%).
- reason: brief explanation (1-2 sentences).
- Only map public MindSpore APIs; for internal or experimental APIs, return "no_matching_impl".
"""
    return prompt


def parse_llm_response(response: str) -> Tuple[str, str, str]:
    """Parse the LLM JSON response.

    Returns:
        (paddle_api, confidence, reason)
    """
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            pd_api = data.get("paddle_api", "no_matching_impl").strip()
            confidence = data.get("confidence", "unknown").strip()
            reason = data.get("reason", "").strip()
            return pd_api, confidence, reason
    except json.JSONDecodeError:
        pass

    if "no_matching_impl" in response:
        return "no_matching_impl", "unknown", "Parse failed but detected no_matching_impl"

    # Try extracting Paddle API from response text.
    pd_pattern = r'(paddle\.[a-zA-Z_][a-zA-Z0-9_\.]*)'
    matches = re.findall(pd_pattern, response)
    if matches:
        return matches[0], "unknown", "extracted_from_response_text"

    return "no_matching_impl", "unknown", "Parse failed"


def query_llm_for_api(
    client: OpenAI,
    ms_api: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_retries: int = 3,
    print_lock: Lock = None,
) -> Tuple[str, str, str]:
    """Call the LLM to get the corresponding PaddlePaddle API.

    Returns:
        (paddle_api, confidence, reason)
    """
    lock = print_lock or Lock()
    api_level = determine_api_level(ms_api)
    prompt = build_prompt_for_api(ms_api, api_level)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024,
            )
            full_response = resp.choices[0].message.content.strip()
            time.sleep(0.5)

            return parse_llm_response(full_response)

        except Exception as e:
            with lock:
                print(
                    f"  ⚠️ {ms_api} LLM call failed: {str(e)[:80]}, "
                    f"retry ({attempt + 1}/{max_retries})"
                )
            time.sleep(2 ** attempt)

    return "no_matching_impl", "unknown", "All retries failed"


def load_existing_mapping(csv_path: str) -> Dict[str, str]:
    """Load existing mapping results (for resume)."""
    if not os.path.exists(csv_path):
        return {}
    existing = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ms_api = row.get("mindspore-api", "").strip()
                pd_api = row.get("paddle-api", "").strip()
                if ms_api:
                    existing[ms_api] = pd_api
    except Exception:
        pass
    return existing


def save_mapping(csv_path: str, mappings: List[Dict[str, str]]):
    """Save mapping results to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=["mindspore-api", "paddle-api", "confidence", "reason"]
        )
        writer.writeheader()
        for m in mappings:
            writer.writerow(m)


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Generate MindSpore -> PaddlePaddle API mapping with an LLM"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(ROOT_DIR, "ms_pd_test_1", "data", "ms_apis_existing.json"),
        help="MS API list file",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "ms_pd_test_1", "data", "ms_pd_mapping.csv"),
        help="Output CSV mapping file path",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"LLM concurrent workers (default {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL,
        help=f"LLM model name (default {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--key-path", "-k", default=DEFAULT_KEY_PATH,
        help=f"API key file path (default {DEFAULT_KEY_PATH})",
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.1,
        help="LLM temperature (default 0.1)",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start processing from this API index (0-indexed)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of APIs to process",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay seconds between API calls",
    )

    args = parser.parse_args()
    workers = max(1, args.workers)

    print("=" * 80)
    print("Step 3: Generate MindSpore -> PaddlePaddle API mapping with an LLM")
    print("=" * 80)

    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        api_data = json.load(f)

    all_apis = [a["ms_api"] for a in api_data.get("apis", [])]
    print(f"📋 Loaded {len(all_apis)} MS APIs")

    # Determine processing range
    start_idx = args.start
    end_idx = start_idx + args.limit if args.limit else len(all_apis)
    end_idx = min(end_idx, len(all_apis))
    apis_to_process = all_apis[start_idx:end_idx]

    # Load existing mappings (resume support)
    existing_mapping = load_existing_mapping(args.output)
    apis_remaining = [a for a in apis_to_process if a not in existing_mapping]

    print(f"📌 Range: [{start_idx}, {end_idx}), total {len(apis_to_process)}")
    print(f"📌 Existing mappings: {len(existing_mapping)} (skipped)")
    print(f"📌 Remaining: {len(apis_remaining)}")
    print(f"📌 Worker count: {workers}")
    print(f"📌 LLM model: {args.model}")

    if not apis_remaining:
        print("✅ All APIs have been processed")
        return

    api_key = load_api_key(args.key_path)
    if not api_key:
        sys.exit(1)

    llm_client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    print_lock = Lock()
    mappings_lock = Lock()

    # Initialize result list (including existing mappings)
    all_mappings: List[Dict[str, str]] = []
    for ms_api, pd_api in existing_mapping.items():
        all_mappings.append({
            "mindspore-api": ms_api,
            "paddle-api": pd_api,
            "confidence": "",
            "reason": "existing_mapping",
        })

    def process_api(ms_api: str) -> Dict[str, str]:
        pd_api, confidence, reason = query_llm_for_api(
            llm_client, ms_api, model=args.model,
            temperature=args.temperature, print_lock=print_lock,
        )
        with print_lock:
            emoji = "✅" if pd_api != "no_matching_impl" else "⏭️"
            print(f"  {emoji} {ms_api} -> {pd_api} ({confidence})")
        time.sleep(args.delay)
        return {
            "mindspore-api": ms_api,
            "paddle-api": pd_api,
            "confidence": confidence,
            "reason": reason,
        }

    print(f"\n🚀 Start generating MS->PD mappings (workers={workers})...\n")
    start_time = time.time()
    completed = 0
    total = len(apis_remaining)

    if workers <= 1:
        for ms_api in apis_remaining:
            result = process_api(ms_api)
            with mappings_lock:
                all_mappings.append(result)
            completed += 1
            if completed % 20 == 0:
                save_mapping(args.output, all_mappings)
                print(f"  💾 Progress: {completed}/{total}, saved intermediate results")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_api = {
                executor.submit(process_api, ms_api): ms_api
                for ms_api in apis_remaining
            }
            for future in as_completed(future_to_api):
                try:
                    result = future.result()
                    with mappings_lock:
                        all_mappings.append(result)
                except Exception as e:
                    api_name = future_to_api[future]
                    with print_lock:
                        print(f"  ❌ {api_name} error: {e}")
                    with mappings_lock:
                        all_mappings.append({
                            "mindspore-api": api_name,
                            "paddle-api": "no_matching_impl",
                            "confidence": "unknown",
                            "reason": f"processing_error: {e}",
                        })

                completed += 1
                if completed % 30 == 0:
                    with mappings_lock:
                        save_mapping(args.output, all_mappings)
                    with print_lock:
                        print(f"  💾 Progress: {completed}/{total}, saved intermediate results")

    all_mappings.sort(key=lambda x: x["mindspore-api"])
    save_mapping(args.output, all_mappings)

    elapsed = time.time() - start_time
    has_impl = sum(1 for m in all_mappings if m["paddle-api"] != "no_matching_impl")

    print(f"\n{'=' * 80}")
    print("📊 Mapping generation complete")
    print(f"{'=' * 80}")
    print(f"  Total APIs: {len(all_mappings)}")
    print(f"  With matching impl: {has_impl}")
    print(f"  No matching impl: {len(all_mappings) - has_impl}")
    print(f"  Elapsed: {elapsed:.1f} s")
    print(f"  💾 Saved to: {args.output}")


if __name__ == "__main__":
    main()
