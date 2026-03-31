#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3: Generate a TensorFlow → PyTorch API mapping table with LLM

Features:
- Read the TF API list from Step 1
- For each TF API, ask the LLM for an equivalent PyTorch API
- Support concurrent LLM calls
- Support resume from existing mapping
- Output CSV mapping

Usage:
    conda activate tf_env
    python tf_pt_test/extract_tf_pt_mapping.py [--input data/tf_apis_existing.json] [--output data/tf_pt_mapping.csv] [--workers 6]

Output: tf_pt_test/data/tf_pt_mapping.csv
"""

import os
import sys
import csv
import json
import time
import re
import argparse
from datetime import datetime
from pathlib import Path
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
    """Load Aliyun API key."""
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
    """
    Determine TF API level: function-level or class-level.

    Rules:
    - tf.keras.layers.XXX (capitalized) -> class-level
    - tf.keras.losses.XXX (capitalized) -> class-level
    - Otherwise -> function-level
    """
    parts = api_name.split(".")
    if "keras" in api_name and "layers" in api_name:
        last_part = parts[-1]
        if last_part and last_part[0].isupper():
            return "class"
    if "keras" in api_name and "losses" in api_name:
        last_part = parts[-1]
        if last_part and last_part[0].isupper():
            return "class"
    return "function"


def build_prompt_for_api(tf_api: str, api_level: str) -> str:
    """Build an LLM prompt for a single TF API."""
    level_desc = "function" if api_level == "function" else "class"
    level_example_tf = "tf.nn.relu" if api_level == "function" else "tf.keras.layers.Conv1D"
    level_example_pt = "torch.nn.functional.relu" if api_level == "function" else "torch.nn.Conv1d"

    prompt = f"""You are an expert in TensorFlow and PyTorch.

[Task]
Find the PyTorch API that is functionally equivalent to the following TensorFlow API.

[TensorFlow API]
{tf_api}

[API level]
This is a **{level_desc}-level** API.
- If the API is a function (e.g., {level_example_tf}), return the corresponding PyTorch function (e.g., {level_example_pt}).
- If the API is a class (e.g., tf.keras.layers.Conv1D), return the corresponding PyTorch class (e.g., torch.nn.Conv1d).

[Requirements]
1. The returned PyTorch API must be equivalent or very close in functionality.
2. Prefer the API with the closest behavior and parameters.
3. If PyTorch has no equivalent API, return "无对应实现".
4. Return only one best API; do not return multiple candidates.

[PyTorch namespace reference]
- Core functions: torch.xxx (e.g., torch.abs, torch.add, torch.matmul)
- NN layers (classes): torch.nn.XXX (e.g., torch.nn.Conv1d, torch.nn.ReLU, torch.nn.Linear)
- NN functions: torch.nn.functional.xxx (e.g., torch.nn.functional.relu, torch.nn.functional.softmax)
- Linear algebra: torch.linalg.xxx (e.g., torch.linalg.det, torch.linalg.inv)
- Random: torch.xxx or torch.distributions.xxx
- Signal processing: torch.fft.xxx
- Image processing: torchvision.transforms.functional.xxx

[Output format]
Return strictly in the following JSON format, with no extra text:

```json
{{
    "tensorflow_api": "{tf_api}",
    "pytorch_api": "<PyTorch API name or '无对应实现'>",
    "confidence": "<high/medium/low>",
    "reason": "<brief mapping reason or why no equivalent>"
}}
```

Notes:
- The pytorch_api field must be the full API name (e.g., torch.abs or torch.nn.Conv1d), or "无对应实现".
- The pytorch_api value must be a real PyTorch API name, not a made-up one.
- confidence reflects your confidence in equivalence (>=85% high, 40-85% medium, <40% low).
- reason should be brief (1-2 sentences).
"""
    return prompt


def parse_llm_response(response: str) -> Tuple[str, str, str]:
    """Parse the LLM JSON response and return (pytorch_api, confidence, reason)."""
    try:
        # Extract JSON block.
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            pt_api = data.get("pytorch_api", "无对应实现").strip()
            confidence = data.get("confidence", "unknown").strip()
            reason = data.get("reason", "").strip()
            return pt_api, confidence, reason
    except json.JSONDecodeError:
        pass

    if "无对应实现" in response:
        return "无对应实现", "unknown", "Parse failed, but detected '无对应实现'"

    # Try regex to extract torch.* API
    torch_pattern = r'(torch\.[a-zA-Z_][a-zA-Z0-9_\.]*)'
    matches = re.findall(torch_pattern, response)
    if matches:
        return matches[0], "unknown", "Extracted from response text"

    return "无对应实现", "unknown", "Parse failed"


def query_llm_for_api(
    client: OpenAI,
    tf_api: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_retries: int = 3,
    print_lock: Lock = None,
) -> Tuple[str, str, str]:
    """Call LLM to get the corresponding PyTorch API."""
    lock = print_lock or Lock()
    api_level = determine_api_level(tf_api)
    prompt = build_prompt_for_api(tf_api, api_level)

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

            pt_api, confidence, reason = parse_llm_response(full_response)
            return pt_api, confidence, reason

        except Exception as e:
            with lock:
                print(f"  ⚠️ {tf_api} LLM call failed: {str(e)[:80]}, retry ({attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)

    return "无对应实现", "unknown", "All retries failed"


def load_existing_mapping(csv_path: str) -> Dict[str, str]:
    """Load existing mappings (for resume)."""
    if not os.path.exists(csv_path):
        return {}
    existing = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tf_api = row.get("tensorflow-api", "").strip()
                pt_api = row.get("pytorch-api", "").strip()
                if tf_api:
                    existing[tf_api] = pt_api
    except Exception:
        pass
    return existing


def save_mapping(csv_path: str, mappings: List[Dict[str, str]]):
    """Save mappings to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["tensorflow-api", "pytorch-api", "confidence", "reason"])
        writer.writeheader()
        for m in mappings:
            writer.writerow(m)


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Generate TensorFlow → PyTorch API mapping with LLM"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(ROOT_DIR, "tf_pt_test", "data", "tf_apis_existing.json"),
        help="Filtered TF API list JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "tf_pt_test", "data", "tf_pt_mapping.csv"),
        help="Output CSV mapping file path"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"LLM concurrent workers (default {DEFAULT_WORKERS})"
    )
    parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL,
        help=f"LLM model name (default {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--key-path", "-k", default=DEFAULT_KEY_PATH,
        help=f"API key file path (default {DEFAULT_KEY_PATH})"
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.1,
        help="LLM temperature (default 0.1; lower is more deterministic)"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start from which API (0-indexed)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max APIs to process"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="API call delay seconds"
    )

    args = parser.parse_args()
    workers = max(1, args.workers)

    print("=" * 80)
    print("Step 3: Generate TensorFlow → PyTorch API mapping with LLM")
    print("=" * 80)

    # Load API list.
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        print("Please generate tf_apis_existing.json first (e.g., run filter_existing_tf_apis.py)")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        api_data = json.load(f)

    all_apis = [a["tf_api"] for a in api_data.get("apis", [])]
    print(f"📋 Loaded {len(all_apis)} TF APIs")

    # Determine processing range.
    start_idx = args.start
    end_idx = start_idx + args.limit if args.limit else len(all_apis)
    end_idx = min(end_idx, len(all_apis))
    apis_to_process = all_apis[start_idx:end_idx]

    # Load existing mapping (resume).
    existing_mapping = load_existing_mapping(args.output)
    apis_remaining = [a for a in apis_to_process if a not in existing_mapping]

    print(f"📌 Processing range: [{start_idx}, {end_idx}), total {len(apis_to_process)}")
    print(f"📌 Existing mappings: {len(existing_mapping)} (skipped)")
    print(f"📌 Pending: {len(apis_remaining)}")
    print(f"📌 Workers: {workers}")
    print(f"📌 LLM model: {args.model}")

    if not apis_remaining:
        print("✅ All APIs processed")
        return

    # Initialize LLM client.
    api_key = load_api_key(args.key_path)
    if not api_key:
        sys.exit(1)

    llm_client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    print_lock = Lock()
    mappings_lock = Lock()

    # Initialize result list (including existing mappings).
    all_mappings: List[Dict[str, str]] = []
    for tf_api, pt_api in existing_mapping.items():
        all_mappings.append({
            "tensorflow-api": tf_api,
            "pytorch-api": pt_api,
            "confidence": "",
            "reason": "existing mapping",
        })

    # Worker function.
    def process_api(tf_api: str) -> Dict[str, str]:
        pt_api, confidence, reason = query_llm_for_api(
            llm_client, tf_api, model=args.model,
            temperature=args.temperature, print_lock=print_lock,
        )
        with print_lock:
            emoji = "✅" if pt_api != "无对应实现" else "⏭️"
            print(f"  {emoji} {tf_api} → {pt_api} ({confidence})")
        time.sleep(args.delay)
        return {
            "tensorflow-api": tf_api,
            "pytorch-api": pt_api,
            "confidence": confidence,
            "reason": reason,
        }

    print(f"\n🚀 Start generating TF→PT mappings (concurrency={workers})...\n")
    start_time = time.time()
    completed = 0
    total = len(apis_remaining)

    if workers <= 1:
        for tf_api in apis_remaining:
            result = process_api(tf_api)
            with mappings_lock:
                all_mappings.append(result)
            completed += 1
            if completed % 20 == 0:
                save_mapping(args.output, all_mappings)
                print(f"  💾 Progress: {completed}/{total}, saved intermediate results")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_api = {
                executor.submit(process_api, tf_api): tf_api
                for tf_api in apis_remaining
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
                            "tensorflow-api": api_name,
                            "pytorch-api": "无对应实现",
                            "confidence": "unknown",
                            "reason": f"Processing error: {e}",
                        })

                completed += 1
                if completed % 30 == 0:
                    with mappings_lock:
                        save_mapping(args.output, all_mappings)
                    with print_lock:
                        print(f"  💾 Progress: {completed}/{total}, saved intermediate results")

    # Sort by TF API name and save.
    all_mappings.sort(key=lambda x: x["tensorflow-api"])
    save_mapping(args.output, all_mappings)

    elapsed = time.time() - start_time
    has_impl = sum(1 for m in all_mappings if m["pytorch-api"] != "无对应实现")
    no_impl = len(all_mappings) - has_impl

    print(f"\n{'=' * 80}")
    print("📊 Mapping generation completed")
    print(f"{'=' * 80}")
    print(f"  Total APIs: {len(all_mappings)}")
    print(f"  With implementation: {has_impl}")
    print(f"  No implementation: {no_impl}")
    print(f"  Runtime: {elapsed:.1f} s")
    print(f"  💾 Saved to: {args.output}")


if __name__ == "__main__":
    main()
