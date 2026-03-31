#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3: Build a TensorFlow → PaddlePaddle API mapping table using an LLM

Purpose:
- Read the TF API list output from Step 1.5
- For each TF API, query the LLM for a functionally equivalent PaddlePaddle API
- Support concurrent LLM calls
- Support resume from existing output
- Output a CSV mapping table

Usage:
    conda activate tf_env
    python tf_pd_test_1/extract_tf_pd_mapping.py

Output: tf_pd_test_1/data/tf_pd_mapping.csv
"""

import os
import sys
import csv
import json
import time
import re
import argparse
from typing import Dict, List, Tuple
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
        with open(key_file, "r", encoding="utf-8") as f:
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
    Determine the TF API level: function-level or class-level.

    Rules:
    - tf.keras.layers.XXX (capitalized) → class level
    - tf.keras.losses.XXX (capitalized) → class level
    - Otherwise → function level
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
    """Build the LLM prompt for a single TF API."""
    level_desc = "function" if api_level == "function" else "class"
    level_example_tf = "tf.nn.relu" if api_level == "function" else "tf.keras.layers.Conv2D"
    level_example_pd = "paddle.nn.functional.relu" if api_level == "function" else "paddle.nn.Conv2D"

    prompt = f"""You are an expert in TensorFlow and PaddlePaddle.

[Task]
Find the functionally equivalent PaddlePaddle API for the following TensorFlow API.

[TensorFlow API]
{tf_api}

[API Level]
This is a **{level_desc}-level** API.
- If the original API is a function (e.g., {level_example_tf}), return the corresponding PaddlePaddle function (e.g., {level_example_pd}).
- If the original API is a class (e.g., tf.keras.layers.Conv2D), return the corresponding PaddlePaddle class (e.g., paddle.nn.Conv2D).

[Requirements]
1. The returned PaddlePaddle API must be functionally equivalent or very close to the TensorFlow API.
2. Prefer the API with the closest functionality and parameters.
3. If PaddlePaddle truly has no equivalent API, return "无对应实现".
4. Return only one best API, not multiple candidates.

[PaddlePaddle API Namespace Reference]
- Basic functions: paddle.xxx (e.g., paddle.abs, paddle.add, paddle.matmul)
- Neural network layers (classes): paddle.nn.XXX (e.g., paddle.nn.Conv2D, paddle.nn.ReLU, paddle.nn.Linear)
- Neural network functions: paddle.nn.functional.xxx (e.g., paddle.nn.functional.relu, paddle.nn.functional.softmax)
- Linear algebra: paddle.linalg.xxx (e.g., paddle.linalg.det, paddle.linalg.inv)
- Random: paddle.rand / paddle.randn / paddle.randint, etc.
- Signal processing: paddle.fft.xxx
- Image processing: paddle.vision.transforms.functional.xxx

[Common TensorFlow → Paddle Mappings]
- tf.nn.xxx ↔ paddle.nn.functional.xxx (most function-level ops)
- tf.keras.layers.XXX ↔ paddle.nn.XXX (class-level layers; common names: Conv2D/BatchNorm2D)
- tf.linalg.xxx ↔ paddle.linalg.xxx
- tf.signal.xxx ↔ paddle.fft.xxx or paddle.signal.xxx (choose by semantics)
- tf.math.xxx / tf.xxx ↔ paddle.xxx

[Output Format]
Return strictly in the following JSON format, with no extra content:

```json
{{
    "tensorflow_api": "{tf_api}",
    "paddle_api": "<PaddlePaddle API name or '无对应实现'>",
    "confidence": "<high/medium/low>",
    "reason": "<brief mapping rationale or why no equivalent exists>"
}}
```

Notes:
- The paddle_api field should be a full API name (e.g., paddle.abs or paddle.nn.Conv2D), or "无对应实现".
- The paddle_api value must be a real PaddlePaddle API name; do not invent APIs.
- confidence indicates your confidence in the mapping (85%+ is high, 40%-85% is medium, below 40% is low).
- reason should be concise (1-2 sentences, not too long).
"""
    return prompt


def parse_llm_response(response: str) -> Tuple[str, str, str]:
    """
    Parse the LLM JSON response.

    Returns:
        (paddle_api, confidence, reason)
    """
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            pd_api = data.get("paddle_api", "无对应实现").strip()
            confidence = data.get("confidence", "unknown").strip()
            reason = data.get("reason", "").strip()
            return pd_api, confidence, reason
    except json.JSONDecodeError:
        pass

    if "无对应实现" in response:
        return "无对应实现", "unknown", "Parse failed, but detected '无对应实现'"

    paddle_pattern = r"(paddle\.[a-zA-Z_][a-zA-Z0-9_\.]*)"
    matches = re.findall(paddle_pattern, response)
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
    """
    Call the LLM to get the corresponding PaddlePaddle API.

    Returns:
        (paddle_api, confidence, reason)
    """
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

            pd_api, confidence, reason = parse_llm_response(full_response)
            return pd_api, confidence, reason

        except Exception as e:
            with lock:
                print(f"  ⚠️ {tf_api} LLM call failed: {str(e)[:80]}, retry ({attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)

    return "无对应实现", "unknown", "All retries failed"


def load_existing_mapping(csv_path: str) -> Dict[str, str]:
    """Load existing mapping results (for resume)."""
    if not os.path.exists(csv_path):
        return {}

    existing = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tf_api = row.get("tensorflow-api", "").strip()
                pd_api = row.get("paddle-api", "").strip()
                if tf_api:
                    existing[tf_api] = pd_api
    except Exception:
        pass
    return existing


def save_mapping(csv_path: str, mappings: List[Dict[str, str]]) -> None:
    """Save mapping results to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tensorflow-api", "paddle-api", "confidence", "reason"])
        writer.writeheader()
        for mapping in mappings:
            writer.writerow(mapping)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: Build a TensorFlow → PaddlePaddle API mapping table with an LLM"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(ROOT_DIR, "tf_pd_test_1", "data", "tf_apis_existing.json"),
        help="Filtered list of existing TF APIs"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "tf_pd_test_1", "data", "tf_pd_mapping.csv"),
        help="Output CSV mapping file path"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"LLM worker threads (default {DEFAULT_WORKERS})"
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
        help="Start index for processing APIs (0-indexed)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of APIs to process"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay seconds between API calls"
    )

    args = parser.parse_args()
    workers = max(1, args.workers)

    print("=" * 80)
    print("Step 3: Build a TensorFlow → PaddlePaddle API mapping table with an LLM")
    print("=" * 80)

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        print("Please ensure tf_apis_existing.json is generated first")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        api_data = json.load(f)

    all_apis = [item["tf_api"] for item in api_data.get("apis", []) if item.get("tf_api")]
    print(f"📋 Loaded {len(all_apis)} TF APIs")

    start_idx = args.start
    end_idx = start_idx + args.limit if args.limit else len(all_apis)
    end_idx = min(end_idx, len(all_apis))
    apis_to_process = all_apis[start_idx:end_idx]

    existing_mapping = load_existing_mapping(args.output)
    apis_remaining = [api for api in apis_to_process if api not in existing_mapping]

    print(f"📌 Range: [{start_idx}, {end_idx}), total {len(apis_to_process)}")
    print(f"📌 Existing mappings: {len(existing_mapping)} (skipped)")
    print(f"📌 Remaining: {len(apis_remaining)}")
    print(f"📌 Workers: {workers}")
    print(f"📌 LLM model: {args.model}")

    if not apis_remaining:
        print("✅ All APIs are processed")
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

    all_mappings: List[Dict[str, str]] = []
    for tf_api, pd_api in existing_mapping.items():
        all_mappings.append(
            {
                "tensorflow-api": tf_api,
                "paddle-api": pd_api,
                "confidence": "",
                "reason": "Existing mapping",
            }
        )

    def process_api(tf_api: str) -> Dict[str, str]:
        pd_api, confidence, reason = query_llm_for_api(
            llm_client,
            tf_api,
            model=args.model,
            temperature=args.temperature,
            print_lock=print_lock,
        )
        with print_lock:
            emoji = "✅" if pd_api != "无对应实现" else "⏭️"
            print(f"  {emoji} {tf_api} → {pd_api} ({confidence})")
        time.sleep(args.delay)
        return {
            "tensorflow-api": tf_api,
            "paddle-api": pd_api,
            "confidence": confidence,
            "reason": reason,
        }

    print(f"\n🚀 Start generating TF→PD mappings (workers={workers})...\n")
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
                        all_mappings.append(
                            {
                                "tensorflow-api": api_name,
                                "paddle-api": "无对应实现",
                                "confidence": "unknown",
                                "reason": f"Processing error: {e}",
                            }
                        )

                completed += 1
                if completed % 30 == 0:
                    with mappings_lock:
                        save_mapping(args.output, all_mappings)
                    with print_lock:
                        print(f"  💾 Progress: {completed}/{total}, saved intermediate results")

    all_mappings.sort(key=lambda item: item["tensorflow-api"])
    save_mapping(args.output, all_mappings)

    elapsed = time.time() - start_time
    has_impl = sum(1 for item in all_mappings if item["paddle-api"] != "无对应实现")
    no_impl = len(all_mappings) - has_impl

    print(f"\n{'=' * 80}")
    print("📊 Mapping generation complete")
    print(f"{'=' * 80}")
    print(f"  Total APIs: {len(all_mappings)}")
    print(f"  With equivalent implementation: {has_impl}")
    print(f"  No equivalent implementation: {no_impl}")
    print(f"  Elapsed: {elapsed:.1f} seconds")
    print(f"  💾 Saved to: {args.output}")


if __name__ == "__main__":
    main()
