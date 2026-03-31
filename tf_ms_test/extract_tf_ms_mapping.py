#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3: Generate a TensorFlow -> MindSpore API mapping table with LLM.

Purpose:
- Read the TF API list from Step 1.5 output.
- For each TF API, call an LLM to find a functionally equivalent MindSpore API.
- Support concurrent LLM calls.
- Support resume from checkpoints.
- Output a CSV mapping table.

Usage:
    conda activate tf_env
    python tf_ms_test_1/extract_tf_ms_mapping.py

Output: tf_ms_test_1/data/tf_ms_mapping.csv
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

DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"
DEFAULT_WORKERS = 6


def load_api_key(key_path: str = DEFAULT_KEY_PATH) -> str:
    """Load the Alibaba Cloud API key."""
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
    - tf.keras.layers.XXX (capitalized) -> class-level
    - tf.keras.losses.XXX (capitalized) -> class-level
    - otherwise -> function-level
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
    level_example_ms = "mindspore.ops.relu" if api_level == "function" else "mindspore.nn.Conv2d"

    prompt = f"""You are a deep learning framework expert fluent in TensorFlow and MindSpore.

[Task]
Find a functionally equivalent MindSpore API for the following TensorFlow API.

[TensorFlow API]
{tf_api}

[API Level]
This is a **{level_desc}-level** API.
- If the original API is a function (e.g., {level_example_tf}), return the corresponding MindSpore function (e.g., {level_example_ms}).
- If the original API is a class (e.g., tf.keras.layers.Conv2D), return the corresponding MindSpore class (e.g., mindspore.nn.Conv2d).

[Requirements]
1. The MindSpore API must be functionally equivalent or very close to the TensorFlow API.
2. Prefer the API with the closest functionality and parameters.
3. If MindSpore has no equivalent API, return "no_implementation".
4. Return only one best API, not multiple candidates.
5. Prefer public APIs available on CPU; avoid internal, experimental, or backend-specific interfaces.

[MindSpore API Namespace Reference]
- Core functions: mindspore.xxx (e.g., mindspore.abs)
- Operator functions: mindspore.ops.xxx (e.g., mindspore.ops.add, mindspore.ops.matmul)
- NN layers (classes): mindspore.nn.XXX (e.g., mindspore.nn.Conv2d, mindspore.nn.ReLU, mindspore.nn.Dense)
- Tensor methods: mindspore.Tensor.xxx (e.g., mindspore.Tensor.add)
- Linear algebra: mindspore.ops.xxx or mindspore.scipy.linalg.xxx
- Random: mindspore.ops.standard_normal / mindspore.ops.uniform

[Common TensorFlow -> MindSpore Mappings]
- tf.math.abs ↔ mindspore.ops.abs (or mindspore.abs)
- tf.add / tf.math.add ↔ mindspore.ops.add
- tf.matmul / tf.linalg.matmul ↔ mindspore.ops.matmul
- tf.nn.relu ↔ mindspore.ops.relu (or mindspore.nn.ReLU class)
- tf.keras.layers.Dense ↔ mindspore.nn.Dense

[Output Format]
Return strictly the following JSON format, with no extra content:

```json
{{
    "tensorflow_api": "{tf_api}",
    "mindspore_api": "<MindSpore API name or 'no_implementation'>",
    "confidence": "<high/medium/low>",
    "reason": "<brief rationale or why no implementation>"
}}
```

Notes:
- mindspore_api must be the full API name (e.g., mindspore.ops.abs or mindspore.nn.Conv2d), or "no_implementation".
- mindspore_api must be a real MindSpore API; do not invent non-existent APIs.
- confidence reflects how confident you are in the mapping (high >85%, medium 40%-85%, low <40%).
- reason should be brief (1-2 sentences).
"""
    return prompt


def parse_llm_response(response: str) -> Tuple[str, str, str]:
    """
    Parse the LLM JSON response.

    Returns:
        (mindspore_api, confidence, reason)
    """
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            ms_api = data.get("mindspore_api", "no_implementation").strip()
            confidence = data.get("confidence", "unknown").strip()
            reason = data.get("reason", "").strip()
            return ms_api, confidence, reason
    except json.JSONDecodeError:
        pass

    if "no_implementation" in response:
        return "no_implementation", "unknown", "Parse failed but detected no_implementation"

    ms_pattern = r"(mindspore\.[a-zA-Z_][a-zA-Z0-9_\.]*)"
    matches = re.findall(ms_pattern, response)
    if matches:
        return matches[0], "unknown", "Extracted from response text"

    return "no_implementation", "unknown", "Parse failed"


def query_llm_for_api(
    client: OpenAI,
    tf_api: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_retries: int = 3,
    print_lock: Lock = None,
) -> Tuple[str, str, str]:
    """
    Call the LLM to get the corresponding MindSpore API.

    Returns:
        (mindspore_api, confidence, reason)
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
            return parse_llm_response(full_response)
        except Exception as e:
            with lock:
                print(f"  ⚠️ {tf_api} LLM call failed: {str(e)[:80]}, retry ({attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)

    return "no_implementation", "unknown", "All retries failed"


def load_existing_mapping(csv_path: str) -> Dict[str, str]:
    """Load existing mappings (for resume)."""
    if not os.path.exists(csv_path):
        return {}

    existing = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tf_api = row.get("tensorflow-api", "").strip()
                ms_api = row.get("mindspore-api", "").strip()
                if tf_api:
                    existing[tf_api] = ms_api
    except Exception:
        pass
    return existing


def save_mapping(csv_path: str, mappings: List[Dict[str, str]]) -> None:
    """Save mappings to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tensorflow-api", "mindspore-api", "confidence", "reason"])
        writer.writeheader()
        for mapping in mappings:
            writer.writerow(mapping)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: Generate a TensorFlow -> MindSpore API mapping with LLM"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(ROOT_DIR, "tf_ms_test_1", "data", "tf_apis_existing.json"),
        help="Filtered list of existing TF APIs"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "tf_ms_test_1", "data", "tf_ms_mapping.csv"),
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
        help="LLM temperature (default 0.1, lower is more deterministic)"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start processing from API index (0-indexed)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of APIs to process"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls in seconds"
    )

    args = parser.parse_args()
    workers = max(1, args.workers)

    print("=" * 80)
    print("Step 3: Generate a TensorFlow -> MindSpore API mapping with LLM")
    print("=" * 80)

    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        print("Please ensure tf_apis_existing.json has been generated")
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

    print(f"📌 Processing range: [{start_idx}, {end_idx}), total {len(apis_to_process)}")
    print(f"📌 Existing mappings: {len(existing_mapping)} (skipped)")
    print(f"📌 Remaining: {len(apis_remaining)}")
    print(f"📌 Worker threads: {workers}")
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

    all_mappings: List[Dict[str, str]] = []
    for tf_api, ms_api in existing_mapping.items():
        all_mappings.append(
            {
                "tensorflow-api": tf_api,
                "mindspore-api": ms_api,
                "confidence": "",
                "reason": "existing mapping",
            }
        )

    def process_api(tf_api: str) -> Dict[str, str]:
        ms_api, confidence, reason = query_llm_for_api(
            llm_client,
            tf_api,
            model=args.model,
            temperature=args.temperature,
            print_lock=print_lock,
        )
        with print_lock:
            emoji = "✅" if ms_api != "no_implementation" else "⏭️"
            print(f"  {emoji} {tf_api} → {ms_api} ({confidence})")
        time.sleep(args.delay)
        return {
            "tensorflow-api": tf_api,
            "mindspore-api": ms_api,
            "confidence": confidence,
            "reason": reason,
        }

    print(f"\n🚀 Start generating TF->MS mapping (workers={workers})...\n")
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
                                "mindspore-api": "no_implementation",
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
    has_impl = sum(1 for item in all_mappings if item["mindspore-api"] != "no_implementation")
    no_impl = len(all_mappings) - has_impl

    print(f"\n{'=' * 80}")
    print("📊 Mapping generation completed")
    print(f"{'=' * 80}")
    print(f"  Total APIs: {len(all_mappings)}")
    print(f"  With implementation: {has_impl}")
    print(f"  No implementation: {no_impl}")
    print(f"  Elapsed: {elapsed:.1f} s")
    print(f"  💾 Saved to: {args.output}")


if __name__ == "__main__":
    main()
