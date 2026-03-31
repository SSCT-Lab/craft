#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3: Generate PaddlePaddle -> PyTorch API mapping table using LLM

Function:
- Read the Paddle API list from Step 1/1.5 output
- For each Paddle API, call the LLM to find the functionally equivalent PyTorch API
- Support concurrent LLM calls and resume from checkpoints
- Output a CSV mapping table

Usage:
    conda activate tf_env
    python pd_pt_test/extract_pd_pt_mapping.py [--input data/pd_apis_existing.json] [--output data/pd_pt_mapping.csv] [--workers 6]

Output: pd_pt_test/data/pd_pt_mapping.csv
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
    """Load the Alibaba Cloud API key."""
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
    Determine the Paddle API level: function or class.

    Rules:
    - paddle.nn.Conv2D (starts with uppercase) -> class
    - paddle.nn.functional.relu -> function
    """
    parts = api_name.split(".")
    if len(parts) >= 2:
        last_part = parts[-1]
        if last_part and last_part[0].isupper():
            return "class"
    return "function"


def build_prompt_for_api(pd_api: str, api_level: str) -> str:
    """Build the LLM prompt for a single Paddle API."""
    level_desc = "function" if api_level == "function" else "class"
    level_example_pd = "paddle.nn.functional.relu" if api_level == "function" else "paddle.nn.Conv2D"
    level_example_pt = "torch.nn.functional.relu" if api_level == "function" else "torch.nn.Conv2d"

    prompt = f"""You are a deep learning framework expert in PaddlePaddle and PyTorch.

[Task]
For the following PaddlePaddle API, find the functionally equivalent PyTorch API.

[PaddlePaddle API]
{pd_api}

[API Level]
This is a **{level_desc}-level** API.
- If the original API is a function (e.g., {level_example_pd}), return the corresponding function in PyTorch (e.g., {level_example_pt}).
- If the original API is a class (e.g., paddle.nn.Conv2D), return the corresponding class in PyTorch (e.g., torch.nn.Conv2d).

[Requirements]
1. The returned PyTorch API must be functionally equivalent or very close to the PaddlePaddle API.
2. Prefer the API with the closest function and parameters.
3. If there is no functionally equivalent API in PyTorch, return "No corresponding implementation".
4. Return only one best API; do not return multiple candidates.

[PyTorch API Namespace Reference]
- Basic functions: torch.xxx (e.g., torch.abs, torch.add, torch.matmul)
- Neural network layers (classes): torch.nn.XXX (e.g., torch.nn.Conv1d, torch.nn.ReLU, torch.nn.Linear)
- Neural network functions: torch.nn.functional.xxx (e.g., torch.nn.functional.relu, torch.nn.functional.softmax)
- Linear algebra: torch.linalg.xxx (e.g., torch.linalg.det, torch.linalg.inv)
- Random: torch.xxx or torch.distributions.xxx
- Signal processing (FFT): torch.fft.xxx
- Image processing: torchvision.transforms.functional.xxx

[Common PaddlePaddle -> PyTorch Mapping Hints]
- paddle.nn.functional.xxx <-> torch.nn.functional.xxx (mostly direct)
- paddle.nn.XXX <-> torch.nn.XXX (mostly direct; note case: Paddle uses Conv2D vs PyTorch Conv2d)
- paddle.linalg.xxx <-> torch.linalg.xxx
- paddle.fft.xxx <-> torch.fft.xxx
- paddle.xxx (e.g., paddle.add, paddle.abs) <-> torch.xxx
- paddle.Tensor.xxx <-> torch.Tensor.xxx

[Output Format]
Return strictly the following JSON format, and nothing else:

```json
{{
    "paddle_api": "{pd_api}",
    "pytorch_api": "<PyTorch API name or 'No corresponding implementation'>",
    "confidence": "<high/medium/low>",
    "reason": "<brief reason for the mapping or why none exists>"
}}
```

Notes:
- The pytorch_api field must be the full API name (e.g., torch.abs or torch.nn.Conv2d), or "No corresponding implementation".
- The pytorch_api value must be a real PyTorch API name; do not fabricate non-existing APIs.
- confidence indicates your confidence in the equivalence (>=85% high, 40%-85% medium, <40% low).
- reason should be brief (1-2 sentences).
- Only map public Paddle APIs; for internal/experimental APIs, return "No corresponding implementation".
"""
    return prompt


def parse_llm_response(response: str) -> Tuple[str, str, str]:
    """
    Parse the LLM JSON response.

    Returns:
        (pytorch_api, confidence, reason)
    """
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            pt_api = data.get("pytorch_api", "No corresponding implementation").strip()
            confidence = data.get("confidence", "unknown").strip()
            reason = data.get("reason", "").strip()
            return pt_api, confidence, reason
    except json.JSONDecodeError:
        pass

    if "No corresponding implementation" in response:
        return "No corresponding implementation", "unknown", "Parse failed but detected No corresponding implementation"

    torch_pattern = r'(torch\.[a-zA-Z_][a-zA-Z0-9_\.]*)'
    matches = re.findall(torch_pattern, response)
    if matches:
        return matches[0], "unknown", "Extracted from response text"

    return "No corresponding implementation", "unknown", "Parse failed"


def query_llm_for_api(
    client: OpenAI,
    pd_api: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_retries: int = 3,
    print_lock: Lock = None,
) -> Tuple[str, str, str]:
    """
    Call the LLM to get the corresponding PyTorch API.

    Returns:
        (pytorch_api, confidence, reason)
    """
    lock = print_lock or Lock()
    api_level = determine_api_level(pd_api)
    prompt = build_prompt_for_api(pd_api, api_level)

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
                print(f"  ⚠️ {pd_api} LLM call failed: {str(e)[:80]}, retry ({attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)

    return "No corresponding implementation", "unknown", "All retries failed"


def load_existing_mapping(csv_path: str) -> Dict[str, str]:
    """Load existing mappings (for resume)."""
    if not os.path.exists(csv_path):
        return {}
    existing = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pd_api = row.get("paddle-api", "").strip()
                pt_api = row.get("pytorch-api", "").strip()
                if pd_api:
                    existing[pd_api] = pt_api
    except Exception:
        pass
    return existing


def save_mapping(csv_path: str, mappings: List[Dict[str, str]]):
    """Save mappings to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["paddle-api", "pytorch-api", "confidence", "reason"])
        writer.writeheader()
        for m in mappings:
            writer.writerow(m)


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Generate PaddlePaddle → PyTorch API mapping table using LLM"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(ROOT_DIR, "pd_pt_test", "data", "pd_apis_existing.json"),
        help="Filtered list of existing Paddle APIs file"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "pd_pt_test", "data", "pd_pt_mapping.csv"),
        help="Output CSV mapping file path"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"LLM concurrent worker count (default {DEFAULT_WORKERS})"
    )
    parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL,
        help=f"LLM model name (default {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--key-path", "-k", default=DEFAULT_KEY_PATH,
        help=f"API key filepath (default {DEFAULT_KEY_PATH})"
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.1,
        help="LLM temperature (default 0.1, lower is more deterministic)"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start processing from API index (0-based)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of APIs to process"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="API call delay in seconds"
    )

    args = parser.parse_args()
    workers = max(1, args.workers)

    print("=" * 80)
    print("Step 3: Generate PaddlePaddle → PyTorch API mapping table using LLM")
    print("=" * 80)

    # Load API list
    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        api_data = json.load(f)

    all_apis = [a["pd_api"] for a in api_data.get("apis", [])]
    print(f"📋 Loaded {len(all_apis)} Paddle APIs")

    # Determine processing range
    start_idx = args.start
    end_idx = start_idx + args.limit if args.limit else len(all_apis)
    end_idx = min(end_idx, len(all_apis))
    apis_to_process = all_apis[start_idx:end_idx]

    # Load existing mappings (resume)
    existing_mapping = load_existing_mapping(args.output)
    apis_remaining = [a for a in apis_to_process if a not in existing_mapping]

    print(f"📌 Processing range: [{start_idx}, {end_idx}), total {len(apis_to_process)} items")
    print(f"📌 Existing mappings: {len(existing_mapping)} items (skip)")
    print(f"📌 Remaining to process: {len(apis_remaining)} items")
    print(f"📌 concurrent worker count: {workers}")
    print(f"📌 LLM model: {args.model}")

    if not apis_remaining:
        print("✅ All APIs processed")
        return

    # Initialize LLM client
    api_key = load_api_key(args.key_path)
    if not api_key:
        sys.exit(1)

    llm_client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    print_lock = Lock()
    mappings_lock = Lock()

    # Initialize results list (including existing mappings)
    all_mappings: List[Dict[str, str]] = []
    for pd_api, pt_api in existing_mapping.items():
        all_mappings.append({
            "paddle-api": pd_api,
            "pytorch-api": pt_api,
            "confidence": "",
            "reason": "existing mapping",
        })

    # Processing function
    def process_api(pd_api: str) -> Dict[str, str]:
        pt_api, confidence, reason = query_llm_for_api(
            llm_client, pd_api, model=args.model,
            temperature=args.temperature, print_lock=print_lock,
        )
        with print_lock:
            emoji = "✅" if pt_api != "No corresponding implementation" else "⏭️"
            print(f"  {emoji} {pd_api} → {pt_api} ({confidence})")
        time.sleep(args.delay)
        return {
            "paddle-api": pd_api,
            "pytorch-api": pt_api,
            "confidence": confidence,
            "reason": reason,
        }

    print(f"\n🚀 Starting PD→PT mapping generation (concurrency={workers})...\n")
    start_time = time.time()
    completed = 0
    total = len(apis_remaining)

    if workers <= 1:
        for pd_api in apis_remaining:
            result = process_api(pd_api)
            with mappings_lock:
                all_mappings.append(result)
            completed += 1
            if completed % 20 == 0:
                save_mapping(args.output, all_mappings)
                print(f"  💾 Progress: {completed}/{total}, saved intermediate results")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_api = {
                executor.submit(process_api, pd_api): pd_api
                for pd_api in apis_remaining
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
                            "paddle-api": api_name,
                            "pytorch-api": "No corresponding implementation",
                            "confidence": "unknown",
                            "reason": f"processing error: {e}",
                        })

                completed += 1
                if completed % 30 == 0:
                    with mappings_lock:
                        save_mapping(args.output, all_mappings)
                    with print_lock:
                        print(f"  💾 Progress: {completed}/{total}, saved intermediate results")

    # Sort by API name then save
    all_mappings.sort(key=lambda x: x["paddle-api"])
    save_mapping(args.output, all_mappings)

    elapsed = time.time() - start_time
    has_impl = sum(1 for m in all_mappings if m["pytorch-api"] != "No corresponding implementation")
    no_impl = len(all_mappings) - has_impl

    print(f"\n{'=' * 80}")
    print("📊 Mapping generation completed")
    print(f"{'=' * 80}")
    print(f"  Total APIs: {len(all_mappings)}")
    print(f"  With corresponding implementation: {has_impl}")
    print(f"  No corresponding implementation: {no_impl}")
    print(f"  Elapsed time: {elapsed:.1f} s")
    print(f"  💾 Saved to: {args.output}")


if __name__ == "__main__":
    main()

