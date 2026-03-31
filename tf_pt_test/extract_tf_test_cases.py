#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2: Extract/generate standardized test cases from TensorFlow official test files with LLM

Features:
- Read the TF API list from Step 1
- For each API, read the corresponding test file content
- Call LLM to extract/generate standardized test cases
- Support concurrent LLM calls
- Output a structured test case set (JSON)

Usage:
    conda activate tf_env
    python tf_pt_test/extract_tf_test_cases.py [--input data/tf_apis_existing.json] [--output data/tf_test_cases.json] [--workers 4]

Output: tf_pt_test/data/tf_test_cases.json
"""

import os
import sys
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
DEFAULT_NUM_CASES = 5
MAX_FILE_CHARS = 8000  # Max test-file characters sent to LLM.


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

    print("❌ API key not found. Ensure aliyun.key exists or set DASHSCOPE_API_KEY")
    return ""


def read_test_file_content(tf_dir: str, source_file: str, max_chars: int = MAX_FILE_CHARS) -> str:
    """
    Read test file content (truncate to max_chars to control token usage).

    Prefer keeping file header (imports/classes) and test methods.
    """
    filepath = os.path.join(tf_dir, source_file)
    if not os.path.exists(filepath):
        return ""

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return ""

    if len(content) <= max_chars:
        return content

    # If the file is too long, keep the header and useful middle parts.
    # Keep 60% for the header and 40% for test methods.
    head_chars = int(max_chars * 0.6)
    tail_chars = max_chars - head_chars

    head = content[:head_chars]
    # Find test methods in the middle.
    remaining = content[head_chars:]
    tail_start = max(0, len(remaining) // 4)
    tail = remaining[tail_start:tail_start + tail_chars]

    return head + "\n\n# ... (middle part omitted) ...\n\n" + tail


def build_extraction_prompt(tf_api: str, file_content: str, num_cases: int = DEFAULT_NUM_CASES) -> str:
    """Build the LLM prompt for test case extraction."""
    prompt = f"""You are a deep learning framework testing expert specializing in TensorFlow operators/APIs.

## Task
From the TensorFlow official test file below, extract or generate test cases for API `{tf_api}` (extract first, generate if insufficient).
Requirements:
- If the test file has more than {num_cases} cases, extract **all** usable cases.
- If it has fewer than {num_cases}, **generate additional cases** until at least {num_cases} are available.

## Test file content
```python
{file_content}
```

## Output requirements
Output strictly in JSON format:

```json
{{
    "api": "{tf_api}",
    "is_class_api": false,
    "test_cases": [
        {{
            "description": "basic functionality",
            "inputs": {{
                "x": {{"shape": [2, 3], "dtype": "float32"}},
                "axis": 1
            }}
        }},
        {{
            "description": "edge case - empty tensor",
            "inputs": {{
                "x": {{"shape": [0, 3], "dtype": "float32"}}
            }}
        }}
    ]
}}
```

## Rules
1. `is_class_api`: determine whether the API is class-based (e.g., `tf.keras.layers.Dense`). If it is a function (e.g., `tf.nn.relu`), set to `false`.
2. In each test case `inputs`:
   - **Tensor params** must use `{{"shape": [...], "dtype": "..."}}` (dtype without the tf. prefix, e.g., "float32", "int64", "bool").
   - **Scalar params** use numeric values, e.g., `"axis": 1`, `"keepdims": true`.
   - **String params** use strings, e.g., `"padding": "SAME"`.
   - **List params** use lists, e.g., `"strides": [1, 1]`.
3. Test cases should cover:
   - Basic functionality (normal inputs)
   - Different dtypes (float32, float64, int32, etc.)
   - Different shapes (1D, 2D, higher dimensions)
   - Edge cases (empty tensors, single element, very large/small values if present)
4. Prefer extracting from the test file. If not possible, generate based on experience. When extracting, capture real input data and parameters from the tests; do not fabricate unreasonable cases. When generating, derive similar variants from existing cases.
5. Ensure shapes are reasonable (no dimension > 10) and dtypes are supported by TensorFlow.
6. **Do not** include markdown or extra text; output pure JSON only.
"""
    return prompt


def extract_test_cases_for_api(
    llm_client: OpenAI,
    model: str,
    tf_api: str,
    file_content: str,
    num_cases: int = DEFAULT_NUM_CASES,
    print_lock: Lock = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Call LLM to extract test cases for a single TF API.

    Args:
        llm_client: OpenAI client
        model: LLM model name
        tf_api: TF API name
        file_content: Test file content
        num_cases: Number of test cases to extract
        print_lock: Thread-safe print lock
        max_retries: Max retries

    Returns:
        Extraction result dict
    """
    lock = print_lock or Lock()
    prompt = build_extraction_prompt(tf_api, file_content, num_cases)

    for attempt in range(max_retries):
        try:
            completion = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a deep learning testing expert who extracts standardized test cases from TensorFlow tests. Output JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4096,
            )

            raw_response = completion.choices[0].message.content.strip()
            time.sleep(0.5)  # Avoid rate limiting.

            # Parse JSON.
            result = _parse_json_response(raw_response)
            if result and "test_cases" in result:
                if len(result["test_cases"]) < num_cases:
                    with lock:
                        print(
                            f"  ⚠️ {tf_api} insufficient cases ({len(result['test_cases'])}/{num_cases}), accepted LLM output"
                        )
                with lock:
                    print(f"  ✅ {tf_api} → {len(result['test_cases'])} test cases")
                return result

            with lock:
                print(f"  ⚠️ {tf_api} invalid format, retry ({attempt + 1}/{max_retries})")

        except Exception as e:
            with lock:
                print(f"  ❌ {tf_api} LLM call failed: {str(e)[:80]}, retry ({attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)

    # All retries failed, return default test case.
    with lock:
        print(f"  ❌ {tf_api} failed, using default test case")
    return _default_test_case(tf_api)


def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse LLM JSON with tolerance."""
    # Remove markdown code fences.
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to extract a JSON object.
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def _default_test_case(tf_api: str) -> Dict[str, Any]:
    """Generate a default test case when extraction fails."""
    return {
        "api": tf_api,
        "is_class_api": False,
        "test_cases": [
            {
                "description": "default test case",
                "inputs": {
                    "x": {"shape": [2, 3], "dtype": "float32"}
                }
            }
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Use LLM to extract test cases from TensorFlow test files"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(ROOT_DIR, "tf_pt_test", "data", "tf_apis_existing.json"),
        help="Path to the TF API list file from Step 1"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "tf_pt_test", "data", "tf_test_cases.json"),
        help="Output test case file path"
    )
    parser.add_argument(
        "--tf-dir",
        default=os.path.join(ROOT_DIR, "tf_testcases"),
        help="tf_testcases directory path"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"LLM concurrent workers (default {DEFAULT_WORKERS})"
    )
    parser.add_argument(
        "--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES,
        help=f"Test cases per API (default {DEFAULT_NUM_CASES})"
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
        "--start", type=int, default=0,
        help="Start from which API (0-indexed, for resume)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max APIs to process"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="LLM call delay seconds (default 0.5)"
    )

    args = parser.parse_args()
    workers = max(1, args.workers)

    print("=" * 80)
    print("Step 2: Use LLM to extract test cases from TensorFlow test files")
    print("=" * 80)

    # Load API list.
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        print("Please run Step 1: python tf_pt_test/extract_tf_apis.py")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        api_data = json.load(f)

    all_apis = api_data.get("apis", [])
    print(f"📋 Loaded {len(all_apis)} TF APIs")

    # Determine processing range.
    start_idx = args.start
    end_idx = start_idx + args.limit if args.limit else len(all_apis)
    end_idx = min(end_idx, len(all_apis))
    apis_to_process = all_apis[start_idx:end_idx]

    print(f"📌 Processing range: [{start_idx}, {end_idx}), total {len(apis_to_process)}")
    print(f"📌 Workers: {workers}")
    print(f"📌 Cases per API: {args.num_cases}")
    print(f"📌 LLM model: {args.model}")

    # Initialize LLM client.
    api_key = load_api_key(args.key_path)
    if not api_key:
        sys.exit(1)

    llm_client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    print_lock = Lock()
    results: Dict[str, Any] = {}

    # Load existing results (resume supported).
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            results = existing_data.get("test_cases", {})
            print(f"📂 Loaded existing results: {len(results)} APIs' test cases")
        except Exception:
            pass

    # Filter out processed APIs.
    apis_remaining = [a for a in apis_to_process if a["tf_api"] not in results]
    print(
        f"📌 Pending: {len(apis_remaining)} (skipped {len(apis_to_process) - len(apis_remaining)} processed)"
    )

    if not apis_remaining:
        print("✅ All APIs processed")
        return

    # Prepare file content cache (group by source_file).
    file_content_cache: Dict[str, str] = {}

    def get_file_content(source_file: str) -> str:
        if source_file not in file_content_cache:
            file_content_cache[source_file] = read_test_file_content(
                args.tf_dir, source_file
            )
        return file_content_cache[source_file]

    # Preload file content.
    source_files_raw = [a["source_file"] for a in apis_remaining]
    print(f"\n📖 Raw test file count: {len(source_files_raw)}")
    print(source_files_raw[:20])
    source_files = set(source_files_raw)
    print(f"📖 Preloaded {len(source_files)} test files...")
    for sf in source_files:
        get_file_content(sf)

    # Concurrent processing.
    def process_api(api_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        tf_api = api_info["tf_api"]
        file_content = get_file_content(api_info["source_file"])
        result = extract_test_cases_for_api(
            llm_client, args.model, tf_api, file_content,
            num_cases=args.num_cases, print_lock=print_lock,
        )
        result["source_file"] = api_info["source_file"]
        result["category"] = api_info["category"]
        time.sleep(args.delay)
        return tf_api, result

    print(f"\n🚀 Start extracting test cases (concurrency={workers})...\n")
    start_time = time.time()
    completed = 0
    total = len(apis_remaining)

    if workers <= 1:
        for api_info in apis_remaining:
            tf_api, result = process_api(api_info)
            results[tf_api] = result
            completed += 1
            if completed % 10 == 0:
                _save_results(args.output, results)
                print(f"  💾 Progress: {completed}/{total}, saved intermediate results")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_api = {
                executor.submit(process_api, api_info): api_info["tf_api"]
                for api_info in apis_remaining
            }
            for future in as_completed(future_to_api):
                try:
                    tf_api, result = future.result()
                    results[tf_api] = result
                except Exception as e:
                    api_name = future_to_api[future]
                    with print_lock:
                        print(f"  ❌ {api_name} processing error: {e}")
                    results[api_name] = _default_test_case(api_name)

                completed += 1
                if completed % 20 == 0:
                    _save_results(args.output, results)
                    with print_lock:
                        print(f"  💾 Progress: {completed}/{total}, saved intermediate results")

    # Final save.
    _save_results(args.output, results)

    elapsed = time.time() - start_time
    total_cases = sum(
        len(v.get("test_cases", [])) for v in results.values()
    )

    print(f"\n{'=' * 80}")
    print("📊 Extraction completed")
    print(f"{'=' * 80}")
    print(f"  Total APIs: {len(results)}")
    print(f"  Total test cases: {total_cases}")
    print(f"  Average per API: {total_cases / max(1, len(results)):.1f} cases")
    print(f"  Runtime: {elapsed:.1f} s")
    print(f"  💾 Saved to: {args.output}")


def _save_results(output_path: str, results: Dict[str, Any]):
    """Save results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_data = {
        "total_apis": len(results),
        "extraction_time": datetime.now().isoformat(),
        "test_cases": results,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
