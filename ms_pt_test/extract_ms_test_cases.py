#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2: Extract/generate standardized test cases from MindSpore official tests with an LLM

Purpose:
- Read the MS API list from Step 1
- For each API, read the corresponding test file content
- Use the LLM to extract/generate standardized test cases
- Support concurrent LLM calls
- Output a structured test case set (JSON)

Usage:
    conda activate tf_env
    python ms_pt_test/extract_ms_test_cases.py [--input data/ms_apis_existing.json] [--output data/ms_test_cases.json] [--workers 6]

Output: ms_pt_test/data/ms_test_cases.json
"""

import os
import sys
import io

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # Force line buffering and immediate flush to avoid missing console output
    # sys.stdout = io.TextIOWrapper(
    #     sys.stdout.buffer,
    #     encoding='utf-8',
    #     errors='replace',
    #     line_buffering=True,
    #     write_through=True,
    # )
    # sys.stderr = io.TextIOWrapper(
    #     sys.stderr.buffer,
    #     encoding='utf-8',
    #     errors='replace',
    #     line_buffering=True,
    #     write_through=True,
    # )

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
DEFAULT_NUM_CASES = 5
MAX_FILE_CHARS = 8000  # Max characters of test file sent to the LLM


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


def read_test_file_content(ms_dir: str, source_file: str, max_chars: int = MAX_FILE_CHARS) -> str:
    """
    Read test file content (truncate to max_chars to control token usage).
    Prefer keeping the file header (imports and class definitions) and test methods.
    """
    filepath = os.path.join(ms_dir, source_file)
    if not os.path.exists(filepath):
        return ""

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return ""

    if len(content) <= max_chars:
        return content

    # If the file is too long, keep the header and a middle slice of test methods
    head_chars = int(max_chars * 0.6)
    tail_chars = max_chars - head_chars

    head = content[:head_chars]
    remaining = content[head_chars:]
    tail_start = max(0, len(remaining) // 4)
    tail = remaining[tail_start:tail_start + tail_chars]

    return head + "\n\n# ... (middle omitted) ...\n\n" + tail


def build_extraction_prompt(ms_api: str, file_content: str, num_cases: int = DEFAULT_NUM_CASES) -> str:
    """Build the LLM prompt for extracting test cases."""
    prompt = f"""You are a deep learning framework testing expert, fluent in MindSpore operators/APIs.

## Task
From the MindSpore official test file below, extract or generate test cases for API `{ms_api}`.
Prefer extraction; generate only if needed.
Requirements:
- If the test file has > {num_cases} cases, extract **all usable** cases.
- If the test file has < {num_cases} cases, **supplement by generation** until at least {num_cases} cases.

## MindSpore API invocation patterns
MindSpore operators can be called in these ways:
1. **Primitive (class-style)**: create `P.Abs()` then call `op(input)` or `self.op(input)`
2. **Functional API**: call `F.abs(input)` or `ops.abs(input)` directly
3. **NN layer (class-style)**: create `nn.Conv2d(in_channels, out_channels, kernel_size)` then call `layer(input)`
4. **Tensor method**: call `tensor.add(other)` on a Tensor

Infer the type from `{ms_api}` and build test cases accordingly.

## Test file content
```python
{file_content}
```

## Output format
Return strict JSON in the following format:

```json
{{
    "api": "{ms_api}",
    "is_class_api": true,
    "init_params": {{}},
    "test_cases": [
        {{
            "description": "basic functionality",
            "inputs": {{
                "x": {{"shape": [2, 3], "dtype": "float32"}},
                "axis": 1
            }}
        }},
        {{
            "description": "different dtypes",
            "inputs": {{
                "x": {{"shape": [3, 4], "dtype": "float64"}}
            }}
        }}
    ]
}}
```

## Rules
1. `is_class_api`: whether the API is class-based.
   - Primitive ops (e.g., `mindspore.ops.Abs`, `mindspore.ops.Conv2D`) -> `true`
   - NN layers (e.g., `mindspore.nn.Conv2d`, `mindspore.nn.BatchNorm2d`) -> `true`
   - Functional APIs (e.g., `mindspore.ops.abs`, `mindspore.ops.relu`) -> `false`
   - Tensor methods (e.g., `mindspore.Tensor.add`) -> `false`

2. `init_params` (class APIs only): initialization parameters.
   - For Primitive ops like `P.Abs()` with no params, use `{}`
   - For NN layers like `nn.Conv2d(3, 64, 3)`, use `{"in_channels": 3, "out_channels": 64, "kernel_size": 3}`
   - For Primitive ops that require params like `P.Conv2D(out_channel=64, kernel_size=3)`, record them

3. In each test case `inputs`:
   - **Tensor params** must be `{"shape": [...], "dtype": "..."}` (dtype without mindspore. prefix, e.g., "float32", "int64", "bool")
   - **Scalar params** use values directly, e.g., `"axis": 1`, `"keepdims": true`
   - **String params** use strings directly
   - **List params** use lists directly

4. Test cases should cover:
   - Basic functionality (valid inputs)
   - Different dtypes (float32, float64, int32, etc.)
   - Different shapes (1D, 2D, higher dims)
   - Edge cases (empty tensors, single element, extreme values, if present in file)

5. Prefer real cases from the test file; only generate when necessary

6. Ensure shapes are reasonable (no dimension > 10) and dtypes are standard

7. **Do not** include markdown or extra text; output pure JSON only
"""
    return prompt


def extract_test_cases_for_api(
    llm_client: OpenAI,
    model: str,
    ms_api: str,
    file_content: str,
    num_cases: int = DEFAULT_NUM_CASES,
    print_lock: Lock = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Call the LLM to extract test cases for a single MS API."""
    lock = print_lock or Lock()
    prompt = build_extraction_prompt(ms_api, file_content, num_cases)

    for attempt in range(max_retries):
        try:
            completion = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a deep learning testing expert who analyzes MindSpore tests and "
                            "extracts standardized test cases. Output JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=4096,
            )

            raw_response = completion.choices[0].message.content.strip()
            time.sleep(0.5)

            result = _parse_json_response(raw_response)
            if result and "test_cases" in result:
                if len(result["test_cases"]) < num_cases:
                    with lock:
                        print(
                            f"  ⚠️ {ms_api} insufficient cases"
                            f" ({len(result['test_cases'])}/{num_cases}); accepted LLM output"
                        )
                with lock:
                    print(f"  ✅ {ms_api} -> {len(result['test_cases'])} test cases")
                return result

            with lock:
                print(
                    f"  ⚠️ {ms_api} invalid response format, retry ({attempt + 1}/{max_retries})"
                )

        except Exception as e:
            with lock:
                print(
                    f"  ❌ {ms_api} LLM call failed: {str(e)[:80]}, "
                    f"retry ({attempt + 1}/{max_retries})"
                )
            time.sleep(2 ** attempt)

    with lock:
        print(f"  ❌ {ms_api} failed; using default test cases")
    return _default_test_case(ms_api)


def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse LLM JSON response with tolerance."""
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def _default_test_case(ms_api: str) -> Dict[str, Any]:
    """Generate default test case for APIs that cannot be extracted."""
    # Infer whether the API is class-based from its name
    parts = ms_api.split(".")
    last_part = parts[-1] if parts else ""
    is_class = last_part and last_part[0].isupper()

    return {
        "api": ms_api,
        "is_class_api": is_class,
        "init_params": {},
        "test_cases": [
            {
                "description": "default test case",
                "inputs": {
                    "x": {"shape": [2, 3], "dtype": "float32"},
                },
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Extract test cases from MindSpore tests with an LLM"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(ROOT_DIR, "ms_pt_test", "data", "ms_apis_existing.json"),
        help="Path to MS API list from Step 1",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "ms_pt_test", "data", "ms_test_cases.json"),
        help="Output test case file path",
    )
    parser.add_argument(
        "--ms-dir",
        default=os.path.join(ROOT_DIR, "testcases_ms"),
        help="Path to testcases_ms directory",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"LLM worker threads (default {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES,
        help=f"Test cases per API (default {DEFAULT_NUM_CASES})",
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
        "--start", type=int, default=0,
        help="Start from API index (0-indexed, for resume)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of APIs to process",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between LLM calls in seconds (default 0.5)",
    )

    args = parser.parse_args()
    workers = max(1, args.workers)

    print("=" * 80)
    print("Step 2: Extract test cases from MindSpore tests with an LLM")
    print("=" * 80)

    # Load API list
    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        print("Please run Step 1: python ms_pt_test/extract_ms_apis.py")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        api_data = json.load(f)

    all_apis = api_data.get("apis", [])
    print(f"📋 Loaded {len(all_apis)} MS APIs")

    # Determine processing range
    start_idx = args.start
    end_idx = start_idx + args.limit if args.limit else len(all_apis)
    end_idx = min(end_idx, len(all_apis))
    apis_to_process = all_apis[start_idx:end_idx]

    print(f"📌 Range: [{start_idx}, {end_idx}), total {len(apis_to_process)}")
    print(f"📌 Worker threads: {workers}")
    print(f"📌 Test cases per API: {args.num_cases}")
    print(f"📌 LLM model: {args.model}")

    # Initialize LLM client
    api_key = load_api_key(args.key_path)
    if not api_key:
        sys.exit(1)

    llm_client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    print_lock = Lock()
    results: Dict[str, Any] = {}

    # Load existing results (resume supported)
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            results = existing_data.get("test_cases", {})
            print(f"📂 Loaded existing results: {len(results)} API test cases")
        except Exception:
            pass

    # Filter already processed APIs
    apis_remaining = [a for a in apis_to_process if a["ms_api"] not in results]
    print(
        f"📌 Remaining: {len(apis_remaining)} "
        f"(skipped {len(apis_to_process) - len(apis_remaining)} processed)"
    )

    if not apis_remaining:
        print("✅ All APIs have been processed")
        return

    # Prepare file content cache
    file_content_cache: Dict[str, str] = {}

    def get_file_content(source_file: str) -> str:
        if source_file not in file_content_cache:
            file_content_cache[source_file] = read_test_file_content(
                args.ms_dir, source_file
            )
        return file_content_cache[source_file]

    # Preload file content
    source_files = set(a["source_file"] for a in apis_remaining)
    print(f"\n📖 Preloading {len(source_files)} test files...")
    for sf in source_files:
        get_file_content(sf)

    # Concurrent processing
    def process_api(api_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        ms_api = api_info["ms_api"]
        file_content = get_file_content(api_info["source_file"])
        result = extract_test_cases_for_api(
            llm_client, args.model, ms_api, file_content,
            num_cases=args.num_cases, print_lock=print_lock,
        )
        result["source_file"] = api_info["source_file"]
        result["api_type"] = api_info.get("api_type", "ops")
        time.sleep(args.delay)
        return ms_api, result

    print(f"\n🚀 Start extracting test cases (workers={workers})...\n")
    start_time = time.time()
    completed = 0
    total = len(apis_remaining)

    if workers <= 1:
        for api_info in apis_remaining:
            ms_api, result = process_api(api_info)
            results[ms_api] = result
            completed += 1
            if completed % 10 == 0:
                _save_results(args.output, results)
                print(f"  💾 Progress: {completed}/{total}, intermediate results saved")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_api = {
                executor.submit(process_api, api_info): api_info["ms_api"]
                for api_info in apis_remaining
            }
            for future in as_completed(future_to_api):
                try:
                    ms_api, result = future.result()
                    results[ms_api] = result
                except Exception as e:
                    api_name = future_to_api[future]
                    with print_lock:
                        print(f"  ❌ {api_name} processing error: {e}")
                    results[api_name] = _default_test_case(api_name)

                completed += 1
                if completed % 20 == 0:
                    _save_results(args.output, results)
                    with print_lock:
                        print(f"  💾 Progress: {completed}/{total}, intermediate results saved")

    # Final save
    _save_results(args.output, results)

    elapsed = time.time() - start_time
    total_cases = sum(len(v.get("test_cases", [])) for v in results.values())

    print(f"\n{'=' * 80}")
    print(f"📊 Extraction complete")
    print(f"{'=' * 80}")
    print(f"  Total APIs: {len(results)}")
    print(f"  Total test cases: {total_cases}")
    print(f"  Avg per API: {total_cases / max(1, len(results)):.1f} cases")
    print(f"  Elapsed: {elapsed:.1f} seconds")
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
