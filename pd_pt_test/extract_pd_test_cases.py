#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2: Use LLM to extract/generate standardized test cases from PaddlePaddle official test files

Function:
- Read the Paddle API list output from Step 1
- For each API, read the corresponding test file content
- Call the LLM to extract/generate standardized test cases
- Support concurrent LLM calls and resume from checkpoints
- Output a structured test case set (JSON)

Usage:
    conda activate tf_env
    python pd_pt_test/extract_pd_test_cases.py [--input data/pd_apis_existing.json] [--output data/pd_test_cases.json] [--workers 4]

Output: pd_pt_test/data/pd_test_cases.json
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
MAX_FILE_CHARS = 8000  # Max test file chars sent to LLM


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

    print("❌ API key not found. Ensure aliyun.key exists or set DASHSCOPE_API_KEY.")
    return ""


def read_test_file_content(pd_dir: str, source_file: str, max_chars: int = MAX_FILE_CHARS) -> str:
    """
    Read test file content (truncate to max_chars to control token usage).

    Paddle test files typically include:
    - import section
    - OpTest subclasses (setUp defines self.inputs, self.outputs, self.attrs)
    - unittest.TestCase subclasses (explicit paddle API calls)
    """
    filepath = os.path.join(pd_dir, source_file)
    if not os.path.exists(filepath):
        return ""

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return ""

    if len(content) <= max_chars:
        return content

    # If the file is too long, keep the head and a useful middle slice.
    head_chars = int(max_chars * 0.6)
    tail_chars = max_chars - head_chars

    head = content[:head_chars]
    remaining = content[head_chars:]
    tail_start = max(0, len(remaining) // 4)
    tail = remaining[tail_start:tail_start + tail_chars]

    return head + "\n\n# ... (middle omitted) ...\n\n" + tail


def build_extraction_prompt(pd_api: str, file_content: str, num_cases: int = DEFAULT_NUM_CASES) -> str:
    """Build the LLM prompt for extracting test cases."""
    prompt = f"""You are a deep learning framework testing expert who understands PaddlePaddle operators and APIs.

## Task
From the PaddlePaddle official test file below, extract or generate test cases for API `{pd_api}` (prefer extraction; generate only if insufficient).
Requirements:
- If the file contains more than {num_cases} cases, extract all available cases.
- If the file contains fewer than {num_cases} cases, extract what exists and generate more until you have at least {num_cases}.

## Paddle test file format
Paddle operator test files typically include:
1. **OpTest pattern**: test classes inherit `OpTest`, and `setUp()` defines:
   - `self.op_type`: operator name
   - `self.python_api`: corresponding Python API
   - `self.inputs`: input dict (e.g., `{{'X': numpy_array}}`)
   - `self.outputs`: expected outputs
   - `self.attrs`: operator attributes
2. **unittest pattern**: explicit calls to `paddle.xxx()` APIs
3. **Subclass variants**: many subclasses override `init_shape()` / `init_dtype()` etc.

Extract input shapes, dtypes, and other parameters from these definitions.

## Test file content
```python
{file_content}
```

## Output requirements
Output strict JSON using this format:

```json
{{
    "api": "{pd_api}",
    "is_class_api": false,
    "test_cases": [
        {{
            "description": "basic functionality test",
            "inputs": {{
                "x": {{"shape": [2, 3], "dtype": "float32"}},
                "axis": 1
            }}
        }},
        {{
            "description": "boundary values test - empty tensor",
            "inputs": {{
                "x": {{"shape": [0, 3], "dtype": "float32"}}
            }}
        }}
    ]
}}
```

## Rules
1. `is_class_api`: determine whether the API is a class (e.g., `paddle.nn.Conv2D`, capitalized class name). If it is a function (e.g., `paddle.nn.functional.relu`), set `false`.
2. In each test case `inputs` dict:
   - **Tensor parameters** must use `{{"shape": [...], "dtype": "..."}}` (dtype without the paddle. prefix, e.g., "float32", "int64", "bool")
   - **Scalar parameters** should be plain values (e.g., `"axis": 1`, `"keepdim": true`)
   - **String parameters** should be strings (e.g., `"padding": "SAME"`)
   - **List parameters** should be lists (e.g., `"strides": [1, 1]`)
   - For Paddle `self.inputs` parameter `X`, map it to `"x"`
3. Test cases should cover:
   - basic functionality (normal inputs)
   - different dtypes (float32, float64, int32, etc.)
   - different shapes (1D, 2D, higher dimensions)
   - boundary values (empty tensor, single element, extreme values, when present in the file)
4. Prefer extracting from the test file; only generate based on experience when extraction fails. When extracting, use real inputs/parameters from the file; do not fabricate unreasonable cases. When generating, create variations similar to existing cases.
5. Ensure shapes are reasonable (not too large; each dimension <= 10) and dtypes are supported by PaddlePaddle.
6. **Do not** include markdown or extra text; output JSON only.
"""
    return prompt


def extract_test_cases_for_api(
    llm_client: OpenAI,
    model: str,
    pd_api: str,
    file_content: str,
    num_cases: int = DEFAULT_NUM_CASES,
    print_lock: Lock = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Call the LLM to extract test cases for a single Paddle API.
    """
    lock = print_lock or Lock()
    prompt = build_extraction_prompt(pd_api, file_content, num_cases)

    for attempt in range(max_retries):
        try:
            completion = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a deep learning testing expert. Analyze PaddlePaddle test code and extract standardized test cases. Output JSON only."
                    },
                    {"role": "user", "content": prompt}
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
                            f"  ⚠️ {pd_api} case count insufficient ({len(result['test_cases'])}/{num_cases}); accepted LLM result"
                        )
                with lock:
                    print(f"  ✅ {pd_api} -> {len(result['test_cases'])} test cases")
                return result

            with lock:
                print(f"  ⚠️ {pd_api} returned invalid format; retry ({attempt + 1}/{max_retries})")

        except Exception as e:
            with lock:
                print(f"  ❌ {pd_api} LLM call failed: {str(e)[:80]}; retry ({attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)

    with lock:
        print(f"  ❌ {pd_api} failed after retries; using default test cases")
    return _default_test_case(pd_api)


def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse LLM JSON response (fault-tolerant)."""
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


def _default_test_case(pd_api: str) -> Dict[str, Any]:
    """Generate default test cases when extraction fails."""
    return {
        "api": pd_api,
        "is_class_api": False,
        "test_cases": [
            {
                "description": "defaultTest cases",
                "inputs": {
                    "x": {"shape": [2, 3], "dtype": "float32"}
                }
            }
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Use LLM to extract test cases from PaddlePaddle test files"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(ROOT_DIR, "pd_pt_test", "data", "pd_apis_existing.json"),
        help="Paddle API list filepath from Step 1/1.5"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "pd_pt_test", "data", "pd_test_cases.json"),
        help="Output test cases filepath"
    )
    parser.add_argument(
        "--pd-dir",
        default=os.path.join(ROOT_DIR, "testcases_pd"),
        help="testcases_pd directory path"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"LLM concurrent worker count (default {DEFAULT_WORKERS})"
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
        help=f"API key filepath (default {DEFAULT_KEY_PATH})"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start processing from API index (0-based, for resume)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of APIs to process"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="LLM call delay in seconds (default 0.5)"
    )

    args = parser.parse_args()
    workers = max(1, args.workers)

    print("=" * 80)
    print("Step 2: Use LLM to extract test cases from PaddlePaddle test files")
    print("=" * 80)

    # Load API list
    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        print("Please run Step 1 first: python pd_pt_test/extract_pd_apis.py")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        api_data = json.load(f)

    all_apis = api_data.get("apis", [])
    print(f"📋 Loaded {len(all_apis)} Paddle APIs")

    # Determine processing range
    start_idx = args.start
    end_idx = start_idx + args.limit if args.limit else len(all_apis)
    end_idx = min(end_idx, len(all_apis))
    apis_to_process = all_apis[start_idx:end_idx]

    print(f"📌 Processing range: [{start_idx}, {end_idx}), total {len(apis_to_process)}")
    print(f"📌 concurrent worker count: {workers}")
    print(f"📌 Cases per API: {args.num_cases}")
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
            print(f"📂 Loaded existing results: {len(results)} APIs")
        except Exception:
            pass

    # Filter out already processed APIs
    apis_remaining = [a for a in apis_to_process if a["pd_api"] not in results]
    print(f"📌 Remaining: {len(apis_remaining)} (skipped {len(apis_to_process) - len(apis_remaining)})")

    if not apis_remaining:
        print("✅ All APIs processed")
        return

    # Preload file contents
    file_content_cache: Dict[str, str] = {}

    def get_file_content(source_file: str) -> str:
        if source_file not in file_content_cache:
            file_content_cache[source_file] = read_test_file_content(
                args.pd_dir, source_file
            )
        return file_content_cache[source_file]

    source_files = set(a["source_file"] for a in apis_remaining)
    print(f"\n📖 Preloading {len(source_files)} test files...")
    for sf in source_files:
        get_file_content(sf)

    # Concurrent processing
    def process_api(api_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        pd_api = api_info["pd_api"]
        file_content = get_file_content(api_info["source_file"])
        result = extract_test_cases_for_api(
            llm_client, args.model, pd_api, file_content,
            num_cases=args.num_cases, print_lock=print_lock,
        )
        result["source_file"] = api_info["source_file"]
        time.sleep(args.delay)
        return pd_api, result

    print(f"\n🚀 Start extracting test cases (concurrency={workers})...\n")
    start_time = time.time()
    completed = 0
    total = len(apis_remaining)

    if workers <= 1:
        for api_info in apis_remaining:
            pd_api, result = process_api(api_info)
            results[pd_api] = result
            completed += 1
            if completed % 10 == 0:
                _save_results(args.output, results)
                print(f"  💾 Progress: {completed}/{total}, saved intermediate results")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_api = {
                executor.submit(process_api, api_info): api_info["pd_api"]
                for api_info in apis_remaining
            }
            for future in as_completed(future_to_api):
                try:
                    pd_api, result = future.result()
                    results[pd_api] = result
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

    # Final save
    _save_results(args.output, results)

    elapsed = time.time() - start_time
    total_cases = sum(
        len(v.get("test_cases", [])) for v in results.values()
    )

    print(f"\n{'=' * 80}")
    print("📊 Extraction completed")
    print(f"{'=' * 80}")
    print(f"  API total count: {len(results)}")
    print(f"  Test cases total count: {total_cases}")
    print(f"  Avg per API: {total_cases / max(1, len(results)):.1f} cases")
    print(f"  Elapsed time: {elapsed:.1f} s")
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

