# ./component/data/extract_tf_api_mapping.py
"""Extract PyTorch-to-TensorFlow API mappings using an LLM."""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import sys

# Add the project root to sys.path so component modules can be imported.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.migration.migrate_generate_tests import get_qwen_client

DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"

# LLM log directory
LOG_DIR = ROOT / "component" / "data" / "llm_logs"


def load_pytorch_apis(csv_path: Path) -> List[str]:
    """Load PyTorch API list from api_mappings.csv."""
    apis: List[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            api = row.get("pytorch-api", "").strip()
            if api:
                apis.append(api)
    return apis


def determine_api_level(api_name: str) -> str:
    """Determine API level: function or class.

    Rules:
    - torch.nn.XXX and capitalized -> class (e.g., torch.nn.Conv1d, torch.nn.ReLU)
    - torch.nn.functional.xxx -> function (e.g., torch.nn.functional.relu)
    - torch.xxx and lowercase -> function (e.g., torch.abs, torch.add)
    - torch.nn.utils.xxx -> function (e.g., torch.nn.utils.clip_grad_norm_)
    """
    parts = api_name.split(".")
    
    # torch.nn.functional.xxx -> function level
    if "functional" in api_name:
        return "function"
    
    # torch.nn.utils.xxx -> function level
    if "utils" in api_name:
        return "function"
    
    # torch.nn.XXX and capitalized last part -> class level
    if len(parts) >= 3 and parts[1] == "nn":
        last_part = parts[-1]
        if last_part and last_part[0].isupper():
            return "class"
    
    # Default to function level
    return "function"


def build_prompt_for_api(pytorch_api: str, api_level: str) -> str:
    """Build prompt text for a single PyTorch API."""

    level_desc = "function" if api_level == "function" else "class"
    level_example_pt = "torch.abs" if api_level == "function" else "torch.nn.Conv1d"
    level_example_tf = "tf.abs" if api_level == "function" else "tf.keras.layers.Conv1D"
    
    prompt = f"""You are a deep learning framework expert fluent in PyTorch and TensorFlow.

[Task]
Find the most functionally similar TensorFlow API for the following PyTorch API.

[PyTorch API]
{pytorch_api}

[API Level]
This is a **{level_desc}** API.
- If the original API is a function (e.g., {level_example_pt}), return the corresponding TensorFlow function (e.g., {level_example_tf}).
- If the original API is a class (e.g., torch.nn.Conv1d), return the corresponding TensorFlow class (e.g., tf.keras.layers.Conv1D).

[Requirements]
1. The TensorFlow API must match the original API level:
   - function to function (e.g., torch.abs -> tf.abs)
   - class to class (e.g., torch.nn.Conv1d -> tf.keras.layers.Conv1D)
2. Prefer the closest match in functionality and parameters.
3. If TensorFlow has no equivalent API, return "无对应实现".
4. Return only one best API, not multiple candidates.

[TensorFlow API Namespace Reference]
- Basic math: tf.xxx (e.g., tf.abs, tf.add, tf.matmul)
- Neural network layers (class): tf.keras.layers.XXX (e.g., tf.keras.layers.Dense, tf.keras.layers.Conv2D)
- Neural network functions: tf.nn.xxx (e.g., tf.nn.relu, tf.nn.softmax)
- Loss functions (class): tf.keras.losses.XXX (e.g., tf.keras.losses.BinaryCrossentropy)
- Loss functions (function): tf.keras.losses.xxx (e.g., tf.keras.losses.binary_crossentropy)
- Linear algebra: tf.linalg.xxx (e.g., tf.linalg.inv, tf.linalg.det)
- Random: tf.random.xxx (e.g., tf.random.normal, tf.random.uniform)
- Signal processing: tf.signal.xxx
- Image processing: tf.image.xxx

[Output Format]
Output strictly in the following JSON format with no extra text:

```json
{{
    "pytorch_api": "{pytorch_api}",
    "tensorflow_api": "<matching TensorFlow API or '无对应实现'>",
    "confidence": "<high/medium/low>",
    "reason": "<Briefly explain the mapping or why no equivalent exists>"
}}
```

Notes:
- The tensorflow_api field must be the full API name (e.g., tf.abs or tf.keras.layers.Conv1D) or "无对应实现".
- confidence reflects your confidence (high >= 85%, medium 40%-85%, low < 40%).
- tensorflow_api must be a real TensorFlow API name; do not invent APIs.
- reason should be brief (one or two sentences).
"""
    return prompt


def parse_llm_response(response: str) -> Tuple[str, str, str]:
    """Parse LLM JSON response.

    Returns:
        (tensorflow_api, confidence, reason)
    """
    try:
        # Try to extract JSON block
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            tf_api = data.get("tensorflow_api", "无对应实现").strip()
            confidence = data.get("confidence", "unknown").strip()
            reason = data.get("reason", "").strip()
            return tf_api, confidence, reason
    except json.JSONDecodeError:
        pass
    
    # If parsing fails, try a simple heuristic
    if "无对应实现" in response:
        return "无对应实现", "unknown", "Parsing failed but detected '无对应实现'"

    # Try to find an API starting with tf.
    import re
    tf_pattern = r'(tf\.[a-zA-Z_][a-zA-Z0-9_\.]*)'
    matches = re.findall(tf_pattern, response)
    if matches:
        return matches[0], "unknown", "Extracted from response text"

    return "无对应实现", "unknown", "Parsing failed"


def query_llm_for_api(
    client,
    pytorch_api: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.8,
    max_retries: int = 3,
) -> Tuple[str, str]:
    """Call the LLM to get a TensorFlow API mapping.

    Args:
        client: LLM client
        pytorch_api: PyTorch API name
        model: LLM model name
        temperature: model temperature (0.0-1.0), lower is more deterministic
        max_retries: max retry count

    Returns:
        (tensorflow_api, full_response)
    """
    api_level = determine_api_level(pytorch_api)
    prompt = build_prompt_for_api(pytorch_api, api_level)
    
    for attempt in range(max_retries):
        try:
            if hasattr(client, "chat"):
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1024,
                )
                full_response = resp.choices[0].message.content.strip()
            else:
                resp = client.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1024,
                )
                full_response = resp.choices[0].message.content.strip()
            
            tf_api, confidence, reason = parse_llm_response(full_response)
            return tf_api, full_response
            
        except Exception as e:
            print(f"[WARN] {pytorch_api} API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue

    return "无对应实现", f"[ERROR] Call failed after {max_retries} retries"


def process_single_api(
    client,
    pt_api: str,
    index: int,
    total: int,
    model: str,
    temperature: float,
    print_lock: Lock,
) -> Tuple[int, str, str, str, str]:
    """Process a single API mapping (for concurrent execution).

    This function is called by multiple threads; each thread processes one PyTorch API.

    Args:
        client: LLM client (thread-safe)
        pt_api: PyTorch API name
        index: API index (0-based)
        total: total API count
        model: LLM model name
        temperature: temperature
        print_lock: lock for thread-safe printing

    Returns:
        (index, pytorch_api, api_level, tensorflow_api, llm_response)
    """
    # 1. Determine API level (function or class)
    api_level = determine_api_level(pt_api)

    # 2. Use a lock to avoid interleaved output
    with print_lock:
        print(f"[INFO] Processing [{index + 1}/{total}] {pt_api} (level: {api_level})")

    # 3. Call LLM to get TensorFlow API mapping (concurrent hot path)
    tf_api, llm_response = query_llm_for_api(
        client,
        pt_api,
        model=model,
        temperature=temperature,
    )

    # 4. Use the lock again for printing
    with print_lock:
        print(f"       [{index + 1}/{total}] {pt_api} -> {tf_api}")

    # 5. Return result (include index for ordering)
    return index, pt_api, api_level, tf_api, llm_response


def save_llm_log(
    log_entries: List[dict],
    log_path: Path,
) -> None:
    """Save LLM logs to a file."""
    with log_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch to TensorFlow API Mapping - LLM Log\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total entries: {len(log_entries)}\n")
        f.write("=" * 80 + "\n\n")
        
        for entry in log_entries:
            f.write("-" * 60 + "\n")
            f.write(f"Index: {entry['index']}\n")
            f.write(f"PyTorch API: {entry['pytorch_api']}\n")
            f.write(f"API level: {entry['api_level']}\n")
            f.write(f"TensorFlow API (extracted): {entry['tensorflow_api']}\n")
            f.write(f"\n【LLM 完整输出】\n{entry['llm_response']}\n")
            f.write("-" * 60 + "\n\n")


def save_updated_csv(
    csv_path: Path,
    output_path: Path,
    api_mappings: List[Tuple[str, str]],
) -> None:
    """Save the updated CSV file."""
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pytorch-api", "tensorflow-api"])
        for pt_api, tf_api in api_mappings:
            writer.writerow([pt_api, tf_api])


def main():
    """CLI entry: batch extract PyTorch-to-TensorFlow API mappings."""
    parser = argparse.ArgumentParser(
        description="Extract PyTorch-to-TensorFlow API mappings with an LLM"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=str(ROOT / "component" / "data" / "api_mappings.csv"),
        help="Path to input api_mappings.csv",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(ROOT / "component" / "data" / "api_mappings.csv"),
        help="Path to output CSV (overwrites by default)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--key-path",
        "-k",
        default=DEFAULT_KEY_PATH,
        help="Path to API key file (default: aliyun.key)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (0-indexed, for resume)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of APIs to process (default: all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls (default: 0.5); no effect in concurrent mode",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.8,
        help="LLM temperature (0.0-1.0). Lower is more deterministic (default: 0.8)",
    )
    parser.add_argument(
        "--log-dir",
        default=str(LOG_DIR),
        help="Output directory for LLM logs",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=5,
        help="Number of worker threads (default: 5)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return

    print(f"[INFO] Loading PyTorch API list: {input_path}")
    pytorch_apis = load_pytorch_apis(input_path)
    
    if not pytorch_apis:
        print("[ERROR] No APIs found in CSV")
        return

    # 处理 start 和 limit 参数
    total_apis = len(pytorch_apis)
    start_idx = args.start
    end_idx = total_apis if args.limit is None else min(start_idx + args.limit, total_apis)
    
    apis_to_process = pytorch_apis[start_idx:end_idx]
    print(
        f"[INFO] Total APIs: {total_apis}; processing range: [{start_idx}, {end_idx}) => {len(apis_to_process)} APIs"
    )

    try:
        client = get_qwen_client(args.key_path)
        print(f"[INFO] LLM client initialized, model: {args.model}")
        print(f"[INFO] Worker threads: {args.workers}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM client: {e}")
        return

    # Store results in a dict to preserve order by index
    results_dict = {}
    # Lock for thread-safe printing
    print_lock = Lock()

    # If resuming mid-run, prefill earlier APIs
    if start_idx > 0:
        for i in range(start_idx):
            results_dict[i] = (pytorch_apis[i], "", "", "", "")

    # ==================== Concurrent processing ====================
    print(f"\n[INFO] Starting concurrent processing with {args.workers} threads...")
    start_time = time.time()
    
    # Create thread pool; max_workers sets concurrency
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Step 1: submit all tasks
        # future_to_index tracks each task's API index
        future_to_index = {}
        
        for i, pt_api in enumerate(apis_to_process, start=start_idx):
            # executor.submit() returns immediately; work runs asynchronously
            future = executor.submit(
                process_single_api,  # 要执行的函数
                client,              # 参数1: LLM 客户端（所有线程共享）
                pt_api,              # 参数2: 当前要处理的 PyTorch API
                i,                   # 参数3: 索引
                total_apis,          # 参数4: 总数
                args.model,          # 参数5: 模型名称
                args.temperature,    # 参数6: 温度参数
                print_lock,          # 参数7: 打印锁
            )
            # Track future to index mapping
            future_to_index[future] = i

        # Step 2: collect results
        # as_completed() yields futures in completion order (not submission order)
        completed = 0
        for future in as_completed(future_to_index):
            try:
                # future.result() blocks until the task completes
                index, pt_api, api_level, tf_api, llm_response = future.result()

                # Store in dict by index to preserve order
                results_dict[index] = (pt_api, api_level, tf_api, llm_response, index)
                completed += 1

                # Progress update every 10 tasks
                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = len(apis_to_process) - completed
                    eta = avg_time * remaining
                    with print_lock:
                        print(
                            f"\n[PROGRESS] Completed {completed}/{len(apis_to_process)}, "
                            f"elapsed {elapsed:.1f}s, ETA {eta:.1f}s\n"
                        )
            except Exception as e:
                # If a task fails, record it but continue
                index = future_to_index[future]
                pt_api = pytorch_apis[index]
                with print_lock:
                    print(f"[ERROR] Exception while processing {pt_api}: {e}")
                results_dict[index] = (pt_api, "unknown", "无对应实现", f"[ERROR] {e}", index)

    # All tasks done; report elapsed time
    elapsed_time = time.time() - start_time
    print(f"\n[INFO] Concurrent processing complete. Total time: {elapsed_time:.1f}s")
    # ==================== End concurrent processing ====================

    # If there are remaining APIs (due to limit), fill with empty values
    if end_idx < total_apis:
        for i in range(end_idx, total_apis):
            results_dict[i] = (pytorch_apis[i], "", "", "", i)

    # Build final results in index order
    api_mappings: List[Tuple[str, str]] = []
    log_entries: List[dict] = []
    
    for i in range(total_apis):
        if i in results_dict:
            pt_api, api_level, tf_api, llm_response, _ = results_dict[i]
            api_mappings.append((pt_api, tf_api))
            
            # Only log entries for processed APIs
            if start_idx <= i < end_idx:
                log_entries.append({
                    "index": i + 1,
                    "pytorch_api": pt_api,
                    "api_level": api_level if api_level else determine_api_level(pt_api),
                    "tensorflow_api": tf_api,
                    "llm_response": llm_response,
                })
        else:
            api_mappings.append((pytorch_apis[i], ""))

    # Save results
    save_updated_csv(input_path, output_path, api_mappings)
    print(f"[SUCCESS] API mappings saved to: {output_path}")

    # Save LLM logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"pt_tf_mapping_log_{timestamp}.txt"
    log_path = log_dir / log_filename
    save_llm_log(log_entries, log_path)
    print(f"[SUCCESS] LLM logs saved to: {log_path}")

    # Summary
    mapped_count = sum(1 for _, tf in api_mappings if tf and tf != "无对应实现")
    no_impl_count = sum(1 for _, tf in api_mappings if tf == "无对应实现")
    empty_count = sum(1 for _, tf in api_mappings if not tf)

    print("\n" + "=" * 50)
    print("[Summary]")
    print(f"  Total APIs: {total_apis}")
    print(f"  Processed this run: {len(apis_to_process)}")
    print(f"  Mapped: {mapped_count}")
    print(f"  No corresponding implementation: {no_impl_count}")
    print(f"  Unprocessed: {empty_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
