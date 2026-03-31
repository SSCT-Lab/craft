# ./component/data/extract_pd_api_mapping.py
"""Extract PyTorch-to-PaddlePaddle API mappings using an LLM."""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

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


def load_csv_data(csv_path: Path) -> Tuple[List[str], List[dict]]:
    """Load full data from api_mappings.csv.

    Returns:
        (fieldnames, rows) - fieldnames list and all rows
    """
    rows: List[dict] = []
    fieldnames: List[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        for row in reader:
            rows.append(dict(row))
    return fieldnames, rows


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
    level_example_pd = "paddle.abs" if api_level == "function" else "paddle.nn.Conv1D"
    
    prompt = f"""You are a deep learning framework expert fluent in PyTorch and PaddlePaddle.

[Task]
Find the functionally equivalent PaddlePaddle API for the following PyTorch API.

[PyTorch API]
{pytorch_api}

[API Level]
This is a **{level_desc}** API.
- If the original API is a function (e.g., {level_example_pt}), return the corresponding PaddlePaddle function (e.g., {level_example_pd}).
- If the original API is a class (e.g., torch.nn.Conv1d), return the corresponding PaddlePaddle class (e.g., paddle.nn.Conv1D).

[Requirements]
1. The PaddlePaddle API must match the original API level:
   - function to function (e.g., torch.abs -> paddle.abs)
   - class to class (e.g., torch.nn.Conv1d -> paddle.nn.Conv1D)
2. Prefer the closest match in functionality and parameters.
3. If PaddlePaddle has no equivalent API, return "无对应实现".
4. Return only one best API, not multiple candidates.

[PaddlePaddle API Namespace Reference]
- Basic math: paddle.xxx (e.g., paddle.abs, paddle.add, paddle.matmul)
- Neural network layers (class): paddle.nn.XXX (e.g., paddle.nn.Linear, paddle.nn.Conv2D)
- Neural network functions: paddle.nn.functional.xxx (e.g., paddle.nn.functional.relu, paddle.nn.functional.softmax)
- Loss functions (class): paddle.nn.XXX (e.g., paddle.nn.CrossEntropyLoss, paddle.nn.MSELoss)
- Loss functions (function): paddle.nn.functional.xxx (e.g., paddle.nn.functional.cross_entropy)
- Linear algebra: paddle.linalg.xxx (e.g., paddle.linalg.inv, paddle.linalg.det)
- Random: paddle.rand, paddle.randn, paddle.randint, etc.
- Tensor ops: paddle.xxx (e.g., paddle.reshape, paddle.concat, paddle.squeeze)
- Distributed: paddle.distributed.xxx
- Vision: paddle.vision.xxx

[Output Format]
Output strictly in the following JSON format with no extra text:

```json
{{
    "pytorch_api": "{pytorch_api}",
    "paddle_api": "<matching PaddlePaddle API or '无对应实现'>",
    "confidence": "<high/medium/low>",
    "reason": "<Briefly explain the mapping or why no equivalent exists>"
}}
```

Notes:
- The paddle_api field must be the full API name (e.g., paddle.abs or paddle.nn.Conv1D) or "无对应实现".
- confidence reflects your confidence (high >= 85%, medium 40%-85%, low < 40%).
- paddle_api must be a real PaddlePaddle API name; do not invent APIs.
- reason should be brief (one or two sentences).
"""
    return prompt


def parse_llm_response(response: str) -> Tuple[str, str, str]:
    """Parse LLM JSON response.

    Returns:
        (paddle_api, confidence, reason)
    """
    try:
        # Try to extract JSON block
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
    
    # If parsing fails, try a simple heuristic
    if "无对应实现" in response:
        return "无对应实现", "unknown", "Parsing failed but detected '无对应实现'"

    # Try to find an API starting with paddle.
    import re
    pd_pattern = r'(paddle\.[a-zA-Z_][a-zA-Z0-9_\.]*)'
    matches = re.findall(pd_pattern, response)
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
    """Call the LLM to get a PaddlePaddle API mapping.

    Args:
        client: LLM client
        pytorch_api: PyTorch API name
        model: LLM model name
        temperature: model temperature (0.0-1.0), lower is more deterministic
        max_retries: max retry count

    Returns:
        (paddle_api, full_response)
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
            
            pd_api, confidence, reason = parse_llm_response(full_response)
            return pd_api, full_response
            
        except Exception as e:
            print(f"[WARN] API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue

    return "无对应实现", f"[ERROR] Call failed after {max_retries} retries"


def save_llm_log(
    log_entries: List[dict],
    log_path: Path,
) -> None:
    """Save LLM logs to a file."""
    with log_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch to PaddlePaddle API Mapping - LLM Log\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total entries: {len(log_entries)}\n")
        f.write("=" * 80 + "\n\n")
        
        for entry in log_entries:
            f.write("-" * 60 + "\n")
            f.write(f"Index: {entry['index']}\n")
            f.write(f"PyTorch API: {entry['pytorch_api']}\n")
            f.write(f"API level: {entry['api_level']}\n")
            f.write(f"PaddlePaddle API (extracted): {entry['paddle_api']}\n")
            f.write(f"\n【LLM 完整输出】\n{entry['llm_response']}\n")
            f.write("-" * 60 + "\n\n")


def save_updated_csv(
    csv_path: Path,
    output_path: Path,
    api_mappings: List[Tuple[str, str]],
) -> None:
    """Save updated CSV; keep original columns and update/add paddle-api."""
    # Read full original data
    fieldnames, rows = load_csv_data(csv_path)

    # Ensure paddle-api column exists
    target_col = "paddle-api"
    if target_col not in fieldnames:
        fieldnames.append(target_col)

    # Build pytorch-api -> paddle-api mapping
    mapping_dict = {pt_api: pd_api for pt_api, pd_api in api_mappings}

    # Update each row
    for row in rows:
        pt_api = row.get("pytorch-api", "").strip()
        if pt_api in mapping_dict:
            row[target_col] = mapping_dict[pt_api]

    # Write updated data
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """CLI entry: batch extract PyTorch-to-PaddlePaddle API mappings."""
    parser = argparse.ArgumentParser(
        description="Extract PyTorch-to-PaddlePaddle API mappings with an LLM"
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
        help="Delay between API calls (default: 0.5)",
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
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM client: {e}")
        return

    # Store results
    api_mappings: List[Tuple[str, str]] = []
    log_entries: List[dict] = []

    # If resuming mid-run, prefill earlier APIs
    if start_idx > 0:
        for i in range(start_idx):
            api_mappings.append((pytorch_apis[i], ""))

    # Process APIs
    for i, pt_api in enumerate(apis_to_process, start=start_idx):
        api_level = determine_api_level(pt_api)
        print(f"[INFO] Processing [{i + 1}/{total_apis}] {pt_api} (level: {api_level})")
        
        pd_api, llm_response = query_llm_for_api(
            client,
            pt_api,
            model=args.model,
            temperature=args.temperature,
        )
        
        api_mappings.append((pt_api, pd_api))
        log_entries.append({
            "index": i + 1,
            "pytorch_api": pt_api,
            "api_level": api_level,
            "paddle_api": pd_api,
            "llm_response": llm_response,
        })
        
        print(f"       -> {pd_api}")

        # Delay to avoid rate limits
        if args.delay > 0 and i < end_idx - 1:
            time.sleep(args.delay)

    # If remaining APIs exist (due to limit), fill with empty values
    if end_idx < total_apis:
        for i in range(end_idx, total_apis):
            api_mappings.append((pytorch_apis[i], ""))

    # Save results
    save_updated_csv(input_path, output_path, api_mappings)
    print(f"[SUCCESS] API mappings saved to: {output_path}")

    # Save LLM logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"pt_pd_mapping_log_{timestamp}.txt"
    log_path = log_dir / log_filename
    save_llm_log(log_entries, log_path)
    print(f"[SUCCESS] LLM logs saved to: {log_path}")

    # Summary
    mapped_count = sum(1 for _, pd in api_mappings if pd and pd != "无对应实现")
    no_impl_count = sum(1 for _, pd in api_mappings if pd == "无对应实现")
    empty_count = sum(1 for _, pd in api_mappings if not pd)

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
