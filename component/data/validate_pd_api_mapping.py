# ./component/data/validate_pd_api_mapping.py
"""Validate PyTorch-to-PaddlePaddle API mappings with an LLM."""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import sys

# Add the project root to sys.path so component modules can be imported.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.migration.migrate_generate_tests import get_qwen_client
from component.doc.doc_crawler_factory import get_doc_content

DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"

# Log directory
LOG_DIR = ROOT / "component" / "data" / "llm_logs"


def load_api_mappings(csv_path: Path) -> List[Dict[str, str]]:
    """Load API mappings from api_mappings.csv."""
    mappings: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt_api = row.get("pytorch-api", "").strip()
            pd_api = row.get("paddle-api", "").strip()
            if pt_api:
                mappings.append({
                    "pytorch_api": pt_api,
                    "paddle_api": pd_api,
                })
    return mappings


def fetch_api_docs(pytorch_api: str, paddle_api: str) -> Tuple[str, str]:
    """Fetch official docs for PyTorch and PaddlePaddle APIs.

    Returns:
        (pytorch_doc, paddle_doc)
    """
    pt_doc = ""
    pd_doc = ""
    
    # Fetch PyTorch docs
    if pytorch_api:
        try:
            doc_text = get_doc_content(pytorch_api, "pytorch")
            if doc_text and len(doc_text.strip()) > 300 and "Unable to" not in doc_text and "not supported" not in doc_text:
                pt_doc = doc_text
            else:
                print(
                    f"[WARN] Unable to fetch PyTorch docs: {pytorch_api} (length: {len(doc_text.strip()) if doc_text else 0})"
                )
        except Exception as e:
            print(f"[WARN] Failed to fetch PyTorch docs {pytorch_api}: {e}")
    
    # Fetch PaddlePaddle docs (if not "无对应实现")
    if paddle_api and paddle_api != "无对应实现":
        try:
            doc_text = get_doc_content(paddle_api, "paddle")
            if doc_text and len(doc_text.strip()) > 300 and "Unable to" not in doc_text and "not supported" not in doc_text:
                pd_doc = doc_text
            else:
                print(
                    f"[WARN] Unable to fetch PaddlePaddle docs: {paddle_api} (length: {len(doc_text.strip()) if doc_text else 0})"
                )
        except Exception as e:
            print(f"[WARN] Failed to fetch PaddlePaddle docs {paddle_api}: {e}")
    
    return pt_doc, pd_doc


def determine_api_level(api_name: str) -> str:
    """Determine API level: function or class.

    Rules:
    - torch.nn.XXX and capitalized -> class (e.g., torch.nn.Conv1d, torch.nn.ReLU)
    - torch.nn.functional.xxx -> function (e.g., torch.nn.functional.relu)
    - torch.xxx and lowercase -> function (e.g., torch.abs, torch.add)
    - torch.nn.utils.xxx -> function (e.g., torch.nn.utils.clip_grad_norm_)
    """
    parts = api_name.split(".")
    
    # torch.nn.functional.xxx -> 函数级别
    if "functional" in api_name:
        return "function"
    
    # torch.nn.utils.xxx -> 函数级别
    if "utils" in api_name:
        return "function"
    
    # torch.nn.XXX 且最后部分首字母大写 -> 类级别
    if len(parts) >= 3 and parts[1] == "nn":
        last_part = parts[-1]
        if last_part and last_part[0].isupper():
            return "class"
    
    # 默认为函数级别
    return "function"


def build_validation_prompt(
    pytorch_api: str,
    paddle_api: str,
    pt_doc: str,
    pd_doc: str,
) -> str:
    """Build the LLM prompt for mapping validation."""

    pt_doc_text = pt_doc if pt_doc else "No relevant PyTorch docs found"
    pd_doc_text = pd_doc if pd_doc else "No relevant PaddlePaddle docs found"

    # Case 1: original mapping is "无对应实现", ask LLM to search again
    if paddle_api == "无对应实现" or not paddle_api:
        prompt = f"""You are a deep learning framework expert fluent in PyTorch and PaddlePaddle.

[Task]
The current record shows that PyTorch API "{pytorch_api}" has "无对应实现" in PaddlePaddle.
Based on the official PyTorch docs below, determine whether PaddlePaddle truly has no functionally equivalent API.
If you believe there is a corresponding PaddlePaddle API, provide its exact name.

[PyTorch API]
{pytorch_api}

[PyTorch Official Docs]
{pt_doc_text}

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

[Requirements]
1. Read the PyTorch docs carefully to understand the API's behavior, parameters, and return values.
2. Based on your PaddlePaddle knowledge, decide whether a functionally equivalent API exists.
3. If it exists, provide the exact PaddlePaddle API name; if it truly does not exist, return "无对应实现".
4. The returned API level must match the original (function-to-function, class-to-class).

[Output Format]
Output strictly in the following JSON format with no extra text:

```json
{{
    "pytorch_api": "{pytorch_api}",
    "paddle_api": "<matching PaddlePaddle API or '无对应实现'>",
    "confidence": "<high/medium/low>",
    "reason": "<Briefly explain why it matches or why no equivalent exists>"
}}
```

Notes:
- The paddle_api field must be the full API name (e.g., paddle.abs or paddle.nn.Conv2D) or "无对应实现".
- confidence reflects your confidence (high >= 85%, medium 40%-85%, low < 40%).
- paddle_api must be a real PaddlePaddle API name; do not invent APIs.
- reason should be brief (one or two sentences).
"""
    # Case 2: PaddlePaddle API exists but docs are empty (likely invalid)
    elif not pd_doc:
        api_level = determine_api_level(pytorch_api)
        level_desc = "function" if api_level == "function" else "class"

        prompt = f"""You are a deep learning framework expert fluent in PyTorch and PaddlePaddle.

[Task]
The current record maps PyTorch API "{pytorch_api}" to PaddlePaddle API "{paddle_api}".
However, we **were unable to fetch the official docs for "{paddle_api}"**, which likely means the API name is incorrect or does not exist.

Using the official PyTorch docs below, re-evaluate whether PaddlePaddle has a functionally equivalent API and provide the correct API name.

[PyTorch API]
{pytorch_api}

[Original (likely invalid) PaddlePaddle API]
{paddle_api}

[PyTorch Official Docs]
{pt_doc_text}

[API Level]
This is a **{level_desc}** API. The returned PaddlePaddle API must be the same level as the original.

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

[Requirements]
1. Read the PyTorch docs carefully to understand the API's behavior, parameters, and return values.
2. Based on your PaddlePaddle knowledge, find the functionally equivalent API.
3. **You must return a real PaddlePaddle API name**; do not invent APIs.
4. If no equivalent API exists, return "无对应实现".

[Output Format]
Output strictly in the following JSON format with no extra text:

```json
{{
    "pytorch_api": "{pytorch_api}",
    "paddle_api": "<correct PaddlePaddle API or '无对应实现'>",
    "confidence": "<high/medium/low>",
    "reason": "<Explain why the original API was wrong and why the new one is correct, or why no equivalent exists>"
}}
```

Notes:
- The paddle_api field must be the full API name (e.g., paddle.abs or paddle.nn.Conv2D) or "无对应实现".
- confidence reflects your confidence (high >= 85%, medium 40%-85%, low < 40%).
- paddle_api must be a real PaddlePaddle API name; do not invent APIs.
- reason should be brief (one or two sentences).
"""
    # Case 3: PaddlePaddle API exists and docs are available
    else:
        prompt = f"""You are a deep learning framework expert fluent in PyTorch and PaddlePaddle.

[Task]
Validate whether the following PyTorch-to-PaddlePaddle API mapping is correct.
Use the provided docs to judge whether the APIs are functionally equivalent or highly similar.

[Current Mapping]
- PyTorch API: {pytorch_api}
- PaddlePaddle API: {paddle_api}

[PyTorch Official Docs]
{pt_doc_text}

[PaddlePaddle Official Docs]
{pd_doc_text}

[Validation Points]
1. **Functionality**: Do the APIs perform the same core function?
2. **Parameter alignment**: Can the key parameters be mapped reasonably?
3. **Return compatibility**: Are return types/meanings compatible?
4. **API level**: Are they both functions or both classes?

[Decision Rules]
- If the APIs are equivalent and parameters align, the mapping is **correct**.
- If there are significant functional differences or parameters cannot align, the mapping is **incorrect**; provide a more appropriate PaddlePaddle API if possible.
- If the current mapping is incorrect and PaddlePaddle has no equivalent API, return "无对应实现".

[Output Format]
Output strictly in the following JSON format with no extra text:

```json
{{
    "pytorch_api": "{pytorch_api}",
    "paddle_api": "<validated PaddlePaddle API: keep if correct, otherwise provide the correct API or '无对应实现'>",
    "confidence": "<high/medium/low>",
    "reason": "<Briefly explain why the mapping is correct/incorrect; if you change it, explain why>"
}}
```

Notes:
- The paddle_api field must be the full API name (e.g., paddle.abs or paddle.nn.Conv2D) or "无对应实现".
- confidence reflects your confidence (high >= 85%, medium 40%-85%, low < 40%).
- paddle_api must be a real PaddlePaddle API name; do not invent APIs.
- reason should be brief (one or two sentences).
"""
    
    return prompt


def parse_llm_response(response: str, original_pd_api: str) -> Tuple[str, str, str]:
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
            pd_api = data.get("paddle_api", original_pd_api).strip()
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

    # Return original mapping
    return original_pd_api, "unknown", "Parsing failed; keeping original mapping"


def validate_mapping_with_llm(
    client,
    pytorch_api: str,
    paddle_api: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> Tuple[str, str, str, str, bool, bool]:
    """Validate API mappings using an LLM.

    Returns:
        (validated_pd_api, confidence, reason, full_response, pt_doc_empty, pd_doc_empty)
    """
    # Fetch docs
    pt_doc, pd_doc = fetch_api_docs(pytorch_api, paddle_api)

    # Track whether docs are empty
    pt_doc_empty = not pt_doc
    pd_doc_empty = not pd_doc and paddle_api and paddle_api != "无对应实现"

    # Build prompt
    prompt = build_validation_prompt(pytorch_api, paddle_api, pt_doc, pd_doc)
    
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
            
            pd_api, confidence, reason = parse_llm_response(full_response, paddle_api)
            return pd_api, confidence, reason, full_response, pt_doc_empty, pd_doc_empty
            
        except Exception as e:
            print(f"[WARN] API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue

    return (
        paddle_api,
        "unknown",
        f"Call failed after {max_retries} retries",
        "[ERROR] LLM call failed",
        pt_doc_empty,
        pd_doc_empty,
    )


def save_validation_log(
    log_entries: List[dict],
    log_path: Path,
) -> None:
    """Save validation logs to a file."""
    with log_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch to PaddlePaddle API Mapping Validation - LLM Log\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total entries: {len(log_entries)}\n")
        f.write("=" * 80 + "\n\n")
        
        for entry in log_entries:
            f.write("-" * 60 + "\n")
            f.write(f"序号: {entry['index']}\n")
            f.write(f"PyTorch API: {entry['pytorch_api']}\n")
            f.write(f"原 PaddlePaddle API: {entry['original_pd_api']}\n")
            f.write(f"验证后 PaddlePaddle API: {entry['validated_pd_api']}\n")
            f.write(f"置信度: {entry['confidence']}\n")
            f.write(f"是否修改: {'是' if entry['changed'] else '否'}\n")
            f.write(f"理由: {entry['reason']}\n")
            f.write(f"\n【LLM 完整输出】\n{entry['llm_response']}\n")
            f.write("-" * 60 + "\n\n")


def save_validated_csv(
    output_path: Path,
    validated_mappings: List[Dict[str, str]],
) -> None:
    """Save the validated CSV file."""
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pytorch-api", "paddle-api", "confidence", "changed"])
        for mapping in validated_mappings:
            writer.writerow([
                mapping["pytorch_api"],
                mapping["paddle_api"],
                mapping.get("confidence", ""),
                "Y" if mapping.get("changed", False) else "N",
            ])


def main():
    """CLI entry: batch validate PyTorch-to-PaddlePaddle API mappings."""
    parser = argparse.ArgumentParser(
        description="Validate PyTorch-to-PaddlePaddle API mappings with an LLM"
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
        default=str(ROOT / "component" / "data" / "api_mappings_pd_validated.csv"),
        help="Path to output validated CSV",
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
        default=1.0,
        help="Delay in seconds between API calls (default: 1.0; docs fetching involved)",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.1,
        help="LLM temperature (0.0-1.0). Lower is more deterministic (default: 0.1)",
    )
    parser.add_argument(
        "--log-dir",
        default=str(LOG_DIR),
        help="Output directory for LLM logs",
    )
    parser.add_argument(
        "--only-no-impl",
        action="store_true",
        help="Only process entries marked '无对应实现'",
    )
    parser.add_argument(
        "--only-has-impl",
        action="store_true",
        help="Only process entries with existing mappings",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return

    print(f"[INFO] Loading API mappings: {input_path}")
    all_mappings = load_api_mappings(input_path)
    
    if not all_mappings:
        print("[ERROR] No API mappings found in CSV")
        return

    # 根据过滤条件筛选
    if args.only_no_impl:
        mappings = [m for m in all_mappings if m["paddle_api"] == "无对应实现" or not m["paddle_api"]]
        print(f"[INFO] Filtered '无对应实现' entries: {len(mappings)}")
    elif args.only_has_impl:
        mappings = [m for m in all_mappings if m["paddle_api"] and m["paddle_api"] != "无对应实现"]
        print(f"[INFO] Filtered entries with existing implementations: {len(mappings)}")
    else:
        mappings = all_mappings

    # 处理 start 和 limit 参数
    total_mappings = len(mappings)
    start_idx = args.start
    end_idx = total_mappings if args.limit is None else min(start_idx + args.limit, total_mappings)
    
    mappings_to_process = mappings[start_idx:end_idx]
    print(
        f"[INFO] Total records: {total_mappings}; processing range: [{start_idx}, {end_idx}) => {len(mappings_to_process)} entries"
    )

    try:
        client = get_qwen_client(args.key_path)
        print(f"[INFO] LLM client initialized, model: {args.model}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM client: {e}")
        return

    # 存储结果
    validated_mappings: List[Dict[str, str]] = []
    log_entries: List[dict] = []

    # Counters
    unchanged_count = 0
    changed_count = 0
    found_new_count = 0  # Previously no-impl, now found

    # Track doc fetch failures
    pt_doc_empty_apis: List[str] = []  # PyTorch docs empty
    pd_doc_empty_apis: List[str] = []  # PaddlePaddle docs empty

    # Process mappings
    for i, mapping in enumerate(mappings_to_process, start=start_idx):
        pt_api = mapping["pytorch_api"]
        original_pd_api = mapping["paddle_api"]

        status_desc = "无对应实现" if (original_pd_api == "无对应实现" or not original_pd_api) else original_pd_api
        print(f"[INFO] Processing [{i + 1}/{total_mappings}] {pt_api} -> {status_desc}")

        validated_pd_api, confidence, reason, llm_response, pt_doc_empty, pd_doc_empty = validate_mapping_with_llm(
            client,
            pt_api,
            original_pd_api,
            model=args.model,
            temperature=args.temperature,
        )

        # Record doc fetch failures
        if pt_doc_empty:
            pt_doc_empty_apis.append(pt_api)
        if pd_doc_empty:
            pd_doc_empty_apis.append(f"{pt_api} -> {original_pd_api}")

        # Detect changes
        changed = validated_pd_api != original_pd_api
        if changed:
            changed_count += 1
            if (original_pd_api == "无对应实现" or not original_pd_api) and validated_pd_api != "无对应实现":
                found_new_count += 1
                print(f"       -> [NEW] {validated_pd_api}")
            else:
                print(f"       -> [UPDATED] {original_pd_api} => {validated_pd_api}")
        else:
            unchanged_count += 1
            print(f"       -> [CONFIRMED] {validated_pd_api}")
        
        validated_mappings.append({
            "pytorch_api": pt_api,
            "paddle_api": validated_pd_api,
            "confidence": confidence,
            "changed": changed,
        })
        
        log_entries.append({
            "index": i + 1,
            "pytorch_api": pt_api,
            "original_pd_api": original_pd_api,
            "validated_pd_api": validated_pd_api,
            "confidence": confidence,
            "reason": reason,
            "changed": changed,
            "llm_response": llm_response,
        })
        
        # Delay to avoid rate limits
        if args.delay > 0 and i < end_idx - 1:
            time.sleep(args.delay)

    # Save validated CSV
    save_validated_csv(output_path, validated_mappings)
    print(f"[SUCCESS] Validation results saved to: {output_path}")

    # Save LLM logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"pt_pd_validation_log_{timestamp}.txt"
    log_path = log_dir / log_filename
    save_validation_log(log_entries, log_path)
    print(f"[SUCCESS] LLM logs saved to: {log_path}")

    # Summary
    print("\n" + "=" * 50)
    print("[Validation Summary]")
    print(f"  Total records: {total_mappings}")
    print(f"  Processed this run: {len(mappings_to_process)}")
    print(f"  Confirmed (unchanged): {unchanged_count}")
    print(f"  Updated mappings: {changed_count}")
    print(f"  Newly found APIs: {found_new_count}")
    print("=" * 50)
    
    # Print doc fetch failure stats
    print("\n" + "=" * 50)
    print("[Doc Fetch Failure Summary]")
    print(f"  PyTorch docs empty: {len(pt_doc_empty_apis)}")
    print(f"  PaddlePaddle docs empty: {len(pd_doc_empty_apis)}")
    print("=" * 50)
    
    if pt_doc_empty_apis:
        print("\n[APIs with empty PyTorch docs]")
        for api in pt_doc_empty_apis:
            print(f"  - {api}")
    
    if pd_doc_empty_apis:
        print("\n[APIs with empty PaddlePaddle docs]")
        for api in pd_doc_empty_apis:
            print(f"  - {api}")


if __name__ == "__main__":
    main()
