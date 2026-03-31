# ./component/data/extract_new_found_paddle_apis.py
"""Extract newly found high-confidence PyTorch -> PaddlePaddle mappings (original is "无对应实现") and generate a new CSV."""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Dict

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]

# Default log directory
LOG_DIR = ROOT / "component" / "data" / "llm_logs"

# Default input CSV path
DEFAULT_INPUT_CSV = ROOT / "component" / "data" / "api_mappings.csv"

# Default output CSV path
DEFAULT_OUTPUT_CSV = ROOT / "component" / "data" / "pd_api_mappings_updated.csv"


def parse_paddle_validation_log(log_path: Path) -> List[Dict[str, str]]:
    """
    Parse PaddlePaddle validation logs and extract each record.

    Returns:
        List of dicts containing record info.
    """
    records: List[Dict[str, str]] = []
    
    with log_path.open("r", encoding="utf-8") as f:
        content = f.read()
    
    # Split records by separator line
    blocks = re.split(r'-{50,}', content)
    
    for block in blocks:
        block = block.strip()
        if not block or "序号:" not in block:
            continue
        
        record = {}
        
        # Extract index
        idx_match = re.search(r'序号:\s*(\d+)', block)
        if idx_match:
            record["index"] = idx_match.group(1)
        
        # Extract PyTorch API
        pt_match = re.search(r'PyTorch API:\s*(.+)', block)
        if pt_match:
            record["pytorch_api"] = pt_match.group(1).strip()
        
        # Extract original PaddlePaddle API (compatible with multiple labels)
        orig_pd_patterns = [
            r'原 PaddlePaddle API:\s*(.+)',
            r'原 Paddle API:\s*(.+)',
            r'原PaddlePaddle API:\s*(.+)',
        ]
        for pattern in orig_pd_patterns:
            orig_pd_match = re.search(pattern, block)
            if orig_pd_match:
                record["original_paddle_api"] = orig_pd_match.group(1).strip()
                break
        
        # Extract validated PaddlePaddle API (compatible with multiple labels)
        validated_pd_patterns = [
            r'验证后 PaddlePaddle API:\s*(.+)',
            r'验证后 Paddle API:\s*(.+)',
            r'验证后PaddlePaddle API:\s*(.+)',
        ]
        for pattern in validated_pd_patterns:
            validated_pd_match = re.search(pattern, block)
            if validated_pd_match:
                record["validated_paddle_api"] = validated_pd_match.group(1).strip()
                break
        
        # Extract confidence
        confidence_match = re.search(r'置信度:\s*(.+)', block)
        if confidence_match:
            record["confidence"] = confidence_match.group(1).strip()
        
        # Extract changed flag
        changed_match = re.search(r'是否修改:\s*(.+)', block)
        if changed_match:
            record["changed"] = changed_match.group(1).strip()
        
        # Extract reason
        reason_match = re.search(r'理由:\s*(.+)', block, re.DOTALL)
        if reason_match:
            reason_text = reason_match.group(1).strip()
            # Only keep the first line (avoid including full LLM output)
            reason_text = reason_text.split('\n')[0].strip()
            record["reason"] = reason_text
        
        if record.get("pytorch_api"):
            records.append(record)
    
    return records


def filter_new_found_high_confidence(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Filter records that match:
    1) Original PaddlePaddle API is "无对应实现"
    2) Validated PaddlePaddle API has a concrete value (not "无对应实现")
    3) Confidence is high
    """
    filtered = []
    for record in records:
        original_pd = record.get("original_paddle_api", "")
        validated_pd = record.get("validated_paddle_api", "")
        confidence = record.get("confidence", "")
        
        # Condition 1: original API is "无对应实现"
        if original_pd != "无对应实现":
            continue
        
        # Condition 2: validated API is concrete (not "无对应实现")
        if validated_pd == "无对应实现" or not validated_pd:
            continue
        
        # Condition 3: confidence is high
        if confidence.lower() != "high":
            continue
        
        filtered.append(record)
    
    return filtered


def read_original_mappings(csv_path: Path) -> Dict[str, str]:
    """
    Read pytorch-api -> paddle-api mappings from the original CSV.

    Returns:
        Dict {pytorch_api: paddle_api}
    """
    mappings = {}
    
    if not csv_path.exists():
        print(f"[WARNING] Input CSV not found: {csv_path}. A new file will be created.")
        return mappings
    
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt_api = row.get("pytorch-api", "")
            pd_api = row.get("paddle-api", "")
            if pt_api:
                mappings[pt_api] = pd_api
    
    return mappings


def create_updated_csv(
    original_mappings: Dict[str, str],
    new_mappings: List[Dict[str, str]],
    output_path: Path,
) -> int:
    """
    Create an updated CSV with only pytorch-api and paddle-api columns.

    Args:
        original_mappings: Original mapping dict
        new_mappings: Newly found mappings list
        output_path: Output file path

    Returns:
        Number of updated records
    """
    # Merge mappings: update original with new mappings
    merged = original_mappings.copy()
    updated_count = 0
    
    for record in new_mappings:
        pt_api = record.get("pytorch_api", "")
        validated_pd = record.get("validated_paddle_api", "")
        if pt_api and validated_pd:
            old_value = merged.get(pt_api, "")
            if old_value != validated_pd:
                merged[pt_api] = validated_pd
                updated_count += 1
                print(f"  [UPDATE] {pt_api}: {old_value} -> {validated_pd}")
    
    # Write CSV (only keep pytorch-api and paddle-api columns)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pytorch-api", "paddle-api"])
        
        # Sort by PyTorch API name
        for pt_api in sorted(merged.keys()):
            pd_api = merged[pt_api]
            writer.writerow([pt_api, pd_api])
    
    return updated_count


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract newly found high-confidence API mappings from PaddlePaddle validation logs"
    )
    parser.add_argument(
        "--log",
        "-l",
        required=True,
        help="Path to the PaddlePaddle validation log file",
    )
    parser.add_argument(
        "--input-csv",
        "-i",
        default=str(DEFAULT_INPUT_CSV),
        help="Input original CSV path (default: component/data/api_mappings.csv)",
    )
    parser.add_argument(
        "--output-csv",
        "-o",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Output CSV path (default: component/data/pd_api_mappings_updated.csv)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show details (including mapped PaddlePaddle APIs and reasons)",
    )

    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[ERROR] Log file not found: {log_path}")
        return

    print(f"[INFO] Parsing log file: {log_path}")
    records = parse_paddle_validation_log(log_path)
    print(f"[INFO] Parsed {len(records)} records")

    # Filter records that meet criteria
    filtered = filter_new_found_high_confidence(records)
    print(
        f"[INFO] Records matching criteria (original is '无对应实现' + validated has value + high confidence): {len(filtered)}"
    )

    # Print details
    if filtered:
        print("\n" + "=" * 70)
        print("Newly found high-confidence PaddlePaddle API mappings")
        print("=" * 70)
        for i, record in enumerate(filtered, start=1):
            pt_api = record.get("pytorch_api", "")
            validated_pd = record.get("validated_paddle_api", "")
            reason = record.get("reason", "")
            
            if args.verbose:
                print(f"\n{i}. {pt_api}")
                print(f"   -> {validated_pd}")
                print(f"   Reason: {reason}")
            else:
                print(f"{i}. {pt_api} -> {validated_pd}")
        print("=" * 70)

    # Read original mappings
    input_csv_path = Path(args.input_csv)
    print(f"\n[INFO] Reading original CSV: {input_csv_path}")
    original_mappings = read_original_mappings(input_csv_path)
    print(f"[INFO] Original mapping count: {len(original_mappings)}")

    # Create updated CSV
    output_csv_path = Path(args.output_csv)
    print(f"\n[INFO] Generating updated CSV: {output_csv_path}")
    updated_count = create_updated_csv(original_mappings, filtered, output_csv_path)
    
    print(f"\n[SUCCESS] Generated CSV: {output_csv_path}")
    print(f"[SUCCESS] Total mappings: {len(original_mappings)}")
    print(f"[SUCCESS] Updated records this run: {updated_count}")

    # Print plain API list for easy copy
    if filtered:
        print("\n" + "=" * 70)
        print("[Newly found PyTorch API Name List (Plain Text)]")
        print("=" * 70)
        for record in filtered:
            print(record.get("pytorch_api", ""))


if __name__ == "__main__":
    main()
