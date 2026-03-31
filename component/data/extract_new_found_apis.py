# ./component/data/extract_new_found_apis.py
"""Extract newly found high-confidence API mappings and optionally update CSV."""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Dict

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]

# Default log directory
LOG_DIR = ROOT / "component" / "data" / "llm_logs"

# Default CSV path
DEFAULT_CSV_PATH = ROOT / "component" / "data" / "api_mappings.csv"


def parse_validation_log(log_path: Path) -> List[Dict[str, str]]:
    """Parse validation log file and extract record info.

    Returns:
        List of dicts with record info
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
        
        # Extract original TensorFlow API
        orig_tf_match = re.search(r'原 TensorFlow API:\s*(.+)', block)
        if orig_tf_match:
            record["original_tf_api"] = orig_tf_match.group(1).strip()
        
        # Extract validated TensorFlow API
        validated_tf_match = re.search(r'验证后 TensorFlow API:\s*(.+)', block)
        if validated_tf_match:
            record["validated_tf_api"] = validated_tf_match.group(1).strip()
        
        # Extract confidence
        confidence_match = re.search(r'置信度:\s*(.+)', block)
        if confidence_match:
            record["confidence"] = confidence_match.group(1).strip()
        
        # Extract change flag
        changed_match = re.search(r'是否修改:\s*(.+)', block)
        if changed_match:
            record["changed"] = changed_match.group(1).strip()
        
        # Extract reason
        reason_match = re.search(r'理由:\s*(.+)', block)
        if reason_match:
            record["reason"] = reason_match.group(1).strip()
        
        if record.get("pytorch_api"):
            records.append(record)
    
    return records


def filter_new_found_high_confidence(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Filter records that meet:
    1) original TensorFlow API is "无对应实现"
    2) validated TensorFlow API has a concrete value (not "无对应实现")
    3) confidence is high
    """
    filtered = []
    for record in records:
        original_tf = record.get("original_tf_api", "")
        validated_tf = record.get("validated_tf_api", "")
        confidence = record.get("confidence", "")
        
        # Condition 1: original is "无对应实现"
        if original_tf != "无对应实现":
            continue

        # Condition 2: validated has concrete value
        if validated_tf == "无对应实现" or not validated_tf:
            continue

        # Condition 3: confidence is high
        if confidence.lower() != "high":
            continue
        
        filtered.append(record)
    
    return filtered


def update_csv_with_new_mappings(
    csv_path: Path,
    new_mappings: List[Dict[str, str]],
    output_path: Path = None,
) -> int:
    """Update CSV with newly found mappings.

    Args:
        csv_path: original CSV path
        new_mappings: list of new mappings
        output_path: output path (overwrite original if None)

    Returns:
        Number of updated records
    """
    if output_path is None:
        output_path = csv_path
    
    # Build PyTorch API -> new TensorFlow API mapping
    update_dict = {}
    for record in new_mappings:
        pt_api = record.get("pytorch_api", "")
        validated_tf = record.get("validated_tf_api", "")
        if pt_api and validated_tf:
            update_dict[pt_api] = validated_tf
    
    # Read original CSV
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    # Update records
    updated_count = 0
    for row in rows:
        pt_api = row.get("pytorch-api", "")
        if pt_api in update_dict:
            old_value = row.get("tensorflow-api", "")
            new_value = update_dict[pt_api]
            if old_value != new_value:
                row["tensorflow-api"] = new_value
                updated_count += 1
                print(f"  [UPDATED] {pt_api}: {old_value} -> {new_value}")
    
    # Write updated CSV
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return updated_count


def main():
    """CLI entry."""
    parser = argparse.ArgumentParser(
        description="Extract newly found high-confidence API mappings from validation logs"
    )
    parser.add_argument(
        "--log",
        "-l",
        required=True,
        help="Validation log path",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path (prints to stdout if omitted)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show details (including TensorFlow API and reason)",
    )
    parser.add_argument(
        "--update-csv",
        "-u",
        action="store_true",
        help="Update corresponding records in api_mappings.csv",
    )
    parser.add_argument(
        "--csv-path",
        "-c",
        default=str(DEFAULT_CSV_PATH),
        help="Path to api_mappings.csv (default: component/data/api_mappings.csv)",
    )
    parser.add_argument(
        "--output-csv",
        help="Output CSV path (overwrites original if omitted)",
    )

    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[ERROR] Log file not found: {log_path}")
        return

    print(f"[INFO] Parsing log file: {log_path}")
    records = parse_validation_log(log_path)
    print(f"[INFO] Parsed {len(records)} records")

    # Filter records
    filtered = filter_new_found_high_confidence(records)
    print(f"[INFO] Records meeting criteria: {len(filtered)}")

    # Build output content
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("Newly found high-confidence API mappings")
    output_lines.append("Criteria: original API is '无对应实现' + validated has concrete API + confidence=high")
    output_lines.append(f"Matching records: {len(filtered)}")
    output_lines.append("=" * 70)
    output_lines.append("")

    for i, record in enumerate(filtered, start=1):
        pt_api = record.get("pytorch_api", "")
        validated_tf = record.get("validated_tf_api", "")
        reason = record.get("reason", "")
        
        if args.verbose:
            output_lines.append(f"{i}. {pt_api}")
            output_lines.append(f"   -> {validated_tf}")
            output_lines.append(f"   Reason: {reason}")
            output_lines.append("")
        else:
            output_lines.append(f"{i}. {pt_api} -> {validated_tf}")

    output_text = "\n".join(output_lines)

    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"[SUCCESS] Results saved to: {output_path}")
    else:
        print("\n" + output_text)

    # Update CSV if requested
    if args.update_csv:
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            print(f"[ERROR] CSV file not found: {csv_path}")
            return
        
        output_csv_path = Path(args.output_csv) if args.output_csv else None
        print(f"\n[INFO] Updating CSV file: {csv_path}")
        if output_csv_path:
            print(f"[INFO] Output CSV path: {output_csv_path}")
        updated_count = update_csv_with_new_mappings(csv_path, filtered, output_csv_path)
        print(f"[SUCCESS] Updated {updated_count} records")
    # Also print plain API list for easy copy
    print("\n" + "=" * 70)
    print("[PyTorch API Name List (Plain Text)]")
    print("=" * 70)
    for record in filtered:
        print(record.get("pytorch_api", ""))


if __name__ == "__main__":
    main()
