# ./component/data/extract_changed_apis.py
"""
Extract API mappings that need updates from validation logs:
1. Original and validated APIs both have values but differ + confidence=high
2. Original API has a value + validated API is "无对应实现" + confidence=high

Optionally update api_mappings.csv.
Supports doc validation: check whether TensorFlow API docs exist; if not, mark as "无对应实现".
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Default log directory
LOG_DIR = ROOT / "component" / "data" / "llm_logs"

# Default CSV path
DEFAULT_CSV_PATH = ROOT / "component" / "data" / "api_mappings.csv"


def check_tf_doc_exists(tf_api: str) -> Tuple[bool, str]:
    """
    Check whether TensorFlow API documentation exists.

    Args:
        tf_api: TensorFlow API name

    Returns:
        (doc_exists, doc_content_or_error)
    """
    # Lazy import to avoid overhead when doc validation is off
    from component.doc.doc_crawler_factory import get_doc_content
    
    # If it is "无对应实现", return True directly
    if tf_api == "无对应实现":
        return True, "无对应实现"
    
    # Handle API names with parameters, e.g., tf.keras.layers.Activation('gelu')
    # Extract base API name
    base_api = tf_api.split('(')[0].strip()
    
    try:
        doc_content = get_doc_content(base_api, "tensorflow")
        if doc_content and len(doc_content.strip()) > 300:
            return True, doc_content[:200] + "..."
        else:
            return False, f"Doc is empty or too short: {len(doc_content) if doc_content else 0} chars"
    except Exception as e:
        return False, f"Crawl failed: {str(e)}"


def parse_validation_log(log_path: Path) -> List[Dict[str, str]]:
    """
    Parse validation log file and extract each record.

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


def filter_changed_mappings(
    records: List[Dict[str, str]]
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Filter records that need updates, split into two types:

    Type 1: original and validated APIs both have values but differ + confidence=high
            (mapping needs update)
    Type 2: original has value + validated is "无对应实现" + confidence=high
            (original mapping is wrong)

    Returns:
        (type1_records, type2_records)
    """
    type1_records: List[Dict[str, str]] = []  # mapping update
    type2_records: List[Dict[str, str]] = []  # original mapping wrong
    
    for record in records:
        original_tf = record.get("original_tf_api", "")
        validated_tf = record.get("validated_tf_api", "")
        confidence = record.get("confidence", "")
        
        # Confidence must be high
        if confidence.lower() != "high":
            continue
        
        # Original API must have a concrete value (not "无对应实现")
        if original_tf == "无对应实现" or not original_tf:
            continue
        
        # Type 1: both have concrete values but differ
        if validated_tf and validated_tf != "无对应实现" and original_tf != validated_tf:
            type1_records.append(record)
        
        # Type 2: validated is "无对应实现"
        elif validated_tf == "无对应实现":
            type2_records.append(record)
    
    return type1_records, type2_records


def update_csv_with_changed_mappings(
    csv_path: Path,
    type1_records: List[Dict[str, str]],
    type2_records: List[Dict[str, str]],
    output_path: Path = None,
    verify_doc: bool = False,
) -> Tuple[int, int, int]:
    """
    Update CSV mappings based on the changed records.

    Args:
        csv_path: original CSV path
        type1_records: type 1 records (mapping updates)
        type2_records: type 2 records (mark as "无对应实现")
        output_path: output path (overwrite if omitted)
        verify_doc: whether to validate TensorFlow API docs

    Returns:
        (type1_update_count, type2_update_count, doc_fallback_count)
    """
    if output_path is None:
        output_path = csv_path
    
    # Build update dicts
    # Type 1: PyTorch API -> (validated TensorFlow API, original TensorFlow API)
    type1_dict = {}
    for record in type1_records:
        pt_api = record.get("pytorch_api", "")
        validated_tf = record.get("validated_tf_api", "")
        original_tf = record.get("original_tf_api", "")
        if pt_api and validated_tf:
            type1_dict[pt_api] = (validated_tf, original_tf)
    
    # Type 2: PyTorch API -> "无对应实现"
    type2_dict = {}
    for record in type2_records:
        pt_api = record.get("pytorch_api", "")
        if pt_api:
            type2_dict[pt_api] = "无对应实现"
    
    # Read original CSV
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    # Apply updates
    type1_count = 0
    type2_count = 0
    doc_fallback_count = 0  # doc validation fallback count
    
    for row in rows:
        pt_api = row.get("pytorch-api", "")
        old_value = row.get("tensorflow-api", "")
        
        # Check type 1 first (mapping update)
        if pt_api in type1_dict:
            validated_tf, original_tf = type1_dict[pt_api]
            new_value = validated_tf
            
            # If doc validation is enabled, check validated TensorFlow API docs
            if verify_doc:
                print(f"\n  Check doc: {pt_api}")
                print(f"    Original mapping: {original_tf}")
                print(f"    New mapping: {validated_tf}")
                
                doc_exists, doc_info = check_tf_doc_exists(validated_tf)
                
                if doc_exists:
                    print(f"    ✅ New mapping doc exists")
                    # Use validated value
                    new_value = validated_tf
                else:
                    print(f"    ❌ New mapping doc missing: {doc_info}")
                    print(f"    ⚠️ Mark as 无对应实现")
                    # Doc missing; mark as "无对应实现"
                    new_value = "无对应实现"
                    doc_fallback_count += 1
            
            if old_value != new_value:
                row["tensorflow-api"] = new_value
                type1_count += 1
                if new_value == "无对应实现":
                    print(f"  [Type1-Doc missing] {pt_api}: {old_value} -> 无对应实现")
                else:
                    print(f"  [Type1-Update mapping] {pt_api}: {old_value} -> {new_value}")
        
        # Check type 2 (mark as "无对应实现")
        elif pt_api in type2_dict:
            new_value = type2_dict[pt_api]
            if old_value != new_value:
                row["tensorflow-api"] = new_value
                type2_count += 1
                print(f"  [Type2-No mapping] {pt_api}: {old_value} -> {new_value}")
    
    # Write updated CSV
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return type1_count, type2_count, doc_fallback_count


def format_output(
    type1_records: List[Dict[str, str]],
    type2_records: List[Dict[str, str]],
    verbose: bool = False,
) -> str:
    """Format output text."""
    lines = []
    
    # Title
    lines.append("=" * 70)
    lines.append("API mappings to update (high confidence)")
    lines.append("=" * 70)
    lines.append("")
    
    # Type 1: mapping update
    lines.append("-" * 70)
    lines.append("[Type 1] Original and validated APIs both have values but differ")
    lines.append(f"Records matched: {len(type1_records)}")
    lines.append("-" * 70)
    
    for i, record in enumerate(type1_records, start=1):
        pt_api = record.get("pytorch_api", "")
        original_tf = record.get("original_tf_api", "")
        validated_tf = record.get("validated_tf_api", "")
        reason = record.get("reason", "")
        
        if verbose:
            lines.append(f"{i}. {pt_api}")
            lines.append(f"   Original: {original_tf}")
            lines.append(f"   New: {validated_tf}")
            lines.append(f"   Reason: {reason}")
            lines.append("")
        else:
            lines.append(f"{i}. {pt_api}: {original_tf} -> {validated_tf}")
    
    lines.append("")
    
    # Type 2: original mapping wrong
    lines.append("-" * 70)
    lines.append("[Type 2] Original has value but validated is '无对应实现'")
    lines.append(f"Records matched: {len(type2_records)}")
    lines.append("-" * 70)
    
    for i, record in enumerate(type2_records, start=1):
        pt_api = record.get("pytorch_api", "")
        original_tf = record.get("original_tf_api", "")
        reason = record.get("reason", "")
        
        if verbose:
            lines.append(f"{i}. {pt_api}")
            lines.append(f"   Original: {original_tf}")
            lines.append(f"   New: 无对应实现")
            lines.append(f"   Reason: {reason}")
            lines.append("")
        else:
            lines.append(f"{i}. {pt_api}: {original_tf} -> 无对应实现")
    
    lines.append("")
    
    # Summary
    lines.append("=" * 70)
    lines.append(f"Summary: Type 1 = {len(type1_records)}, Type 2 = {len(type2_records)}")
    lines.append(f"Total to update: {len(type1_records) + len(type2_records)}")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract API mappings to update from validation logs (high confidence)"
    )
    parser.add_argument(
        "--log",
        "-l",
        required=True,
        help="Validation log file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (print to stdout if omitted)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show details (original mapping, new mapping, reason)",
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
        help="api_mappings.csv path (default: component/data/api_mappings.csv)",
    )
    parser.add_argument(
        "--output-csv",
        help="Updated CSV output path (overwrite if omitted)",
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["all", "1", "2"],
        default="all",
        help="Filter type: all=all, 1=type1(mapping update), 2=type2(无对应实现)",
    )
    parser.add_argument(
        "--verify-doc",
        action="store_true",
        help="Enable doc validation: if TensorFlow API doc is missing, mark as '无对应实现'",
    )

    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[ERROR] Log file does not exist: {log_path}")
        return

    print(f"[INFO] Parsing log file: {log_path}")
    records = parse_validation_log(log_path)
    print(f"[INFO] Parsed {len(records)} records")

    # Filter matching records
    type1_records, type2_records = filter_changed_mappings(records)
    
    # Filter by --type
    if args.type == "1":
        type2_records = []
    elif args.type == "2":
        type1_records = []
    
    print(f"[INFO] Type 1 (mapping update) records: {len(type1_records)}")
    print(f"[INFO] Type 2 (无对应实现) records: {len(type2_records)}")

    # Build output content
    output_text = format_output(type1_records, type2_records, args.verbose)

    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"[SUCCESS] Results saved to: {output_path}")
    else:
        print("\n" + output_text)

    # If --update-csv is specified, update CSV
    if args.update_csv:
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            print(f"[ERROR] CSV file does not exist: {csv_path}")
            return
        
        print(f"\n[INFO] Updating CSV: {csv_path}")
        output_csv_path = Path(args.output_csv) if args.output_csv else None
        if output_csv_path:
            print(f"[INFO] Writing to new file: {output_csv_path}")
        
        if args.verify_doc:
            print("[INFO] Doc validation enabled; will crawl TensorFlow API docs")
        
        type1_count, type2_count, doc_fallback_count = update_csv_with_changed_mappings(
            csv_path, type1_records, type2_records, output_csv_path, args.verify_doc
        )
        
        total_updated = type1_count + type2_count
        print(f"\n[SUCCESS] Updated {total_updated} records")
        print(f"  - Type 1 (mapping update): {type1_count}")
        print(f"  - Type 2 (无对应实现): {type2_count}")
        
        if args.verify_doc and doc_fallback_count > 0:
            print(f"  - Converted to '无对应实现' due to missing docs: {doc_fallback_count}")

    # Print plain API name list (easy to copy)
    print("\n" + "=" * 70)
    print("[PyTorch API Name List (Plain Text)]")
    print("=" * 70)
    
    if type1_records:
        print("\n--- Type 1: mapping update ---")
        for record in type1_records:
            print(record.get("pytorch_api", ""))
    
    if type2_records:
        print("\n--- Type 2: 无对应实现 ---")
        for record in type2_records:
            print(record.get("pytorch_api", ""))


if __name__ == "__main__":
    main()
