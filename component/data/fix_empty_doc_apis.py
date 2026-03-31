# ./component/data/fix_empty_doc_apis.py
"""
Fix API mappings where TensorFlow docs are empty.

Steps:
1. Read the list of APIs with failed doc fetches
2. Look up matching records in api_mappings_validated.csv
3. Try to fetch docs for the tensorflow-api in the validated file
4. If docs exist or value is "无对应实现", update api_mappings.csv with the validated value
5. Generate a new CSV result file
"""

import csv
import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from component.doc.doc_crawler_factory import get_doc_content


# APIs whose doc fetch failed (PyTorch API -> current TensorFlow API)
EMPTY_DOC_APIS = [
    ("torch.addmm", "tf.linalg.LinearOperator.matmul"),
    ("torch.copysign", "tf.math.copysign"),
    ("torch.count_nonzero", "tf.count_nonzero"),
    ("torch.floor_divide", "tf.floor_div"),
    ("torch.frac", "tf.math.fractional_part"),
    ("torch.gcd", "tf.math.gcd"),
    ("torch.hypot", "tf.math.hypot"),
    ("torch.isreal", "tf.math.is_real"),
    ("torch.kron", "tf.linalg.kronecker"),
    ("torch.lerp", "tf.raw_ops.Lerp"),
    ("torch.log10", "tf.math.log10"),
    ("torch.logaddexp", "tf.math.logaddexp"),
    ("torch.logaddexp2", "tf.math.logaddexp"),
    ("torch.matrix_power", "tf.linalg.matrix_power"),
    ("torch.median", "tf.math.reduce_median"),
    ("torch.nn.CTCLoss", "tf.keras.losses.CTCloss"),
    ("torch.nn.GELU", "tf.keras.layers.Activation('gelu')"),
    ("torch.nn.Hardsigmoid", "tf.keras.layers.HardSigmoid"),
    ("torch.nn.Hardswish", "tf.keras.layers.Activation('hard_swish')"),
    ("torch.nn.MarginRankingLoss", "tf.keras.losses.MarginRankingLoss"),
    ("torch.nn.SiLU", "tf.keras.layers.Activation('swish')"),
    ("torch.nn.TransformerDecoderLayer", "tf.keras.layers.TransformerDecoderLayer"),
    ("torch.nn.TransformerEncoderLayer", "tf.keras.layers.TransformerEncoderLayer"),
    ("torch.nn.TripletMarginLoss", "tf.keras.losses.TripletSemiHardLoss"),
    ("torch.nn.functional.adaptive_max_pool2d", "tf.nn.adaptive_max_pooling_2d"),
    ("torch.nn.functional.celu", "tf.nn.celu"),
    ("torch.nn.functional.hardshrink", "tf.nn.hard_shrink"),
    ("torch.nn.functional.instance_norm", "tf.nn.instance_norm"),
    ("torch.nn.functional.l1_loss", "tf.keras.losses.mean_absolute_error"),
    ("torch.nn.functional.layer_norm", "tf.keras.utils.layer_normalization"),
    ("torch.nn.functional.logsigmoid", "tf.nn.log_sigmoid"),
    ("torch.nn.functional.mse_loss", "tf.keras.losses.mean_squared_error"),
    ("torch.nn.functional.rrelu", "tf.nn.rrelu"),
    ("torch.nn.functional.rrelu_", "tf.nn.rrelu"),
    ("torch.nn.functional.softshrink", "tf.nn.softshrink"),
    ("torch.nn.functional.triplet_margin_loss", "tf.keras.losses.TripletSemiHardLoss"),
    ("torch.quantile", "tf.math.quantile"),
    ("torch.rand_like", "tf.random.uniform_like"),
    ("torch.trunc", "tf.truncated_normal"),
]


def load_validated_csv(csv_path: str) -> Dict[str, Tuple[str, str, str]]:
    """
    Load the validated CSV file.

    Returns:
        Dict[pytorch_api, (tensorflow_api, confidence, changed)]
    """
    result = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt_api = row['pytorch-api']
            tf_api = row['tensorflow-api']
            confidence = row.get('confidence', '')
            changed = row.get('changed', '')
            result[pt_api] = (tf_api, confidence, changed)
    return result


def load_original_csv(csv_path: str) -> List[Dict[str, str]]:
    """
    Load the original CSV file.

    Returns:
        List of rows as dicts
    """
    result = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            result.append(dict(row))
    return result


def check_tf_doc_exists(tf_api: str) -> Tuple[bool, str]:
    """
    Check whether a TensorFlow API doc exists.

    Args:
        tf_api: TensorFlow API name

    Returns:
        (doc_exists, doc content or error info)
    """
    # If it's "无对应实现", return True directly
    if tf_api == "无对应实现":
        return True, "无对应实现"
    
    # Handle API names with parameters, e.g. tf.keras.layers.Activation('gelu')
    # Extract base API name
    base_api = tf_api.split('(')[0].strip()
    
    try:
        doc_content = get_doc_content(base_api, "tensorflow")
        if doc_content and len(doc_content.strip()) > 50:
            return True, doc_content[:200] + "..."
        else:
            return False, f"Doc is empty or too short: {len(doc_content) if doc_content else 0} chars"
    except Exception as e:
        return False, f"Fetch failed: {str(e)}"


def fix_empty_doc_apis(
    original_csv: str,
    validated_csv: str,
    output_csv: str,
    dry_run: bool = False
) -> Tuple[int, int, List[Tuple[str, str, str, str, str]]]:
    """
    Fix API mappings with empty docs.

    Args:
        original_csv: Original CSV path
        validated_csv: Validated CSV path
        output_csv: Output CSV path
        dry_run: Whether to print only and skip writing

    Returns:
        (updated_count, skipped_count, update_details)
    """
    # Load files
    validated_data = load_validated_csv(validated_csv)
    original_data = load_original_csv(original_csv)
    
    # Map pytorch-api to row index
    pt_to_index = {row['pytorch-api']: i for i, row in enumerate(original_data)}
    
    # Counters
    updated_count = 0
    skipped_count = 0
    update_details = []  # (pt_api, old_tf_api, new_tf_api, status, reason)
    
    print("=" * 80)
    print("Start checking API mappings with empty docs")
    print("=" * 80)
    
    for pt_api, current_tf_api in EMPTY_DOC_APIS:
        print(f"\nChecking: {pt_api}")
        print(f"  Current mapping: {current_tf_api}")
        
        # Look up in validated file
        if pt_api not in validated_data:
            print("  ❌ Not found in validated file")
            skipped_count += 1
            update_details.append((pt_api, current_tf_api, "", "skipped", "missing in validated file"))
            continue
        
        validated_tf_api, confidence, changed = validated_data[pt_api]
        print(f"  Validated mapping: {validated_tf_api} (confidence={confidence}, changed={changed})")
        
        # If validated value is the same as current, skip
        if validated_tf_api == current_tf_api:
            print("  ⏭️ Same value, skip")
            skipped_count += 1
            update_details.append((pt_api, current_tf_api, validated_tf_api, "skipped", "same value"))
            continue
        
        # Try to fetch TensorFlow docs for the validated API
        doc_exists, doc_info = check_tf_doc_exists(validated_tf_api)
        
        if doc_exists:
            print("  ✅ Docs exist for validated value")
            print(f"     Doc preview: {doc_info[:100]}...")
            
            # Find and update in original data
            if pt_api in pt_to_index:
                idx = pt_to_index[pt_api]
                old_value = original_data[idx]['tensorflow-api']
                original_data[idx]['tensorflow-api'] = validated_tf_api
                updated_count += 1
                update_details.append((pt_api, old_value, validated_tf_api, "updated", "docs exist"))
                print(f"  📝 Update: {old_value} -> {validated_tf_api}")
            else:
                skipped_count += 1
                update_details.append((pt_api, current_tf_api, validated_tf_api, "skipped", "missing in original file"))
                print("  ❌ Not found in original file")
        else:
            # Neither original nor validated docs exist; mark as "无对应实现"
            print(f"  ❌ Docs also missing for validated value: {doc_info}")
            print("  ⚠️ Docs missing for both original and validated; mark as 无对应实现")
            
            if pt_api in pt_to_index:
                idx = pt_to_index[pt_api]
                old_value = original_data[idx]['tensorflow-api']
                original_data[idx]['tensorflow-api'] = "无对应实现"
                updated_count += 1
                update_details.append(
                    (pt_api, old_value, "无对应实现", "updated", "docs missing for both; marked as 无对应实现")
                )
                print(f"  📝 Update: {old_value} -> 无对应实现")
            else:
                skipped_count += 1
                update_details.append((pt_api, current_tf_api, "无对应实现", "skipped", "missing in original file"))
    
    # Write results
    if not dry_run:
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['pytorch-api', 'tensorflow-api']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in original_data:
                writer.writerow({
                    'pytorch-api': row['pytorch-api'],
                    'tensorflow-api': row['tensorflow-api']
                })
        print(f"\n✅ Wrote result file: {output_csv}")
    else:
        print("\n[DRY RUN] File not written")
    
    return updated_count, skipped_count, update_details


def print_summary(updated: int, skipped: int, details: List[Tuple[str, str, str, str, str]]):
    """Print summary."""
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total processed: {len(EMPTY_DOC_APIS)} APIs")
    print(f"Updated: {updated}")
    print(f"Skipped: {skipped}")
    
    if updated > 0:
        print("\n[Updated mappings]")
        for pt_api, old_tf, new_tf, status, reason in details:
            if status == "updated":
                print(f"  {pt_api}:")
                print(f"    Old value: {old_tf}")
                print(f"    New value: {new_tf}")
    
    skipped_details = [(d[0], d[4]) for d in details if d[3] == "skipped"]
    if skipped_details:
        print("\n[Skipped mappings]")
        for pt_api, reason in skipped_details:
            print(f"  {pt_api}: {reason}")


def main():
    parser = argparse.ArgumentParser(
        description='Fix API mappings where TensorFlow docs are empty'
    )
    parser.add_argument(
        '--original-csv',
        default='component/data/api_mappings.csv',
        help='Original CSV path (default: component/data/api_mappings.csv)'
    )
    parser.add_argument(
        '--validated-csv',
        default='component/data/api_mappings_validated.csv',
        help='Validated CSV path (default: component/data/api_mappings_validated.csv)'
    )
    parser.add_argument(
        '--output-csv',
        default='component/data/api_mappings_fixed.csv',
        help='Output CSV path (default: component/data/api_mappings_fixed.csv)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print only; do not write file'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    original_csv = os.path.join(project_root, args.original_csv) if not os.path.isabs(args.original_csv) else args.original_csv
    validated_csv = os.path.join(project_root, args.validated_csv) if not os.path.isabs(args.validated_csv) else args.validated_csv
    output_csv = os.path.join(project_root, args.output_csv) if not os.path.isabs(args.output_csv) else args.output_csv
    
    # Check files exist
    if not os.path.exists(original_csv):
        print(f"Error: Original CSV not found: {original_csv}")
        sys.exit(1)
    if not os.path.exists(validated_csv):
        print(f"Error: Validated CSV not found: {validated_csv}")
        sys.exit(1)
    
    print(f"Original CSV: {original_csv}")
    print(f"Validated CSV: {validated_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Dry Run: {args.dry_run}")
    
    # Execute fix
    updated, skipped, details = fix_empty_doc_apis(
        original_csv,
        validated_csv,
        output_csv,
        args.dry_run
    )
    
    # Print summary
    print_summary(updated, skipped, details)


if __name__ == '__main__':
    main()
