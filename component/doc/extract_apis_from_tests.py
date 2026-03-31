"""Extract TF/PT APIs from all test files for batch doc downloads.

Extract APIs from:
1. dev/tf_core/*.py - TF core tests
2. dev/tf_fuzz/*.py - TF fuzzing tests
3. dev/pt_migrated/*.py - PT migrated tests
"""
import re
import json
import argparse
from pathlib import Path
from typing import Set
from tqdm import tqdm

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def extract_apis_from_file(file_path: Path) -> tuple[Set[str], Set[str]]:
    """Extract TF and PT APIs from a file."""
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Extract TF APIs (tf.xxx)
        tf_apis = set(re.findall(r'tf\.\w+(?:\.\w+)*', content))
        
        # Extract PT APIs (torch.xxx)
        pt_apis = set(re.findall(r'torch\.\w+(?:\.\w+)*', content))
        
        return tf_apis, pt_apis
    except Exception as e:
        print(f"[WARN] Failed to read {file_path}: {e}")
        return set(), set()


def main():
    parser = argparse.ArgumentParser(description="Extract all APIs from test files")
    parser.add_argument(
        "--output",
        default="data/analysis/test_apis.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--tf-core-dir",
        default="dev/tf_core",
        help="TF core test directory"
    )
    parser.add_argument(
        "--tf-fuzz-dir",
        default="dev/tf_fuzz",
        help="TF fuzzing test directory"
    )
    parser.add_argument(
        "--pt-migrated-dir",
        default="dev/pt_migrated",
        help="PT migrated test directory"
    )
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    tf_core_dir = Path(args.tf_core_dir)
    tf_fuzz_dir = Path(args.tf_fuzz_dir)
    pt_migrated_dir = Path(args.pt_migrated_dir)
    
    all_tf_apis: Set[str] = set()
    all_pt_apis: Set[str] = set()
    
    # Extract from TF core tests
    if tf_core_dir.exists():
        print(f"[INFO] Scanning {tf_core_dir} ...")
        for f in tqdm(list(tf_core_dir.glob("*.py")), desc="TF core"):
            tf_apis, pt_apis = extract_apis_from_file(f)
            all_tf_apis.update(tf_apis)
            all_pt_apis.update(pt_apis)
    
    # Extract from TF fuzzing tests
    if tf_fuzz_dir.exists():
        print(f"[INFO] Scanning {tf_fuzz_dir} ...")
        for f in tqdm(list(tf_fuzz_dir.glob("*.py")), desc="TF fuzz"):
            tf_apis, pt_apis = extract_apis_from_file(f)
            all_tf_apis.update(tf_apis)
            all_pt_apis.update(pt_apis)
    
    # Extract from PT migrated tests
    if pt_migrated_dir.exists():
        print(f"[INFO] Scanning {pt_migrated_dir} ...")
        for f in tqdm(list(pt_migrated_dir.glob("*.py")), desc="PT migrated"):
            tf_apis, pt_apis = extract_apis_from_file(f)
            all_tf_apis.update(tf_apis)
            all_pt_apis.update(pt_apis)
    
    # Write output file
    with output_path.open("w", encoding="utf-8") as fout:
        # Write TF APIs
        for api in sorted(all_tf_apis):
            fout.write(json.dumps({
                "framework": "tensorflow",
                "api": api
            }, ensure_ascii=False) + "\n")
        
        # Write PT APIs
        for api in sorted(all_pt_apis):
            fout.write(json.dumps({
                "framework": "pytorch",
                "api": api
            }, ensure_ascii=False) + "\n")
    
    print("\n[DONE] Extraction complete:")
    print(f"  - TF APIs: {len(all_tf_apis)}")
    print(f"  - PT APIs: {len(all_pt_apis)}")
    print(f"  - Output file: {output_path}")


if __name__ == "__main__":
    main()


