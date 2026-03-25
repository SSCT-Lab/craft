"""从所有测试文件中提取 TF/PT API，用于批量下载文档

从以下位置提取 API：
1. dev/tf_core/*.py - TF core 测试
2. dev/tf_fuzz/*.py - TF fuzzing 测试
3. dev/pt_migrated/*.py - PT 迁移测试
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
    """从文件中提取 TF 和 PT API"""
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # 提取 TF API (tf.xxx)
        tf_apis = set(re.findall(r'tf\.\w+(?:\.\w+)*', content))
        
        # 提取 PT API (torch.xxx)
        pt_apis = set(re.findall(r'torch\.\w+(?:\.\w+)*', content))
        
        return tf_apis, pt_apis
    except Exception as e:
        print(f"[WARN] 读取 {file_path} 失败: {e}")
        return set(), set()


def main():
    parser = argparse.ArgumentParser(description="从测试文件中提取所有 API")
    parser.add_argument(
        "--output",
        default="data/analysis/test_apis.jsonl",
        help="输出 JSONL 文件路径"
    )
    parser.add_argument(
        "--tf-core-dir",
        default="dev/tf_core",
        help="TF core 测试目录"
    )
    parser.add_argument(
        "--tf-fuzz-dir",
        default="dev/tf_fuzz",
        help="TF fuzzing 测试目录"
    )
    parser.add_argument(
        "--pt-migrated-dir",
        default="dev/pt_migrated",
        help="PT 迁移测试目录"
    )
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    tf_core_dir = Path(args.tf_core_dir)
    tf_fuzz_dir = Path(args.tf_fuzz_dir)
    pt_migrated_dir = Path(args.pt_migrated_dir)
    
    all_tf_apis: Set[str] = set()
    all_pt_apis: Set[str] = set()
    
    # 从 TF core 测试提取
    if tf_core_dir.exists():
        print(f"[INFO] 扫描 {tf_core_dir} ...")
        for f in tqdm(list(tf_core_dir.glob("*.py")), desc="TF core"):
            tf_apis, pt_apis = extract_apis_from_file(f)
            all_tf_apis.update(tf_apis)
            all_pt_apis.update(pt_apis)
    
    # 从 TF fuzzing 测试提取
    if tf_fuzz_dir.exists():
        print(f"[INFO] 扫描 {tf_fuzz_dir} ...")
        for f in tqdm(list(tf_fuzz_dir.glob("*.py")), desc="TF fuzz"):
            tf_apis, pt_apis = extract_apis_from_file(f)
            all_tf_apis.update(tf_apis)
            all_pt_apis.update(pt_apis)
    
    # 从 PT 迁移测试提取
    if pt_migrated_dir.exists():
        print(f"[INFO] 扫描 {pt_migrated_dir} ...")
        for f in tqdm(list(pt_migrated_dir.glob("*.py")), desc="PT migrated"):
            tf_apis, pt_apis = extract_apis_from_file(f)
            all_tf_apis.update(tf_apis)
            all_pt_apis.update(pt_apis)
    
    # 写入输出文件
    with output_path.open("w", encoding="utf-8") as fout:
        # 写入 TF APIs
        for api in sorted(all_tf_apis):
            fout.write(json.dumps({
                "framework": "tensorflow",
                "api": api
            }, ensure_ascii=False) + "\n")
        
        # 写入 PT APIs
        for api in sorted(all_pt_apis):
            fout.write(json.dumps({
                "framework": "pytorch",
                "api": api
            }, ensure_ascii=False) + "\n")
    
    print(f"\n[DONE] 提取完成:")
    print(f"  - TF APIs: {len(all_tf_apis)}")
    print(f"  - PT APIs: {len(all_pt_apis)}")
    print(f"  - 输出文件: {output_path}")


if __name__ == "__main__":
    main()


