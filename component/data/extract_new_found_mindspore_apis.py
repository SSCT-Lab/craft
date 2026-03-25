# ./component/data/extract_new_found_mindspore_apis.py
"""从验证日志中提取新发现的高置信度 PyTorch -> MindSpore API 映射（原来值为"无对应实现"），生成新的 CSV 文件"""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Dict

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]

# 默认日志目录
LOG_DIR = ROOT / "component" / "data" / "llm_logs"

# 默认输入 CSV 文件路径
DEFAULT_INPUT_CSV = ROOT / "component" / "data" / "api_mappings.csv"

# 默认输出 CSV 文件路径
DEFAULT_OUTPUT_CSV = ROOT / "component" / "data" / "ms_api_mappings_updated.csv"


def parse_mindspore_validation_log(log_path: Path) -> List[Dict[str, str]]:
    """
    解析 MindSpore 验证日志文件，提取每条记录的信息
    
    Returns:
        包含每条记录信息的字典列表
    """
    records: List[Dict[str, str]] = []
    
    with log_path.open("r", encoding="utf-8") as f:
        content = f.read()
    
    # 按分隔线分割记录
    blocks = re.split(r'-{50,}', content)
    
    for block in blocks:
        block = block.strip()
        if not block or "序号:" not in block:
            continue
        
        record = {}
        
        # 提取序号
        idx_match = re.search(r'序号:\s*(\d+)', block)
        if idx_match:
            record["index"] = idx_match.group(1)
        
        # 提取 PyTorch API
        pt_match = re.search(r'PyTorch API:\s*(.+)', block)
        if pt_match:
            record["pytorch_api"] = pt_match.group(1).strip()
        
        # 提取原 MindSpore API（兼容多种命名）
        orig_ms_patterns = [
            r'原 MindSpore API:\s*(.+)',
            r'原MindSpore API:\s*(.+)',
            r'原 MS API:\s*(.+)',
        ]
        for pattern in orig_ms_patterns:
            orig_ms_match = re.search(pattern, block)
            if orig_ms_match:
                record["original_mindspore_api"] = orig_ms_match.group(1).strip()
                break
        
        # 提取验证后 MindSpore API（兼容多种命名）
        validated_ms_patterns = [
            r'验证后 MindSpore API:\s*(.+)',
            r'验证后MindSpore API:\s*(.+)',
            r'验证后 MS API:\s*(.+)',
        ]
        for pattern in validated_ms_patterns:
            validated_ms_match = re.search(pattern, block)
            if validated_ms_match:
                record["validated_mindspore_api"] = validated_ms_match.group(1).strip()
                break
        
        # 提取置信度
        confidence_match = re.search(r'置信度:\s*(.+)', block)
        if confidence_match:
            record["confidence"] = confidence_match.group(1).strip()
        
        # 提取是否修改
        changed_match = re.search(r'是否修改:\s*(.+)', block)
        if changed_match:
            record["changed"] = changed_match.group(1).strip()
        
        # 提取理由
        reason_match = re.search(r'理由:\s*(.+)', block, re.DOTALL)
        if reason_match:
            reason_text = reason_match.group(1).strip()
            # 只取第一行（避免包含 LLM 完整输出）
            reason_text = reason_text.split('\n')[0].strip()
            record["reason"] = reason_text
        
        if record.get("pytorch_api"):
            records.append(record)
    
    return records


def filter_new_found_high_confidence(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    筛选出满足条件的记录：
    1. 原 MindSpore API 是 "无对应实现"
    2. 验证后 MindSpore API 有具体值（不是"无对应实现"）
    3. 置信度为 high
    """
    filtered = []
    for record in records:
        original_ms = record.get("original_mindspore_api", "")
        validated_ms = record.get("validated_mindspore_api", "")
        confidence = record.get("confidence", "")
        
        # 条件1: 原 API 是"无对应实现"
        if original_ms != "无对应实现":
            continue
        
        # 条件2: 验证后有具体 API（不是"无对应实现"）
        if validated_ms == "无对应实现" or not validated_ms:
            continue
        
        # 条件3: 置信度为 high
        if confidence.lower() != "high":
            continue
        
        filtered.append(record)
    
    return filtered


def read_original_mappings(csv_path: Path) -> Dict[str, str]:
    """
    读取原始 CSV 文件中的 pytorch-api 到 mindspore-api 的映射
    
    Returns:
        字典 {pytorch_api: mindspore_api}
    """
    mappings = {}
    
    if not csv_path.exists():
        print(f"[WARNING] 输入 CSV 文件不存在: {csv_path}，将创建新文件")
        return mappings
    
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt_api = row.get("pytorch-api", "")
            ms_api = row.get("mindspore-api", "")
            if pt_api:
                mappings[pt_api] = ms_api
    
    return mappings


def create_updated_csv(
    original_mappings: Dict[str, str],
    new_mappings: List[Dict[str, str]],
    output_path: Path,
) -> int:
    """
    创建更新后的 CSV 文件，只包含 pytorch-api 和 mindspore-api 两列
    
    Args:
        original_mappings: 原始映射字典
        new_mappings: 新发现的映射列表
        output_path: 输出文件路径
    
    Returns:
        更新的记录数
    """
    # 合并映射：用新发现的映射更新原映射
    merged = original_mappings.copy()
    updated_count = 0
    
    for record in new_mappings:
        pt_api = record.get("pytorch_api", "")
        validated_ms = record.get("validated_mindspore_api", "")
        if pt_api and validated_ms:
            old_value = merged.get(pt_api, "")
            if old_value != validated_ms:
                merged[pt_api] = validated_ms
                updated_count += 1
                print(f"  [更新] {pt_api}: {old_value} -> {validated_ms}")
    
    # 写入 CSV（只保留 pytorch-api 和 mindspore-api 两列）
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pytorch-api", "mindspore-api"])
        
        # 按 PyTorch API 名称排序
        for pt_api in sorted(merged.keys()):
            ms_api = merged[pt_api]
            writer.writerow([pt_api, ms_api])
    
    return updated_count


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="从 MindSpore 验证日志中提取新发现的高置信度 API 映射"
    )
    parser.add_argument(
        "--log",
        "-l",
        required=True,
        help="MindSpore 验证日志文件路径",
    )
    parser.add_argument(
        "--input-csv",
        "-i",
        default=str(DEFAULT_INPUT_CSV),
        help="输入的原始 CSV 文件路径（默认：component/data/api_mappings.csv）",
    )
    parser.add_argument(
        "--output-csv",
        "-o",
        default=str(DEFAULT_OUTPUT_CSV),
        help="输出的 CSV 文件路径（默认：component/data/ms_api_mappings_updated.csv）",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="显示详细信息（包括映射的 MindSpore API 和理由）",
    )

    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[ERROR] 日志文件不存在: {log_path}")
        return

    print(f"[INFO] 正在解析日志文件: {log_path}")
    records = parse_mindspore_validation_log(log_path)
    print(f"[INFO] 共解析到 {len(records)} 条记录")

    # 筛选满足条件的记录
    filtered = filter_new_found_high_confidence(records)
    print(f"[INFO] 满足条件的记录数（原值为'无对应实现' + 验证后有具体API + 置信度high）: {len(filtered)}")

    # 打印详细信息
    if filtered:
        print("\n" + "=" * 70)
        print("新发现的高置信度 MindSpore API 映射")
        print("=" * 70)
        for i, record in enumerate(filtered, start=1):
            pt_api = record.get("pytorch_api", "")
            validated_ms = record.get("validated_mindspore_api", "")
            reason = record.get("reason", "")
            
            if args.verbose:
                print(f"\n{i}. {pt_api}")
                print(f"   -> {validated_ms}")
                print(f"   理由: {reason}")
            else:
                print(f"{i}. {pt_api} -> {validated_ms}")
        print("=" * 70)

    # 读取原始映射
    input_csv_path = Path(args.input_csv)
    print(f"\n[INFO] 正在读取原始 CSV 文件: {input_csv_path}")
    original_mappings = read_original_mappings(input_csv_path)
    print(f"[INFO] 原始映射数: {len(original_mappings)}")

    # 创建更新后的 CSV
    output_csv_path = Path(args.output_csv)
    print(f"\n[INFO] 正在生成更新后的 CSV 文件: {output_csv_path}")
    updated_count = create_updated_csv(original_mappings, filtered, output_csv_path)
    
    print(f"\n[SUCCESS] 已生成 CSV 文件: {output_csv_path}")
    print(f"[SUCCESS] 总映射数: {len(original_mappings)}")
    print(f"[SUCCESS] 本次更新: {updated_count} 条记录")
    
    # 打印纯 API 名称列表（方便复制）
    if filtered:
        print("\n" + "=" * 70)
        print("【新发现的 PyTorch API 名称列表（纯文本）】")
        print("=" * 70)
        for record in filtered:
            print(record.get("pytorch_api", ""))


if __name__ == "__main__":
    main()
