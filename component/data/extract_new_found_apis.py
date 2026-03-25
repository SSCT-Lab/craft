# ./component/data/extract_new_found_apis.py
"""从验证日志中提取新发现的高置信度 API 映射（原来值为“无对应实现”），并可选择更新 CSV 文件"""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Dict

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]

# 默认日志目录
LOG_DIR = ROOT / "component" / "data" / "llm_logs"

# 默认 CSV 文件路径
DEFAULT_CSV_PATH = ROOT / "component" / "data" / "api_mappings.csv"


def parse_validation_log(log_path: Path) -> List[Dict[str, str]]:
    """
    解析验证日志文件，提取每条记录的信息
    
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
        
        # 提取原 TensorFlow API
        orig_tf_match = re.search(r'原 TensorFlow API:\s*(.+)', block)
        if orig_tf_match:
            record["original_tf_api"] = orig_tf_match.group(1).strip()
        
        # 提取验证后 TensorFlow API
        validated_tf_match = re.search(r'验证后 TensorFlow API:\s*(.+)', block)
        if validated_tf_match:
            record["validated_tf_api"] = validated_tf_match.group(1).strip()
        
        # 提取置信度
        confidence_match = re.search(r'置信度:\s*(.+)', block)
        if confidence_match:
            record["confidence"] = confidence_match.group(1).strip()
        
        # 提取是否修改
        changed_match = re.search(r'是否修改:\s*(.+)', block)
        if changed_match:
            record["changed"] = changed_match.group(1).strip()
        
        # 提取理由
        reason_match = re.search(r'理由:\s*(.+)', block)
        if reason_match:
            record["reason"] = reason_match.group(1).strip()
        
        if record.get("pytorch_api"):
            records.append(record)
    
    return records


def filter_new_found_high_confidence(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    筛选出满足条件的记录：
    1. 原 TensorFlow API 是 "无对应实现"
    2. 验证后 TensorFlow API 有具体值（不是"无对应实现"）
    3. 置信度为 high
    """
    filtered = []
    for record in records:
        original_tf = record.get("original_tf_api", "")
        validated_tf = record.get("validated_tf_api", "")
        confidence = record.get("confidence", "")
        
        # 条件1: 原 API 是"无对应实现"
        if original_tf != "无对应实现":
            continue
        
        # 条件2: 验证后有具体 API（不是"无对应实现"）
        if validated_tf == "无对应实现" or not validated_tf:
            continue
        
        # 条件3: 置信度为 high
        if confidence.lower() != "high":
            continue
        
        filtered.append(record)
    
    return filtered


def update_csv_with_new_mappings(
    csv_path: Path,
    new_mappings: List[Dict[str, str]],
    output_path: Path = None,
) -> int:
    """
    根据新发现的映射更新 CSV 文件
    
    Args:
        csv_path: 原 CSV 文件路径
        new_mappings: 新发现的映射列表
        output_path: 输出文件路径（不指定则覆盖原文件）
    
    Returns:
        更新的记录数
    """
    if output_path is None:
        output_path = csv_path
    
    # 构建 PyTorch API -> 新 TensorFlow API 的映射字典
    update_dict = {}
    for record in new_mappings:
        pt_api = record.get("pytorch_api", "")
        validated_tf = record.get("validated_tf_api", "")
        if pt_api and validated_tf:
            update_dict[pt_api] = validated_tf
    
    # 读取原 CSV
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    # 更新记录
    updated_count = 0
    for row in rows:
        pt_api = row.get("pytorch-api", "")
        if pt_api in update_dict:
            old_value = row.get("tensorflow-api", "")
            new_value = update_dict[pt_api]
            if old_value != new_value:
                row["tensorflow-api"] = new_value
                updated_count += 1
                print(f"  [更新] {pt_api}: {old_value} -> {new_value}")
    
    # 写入更新后的 CSV
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return updated_count


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="从验证日志中提取新发现的高置信度 API 映射"
    )
    parser.add_argument(
        "--log",
        "-l",
        required=True,
        help="验证日志文件路径",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="输出文件路径（不指定则打印到控制台）",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="显示详细信息（包括映射的 TensorFlow API 和理由）",
    )
    parser.add_argument(
        "--update-csv",
        "-u",
        action="store_true",
        help="更新 api_mappings.csv 文件中的对应记录",
    )
    parser.add_argument(
        "--csv-path",
        "-c",
        default=str(DEFAULT_CSV_PATH),
        help="api_mappings.csv 文件路径（默认为 component/data/api_mappings.csv）",
    )
    parser.add_argument(
        "--output-csv",
        help="更新后的 CSV 输出文件路径（不指定则覆盖原文件）",
    )

    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[ERROR] 日志文件不存在: {log_path}")
        return

    print(f"[INFO] 正在解析日志文件: {log_path}")
    records = parse_validation_log(log_path)
    print(f"[INFO] 共解析到 {len(records)} 条记录")

    # 筛选满足条件的记录
    filtered = filter_new_found_high_confidence(records)
    print(f"[INFO] 满足条件的记录数: {len(filtered)}")

    # 构建输出内容
    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("新发现的高置信度 API 映射")
    output_lines.append(f"条件: 原API为'无对应实现' + 验证后有具体API + 置信度为high")
    output_lines.append(f"符合条件的记录数: {len(filtered)}")
    output_lines.append("=" * 70)
    output_lines.append("")

    for i, record in enumerate(filtered, start=1):
        pt_api = record.get("pytorch_api", "")
        validated_tf = record.get("validated_tf_api", "")
        reason = record.get("reason", "")
        
        if args.verbose:
            output_lines.append(f"{i}. {pt_api}")
            output_lines.append(f"   -> {validated_tf}")
            output_lines.append(f"   理由: {reason}")
            output_lines.append("")
        else:
            output_lines.append(f"{i}. {pt_api} -> {validated_tf}")

    output_text = "\n".join(output_lines)

    # 输出结果
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"[SUCCESS] 结果已保存到: {output_path}")
    else:
        print("\n" + output_text)

    # 如果指定了 --update-csv，则更新 CSV 文件
    if args.update_csv:
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            print(f"[ERROR] CSV 文件不存在: {csv_path}")
            return
        
        output_csv_path = Path(args.output_csv) if args.output_csv else None
        print(f"\n[INFO] 正在更新 CSV 文件: {csv_path}")
        if output_csv_path:
            print(f"[INFO] 输出文件路径: {output_csv_path}")
        updated_count = update_csv_with_new_mappings(csv_path, filtered, output_csv_path)
        print(f"[SUCCESS] 已更新 {updated_count} 条记录")
    # 额外打印纯 API 名称列表（方便复制）
    print("\n" + "=" * 70)
    print("【PyTorch API 名称列表（纯文本）】")
    print("=" * 70)
    for record in filtered:
        print(record.get("pytorch_api", ""))


if __name__ == "__main__":
    main()
