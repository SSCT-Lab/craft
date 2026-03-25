# ./component/data/extract_changed_mindspore_apis.py
"""
从 MindSpore 验证日志中提取需要修改的 API 映射：
1. 原API和验证后API都有具体值但值不同 + 置信度为high
2. 原API有具体值 + 验证后API为"无对应实现" + 置信度为high

支持可选地更新 CSV 文件（生成只包含 pytorch-api 和 mindspore-api 的 CSV）
支持文档验证功能：验证 MindSpore API 文档是否存在，不存在则标记为"无对应实现"
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 默认日志目录
LOG_DIR = ROOT / "component" / "data" / "llm_logs"

# 默认输入 CSV 文件路径
DEFAULT_INPUT_CSV = ROOT / "component" / "data" / "api_mappings.csv"

# 默认输出 CSV 文件路径
DEFAULT_OUTPUT_CSV = ROOT / "component" / "data" / "ms_api_mappings_updated.csv"


def check_mindspore_doc_exists(mindspore_api: str) -> Tuple[bool, str]:
    """
    检查 MindSpore API 文档是否存在
    
    Args:
        mindspore_api: MindSpore API 名称
        
    Returns:
        (文档是否存在, 文档内容或错误信息)
    """
    # 延迟导入，避免未启用文档验证时的导入开销
    from component.doc.doc_crawler_factory import get_doc_content
    
    # 如果是"无对应实现"，直接返回 True
    if mindspore_api == "无对应实现":
        return True, "无对应实现"
    
    # 处理带参数的 API 名称
    base_api = mindspore_api.split('(')[0].strip()
    
    try:
        doc_content = get_doc_content(base_api, "mindspore")
        if doc_content and len(doc_content.strip()) > 300:
            return True, doc_content[:200] + "..."
        else:
            return False, f"文档为空或太短: {len(doc_content) if doc_content else 0} 字符"
    except Exception as e:
        return False, f"爬取失败: {str(e)}"


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
            r'原 MS API:\s*(.+)',
            r'原MindSpore API:\s*(.+)',
        ]
        for pattern in orig_ms_patterns:
            orig_ms_match = re.search(pattern, block)
            if orig_ms_match:
                record["original_mindspore_api"] = orig_ms_match.group(1).strip()
                break
        
        # 提取验证后 MindSpore API（兼容多种命名）
        validated_ms_patterns = [
            r'验证后 MindSpore API:\s*(.+)',
            r'验证后 MS API:\s*(.+)',
            r'验证后MindSpore API:\s*(.+)',
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


def filter_changed_mappings(
    records: List[Dict[str, str]]
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    筛选出需要修改的记录，分为两类：
    
    类型1: 原API和验证后API都有具体值但值不同 + 置信度为high（映射需更新）
    类型2: 原API有具体值 + 验证后API为"无对应实现" + 置信度为high（原映射错误）
    
    Returns:
        (类型1记录列表, 类型2记录列表)
    """
    type1_records: List[Dict[str, str]] = []  # 映射更新
    type2_records: List[Dict[str, str]] = []  # 原映射错误
    
    for record in records:
        original_ms = record.get("original_mindspore_api", "")
        validated_ms = record.get("validated_mindspore_api", "")
        confidence = record.get("confidence", "")
        
        # 置信度必须为 high
        if confidence.lower() != "high":
            continue
        
        # 原 API 必须有具体值（不是"无对应实现"）
        if original_ms == "无对应实现" or not original_ms:
            continue
        
        # 类型1: 两者都有具体值但不同
        if validated_ms and validated_ms != "无对应实现" and original_ms != validated_ms:
            type1_records.append(record)
        
        # 类型2: 验证后为"无对应实现"
        elif validated_ms == "无对应实现":
            type2_records.append(record)
    
    return type1_records, type2_records


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


def update_csv_with_changed_mappings(
    original_mappings: Dict[str, str],
    type1_records: List[Dict[str, str]],
    type2_records: List[Dict[str, str]],
    output_path: Path,
    verify_doc: bool = False,
) -> Tuple[int, int, int]:
    """
    根据需要修改的映射更新 CSV 文件（生成只包含 pytorch-api 和 mindspore-api 的 CSV）
    
    Args:
        original_mappings: 原始映射字典
        type1_records: 类型1记录（映射更新）
        type2_records: 类型2记录（标记为无对应实现）
        output_path: 输出文件路径
        verify_doc: 是否验证 MindSpore API 文档存在性
    
    Returns:
        (类型1更新数, 类型2更新数, 文档验证失败转为无对应实现的数量)
    """
    # 构建更新字典
    # 类型1: PyTorch API -> (验证后的 MindSpore API, 原 MindSpore API)
    type1_dict = {}
    for record in type1_records:
        pt_api = record.get("pytorch_api", "")
        validated_ms = record.get("validated_mindspore_api", "")
        original_ms = record.get("original_mindspore_api", "")
        if pt_api and validated_ms:
            type1_dict[pt_api] = (validated_ms, original_ms)
    
    # 类型2: PyTorch API -> "无对应实现"
    type2_dict = {}
    for record in type2_records:
        pt_api = record.get("pytorch_api", "")
        if pt_api:
            type2_dict[pt_api] = "无对应实现"
    
    # 复制原始映射
    merged_mappings = original_mappings.copy()
    
    # 更新记录
    type1_count = 0
    type2_count = 0
    doc_fallback_count = 0  # 文档验证失败转为无对应实现的数量
    
    for pt_api in merged_mappings.keys():
        old_value = merged_mappings[pt_api]
        
        # 优先检查类型1（映射更新）
        if pt_api in type1_dict:
            validated_ms, original_ms = type1_dict[pt_api]
            new_value = validated_ms
            
            # 如果启用文档验证，检查 validated MindSpore API 的文档是否存在
            if verify_doc:
                print(f"\n  检查文档: {pt_api}")
                print(f"    原映射: {original_ms}")
                print(f"    新映射: {validated_ms}")
                
                doc_exists, doc_info = check_mindspore_doc_exists(validated_ms)
                
                if doc_exists:
                    print(f"    ✅ 新映射文档存在")
                    new_value = validated_ms
                else:
                    print(f"    ❌ 新映射文档不存在: {doc_info}")
                    print(f"    ⚠️ 标记为无对应实现")
                    new_value = "无对应实现"
                    doc_fallback_count += 1
            
            if old_value != new_value:
                merged_mappings[pt_api] = new_value
                type1_count += 1
                if new_value == "无对应实现":
                    print(f"  [类型1-文档不存在] {pt_api}: {old_value} -> 无对应实现")
                else:
                    print(f"  [类型1-更新映射] {pt_api}: {old_value} -> {new_value}")
        
        # 检查类型2（标记为无对应实现）
        elif pt_api in type2_dict:
            new_value = type2_dict[pt_api]
            if old_value != new_value:
                merged_mappings[pt_api] = new_value
                type2_count += 1
                print(f"  [类型2-无对应] {pt_api}: {old_value} -> {new_value}")
    
    # 写入 CSV（只保留 pytorch-api 和 mindspore-api 两列）
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pytorch-api", "mindspore-api"])
        
        # 按 PyTorch API 名称排序
        for pt_api in sorted(merged_mappings.keys()):
            ms_api = merged_mappings[pt_api]
            writer.writerow([pt_api, ms_api])
    
    return type1_count, type2_count, doc_fallback_count


def format_output(
    type1_records: List[Dict[str, str]],
    type2_records: List[Dict[str, str]],
    verbose: bool = False,
) -> str:
    """格式化输出结果"""
    lines = []
    
    # 标题
    lines.append("=" * 70)
    lines.append("需要修改的 MindSpore API 映射（高置信度）")
    lines.append("=" * 70)
    lines.append("")
    
    # 类型1: 映射更新
    lines.append("-" * 70)
    lines.append("【类型1】原API和验证后API都有具体值但值不同")
    lines.append(f"符合条件的记录数: {len(type1_records)}")
    lines.append("-" * 70)
    
    for i, record in enumerate(type1_records, start=1):
        pt_api = record.get("pytorch_api", "")
        original_ms = record.get("original_mindspore_api", "")
        validated_ms = record.get("validated_mindspore_api", "")
        reason = record.get("reason", "")
        
        if verbose:
            lines.append(f"{i}. {pt_api}")
            lines.append(f"   原映射: {original_ms}")
            lines.append(f"   新映射: {validated_ms}")
            lines.append(f"   理由: {reason}")
            lines.append("")
        else:
            lines.append(f"{i}. {pt_api}: {original_ms} -> {validated_ms}")
    
    lines.append("")
    
    # 类型2: 原映射错误
    lines.append("-" * 70)
    lines.append("【类型2】原API有具体值但验证后为'无对应实现'")
    lines.append(f"符合条件的记录数: {len(type2_records)}")
    lines.append("-" * 70)
    
    for i, record in enumerate(type2_records, start=1):
        pt_api = record.get("pytorch_api", "")
        original_ms = record.get("original_mindspore_api", "")
        reason = record.get("reason", "")
        
        if verbose:
            lines.append(f"{i}. {pt_api}")
            lines.append(f"   原映射: {original_ms}")
            lines.append(f"   新映射: 无对应实现")
            lines.append(f"   理由: {reason}")
            lines.append("")
        else:
            lines.append(f"{i}. {pt_api}: {original_ms} -> 无对应实现")
    
    lines.append("")
    
    # 汇总
    lines.append("=" * 70)
    lines.append(f"汇总: 类型1共 {len(type1_records)} 条，类型2共 {len(type2_records)} 条")
    lines.append(f"总计需修改: {len(type1_records) + len(type2_records)} 条")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="从 MindSpore 验证日志中提取需要修改的 API 映射（高置信度）"
    )
    parser.add_argument(
        "--log",
        "-l",
        required=True,
        help="MindSpore 验证日志文件路径",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="输出文本文件路径（不指定则打印到控制台）",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="显示详细信息（包括原映射、新映射和理由）",
    )
    parser.add_argument(
        "--update-csv",
        "-u",
        action="store_true",
        help="生成更新后的 CSV 文件",
    )
    parser.add_argument(
        "--input-csv",
        "-i",
        default=str(DEFAULT_INPUT_CSV),
        help="输入的原始 CSV 文件路径（默认：component/data/api_mappings.csv）",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="更新后的 CSV 输出路径（默认：component/data/ms_api_mappings_updated.csv）",
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["all", "1", "2"],
        default="all",
        help="筛选类型: all=全部, 1=仅类型1(映射更新), 2=仅类型2(无对应实现)",
    )
    parser.add_argument(
        "--verify-doc",
        action="store_true",
        help="启用文档验证：爬取验证 MindSpore API 文档是否存在，不存在则标记为'无对应实现'",
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
    type1_records, type2_records = filter_changed_mappings(records)
    
    # 根据 --type 参数过滤
    if args.type == "1":
        type2_records = []
    elif args.type == "2":
        type1_records = []
    
    print(f"[INFO] 类型1（映射更新）记录数: {len(type1_records)}")
    print(f"[INFO] 类型2（无对应实现）记录数: {len(type2_records)}")

    # 构建输出内容
    output_text = format_output(type1_records, type2_records, args.verbose)

    # 输出结果
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"[SUCCESS] 结果已保存到: {output_path}")
    else:
        print("\n" + output_text)

    # 如果指定了 --update-csv，则生成更新后的 CSV
    if args.update_csv:
        input_csv_path = Path(args.input_csv)
        print(f"\n[INFO] 正在读取原始 CSV 文件: {input_csv_path}")
        original_mappings = read_original_mappings(input_csv_path)
        print(f"[INFO] 原始映射数: {len(original_mappings)}")
        
        output_csv_path = Path(args.output_csv)
        print(f"\n[INFO] 正在生成更新后的 CSV 文件: {output_csv_path}")
        
        if args.verify_doc:
            print(f"[INFO] 已启用文档验证，将爬取 MindSpore API 文档进行验证")
        
        type1_count, type2_count, doc_fallback_count = update_csv_with_changed_mappings(
            original_mappings, type1_records, type2_records, output_csv_path, args.verify_doc
        )
        
        total_updated = type1_count + type2_count
        print(f"\n[SUCCESS] 已生成 CSV 文件: {output_csv_path}")
        print(f"[SUCCESS] 总映射数: {len(original_mappings)}")
        print(f"[SUCCESS] 本次更新: {total_updated} 条记录")
        print(f"  - 类型1（映射更新）: {type1_count} 条")
        print(f"  - 类型2（无对应实现）: {type2_count} 条")
        
        if args.verify_doc and doc_fallback_count > 0:
            print(f"  - 其中因文档不存在转为'无对应实现': {doc_fallback_count} 条")

    # 额外打印纯 API 名称列表（方便复制）
    if type1_records or type2_records:
        print("\n" + "=" * 70)
        print("【PyTorch API 名称列表（纯文本）】")
        print("=" * 70)
        
        if type1_records:
            print("\n--- 类型1: 映射更新 ---")
            for record in type1_records:
                print(record.get("pytorch_api", ""))
        
        if type2_records:
            print("\n--- 类型2: 无对应实现 ---")
            for record in type2_records:
                print(record.get("pytorch_api", ""))


if __name__ == "__main__":
    main()
