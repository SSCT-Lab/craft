# ./component/data/fix_empty_doc_apis.py
"""
修复 TensorFlow 文档为空的 API 映射

功能：
1. 读取文档爬取失败的 API 列表
2. 在 api_mappings_validated.csv 中查找对应记录
3. 尝试爬取 validated 文件中的 tensorflow-api 的文档
4. 如果文档存在或值为"无对应实现"，则用 validated 的值更新 api_mappings.csv
5. 生成新的 csv 结果文件
"""

import csv
import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from component.doc.doc_crawler_factory import get_doc_content


# 文档爬取失败的 API 列表（PyTorch API -> 当前的 TensorFlow API）
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
    加载 validated CSV 文件
    
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
    加载原始 CSV 文件
    
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
    检查 TensorFlow API 文档是否存在
    
    Args:
        tf_api: TensorFlow API 名称
        
    Returns:
        (文档是否存在, 文档内容或错误信息)
    """
    # 如果是"无对应实现"，直接返回 True
    if tf_api == "无对应实现":
        return True, "无对应实现"
    
    # 处理带参数的 API 名称，如 tf.keras.layers.Activation('gelu')
    # 提取基础 API 名称
    base_api = tf_api.split('(')[0].strip()
    
    try:
        doc_content = get_doc_content(base_api, "tensorflow")
        if doc_content and len(doc_content.strip()) > 50:
            return True, doc_content[:200] + "..."
        else:
            return False, f"文档为空或文档太短: {len(doc_content) if doc_content else 0} 字符"
    except Exception as e:
        return False, f"爬取失败: {str(e)}"


def fix_empty_doc_apis(
    original_csv: str,
    validated_csv: str,
    output_csv: str,
    dry_run: bool = False
) -> Tuple[int, int, List[Tuple[str, str, str, str, str]]]:
    """
    修复文档为空的 API 映射
    
    Args:
        original_csv: 原始 CSV 文件路径
        validated_csv: 验证后的 CSV 文件路径
        output_csv: 输出 CSV 文件路径
        dry_run: 是否只打印不写入
        
    Returns:
        (更新数量, 跳过数量, 更新详情列表)
    """
    # 加载文件
    validated_data = load_validated_csv(validated_csv)
    original_data = load_original_csv(original_csv)
    
    # 创建 pytorch-api 到行索引的映射
    pt_to_index = {row['pytorch-api']: i for i, row in enumerate(original_data)}
    
    # 统计
    updated_count = 0
    skipped_count = 0
    update_details = []  # (pt_api, old_tf_api, new_tf_api, status, reason)
    
    print("=" * 80)
    print("开始检查文档为空的 API 映射")
    print("=" * 80)
    
    for pt_api, current_tf_api in EMPTY_DOC_APIS:
        print(f"\n检查: {pt_api}")
        print(f"  当前映射: {current_tf_api}")
        
        # 在 validated 文件中查找
        if pt_api not in validated_data:
            print(f"  ❌ 在 validated 文件中未找到")
            skipped_count += 1
            update_details.append((pt_api, current_tf_api, "", "跳过", "validated 文件中不存在"))
            continue
        
        validated_tf_api, confidence, changed = validated_data[pt_api]
        print(f"  Validated 映射: {validated_tf_api} (confidence={confidence}, changed={changed})")
        
        # 如果 validated 的值和当前值相同，跳过
        if validated_tf_api == current_tf_api:
            print(f"  ⏭️ 值相同，跳过")
            skipped_count += 1
            update_details.append((pt_api, current_tf_api, validated_tf_api, "跳过", "值相同"))
            continue
        
        # 尝试爬取 validated 中的 TensorFlow API 文档
        doc_exists, doc_info = check_tf_doc_exists(validated_tf_api)
        
        if doc_exists:
            print(f"  ✅ Validated 值的文档存在")
            print(f"     文档预览: {doc_info[:100]}...")
            
            # 在原始数据中查找并更新
            if pt_api in pt_to_index:
                idx = pt_to_index[pt_api]
                old_value = original_data[idx]['tensorflow-api']
                original_data[idx]['tensorflow-api'] = validated_tf_api
                updated_count += 1
                update_details.append((pt_api, old_value, validated_tf_api, "更新", "文档存在"))
                print(f"  📝 更新: {old_value} -> {validated_tf_api}")
            else:
                skipped_count += 1
                update_details.append((pt_api, current_tf_api, validated_tf_api, "跳过", "原始文件中不存在"))
                print(f"  ❌ 在原始文件中未找到")
        else:
            # 原始文档不存在，validated 文档也不存在，标记为"无对应实现"
            print(f"  ❌ Validated 值的文档也不存在: {doc_info}")
            print(f"  ⚠️ 原始和 validated 文档均不存在，标记为无对应实现")
            
            if pt_api in pt_to_index:
                idx = pt_to_index[pt_api]
                old_value = original_data[idx]['tensorflow-api']
                original_data[idx]['tensorflow-api'] = "无对应实现"
                updated_count += 1
                update_details.append((pt_api, old_value, "无对应实现", "更新", f"文档均不存在，标记为无对应实现"))
                print(f"  📝 更新: {old_value} -> 无对应实现")
            else:
                skipped_count += 1
                update_details.append((pt_api, current_tf_api, "无对应实现", "跳过", "原始文件中不存在"))
    
    # 写入结果
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
        print(f"\n✅ 已写入结果文件: {output_csv}")
    else:
        print(f"\n[DRY RUN] 未写入文件")
    
    return updated_count, skipped_count, update_details


def print_summary(updated: int, skipped: int, details: List[Tuple[str, str, str, str, str]]):
    """打印摘要"""
    print("\n" + "=" * 80)
    print("处理摘要")
    print("=" * 80)
    print(f"总计处理: {len(EMPTY_DOC_APIS)} 个 API")
    print(f"已更新: {updated} 个")
    print(f"已跳过: {skipped} 个")
    
    if updated > 0:
        print("\n【已更新的映射】")
        for pt_api, old_tf, new_tf, status, reason in details:
            if status == "更新":
                print(f"  {pt_api}:")
                print(f"    旧值: {old_tf}")
                print(f"    新值: {new_tf}")
    
    skipped_details = [(d[0], d[4]) for d in details if d[3] == "跳过"]
    if skipped_details:
        print("\n【跳过的映射】")
        for pt_api, reason in skipped_details:
            print(f"  {pt_api}: {reason}")


def main():
    parser = argparse.ArgumentParser(
        description='修复 TensorFlow 文档为空的 API 映射'
    )
    parser.add_argument(
        '--original-csv',
        default='component/data/api_mappings.csv',
        help='原始 CSV 文件路径 (默认: component/data/api_mappings.csv)'
    )
    parser.add_argument(
        '--validated-csv',
        default='component/data/api_mappings_validated.csv',
        help='验证后的 CSV 文件路径 (默认: component/data/api_mappings_validated.csv)'
    )
    parser.add_argument(
        '--output-csv',
        default='component/data/api_mappings_fixed.csv',
        help='输出 CSV 文件路径 (默认: component/data/api_mappings_fixed.csv)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只打印不实际写入文件'
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    original_csv = os.path.join(project_root, args.original_csv) if not os.path.isabs(args.original_csv) else args.original_csv
    validated_csv = os.path.join(project_root, args.validated_csv) if not os.path.isabs(args.validated_csv) else args.validated_csv
    output_csv = os.path.join(project_root, args.output_csv) if not os.path.isabs(args.output_csv) else args.output_csv
    
    # 检查文件是否存在
    if not os.path.exists(original_csv):
        print(f"错误: 原始 CSV 文件不存在: {original_csv}")
        sys.exit(1)
    if not os.path.exists(validated_csv):
        print(f"错误: 验证后的 CSV 文件不存在: {validated_csv}")
        sys.exit(1)
    
    print(f"原始 CSV: {original_csv}")
    print(f"验证后 CSV: {validated_csv}")
    print(f"输出 CSV: {output_csv}")
    print(f"Dry Run: {args.dry_run}")
    
    # 执行修复
    updated, skipped, details = fix_empty_doc_apis(
        original_csv,
        validated_csv,
        output_csv,
        args.dry_run
    )
    
    # 打印摘要
    print_summary(updated, skipped, details)


if __name__ == '__main__':
    main()
