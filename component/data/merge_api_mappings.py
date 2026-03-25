#!/usr/bin/env python3
"""
合并三张API映射表为一张四列的跨框架映射表

将以下三张CSV文件合并：
- api_mappings_final.csv (PyTorch -> TensorFlow)
- pd_api_mappings_final.csv (PyTorch -> PaddlePaddle)
- ms_api_mappings_final.csv (PyTorch -> MindSpore)

输出:
- unified_api_mappings.csv (PyTorch, TensorFlow, PaddlePaddle, MindSpore 四列)
"""

import os
import pandas as pd
from pathlib import Path


def merge_api_mappings(output_filename: str = "unified_api_mappings.csv") -> None:
    """
    合并三张API映射表为一张四列表
    
    Args:
        output_filename: 输出文件名
    """
    # 获取当前目录
    current_dir = Path(__file__).parent
    
    # 定义输入文件路径
    tf_mapping_file = current_dir / "api_mappings_final.csv"
    pd_mapping_file = current_dir / "pd_api_mappings_final.csv"
    ms_mapping_file = current_dir / "ms_api_mappings_final.csv"
    
    # 检查文件是否存在
    missing_files = []
    for f in [tf_mapping_file, pd_mapping_file, ms_mapping_file]:
        if not f.exists():
            missing_files.append(str(f))
    
    if missing_files:
        print(f"❌ 缺少以下文件：")
        for f in missing_files:
            print(f"   - {f}")
        return
    
    print("📂 正在读取CSV文件...")
    
    # 读取三个CSV文件
    # PyTorch -> TensorFlow 映射
    df_tf = pd.read_csv(tf_mapping_file)
    df_tf.columns = ['pytorch-api', 'tensorflow-api']
    print(f"   ✅ 读取 TensorFlow 映射: {len(df_tf)} 条记录")
    
    # PyTorch -> PaddlePaddle 映射
    df_pd = pd.read_csv(pd_mapping_file)
    df_pd.columns = ['pytorch-api', 'paddle-api']
    print(f"   ✅ 读取 PaddlePaddle 映射: {len(df_pd)} 条记录")
    
    # PyTorch -> MindSpore 映射
    df_ms = pd.read_csv(ms_mapping_file)
    df_ms.columns = ['pytorch-api', 'mindspore-api']
    print(f"   ✅ 读取 MindSpore 映射: {len(df_ms)} 条记录")
    
    print("\n🔄 正在合并映射表...")
    
    # 以 PyTorch API 为主键进行合并
    # 使用外连接确保不丢失任何API
    df_merged = df_tf.merge(df_pd, on='pytorch-api', how='outer')
    df_merged = df_merged.merge(df_ms, on='pytorch-api', how='outer')
    
    # 填充缺失值为 "无对应实现"
    df_merged = df_merged.fillna("无对应实现")
    
    # 按 PyTorch API 排序
    df_merged = df_merged.sort_values('pytorch-api').reset_index(drop=True)
    
    # 输出文件
    output_path = current_dir / output_filename
    df_merged.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n✅ 合并完成！")
    print(f"   📄 输出文件: {output_path}")
    print(f"   📊 总记录数: {len(df_merged)} 条")
    
    # 打印统计信息
    print("\n📊 映射统计信息:")
    
    # TensorFlow 映射统计
    tf_valid = df_merged[df_merged['tensorflow-api'] != '无对应实现'].shape[0]
    print(f"   - TensorFlow: {tf_valid}/{len(df_merged)} 有对应实现 ({tf_valid/len(df_merged)*100:.1f}%)")
    
    # PaddlePaddle 映射统计
    pd_valid = df_merged[df_merged['paddle-api'] != '无对应实现'].shape[0]
    print(f"   - PaddlePaddle: {pd_valid}/{len(df_merged)} 有对应实现 ({pd_valid/len(df_merged)*100:.1f}%)")
    
    # MindSpore 映射统计
    ms_valid = df_merged[df_merged['mindspore-api'] != '无对应实现'].shape[0]
    print(f"   - MindSpore: {ms_valid}/{len(df_merged)} 有对应实现 ({ms_valid/len(df_merged)*100:.1f}%)")
    
    # 两两框架都有实现的统计（用于差分测试）
    print("\n📊 跨框架对比统计（可用于差分测试）:")
    
    # TensorFlow & PaddlePaddle 都有实现
    tf_pd_both = df_merged[
        (df_merged['tensorflow-api'] != '无对应实现') & 
        (df_merged['paddle-api'] != '无对应实现')
    ].shape[0]
    print(f"   - TensorFlow ∩ PaddlePaddle: {tf_pd_both} 个API")
    
    # TensorFlow & MindSpore 都有实现
    tf_ms_both = df_merged[
        (df_merged['tensorflow-api'] != '无对应实现') & 
        (df_merged['mindspore-api'] != '无对应实现')
    ].shape[0]
    print(f"   - TensorFlow ∩ MindSpore: {tf_ms_both} 个API")
    
    # PaddlePaddle & MindSpore 都有实现
    pd_ms_both = df_merged[
        (df_merged['paddle-api'] != '无对应实现') & 
        (df_merged['mindspore-api'] != '无对应实现')
    ].shape[0]
    print(f"   - PaddlePaddle ∩ MindSpore: {pd_ms_both} 个API")
    
    # 三个框架都有实现
    all_three = df_merged[
        (df_merged['tensorflow-api'] != '无对应实现') & 
        (df_merged['paddle-api'] != '无对应实现') &
        (df_merged['mindspore-api'] != '无对应实现')
    ].shape[0]
    print(f"   - 三框架都有实现: {all_three} 个API")
    
    # 预览前几行
    print("\n📋 映射表预览（前10行）:")
    print(df_merged.head(10).to_string(index=False))
    
    return df_merged


def get_tf_pd_mapping() -> pd.DataFrame:
    """
    获取TensorFlow和PaddlePaddle都有实现的API映射
    
    Returns:
        包含 pytorch-api, tensorflow-api, paddle-api 三列的DataFrame
    """
    current_dir = Path(__file__).parent
    unified_file = current_dir / "unified_api_mappings.csv"
    
    if not unified_file.exists():
        print("⚠️ 统一映射表不存在，正在生成...")
        merge_api_mappings()
    
    df = pd.read_csv(unified_file)
    
    # 筛选 TensorFlow 和 PaddlePaddle 都有实现的 API
    df_tf_pd = df[
        (df['tensorflow-api'] != '无对应实现') & 
        (df['paddle-api'] != '无对应实现')
    ][['pytorch-api', 'tensorflow-api', 'paddle-api']].copy()
    
    return df_tf_pd


if __name__ == "__main__":
    merge_api_mappings()
