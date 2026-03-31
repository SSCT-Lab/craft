#!/usr/bin/env python3
"""
Merge three API mapping tables into a four-column cross-framework mapping table.

Merge the following CSV files:
- api_mappings_final.csv (PyTorch -> TensorFlow)
- pd_api_mappings_final.csv (PyTorch -> PaddlePaddle)
- ms_api_mappings_final.csv (PyTorch -> MindSpore)

Output:
- unified_api_mappings.csv (PyTorch, TensorFlow, PaddlePaddle, MindSpore columns)
"""

import os
import pandas as pd
from pathlib import Path


def merge_api_mappings(output_filename: str = "unified_api_mappings.csv") -> None:
    """
    Merge three API mapping tables into a four-column table.

    Args:
        output_filename: output file name
    """
    # Current directory
    current_dir = Path(__file__).parent
    
    # Define input file paths
    tf_mapping_file = current_dir / "api_mappings_final.csv"
    pd_mapping_file = current_dir / "pd_api_mappings_final.csv"
    ms_mapping_file = current_dir / "ms_api_mappings_final.csv"
    
    # Check file existence
    missing_files = []
    for f in [tf_mapping_file, pd_mapping_file, ms_mapping_file]:
        if not f.exists():
            missing_files.append(str(f))
    
    if missing_files:
        print("❌ Missing files:")
        for f in missing_files:
            print(f"   - {f}")
        return
    
    print("📂 Reading CSV files...")
    
    # Read three CSV files
    # PyTorch -> TensorFlow mapping
    df_tf = pd.read_csv(tf_mapping_file)
    df_tf.columns = ['pytorch-api', 'tensorflow-api']
    print(f"   ✅ Read TensorFlow mapping: {len(df_tf)} records")
    
    # PyTorch -> PaddlePaddle mapping
    df_pd = pd.read_csv(pd_mapping_file)
    df_pd.columns = ['pytorch-api', 'paddle-api']
    print(f"   ✅ Read PaddlePaddle mapping: {len(df_pd)} records")
    
    # PyTorch -> MindSpore mapping
    df_ms = pd.read_csv(ms_mapping_file)
    df_ms.columns = ['pytorch-api', 'mindspore-api']
    print(f"   ✅ Read MindSpore mapping: {len(df_ms)} records")
    
    print("\n🔄 Merging mapping tables...")
    
    # Merge by PyTorch API key
    # Use outer joins to keep all APIs
    df_merged = df_tf.merge(df_pd, on='pytorch-api', how='outer')
    df_merged = df_merged.merge(df_ms, on='pytorch-api', how='outer')
    
    # Fill missing values with "无对应实现"
    df_merged = df_merged.fillna("无对应实现")
    
    # Sort by PyTorch API
    df_merged = df_merged.sort_values('pytorch-api').reset_index(drop=True)
    
    # Output file
    output_path = current_dir / output_filename
    df_merged.to_csv(output_path, index=False, encoding='utf-8')
    
    print("\n✅ Merge complete!")
    print(f"   📄 Output file: {output_path}")
    print(f"   📊 Total records: {len(df_merged)}")
    
    # Print summary stats
    print("\n📊 Mapping stats:")
    
    # TensorFlow mapping stats
    tf_valid = df_merged[df_merged['tensorflow-api'] != '无对应实现'].shape[0]
    print(f"   - TensorFlow: {tf_valid}/{len(df_merged)} mapped ({tf_valid/len(df_merged)*100:.1f}%)")
    
    # PaddlePaddle mapping stats
    pd_valid = df_merged[df_merged['paddle-api'] != '无对应实现'].shape[0]
    print(f"   - PaddlePaddle: {pd_valid}/{len(df_merged)} mapped ({pd_valid/len(df_merged)*100:.1f}%)")
    
    # MindSpore mapping stats
    ms_valid = df_merged[df_merged['mindspore-api'] != '无对应实现'].shape[0]
    print(f"   - MindSpore: {ms_valid}/{len(df_merged)} mapped ({ms_valid/len(df_merged)*100:.1f}%)")
    
    # Pairwise stats (for differential testing)
    print("\n📊 Cross-framework overlap stats (for differential testing):")
    
    # TensorFlow & PaddlePaddle both present
    tf_pd_both = df_merged[
        (df_merged['tensorflow-api'] != '无对应实现') & 
        (df_merged['paddle-api'] != '无对应实现')
    ].shape[0]
    print(f"   - TensorFlow ∩ PaddlePaddle: {tf_pd_both} APIs")
    
    # TensorFlow & MindSpore both present
    tf_ms_both = df_merged[
        (df_merged['tensorflow-api'] != '无对应实现') & 
        (df_merged['mindspore-api'] != '无对应实现')
    ].shape[0]
    print(f"   - TensorFlow ∩ MindSpore: {tf_ms_both} APIs")
    
    # PaddlePaddle & MindSpore both present
    pd_ms_both = df_merged[
        (df_merged['paddle-api'] != '无对应实现') & 
        (df_merged['mindspore-api'] != '无对应实现')
    ].shape[0]
    print(f"   - PaddlePaddle ∩ MindSpore: {pd_ms_both} APIs")
    
    # All three frameworks present
    all_three = df_merged[
        (df_merged['tensorflow-api'] != '无对应实现') & 
        (df_merged['paddle-api'] != '无对应实现') &
        (df_merged['mindspore-api'] != '无对应实现')
    ].shape[0]
    print(f"   - All three frameworks: {all_three} APIs")
    
    # Preview first rows
    print("\n📋 Mapping preview (first 10 rows):")
    print(df_merged.head(10).to_string(index=False))
    
    return df_merged


def get_tf_pd_mapping() -> pd.DataFrame:
    """
    Get API mappings where both TensorFlow and PaddlePaddle are present.

    Returns:
        DataFrame with pytorch-api, tensorflow-api, paddle-api columns
    """
    current_dir = Path(__file__).parent
    unified_file = current_dir / "unified_api_mappings.csv"
    
    if not unified_file.exists():
        print("⚠️ Unified mapping file not found, generating...")
        merge_api_mappings()
    
    df = pd.read_csv(unified_file)
    
    # Filter APIs where both TensorFlow and PaddlePaddle are present
    df_tf_pd = df[
        (df['tensorflow-api'] != '无对应实现') & 
        (df['paddle-api'] != '无对应实现')
    ][['pytorch-api', 'tensorflow-api', 'paddle-api']].copy()
    
    return df_tf_pd


if __name__ == "__main__":
    merge_api_mappings()
