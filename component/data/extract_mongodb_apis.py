"""
从MongoDB数据库中提取所有PyTorch算子名称并保存到CSV文件

数据库: freefuzz-torch
集合: argVS
输出: pytorch_apis.csv (列名: pytorch-api)
"""

import os
import csv
from typing import List, Set

import pymongo


def extract_pytorch_apis(
    mongo_uri: str = "mongodb://localhost:27017/",
    db_name: str = "freefuzz-torch",
    collection_name: str = "argVS"
) -> List[str]:
    """
    从MongoDB中提取所有PyTorch算子名称
    
    Args:
        mongo_uri: MongoDB连接URI
        db_name: 数据库名称
        collection_name: 集合名称
    
    Returns:
        去重后的算子名称列表（按字母排序）
    """
    # 连接MongoDB
    client = pymongo.MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    # 使用distinct方法直接获取去重后的api字段值
    # 这比遍历所有文档更高效
    api_names: List[str] = collection.distinct("api")
    
    # 过滤掉空值和None
    api_names = [api for api in api_names if api]
    
    # 按字母顺序排序
    api_names.sort()
    
    # 关闭连接
    client.close()
    
    return api_names


def save_to_csv(api_names: List[str], output_path: str) -> None:
    """
    将算子名称保存到CSV文件
    
    Args:
        api_names: 算子名称列表
        output_path: 输出文件路径
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入列名
        writer.writerow(['pytorch-api'])
        # 写入数据
        for api_name in api_names:
            writer.writerow([api_name])


def main():
    """主函数"""
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'api_mappings.csv')
    
    print("=" * 50)
    print("📊 PyTorch算子名称提取工具")
    print("=" * 50)
    
    try:
        # 提取算子名称
        print("\n🔗 正在连接MongoDB数据库...")
        api_names = extract_pytorch_apis()
        
        print(f"✅ 成功提取 {len(api_names)} 个唯一算子名称")
        
        # 保存到CSV
        save_to_csv(api_names, output_path)
        print(f"✅ 已保存到: {output_path}")
        
        # 打印前10个算子作为预览
        print(f"\n📋 算子名称预览 (前10个):")
        print("-" * 30)
        for i, api_name in enumerate(api_names[:10], 1):
            print(f"  {i}. {api_name}")
        
        if len(api_names) > 10:
            print(f"  ... (共 {len(api_names)} 个)")
        
        print("\n" + "=" * 50)
        print("🎉 提取完成！")
        print("=" * 50)
        
    except pymongo.errors.ConnectionFailure as e:
        print(f"❌ MongoDB连接失败: {e}")
        print("   请确保MongoDB服务已启动")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
