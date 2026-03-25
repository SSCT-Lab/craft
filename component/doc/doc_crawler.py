# ./component/doc_crawler.py
"""爬取深度学习框架官方文档（统一接口）"""
# 向后兼容：导入新的工厂接口
from component.doc.doc_crawler_factory import (
    crawl_doc,
    get_doc_content,
    detect_framework,
    get_crawler,
    list_supported_frameworks
)


def main():
    """命令行工具"""
    import argparse
    from component.doc.doc_crawler_factory import list_supported_frameworks
    
    parser = argparse.ArgumentParser(description="爬取深度学习框架官方文档")
    parser.add_argument("api", help="API 名称，例如: torch.nn.Conv2d 或 tf.keras.layers.Dense")
    parser.add_argument("--framework", "-f", 
                       help=f"框架名称（可选，支持: {', '.join(list_supported_frameworks())}）", 
                       default=None)
    parser.add_argument("--output", "-o", help="输出文件路径（JSON格式）", default=None)
    
    args = parser.parse_args()
    
    # 使用工厂接口爬取文档
    doc = crawl_doc(args.api, args.framework)
    
    if doc:
        if args.output:
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            print(f"[SUCCESS] 文档已保存到: {args.output}")
        else:
            import json
            print(json.dumps(doc, ensure_ascii=False, indent=2))
    else:
        print(f"[ERROR] 无法获取文档")


if __name__ == "__main__":
    main()

