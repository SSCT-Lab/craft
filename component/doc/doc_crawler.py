# ./component/doc_crawler.py
"""Crawl official deep learning framework docs (unified interface)."""
# Backward compatibility: import the new factory interface
from component.doc.doc_crawler_factory import (
    crawl_doc,
    get_doc_content,
    detect_framework,
    get_crawler,
    list_supported_frameworks
)


def main():
    """CLI tool."""
    import argparse
    from component.doc.doc_crawler_factory import list_supported_frameworks
    
    parser = argparse.ArgumentParser(description="Crawl official deep learning framework docs")
    parser.add_argument("api", help="API name, e.g., torch.nn.Conv2d or tf.keras.layers.Dense")
    parser.add_argument("--framework", "-f", 
                       help=f"Framework name (optional, supported: {', '.join(list_supported_frameworks())})", 
                       default=None)
    parser.add_argument("--output", "-o", help="Output file path (JSON)", default=None)
    
    args = parser.parse_args()
    
    # Crawl docs via factory interface
    doc = crawl_doc(args.api, args.framework)
    
    if doc:
        if args.output:
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            print(f"[SUCCESS] Docs saved to: {args.output}")
        else:
            import json
            print(json.dumps(doc, ensure_ascii=False, indent=2))
    else:
        print("[ERROR] Unable to fetch docs")


if __name__ == "__main__":
    main()

