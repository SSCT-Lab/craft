# ./component/doc_crawler_factory.py
"""Documentation crawler factory."""
from typing import Dict, Optional
from component.doc.doc_crawler_base import DocCrawler
from component.doc.doc_crawler_pytorch import PyTorchDocCrawler
from component.doc.doc_crawler_tensorflow import TensorFlowDocCrawler
from component.doc.doc_crawler_paddle import PaddleDocCrawler
from component.doc.doc_crawler_mindspore import MindSporeDocCrawler

# Registered crawlers
_CRAWLERS: Dict[str, type] = {
    'pytorch': PyTorchDocCrawler,
    'pt': PyTorchDocCrawler,
    'torch': PyTorchDocCrawler,
    'tensorflow': TensorFlowDocCrawler,
    'tf': TensorFlowDocCrawler,
    'paddle': PaddleDocCrawler,
    'paddlepaddle': PaddleDocCrawler,
    'pd': PaddleDocCrawler,
    'mindspore': MindSporeDocCrawler,
    'ms': MindSporeDocCrawler,
}


def get_crawler(framework: str) -> Optional[DocCrawler]:
    """Get a documentation crawler for the given framework."""
    framework_lower = framework.lower()
    
    # Direct match
    if framework_lower in _CRAWLERS:
        crawler_class = _CRAWLERS[framework_lower]
        return crawler_class()
    
    # Fuzzy match
    for key, crawler_class in _CRAWLERS.items():
        if key in framework_lower or framework_lower in key:
            return crawler_class()
    
    return None


def detect_framework(api_name: str) -> Optional[str]:
    """Detect framework from API name."""
    api_lower = api_name.lower()
    
    if api_lower.startswith('tf.') or api_lower.startswith('tensorflow.'):
        return 'tensorflow'
    elif api_lower.startswith('torch.') or api_lower.startswith('pt.'):
        return 'pytorch'
    elif api_lower.startswith('paddle.'):
        return 'paddle'
    elif api_lower.startswith('ms.') or api_lower.startswith('mindspore.'):
        return 'mindspore'
    
    return None


def crawl_doc(api_name: str, framework: Optional[str] = None) -> Optional[Dict]:
    """Crawl documentation (unified interface)."""
    # If framework is not specified, try auto-detection
    if not framework:
        framework = detect_framework(api_name)
    
    if not framework:
        print("[ERROR] Cannot detect framework, please specify --framework")
        return None
    
    # Get crawler
    crawler = get_crawler(framework)
    if not crawler:
        print(f"[ERROR] Unsupported framework: {framework}")
        print(f"[INFO] Supported frameworks: {', '.join(_CRAWLERS.keys())}")
        return None
    
    # Crawl docs
    return crawler.crawl(api_name)


def get_doc_content(api_name: str, framework: Optional[str] = None) -> str:
    """Get documentation text content (for LLM analysis)."""
    # If framework is not specified, try auto-detection
    if not framework:
        framework = detect_framework(api_name)
    
    if not framework:
        return "Cannot detect framework; please specify a framework"
    
    # Get crawler
    crawler = get_crawler(framework)
    if not crawler:
        return f"Unsupported framework: {framework}"
    
    # Get doc text
    return crawler.get_doc_text(api_name)


def register_crawler(framework: str, crawler_class: type):
    """Register a new documentation crawler (for extensions)."""
    _CRAWLERS[framework.lower()] = crawler_class


def list_supported_frameworks() -> list:
    """List supported frameworks."""
    return list(_CRAWLERS.keys())

