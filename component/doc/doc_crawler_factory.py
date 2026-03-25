# ./component/doc_crawler_factory.py
"""文档爬取器工厂"""
from typing import Dict, Optional
from component.doc.doc_crawler_base import DocCrawler
from component.doc.doc_crawler_pytorch import PyTorchDocCrawler
from component.doc.doc_crawler_tensorflow import TensorFlowDocCrawler
from component.doc.doc_crawler_paddle import PaddleDocCrawler
from component.doc.doc_crawler_mindspore import MindSporeDocCrawler

# 注册的爬取器
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
    """获取指定框架的文档爬取器"""
    framework_lower = framework.lower()
    
    # 直接匹配
    if framework_lower in _CRAWLERS:
        crawler_class = _CRAWLERS[framework_lower]
        return crawler_class()
    
    # 尝试模糊匹配
    for key, crawler_class in _CRAWLERS.items():
        if key in framework_lower or framework_lower in key:
            return crawler_class()
    
    return None


def detect_framework(api_name: str) -> Optional[str]:
    """从 API 名称自动检测框架"""
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
    """爬取文档（统一接口）"""
    # 如果没有指定框架，尝试自动检测
    if not framework:
        framework = detect_framework(api_name)
    
    if not framework:
        print(f"[ERROR] 无法检测框架，请指定 --framework 参数")
        return None
    
    # 获取爬取器
    crawler = get_crawler(framework)
    if not crawler:
        print(f"[ERROR] 不支持的框架: {framework}")
        print(f"[INFO] 支持的框架: {', '.join(_CRAWLERS.keys())}")
        return None
    
    # 爬取文档
    return crawler.crawl(api_name)


def get_doc_content(api_name: str, framework: Optional[str] = None) -> str:
    """获取文档的文本内容（用于大模型分析）"""
    # 如果没有指定框架，尝试自动检测
    if not framework:
        framework = detect_framework(api_name)
    
    if not framework:
        return f"无法检测框架，请指定框架参数"
    
    # 获取爬取器
    crawler = get_crawler(framework)
    if not crawler:
        return f"不支持的框架: {framework}"
    
    # 获取文档文本
    return crawler.get_doc_text(api_name)


def register_crawler(framework: str, crawler_class: type):
    """注册新的文档爬取器（用于扩展）"""
    _CRAWLERS[framework.lower()] = crawler_class


def list_supported_frameworks() -> list:
    """列出支持的框架"""
    return list(_CRAWLERS.keys())

