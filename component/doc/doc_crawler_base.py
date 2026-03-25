# ./component/doc_crawler_base.py
"""文档爬取器基类和接口定义"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
from pathlib import Path
import json
import hashlib
import time
import requests
from bs4 import BeautifulSoup

# ==================== 常量定义 ====================
DOC_CACHE_DIR = Path("data/docs_cache")
DOC_CACHE_DIR.mkdir(exist_ok=True, parents=True)

REQUEST_TIMEOUT = 10
REQUEST_DELAY = 0.5


class DocCrawler(ABC):
    """文档爬取器抽象基类"""
    
    def __init__(self, framework_name: str):
        self.framework_name = framework_name
        self.cache_dir = DOC_CACHE_DIR
    
    def get_cache_path(self, api_name: str) -> Path:
        """获取文档缓存路径"""
        api_hash = hashlib.md5(api_name.encode()).hexdigest()
        return self.cache_dir / f"{self.framework_name}_{api_hash}.json"
    
    def load_cached_doc(self, api_name: str) -> Optional[Dict]:
        """从缓存加载文档"""
        cache_path = self.get_cache_path(api_name)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    def save_cached_doc(self, api_name: str, doc_content: Dict):
        """保存文档到缓存"""
        cache_path = self.get_cache_path(api_name)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(doc_content, f, ensure_ascii=False, indent=2)
    
    def normalize_api_name(self, api_name: str) -> str:
        """规范化 API 名称（尽量保持原始路径，只剥掉版本后缀）"""
        import re
        # 去掉首尾空白
        api_name = api_name.strip()
        # 仅移除结尾的 `_vX` 版本后缀，例如：conv2d_v2 -> conv2d
        api_name = re.sub(r"_v\d+$", "", api_name)
        return api_name
    
    def fetch_url(self, url: str) -> Optional[requests.Response]:
        """获取 URL 内容（带延迟和错误处理）"""
        try:
            time.sleep(REQUEST_DELAY)
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            return response if response.status_code == 200 else None
        except Exception as e:
            print(f"[WARN] 请求失败 {url}: {e}")
            return None
    
    @abstractmethod
    def build_doc_url(self, api_name: str) -> str:
        """构建文档 URL（子类必须实现）"""
        pass
    
    @abstractmethod
    def parse_doc_content(self, soup: BeautifulSoup, api_name: str, url: str) -> Dict:
        """解析文档内容（子类必须实现）"""
        pass
    
    def crawl(self, api_name: str) -> Optional[Dict]:
        """爬取文档（主流程）"""
        api_name = self.normalize_api_name(api_name)
        
        # 检查缓存
        cached = self.load_cached_doc(api_name)
        if cached:
            return cached
        
        # 构建 URL
        url = self.build_doc_url(api_name)
        
        # 获取页面
        response = self.fetch_url(url)
        if not response:
            return None
        
        # 解析内容
        soup = BeautifulSoup(response.content, 'html.parser')
        doc_content = self.parse_doc_content(soup, api_name, url)
        
        # 保存缓存
        if doc_content:
            self.save_cached_doc(api_name, doc_content)
        
        return doc_content
    
    def get_doc_text(self, api_name: str) -> str:
        """获取文档的文本内容（用于大模型分析）"""
        doc = self.crawl(api_name)
        if not doc:
            return f"无法获取 {self.framework_name} API {api_name} 的文档"
        
        # 格式化文档内容
        content_parts = [
            f"API: {doc['api_name']}",
            f"框架: {doc['framework']}",
            f"URL: {doc['url']}",
        ]
        
        if doc.get('description'):
            content_parts.append(f"\n描述:\n{doc['description']}")
        
        if doc.get('parameters'):
            content_parts.append("\n参数:")
            for param in doc['parameters']:
                content_parts.append(f"  - {param['name']}: {param.get('description', '')}")
        
        if doc.get('returns'):
            content_parts.append(f"\n返回值:\n{doc['returns']}")
        
        return "\n".join(content_parts)

