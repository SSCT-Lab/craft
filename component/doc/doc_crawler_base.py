# ./component/doc_crawler_base.py
"""Documentation crawler base class and interface definitions."""
from abc import ABC, abstractmethod
from typing import Dict, Optional
from pathlib import Path
import json
import hashlib
import time
import requests
from bs4 import BeautifulSoup

# ==================== Constants ====================
DOC_CACHE_DIR = Path("data/docs_cache")
DOC_CACHE_DIR.mkdir(exist_ok=True, parents=True)

REQUEST_TIMEOUT = 10
REQUEST_DELAY = 0.5


class DocCrawler(ABC):
    """Abstract base class for documentation crawlers."""
    
    def __init__(self, framework_name: str):
        self.framework_name = framework_name
        self.cache_dir = DOC_CACHE_DIR
    
    def get_cache_path(self, api_name: str) -> Path:
        """Get doc cache path."""
        api_hash = hashlib.md5(api_name.encode()).hexdigest()
        return self.cache_dir / f"{self.framework_name}_{api_hash}.json"
    
    def load_cached_doc(self, api_name: str) -> Optional[Dict]:
        """Load doc from cache."""
        cache_path = self.get_cache_path(api_name)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    def save_cached_doc(self, api_name: str, doc_content: Dict):
        """Save doc to cache."""
        cache_path = self.get_cache_path(api_name)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(doc_content, f, ensure_ascii=False, indent=2)
    
    def normalize_api_name(self, api_name: str) -> str:
        """Normalize API name (keep original path, only strip version suffix)."""
        import re
        # Trim leading/trailing whitespace
        api_name = api_name.strip()
        # Remove trailing `_vX` suffix, e.g., conv2d_v2 -> conv2d
        api_name = re.sub(r"_v\d+$", "", api_name)
        return api_name
    
    def fetch_url(self, url: str) -> Optional[requests.Response]:
        """Fetch URL content (with delay and error handling)."""
        try:
            time.sleep(REQUEST_DELAY)
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            return response if response.status_code == 200 else None
        except Exception as e:
            print(f"[WARN] Request failed {url}: {e}")
            return None
    
    @abstractmethod
    def build_doc_url(self, api_name: str) -> str:
        """Build doc URL (subclass must implement)."""
        pass
    
    @abstractmethod
    def parse_doc_content(self, soup: BeautifulSoup, api_name: str, url: str) -> Dict:
        """Parse doc content (subclass must implement)."""
        pass
    
    def crawl(self, api_name: str) -> Optional[Dict]:
        """Crawl documentation (main flow)."""
        api_name = self.normalize_api_name(api_name)
        
        # Check cache
        cached = self.load_cached_doc(api_name)
        if cached:
            return cached
        
        # Build URL
        url = self.build_doc_url(api_name)
        
        # Fetch page
        response = self.fetch_url(url)
        if not response:
            return None
        
        # Parse content
        soup = BeautifulSoup(response.content, 'html.parser')
        doc_content = self.parse_doc_content(soup, api_name, url)
        
        # Save cache
        if doc_content:
            self.save_cached_doc(api_name, doc_content)
        
        return doc_content
    
    def get_doc_text(self, api_name: str) -> str:
        """Get documentation text content (for LLM analysis)."""
        doc = self.crawl(api_name)
        if not doc:
            return f"Unable to fetch {self.framework_name} API {api_name} docs"
        
        # Format doc content
        content_parts = [
            f"API: {doc['api_name']}",
            f"Framework: {doc['framework']}",
            f"URL: {doc['url']}",
        ]
        
        if doc.get('description'):
            content_parts.append(f"\nDescription:\n{doc['description']}")
        
        if doc.get('parameters'):
            content_parts.append("\nParameters:")
            for param in doc['parameters']:
                content_parts.append(f"  - {param['name']}: {param.get('description', '')}")
        
        if doc.get('returns'):
            content_parts.append(f"\nReturns:\n{doc['returns']}")
        
        return "\n".join(content_parts)

