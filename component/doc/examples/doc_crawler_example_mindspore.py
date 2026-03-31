# ./component/doc_crawler_example_mindspore.py
"""MindSpore doc crawler example (for future extension)."""
from typing import Dict
from component.doc.doc_crawler_base import DocCrawler
from bs4 import BeautifulSoup
import re

MINSPORE_DOC_BASE = "https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/"


class MindSporeDocCrawler(DocCrawler):
    """MindSpore doc crawler."""
    
    def __init__(self):
        super().__init__("mindspore")
    
    def normalize_api_name(self, api_name: str) -> str:
        """Normalize MindSpore API name."""
        api_name = super().normalize_api_name(api_name)
        # MindSpore-specific normalization
        api_name = api_name.replace('mindspore.', '').replace('ms.', '')
        return api_name.strip()
    
    def build_doc_url(self, api_name: str) -> str:
        """Build MindSpore doc URL."""
        # MindSpore doc URL format example
        # https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore/mindspore.xxx.html
        api_parts = api_name.split('.')
        
        if len(api_parts) == 1:
            return f"{MINSPORE_DOC_BASE}mindspore/mindspore.{api_name}.html"
        else:
            module_path = '/'.join(api_parts[:-1])
            func_name = api_parts[-1]
            return f"{MINSPORE_DOC_BASE}mindspore/{module_path}/mindspore.{func_name}.html"
    
    def parse_doc_content(self, soup: BeautifulSoup, api_name: str, url: str) -> Dict:
        """Parse MindSpore doc content."""
        doc_content = {
            "api_name": api_name,
            "framework": "mindspore",
            "url": url,
            "title": soup.find('title').text if soup.find('title') else "",
            "description": "",
            "parameters": [],
            "returns": "",
            "examples": [],
            "raw_html": str(soup.find('main') or soup.find('article') or '')
        }
        
        # MindSpore-specific parsing logic
        main_content = soup.find('article') or soup.find('main')
        if main_content:
            # Extract description
            first_p = main_content.find('p')
            if first_p:
                doc_content["description"] = first_p.get_text(strip=True)
            
            # Extract parameters (per MindSpore doc structure)
            # ... implement parsing logic here
        
        return doc_content


# Usage example:
# from component.doc.doc_crawler_factory import register_crawler
# register_crawler('mindspore', MindSporeDocCrawler)
# register_crawler('ms', MindSporeDocCrawler)

