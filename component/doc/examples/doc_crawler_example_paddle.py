# ./component/doc_crawler_example_paddle.py
"""PaddlePaddle doc crawler example (for future extension)."""
from typing import Dict
from component.doc.doc_crawler_base import DocCrawler
from bs4 import BeautifulSoup
import re

PADDLE_DOC_BASE = "https://www.paddlepaddle.org.cn/documentation/docs/zh/api/"


class PaddleDocCrawler(DocCrawler):
    """PaddlePaddle doc crawler."""
    
    def __init__(self):
        super().__init__("paddle")
    
    def normalize_api_name(self, api_name: str) -> str:
        """Normalize PaddlePaddle API name."""
        api_name = super().normalize_api_name(api_name)
        # PaddlePaddle-specific normalization
        api_name = api_name.replace('paddle.', '')
        return api_name.strip()
    
    def build_doc_url(self, api_name: str) -> str:
        """Build PaddlePaddle doc URL."""
        # PaddlePaddle doc URL format examples
        # https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/xxx_cn.html
        api_parts = api_name.split('.')
        
        if len(api_parts) == 1:
            return f"{PADDLE_DOC_BASE}paddle/{api_name}_cn.html"
        else:
            module_path = '/'.join(api_parts[:-1])
            func_name = api_parts[-1]
            return f"{PADDLE_DOC_BASE}paddle/{module_path}/{func_name}_cn.html"
    
    def parse_doc_content(self, soup: BeautifulSoup, api_name: str, url: str) -> Dict:
        """Parse PaddlePaddle doc content."""
        doc_content = {
            "api_name": api_name,
            "framework": "paddle",
            "url": url,
            "title": soup.find('title').text if soup.find('title') else "",
            "description": "",
            "parameters": [],
            "returns": "",
            "examples": [],
            "raw_html": str(soup.find('main') or soup.find('article') or '')
        }
        
        # PaddlePaddle-specific parsing logic
        main_content = soup.find('article') or soup.find('main')
        if main_content:
            # Extract description
            first_p = main_content.find('p')
            if first_p:
                doc_content["description"] = first_p.get_text(strip=True)
            
            # Extract parameters (per PaddlePaddle doc structure)
            # ... implement parsing logic here
        
        return doc_content


# Usage example:
# from component.doc_crawler_factory import register_crawler
# register_crawler('paddle', PaddleDocCrawler)
# register_crawler('paddlepaddle', PaddleDocCrawler)

