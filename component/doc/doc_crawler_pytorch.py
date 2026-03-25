# ./component/doc_crawler_pytorch.py
"""PyTorch 文档爬取器"""
import re
from typing import Dict
from bs4 import BeautifulSoup
from component.doc.doc_crawler_base import DocCrawler

PT_DOC_BASE = "https://pytorch.org/docs/stable/"


class PyTorchDocCrawler(DocCrawler):
    """PyTorch 文档爬取器"""
    
    def __init__(self):
        super().__init__("pytorch")
    
    def build_doc_url(self, api_name: str) -> str:
        """构建 PyTorch 文档 URL（优先使用 generated 页面）.

        典型示例:
        - torch.nn.Conv2d -> https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        - torch.add        -> https://pytorch.org/docs/stable/generated/torch.add.html
        """
        api_name = api_name.lstrip(".")
        # 使用官方推荐的 generated 路径
        return f"{PT_DOC_BASE}generated/{api_name}.html"
    
    def parse_doc_content(self, soup: BeautifulSoup, api_name: str, url: str) -> Dict:
        """解析 PyTorch 文档内容"""
        doc_content = {
            "api_name": api_name,
            "framework": "pytorch",
            "url": url,
            "title": soup.find('title').text if soup.find('title') else "",
            "description": "",
            "parameters": [],
            "returns": "",
            "examples": [],
            "raw_html": str(soup.find('main') or soup.find('body') or '')
        }
        
        # 提取主要描述
        main_content = soup.find('main') or soup.find('div', class_='document')
        if main_content:
            # 提取第一个段落作为描述
            first_p = main_content.find('p')
            if first_p:
                doc_content["description"] = first_p.get_text(strip=True)
            
            # 提取参数说明
            params_section = main_content.find('dl', class_='field-list') or main_content.find('dl')
            if params_section:
                params = []
                for dt in params_section.find_all('dt'):
                    param_name = dt.get_text(strip=True)
                    dd = dt.find_next_sibling('dd')
                    param_desc = dd.get_text(strip=True) if dd else ""
                    params.append({"name": param_name, "description": param_desc})
                doc_content["parameters"] = params
            
            # 提取返回值说明
            returns_section = main_content.find(string=re.compile(r'Returns?', re.I))
            if returns_section:
                parent = returns_section.find_parent()
                if parent:
                    doc_content["returns"] = parent.get_text(strip=True)
        
        return doc_content

