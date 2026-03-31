# ./component/doc_crawler_pytorch.py
"""PyTorch documentation crawler."""
import re
from typing import Dict
from bs4 import BeautifulSoup
from component.doc.doc_crawler_base import DocCrawler

PT_DOC_BASE = "https://pytorch.org/docs/stable/"


class PyTorchDocCrawler(DocCrawler):
    """PyTorch documentation crawler."""
    
    def __init__(self):
        super().__init__("pytorch")
    
    def build_doc_url(self, api_name: str) -> str:
        """Build PyTorch doc URL (prefer generated pages).

        Typical examples:
        - torch.nn.Conv2d -> https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        - torch.add        -> https://pytorch.org/docs/stable/generated/torch.add.html
        """
        api_name = api_name.lstrip(".")
        # Use official recommended generated path
        return f"{PT_DOC_BASE}generated/{api_name}.html"
    
    def parse_doc_content(self, soup: BeautifulSoup, api_name: str, url: str) -> Dict:
        """Parse PyTorch doc content."""
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
        
        # Extract main description
        main_content = soup.find('main') or soup.find('div', class_='document')
        if main_content:
            # Use first paragraph as description
            first_p = main_content.find('p')
            if first_p:
                doc_content["description"] = first_p.get_text(strip=True)
            
            # Extract parameters
            params_section = main_content.find('dl', class_='field-list') or main_content.find('dl')
            if params_section:
                params = []
                for dt in params_section.find_all('dt'):
                    param_name = dt.get_text(strip=True)
                    dd = dt.find_next_sibling('dd')
                    param_desc = dd.get_text(strip=True) if dd else ""
                    params.append({"name": param_name, "description": param_desc})
                doc_content["parameters"] = params
            
            # Extract returns
            returns_section = main_content.find(string=re.compile(r'Returns?', re.I))
            if returns_section:
                parent = returns_section.find_parent()
                if parent:
                    doc_content["returns"] = parent.get_text(strip=True)
        
        return doc_content

