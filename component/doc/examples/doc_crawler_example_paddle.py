# ./component/doc_crawler_example_paddle.py
"""PaddlePaddle 文档爬取器示例（未来扩展用）"""
from typing import Dict
from component.doc.doc_crawler_base import DocCrawler
from bs4 import BeautifulSoup
import re

PADDLE_DOC_BASE = "https://www.paddlepaddle.org.cn/documentation/docs/zh/api/"


class PaddleDocCrawler(DocCrawler):
    """PaddlePaddle 文档爬取器"""
    
    def __init__(self):
        super().__init__("paddle")
    
    def normalize_api_name(self, api_name: str) -> str:
        """规范化 PaddlePaddle API 名称"""
        api_name = super().normalize_api_name(api_name)
        # PaddlePaddle 特定的规范化逻辑
        api_name = api_name.replace('paddle.', '')
        return api_name.strip()
    
    def build_doc_url(self, api_name: str) -> str:
        """构建 PaddlePaddle 文档 URL"""
        # PaddlePaddle 文档 URL 格式示例
        # https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/xxx_cn.html
        api_parts = api_name.split('.')
        
        if len(api_parts) == 1:
            return f"{PADDLE_DOC_BASE}paddle/{api_name}_cn.html"
        else:
            module_path = '/'.join(api_parts[:-1])
            func_name = api_parts[-1]
            return f"{PADDLE_DOC_BASE}paddle/{module_path}/{func_name}_cn.html"
    
    def parse_doc_content(self, soup: BeautifulSoup, api_name: str, url: str) -> Dict:
        """解析 PaddlePaddle 文档内容"""
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
        
        # PaddlePaddle 特定的解析逻辑
        main_content = soup.find('article') or soup.find('main')
        if main_content:
            # 提取描述
            first_p = main_content.find('p')
            if first_p:
                doc_content["description"] = first_p.get_text(strip=True)
            
            # 提取参数（根据 PaddlePaddle 文档结构）
            # ... 实现具体的解析逻辑
        
        return doc_content


# 使用示例：
# from component.doc_crawler_factory import register_crawler
# register_crawler('paddle', PaddleDocCrawler)
# register_crawler('paddlepaddle', PaddleDocCrawler)

