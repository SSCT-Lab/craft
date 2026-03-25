# ./component/doc_crawler_tensorflow.py
"""TensorFlow 文档爬取器"""
import re
from typing import Dict
from bs4 import BeautifulSoup
from component.doc.doc_crawler_base import DocCrawler

TF_DOC_BASE = "https://www.tensorflow.org/api_docs/python/"


class TensorFlowDocCrawler(DocCrawler):
    """TensorFlow 文档爬取器"""
    
    def __init__(self):
        super().__init__("tensorflow")
    
    def build_doc_url(self, api_name: str) -> str:
        """构建 TensorFlow 文档 URL"""
        api_parts = api_name.split('.')
        
        if len(api_parts) == 1:
            return f"{TF_DOC_BASE}{api_name}"
        else:
            # 模块.函数名格式
            module_path = '/'.join(api_parts[:-1])
            func_name = api_parts[-1]
            return f"{TF_DOC_BASE}{module_path}/{func_name}"
    
    def parse_doc_content(self, soup: BeautifulSoup, api_name: str, url: str) -> Dict:
        """解析 TensorFlow 文档内容"""
        doc_content = {
            "api_name": api_name,
            "framework": "tensorflow",
            "url": url,
            "title": soup.find('title').text if soup.find('title') else "",
            "description": "",
            "parameters": [],
            "returns": "",
            "examples": [],
            "raw_html": ""
        }
        
        # 提取主要描述
        main_content = soup.find('main') or soup.find('div', class_='devsite-article-body')
        if main_content:
            # 存储原始HTML（可用于调试）
            doc_content["raw_html"] = str(main_content)[:5000]
            
            # 提取描述：遍历所有p标签，找到第一个有内容的
            description_parts = []
            for p in main_content.find_all('p'):
                text = p.get_text(strip=True)
                if text and len(text) > 10:  # 忽略空的或太短的段落
                    description_parts.append(text)
                    if len(description_parts) >= 3:  # 取前3个有效段落
                        break
            
            if description_parts:
                doc_content["description"] = "\n".join(description_parts)
            
            # 提取API签名（通常在pre标签中）
            signature_pre = main_content.find('pre')
            if signature_pre:
                signature_text = signature_pre.get_text(strip=True)
                if signature_text and len(signature_text) < 500:  # 合理长度的签名
                    doc_content["signature"] = signature_text
            
            # 提取参数说明（TensorFlow 使用特定的 HTML 结构）
            # 尝试多种方式找到Args部分
            params_section = None
            
            # 方式1：通过section id
            params_section = main_content.find('section', {'id': 'args'})
            
            # 方式2：通过h2标签
            if not params_section:
                args_h2 = main_content.find('h2', string=re.compile(r'^Args?$', re.I))
                if args_h2:
                    params_section = args_h2.find_parent('section') or args_h2
            
            # 方式3：通过包含"Args"的h3标签
            if not params_section:
                args_h3 = main_content.find('h3', string=re.compile(r'^Args?$', re.I))
                if args_h3:
                    params_section = args_h3.find_parent('section') or args_h3
            
            if params_section:
                params = []
                # TensorFlow 文档中参数通常在 <code> 标签中
                for code in params_section.find_all('code'):
                    param_name = code.get_text(strip=True)
                    if param_name and len(param_name) < 50:  # 合理长度的参数名
                        # 查找参数描述
                        parent = code.find_parent()
                        if parent:
                            param_desc = parent.get_text(strip=True).replace(param_name, '', 1).strip()
                            if param_desc and len(param_desc) > 5:
                                params.append({"name": param_name, "description": param_desc[:200]})
                doc_content["parameters"] = params[:20]  # 最多20个参数
            
            # 提取返回值说明
            returns_section = None
            returns_section = main_content.find('section', {'id': 'returns'})
            if not returns_section:
                returns_h2 = main_content.find('h2', string=re.compile(r'^Returns?$', re.I))
                if returns_h2:
                    returns_section = returns_h2.find_parent('section') or returns_h2
            
            if returns_section:
                returns_text = returns_section.get_text(strip=True)
                if returns_text:
                    doc_content["returns"] = returns_text[:500]
        
        return doc_content

