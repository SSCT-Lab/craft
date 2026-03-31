# ./component/doc_crawler_tensorflow.py
"""TensorFlow documentation crawler."""
import re
from typing import Dict
from bs4 import BeautifulSoup
from component.doc.doc_crawler_base import DocCrawler

TF_DOC_BASE = "https://www.tensorflow.org/api_docs/python/"


class TensorFlowDocCrawler(DocCrawler):
    """TensorFlow documentation crawler."""
    
    def __init__(self):
        super().__init__("tensorflow")
    
    def build_doc_url(self, api_name: str) -> str:
        """Build TensorFlow doc URL."""
        api_parts = api_name.split('.')
        
        if len(api_parts) == 1:
            return f"{TF_DOC_BASE}{api_name}"
        else:
            # module.function format
            module_path = '/'.join(api_parts[:-1])
            func_name = api_parts[-1]
            return f"{TF_DOC_BASE}{module_path}/{func_name}"
    
    def parse_doc_content(self, soup: BeautifulSoup, api_name: str, url: str) -> Dict:
        """Parse TensorFlow doc content."""
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
        
        # Extract main description
        main_content = soup.find('main') or soup.find('div', class_='devsite-article-body')
        if main_content:
            # Store raw HTML (useful for debugging)
            doc_content["raw_html"] = str(main_content)[:5000]
            
            # Extract description: scan p tags for first meaningful text
            description_parts = []
            for p in main_content.find_all('p'):
                text = p.get_text(strip=True)
                if text and len(text) > 10:  # Skip empty/too-short paragraphs
                    description_parts.append(text)
                    if len(description_parts) >= 3:  # Take first 3 valid paragraphs
                        break
            
            if description_parts:
                doc_content["description"] = "\n".join(description_parts)
            
            # Extract API signature (usually in a pre tag)
            signature_pre = main_content.find('pre')
            if signature_pre:
                signature_text = signature_pre.get_text(strip=True)
                if signature_text and len(signature_text) < 500:  # Reasonable signature length
                    doc_content["signature"] = signature_text
            
            # Extract parameters (TensorFlow uses specific HTML structure)
            # Try multiple ways to find the Args section
            params_section = None
            
            # Method 1: section id
            params_section = main_content.find('section', {'id': 'args'})
            
            # Method 2: h2 tag
            if not params_section:
                args_h2 = main_content.find('h2', string=re.compile(r'^Args?$', re.I))
                if args_h2:
                    params_section = args_h2.find_parent('section') or args_h2
            
            # Method 3: h3 tag containing "Args"
            if not params_section:
                args_h3 = main_content.find('h3', string=re.compile(r'^Args?$', re.I))
                if args_h3:
                    params_section = args_h3.find_parent('section') or args_h3
            
            if params_section:
                params = []
                # TensorFlow params are usually in <code> tags
                for code in params_section.find_all('code'):
                    param_name = code.get_text(strip=True)
                    if param_name and len(param_name) < 50:  # Reasonable param name length
                        # Find parameter description
                        parent = code.find_parent()
                        if parent:
                            param_desc = parent.get_text(strip=True).replace(param_name, '', 1).strip()
                            if param_desc and len(param_desc) > 5:
                                params.append({"name": param_name, "description": param_desc[:200]})
                doc_content["parameters"] = params[:20]  # Up to 20 parameters
            
            # Extract returns
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

