"""MindSpore 文档爬取器（适配 MindSpore 2.6.0）"""
import re
from typing import Dict
from bs4 import BeautifulSoup
from component.doc.doc_crawler_base import DocCrawler

MS_VERSION = "r2.8.0"
# MS_DOC_BASE = f"https://www.mindspore.cn/docs/zh-CN/{MS_VERSION}/api_python/"
MS_DOC_BASE = f"https://www.mindspore.cn/docs/en/{MS_VERSION}/api_python/"


class MindSporeDocCrawler(DocCrawler):
    """MindSpore 文档爬取器"""

    def __init__(self):
        super().__init__("mindspore")

    def build_doc_url(self, api_name: str) -> str:
        """
        构建 MindSpore API 文档 URL（自动识别子模块）

        示例：
        - mindspore.Tensor
          -> api_python/mindspore/mindspore.Tensor.html
        - mindspore.nn.Conv1d
          -> api_python/nn/mindspore.nn.Conv1d.html
        - mindspore.ops.eye
          -> api_python/ops/mindspore.ops.eye.html
        """
        api_name = api_name.lstrip(".")
        parts = api_name.split(".")

        # 默认兜底
        sub_module = "mindspore"

        # 旧逻辑（保留注释）：只按二级模块定位
        # if len(parts) >= 2:
        #     sub_module = parts[1]
        # return f"{MS_DOC_BASE}{sub_module}/{api_name}.html"

        # 新逻辑：兼容 Tensor 方法路径
        # Tensor 方法示例：mindspore.Tensor.add
        # URL: https://www.mindspore.cn/docs/en/r2.8.0/api_python/mindspore/Tensor/mindspore.Tensor.add.html
        if api_name.startswith("mindspore.Tensor"):
            sub_module = "mindspore/Tensor"
        elif len(parts) >= 2:
            sub_module = parts[1]

        return f"{MS_DOC_BASE}{sub_module}/{api_name}.html"

    def parse_doc_content(
        self, soup: BeautifulSoup, api_name: str, url: str
    ) -> Dict:
        """解析 MindSpore 文档内容"""
        doc_content = {
            "api_name": api_name,
            "framework": "mindspore",
            "url": url,
            "title": soup.find("title").text if soup.find("title") else "",
            "description": "",
            "parameters": [],
            "returns": "",
            "examples": [],
            "raw_html": str(
                soup.find("main")
                or soup.find("div", class_="section")
                or soup.find("body")
                or ""
            ),
        }

        main_content = (
            soup.find("main")
            or soup.find("div", class_="section")
            or soup.find("div", class_="document")
        )

        if not main_content:
            return doc_content

        # ========== 1. API 描述 ==========
        first_p = main_content.find("p")
        if first_p:
            doc_content["description"] = first_p.get_text(strip=True)

        # ========== 2. 参数解析 ==========
        params = []

        # 情况 1：<dl class="field-list">
        dl = main_content.find("dl", class_="field-list")
        if dl:
            for dt in dl.find_all("dt"):
                name = dt.get_text(strip=True)
                dd = dt.find_next_sibling("dd")
                desc = dd.get_text(strip=True) if dd else ""
                params.append({"name": name, "description": desc})

        # 情况 2：参数 / Inputs 标题下的列表
        if not params:
            headers = main_content.find_all(
                string=re.compile(r"参数|Inputs?", re.I)
            )
            for h in headers:
                section = h.find_parent(["section", "div"])
                if not section:
                    continue
                for li in section.find_all("li"):
                    text = li.get_text(" ", strip=True)
                    if ":" in text:
                        name, desc = text.split(":", 1)
                        params.append(
                            {"name": name.strip(), "description": desc.strip()}
                        )

        doc_content["parameters"] = params

        # ========== 3. 返回值解析 ==========
        ret_headers = main_content.find_all(
            string=re.compile(r"返回值|Outputs?|Returns?", re.I)
        )
        if ret_headers:
            parent = ret_headers[0].find_parent(["p", "div", "section"])
            if parent:
                doc_content["returns"] = parent.get_text(strip=True)

        return doc_content
