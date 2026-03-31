"""PaddlePaddle documentation crawler."""
import re
from typing import Dict
from bs4 import BeautifulSoup
from component.doc.doc_crawler_base import DocCrawler

# PADDLE_DOC_BASE = "https://www.paddlepaddle.org.cn/documentation/docs/zh/api/"
PADDLE_DOC_BASE = "https://www.paddlepaddle.org.cn/documentation/docs/en/api/"


class PaddleDocCrawler(DocCrawler):
    """PaddlePaddle documentation crawler."""

    def __init__(self):
        super().__init__("paddle")

    def build_doc_url(self, api_name: str) -> str:
        """
                Build Paddle API doc URL.

                Examples:
        - paddle.nn.Conv2D
          -> https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Conv2D_cn.html
        - paddle.add
          -> https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/add_cn.html
        """
        api_name = api_name.lstrip(".")
        path = api_name.replace(".", "/")
        return f"{PADDLE_DOC_BASE}{path}_en.html"

    def parse_doc_content(
        self, soup: BeautifulSoup, api_name: str, url: str
    ) -> Dict:
        """Parse Paddle doc content."""
        doc_content = {
            "api_name": api_name,
            "framework": "paddle",
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

        # ========== 1. API description ==========
        first_p = main_content.find("p")
        if first_p:
            doc_content["description"] = first_p.get_text(strip=True)

        # ========== 2. Parameter parsing ==========
        params = []

        # Case 1: <dl class="field-list">
        dl = main_content.find("dl", class_="field-list")
        if dl:
            for dt in dl.find_all("dt"):
                name = dt.get_text(strip=True)
                dd = dt.find_next_sibling("dd")
                desc = dd.get_text(strip=True) if dd else ""
                params.append({"name": name, "description": desc})

        # Case 2: parameters in tables
        if not params:
            tables = main_content.find_all("table")
            for table in tables:
                headers = [th.get_text(strip=True) for th in table.find_all("th")]
                if "参数" in headers or "Parameter" in headers:
                    for row in table.find_all("tr")[1:]:
                        cols = row.find_all("td")
                        if len(cols) >= 2:
                            params.append(
                                {
                                    "name": cols[0].get_text(strip=True),
                                    "description": cols[1].get_text(strip=True),
                                }
                            )

        doc_content["parameters"] = params

        # ========== 3. Return parsing ==========
        return_keywords = re.compile(r"返回值|返回|Returns?", re.I)
        ret_node = main_content.find(string=return_keywords)
        if ret_node:
            parent = ret_node.find_parent(["p", "div", "section"])
            if parent:
                doc_content["returns"] = parent.get_text(strip=True)

        return doc_content
