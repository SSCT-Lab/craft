"""导出指定测试的完整评估提示词到一个 txt 文件，给老板试读用。

说明：
- 不调用 LLM，只复用当前的提示词结构（build_analysis_prompt）。
- TF/PT 代码、文档、输出都来自本地文件：
  - 代码：从迁移后的测试文件中用正则提取 TF / PT 测试函数。
  - 文档：从 migrated_tests/用所选项目新建的文件夹/api_docs.json 中读取。
  - 输出：直接嵌入 data/logs/*.log 的真实内容（当前主要是 PT 侧日志）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_analyzer import build_analysis_prompt  # noqa: E402


TARGET_TEST_FILES = [
    "migrated_tests/用所选项目新建的文件夹/testArgRenames.py",
    "migrated_tests/用所选项目新建的文件夹/testBasic.py",
    "migrated_tests/用所选项目新建的文件夹/testCanLoadWithPkgutil.py",
]


def load_api_docs() -> tuple[dict, dict]:
    """加载 per-file API 列表和 API 对应文档内容。"""
    api_json = ROOT / "migrated_tests/用所选项目新建的文件夹/api_docs.json"
    if not api_json.exists():
        raise FileNotFoundError(f"找不到 api_docs.json: {api_json}")
    data = json.loads(api_json.read_text(encoding="utf-8"))
    file_apis = data.get("files") or {}
    api_docs = data.get("docs") or {}
    return file_apis, api_docs


def extract_tf_pt_code(test_file: Path) -> tuple[str, str]:
    """从迁移后的测试文件中粗略提取 TF / PT 测试函数代码片段。"""
    content = test_file.read_text(encoding="utf-8", errors="ignore")

    # 与 analyze_test_error 中保持一致的正则
    tf_match = re.search(
        r"def\s+(test\w+)\(\):.*?(?=def\s+test.*?_pt|# ===== PyTorch|if __name__)",
        content,
        re.DOTALL,
    )
    tf_code = tf_match.group(0).strip() if tf_match else ""

    pt_match = re.search(
        r"def\s+(test.*?_pt)\(\):.*?(?=def\s+test.*?|# ===== Main|if __name__)",
        content,
        re.DOTALL,
    )
    pt_code = pt_match.group(0).strip() if pt_match else ""

    return tf_code, pt_code


def load_docs_for_file(
    file_name: str, file_apis: dict, api_docs: dict
) -> tuple[list[str], list[str]]:
    """根据 api_docs.json 为指定文件收集 TF / PT 文档文本。"""
    apis = file_apis.get(file_name, []) or []
    tf_docs: list[str] = []
    pt_docs: list[str] = []

    for api in apis:
        doc = api_docs.get(api)
        if not doc:
            continue
        content = doc.get("content") or ""
        if not content:
            continue
        if api.startswith("tf."):
            tf_docs.append(content)
        elif api.startswith("torch."):
            pt_docs.append(content)

    return tf_docs, pt_docs


def load_log_output(stem: str) -> str:
    """读取对应的日志文件内容（主要是 PT 侧执行日志）。"""
    log_path = ROOT / "data" / "logs" / f"{stem}.log"
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="ignore")


def build_prompt_for_test(test_path: str, file_apis: dict, api_docs: dict) -> str:
    """构造单个测试文件的完整提示词字符串。"""
    p = ROOT / test_path
    name = p.name

    tf_code, pt_code = extract_tf_pt_code(p)
    tf_docs, pt_docs = load_docs_for_file(name, file_apis, api_docs)

    # 当前执行环境主要是跑 PyTorch 侧，因此 TF 输出留空占位，PT 输出用真实日志
    tf_output = "=== STDOUT ===\n\n=== STDERR ===\n"
    pt_output = load_log_output(p.stem)

    # 为了让提示词看起来更自然，这里用一句话说明去哪里看日志
    error_message = f"该 case 的 PyTorch 执行日志见 data/logs/{p.stem}.log，请结合下面提供的完整 stderr/stdout 进行分析。"

    prompt = build_analysis_prompt(
        error_message=error_message,
        tf_code=tf_code,
        pt_code=pt_code,
        tf_docs=tf_docs,
        pt_docs=pt_docs,
        tf_output=tf_output,
        pt_output=pt_output,
        context=None,
    )

    header = f"{'='*30}\nPrompt: {test_path}\n{'='*30}\n\n"
    return header + prompt + "\n\n"


def main() -> None:
    file_apis, api_docs = load_api_docs()

    out_path = ROOT / "migrated_tests/用所选项目新建的文件夹/prompts_for_boss_generated1.txt"
    pieces: list[str] = []

    for test in TARGET_TEST_FILES:
        pieces.append(build_prompt_for_test(test, file_apis, api_docs))

    out_path.write_text("".join(pieces), encoding="utf-8")
    print(f"[DONE] 已生成提示词到: {out_path}")


if __name__ == "__main__":
    main()


