"""Export full evaluation prompts for selected tests to a txt file (for review).

Notes:
- Does not call the LLM; reuses the current prompt structure (build_analysis_prompt).
- TF/PT code, docs, and outputs are from local files:
    - Code: extract TF/PT test functions from migrated tests using regex.
    - Docs: read from migrated_tests/用所选项目新建的文件夹/api_docs.json.
    - Output: embed real data/logs/*.log content (currently mostly PT logs).
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
    """Load per-file API lists and corresponding doc content."""
    api_json = ROOT / "migrated_tests/用所选项目新建的文件夹/api_docs.json"
    if not api_json.exists():
        raise FileNotFoundError(f"api_docs.json not found: {api_json}")
    data = json.loads(api_json.read_text(encoding="utf-8"))
    file_apis = data.get("files") or {}
    api_docs = data.get("docs") or {}
    return file_apis, api_docs


def extract_tf_pt_code(test_file: Path) -> tuple[str, str]:
    """Roughly extract TF/PT test function code from migrated tests."""
    content = test_file.read_text(encoding="utf-8", errors="ignore")

    # Keep regex consistent with analyze_test_error
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
    """Collect TF/PT doc text for a given file from api_docs.json."""
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
    """Read corresponding log content (mainly PT execution logs)."""
    log_path = ROOT / "data" / "logs" / f"{stem}.log"
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="ignore")


def build_prompt_for_test(test_path: str, file_apis: dict, api_docs: dict) -> str:
    """Build the full prompt text for a single test file."""
    p = ROOT / test_path
    name = p.name

    tf_code, pt_code = extract_tf_pt_code(p)
    tf_docs, pt_docs = load_docs_for_file(name, file_apis, api_docs)

    # Current environment runs PyTorch side; keep TF output empty, PT output from logs
    tf_output = "=== STDOUT ===\n\n=== STDERR ===\n"
    pt_output = load_log_output(p.stem)

    # Add a note to point to logs for context
    error_message = (
        f"PyTorch execution log for this case is in data/logs/{p.stem}.log. "
        "Please analyze using the full stderr/stdout below."
    )

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
    print(f"[DONE] Prompts generated at: {out_path}")


if __name__ == "__main__":
    main()


