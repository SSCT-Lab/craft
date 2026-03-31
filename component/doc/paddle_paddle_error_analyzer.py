# ./component/doc/paddle_paddle_error_analyzer.py
"""Analyze paddle_error reports using docs and an LLM."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

# Add the project root to sys.path so component modules can be imported.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_crawler_factory import get_doc_content
from component.migration.migrate_generate_tests import get_qwen_client

DEFAULT_MODEL = "qwen-flash"
DEFAULT_KEY_PATH = "aliyun.key"


def parse_paddle_error_report(report_path: Path) -> List[Dict[str, Any]]:
    """Parse paddle_error_samples_report.txt and extract sample info.

    Report format:
    ==============================================================================
    文件: xxx.json
    --------------------------------------------------------------------------------
    样例 1:
    paddle_error: xxx
    torch_test_case:
    {...}
    paddle_test_case:
    {...}
    """
    samples: List[Dict[str, Any]] = []

    current_file: Optional[str] = None
    current_index: Optional[int] = None
    current_error: Optional[str] = None
    torch_lines: List[str] = []
    paddle_lines: List[str] = []
    mode: Optional[str] = None  # "torch" / "paddle" / None

    def flush_sample():
        """Flush the current sample when switching file/sample."""
        nonlocal current_file, current_index, current_error, torch_lines, paddle_lines
        if current_index is None:
            return
        try:
            torch_str = "".join(torch_lines).strip()
            paddle_str = "".join(paddle_lines).strip()
            if not torch_str or not paddle_str:
                return
            torch_case = json.loads(torch_str)
            paddle_case = json.loads(paddle_str)
        except Exception as e:
            print(
                f"[WARN] Sample parsing failed (file: {current_file}, index: {current_index}): {e}"
            )
            return

        samples.append(
            {
                "file": current_file,
                "index": current_index,
                "paddle_error": current_error or "",
                "torch_test_case": torch_case,
                "paddle_test_case": paddle_case,
            }
        )

    with report_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            # Handle separator lines of '=' or '-' to avoid JSON pollution
            if stripped and (set(stripped) == {"="} or set(stripped) == {"-"}):
                flush_sample()
                current_index = None
                current_error = None
                torch_lines = []
                paddle_lines = []
                mode = None
                continue

            if stripped.startswith("文件:"):
                # New file begins; flush any pending sample
                flush_sample()
                current_file = stripped.split("文件:", 1)[1].strip()
                current_index = None
                current_error = None
                torch_lines = []
                paddle_lines = []
                mode = None
                continue

            if stripped.startswith("样例"):
                # New sample begins; flush previous sample
                flush_sample()
                try:
                    # e.g., "样例 1:" -> 1
                    idx_part = stripped.split("样例", 1)[1]
                    idx_part = idx_part.strip(" :")
                    current_index = int(idx_part)
                except Exception:
                    current_index = None
                current_error = None
                torch_lines = []
                paddle_lines = []
                mode = None
                continue

            if stripped.startswith("paddle_error:"):
                current_error = stripped.split("paddle_error:", 1)[1].strip()
                continue

            if stripped.startswith("torch_test_case:"):
                mode = "torch"
                continue

            if stripped.startswith("paddle_test_case:"):
                mode = "paddle"
                continue

            # Collect JSON content
            if mode == "torch":
                torch_lines.append(raw_line)
            elif mode == "paddle":
                paddle_lines.append(raw_line)

    # Flush the last sample at EOF
    flush_sample()

    return samples


def build_sample_prompt(
    sample: Dict[str, Any],
    paddle_docs: List[str],
    pt_docs: List[str],
) -> str:
    """Build the LLM prompt for a single paddle_error sample."""
    file_name = sample.get("file") or ""
    index = sample.get("index")
    paddle_error = sample.get("paddle_error", "")
    torch_case = sample.get("torch_test_case", {})
    paddle_case = sample.get("paddle_test_case", {})

    torch_api = torch_case.get("api", "")
    paddle_api = paddle_case.get("api", "")

    paddle_docs_text = "\n\n".join(paddle_docs) if paddle_docs else "No relevant PaddlePaddle docs found"
    pt_docs_text = "\n\n".join(pt_docs) if pt_docs else "No relevant PyTorch docs found"

    torch_case_json = json.dumps(torch_case, ensure_ascii=False, indent=2)
    paddle_case_json = json.dumps(paddle_case, ensure_ascii=False, indent=2)

    prompt = f"""You are a senior framework expert familiar with PyTorch and PaddlePaddle, analyzing a PaddlePaddle execution error sample.

[Sample Info]
- Source file: {file_name}
- Sample index: {index}
- paddle_error message: {paddle_error}

[Test Case Info]
1. PyTorch test case (torch_test_case, JSON):
```json
{torch_case_json}
```

2. PaddlePaddle test case (paddle_test_case, JSON):
```json
{paddle_case_json}
```

[Candidate API Mapping]
- PyTorch API: {torch_api}
- PaddlePaddle API: {paddle_api}

[Relevant Official Docs (PaddlePaddle)]
{paddle_docs_text}

[Relevant Official Docs (PyTorch)]
{pt_docs_text}

----------------------------------------
[Analysis Task]
Based on the information above, analyze the most likely cause of paddle_error in this sample. Focus on distinguishing:

1. Framework behavior differences (Category A)
    - Examples: different broadcasting rules, PaddlePaddle not supporting certain shapes (e.g., empty tensors with 0 in shape), or differing implementations causing failures.
    - Common errors: "Broadcast dimension mismatch" (broadcasting rule differences), dtype support differences (e.g., bitwise ops not supporting float32), etc.

2. Test case / parameter mapping mismatch (Category B)
    - Examples: parameter name mapping errors (e.g., PyTorch `dim` should map to PaddlePaddle `axis`),
    - Incorrect parameter format conversions (e.g., dtype string format incorrect; 'torch.float32' should map to 'float32'),
    - Nested dicts not converted to actual tensors (e.g., PaddlePaddle receives a dict instead of a tensor),
    - API not found (e.g., "PaddlePaddle算子 xxx 未找到" indicates a misspelled or nonexistent API name).

3. API mapping error (Category C)
    - Examples: a PyTorch API mapped to a semantically different PaddlePaddle API;
    - Or PaddlePaddle has no direct counterpart and requires composition.

4. Other or insufficient information (Category D)

----------------------------------------
[Output Requirements]
Provide a rigorous technical analysis and follow this structure:

1. Conclusion label:
    - Provide a single label line in the format:
      - Conclusion: A Framework behavior difference
      - Conclusion: B Test case/parameter mapping issue
      - Conclusion: C API mapping error
      - Conclusion: D Other/insufficient info

2. Root cause analysis (1-2 concise points):
    - Use the paddle_error message, input shape/dtype, and doc constraints/behavior to explain the failure.
    - If you choose C (API mapping error), clearly and concisely state the key semantic/parameter differences.

3. Fix suggestions (1-2 short sentences):
    - If Category A, decide whether it's worth filing an issue with a brief reason (only for previously unknown inconsistencies/bugs).
    - If Category C/D, no fix suggestions are needed.
    - If Category B, provide concise suggestions to adjust mappings or test cases, for example:
      - Fix API name spelling (e.g., padle.add -> paddle.add);
      - Adjust parameter name mapping (e.g., dim -> axis, x1 -> x);
      - Fix dtype format conversion logic (e.g., torch.float32 -> float32);
      - Ensure nested API calls are expanded and executed correctly;
      - Add/adjust parameters to keep alignment.
"""

    return prompt


def analyze_sample_with_llm(
    client,
    sample: Dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> Optional[str]:
    """Analyze a single paddle_error sample with an LLM."""
    torch_case = sample.get("torch_test_case", {})
    paddle_case = sample.get("paddle_test_case", {})

    torch_api = torch_case.get("api", "")
    paddle_api = paddle_case.get("api", "")

    paddle_docs: List[str] = []
    pt_docs: List[str] = []

    # Fetch Paddle docs
    if paddle_api:
        try:
            doc_text = get_doc_content(paddle_api, "paddle")
            if doc_text and "Unable to fetch" not in doc_text:
                paddle_docs.append(doc_text)
        except Exception as e:
            print(f"[WARN] Failed to fetch PaddlePaddle docs {paddle_api}: {e}")

    # Fetch PT docs
    if torch_api:
        try:
            doc_text = get_doc_content(torch_api, "pytorch")
            if doc_text and "Unable to fetch" not in doc_text:
                pt_docs.append(doc_text)
        except Exception as e:
            print(f"[WARN] Failed to fetch PyTorch docs {torch_api}: {e}")

    prompt = build_sample_prompt(sample, paddle_docs=paddle_docs, pt_docs=pt_docs)

    try:
        if hasattr(client, "chat"):
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048,
            )
            return resp.choices[0].message.content.strip()
        else:
            resp = client.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        return None


def save_categorized_sample(
    sample: Dict[str, Any],
    analysis: str,
    category: str,
) -> None:
    """Save a categorized sample to its folder."""
    base_dir = Path("pt_pd_test") / "analysis"
    if category == "A":
        out_dir = base_dir / "paddle_a"
    elif category == "D":
        out_dir = base_dir / "paddle_d"
    else:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    file_name = sample.get("file") or "unknown_file"
    index = sample.get("index")
    paddle_error = sample.get("paddle_error", "")
    torch_case = sample.get("torch_test_case", {})
    paddle_case = sample.get("paddle_test_case", {})

    torch_case_json = json.dumps(torch_case, ensure_ascii=False, indent=2)
    paddle_case_json = json.dumps(paddle_case, ensure_ascii=False, indent=2)

    safe_file_name = file_name.replace("/", "_").replace("\\", "_")
    out_path = out_dir / f"{safe_file_name}_sample{index}.txt"

    content_parts = [
        f"Source file: {file_name}",
        f"Sample index: {index}",
        f"paddle_error message: {paddle_error}",
        "",
        "[PyTorch Test Case]",
        torch_case_json,
        "",
        "[PaddlePaddle Test Case]",
        paddle_case_json,
        "",
        "[LLM Analysis Result]",
        analysis,
    ]
    out_path.write_text("\n".join(content_parts), encoding="utf-8")


def main():
    """CLI entry: batch analyze paddle_error report samples."""
    parser = argparse.ArgumentParser(
        description="Analyze causes of paddle_error samples"
    )
    parser.add_argument(
        "--report",
        "-r",
        required=True,
        help="Path to paddle_error_samples_report.txt",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples to analyze (default: all)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help="LLM model name (default: qwen-flash)",
    )
    parser.add_argument(
        "--key-path",
        "-k",
        default=DEFAULT_KEY_PATH,
        help="Path to API key file (default: aliyun.key)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for analysis results (defaults to same directory)",
    )

    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"[ERROR] Report file not found: {report_path}")
        return

    print(f"[INFO] Parsing report file: {report_path}")
    samples = parse_paddle_error_report(report_path)
    if not samples:
        print("[ERROR] No samples parsed from report")
        return

    if args.limit is not None:
        samples = samples[: args.limit]

    print(f"[INFO] Total samples to analyze: {len(samples)}")

    try:
        client = get_qwen_client(args.key_path)
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM client: {e}")
        return

    # Determine output file path
    if args.output:
        out_path = Path(args.output)
    else:
        stem = report_path.stem
        if stem == "paddle_error_samples_report":
            out_name = "llm_paddle_error_analysis.txt"
        else:
            out_name = f"{stem}_llm_paddle_error_analysis.txt"
        out_path = report_path.with_name(out_name)

    outputs_batch: List[str] = []

    for i, sample in enumerate(samples, start=1):
        file_name = sample.get("file") or ""
        index = sample.get("index")
        print(f"[INFO] Analyzing sample {i}/{len(samples)} (file: {file_name}, index: {index})")

        analysis = analyze_sample_with_llm(client, sample, model=args.model)

        if analysis:
            if "Conclusion: A" in analysis:
                save_categorized_sample(sample, analysis, "A")
            elif "Conclusion: D" in analysis:
                save_categorized_sample(sample, analysis, "D")

        header = f"Sample {index} (file: {file_name}) analysis results"
        sep = "=" * 80
        block = [sep, header, sep]
        if analysis:
            block.append(analysis)
        else:
            block.append("[ERROR] Sample analysis failed")
        outputs_batch.append("\n".join(block))

        # Write every 50 samples, or on the last sample
        if i % 50 == 0 or i == len(samples):
            batch_text = "\n\n".join(outputs_batch)
            mode = "w" if i <= 50 else "a"
            with out_path.open(mode, encoding="utf-8") as f:
                if mode == "a":
                    f.write("\n\n")
                f.write(batch_text)
            outputs_batch = []

    print(f"[SUCCESS] Analysis results saved to: {out_path}")


if __name__ == "__main__":
    main()
