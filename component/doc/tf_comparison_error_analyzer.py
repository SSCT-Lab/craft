# ./component/doc/comparison_error_analyzer.py
"""Analyze comparison_error reports using docs and an LLM."""

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


def parse_comparison_error_report(report_path: Path) -> List[Dict[str, Any]]:
    """Parse comparison_error_samples_report.txt and extract sample info."""
    samples: List[Dict[str, Any]] = []

    current_file: Optional[str] = None
    current_index: Optional[int] = None
    current_error: Optional[str] = None
    torch_lines: List[str] = []
    tf_lines: List[str] = []
    mode: Optional[str] = None  # "torch" / "tf" / None

    def flush_sample():
        """Flush the current sample when switching file/sample."""
        nonlocal current_file, current_index, current_error, torch_lines, tf_lines
        if current_index is None:
            return
        try:
            torch_str = "".join(torch_lines).strip()
            tf_str = "".join(tf_lines).strip()
            if not torch_str or not tf_str:
                return
            torch_case = json.loads(torch_str)
            tf_case = json.loads(tf_str)
        except Exception as e:
            print(
                f"[WARN] Sample parsing failed (file: {current_file}, index: {current_index}): {e}"
            )
            return

        samples.append(
            {
                "file": current_file,
                "index": current_index,
                "comparison_error": current_error or "",
                "torch_test_case": torch_case,
                "tensorflow_test_case": tf_case,
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
                tf_lines = []
                mode = None
                continue

            if stripped.startswith("文件:"):
                # New file begins; flush any pending sample
                flush_sample()
                current_file = stripped.split("文件:", 1)[1].strip()
                current_index = None
                current_error = None
                torch_lines = []
                tf_lines = []
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
                tf_lines = []
                mode = None
                continue

            if stripped.startswith("comparison_error:"):
                current_error = stripped.split("comparison_error:", 1)[1].strip()
                continue

            if stripped.startswith("torch_test_case:"):
                mode = "torch"
                continue

            if stripped.startswith("tensorflow_test_case:"):
                mode = "tf"
                continue

            # Collect JSON content
            if mode == "torch":
                torch_lines.append(raw_line)
            elif mode == "tf":
                tf_lines.append(raw_line)

    # Flush the last sample at EOF
    flush_sample()

    return samples


def build_sample_prompt(
    sample: Dict[str, Any],
    tf_docs: List[str],
    pt_docs: List[str],
) -> str:
    """Build the LLM prompt for a single comparison_error sample."""
    file_name = sample.get("file") or ""
    index = sample.get("index")
    comparison_error = sample.get("comparison_error", "")
    torch_case = sample.get("torch_test_case", {})
    tf_case = sample.get("tensorflow_test_case", {})

    torch_api = torch_case.get("api", "")
    tf_api = tf_case.get("api", "")

    tf_docs_text = "\n\n".join(tf_docs) if tf_docs else "No relevant TensorFlow docs found"
    pt_docs_text = "\n\n".join(pt_docs) if pt_docs else "No relevant PyTorch docs found"

    torch_case_json = json.dumps(torch_case, ensure_ascii=False, indent=2)
    tf_case_json = json.dumps(tf_case, ensure_ascii=False, indent=2)

    prompt = f"""You are a senior framework expert familiar with PyTorch and TensorFlow, analyzing a comparison error sample.

[Sample Info]
- Source file: {file_name}
- Sample index: {index}
- comparison_error description: {comparison_error}

[Test Case Info]
1. PyTorch test case (torch_test_case, JSON):
```json
{torch_case_json}
```

2. TensorFlow test case (tensorflow_test_case, JSON):
```json
{tf_case_json}
```

[Candidate API Mapping]
- PyTorch API: {torch_api}
- TensorFlow API: {tf_api}

[Relevant Official Docs (TensorFlow)]
{tf_docs_text}

[Relevant Official Docs (PyTorch)]
{pt_docs_text}

----------------------------------------
[Analysis Task]
Based on the information above, analyze the most likely cause of comparison_error and distinguish:

1. Framework behavior differences (Category A)
    - Examples: different broadcasting rules, numerical stability/precision strategies, or differing implementations leading to inconsistent outputs for identical inputs.

2. Test case / input construction mismatch (Category B)
    - Examples: mismatched input shape/dtype, misaligned default parameters, or extra operations on the TensorFlow side.

3. API mapping error (Category C)
    - Examples: should map to tf.nn.softmax but incorrectly mapped to tf.math.softmax;
    - Or PyTorch and TensorFlow APIs are semantically different and cannot be treated as the same operator.

4. Other or insufficient information (Category D)

----------------------------------------
[Output Requirements]
Provide a rigorous technical analysis and follow this structure:

1. Conclusion label:
    - Provide a single label line in the format:
      - Conclusion: A Framework behavior difference
      - Conclusion: B Test case construction issue
      - Conclusion: C API mapping error
      - Conclusion: D Other/insufficient info

2. Detailed root cause analysis:
    - Use the comparison_error description, input shape/dtype, and doc constraints/behavior to explain the mismatch.
    - If you choose C (API mapping error), clearly state the key semantic/parameter differences.

3. Fix suggestions:
    - If Category A, decide whether it's worth filing an issue with a brief reason (only for previously unknown inconsistencies/bugs).
    - If Category C/D, no fix suggestions are needed.
    - If Category B, suggest how to adjust mappings or test cases, for example:
      - Map to a more appropriate Paddle or PyTorch API;
      - Add/adjust parameters;
      - Align input shape/dtype.
"""

    return prompt


def analyze_sample_with_llm(
    client,
    sample: Dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> Optional[str]:
    torch_case = sample.get("torch_test_case", {})
    tf_case = sample.get("tensorflow_test_case", {})

    torch_api = torch_case.get("api", "")
    tf_api = tf_case.get("api", "")

    tf_docs: List[str] = []
    pt_docs: List[str] = []

    # Fetch TF docs
    if tf_api:
        try:
            doc_text = get_doc_content(tf_api, "tensorflow")
            if doc_text and "Unable to fetch" not in doc_text:
                tf_docs.append(doc_text)
        except Exception as e:
            print(f"[WARN] Failed to fetch TensorFlow docs {tf_api}: {e}")

    # Fetch PT docs
    if torch_api:
        try:
            doc_text = get_doc_content(torch_api, "pytorch")
            if doc_text and "Unable to fetch" not in doc_text:
                pt_docs.append(doc_text)
        except Exception as e:
            print(f"[WARN] Failed to fetch PyTorch docs {torch_api}: {e}")

    prompt = build_sample_prompt(sample, tf_docs=tf_docs, pt_docs=pt_docs)

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
    base_dir = Path("pt_tf_test") / "analysis"
    if category == "A":
        out_dir = base_dir / "comparison_a"
    elif category == "D":
        out_dir = base_dir / "comparison_d"
    else:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    file_name = sample.get("file") or "unknown_file"
    index = sample.get("index")
    comparison_error = sample.get("comparison_error", "")
    torch_case = sample.get("torch_test_case", {})
    tf_case = sample.get("tensorflow_test_case", {})

    torch_case_json = json.dumps(torch_case, ensure_ascii=False, indent=2)
    tf_case_json = json.dumps(tf_case, ensure_ascii=False, indent=2)

    safe_file_name = file_name.replace("/", "_").replace("\\", "_")
    out_path = out_dir / f"{safe_file_name}_sample{index}.txt"

    content_parts = [
        f"Source file: {file_name}",
        f"Sample index: {index}",
        f"comparison_error description: {comparison_error}",
        "",
        torch_case_json,
        "",
        tf_case_json,
    ]
    out_path.write_text("\n".join(content_parts), encoding="utf-8")


def main():
    """CLI entry: batch analyze comparison_error report samples."""
    parser = argparse.ArgumentParser(
        description="Analyze causes of comparison_error samples"
    )
    parser.add_argument(
        "--report",
        "-r",
        required=True,
        help="Path to comparison_error_samples_report.txt",
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
        help="Output path for analysis results (defaults to stdout)",
    )

    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"[ERROR] Report file not found: {report_path}")
        return

    print(f"[INFO] Parsing report file: {report_path}")
    samples = parse_comparison_error_report(report_path)
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

    outputs: List[str] = []

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
        outputs.append("\n".join(block))

    result_text = "\n\n".join(outputs)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(result_text, encoding="utf-8")
        print(f"[SUCCESS] Analysis results saved to: {out_path}")
    else:
        print(result_text)


if __name__ == "__main__":
    main()

