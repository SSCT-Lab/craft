"""Analyze PyTorch vs MindSpore comparison_error reports using docs and LLM."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

# Add project root to path to import component modules
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
    ms_lines: List[str] = []
    mode: Optional[str] = None

    def flush_sample():
        """Flush the current sample when switching file/sample."""
        nonlocal current_file, current_index, current_error, torch_lines, ms_lines
        if current_index is None:
            return
        try:
            # Merge accumulated JSON lines into strings
            torch_str = "".join(torch_lines).strip()
            ms_str = "".join(ms_lines).strip()
            if not torch_str or not ms_str:
                return
            torch_case = json.loads(torch_str)
            ms_case = json.loads(ms_str)
        except Exception as e:
            print(
                f"[WARN] Sample parse failed (file: {current_file}, sample: {current_index}): {e}"
            )
            return

        samples.append(
            {
                "file": current_file,
                "index": current_index,
                "comparison_error": current_error or "",
                "torch_test_case": torch_case,
                "mindspore_test_case": ms_case,
            }
        )

    with report_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            # Handle separator lines made of '=' to avoid mixing into JSON
            if stripped and set(stripped) == {"="}:
                flush_sample()
                current_index = None
                current_error = None
                torch_lines = []
                ms_lines = []
                mode = None
                continue

            # Start of each JSON file block
            if stripped.startswith("文件:"):
                flush_sample()
                current_file = stripped.split("文件:", 1)[1].strip()
                current_index = None
                current_error = None
                torch_lines = []
                ms_lines = []
                mode = None
                continue

            # Start of each sample, e.g., "样例 1:"
            if stripped.startswith("样例"):
                flush_sample()
                try:
                    idx_part = stripped.split("样例", 1)[1]
                    idx_part = idx_part.strip(" :")
                    current_index = int(idx_part)
                except Exception:
                    current_index = None
                current_error = None
                torch_lines = []
                ms_lines = []
                mode = None
                continue

            if stripped.startswith("comparison_error:"):
                current_error = stripped.split("comparison_error:", 1)[1].strip()
                continue

            # Mark which framework JSON content follows
            if stripped.startswith("torch_test_case:"):
                mode = "torch"
                continue

            if stripped.startswith("mindspore_test_case:"):
                mode = "mindspore"
                continue

            # Collect JSON lines based on current mode
            if mode == "torch":
                torch_lines.append(raw_line)
            elif mode == "mindspore":
                ms_lines.append(raw_line)

    flush_sample()

    return samples


def build_sample_prompt(
    sample: Dict[str, Any],
    ms_docs: List[str],
    pt_docs: List[str],
) -> str:
    """Build prompt text for a single comparison_error sample."""
    file_name = sample.get("file") or ""
    index = sample.get("index")
    comparison_error = sample.get("comparison_error", "")
    torch_case = sample.get("torch_test_case", {})
    ms_case = sample.get("mindspore_test_case", {})

    torch_api = torch_case.get("api", "")
    ms_api = ms_case.get("api", "")

    ms_docs_text = "\n\n".join(ms_docs) if ms_docs else "No relevant MindSpore docs found"
    pt_docs_text = "\n\n".join(pt_docs) if pt_docs else "No relevant PyTorch docs found"

    torch_case_json = json.dumps(torch_case, ensure_ascii=False, indent=2)
    ms_case_json = json.dumps(ms_case, ensure_ascii=False, indent=2)

    prompt = f"""You are a senior framework expert familiar with PyTorch and MindSpore. Analyze a comparison error sample.

[Sample Info]
- Source file: {file_name}
- Sample index: {index}
- comparison_error description: {comparison_error}

[Test Case Info]
1. PyTorch test case (torch_test_case, JSON):
```json
{torch_case_json}
```

2. MindSpore test case (mindspore_test_case, JSON):
```json
{ms_case_json}
```

[Candidate API Mapping]
- PyTorch API: {torch_api}
- MindSpore API: {ms_api}

[Official Docs (MindSpore)]
{ms_docs_text}

[Official Docs (PyTorch)]
{pt_docs_text}

----------------------------------------
[Analysis Task]
Based on the above, analyze the most likely cause of comparison_error and distinguish:

1. Framework behavior differences (Category A)
    - e.g., different broadcasting rules, numeric stability/precision policies, or implementation
      differences causing output mismatches for the same inputs.

2. Test case / input construction mismatch (Category B)
    - e.g., shape/dtype mismatches, default parameters not aligned, extra operations on one side.

3. API mapping error (Category C)
    - e.g., should map to mindspore.ops.softmax but was mapped to mindspore.nn.Softmax;
    - or PyTorch and MindSpore API semantics clearly differ and are not equivalent.

4. Other causes or insufficient information (Category D)

----------------------------------------
[Output Requirements]
Provide a rigorous technical analysis with the following structure:

1. Conclusion label:
    - Provide one label line, format examples:
      - Conclusion: A framework behavior difference
      - Conclusion: B test case issue
      - Conclusion: C API mapping error
      - Conclusion: D other/insufficient info

2. Detailed analysis:
    - Use the comparison_error description, input shape/dtype, and doc constraints/behavior to
      explain the error.
    - If you choose C (API mapping error), call out key semantic/parameter differences.

3. Fix suggestions:
    - For A, decide whether to file an issue and provide a brief reason (no full issue text).
    - For C/D, no fix suggestion needed.
    - For B, suggest how to adjust the mapping or test case, e.g.:
      - map to a more appropriate MindSpore or PyTorch API;
      - add/modify a parameter;
      - adjust input shape/dtype to align.
"""

    return prompt


def analyze_sample_with_llm(
    client,
    sample: Dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> Optional[str]:
    """Analyze a single sample with the LLM."""
    torch_case = sample.get("torch_test_case", {})
    ms_case = sample.get("mindspore_test_case", {})

    # Extract API names from sample
    torch_api = torch_case.get("api", "")
    ms_api = ms_case.get("api", "")

    ms_docs: List[str] = []
    pt_docs: List[str] = []

    # Fetch MindSpore official docs
    if ms_api:
        try:
            doc_text = get_doc_content(ms_api, "mindspore")
            if doc_text and "Unable to fetch" not in doc_text:
                ms_docs.append(doc_text)
        except Exception as e:
            print(f"[WARN] Failed to fetch MindSpore docs {ms_api}: {e}")

    # Fetch PyTorch official docs
    if torch_api:
        try:
            doc_text = get_doc_content(torch_api, "pytorch")
            if doc_text and "Unable to fetch" not in doc_text:
                pt_docs.append(doc_text)
        except Exception as e:
            print(f"[WARN] Failed to fetch PyTorch docs {torch_api}: {e}")

    prompt = build_sample_prompt(sample, ms_docs=ms_docs, pt_docs=pt_docs)

    try:
        if hasattr(client, "chat"):
            # New SDK call style
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048,
            )
            return resp.choices[0].message.content.strip()
        else:
            # Legacy SDK call style
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
    """Save categorized sample info to the appropriate directory."""
    base_dir = Path("pt_ms_test") / "analysis"
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
    ms_case = sample.get("mindspore_test_case", {})

    torch_case_json = json.dumps(torch_case, ensure_ascii=False, indent=2)
    ms_case_json = json.dumps(ms_case, ensure_ascii=False, indent=2)

    safe_file_name = file_name.replace("/", "_").replace("\\", "_")
    out_path = out_dir / f"{safe_file_name}_sample{index}.txt"

    content_parts = [
        f"Source file: {file_name}",
        f"Sample index: {index}",
        f"comparison_error description: {comparison_error}",
        "",
        torch_case_json,
        "",
        ms_case_json,
    ]
    out_path.write_text("\n".join(content_parts), encoding="utf-8")


def main():
    """CLI entry point: batch analyze PyTorch vs MindSpore comparison_error samples."""
    parser = argparse.ArgumentParser(
        description="Analyze causes of PyTorch vs MindSpore comparison_error samples"
    )
    parser.add_argument(
        "--report",
        "-r",
        required=True,
        help="Path to pt_ms_test/analysis/comparison_error_samples_report.txt",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples to analyze (default all)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help="LLM model name (default qwen-flash)",
    )
    parser.add_argument(
        "--key-path",
        "-k",
        default=DEFAULT_KEY_PATH,
        help="API key file path (default aliyun.key)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (auto-generate in same dir if omitted)",
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

    print(f"[INFO] Samples to analyze: {len(samples)}")

    try:
        client = get_qwen_client(args.key_path)
    except Exception as e:
        print(f"[ERROR] Unable to initialize LLM client: {e}")
        return

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        stem = report_path.stem
        if stem == "comparison_error_samples_report":
            out_name = "llm_comparison_error_analysis.txt"
        else:
            out_name = f"{stem}_llm_comparison_error_analysis.txt"
        out_path = report_path.with_name(out_name)

    outputs_batch: List[str] = []
    processed_count = 0
    has_written = False
    skipped_due_to_error = 0
    llm_success = 0
    llm_failed = 0

    for i, sample in enumerate(samples, start=1):
        comparison_error = sample.get("comparison_error", "")
        if "比较过程出错" in comparison_error:
            skipped_due_to_error += 1
            # print(
            #     f"[DEBUG] Skip sample (comparison_error contains '比较过程出错'): "
            #     f"file={sample.get('file')}, sample={sample.get('index')}"
            # )
            continue

        file_name = sample.get("file") or ""
        index = sample.get("index")
        print(
            f"[INFO] Analyzing sample {i}/{len(samples)} (file: {file_name}, sample: {index})"
        )

        analysis = analyze_sample_with_llm(client, sample, model=args.model)

        if analysis:
            llm_success += 1
            print(
                f"[DEBUG] LLM succeeded: file={file_name}, sample={index}"
            )
            if "Conclusion: A" in analysis:
                save_categorized_sample(sample, analysis, "A")
            elif "Conclusion: D" in analysis:
                save_categorized_sample(sample, analysis, "D")
        else:
            llm_failed += 1
            print(
                f"[DEBUG] LLM returned empty or error: file={file_name}, sample={index}"
            )

        header = f"Sample {index} (file: {file_name}) analysis"
        sep = "=" * 80
        block = [sep, header, sep]
        if analysis:
            block.append(analysis)
        else:
            block.append("[ERROR] Sample analysis failed")
        outputs_batch.append("\n".join(block))

        processed_count += 1

        # Write results after every 50 processed samples
        if processed_count % 50 == 0:
            batch_text = "\n\n".join(outputs_batch)
            mode = "w" if not has_written else "a"
            with out_path.open(mode, encoding="utf-8") as f:
                if mode == "a":
                    f.write("\n\n")
                f.write(batch_text)
            outputs_batch = []
            has_written = True

    # After loop, write any remaining results
    if outputs_batch:
        batch_text = "\n\n".join(outputs_batch)
        mode = "w" if not has_written else "a"
        with out_path.open(mode, encoding="utf-8") as f:
            if mode == "a":
                f.write("\n\n")
            f.write(batch_text)
        has_written = True

    print(
        f"[DEBUG] Summary: total={len(samples)}, "
        f"skipped(comparison_error has '比较过程出错')={skipped_due_to_error}, "
        f"processed={processed_count}, "
        f"LLM success={llm_success}, LLM empty/error={llm_failed}"
    )

    if has_written:
        print(f"[SUCCESS] Analysis saved to: {out_path}")
    else:
        print("[INFO] All samples had '比较过程出错'; no output file generated")


if __name__ == "__main__":
    main()
