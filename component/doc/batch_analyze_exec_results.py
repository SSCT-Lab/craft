"""Batch doc+LLM analysis from execution result JSONL (no migrate_comparison.jsonl).

Inputs:
- PT execution results: dev/results/pt_exec.jsonl
- TF execution results (optional):
    - dev/results/tf_core_exec.jsonl
    - dev/results/tf_fuzz_exec.jsonl

Approach:
- Align TF/PT execution results by logical test name, e.g.:
    - TF: dev/tf_core/tf_core_testBasic.py      -> key: testBasic
    - PT: dev/pt_migrated/pt_testBasic.py       -> key: testBasic
    - TF: dev/tf_fuzz/tf_core_testBasic_fuzz_0.py -> key: testBasic_fuzz_0
    - PT: dev/pt_migrated/pt_testBasic_fuzz_0.py  -> key: testBasic_fuzz_0
- Iterate PT results and find corresponding TF results by key (if any)
- Filter "problematic" cases and call doc_analyzer.analyze_test_error
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_analyzer import analyze_test_error  # noqa: E402
from component.doc.batch_analyze_results import build_error_message  # reuse existing logic


def load_jsonl(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def make_pt_key(pt_file: str) -> str:
    """Generate logical key from PT filename for TF alignment."""
    name = Path(pt_file).stem  # pt_testArgRenames_fuzz_0
    if name.startswith("pt_"):
        name = name[3:]
    return name


def make_tf_key(tf_file: str) -> str:
    """Generate logical key from TF filename for PT alignment."""
    name = Path(tf_file).stem  # tf_core_testArgRenames_fuzz_0 / tf_core_testBasic
    if name.startswith("tf_core_"):
        name = name[len("tf_core_") :]
    return name


def load_tf_maps(
    tf_core_path: Path, tf_fuzz_path: Path
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Load TF core/fuzz execution results, indexed by key."""
    tf_core_map: Dict[str, Dict] = {}
    tf_fuzz_map: Dict[str, Dict] = {}

    if tf_core_path.exists():
        for rec in load_jsonl(tf_core_path):
            tf_file = rec.get("tf_file")
            if not tf_file:
                continue
            key = make_tf_key(tf_file)
            tf_core_map[key] = rec.get("tf_result") or {}

    if tf_fuzz_path.exists():
        for rec in load_jsonl(tf_fuzz_path):
            tf_file = rec.get("tf_file")
            if not tf_file:
                continue
            key = make_tf_key(tf_file)
            tf_fuzz_map[key] = rec.get("tf_result") or {}

    return tf_core_map, tf_fuzz_map


def build_combined_records(
    pt_exec_path: Path,
    tf_core_path: Path,
    tf_fuzz_path: Path,
) -> List[Dict]:
    """Combine TF/PT execution results into a unified structure."""
    tf_core_map, tf_fuzz_map = load_tf_maps(tf_core_path, tf_fuzz_path)
    records: List[Dict] = []

    for rec in load_jsonl(pt_exec_path):
        pt_file = rec.get("pt_file", "")
        pt_result = rec.get("pt_result") or {}
        key = make_pt_key(pt_file)

        # Prefer fuzz TF results, then core
        tf_result = tf_fuzz_map.get(key) or tf_core_map.get(key) or {}

        combined = {
            "file": pt_file,  # keep field name compatible with batch_analyze_results
            "pt_file": pt_file,
            "test_name": key,
            "pt_result": pt_result,
            "tf_result": tf_result,
            # Skip fine-grained comparison; LLM will inspect status + logs
            "comparison": {
                "tf_status": tf_result.get("status"),
                "pt_status": pt_result.get("status"),
                "match": (
                    tf_result.get("status") == pt_result.get("status")
                    if tf_result
                    else None
                ),
                "notes": [],
            },
        }
        records.append(combined)

    return records


def pick_interesting_cases_exec(
    items: Iterable[Dict],
    limit: int = -1,
) -> List[Dict]:
    """Simple filtering for exec results:
    - Focus on PT failures/errors
    - If TF has results and TF/PT status mismatch, include
    """
    selected: List[Dict] = []

    for r in items:
        pt_res = (r.get("pt_result") or {})
        tf_res = (r.get("tf_result") or {})

        pt_status = pt_res.get("status")
        tf_status = tf_res.get("status")

        interesting = False

        # Case 1: PT not pass (fail / error / timeout, etc.)
        if pt_status not in (None, "pass"):
            interesting = True

        # Case 2: TF has results and TF/PT status mismatch
        if tf_status and pt_status and tf_status != pt_status:
            interesting = True

        if not interesting:
            continue

        selected.append(r)
        if limit > 0 and len(selected) >= limit:
            break

    return selected


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch analyze migration quality using exec results (jsonl) + docs + LLM"
    )
    ap.add_argument(
        "--pt-exec",
        default="dev/results/pt_exec.jsonl",
        help="PyTorch execution result file",
    )
    ap.add_argument(
        "--tf-core-exec",
        default="dev/results/tf_core_exec.jsonl",
        help="TF core execution result file (optional)",
    )
    ap.add_argument(
        "--tf-fuzz-exec",
        default="dev/results/tf_fuzz_exec.jsonl",
        help="TF fuzz execution result file (optional)",
    )
    ap.add_argument(
        "--out-json",
        default="data/analysis/doc_llm_analysis_exec.jsonl",
        help="Detailed analysis output per case (JSONL)",
    )
    ap.add_argument(
        "--out-md",
        default="reports/doc_llm_summary_exec.md",
        help="Markdown summary for stakeholders",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max cases to analyze (default 50, -1 for all)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Worker threads for analysis (default 4, mind rate limits)",
    )
    args = ap.parse_args()

    pt_exec_path = Path(args.pt_exec)
    tf_core_exec_path = Path(args.tf_core_exec)
    tf_fuzz_exec_path = Path(args.tf_fuzz_exec)

    if not pt_exec_path.exists():
        print(f"[ERROR] PT execution results not found: {pt_exec_path}")
        return

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading PT execution results: {pt_exec_path}")
    combined_records = build_combined_records(
        pt_exec_path, tf_core_exec_path, tf_fuzz_exec_path
    )
    print(f"[INFO] Total records: {len(combined_records)}")

    # If limit is -1, analyze all cases
    if args.limit == -1:
        cases = list(combined_records)
        print(f"[INFO] Analyzing all {len(cases)} cases (no filtering)")
    else:
        cases = pick_interesting_cases_exec(combined_records, limit=args.limit)
        print(f"[INFO] Selected {len(cases)} cases to analyze (failure/mismatch filter)")

    analyses: List[Dict] = []

    def process_case(idx_and_rec):
        idx, rec = idx_and_rec
        pt_file: str = rec.get("pt_file") or rec.get("file") or ""
        test_name: str = rec.get("test_name") or ""

        err = build_error_message(rec)
        error_msg = err["summary"]

        analysis_text: Optional[str] = analyze_test_error(
            error_message=error_msg,
            test_file=pt_file,
            tf_apis=None,
            pt_apis=None,
            tf_output=err.get("tf_output"),
            pt_output=err.get("pt_output"),
            context=None,
        )

        analysis_rec = {
            "pt_file": pt_file,
            "test_name": test_name,
            "tf_status": (rec.get("tf_result") or {}).get("status"),
            "pt_status": (rec.get("pt_result") or {}).get("status"),
            "comparison": rec.get("comparison"),
            "analysis": analysis_text or "",
        }
        return idx, analysis_rec

    with out_json.open("w", encoding="utf-8") as fj:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_case, (idx, rec)): idx
                for idx, rec in enumerate(cases)
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Analyzing exec results with docs + LLM",
            ):
                idx, analysis_rec = fut.result()
                analyses.append(analysis_rec)
                fj.write(json.dumps(analysis_rec, ensure_ascii=False) + "\n")
                fj.flush()

    # Summarize as Markdown report (same style as batch_analyze_results)
    lines: List[str] = []
    lines.append("# Docs + LLM Exec Result Summary\n")
    lines.append(f"- Total records: **{len(combined_records)}**")
    lines.append(f"- Analyzed cases: **{len(analyses)}**")
    lines.append(f"- PT result file: `{pt_exec_path}`")
    if tf_core_exec_path.exists():
        lines.append(f"- TF core result file: `{tf_core_exec_path}`")
    if tf_fuzz_exec_path.exists():
        lines.append(f"- TF fuzz result file: `{tf_fuzz_exec_path}`")
    lines.append("")

    for idx, a in enumerate(analyses, 1):
        title = f"{idx}. `{a['pt_file']}` / `{a.get('test_name') or ''}`"
        tf_status = a.get("tf_status") or "unknown"
        pt_status = a.get("pt_status") or "unknown"

        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- **TF status**: `{tf_status}`")
        lines.append(f"- **PT status**: `{pt_status}`")

        comp = a.get("comparison") or {}
        if comp:
            lines.append(f"- **Comparison (match)**: `{comp.get('match')}`")
            notes = comp.get("notes") or []
            if notes:
                lines.append(f"- **Comparison notes**: {', '.join(notes)}")

        lines.append("")
        lines.append("### LLM Analysis")
        lines.append("")
        analysis_txt = a.get("analysis") or "(No result; LLM call may have failed)"
        lines.append(analysis_txt)
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] Detailed analysis written to: {out_json}")
    print(f"[DONE] Summary report written to: {out_md}")


if __name__ == "__main__":
    main()


