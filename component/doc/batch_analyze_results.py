"""Batch judge test results using official docs + LLM for reporting.

Main flow:
1. Read `data/results/migrate_comparison.jsonl` (or other result files)
2. Select cases of interest (e.g., TF/PT mismatch, PT failure)
3. For each case, call `doc_analyzer.analyze_test_error`:
    - Auto-crawl TF/PT official docs (with cache)
    - Extract TF/PT test code
    - Call LLM for detailed analysis
4. Outputs:
    - JSONL: detailed analysis per case
    - Markdown: summary report for stakeholders

Usage example::

     # Analyze comparison results (recommended)
     python3 component/doc/batch_analyze_results.py \
          --results data/results/migrate_comparison.jsonl \
          --out-json data/analysis/doc_llm_analysis.jsonl \
          --out-md reports/doc_llm_summary.md \
          --limit 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, List

from tqdm import tqdm

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_analyzer import analyze_test_error  # noqa: E402


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


def pick_interesting_cases(
    items: Iterable[Dict],
    limit: int = -1,
    only_mismatch: bool = False,
) -> List[Dict]:
    """Pick cases that need doc+LLM analysis."""
    selected: List[Dict] = []

    for r in items:
        pt_res = (r.get("pt_result") or {})
        tf_res = (r.get("tf_result") or {})
        comp = (r.get("comparison") or {})

        pt_status = pt_res.get("status")
        tf_status = tf_res.get("status")
        match = comp.get("match")

        interesting = False

        # Case 1: TF/PT status mismatch (primary)
        if tf_status and pt_status and tf_status != pt_status:
            interesting = True

        # Case 2: only mismatched comparisons
        if only_mismatch and comp and match is False:
            interesting = True

        # Case 3: no TF result, but PT failed
        if not tf_status and pt_status not in (None, "pass"):
            interesting = True

        if not interesting:
            continue

        selected.append(r)
        if limit > 0 and len(selected) >= limit:
            break

    return selected


def build_error_message(rec: Dict) -> Dict[str, str]:
    """Compress a record into a concise error summary (TF/PT outputs separated)."""
    pt_res = rec.get("pt_result") or {}
    tf_res = rec.get("tf_result") or {}
    comp = rec.get("comparison") or {}

    pt_status = pt_res.get("status", "unknown")
    tf_status = tf_res.get("status", "unknown")

    # Summary line for LLM
    summary = f"TF status={tf_status}, PT status={pt_status}"

    # Truncate TF/PT output to avoid overly long prompts
    def _short_snippet(res: Dict) -> str:
        text = (res.get("stderr") or "") + "\n" + (res.get("stdout") or "")
        text = text.strip()
        if not text:
            return ""
        max_len = 1200
        if len(text) > max_len:
            text = text[:max_len] + "\n...[truncated]..."
        return text

    tf_snippet = _short_snippet(tf_res)
    pt_snippet = _short_snippet(pt_res)

    notes = ""
    if comp:
        notes = f"comparison meta: {json.dumps(comp, ensure_ascii=False)}"

    parts = [summary]
    if notes:
        parts.append(notes)

    # TF output block
    if tf_snippet:
        parts.append("TensorFlow output / error snippet:\n" + tf_snippet)
    else:
        parts.append("TensorFlow output / error snippet: (no logs)")

    # PT output block
    if pt_snippet:
        parts.append("PyTorch output / error snippet:\n" + pt_snippet)
    else:
        parts.append("PyTorch output / error snippet: (no logs)")

    return {
        "summary": "\n\n".join(parts),
        "tf_output": tf_snippet or "(no logs)",
        "pt_output": pt_snippet or "(no logs)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch analyze migration test results with docs + LLM")
    ap.add_argument(
        "--results",
        default="data/results/migrate_comparison.jsonl",
        help="Migration result file (from migrate_compare.py)",
    )
    ap.add_argument(
        "--out-json",
        default="data/analysis/doc_llm_analysis.jsonl",
        help="Detailed analysis output per case (JSONL)",
    )
    ap.add_argument(
        "--out-md",
        default="reports/doc_llm_summary.md",
        help="Markdown summary for stakeholders",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max cases to analyze (default 20, -1 for all)",
    )
    ap.add_argument(
        "--only-mismatch",
        action="store_true",
        help="Only analyze TF/PT mismatched cases",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Worker threads for analysis (default 4, avoid LLM rate limits)",
    )
    args = ap.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        return

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading results file: {results_path}")
    all_items = list(load_jsonl(results_path))
    print(f"[INFO] Total results: {len(all_items)}")

    cases = pick_interesting_cases(
        all_items,
        limit=args.limit,
        only_mismatch=args.only_mismatch,
    )
    print(f"[INFO] Selected {len(cases)} cases to analyze")

    analyses: List[Dict] = []
    
    # Concurrent analysis: analyze each case (uses local doc cache)
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
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Analyzing with docs + LLM"):
                idx, analysis_rec = fut.result()
                analyses.append(analysis_rec)
                fj.write(json.dumps(analysis_rec, ensure_ascii=False) + "\n")
                fj.flush()

    # Summarize as Markdown report
    lines: List[str] = []
    lines.append("# Docs + LLM Migration Result Summary\n")
    lines.append(f"- Total results: **{len(all_items)}**")
    lines.append(f"- Analyzed cases: **{len(analyses)}**")
    lines.append(f"- Results file: `{results_path}`")
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


