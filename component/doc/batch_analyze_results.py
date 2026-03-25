"""批量基于“官方文档 + LLM”对测试结果做判断，用于对外汇报.

主流程：
1. 读取 `data/results/migrate_comparison.jsonl`（或其它结果文件）
2. 选出「需要关注」的 case（例如 TF/PT 结果不一致、PT 失败等）
3. 对每个 case 调用 `doc_analyzer.analyze_test_error`：
   - 自动爬取 TF/PT 官方文档（带缓存）
   - 提取 TF/PT 测试代码
   - 调用大模型做详细分析
4. 输出：
   - JSONL: 每条 case 的详细分析
   - Markdown: 给老板看的总结报告

使用示例::

    # 对对比结果做分析（推荐）
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
    """从结果中挑出需要做文档+LLM分析的 case."""
    selected: List[Dict] = []

    for r in items:
        pt_res = (r.get("pt_result") or {})
        tf_res = (r.get("tf_result") or {})
        comp = (r.get("comparison") or {})

        pt_status = pt_res.get("status")
        tf_status = tf_res.get("status")
        match = comp.get("match")

        interesting = False

        # 情况 1: TF/PT 结果不一致（重点）
        if tf_status and pt_status and tf_status != pt_status:
            interesting = True

        # 情况 2: 只看不匹配的对比结果
        if only_mismatch and comp and match is False:
            interesting = True

        # 情况 3: 没有 TF 结果，但 PT 失败
        if not tf_status and pt_status not in (None, "pass"):
            interesting = True

        if not interesting:
            continue

        selected.append(r)
        if limit > 0 and len(selected) >= limit:
            break

    return selected


def build_error_message(rec: Dict) -> Dict[str, str]:
    """把一条记录压缩成简洁的错误信息结构（summary + TF / PT 输出分开展示）."""
    pt_res = rec.get("pt_result") or {}
    tf_res = rec.get("tf_result") or {}
    comp = rec.get("comparison") or {}

    pt_status = pt_res.get("status", "unknown")
    tf_status = tf_res.get("status", "unknown")

    # 为大模型提供一个总览行
    summary = f"TF status={tf_status}, PT status={pt_status}"

    # 分别截断 TF / PT 的输出，避免提示词过长
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

    # TF 输出单独一段
    if tf_snippet:
        parts.append("TensorFlow 执行输出 / 错误片段:\n" + tf_snippet)
    else:
        parts.append("TensorFlow 执行输出 / 错误片段: (无可用日志)")

    # PT 输出单独一段
    if pt_snippet:
        parts.append("PyTorch 执行输出 / 错误片段:\n" + pt_snippet)
    else:
        parts.append("PyTorch 执行输出 / 错误片段: (无可用日志)")

    return {
        "summary": "\n\n".join(parts),
        "tf_output": tf_snippet or "（无可用日志）",
        "pt_output": pt_snippet or "（无可用日志）",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="批量基于文档+LLM 分析迁移测试结果")
    ap.add_argument(
        "--results",
        default="data/results/migrate_comparison.jsonl",
        help="迁移结果文件（来自 migrate_compare.py）",
    )
    ap.add_argument(
        "--out-json",
        default="data/analysis/doc_llm_analysis.jsonl",
        help="每条 case 的详细分析输出(JSONL)",
    )
    ap.add_argument(
        "--out-md",
        default="reports/doc_llm_summary.md",
        help="给老板看的 Markdown 总结",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=20,
        help="最多分析多少条 case（默认 20 条，-1 表示全部）",
    )
    ap.add_argument(
        "--only-mismatch",
        action="store_true",
        help="只分析 TF/PT 结果不匹配的 case",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并发分析的线程数（默认 4，注意不要开太大以免触发 LLM 速率限制）",
    )
    args = ap.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"[ERROR] 结果文件不存在: {results_path}")
        return

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 读取结果文件: {results_path}")
    all_items = list(load_jsonl(results_path))
    print(f"[INFO] 总结果数: {len(all_items)}")

    cases = pick_interesting_cases(
        all_items,
        limit=args.limit,
        only_mismatch=args.only_mismatch,
    )
    print(f"[INFO] 选中 {len(cases)} 条需要分析的 case")

    analyses: List[Dict] = []
    
    # 并发分析：每个 case 单独调用 analyze_test_error（内部会使用本地文档缓存）
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

    # 汇总为 Markdown 报告
    lines: List[str] = []
    lines.append("# 文档 + LLM 迁移结果分析总结\n")
    lines.append(f"- 总结果数: **{len(all_items)}**")
    lines.append(f"- 分析条数: **{len(analyses)}**")
    lines.append(f"- 结果文件: `{results_path}`")
    lines.append("")

    for idx, a in enumerate(analyses, 1):
        title = f"{idx}. `{a['pt_file']}` / `{a.get('test_name') or ''}`"
        tf_status = a.get("tf_status") or "unknown"
        pt_status = a.get("pt_status") or "unknown"

        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- **TF 状态**: `{tf_status}`")
        lines.append(f"- **PT 状态**: `{pt_status}`")

        comp = a.get("comparison") or {}
        if comp:
            lines.append(f"- **对比结果(match)**: `{comp.get('match')}`")
            notes = comp.get("notes") or []
            if notes:
                lines.append(f"- **对比备注**: {', '.join(notes)}")

        lines.append("")
        lines.append("### LLM 分析结论")
        lines.append("")
        analysis_txt = a.get("analysis") or "（无结果，可能是 LLM 调用失败）"
        lines.append(analysis_txt)
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] 详细分析已写入: {out_json}")
    print(f"[DONE] 总结报告已写入: {out_md}")


if __name__ == "__main__":
    main()


