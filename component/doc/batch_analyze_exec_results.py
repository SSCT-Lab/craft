"""基于执行结果 jsonl 做批量文档+LLM 分析（不依赖 migrate_comparison.jsonl）

输入：
- PT 执行结果：dev/results/pt_exec.jsonl
- TF 执行结果（可选）：
  - dev/results/tf_core_exec.jsonl
  - dev/results/tf_fuzz_exec.jsonl

思路：
- 先把 TF / PT 执行结果按「逻辑测试名」对齐，例如：
  - TF: dev/tf_core/tf_core_testBasic.py      -> key: testBasic
  - PT: dev/pt_migrated/pt_testBasic.py       -> key: testBasic
  - TF: dev/tf_fuzz/tf_core_testBasic_fuzz_0.py -> key: testBasic_fuzz_0
  - PT: dev/pt_migrated/pt_testBasic_fuzz_0.py  -> key: testBasic_fuzz_0
- 然后只遍历 PT 侧的执行结果，按 key 找到对应的 TF 结果（如果有）
- 按「有问题的 case」过滤后，调用 doc_analyzer.analyze_test_error 做分析
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
from component.doc.batch_analyze_results import build_error_message  # 直接复用现有逻辑


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
    """根据 PT 文件名生成逻辑 key，用于和 TF 对齐"""
    name = Path(pt_file).stem  # pt_testArgRenames_fuzz_0
    if name.startswith("pt_"):
        name = name[3:]
    return name


def make_tf_key(tf_file: str) -> str:
    """根据 TF 文件名生成逻辑 key，用于和 PT 对齐"""
    name = Path(tf_file).stem  # tf_core_testArgRenames_fuzz_0 / tf_core_testBasic
    if name.startswith("tf_core_"):
        name = name[len("tf_core_") :]
    return name


def load_tf_maps(
    tf_core_path: Path, tf_fuzz_path: Path
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """加载 TF core / fuzz 执行结果，按 key 索引"""
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
    """把 TF / PT 执行结果拼成统一结构，类似 migrate_comparison.jsonl"""
    tf_core_map, tf_fuzz_map = load_tf_maps(tf_core_path, tf_fuzz_path)
    records: List[Dict] = []

    for rec in load_jsonl(pt_exec_path):
        pt_file = rec.get("pt_file", "")
        pt_result = rec.get("pt_result") or {}
        key = make_pt_key(pt_file)

        # 优先用 fuzz TF 结果，其次 core
        tf_result = tf_fuzz_map.get(key) or tf_core_map.get(key) or {}

        combined = {
            "file": pt_file,  # 兼容 batch_analyze_results 的字段名
            "pt_file": pt_file,
            "test_name": key,
            "pt_result": pt_result,
            "tf_result": tf_result,
            # 暂时不做细粒度 comparison，交给 LLM 直接看 status + 日志
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
    """针对 exec 结果的简单筛选策略：
    - 重点关注 PT 失败 / 错误的 case
    - 如果 TF 有结果，且 TF/PT status 不一致，也算一类
    """
    selected: List[Dict] = []

    for r in items:
        pt_res = (r.get("pt_result") or {})
        tf_res = (r.get("tf_result") or {})

        pt_status = pt_res.get("status")
        tf_status = tf_res.get("status")

        interesting = False

        # 情况 1: PT 非 pass（fail / error / timeout 等）
        if pt_status not in (None, "pass"):
            interesting = True

        # 情况 2: TF 有结果且 TF/PT 状态不一致
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
        description="基于执行结果(jsonl) + 文档 + LLM 批量分析迁移质量"
    )
    ap.add_argument(
        "--pt-exec",
        default="dev/results/pt_exec.jsonl",
        help="PyTorch 执行结果文件",
    )
    ap.add_argument(
        "--tf-core-exec",
        default="dev/results/tf_core_exec.jsonl",
        help="TF core 执行结果文件（可选）",
    )
    ap.add_argument(
        "--tf-fuzz-exec",
        default="dev/results/tf_fuzz_exec.jsonl",
        help="TF fuzz 执行结果文件（可选）",
    )
    ap.add_argument(
        "--out-json",
        default="data/analysis/doc_llm_analysis_exec.jsonl",
        help="每条 case 的详细分析输出(JSONL)",
    )
    ap.add_argument(
        "--out-md",
        default="reports/doc_llm_summary_exec.md",
        help="给老板看的 Markdown 总结",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=50,
        help="最多分析多少条 case（默认 50 条，-1 表示全部）",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并发分析的线程数（默认 4，注意速率限制）",
    )
    args = ap.parse_args()

    pt_exec_path = Path(args.pt_exec)
    tf_core_exec_path = Path(args.tf_core_exec)
    tf_fuzz_exec_path = Path(args.tf_fuzz_exec)

    if not pt_exec_path.exists():
        print(f"[ERROR] PT 执行结果不存在: {pt_exec_path}")
        return

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 读取 PT 执行结果: {pt_exec_path}")
    combined_records = build_combined_records(
        pt_exec_path, tf_core_exec_path, tf_fuzz_exec_path
    )
    print(f"[INFO] 总记录数: {len(combined_records)}")

    # 如果 limit 为 -1，则不做过滤，直接分析全部 case
    if args.limit == -1:
        cases = list(combined_records)
        print(f"[INFO] 按要求分析全部 {len(cases)} 条 case（不做筛选）")
    else:
        cases = pick_interesting_cases_exec(combined_records, limit=args.limit)
        print(f"[INFO] 选中 {len(cases)} 条需要分析的 case（基于失败/不匹配筛选）")

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

    # 汇总为 Markdown 报告（复用 batch_analyze_results 的风格）
    lines: List[str] = []
    lines.append("# 文档 + LLM 执行结果分析总结\n")
    lines.append(f"- 总记录数: **{len(combined_records)}**")
    lines.append(f"- 分析条数: **{len(analyses)}**")
    lines.append(f"- PT 结果文件: `{pt_exec_path}`")
    if tf_core_exec_path.exists():
        lines.append(f"- TF core 结果文件: `{tf_core_exec_path}`")
    if tf_fuzz_exec_path.exists():
        lines.append(f"- TF fuzz 结果文件: `{tf_fuzz_exec_path}`")
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


