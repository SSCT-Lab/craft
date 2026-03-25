import re
import json
from pathlib import Path

def parse_report(report_path):
    files = []
    current_file = None
    has_comparison_error_field = False
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"文件名:\s+([^\s]+\.json)", line)
            if m:
                if current_file and has_comparison_error_field:
                    files.append(current_file)
                current_file = m.group(1).strip()
                has_comparison_error_field = False
                continue
            if "comparison_error 非null值个数" in line:
                has_comparison_error_field = True
        if current_file and has_comparison_error_field:
            files.append(current_file)
    return files

def collect_samples(json_path):
    results = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("results", []):
        exec_res = item.get("execution_result") or {}
        comp_err = exec_res.get("comparison_error")
        if comp_err is not None:
            entry = {
                "iteration": item.get("iteration"),
                "comparison_error": comp_err,
            }
            if "torch_test_case" in item or "paddle_test_case" in item:
                entry["torch_test_case"] = item.get("torch_test_case")
                entry["paddle_test_case"] = item.get("paddle_test_case")
            elif "test_case" in item:
                entry["test_case"] = item.get("test_case")
            else:
                entry["raw_item"] = item
            results.append(entry)
    return results

def format_section(filename, samples):
    lines = []
    lines.append("=" * 80)
    lines.append(f"文件: {filename}")
    lines.append("-" * 80)
    for idx, s in enumerate(samples, 1):
        lines.append(f"样例 {idx}:")
        lines.append(f"comparison_error: {s.get('comparison_error')}")
        if "torch_test_case" in s or "paddle_test_case" in s:
            if s.get("torch_test_case") is not None:
                lines.append("torch_test_case:")
                lines.append(json.dumps(s["torch_test_case"], ensure_ascii=False, indent=2))
            if s.get("paddle_test_case") is not None:
                lines.append("paddle_test_case:")
                lines.append(json.dumps(s["paddle_test_case"], ensure_ascii=False, indent=2))
        elif "test_case" in s:
            lines.append("test_case:")
            lines.append(json.dumps(s["test_case"], ensure_ascii=False, indent=2))
        else:
            lines.append("raw_item:")
            lines.append(json.dumps(s.get("raw_item"), ensure_ascii=False, indent=2))
        lines.append("")
    return "\n".join(lines)

def main():
    base_dir = Path(r"d:\graduate\DFrameworkTest\pt_pd_test")
    report_path = base_dir / "error_analysis_report.txt"
    log_dir = base_dir / "pt_pd_log"
    output_dir = base_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comparison_error_samples_report.txt"

    target_files = parse_report(report_path)
    sections = []
    for fname in target_files:
        jpath = log_dir / fname
        if not jpath.exists():
            continue
        samples = collect_samples(jpath)
        if samples:
            sections.append(format_section(fname, samples))
    content = "\n".join(sections) if sections else "无comparison_error非null的样例"
    with open(output_path, "w", encoding="utf-8") as wf:
        wf.write(content)
    print(str(output_path))

if __name__ == "__main__":
    main()
