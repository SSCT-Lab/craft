import re
import json
from pathlib import Path

def parse_report(report_path):
    files = []
    current_file = None
    has_field = False
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"File name:\s+([^\s]+\.json)", line)
            if m:
                if current_file and has_field:
                    files.append(current_file)
                current_file = m.group(1).strip()
                has_field = False
                continue
            if "torch_error non-null count" in line:
                has_field = True
        if current_file and has_field:
            files.append(current_file)
    return files

def collect_samples(json_path):
    results = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("results", []):
        exec_res = item.get("execution_result") or {}
        val = exec_res.get("torch_error")
        if val is not None:
            entry = {
                "iteration": item.get("iteration"),
                "torch_error": val,
            }
            if "torch_test_case" in item or "mindspore_test_case" in item:
                entry["torch_test_case"] = item.get("torch_test_case")
                entry["mindspore_test_case"] = item.get("mindspore_test_case")
            elif "test_case" in item:
                entry["test_case"] = item.get("test_case")
            else:
                entry["raw_item"] = item
            results.append(entry)
    return results

def format_section(filename, samples):
    lines = []
    lines.append("=" * 80)
    lines.append(f"File: {filename}")
    lines.append("-" * 80)
    for idx, s in enumerate(samples, 1):
        lines.append(f"Sample {idx}:")
        lines.append(f"torch_error: {s.get('torch_error')}")
        if "torch_test_case" in s or "mindspore_test_case" in s:
            if s.get("torch_test_case") is not None:
                lines.append("torch_test_case:")
                lines.append(json.dumps(s["torch_test_case"], ensure_ascii=False, indent=2))
            if s.get("mindspore_test_case") is not None:
                lines.append("mindspore_test_case:")
                lines.append(json.dumps(s["mindspore_test_case"], ensure_ascii=False, indent=2))
        elif "test_case" in s:
            lines.append("test_case:")
            lines.append(json.dumps(s["test_case"], ensure_ascii=False, indent=2))
        else:
            lines.append("raw_item:")
            lines.append(json.dumps(s.get("raw_item"), ensure_ascii=False, indent=2))
        lines.append("")
    return "\n".join(lines)

def main():
    base_dir = Path(r"d:\graduate\DFrameworkTest\pt_ms_test")
    report_path = base_dir / "error_analysis_report.txt"
    log_dir = base_dir / "pt_ms_log"
    output_dir = base_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "torch_error_samples_report.txt"
    target_files = parse_report(report_path)
    sections = []
    for fname in target_files:
        jpath = log_dir / fname
        if not jpath.exists():
            continue
        samples = collect_samples(jpath)
        if samples:
            sections.append(format_section(fname, samples))
    content = "\n".join(sections) if sections else "No samples with non-null torch_error"
    with open(output_path, "w", encoding="utf-8") as wf:
        wf.write(content)
    print(str(output_path))

if __name__ == "__main__":
    main()
