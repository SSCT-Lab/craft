import re
import json
from pathlib import Path

def parse_report(report_path):
    files = []
    current_file = None
    has_field = False
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"file name:\s+([^\s]+\.json)", line)
            if m:
                if current_file and has_field:
                    files.append(current_file)
                current_file = m.group(1).strip()
                has_field = False
                continue
            if "Only the number of examples of TensorFlow error reports" in line:
                has_field = True
        if current_file and has_field:
            files.append(current_file)
    return files

def collect_samples(json_path):
    """
    Collect all TensorFlow-only error examples from the JSON test result file     (tensorflow_error is not null and torch_error is null）
    """
    results = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("results", []):
        exec_res = item.get("execution_result") or {}
        tensorflow_err = exec_res.get("tensorflow_error")
        torch_err = exec_res.get("torch_error")
        # Extract only samples with TensorFlow errors (PyTorch is normal)）
        if tensorflow_err is not None and torch_err is None:
            val = tensorflow_err
            entry = {
                "iteration": item.get("iteration"),
                "tensorflow_error": val,
            }
            if "torch_test_case" in item or "tensorflow_test_case" in item:
                entry["torch_test_case"] = item.get("torch_test_case")
                entry["tensorflow_test_case"] = item.get("tensorflow_test_case")
            elif "test_case" in item:
                entry["test_case"] = item.get("test_case")
            else:
                entry["raw_item"] = item
            results.append(entry)
    return results

def format_section(filename, samples):
    lines = []
    lines.append("=" * 80)
    lines.append(f"document: {filename}")
    lines.append("-" * 80)
    for idx, s in enumerate(samples, 1):
        lines.append(f"Sample {idx}:")
        lines.append(f"tensorflow_error: {s.get('tensorflow_error')}")
        if "torch_test_case" in s or "tensorflow_test_case" in s:
            if s.get("torch_test_case") is not None:
                lines.append("torch_test_case:")
                lines.append(json.dumps(s["torch_test_case"], ensure_ascii=False, indent=2))
            if s.get("tensorflow_test_case") is not None:
                lines.append("tensorflow_test_case:")
                lines.append(json.dumps(s["tensorflow_test_case"], ensure_ascii=False, indent=2))
        elif "test_case" in s:
            lines.append("test_case:")
            lines.append(json.dumps(s["test_case"], ensure_ascii=False, indent=2))
        else:
            lines.append("raw_item:")
            lines.append(json.dumps(s.get("raw_item"), ensure_ascii=False, indent=2))
        lines.append("")
    return "\n".join(lines)

def main():
    # Configuration path
    base_dir = Path(r"d:\graduate\DFrameworkTest\pt_tf_test")
    report_path = base_dir / "pt_tf_log_1" / "error_analysis_report_new.txt"  # Input: error analysis report
    log_dir = base_dir / "pt_tf_log_1"                                        # JSON Log file directory
    output_dir = base_dir / "analysis"                                        # Output directory
    output_dir.mkdir(parents=True, exist_ok=True)                             # Make sure the output directory exists
    output_path = output_dir / "new_tensorflow_error_samples_report.txt"      # Output file path
    target_files = parse_report(report_path)
    sections = []
    for fname in target_files:
        jpath = log_dir / fname
        if not jpath.exists():
            continue
        samples = collect_samples(jpath)
        if samples:
            sections.append(format_section(fname, samples))
    content = "\n".join(sections) if sections else "No examples of TensorFlow error reporting"
    with open(output_path, "w", encoding="utf-8") as wf:
        wf.write(content)
    print(f"✅ Report generated: {output_path}")
    print(f"📊 Processed in total {len(target_files)} files containing only TensorFlow errors")
    print(f"📝 symbiosis {len(sections)} Sample report for files")

if __name__ == "__main__":
    main()
