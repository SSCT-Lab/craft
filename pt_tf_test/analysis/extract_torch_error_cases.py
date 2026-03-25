import re
import json
from pathlib import Path

def parse_report(report_path):
    files = []
    current_file = None
    has_field = False
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"文件名:\s+([^\s]+\.json)", line)
            if m:
                if current_file and has_field:
                    files.append(current_file)
                current_file = m.group(1).strip()
                has_field = False
                continue
            if "仅 PyTorch 报错的样例数" in line:
                has_field = True
        if current_file and has_field:
            files.append(current_file)
    return files

def collect_samples(json_path):
    """
    从 JSON 测试结果文件中收集所有仅 PyTorch 报错的样例
    （torch_error 非 null 且 tensorflow_error 为 null）
    """
    results = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("results", []):
        exec_res = item.get("execution_result") or {}
        torch_err = exec_res.get("torch_error")
        tensorflow_err = exec_res.get("tensorflow_error")
        # 只提取仅 PyTorch 报错的样例（TensorFlow 正常）
        if torch_err is not None and tensorflow_err is None:
            val = torch_err
            entry = {
                "iteration": item.get("iteration"),
                "torch_error": val,
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
    lines.append(f"文件: {filename}")
    lines.append("-" * 80)
    for idx, s in enumerate(samples, 1):
        lines.append(f"样例 {idx}:")
        lines.append(f"torch_error: {s.get('torch_error')}")
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
    # 配置路径
    base_dir = Path(r"d:\graduate\DFrameworkTest\pt_tf_test")
    report_path = base_dir / "pt_tf_log_1" / "error_analysis_report_new.txt"  # 输入：错误分析报告
    log_dir = base_dir / "pt_tf_log_1"                                        # JSON 日志文件目录
    output_dir = base_dir / "analysis"                                        # 输出目录
    output_dir.mkdir(parents=True, exist_ok=True)                             # 确保输出目录存在
    output_path = output_dir / "new_torch_error_samples_report.txt"           # 输出文件路径
    target_files = parse_report(report_path)
    sections = []
    for fname in target_files:
        jpath = log_dir / fname
        if not jpath.exists():
            continue
        samples = collect_samples(jpath)
        if samples:
            sections.append(format_section(fname, samples))
    content = "\n".join(sections) if sections else "无仅 PyTorch 报错的样例"
    with open(output_path, "w", encoding="utf-8") as wf:
        wf.write(content)
    print(f"✅ 报告已生成: {output_path}")
    print(f"📊 共处理 {len(target_files)} 个包含仅 PyTorch 错误的文件")
    print(f"📝 共生成 {len(sections)} 个文件的样例报告")

if __name__ == "__main__":
    main()
