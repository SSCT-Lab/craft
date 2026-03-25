# ./component/migrate_report.py
# 生成迁移测试的对比报告
import json
import argparse
from pathlib import Path
from collections import Counter

def load_jsonl(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []

def generate_report(comparison_file, output_file):
    """生成 HTML 格式的对比报告"""
    results = load_jsonl(comparison_file)
    
    if not results:
        print(f"[ERROR] 无法读取对比结果文件: {comparison_file}")
        return
    
    # 统计信息
    total = len(results)
    
    # 安全获取 pt_result
    pt_pass = sum(1 for r in results if (r.get("pt_result") or {}).get("status") == "pass")
    pt_fail = sum(1 for r in results if (r.get("pt_result") or {}).get("status") == "fail")
    pt_error = sum(1 for r in results if (r.get("pt_result") or {}).get("status") in ["error", "timeout", "not_found"])
    
    # 安全获取 tf_result（可能为 None）
    tf_pass = sum(1 for r in results if (r.get("tf_result") or {}).get("status") == "pass")
    tf_fail = sum(1 for r in results if (r.get("tf_result") or {}).get("status") == "fail")
    tf_error = sum(1 for r in results if (r.get("tf_result") or {}).get("status") in ["error", "timeout", "not_found"])
    
    matches = sum(1 for r in results if (r.get("comparison") or {}).get("match", False))
    
    # 生成 HTML 报告
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>迁移测试对比报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .summary h2 {{ margin-top: 0; }}
        .summary table {{ width: 100%; border-collapse: collapse; }}
        .summary td, .summary th {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
        .summary th {{ background: #4CAF50; color: white; }}
        .test-item {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .test-item.pass {{ background: #d4edda; }}
        .test-item.fail {{ background: #f8d7da; }}
        .test-item.error {{ background: #fff3cd; }}
        .test-header {{ font-weight: bold; margin-bottom: 10px; }}
        .test-details {{ margin-left: 20px; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>迁移测试对比报告</h1>
    
    <div class="summary">
        <h2>统计摘要</h2>
        <table>
            <tr>
                <th>指标</th>
                <th>数量</th>
                <th>百分比</th>
            </tr>
            <tr>
                <td>总测试数</td>
                <td>{total}</td>
                <td>100%</td>
            </tr>
            <tr>
                <td>PyTorch 通过</td>
                <td>{pt_pass}</td>
                <td>{pt_pass/total*100:.1f}%</td>
            </tr>
            <tr>
                <td>PyTorch 失败</td>
                <td>{pt_fail}</td>
                <td>{pt_fail/total*100:.1f}%</td>
            </tr>
            <tr>
                <td>TensorFlow 通过</td>
                <td>{tf_pass}</td>
                <td>{tf_pass/total*100:.1f}%</td>
            </tr>
            <tr>
                <td>TensorFlow 失败</td>
                <td>{tf_fail}</td>
                <td>{tf_fail/total*100:.1f}%</td>
            </tr>
            <tr>
                <td>结果匹配</td>
                <td>{matches}</td>
                <td>{matches/total*100:.1f}%</td>
            </tr>
        </table>
    </div>
    
    <h2>详细结果</h2>
"""
    
    for r in results:
        pt_status = (r.get("pt_result") or {}).get("status", "unknown")
        tf_status = (r.get("tf_result") or {}).get("status", "unknown")
        match = (r.get("comparison") or {}).get("match", False)
        
        status_class = "error"
        if pt_status == "pass":
            status_class = "pass"
        elif pt_status == "fail":
            status_class = "fail"
        
        html += f"""
    <div class="test-item {status_class}">
        <div class="test-header">
            {Path(r.get("pt_file", "")).name} 
            <span style="color: {'green' if match else 'red'}">({'匹配' if match else '不匹配'})</span>
        </div>
        <div class="test-details">
            <p><strong>TensorFlow 文件:</strong> {r.get("tf_file", "N/A")}</p>
            <p><strong>测试函数:</strong> {r.get("test_name", "N/A")}</p>
            <p><strong>TensorFlow 状态:</strong> {tf_status}</p>
            <p><strong>PyTorch 状态:</strong> {pt_status}</p>
"""
        
        pt_result = r.get("pt_result") or {}
        if pt_result.get("stderr"):
            html += f"""
            <details>
                <summary>PyTorch 错误信息</summary>
                <pre>{pt_result.get("stderr", "")[:500]}</pre>
            </details>
"""
        
        tf_result = r.get("tf_result")
        if tf_result and tf_result.get("stderr"):
            html += f"""
            <details>
                <summary>TensorFlow 错误信息</summary>
                <pre>{tf_result.get("stderr", "")[:500]}</pre>
            </details>
"""
        
        html += """
        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"[DONE] 报告已生成: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="data/results/migrate_comparison.jsonl", help="对比结果文件")
    parser.add_argument("--output", default="reports/migration_comparison.html", help="输出 HTML 报告文件")
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    generate_report(args.comparison, args.output)

if __name__ == "__main__":
    main()

