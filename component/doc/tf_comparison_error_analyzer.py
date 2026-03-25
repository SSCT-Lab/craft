# ./component/doc/comparison_error_analyzer.py
"""分析 comparison_error 报告的工具：结合官方文档和 LLM 判断误差原因"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

# 添加项目根目录到路径，保证可以导入 component 下的模块
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.doc.doc_crawler_factory import get_doc_content
from component.migration.migrate_generate_tests import get_qwen_client

DEFAULT_MODEL = "qwen-flash"
DEFAULT_KEY_PATH = "aliyun.key"


def parse_comparison_error_report(report_path: Path) -> List[Dict[str, Any]]:
    """解析 comparison_error_samples_report.txt，提取每个样例的信息"""
    samples: List[Dict[str, Any]] = []

    current_file: Optional[str] = None
    current_index: Optional[int] = None
    current_error: Optional[str] = None
    torch_lines: List[str] = []
    tf_lines: List[str] = []
    mode: Optional[str] = None  # "torch" / "tf" / None

    def flush_sample():
        """在切换样例或文件时，将当前样例落地"""
        nonlocal current_file, current_index, current_error, torch_lines, tf_lines
        if current_index is None:
            return
        try:
            torch_str = "".join(torch_lines).strip()
            tf_str = "".join(tf_lines).strip()
            if not torch_str or not tf_str:
                return
            torch_case = json.loads(torch_str)
            tf_case = json.loads(tf_str)
        except Exception as e:
            print(
                f"[WARN] 样例解析失败 (文件: {current_file}, 样例编号: {current_index}): {e}"
            )
            return

        samples.append(
            {
                "file": current_file,
                "index": current_index,
                "comparison_error": current_error or "",
                "torch_test_case": torch_case,
                "tensorflow_test_case": tf_case,
            }
        )

    with report_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            # 处理由 = 或 - 构成的分隔线，避免被拼进 JSON
            if stripped and (set(stripped) == {"="} or set(stripped) == {"-"}):
                flush_sample()
                current_index = None
                current_error = None
                torch_lines = []
                tf_lines = []
                mode = None
                continue

            if stripped.startswith("文件:"):
                # 新文件开始，刷新之前尚未落地的样例
                flush_sample()
                current_file = stripped.split("文件:", 1)[1].strip()
                current_index = None
                current_error = None
                torch_lines = []
                tf_lines = []
                mode = None
                continue

            if stripped.startswith("样例"):
                # 新样例开始，先落地上一个样例
                flush_sample()
                try:
                    # 形如 "样例 1:" -> 1
                    idx_part = stripped.split("样例", 1)[1]
                    idx_part = idx_part.strip(" :")
                    current_index = int(idx_part)
                except Exception:
                    current_index = None
                current_error = None
                torch_lines = []
                tf_lines = []
                mode = None
                continue

            if stripped.startswith("comparison_error:"):
                current_error = stripped.split("comparison_error:", 1)[1].strip()
                continue

            if stripped.startswith("torch_test_case:"):
                mode = "torch"
                continue

            if stripped.startswith("tensorflow_test_case:"):
                mode = "tf"
                continue

            # 收集 JSON 内容
            if mode == "torch":
                torch_lines.append(raw_line)
            elif mode == "tf":
                tf_lines.append(raw_line)

    # 文件结束后，尝试落地最后一个样例
    flush_sample()

    return samples


def build_sample_prompt(
    sample: Dict[str, Any],
    tf_docs: List[str],
    pt_docs: List[str],
) -> str:
    """为单个 comparison_error 样例构建提示词"""
    file_name = sample.get("file") or ""
    index = sample.get("index")
    comparison_error = sample.get("comparison_error", "")
    torch_case = sample.get("torch_test_case", {})
    tf_case = sample.get("tensorflow_test_case", {})

    torch_api = torch_case.get("api", "")
    tf_api = tf_case.get("api", "")

    tf_docs_text = "\n\n".join(tf_docs) if tf_docs else "未找到相关 TensorFlow 文档"
    pt_docs_text = "\n\n".join(pt_docs) if pt_docs else "未找到相关 PyTorch 文档"

    torch_case_json = json.dumps(torch_case, ensure_ascii=False, indent=2)
    tf_case_json = json.dumps(tf_case, ensure_ascii=False, indent=2)

    prompt = f"""你是一个熟悉 PyTorch 和 TensorFlow 的资深框架专家，现在要分析一个比较误差样例。

【样例基本信息】
- 来源文件: {file_name}
- 样例编号: {index}
- comparison_error 描述: {comparison_error}

【测试用例信息】
1. PyTorch 测试用例 (torch_test_case，JSON):
```json
{torch_case_json}
```

2. TensorFlow 测试用例 (tensorflow_test_case，JSON):
```json
{tf_case_json}
```

【候选 API 映射】
- PyTorch API: {torch_api}
- TensorFlow API: {tf_api}

【相关官方文档（TensorFlow）】
{tf_docs_text}

【相关官方文档（PyTorch）】
{pt_docs_text}

----------------------------------------
【分析任务】
请你结合以上信息，分析本样例中出现 comparison_error 的最可能原因，并重点区分以下几类：

1. 框架行为差异 (类别 A)
   - 例如: 广播规则不同、数值稳定性/精度策略不同、某些行为在两个框架中的具体实现不同等原因导致的”相同输入理论上输出相同结果但最终输出存在不一致“现象。

2. 测试用例 / 输入构造不一致问题 (类别 B)
   - 例如: 两侧输入 shape/dtype 不一致、某些参数默认值没有对齐、TensorFlow 侧绕了一层额外操作等。

3. API 匹配错误 (类别 C)
   - 例如: 实际上应该对应到 tf.nn.softmax，却错误映射成了 tf.math.softmax；
   - 或者 PyTorch 与 TensorFlow 的 API 语义明显不同，无法视为“同一个算子”。

4. 其他原因或信息不足 (类别 D)

----------------------------------------
【输出要求】
请你给出严格的技术分析，并按照下面结构回答：

1. 结论标签：
   - 请在一行中给出一个标签，格式形如：
     - 结论标签：A 框架行为差异
     - 结论标签：B 测试用例构造问题
     - 结论标签：C API 匹配错误
     - 结论标签：D 其他/信息不足

2. 详细原因分析：
   - 结合 comparison_error 描述、输入 shape/dtype、官方文档中的限制或行为说明，解释为什么会出现当前误差。
   - 如果你选择 C（API 匹配错误），请明确指出两侧 API 语义或参数上的关键差异。

3. 修复建议：
   - 如果是 A 类问题，判断是否值得提交为一个issue给官方+一句简单的原因，不用写出具体的issue内容。（一般认为发现一个之前未被发现的框架不一致或者漏洞才更值得提交为一个issue）
   - 如果是 C/D 类问题，无需给出修复建议。
   - 如果是 B 类问题，请给出如何修改映射或测试用例的建议，例如：
     - 调整为更合适的 Paddle API 或 PyTorch API；
     - 补齐/修改某个参数；
     - 调整输入 shape/dtype 以保持对齐。
"""

    return prompt


def analyze_sample_with_llm(
    client,
    sample: Dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> Optional[str]:
    torch_case = sample.get("torch_test_case", {})
    tf_case = sample.get("tensorflow_test_case", {})

    torch_api = torch_case.get("api", "")
    tf_api = tf_case.get("api", "")

    tf_docs: List[str] = []
    pt_docs: List[str] = []

    # 拉取 TF 文档
    if tf_api:
        try:
            doc_text = get_doc_content(tf_api, "tensorflow")
            if doc_text and "无法获取" not in doc_text:
                tf_docs.append(doc_text)
        except Exception as e:
            print(f"[WARN] 获取 TensorFlow 文档失败 {tf_api}: {e}")

    # 拉取 PT 文档
    if torch_api:
        try:
            doc_text = get_doc_content(torch_api, "pytorch")
            if doc_text and "无法获取" not in doc_text:
                pt_docs.append(doc_text)
        except Exception as e:
            print(f"[WARN] 获取 PyTorch 文档失败 {torch_api}: {e}")

    prompt = build_sample_prompt(sample, tf_docs=tf_docs, pt_docs=pt_docs)

    try:
        if hasattr(client, "chat"):
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048,
            )
            return resp.choices[0].message.content.strip()
        else:
            resp = client.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] LLM 调用失败: {e}")
        return None


def save_categorized_sample(
    sample: Dict[str, Any],
    analysis: str,
    category: str,
) -> None:
    base_dir = Path("pt_tf_test") / "analysis"
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
    tf_case = sample.get("tensorflow_test_case", {})

    torch_case_json = json.dumps(torch_case, ensure_ascii=False, indent=2)
    tf_case_json = json.dumps(tf_case, ensure_ascii=False, indent=2)

    safe_file_name = file_name.replace("/", "_").replace("\\", "_")
    out_path = out_dir / f"{safe_file_name}_sample{index}.txt"

    content_parts = [
        f"来源文件: {file_name}",
        f"样例编号: {index}",
        f"comparison_error 描述: {comparison_error}",
        "",
        torch_case_json,
        "",
        tf_case_json,
    ]
    out_path.write_text("\n".join(content_parts), encoding="utf-8")


def main():
    """命令行入口：批量分析 comparison_error 报告中的样例"""
    parser = argparse.ArgumentParser(description="分析 comparison_error 样例产生的原因")
    parser.add_argument(
        "--report",
        "-r",
        required=True,
        help="comparison_error_samples_report.txt 的路径",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多分析多少个样例（默认全部）",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help="LLM 模型名称（默认 qwen-flash）",
    )
    parser.add_argument(
        "--key-path",
        "-k",
        default=DEFAULT_KEY_PATH,
        help="API key 文件路径（默认 aliyun.key）",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="分析结果输出文件路径（不指定则打印到控制台）",
    )

    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"[ERROR] 报告文件不存在: {report_path}")
        return

    print(f"[INFO] 正在解析报告文件: {report_path}")
    samples = parse_comparison_error_report(report_path)
    if not samples:
        print("[ERROR] 未从报告中解析到任何样例")
        return

    if args.limit is not None:
        samples = samples[: args.limit]

    print(f"[INFO] 共需分析样例数: {len(samples)}")

    try:
        client = get_qwen_client(args.key_path)
    except Exception as e:
        print(f"[ERROR] 无法初始化 LLM 客户端: {e}")
        return

    outputs: List[str] = []

    for i, sample in enumerate(samples, start=1):
        file_name = sample.get("file") or ""
        index = sample.get("index")
        print(f"[INFO] 分析第 {i}/{len(samples)} 个样例 (文件: {file_name}, 样例编号: {index})")

        analysis = analyze_sample_with_llm(client, sample, model=args.model)

        if analysis:
            if "标签：A" in analysis:
                save_categorized_sample(sample, analysis, "A")
            elif "标签：D" in analysis:
                save_categorized_sample(sample, analysis, "D")

        header = f"样例 {index}（文件: {file_name}）分析结果"
        sep = "=" * 80
        block = [sep, header, sep]
        if analysis:
            block.append(analysis)
        else:
            block.append("[ERROR] 本样例分析失败")
        outputs.append("\n".join(block))

    result_text = "\n\n".join(outputs)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(result_text, encoding="utf-8")
        print(f"[SUCCESS] 分析结果已保存到: {out_path}")
    else:
        print(result_text)


if __name__ == "__main__":
    main()

