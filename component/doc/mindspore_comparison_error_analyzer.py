"""分析 PyTorch 与 MindSpore comparison_error 报告的工具：结合官方文档和 LLM 判断误差原因"""

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
    ms_lines: List[str] = []
    mode: Optional[str] = None

    def flush_sample():
        """在切换样例或文件时，将当前样例落地"""
        nonlocal current_file, current_index, current_error, torch_lines, ms_lines
        if current_index is None:
            return
        try:
            # 将累积的 JSON 行合并成字符串
            torch_str = "".join(torch_lines).strip()
            ms_str = "".join(ms_lines).strip()
            if not torch_str or not ms_str:
                return
            torch_case = json.loads(torch_str)
            ms_case = json.loads(ms_str)
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
                "mindspore_test_case": ms_case,
            }
        )

    with report_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            # 处理由 = 构成的分隔线，避免被拼进 JSON
            if stripped and set(stripped) == {"="}:
                flush_sample()
                current_index = None
                current_error = None
                torch_lines = []
                ms_lines = []
                mode = None
                continue

            # 每个 JSON 文件块的开头
            if stripped.startswith("文件:"):
                flush_sample()
                current_file = stripped.split("文件:", 1)[1].strip()
                current_index = None
                current_error = None
                torch_lines = []
                ms_lines = []
                mode = None
                continue

            # 每个样例的开头，如“样例 1:”
            if stripped.startswith("样例"):
                flush_sample()
                try:
                    idx_part = stripped.split("样例", 1)[1]
                    idx_part = idx_part.strip(" :")
                    current_index = int(idx_part)
                except Exception:
                    current_index = None
                current_error = None
                torch_lines = []
                ms_lines = []
                mode = None
                continue

            if stripped.startswith("comparison_error:"):
                current_error = stripped.split("comparison_error:", 1)[1].strip()
                continue

            # 标记接下来是哪个框架的 JSON 内容
            if stripped.startswith("torch_test_case:"):
                mode = "torch"
                continue

            if stripped.startswith("mindspore_test_case:"):
                mode = "mindspore"
                continue

            # 根据当前模式，收集对应的 JSON 行
            if mode == "torch":
                torch_lines.append(raw_line)
            elif mode == "mindspore":
                ms_lines.append(raw_line)

    flush_sample()

    return samples


def build_sample_prompt(
    sample: Dict[str, Any],
    ms_docs: List[str],
    pt_docs: List[str],
) -> str:
    """为单个 comparison_error 样例构建提示词"""
    file_name = sample.get("file") or ""
    index = sample.get("index")
    comparison_error = sample.get("comparison_error", "")
    torch_case = sample.get("torch_test_case", {})
    ms_case = sample.get("mindspore_test_case", {})

    torch_api = torch_case.get("api", "")
    ms_api = ms_case.get("api", "")

    ms_docs_text = "\n\n".join(ms_docs) if ms_docs else "未找到相关 MindSpore 文档"
    pt_docs_text = "\n\n".join(pt_docs) if pt_docs else "未找到相关 PyTorch 文档"

    torch_case_json = json.dumps(torch_case, ensure_ascii=False, indent=2)
    ms_case_json = json.dumps(ms_case, ensure_ascii=False, indent=2)

    prompt = f"""你是一个熟悉 PyTorch 和 MindSpore 的资深框架专家，现在要分析一个比较误差样例。

【样例基本信息】
- 来源文件: {file_name}
- 样例编号: {index}
- comparison_error 描述: {comparison_error}

【测试用例信息】
1. PyTorch 测试用例 (torch_test_case，JSON):
```json
{torch_case_json}
```

2. MindSpore 测试用例 (mindspore_test_case，JSON):
```json
{ms_case_json}
```

【候选 API 映射】
- PyTorch API: {torch_api}
- MindSpore API: {ms_api}

【相关官方文档（MindSpore）】
{ms_docs_text}

【相关官方文档（PyTorch）】
{pt_docs_text}

----------------------------------------
【分析任务】
请你结合以上信息，分析本样例中出现 comparison_error 的最可能原因，并重点区分以下几类：

1. 框架行为差异 (类别 A)
   - 例如: 广播规则不同、数值稳定性/精度策略不同、某些行为在两个框架中的具体实现不同等原因导致的”相同输入理论上输出相同结果但最终输出存在不一致“现象。

2. 测试用例 / 输入构造不一致问题 (类别 B)
   - 例如: 两个框架输入的 shape/dtype 不一致、某些参数默认值没有对齐、一侧绕了一层额外操作等。

3. API 匹配错误 (类别 C)
   - 例如: 实际上应该对应到 mindspore.ops.softmax，却错误映射成了 mindspore.nn.Softmax；
   - 或者 PyTorch 与 MindSpore 的 API 语义明显不同，无法视为“同一个算子”。

4. 其他原因或信息不足 (类别 D)

----------------------------------------
【输出要求】
请你给出严格的技术分析，并按照下面结构回答：

1. 结论标签：
   - 请在一行中给出一个标签，格式形如：
     - 结论标签：A 框架行为差异
     - 结论标签：B 测试用例问题
     - 结论标签：C API 匹配错误
     - 结论标签：D 其他/信息不足

2. 详细原因分析：
   - 结合 comparison_error 描述、输入 shape/dtype、官方文档中的限制或行为说明，解释为什么会出现当前误差。
   - 如果你选择 C（API 匹配错误），请明确指出两侧 API 语义或参数上的关键差异。

3. 修复建议：
   - 如果是 A 类问题，判断是否值得提交为一个issue给官方+一句简单的原因，不用写出具体的issue内容。
   - 如果是 C/D 类问题，无需给出修复建议。
   - 如果是 B 类问题，请给出如何修改映射或测试用例的建议，例如：
     - 调整为更合适的 MindSpore API 或 PyTorch API；
     - 补齐/修改某个参数；
     - 调整输入 shape/dtype 以保持对齐。
"""

    return prompt


def analyze_sample_with_llm(
    client,
    sample: Dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> Optional[str]:
    """调用大模型，对单个样例进行分析"""
    torch_case = sample.get("torch_test_case", {})
    ms_case = sample.get("mindspore_test_case", {})

    # 从样例中提取 API 名称
    torch_api = torch_case.get("api", "")
    ms_api = ms_case.get("api", "")

    ms_docs: List[str] = []
    pt_docs: List[str] = []

    # 拉取 MindSpore 官方文档
    if ms_api:
        try:
            doc_text = get_doc_content(ms_api, "mindspore")
            if doc_text and "无法获取" not in doc_text:
                ms_docs.append(doc_text)
        except Exception as e:
            print(f"[WARN] 获取 MindSpore 文档失败 {ms_api}: {e}")

    # 拉取 PyTorch 官方文档
    if torch_api:
        try:
            doc_text = get_doc_content(torch_api, "pytorch")
            if doc_text and "无法获取" not in doc_text:
                pt_docs.append(doc_text)
        except Exception as e:
            print(f"[WARN] 获取 PyTorch 文档失败 {torch_api}: {e}")

    prompt = build_sample_prompt(sample, ms_docs=ms_docs, pt_docs=pt_docs)

    try:
        if hasattr(client, "chat"):
            # 适配新版 SDK 的调用方式
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048,
            )
            return resp.choices[0].message.content.strip()
        else:
            # 兼容旧版 SDK 接口
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
    """根据 LLM 分类结果，将样例信息保存到对应目录"""
    base_dir = Path("pt_ms_test") / "analysis"
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
    ms_case = sample.get("mindspore_test_case", {})

    torch_case_json = json.dumps(torch_case, ensure_ascii=False, indent=2)
    ms_case_json = json.dumps(ms_case, ensure_ascii=False, indent=2)

    safe_file_name = file_name.replace("/", "_").replace("\\", "_")
    out_path = out_dir / f"{safe_file_name}_sample{index}.txt"

    content_parts = [
        f"来源文件: {file_name}",
        f"样例编号: {index}",
        f"comparison_error 描述: {comparison_error}",
        "",
        torch_case_json,
        "",
        ms_case_json,
    ]
    out_path.write_text("\n".join(content_parts), encoding="utf-8")


def main():
    """命令行入口：批量分析 PyTorch 与 MindSpore comparison_error 样例"""
    parser = argparse.ArgumentParser(
        description="分析 PyTorch 与 MindSpore comparison_error 样例产生的原因"
    )
    parser.add_argument(
        "--report",
        "-r",
        required=True,
        help="pt_ms_test/analysis/comparison_error_samples_report.txt 的路径",
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
        help="分析结果输出文件路径（不指定则自动在同目录生成）",
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

    # 确定输出文件路径
    if args.output:
        out_path = Path(args.output)
    else:
        stem = report_path.stem
        if stem == "comparison_error_samples_report":
            out_name = "llm_comparison_error_analysis.txt"
        else:
            out_name = f"{stem}_llm_comparison_error_analysis.txt"
        out_path = report_path.with_name(out_name)

    outputs_batch: List[str] = []
    processed_count = 0
    has_written = False
    skipped_due_to_error = 0
    llm_success = 0
    llm_failed = 0

    for i, sample in enumerate(samples, start=1):
        comparison_error = sample.get("comparison_error", "")
        if "比较过程出错" in comparison_error:
            skipped_due_to_error += 1
            # print(
            #     f"[DEBUG] 跳过样例 (comparison_error 含“比较过程出错”): "
            #     f"文件={sample.get('file')}, 样例编号={sample.get('index')}"
            # )
            continue

        file_name = sample.get("file") or ""
        index = sample.get("index")
        print(
            f"[INFO] 分析第 {i}/{len(samples)} 个样例 (文件: {file_name}, 样例编号: {index})"
        )

        analysis = analyze_sample_with_llm(client, sample, model=args.model)

        if analysis:
            llm_success += 1
            print(
                f"[DEBUG] LLM 返回成功: 文件={file_name}, 样例编号={index}"
            )
            if "标签：A" in analysis:
                save_categorized_sample(sample, analysis, "A")
            elif "标签：D" in analysis:
                save_categorized_sample(sample, analysis, "D")
        else:
            llm_failed += 1
            print(
                f"[DEBUG] LLM 返回为空或出错: 文件={file_name}, 样例编号={index}"
            )

        header = f"样例 {index}（文件: {file_name}）分析结果"
        sep = "=" * 80
        block = [sep, header, sep]
        if analysis:
            block.append(analysis)
        else:
            block.append("[ERROR] 本样例分析失败")
        outputs_batch.append("\n".join(block))

        processed_count += 1

        # 每处理完 50 个“实际分析”的样例，就将结果写入文件
        if processed_count % 50 == 0:
            batch_text = "\n\n".join(outputs_batch)
            mode = "w" if not has_written else "a"
            with out_path.open(mode, encoding="utf-8") as f:
                if mode == "a":
                    f.write("\n\n")
                f.write(batch_text)
            outputs_batch = []
            has_written = True

    # 循环结束后，如果还有未写入的结果，一次性写入
    if outputs_batch:
        batch_text = "\n\n".join(outputs_batch)
        mode = "w" if not has_written else "a"
        with out_path.open(mode, encoding="utf-8") as f:
            if mode == "a":
                f.write("\n\n")
            f.write(batch_text)
        has_written = True

    print(
        f"[DEBUG] 汇总: 总样例数={len(samples)}, "
        f"comparison_error 含“比较过程出错”跳过={skipped_due_to_error}, "
        f"实际分析样例数={processed_count}, "
        f"LLM 成功返回={llm_success}, LLM 返回为空或出错={llm_failed}"
    )

    if has_written:
        print(f"[SUCCESS] 分析结果已保存到: {out_path}")
    else:
        print("[INFO] 所有样例均为“比较过程出错”，未生成分析结果文件")


if __name__ == "__main__":
    main()
