#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用千问 qwen3-max 对错误样例进行“是否值得提交 issue”筛选。

输入文件（可通过 --input 指定任意一个）通常为：
- tf_pt_test/analysis/pytorch_error_only_samples_20260221_211516.json
- tf_pt_test/analysis/both_error_samples_20260221_211516.json

输出文件：
- 默认输出到输入文件同目录，文件名格式为 issue_candidates_xxx.json（前缀）

筛选目标：
- 让 LLM 判断每个样例是否更可能是“框架本身问题”而非“迁移/参数不一致导致”。
- 保存所有样例判断：is_issue=true 保留完整样例，is_issue=false 保留最小可区分信息。
- 保存时附带 LLM 的判断与简短中文理由。

用法：
    conda activate tf_env
    python tf_pt_test/analysisCode/filter_issue_candidates_with_llm.py `
        --input tf_pt_test/analysis/pytorch_error_only_samples_20260221_211516.json

小规模测试：
    conda activate tf_env
    python tf_pt_test/analysisCode/filter_issue_candidates_with_llm.py \
        --input tf_pt_test/analysis/pytorch_error_only_samples_20260221_211516.json \
        --max-samples 8 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "qwen3-max"
DEFAULT_KEY_PATH = "aliyun.key"
DEFAULT_WORKERS = 6


def load_api_key(key_path: str = DEFAULT_KEY_PATH) -> str:
    """加载阿里云 API 密钥。"""
    candidate = Path(key_path)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / key_path

    if candidate.exists():
        api_key = candidate.read_text(encoding="utf-8").strip()
        if api_key:
            return api_key

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if api_key:
        return api_key

    print("❌ 未找到 API 密钥，请检查 aliyun.key 或 DASHSCOPE_API_KEY")
    return ""


def build_default_output_path(input_path: Path) -> Path:
    """构建默认输出路径。"""
    return input_path.with_name(f"issue_candidates_{input_path.stem}.json")


def simplify_sample_for_prompt(sample: Dict[str, Any]) -> Dict[str, Any]:
    """精简样例信息，降低 token 消耗，同时保留判断所需关键字段。"""
    execution_result = sample.get("execution_result", {}) if isinstance(sample, dict) else {}
    tf_case = sample.get("tf_test_case", {}) if isinstance(sample, dict) else {}
    pt_case = sample.get("pytorch_test_case", {}) if isinstance(sample, dict) else {}
    llm_operation = sample.get("llm_operation", {}) if isinstance(sample, dict) else {}

    return {
        "iteration": sample.get("iteration") if isinstance(sample, dict) else None,
        "case_number": sample.get("case_number") if isinstance(sample, dict) else None,
        "is_llm_generated": sample.get("is_llm_generated") if isinstance(sample, dict) else None,
        "tf_test_case": tf_case,
        "pytorch_test_case": pt_case,
        "execution_result": {
            "tf_api": execution_result.get("tf_api"),
            "pytorch_api": execution_result.get("pytorch_api"),
            "tf_success": execution_result.get("tf_success"),
            "pytorch_success": execution_result.get("pytorch_success"),
            "tf_error": execution_result.get("tf_error"),
            "pytorch_error": execution_result.get("pytorch_error"),
            "comparison_error": execution_result.get("comparison_error"),
            "tf_shape": execution_result.get("tf_shape"),
            "pytorch_shape": execution_result.get("pytorch_shape"),
            "tf_dtype": execution_result.get("tf_dtype"),
            "pytorch_dtype": execution_result.get("pytorch_dtype"),
            "status": execution_result.get("status"),
        },
        "llm_operation": {
            "operation": llm_operation.get("operation"),
            "reason": llm_operation.get("reason"),
        },
    }


def build_minimal_discriminative_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """构建最小可区分样例信息，用于非 issue 样例。"""
    execution_result = sample.get("execution_result", {}) if isinstance(sample, dict) else {}
    return {
        "iteration": sample.get("iteration") if isinstance(sample, dict) else None,
        "case_number": sample.get("case_number") if isinstance(sample, dict) else None,
        "is_llm_generated": sample.get("is_llm_generated") if isinstance(sample, dict) else None,
        "execution_result": {
            "tf_api": execution_result.get("tf_api"),
            "pytorch_api": execution_result.get("pytorch_api"),
            "status": execution_result.get("status"),
        },
    }


def build_prompt(sample: Dict[str, Any]) -> str:
    """构建用于问题筛选的提示词。"""
    compact = simplify_sample_for_prompt(sample)
    sample_json = json.dumps(compact, ensure_ascii=False, indent=2)

    return f"""你是深度学习框架差分测试专家。请判断下面这个样例是否“值得提交为框架官方 issue”。

【判断目标】
只在“更可能是框架本身问题”时返回 true。若更可能是测试构造/参数映射/迁移差异导致的问题，返回 false。

【高价值 issue 参考特征】
1. 严重崩溃：段错误、core dump、进程崩溃，而非普通 Python 异常。
2. 显著数值错误：合法输入下出现远超浮点误差的异常结果（如反常 NaN/Inf、巨大偏差）。
3. 违反文档：输入符合官方文档却报不应出现的异常。
4. 静默错误：返回成功但 shape/dtype/语义明显错误。
5. 稳定性问题：卡死、死锁、明显资源泄漏。

【应排除的情况（通常返回 false）】
1. TF/PT 参数名不一致、参数缺失、dtype 字符串与对象不匹配等迁移问题。
2. 用例本身输入非法或语义不对齐（如把 dict 当 tensor 传入）。
3. 两框架本就非等价 API 的强行比较。
4. 明显是测试框架组装错误，而非算子实现错误。

【样例】
```json
{sample_json}
```

【输出要求】
仅输出 JSON，不要任何额外文本：
{{
  "is_issue": true 或 false,
  "reason": "中文简短理由（1-2句话，不超过60字）"
}}
"""


def parse_llm_json(text: str) -> Tuple[bool, str]:
    """解析 LLM 返回，失败时给保守默认值。"""
    if not isinstance(text, str) or not text.strip():
        return False, "LLM返回为空"

    raw = text.strip()
    # 先尝试直接 json
    try:
        data = json.loads(raw)
        return normalize_llm_result(data)
    except json.JSONDecodeError:
        pass

    # 再尝试提取最外层 JSON 块
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            data = json.loads(match.group(0))
            return normalize_llm_result(data)
        except json.JSONDecodeError:
            pass

    return False, "LLM返回解析失败"


def normalize_llm_result(data: Dict[str, Any]) -> Tuple[bool, str]:
    """归一化 is_issue/reason 字段。"""
    issue_val = data.get("is_issue", False)
    if isinstance(issue_val, bool):
        is_issue = issue_val
    elif isinstance(issue_val, str):
        is_issue = issue_val.strip().lower() in {"true", "1", "yes", "y"}
    else:
        is_issue = False

    reason = data.get("reason", "")
    if not isinstance(reason, str) or not reason.strip():
        reason = "未提供有效理由"

    return is_issue, reason.strip()[:120]


def evaluate_one_sample(
    client: OpenAI,
    sample: Dict[str, Any],
    model: str,
    temperature: float,
    max_retries: int,
    print_lock: Lock,
) -> Tuple[bool, str]:
    """调用 LLM 判断单个样例。"""
    prompt = build_prompt(sample)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是严谨的深度学习框架缺陷分诊专家，只输出合法JSON。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=256,
            )

            content = response.choices[0].message.content
            is_issue, reason = parse_llm_json(content)
            return is_issue, reason

        except Exception as exc:
            with print_lock:
                print(f"  ⚠️ LLM调用失败，重试 {attempt + 1}/{max_retries}: {str(exc)[:120]}")
            time.sleep(2 ** attempt)

    return False, "LLM调用失败（重试耗尽）"


def load_input_samples(input_path: Path) -> Dict[str, Any]:
    """加载输入 JSON。"""
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data.get("samples"), list):
        raise ValueError("输入JSON缺少 samples 列表")

    return data


def save_output(output_path: Path, payload: Dict[str, Any]) -> None:
    """保存输出 JSON。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    default_input = ROOT_DIR / "tf_pt_test" / "analysis" / "pytorch_error_only_samples_20260221_211516.json"

    parser = argparse.ArgumentParser(
        description="使用 qwen3-max 筛选更可能提交 issue 的差分测试错误样例"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"输入JSON文件路径（默认: {default_input}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出JSON路径（默认自动在输入目录生成 issue_candidates_xxx.json）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM模型名称（默认: {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并发线程数（默认: {DEFAULT_WORKERS}）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="仅处理前N个样例（用于小测试）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM温度（默认0.0，更稳定）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="单样例最大重试次数（默认3）",
    )
    parser.add_argument(
        "--key-path",
        type=str,
        default=DEFAULT_KEY_PATH,
        help=f"API key 文件路径（默认: {DEFAULT_KEY_PATH}）",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="每次成功请求后延迟秒数（默认0.2）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else build_default_output_path(input_path)
    workers = max(1, int(args.workers))

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    api_key = load_api_key(args.key_path)
    if not api_key:
        raise RuntimeError("缺少可用的 API key")

    raw_data = load_input_samples(input_path)
    all_samples: List[Dict[str, Any]] = raw_data.get("samples", [])

    if args.max_samples is not None and args.max_samples > 0:
        samples = all_samples[: args.max_samples]
    else:
        samples = all_samples

    print("=" * 80)
    print("Issue 候选筛选（qwen3-max）")
    print("=" * 80)
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"样例总数: {len(all_samples)}")
    print(f"本次处理: {len(samples)}")
    print(f"并发线程: {workers}")
    print(f"模型: {args.model}")
    print("=" * 80)

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    print_lock = Lock()
    selected_lock = Lock()
    all_results_lock = Lock()
    selected_samples: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []
    started_at = time.time()

    def process(idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        is_issue, reason = evaluate_one_sample(
            client=client,
            sample=sample,
            model=args.model,
            temperature=float(args.temperature),
            max_retries=int(args.max_retries),
            print_lock=print_lock,
        )

        result = {
            "index": idx,
            "is_issue": is_issue,
            "reason": reason,
            "sample": sample if is_issue else build_minimal_discriminative_sample(sample),
        }

        with print_lock:
            flag = "✅" if is_issue else "⏭️"
            print(f"  {flag} 样例#{idx}: is_issue={is_issue} | {reason[:40]}")

        if args.delay > 0:
            time.sleep(args.delay)
        return result

    if workers <= 1:
        for idx, sample in enumerate(samples, start=1):
            outcome = process(idx, sample)
            with all_results_lock:
                all_results.append(outcome)
            if outcome["is_issue"]:
                with selected_lock:
                    selected_samples.append(outcome)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(process, idx, sample): idx
                for idx, sample in enumerate(samples, start=1)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    outcome = future.result()
                except Exception as exc:
                    with print_lock:
                        print(f"  ❌ 样例#{idx} 处理异常: {str(exc)[:120]}")
                    continue

                with all_results_lock:
                    all_results.append(outcome)
                if outcome["is_issue"]:
                    with selected_lock:
                        selected_samples.append(outcome)

    all_results.sort(key=lambda x: x["index"])
    selected_samples.sort(key=lambda x: x["index"])

    payload = {
        "generated_at": datetime.now().isoformat(),
        "source_file": str(input_path),
        "model": args.model,
        "criteria": "筛选更可能属于框架本身问题、而非迁移/参数映射问题的样例",
        "total_samples": len(all_samples),
        "processed_samples": len(samples),
        "non_issue_count": len(samples) - len(selected_samples),
        "issue_candidate_count": len(selected_samples),
        "results": all_results,
        "candidates": selected_samples,
    }
    save_output(output_path, payload)

    elapsed = time.time() - started_at
    print("=" * 80)
    print("筛选完成")
    print("=" * 80)
    print(f"候选数量: {len(selected_samples)} / {len(samples)}")
    print(f"耗时: {elapsed:.1f} 秒")
    print(f"结果文件: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
