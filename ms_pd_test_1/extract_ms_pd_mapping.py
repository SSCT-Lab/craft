#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3: 基于 LLM 生成 MindSpore → PaddlePaddle 的 API 映射表

功能：
- 读取 Step 1/1.5 输出的 MS API 列表
- 对每个 MS API，调用 LLM 查找功能等价的 PaddlePaddle API
- 支持并发调用 LLM、断点续传
- 输出 CSV 映射表

PaddlePaddle 特性注意：
- Paddle 默认使用 NCHW 格式（与 MindSpore 一致），无需做格式转换
- Paddle 全连接层为 paddle.nn.Linear（MindSpore 为 mindspore.nn.Dense）
- Paddle 的函数式 API 在 paddle.nn.functional 和 paddle 顶层命名空间下
- Paddle 的卷积类名大写：paddle.nn.Conv2D（注意 D 大写）

用法：
    conda activate tf_env
    python ms_pd_test_1/extract_ms_pd_mapping.py [--input data/ms_apis_existing.json] [--output data/ms_pd_mapping.csv] [--workers 6]

输出：ms_pd_test_1/data/ms_pd_mapping.csv
"""

import os
import sys

import csv
import json
import time
import re
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== 常量定义 ====================
DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"
DEFAULT_WORKERS = 6


def load_api_key(key_path: str = DEFAULT_KEY_PATH) -> str:
    """加载阿里云 API 密钥"""
    if not os.path.isabs(key_path):
        key_file = os.path.join(ROOT_DIR, key_path)
    else:
        key_file = key_path

    if os.path.exists(key_file):
        with open(key_file, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()
        if api_key:
            return api_key

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        return api_key

    print("❌ 未找到 API 密钥")
    return ""


def determine_api_level(api_name: str) -> str:
    """判断 MS API 的级别：函数/类/Tensor方法"""
    parts = api_name.split(".")
    last_part = parts[-1] if parts else ""

    if "Tensor" in api_name:
        return "tensor_method"
    if ".nn." in api_name and last_part and last_part[0].isupper():
        return "class"
    if ".ops." in api_name and last_part and last_part[0].isupper():
        return "class"
    return "function"


def build_prompt_for_api(ms_api: str, api_level: str) -> str:
    """为单个 MS API 构建 LLM 提示词（MS→PD 映射）"""
    level_desc = {"function": "函数", "class": "类", "tensor_method": "Tensor方法"}.get(api_level, "函数")

    prompt = f"""你是一个精通 MindSpore 和 PaddlePaddle 的深度学习框架专家。

【任务】
请为以下 MindSpore API 找到 PaddlePaddle 中功能等价的对应 API。

【MindSpore API】
{ms_api}

【API 级别】
这是一个 **{level_desc}级别** 的 API。

【MindSpore API 到 PaddlePaddle 的常见映射参考】
- mindspore.ops.Abs / mindspore.ops.abs → paddle.abs
- mindspore.ops.Add / mindspore.ops.add → paddle.add
- mindspore.ops.MatMul / mindspore.ops.matmul → paddle.matmul
- mindspore.ops.Conv2D → paddle.nn.functional.conv2d
- mindspore.nn.Conv2d → paddle.nn.Conv2D（注意 Paddle 中 D 大写）
- mindspore.nn.BatchNorm2d → paddle.nn.BatchNorm2D（注意 Paddle 中 D 大写）
- mindspore.nn.Dense → paddle.nn.Linear
- mindspore.nn.ReLU → paddle.nn.ReLU
- mindspore.ops.Softmax / mindspore.ops.softmax → paddle.nn.functional.softmax
- mindspore.Tensor.add → paddle.Tensor.add

【要求】
1. 返回的 PaddlePaddle API 必须与原 MindSpore API 功能等价或极为接近
2. 优先选择功能和参数最接近的 API
3. 如果 PaddlePaddle 中确实没有功能等价的对应 API，请返回 "无对应实现"
4. 只返回一个最合适的 API，不要返回多个候选
5. 注意区分：
   - MindSpore Primitive 算子（如 mindspore.ops.Abs）通常对应 Paddle 的顶层函数（如 paddle.abs）
   - MindSpore NN 层（如 mindspore.nn.Conv2d）通常对应 Paddle 的 NN 层（如 paddle.nn.Conv2D）
   - MindSpore 函数式 API（如 mindspore.ops.relu）通常对应 Paddle 的 paddle.nn.functional.xxx
   - MindSpore Tensor 方法对应 Paddle 的 Tensor 方法

【PaddlePaddle API 命名空间参考】
- 基础函数：paddle.xxx（如 paddle.abs, paddle.add, paddle.matmul, paddle.reshape）
- 神经网络层（类）：paddle.nn.XXX（如 paddle.nn.Conv2D, paddle.nn.ReLU, paddle.nn.Linear）
  注意：Paddle 中卷积/归一化层后缀大写，如 Conv2D、BatchNorm2D（不同于 MindSpore 的 Conv2d、BatchNorm2d）
- 神经网络函数：paddle.nn.functional.xxx（如 paddle.nn.functional.relu, paddle.nn.functional.softmax）
- 线性代数：paddle.linalg.xxx（如 paddle.linalg.det, paddle.linalg.inv）
- 信号处理：paddle.fft.xxx（如 paddle.fft.fft, paddle.fft.ifft）
- 随机数：paddle.xxx（如 paddle.randn, paddle.rand）或 paddle.distribution.xxx
- 张量方法：paddle.Tensor.xxx（如 paddle.Tensor.add, paddle.Tensor.reshape）

【数据格式】
MindSpore 和 PaddlePaddle 默认都使用 NCHW 格式，无需转换。

【输出格式】
请严格按照以下 JSON 格式输出，不要包含任何其他内容：

```json
{{
    "mindspore_api": "{ms_api}",
    "paddle_api": "<对应的PaddlePaddle API 名称或'无对应实现'>",
    "confidence": "<high/medium/low>",
    "reason": "<简要说明映射理由或为何无对应实现>"
}}
```

注意：
- paddle_api 字段只填写 API 全名（如 paddle.abs 或 paddle.nn.Conv2D），或 "无对应实现"
- paddle_api 字段的值一定要是真实存在的 PaddlePaddle API 名称，不能自己编造
- confidence: 置信度，代表你对这个判断的自信程度。85%以上是high，40%-85%是medium，40%以下是low
- reason: 简要说明理由（一两句话）
- 只为公开的 MindSpore API 提供映射，针对内部或实验性 API，paddle_api 字段直接返回"无对应实现"。
"""
    return prompt


def parse_llm_response(response: str) -> Tuple[str, str, str]:
    """
    解析 LLM 的 JSON 响应

    Returns:
        (paddle_api, confidence, reason)
    """
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            pd_api = data.get("paddle_api", "无对应实现").strip()
            confidence = data.get("confidence", "unknown").strip()
            reason = data.get("reason", "").strip()
            return pd_api, confidence, reason
    except json.JSONDecodeError:
        pass

    if "无对应实现" in response:
        return "无对应实现", "unknown", "解析失败，但检测到无对应实现"

    # 尝试从响应文本中提取 Paddle API
    pd_pattern = r'(paddle\.[a-zA-Z_][a-zA-Z0-9_\.]*)'
    matches = re.findall(pd_pattern, response)
    if matches:
        return matches[0], "unknown", "从响应文本中提取"

    return "无对应实现", "unknown", "解析失败"


def query_llm_for_api(
    client: OpenAI,
    ms_api: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_retries: int = 3,
    print_lock: Lock = None,
) -> Tuple[str, str, str]:
    """
    调用 LLM 获取对应的 PaddlePaddle API

    Returns:
        (paddle_api, confidence, reason)
    """
    lock = print_lock or Lock()
    api_level = determine_api_level(ms_api)
    prompt = build_prompt_for_api(ms_api, api_level)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024,
            )
            full_response = resp.choices[0].message.content.strip()
            time.sleep(0.5)

            return parse_llm_response(full_response)

        except Exception as e:
            with lock:
                print(
                    f"  ⚠️ {ms_api} LLM调用失败: {str(e)[:80]}，"
                    f"重试 ({attempt + 1}/{max_retries})"
                )
            time.sleep(2 ** attempt)

    return "无对应实现", "unknown", "所有重试均失败"


def load_existing_mapping(csv_path: str) -> Dict[str, str]:
    """加载已有的映射结果（用于断点续传）"""
    if not os.path.exists(csv_path):
        return {}
    existing = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ms_api = row.get("mindspore-api", "").strip()
                pd_api = row.get("paddle-api", "").strip()
                if ms_api:
                    existing[ms_api] = pd_api
    except Exception:
        pass
    return existing


def save_mapping(csv_path: str, mappings: List[Dict[str, str]]):
    """保存映射结果到 CSV"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=["mindspore-api", "paddle-api", "confidence", "reason"]
        )
        writer.writeheader()
        for m in mappings:
            writer.writerow(m)


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: 基于 LLM 生成 MindSpore → PaddlePaddle API 映射表"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(ROOT_DIR, "ms_pd_test_1", "data", "ms_apis_existing.json"),
        help="MS API 列表文件",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "ms_pd_test_1", "data", "ms_pd_mapping.csv"),
        help="输出的 CSV 映射文件路径",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"LLM 并发线程数（默认 {DEFAULT_WORKERS}）",
    )
    parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL,
        help=f"LLM 模型名称（默认 {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--key-path", "-k", default=DEFAULT_KEY_PATH,
        help=f"API key 文件路径（默认 {DEFAULT_KEY_PATH}）",
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.1,
        help="LLM 温度参数（默认 0.1）",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="从第几个 API 开始处理（0-indexed）",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="最多处理多少个 API",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="API 调用间隔秒数",
    )

    args = parser.parse_args()
    workers = max(1, args.workers)

    print("=" * 80)
    print("Step 3: 基于 LLM 生成 MindSpore → PaddlePaddle API 映射表")
    print("=" * 80)

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        api_data = json.load(f)

    all_apis = [a["ms_api"] for a in api_data.get("apis", [])]
    print(f"📋 共加载 {len(all_apis)} 个 MS API")

    # 确定处理范围
    start_idx = args.start
    end_idx = start_idx + args.limit if args.limit else len(all_apis)
    end_idx = min(end_idx, len(all_apis))
    apis_to_process = all_apis[start_idx:end_idx]

    # 加载已有映射（断点续传）
    existing_mapping = load_existing_mapping(args.output)
    apis_remaining = [a for a in apis_to_process if a not in existing_mapping]

    print(f"📌 处理范围: [{start_idx}, {end_idx})，共 {len(apis_to_process)} 个")
    print(f"📌 已有映射: {len(existing_mapping)} 个（跳过）")
    print(f"📌 待处理: {len(apis_remaining)} 个")
    print(f"📌 并发线程数: {workers}")
    print(f"📌 LLM模型: {args.model}")

    if not apis_remaining:
        print("✅ 所有API已处理完毕")
        return

    api_key = load_api_key(args.key_path)
    if not api_key:
        sys.exit(1)

    llm_client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    print_lock = Lock()
    mappings_lock = Lock()

    # 初始化结果列表（包含已有映射）
    all_mappings: List[Dict[str, str]] = []
    for ms_api, pd_api in existing_mapping.items():
        all_mappings.append({
            "mindspore-api": ms_api,
            "paddle-api": pd_api,
            "confidence": "",
            "reason": "已有映射",
        })

    def process_api(ms_api: str) -> Dict[str, str]:
        pd_api, confidence, reason = query_llm_for_api(
            llm_client, ms_api, model=args.model,
            temperature=args.temperature, print_lock=print_lock,
        )
        with print_lock:
            emoji = "✅" if pd_api != "无对应实现" else "⏭️"
            print(f"  {emoji} {ms_api} → {pd_api} ({confidence})")
        time.sleep(args.delay)
        return {
            "mindspore-api": ms_api,
            "paddle-api": pd_api,
            "confidence": confidence,
            "reason": reason,
        }

    print(f"\n🚀 开始生成 MS→PD 映射 (并发={workers})...\n")
    start_time = time.time()
    completed = 0
    total = len(apis_remaining)

    if workers <= 1:
        for ms_api in apis_remaining:
            result = process_api(ms_api)
            with mappings_lock:
                all_mappings.append(result)
            completed += 1
            if completed % 20 == 0:
                save_mapping(args.output, all_mappings)
                print(f"  💾 进度: {completed}/{total}，已保存中间结果")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_api = {
                executor.submit(process_api, ms_api): ms_api
                for ms_api in apis_remaining
            }
            for future in as_completed(future_to_api):
                try:
                    result = future.result()
                    with mappings_lock:
                        all_mappings.append(result)
                except Exception as e:
                    api_name = future_to_api[future]
                    with print_lock:
                        print(f"  ❌ {api_name} 异常: {e}")
                    with mappings_lock:
                        all_mappings.append({
                            "mindspore-api": api_name,
                            "paddle-api": "无对应实现",
                            "confidence": "unknown",
                            "reason": f"处理异常: {e}",
                        })

                completed += 1
                if completed % 30 == 0:
                    with mappings_lock:
                        save_mapping(args.output, all_mappings)
                    with print_lock:
                        print(f"  💾 进度: {completed}/{total}，已保存中间结果")

    all_mappings.sort(key=lambda x: x["mindspore-api"])
    save_mapping(args.output, all_mappings)

    elapsed = time.time() - start_time
    has_impl = sum(1 for m in all_mappings if m["paddle-api"] != "无对应实现")

    print(f"\n{'=' * 80}")
    print(f"📊 映射生成完成")
    print(f"{'=' * 80}")
    print(f"  总 API 数: {len(all_mappings)}")
    print(f"  有对应实现: {has_impl}")
    print(f"  无对应实现: {len(all_mappings) - has_impl}")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  💾 已保存到: {args.output}")


if __name__ == "__main__":
    main()
