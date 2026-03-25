#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3: 基于 LLM 生成 TensorFlow → MindSpore 的 API 映射表

功能：
- 读取 Step 1.5 输出的 TF API 列表
- 对每个 TF API，调用 LLM 查找功能等价的 MindSpore API
- 支持并发调用 LLM
- 支持断点续传
- 输出 CSV 映射表

用法：
    conda activate tf_env
    python tf_ms_test_1/extract_tf_ms_mapping.py

输出：tf_ms_test_1/data/tf_ms_mapping.csv
"""

import os
import sys
import csv
import json
import time
import re
import argparse
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
        with open(key_file, "r", encoding="utf-8") as f:
            api_key = f.read().strip()
        if api_key:
            return api_key

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        return api_key

    print("❌ 未找到 API 密钥")
    return ""


def determine_api_level(api_name: str) -> str:
    """
    判断 TF API 的级别：函数级别 or 类级别

    规则：
    - tf.keras.layers.XXX（首字母大写）→ 类级别
    - tf.keras.losses.XXX（首字母大写）→ 类级别
    - 其他 → 函数级别
    """
    parts = api_name.split(".")
    if "keras" in api_name and "layers" in api_name:
        last_part = parts[-1]
        if last_part and last_part[0].isupper():
            return "class"
    if "keras" in api_name and "losses" in api_name:
        last_part = parts[-1]
        if last_part and last_part[0].isupper():
            return "class"
    return "function"


def build_prompt_for_api(tf_api: str, api_level: str) -> str:
    """为单个 TF API 构建 LLM 提示词"""
    level_desc = "函数" if api_level == "function" else "类"
    level_example_tf = "tf.nn.relu" if api_level == "function" else "tf.keras.layers.Conv2D"
    level_example_ms = "mindspore.ops.relu" if api_level == "function" else "mindspore.nn.Conv2d"

    prompt = f"""你是一个精通 TensorFlow 和 MindSpore 的深度学习框架专家。

【任务】
请为以下 TensorFlow API 找到 MindSpore 中功能等价的对应 API。

【TensorFlow API】
{tf_api}

【API 级别】
这是一个 **{level_desc}级别** 的 API。
- 如果原 API 是函数（如 {level_example_tf}），请返回 MindSpore 中对应的函数（如 {level_example_ms}）。
- 如果原 API 是类（如 tf.keras.layers.Conv2D），请返回 MindSpore 中对应的类（如 mindspore.nn.Conv2d）。

【要求】
1. 返回的 MindSpore API 必须与原 TensorFlow API 功能等价或极为接近。
2. 优先选择功能和参数最接近的 API。
3. 如果 MindSpore 中确实没有功能等价的对应 API，请返回 "无对应实现"。
4. 只返回一个最合适的 API，不要返回多个候选。
5. 优先返回 CPU 可用的公开 API，不要返回内部、实验性或仅特定后端可用的接口。

【MindSpore API 命名空间参考】
- 基础函数：mindspore.xxx（如 mindspore.abs）
- 算子函数：mindspore.ops.xxx（如 mindspore.ops.add, mindspore.ops.matmul）
- 神经网络层（类）：mindspore.nn.XXX（如 mindspore.nn.Conv2d, mindspore.nn.ReLU, mindspore.nn.Dense）
- Tensor 方法：mindspore.Tensor.xxx（如 mindspore.Tensor.add）
- 线性代数：mindspore.ops.xxx 或 mindspore.scipy.linalg.xxx
- 随机数：mindspore.ops.standard_normal / mindspore.ops.uniform 等

【TensorFlow → MindSpore 常见映射参考】
- tf.math.abs ↔ mindspore.ops.abs（或 mindspore.abs）
- tf.add / tf.math.add ↔ mindspore.ops.add
- tf.matmul / tf.linalg.matmul ↔ mindspore.ops.matmul
- tf.nn.relu ↔ mindspore.ops.relu（或 mindspore.nn.ReLU 类）
- tf.keras.layers.Dense ↔ mindspore.nn.Dense

【输出格式】
请严格按照以下 JSON 格式输出，不要包含任何其他内容：

```json
{{
    "tensorflow_api": "{tf_api}",
    "mindspore_api": "<对应的MindSpore API 名称或'无对应实现'>",
    "confidence": "<high/medium/low>",
    "reason": "<简要说明映射理由或为何无对应实现>"
}}
```

注意：
- mindspore_api 字段只填写 API 全名（如 mindspore.ops.abs 或 mindspore.nn.Conv2d），或 "无对应实现"
- mindspore_api 字段的值一定要是真实存在的 MindSpore API 名称，不能自己编造不存在的 API
- confidence 表示你对这个映射等价的信心程度（85%以上是high，40%-85%是medium，40%以下是low）
- reason 简要说明这个映射的理由（一两句话即可，不要太长）
"""
    return prompt


def parse_llm_response(response: str) -> Tuple[str, str, str]:
    """
    解析 LLM 的 JSON 响应

    Returns:
        (mindspore_api, confidence, reason)
    """
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            ms_api = data.get("mindspore_api", "无对应实现").strip()
            confidence = data.get("confidence", "unknown").strip()
            reason = data.get("reason", "").strip()
            return ms_api, confidence, reason
    except json.JSONDecodeError:
        pass

    if "无对应实现" in response:
        return "无对应实现", "unknown", "解析失败，但检测到无对应实现"

    ms_pattern = r"(mindspore\.[a-zA-Z_][a-zA-Z0-9_\.]*)"
    matches = re.findall(ms_pattern, response)
    if matches:
        return matches[0], "unknown", "从响应文本中提取"

    return "无对应实现", "unknown", "解析失败"


def query_llm_for_api(
    client: OpenAI,
    tf_api: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_retries: int = 3,
    print_lock: Lock = None,
) -> Tuple[str, str, str]:
    """
    调用 LLM 获取对应的 MindSpore API

    Returns:
        (mindspore_api, confidence, reason)
    """
    lock = print_lock or Lock()
    api_level = determine_api_level(tf_api)
    prompt = build_prompt_for_api(tf_api, api_level)

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
                print(f"  ⚠️ {tf_api} LLM调用失败: {str(e)[:80]}，重试 ({attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)

    return "无对应实现", "unknown", "所有重试均失败"


def load_existing_mapping(csv_path: str) -> Dict[str, str]:
    """加载已有的映射结果（用于断点续传）"""
    if not os.path.exists(csv_path):
        return {}

    existing = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tf_api = row.get("tensorflow-api", "").strip()
                ms_api = row.get("mindspore-api", "").strip()
                if tf_api:
                    existing[tf_api] = ms_api
    except Exception:
        pass
    return existing


def save_mapping(csv_path: str, mappings: List[Dict[str, str]]) -> None:
    """保存映射结果到 CSV"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tensorflow-api", "mindspore-api", "confidence", "reason"])
        writer.writeheader()
        for mapping in mappings:
            writer.writerow(mapping)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: 基于 LLM 生成 TensorFlow → MindSpore API 映射表"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(ROOT_DIR, "tf_ms_test_1", "data", "tf_apis_existing.json"),
        help="过滤后真实存在的 TF API 列表文件"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "tf_ms_test_1", "data", "tf_ms_mapping.csv"),
        help="输出的 CSV 映射文件路径"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"LLM 并发线程数（默认 {DEFAULT_WORKERS}）"
    )
    parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL,
        help=f"LLM 模型名称（默认 {DEFAULT_MODEL}）"
    )
    parser.add_argument(
        "--key-path", "-k", default=DEFAULT_KEY_PATH,
        help=f"API key 文件路径（默认 {DEFAULT_KEY_PATH}）"
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.1,
        help="LLM 温度参数（默认 0.1，低温度更确定）"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="从第几个 API 开始处理（0-indexed）"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="最多处理多少个 API"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="API 调用间隔秒数"
    )

    args = parser.parse_args()
    workers = max(1, args.workers)

    print("=" * 80)
    print("Step 3: 基于 LLM 生成 TensorFlow → MindSpore API 映射表")
    print("=" * 80)

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        print("请先确认 tf_apis_existing.json 已生成")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        api_data = json.load(f)

    all_apis = [item["tf_api"] for item in api_data.get("apis", []) if item.get("tf_api")]
    print(f"📋 共加载 {len(all_apis)} 个 TF API")

    start_idx = args.start
    end_idx = start_idx + args.limit if args.limit else len(all_apis)
    end_idx = min(end_idx, len(all_apis))
    apis_to_process = all_apis[start_idx:end_idx]

    existing_mapping = load_existing_mapping(args.output)
    apis_remaining = [api for api in apis_to_process if api not in existing_mapping]

    print(f"📌 处理范围: [{start_idx}, {end_idx})，共 {len(apis_to_process)} 个")
    print(f"📌 已有映射: {len(existing_mapping)} 个（跳过）")
    print(f"📌 待处理: {len(apis_remaining)} 个")
    print(f"📌 并发线程数: {workers}")
    print(f"📌 LLM模型: {args.model}")

    if not apis_remaining:
        print("✅ 所有 API 已处理完毕")
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

    all_mappings: List[Dict[str, str]] = []
    for tf_api, ms_api in existing_mapping.items():
        all_mappings.append(
            {
                "tensorflow-api": tf_api,
                "mindspore-api": ms_api,
                "confidence": "",
                "reason": "已有映射",
            }
        )

    def process_api(tf_api: str) -> Dict[str, str]:
        ms_api, confidence, reason = query_llm_for_api(
            llm_client,
            tf_api,
            model=args.model,
            temperature=args.temperature,
            print_lock=print_lock,
        )
        with print_lock:
            emoji = "✅" if ms_api != "无对应实现" else "⏭️"
            print(f"  {emoji} {tf_api} → {ms_api} ({confidence})")
        time.sleep(args.delay)
        return {
            "tensorflow-api": tf_api,
            "mindspore-api": ms_api,
            "confidence": confidence,
            "reason": reason,
        }

    print(f"\n🚀 开始生成 TF→MS 映射 (并发={workers})...\n")
    start_time = time.time()
    completed = 0
    total = len(apis_remaining)

    if workers <= 1:
        for tf_api in apis_remaining:
            result = process_api(tf_api)
            with mappings_lock:
                all_mappings.append(result)
            completed += 1
            if completed % 20 == 0:
                save_mapping(args.output, all_mappings)
                print(f"  💾 进度: {completed}/{total}，已保存中间结果")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_api = {
                executor.submit(process_api, tf_api): tf_api
                for tf_api in apis_remaining
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
                        all_mappings.append(
                            {
                                "tensorflow-api": api_name,
                                "mindspore-api": "无对应实现",
                                "confidence": "unknown",
                                "reason": f"处理异常: {e}",
                            }
                        )

                completed += 1
                if completed % 30 == 0:
                    with mappings_lock:
                        save_mapping(args.output, all_mappings)
                    with print_lock:
                        print(f"  💾 进度: {completed}/{total}，已保存中间结果")

    all_mappings.sort(key=lambda item: item["tensorflow-api"])
    save_mapping(args.output, all_mappings)

    elapsed = time.time() - start_time
    has_impl = sum(1 for item in all_mappings if item["mindspore-api"] != "无对应实现")
    no_impl = len(all_mappings) - has_impl

    print(f"\n{'=' * 80}")
    print("📊 映射生成完成")
    print(f"{'=' * 80}")
    print(f"  总 API 数: {len(all_mappings)}")
    print(f"  有对应实现: {has_impl}")
    print(f"  无对应实现: {no_impl}")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  💾 已保存到: {args.output}")


if __name__ == "__main__":
    main()
