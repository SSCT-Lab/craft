#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3: 基于 LLM 生成 PaddlePaddle → PyTorch 的 API 映射表

功能：
- 读取 Step 1/1.5 输出的 Paddle API 列表
- 对每个 Paddle API，调用 LLM 查找功能等价的 PyTorch API
- 支持并发调用 LLM、断点续传
- 输出 CSV 映射表

用法：
    conda activate tf_env
    python pd_pt_test/extract_pd_pt_mapping.py [--input data/pd_apis_existing.json] [--output data/pd_pt_mapping.csv] [--workers 6]

输出：pd_pt_test/data/pd_pt_mapping.csv
"""

import os
import sys
import csv
import json
import time
import re
import argparse
from datetime import datetime
from pathlib import Path
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
    """
    判断 Paddle API 的级别：函数级别 or 类级别

    规则：
    - paddle.nn.Conv2D（首字母大写）→ 类级别
    - paddle.nn.functional.relu → 函数级别
    """
    parts = api_name.split(".")
    if len(parts) >= 2:
        last_part = parts[-1]
        if last_part and last_part[0].isupper():
            return "class"
    return "function"


def build_prompt_for_api(pd_api: str, api_level: str) -> str:
    """为单个 Paddle API 构建 LLM 提示词"""
    level_desc = "函数" if api_level == "function" else "类"
    level_example_pd = "paddle.nn.functional.relu" if api_level == "function" else "paddle.nn.Conv2D"
    level_example_pt = "torch.nn.functional.relu" if api_level == "function" else "torch.nn.Conv2d"

    prompt = f"""你是一个精通 PaddlePaddle 和 PyTorch 的深度学习框架专家。

【任务】
请为以下 PaddlePaddle API 找到 PyTorch 中功能等价的对应 API。

【PaddlePaddle API】
{pd_api}

【API 级别】
这是一个 **{level_desc}级别** 的 API。
- 如果原 API 是函数（如 {level_example_pd}），请返回 PyTorch 中对应的函数（如 {level_example_pt}）。
- 如果原 API 是类（如 paddle.nn.Conv2D），请返回 PyTorch 中对应的类（如 torch.nn.Conv2d）。

【要求】
1. 返回的 PyTorch API 必须与原 PaddlePaddle API 功能等价或极为接近。
2. 优先选择功能和参数最接近的 API。
3. 如果 PyTorch 中确实没有功能等价的对应 API，请返回 "无对应实现"。
4. 只返回一个最合适的 API，不要返回多个候选。

【PyTorch API 命名空间参考】
- 基础函数：torch.xxx（如 torch.abs, torch.add, torch.matmul）
- 神经网络层（类）：torch.nn.XXX（如 torch.nn.Conv1d, torch.nn.ReLU, torch.nn.Linear）
- 神经网络函数：torch.nn.functional.xxx（如 torch.nn.functional.relu, torch.nn.functional.softmax）
- 线性代数：torch.linalg.xxx（如 torch.linalg.det, torch.linalg.inv）
- 随机数：torch.xxx 或 torch.distributions.xxx
- 信号处理（FFT）：torch.fft.xxx
- 图像处理：torchvision.transforms.functional.xxx

【PaddlePaddle → PyTorch 常见映射参考】
- paddle.nn.functional.xxx ↔ torch.nn.functional.xxx（大部分直接对应）
- paddle.nn.XXX ↔ torch.nn.XXX（大部分直接对应，注意大小写：Paddle用大驼峰如Conv2D，PyTorch用Conv2d）
- paddle.linalg.xxx ↔ torch.linalg.xxx
- paddle.fft.xxx ↔ torch.fft.xxx
- paddle.xxx（如 paddle.add, paddle.abs）↔ torch.xxx
- paddle.Tensor.xxx ↔ torch.Tensor.xxx

【输出格式】
请严格按照以下 JSON 格式输出，不要包含任何其他内容：

```json
{{
    "paddle_api": "{pd_api}",
    "pytorch_api": "<对应的PyTorch API 名称或'无对应实现'>",
    "confidence": "<high/medium/low>",
    "reason": "<简要说明映射理由或为何无对应实现>"
}}
```

注意：
- pytorch_api 字段只填写 API 全名（如 torch.abs 或 torch.nn.Conv2d），或 "无对应实现"
- pytorch_api 字段的值一定要是真实存在的 PyTorch API 名称，不能自己编造不存在的 API
- confidence 表示你对这个映射等价的信心程度（85%以上是high，40%-85%是medium，40%以下是low）
- reason 简要说明这个映射的理由(一两句话即可，不要太长)
- 只为公开的 Paddle API 提供映射，针对内部或实验性 API，pytorch_api 字段直接返回"无对应实现"。
"""
    return prompt


def parse_llm_response(response: str) -> Tuple[str, str, str]:
    """
    解析 LLM 的 JSON 响应

    Returns:
        (pytorch_api, confidence, reason)
    """
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            pt_api = data.get("pytorch_api", "无对应实现").strip()
            confidence = data.get("confidence", "unknown").strip()
            reason = data.get("reason", "").strip()
            return pt_api, confidence, reason
    except json.JSONDecodeError:
        pass

    if "无对应实现" in response:
        return "无对应实现", "unknown", "解析失败，但检测到无对应实现"

    torch_pattern = r'(torch\.[a-zA-Z_][a-zA-Z0-9_\.]*)'
    matches = re.findall(torch_pattern, response)
    if matches:
        return matches[0], "unknown", "从响应文本中提取"

    return "无对应实现", "unknown", "解析失败"


def query_llm_for_api(
    client: OpenAI,
    pd_api: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_retries: int = 3,
    print_lock: Lock = None,
) -> Tuple[str, str, str]:
    """
    调用 LLM 获取对应的 PyTorch API

    Returns:
        (pytorch_api, confidence, reason)
    """
    lock = print_lock or Lock()
    api_level = determine_api_level(pd_api)
    prompt = build_prompt_for_api(pd_api, api_level)

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

            pt_api, confidence, reason = parse_llm_response(full_response)
            return pt_api, confidence, reason

        except Exception as e:
            with lock:
                print(f"  ⚠️ {pd_api} LLM调用失败: {str(e)[:80]}，重试 ({attempt + 1}/{max_retries})")
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
                pd_api = row.get("paddle-api", "").strip()
                pt_api = row.get("pytorch-api", "").strip()
                if pd_api:
                    existing[pd_api] = pt_api
    except Exception:
        pass
    return existing


def save_mapping(csv_path: str, mappings: List[Dict[str, str]]):
    """保存映射结果到 CSV"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["paddle-api", "pytorch-api", "confidence", "reason"])
        writer.writeheader()
        for m in mappings:
            writer.writerow(m)


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: 基于 LLM 生成 PaddlePaddle → PyTorch API 映射表"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(ROOT_DIR, "pd_pt_test", "data", "pd_apis_existing.json"),
        help="过滤后真实存在的 Paddle API 列表文件"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "pd_pt_test", "data", "pd_pt_mapping.csv"),
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
    print("Step 3: 基于 LLM 生成 PaddlePaddle → PyTorch API 映射表")
    print("=" * 80)

    # 加载 API 列表
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        api_data = json.load(f)

    all_apis = [a["pd_api"] for a in api_data.get("apis", [])]
    print(f"📋 共加载 {len(all_apis)} 个 Paddle API")

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

    # 初始化 LLM 客户端
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
    for pd_api, pt_api in existing_mapping.items():
        all_mappings.append({
            "paddle-api": pd_api,
            "pytorch-api": pt_api,
            "confidence": "",
            "reason": "已有映射",
        })

    # 处理函数
    def process_api(pd_api: str) -> Dict[str, str]:
        pt_api, confidence, reason = query_llm_for_api(
            llm_client, pd_api, model=args.model,
            temperature=args.temperature, print_lock=print_lock,
        )
        with print_lock:
            emoji = "✅" if pt_api != "无对应实现" else "⏭️"
            print(f"  {emoji} {pd_api} → {pt_api} ({confidence})")
        time.sleep(args.delay)
        return {
            "paddle-api": pd_api,
            "pytorch-api": pt_api,
            "confidence": confidence,
            "reason": reason,
        }

    print(f"\n🚀 开始生成 PD→PT 映射 (并发={workers})...\n")
    start_time = time.time()
    completed = 0
    total = len(apis_remaining)

    if workers <= 1:
        for pd_api in apis_remaining:
            result = process_api(pd_api)
            with mappings_lock:
                all_mappings.append(result)
            completed += 1
            if completed % 20 == 0:
                save_mapping(args.output, all_mappings)
                print(f"  💾 进度: {completed}/{total}，已保存中间结果")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_api = {
                executor.submit(process_api, pd_api): pd_api
                for pd_api in apis_remaining
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
                            "paddle-api": api_name,
                            "pytorch-api": "无对应实现",
                            "confidence": "unknown",
                            "reason": f"处理异常: {e}",
                        })

                completed += 1
                if completed % 30 == 0:
                    with mappings_lock:
                        save_mapping(args.output, all_mappings)
                    with print_lock:
                        print(f"  💾 进度: {completed}/{total}，已保存中间结果")

    # 按 API 名排序后保存
    all_mappings.sort(key=lambda x: x["paddle-api"])
    save_mapping(args.output, all_mappings)

    elapsed = time.time() - start_time
    has_impl = sum(1 for m in all_mappings if m["pytorch-api"] != "无对应实现")
    no_impl = len(all_mappings) - has_impl

    print(f"\n{'=' * 80}")
    print(f"📊 映射生成完成")
    print(f"{'=' * 80}")
    print(f"  总 API 数: {len(all_mappings)}")
    print(f"  有对应实现: {has_impl}")
    print(f"  无对应实现: {no_impl}")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  💾 已保存到: {args.output}")


if __name__ == "__main__":
    main()
