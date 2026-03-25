#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2: 基于 LLM 从 MindSpore 官方测试文件中提取/生成标准化测试用例

功能：
- 读取 Step 1 输出的 MS API 列表
- 对每个 API，读取对应的测试文件内容
- 调用 LLM 从测试文件中提取/生成标准化的测试用例
- 支持并发调用 LLM
- 输出结构化的测试用例集（JSON 格式）

用法：
    conda activate tf_env
    python ms_pt_test/extract_ms_test_cases.py [--input data/ms_apis_existing.json] [--output data/ms_test_cases.json] [--workers 6]

输出：ms_pt_test/data/ms_test_cases.json
"""

import os
import sys
import io

# Windows 环境下强制使用 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # 强制行缓冲与即时刷新，避免控制台无输出
    # sys.stdout = io.TextIOWrapper(
    #     sys.stdout.buffer,
    #     encoding='utf-8',
    #     errors='replace',
    #     line_buffering=True,
    #     write_through=True,
    # )
    # sys.stderr = io.TextIOWrapper(
    #     sys.stderr.buffer,
    #     encoding='utf-8',
    #     errors='replace',
    #     line_buffering=True,
    #     write_through=True,
    # )

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
DEFAULT_NUM_CASES = 5
MAX_FILE_CHARS = 8000  # 发送给 LLM 的测试文件最大字符数


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

    print("❌ 未找到 API 密钥，请确保 aliyun.key 文件存在或设置 DASHSCOPE_API_KEY 环境变量")
    return ""


def read_test_file_content(ms_dir: str, source_file: str, max_chars: int = MAX_FILE_CHARS) -> str:
    """
    读取测试文件内容（截断到 max_chars 字符以控制 token 消耗）
    优先保留文件开头（import 和类定义）和测试方法
    """
    filepath = os.path.join(ms_dir, source_file)
    if not os.path.exists(filepath):
        return ""

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return ""

    if len(content) <= max_chars:
        return content

    # 文件过长时，保留开头和中间的测试方法
    head_chars = int(max_chars * 0.6)
    tail_chars = max_chars - head_chars

    head = content[:head_chars]
    remaining = content[head_chars:]
    tail_start = max(0, len(remaining) // 4)
    tail = remaining[tail_start:tail_start + tail_chars]

    return head + "\n\n# ... (中间部分省略) ...\n\n" + tail


def build_extraction_prompt(ms_api: str, file_content: str, num_cases: int = DEFAULT_NUM_CASES) -> str:
    """构建 LLM 提取测试用例的提示词"""
    prompt = f"""你是一个深度学习框架测试专家，精通 MindSpore 框架的各种算子/API。

## 任务
请从以下 MindSpore 官方测试文件中，为 API `{ms_api}` 提取或生成测试用例。（优先提取，不够再生成）
要求：
- 如果测试文件中用例数 > {num_cases}，请把可用的用例**全部提取**出来。
- 如果测试文件中用例数 < {num_cases}，请在提取基础上**补充生成**，直到数量**至少为 {num_cases}**。

## MindSpore API 调用模式说明
MindSpore 算子有以下几种调用方式：
1. **Primitive算子（类式）**: `P.Abs()`创建实例后调用 `op(input)` 或 `self.op(input)`
2. **函数式API**: `F.abs(input)` 或 `ops.abs(input)` 直接调用
3. **NN层（类式）**: `nn.Conv2d(in_channels, out_channels, kernel_size)` 创建实例后 `layer(input)` 调用
4. **Tensor方法**: `tensor.add(other)` 直接在 Tensor 上调用

请根据 `{ms_api}` 的命名判断它属于哪种类型，并相应地构造测试用例。

## 测试文件内容
```python
{file_content}
```

## 输出要求
请输出严格的 JSON 格式，格式如下：

```json
{{
    "api": "{ms_api}",
    "is_class_api": true,
    "init_params": {{}},
    "test_cases": [
        {{
            "description": "基本功能测试",
            "inputs": {{
                "x": {{"shape": [2, 3], "dtype": "float32"}},
                "axis": 1
            }}
        }},
        {{
            "description": "不同数据类型测试",
            "inputs": {{
                "x": {{"shape": [3, 4], "dtype": "float64"}}
            }}
        }}
    ]
}}
```

## 规则
1. `is_class_api`: 判断该 API 是否为类形式。
   - Primitive 算子（如 `mindspore.ops.Abs`、`mindspore.ops.Conv2D`）→ `true`
   - NN 层（如 `mindspore.nn.Conv2d`、`mindspore.nn.BatchNorm2d`）→ `true`
   - 函数式 API（如 `mindspore.ops.abs`、`mindspore.ops.relu`）→ `false`
   - Tensor 方法（如 `mindspore.Tensor.add`）→ `false`

2. `init_params` (仅类式 API 需要): 类的初始化参数。
   - 对于 Primitive 算子，如 `P.Abs()` 不需要参数则为空 `{{}}`
   - 对于 NN 层，如 `nn.Conv2d(3, 64, 3)` 需要指定 `{{"in_channels": 3, "out_channels": 64, "kernel_size": 3}}`
   - 对于需要指定参数的 Primitive，如 `P.Conv2D(out_channel=64, kernel_size=3)` 需要对应记录

3. 每个测试用例的 `inputs` 字典中：
   - **张量参数**必须用 `{{"shape": [...], "dtype": "..."}}` 格式描述（dtype 不带 mindspore. 前缀，如 "float32"、"int64"、"bool"）
   - **标量参数**直接用数值，如 `"axis": 1`、`"keepdims": true`
   - **字符串参数**直接用字符串
   - **列表参数**直接用列表

4. 测试用例需要覆盖：
   - 基本功能（正常输入）
   - 不同数据类型（float32, float64, int32 等）
   - 不同形状（1D, 2D, 高维）
   - 边界值（空张量、单元素、极大/极小值等，如果文件中有的话）

5. 优先从测试文件中提取真实用例，实在提取不到再根据经验生成

6. 确保 shape 是合理的（不要太大，每个维度不超过 10），dtype 是标准的

7. **不要**包含 markdown 标记或其他额外文字，只输出纯 JSON
"""
    return prompt


def extract_test_cases_for_api(
    llm_client: OpenAI,
    model: str,
    ms_api: str,
    file_content: str,
    num_cases: int = DEFAULT_NUM_CASES,
    print_lock: Lock = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """调用 LLM 为单个 MS API 提取测试用例"""
    lock = print_lock or Lock()
    prompt = build_extraction_prompt(ms_api, file_content, num_cases)

    for attempt in range(max_retries):
        try:
            completion = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是深度学习测试专家，擅长分析MindSpore测试代码并提取标准化的测试用例。"
                            "只输出JSON，不要输出其他内容。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=4096,
            )

            raw_response = completion.choices[0].message.content.strip()
            time.sleep(0.5)

            result = _parse_json_response(raw_response)
            if result and "test_cases" in result:
                if len(result["test_cases"]) < num_cases:
                    with lock:
                        print(
                            f"  ⚠️ {ms_api} 用例数不足"
                            f"（{len(result['test_cases'])}/{num_cases}），已接受LLM结果"
                        )
                with lock:
                    print(f"  ✅ {ms_api} → {len(result['test_cases'])} 个测试用例")
                return result

            with lock:
                print(
                    f"  ⚠️ {ms_api} 返回格式异常，重试 ({attempt + 1}/{max_retries})"
                )

        except Exception as e:
            with lock:
                print(
                    f"  ❌ {ms_api} LLM调用失败: {str(e)[:80]}，"
                    f"重试 ({attempt + 1}/{max_retries})"
                )
            time.sleep(2 ** attempt)

    with lock:
        print(f"  ❌ {ms_api} 最终失败，使用默认测试用例")
    return _default_test_case(ms_api)


def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    """解析 LLM 返回的 JSON（带容错）"""
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def _default_test_case(ms_api: str) -> Dict[str, Any]:
    """为无法提取的 API 生成默认测试用例"""
    # 根据 API 名推断是否为类式
    parts = ms_api.split(".")
    last_part = parts[-1] if parts else ""
    is_class = last_part and last_part[0].isupper()

    return {
        "api": ms_api,
        "is_class_api": is_class,
        "init_params": {},
        "test_cases": [
            {
                "description": "默认测试用例",
                "inputs": {
                    "x": {"shape": [2, 3], "dtype": "float32"},
                },
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: 基于 LLM 从 MindSpore 测试文件中提取测试用例"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(ROOT_DIR, "ms_pt_test", "data", "ms_apis_existing.json"),
        help="Step 1 输出的 MS API 列表文件路径",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "ms_pt_test", "data", "ms_test_cases.json"),
        help="输出的测试用例文件路径",
    )
    parser.add_argument(
        "--ms-dir",
        default=os.path.join(ROOT_DIR, "testcases_ms"),
        help="testcases_ms 目录路径",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"LLM 并发线程数（默认 {DEFAULT_WORKERS}）",
    )
    parser.add_argument(
        "--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES,
        help=f"每个 API 提取的测试用例数（默认 {DEFAULT_NUM_CASES}）",
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
        "--start", type=int, default=0,
        help="从第几个 API 开始处理（0-indexed，用于断点续传）",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="最多处理多少个 API",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="LLM 调用间隔秒数（默认 0.5）",
    )

    args = parser.parse_args()
    workers = max(1, args.workers)

    print("=" * 80)
    print("Step 2: 基于 LLM 从 MindSpore 测试文件中提取测试用例")
    print("=" * 80)

    # 加载 API 列表
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        print("请先运行 Step 1: python ms_pt_test/extract_ms_apis.py")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        api_data = json.load(f)

    all_apis = api_data.get("apis", [])
    print(f"📋 共加载 {len(all_apis)} 个 MS API")

    # 确定处理范围
    start_idx = args.start
    end_idx = start_idx + args.limit if args.limit else len(all_apis)
    end_idx = min(end_idx, len(all_apis))
    apis_to_process = all_apis[start_idx:end_idx]

    print(f"📌 处理范围: [{start_idx}, {end_idx})，共 {len(apis_to_process)} 个")
    print(f"📌 并发线程数: {workers}")
    print(f"📌 每个API提取用例数: {args.num_cases}")
    print(f"📌 LLM模型: {args.model}")

    # 初始化 LLM 客户端
    api_key = load_api_key(args.key_path)
    if not api_key:
        sys.exit(1)

    llm_client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    print_lock = Lock()
    results: Dict[str, Any] = {}

    # 加载已有结果（支持断点续传）
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            results = existing_data.get("test_cases", {})
            print(f"📂 加载已有结果: {len(results)} 个API的测试用例")
        except Exception:
            pass

    # 过滤掉已处理的 API
    apis_remaining = [a for a in apis_to_process if a["ms_api"] not in results]
    print(f"📌 待处理: {len(apis_remaining)} 个"
          f"（跳过已处理的 {len(apis_to_process) - len(apis_remaining)} 个）")

    if not apis_remaining:
        print("✅ 所有API已处理完毕")
        return

    # 准备文件内容缓存
    file_content_cache: Dict[str, str] = {}

    def get_file_content(source_file: str) -> str:
        if source_file not in file_content_cache:
            file_content_cache[source_file] = read_test_file_content(
                args.ms_dir, source_file
            )
        return file_content_cache[source_file]

    # 预加载文件内容
    source_files = set(a["source_file"] for a in apis_remaining)
    print(f"\n📖 预加载 {len(source_files)} 个测试文件...")
    for sf in source_files:
        get_file_content(sf)

    # 并发处理
    def process_api(api_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        ms_api = api_info["ms_api"]
        file_content = get_file_content(api_info["source_file"])
        result = extract_test_cases_for_api(
            llm_client, args.model, ms_api, file_content,
            num_cases=args.num_cases, print_lock=print_lock,
        )
        result["source_file"] = api_info["source_file"]
        result["api_type"] = api_info.get("api_type", "ops")
        time.sleep(args.delay)
        return ms_api, result

    print(f"\n🚀 开始提取测试用例 (并发={workers})...\n")
    start_time = time.time()
    completed = 0
    total = len(apis_remaining)

    if workers <= 1:
        for api_info in apis_remaining:
            ms_api, result = process_api(api_info)
            results[ms_api] = result
            completed += 1
            if completed % 10 == 0:
                _save_results(args.output, results)
                print(f"  💾 进度: {completed}/{total}，已保存中间结果")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_api = {
                executor.submit(process_api, api_info): api_info["ms_api"]
                for api_info in apis_remaining
            }
            for future in as_completed(future_to_api):
                try:
                    ms_api, result = future.result()
                    results[ms_api] = result
                except Exception as e:
                    api_name = future_to_api[future]
                    with print_lock:
                        print(f"  ❌ {api_name} 处理异常: {e}")
                    results[api_name] = _default_test_case(api_name)

                completed += 1
                if completed % 20 == 0:
                    _save_results(args.output, results)
                    with print_lock:
                        print(f"  💾 进度: {completed}/{total}，已保存中间结果")

    # 最终保存
    _save_results(args.output, results)

    elapsed = time.time() - start_time
    total_cases = sum(len(v.get("test_cases", [])) for v in results.values())

    print(f"\n{'=' * 80}")
    print(f"📊 提取完成")
    print(f"{'=' * 80}")
    print(f"  API总数: {len(results)}")
    print(f"  测试用例总数: {total_cases}")
    print(f"  平均每个API: {total_cases / max(1, len(results)):.1f} 个用例")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  💾 已保存到: {args.output}")


def _save_results(output_path: str, results: Dict[str, Any]):
    """保存结果到 JSON 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_data = {
        "total_apis": len(results),
        "extraction_time": datetime.now().isoformat(),
        "test_cases": results,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
