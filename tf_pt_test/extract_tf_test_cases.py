#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2: 基于 LLM 从 TensorFlow 官方测试文件中提取/生成标准化测试用例

功能：
- 读取 Step 1 输出的 TF API 列表
- 对每个 API，读取对应的测试文件内容
- 调用 LLM 从测试文件中提取/生成标准化的测试用例
- 支持并发调用 LLM
- 输出结构化的测试用例集（JSON 格式）

用法：
    conda activate tf_env
    python tf_pt_test/extract_tf_test_cases.py [--input data/tf_apis_existing.json] [--output data/tf_test_cases.json] [--workers 4]

输出：tf_pt_test/data/tf_test_cases.json
"""

import os
import sys
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


def read_test_file_content(tf_dir: str, source_file: str, max_chars: int = MAX_FILE_CHARS) -> str:
    """
    读取测试文件内容（截断到 max_chars 字符以控制 token 消耗）
    
    优先保留文件开头（import 和类定义）和测试方法
    """
    filepath = os.path.join(tf_dir, source_file)
    if not os.path.exists(filepath):
        return ""

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return ""

    if len(content) <= max_chars:
        return content

    # 文件过长时，保留开头和中间有用的部分
    # 保留前 60% 的配额给开头，40% 给中间的测试方法
    head_chars = int(max_chars * 0.6)
    tail_chars = max_chars - head_chars

    head = content[:head_chars]
    # 从中间找测试方法
    remaining = content[head_chars:]
    tail_start = max(0, len(remaining) // 4)
    tail = remaining[tail_start:tail_start + tail_chars]

    return head + "\n\n# ... (中间部分省略) ...\n\n" + tail


def build_extraction_prompt(tf_api: str, file_content: str, num_cases: int = DEFAULT_NUM_CASES) -> str:
    """构建 LLM 提取测试用例的提示词"""
    prompt = f"""你是一个深度学习框架测试专家，精通 TensorFlow 框架的各种算子/API。

## 任务
请从以下 TensorFlow 官方测试文件中，为 API `{tf_api}` 提取或生成测试用例。（优先提取，不够再生成）
要求：
- 如果测试文件中用例数 > {num_cases}，请把可用的用例**全部提取**出来。
- 如果测试文件中用例数 < {num_cases}，请在提取基础上**补充生成**，直到数量**至少为 {num_cases}**。

## 测试文件内容
```python
{file_content}
```

## 输出要求
请输出严格的 JSON 格式，格式如下：

```json
{{
    "api": "{tf_api}",
    "is_class_api": false,
    "test_cases": [
        {{
            "description": "基本功能测试",
            "inputs": {{
                "x": {{"shape": [2, 3], "dtype": "float32"}},
                "axis": 1
            }}
        }},
        {{
            "description": "边界值测试 - 空张量",
            "inputs": {{
                "x": {{"shape": [0, 3], "dtype": "float32"}}
            }}
        }}
    ]
}}
```

## 规则
1. `is_class_api`: 判断该 API 是否为类形式（如 `tf.keras.layers.Dense`）。如果是函数形式（如 `tf.nn.relu`），设为 `false`。
2. 每个测试用例的 `inputs` 字典中：
   - **张量参数**必须用 `{{"shape": [...], "dtype": "..."}}` 格式描述（dtype 不带 tf. 前缀，如 "float32"、"int64"、"bool"）
   - **标量参数**直接用数值，如 `"axis": 1`、`"keepdims": true`
   - **字符串参数**直接用字符串，如 `"padding": "SAME"`
   - **列表参数**直接用列表，如 `"strides": [1, 1]`
3. 测试用例需要覆盖：
   - 基本功能（正常输入）
   - 不同数据类型（float32, float64, int32 等）
   - 不同形状（1D, 2D, 高维）
   - 边界值（空张量、单元素、极大/极小值等，如果文件中有的话）
4. 优先分析并从测试文件中提取测试用例，实在提取不到再根据经验生成。提取时需要收集测试文件中真实使用的输入数据和参数，不要凭空捏造不合理的测试用例；生成时尽量根据现有的用例生成类似的变体
5. 确保 shape 是合理的（不要太大，每个维度不超过 10），dtype 是 TensorFlow 支持的
6. **不要**包含 markdown 标记或其他额外文字，只输出纯 JSON
"""
    return prompt


def extract_test_cases_for_api(
    llm_client: OpenAI,
    model: str,
    tf_api: str,
    file_content: str,
    num_cases: int = DEFAULT_NUM_CASES,
    print_lock: Lock = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    调用 LLM 为单个 TF API 提取测试用例
    
    Args:
        llm_client: OpenAI 客户端
        model: LLM 模型名称
        tf_api: TF API 名称
        file_content: 测试文件内容
        num_cases: 要提取的测试用例数
        print_lock: 线程安全打印锁
        max_retries: 最大重试次数
    
    Returns:
        提取结果字典
    """
    lock = print_lock or Lock()
    prompt = build_extraction_prompt(tf_api, file_content, num_cases)

    for attempt in range(max_retries):
        try:
            completion = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是深度学习测试专家，擅长分析TensorFlow测试代码并提取标准化的测试用例。只输出JSON，不要输出其他内容。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4096,
            )

            raw_response = completion.choices[0].message.content.strip()
            time.sleep(0.5)  # 避免频控

            # 解析 JSON
            result = _parse_json_response(raw_response)
            if result and "test_cases" in result:
                if len(result["test_cases"]) < num_cases:
                    with lock:
                        print(
                            f"  ⚠️ {tf_api} 用例数不足（{len(result['test_cases'])}/{num_cases}），已接受LLM结果"
                        )
                with lock:
                    print(f"  ✅ {tf_api} → {len(result['test_cases'])} 个测试用例")
                return result

            with lock:
                print(f"  ⚠️ {tf_api} 返回格式异常，重试 ({attempt + 1}/{max_retries})")

        except Exception as e:
            with lock:
                print(f"  ❌ {tf_api} LLM调用失败: {str(e)[:80]}，重试 ({attempt + 1}/{max_retries})")
            time.sleep(2 ** attempt)

    # 所有重试都失败，返回空结果
    with lock:
        print(f"  ❌ {tf_api} 最终失败，使用默认测试用例")
    return _default_test_case(tf_api)


def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    """解析 LLM 返回的 JSON（带容错）"""
    # 去除 markdown 代码块
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 对象
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def _default_test_case(tf_api: str) -> Dict[str, Any]:
    """为无法提取的 API 生成默认测试用例"""
    return {
        "api": tf_api,
        "is_class_api": False,
        "test_cases": [
            {
                "description": "默认测试用例",
                "inputs": {
                    "x": {"shape": [2, 3], "dtype": "float32"}
                }
            }
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: 基于 LLM 从 TensorFlow 测试文件中提取测试用例"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(ROOT_DIR, "tf_pt_test", "data", "tf_apis_existing.json"),
        help="Step 1 输出的 TF API 列表文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(ROOT_DIR, "tf_pt_test", "data", "tf_test_cases.json"),
        help="输出的测试用例文件路径"
    )
    parser.add_argument(
        "--tf-dir",
        default=os.path.join(ROOT_DIR, "tf_testcases"),
        help="tf_testcases 目录路径"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"LLM 并发线程数（默认 {DEFAULT_WORKERS}）"
    )
    parser.add_argument(
        "--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES,
        help=f"每个 API 提取的测试用例数（默认 {DEFAULT_NUM_CASES}）"
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
        "--start", type=int, default=0,
        help="从第几个 API 开始处理（0-indexed，用于断点续传）"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="最多处理多少个 API"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="LLM 调用间隔秒数（默认 0.5）"
    )

    args = parser.parse_args()
    workers = max(1, args.workers)

    print("=" * 80)
    print("Step 2: 基于 LLM 从 TensorFlow 测试文件中提取测试用例")
    print("=" * 80)

    # 加载 API 列表
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        print("请先运行 Step 1: python tf_pt_test/extract_tf_apis.py")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        api_data = json.load(f)

    all_apis = api_data.get("apis", [])
    print(f"📋 共加载 {len(all_apis)} 个 TF API")

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
    apis_remaining = [a for a in apis_to_process if a["tf_api"] not in results]
    print(f"📌 待处理: {len(apis_remaining)} 个（跳过已处理的 {len(apis_to_process) - len(apis_remaining)} 个）")

    if not apis_remaining:
        print("✅ 所有API已处理完毕")
        return

    # 准备文件内容缓存（按 source_file 分组）
    file_content_cache: Dict[str, str] = {}

    def get_file_content(source_file: str) -> str:
        if source_file not in file_content_cache:
            file_content_cache[source_file] = read_test_file_content(
                args.tf_dir, source_file
            )
        return file_content_cache[source_file]

    # 预加载文件内容
    source_files_raw = [a["source_file"] for a in apis_remaining]
    print(f"\n📖 未去重测试文件数: {len(source_files_raw)}")
    print(source_files_raw[:20])
    source_files = set(source_files_raw)
    print(f"📖 预加载 {len(source_files)} 个测试文件...")
    for sf in source_files:
        get_file_content(sf)

    # 并发处理
    def process_api(api_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        tf_api = api_info["tf_api"]
        file_content = get_file_content(api_info["source_file"])
        result = extract_test_cases_for_api(
            llm_client, args.model, tf_api, file_content,
            num_cases=args.num_cases, print_lock=print_lock,
        )
        result["source_file"] = api_info["source_file"]
        result["category"] = api_info["category"]
        time.sleep(args.delay)
        return tf_api, result

    print(f"\n🚀 开始提取测试用例 (并发={workers})...\n")
    start_time = time.time()
    completed = 0
    total = len(apis_remaining)

    if workers <= 1:
        for api_info in apis_remaining:
            tf_api, result = process_api(api_info)
            results[tf_api] = result
            completed += 1
            if completed % 10 == 0:
                _save_results(args.output, results)
                print(f"  💾 进度: {completed}/{total}，已保存中间结果")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_api = {
                executor.submit(process_api, api_info): api_info["tf_api"]
                for api_info in apis_remaining
            }
            for future in as_completed(future_to_api):
                try:
                    tf_api, result = future.result()
                    results[tf_api] = result
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
    total_cases = sum(
        len(v.get("test_cases", [])) for v in results.values()
    )

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
