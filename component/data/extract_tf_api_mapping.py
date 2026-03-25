# ./component/data/extract_tf_api_mapping.py
"""基于 LLM 提取 PyTorch API 对应的 TensorFlow API 映射"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import sys

# 添加项目根目录到路径，保证可以导入 component 下的模块
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.migration.migrate_generate_tests import get_qwen_client

DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"

# LLM 日志文件路径
LOG_DIR = ROOT / "component" / "data" / "llm_logs"


def load_pytorch_apis(csv_path: Path) -> List[str]:
    """从 api_mappings.csv 中加载 PyTorch API 列表"""
    apis: List[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            api = row.get("pytorch-api", "").strip()
            if api:
                apis.append(api)
    return apis


def determine_api_level(api_name: str) -> str:
    """
    判断 API 的级别：函数级别 or 类级别
    
    规则：
    - torch.nn.XXX 且首字母大写 -> 类级别 (如 torch.nn.Conv1d, torch.nn.ReLU)
    - torch.nn.functional.xxx -> 函数级别 (如 torch.nn.functional.relu)
    - torch.xxx 且首字母小写 -> 函数级别 (如 torch.abs, torch.add)
    - torch.nn.utils.xxx -> 函数级别 (如 torch.nn.utils.clip_grad_norm_)
    """
    parts = api_name.split(".")
    
    # torch.nn.functional.xxx -> 函数级别
    if "functional" in api_name:
        return "function"
    
    # torch.nn.utils.xxx -> 函数级别
    if "utils" in api_name:
        return "function"
    
    # torch.nn.XXX 且最后部分首字母大写 -> 类级别
    if len(parts) >= 3 and parts[1] == "nn":
        last_part = parts[-1]
        if last_part and last_part[0].isupper():
            return "class"
    
    # 默认为函数级别
    return "function"


def build_prompt_for_api(pytorch_api: str, api_level: str) -> str:
    """为单个 PyTorch API 构建提示词"""
    
    level_desc = "函数" if api_level == "function" else "类"
    level_example_pt = "torch.abs" if api_level == "function" else "torch.nn.Conv1d"
    level_example_tf = "tf.abs" if api_level == "function" else "tf.keras.layers.Conv1D"
    
    prompt = f"""你是一个精通 PyTorch 和 TensorFlow 的深度学习框架专家。

【任务】
请为以下 PyTorch API 找到 TensorFlow 中功能最相近的对应 API。

【PyTorch API】
{pytorch_api}

【API 级别】
这是一个 **{level_desc}级别** 的 API。
- 如果原 API 是函数（如 {level_example_pt}），请返回 TensorFlow 中对应的函数（如 {level_example_tf}）。
- 如果原 API 是类（如 torch.nn.Conv1d），请返回 TensorFlow 中对应的类（如 tf.keras.layers.Conv1D）。

【要求】
1. 返回的 TensorFlow API 必须与原 PyTorch API 级别一致：
   - 函数对应函数（如 torch.abs -> tf.abs）
   - 类对应类（如 torch.nn.Conv1d -> tf.keras.layers.Conv1D）
2. 优先选择功能和参数最接近的 API。
3. 如果 TensorFlow 中确实没有功能等价的对应 API，请返回 "无对应实现"。
4. 只返回一个最合适的 API，不要返回多个候选。

【TensorFlow API 命名空间参考】
- 基础数学运算：tf.xxx（如 tf.abs, tf.add, tf.matmul）
- 神经网络层（类）：tf.keras.layers.XXX（如 tf.keras.layers.Dense, tf.keras.layers.Conv2D）
- 神经网络函数：tf.nn.xxx（如 tf.nn.relu, tf.nn.softmax）
- 损失函数（类）：tf.keras.losses.XXX（如 tf.keras.losses.BinaryCrossentropy）
- 损失函数（函数）：tf.keras.losses.xxx（如 tf.keras.losses.binary_crossentropy）
- 线性代数：tf.linalg.xxx（如 tf.linalg.inv, tf.linalg.det）
- 随机数：tf.random.xxx（如 tf.random.normal, tf.random.uniform）
- 信号处理：tf.signal.xxx
- 图像处理：tf.image.xxx

【输出格式】
请严格按照以下 JSON 格式输出，不要包含任何其他内容：

```json
{{
    "pytorch_api": "{pytorch_api}",
    "tensorflow_api": "<对应的TensorFlow API 名称或'无对应实现'>",
    "confidence": "<high/medium/low>",
    "reason": "<简要说明映射理由或为何无对应实现>"
}}
```

注意：
- tensorflow_api 字段只填写 API 全名（如 tf.abs 或 tf.keras.layers.Conv1D），或 "无对应实现"
- confidence 表示你对这个映射的信心程度（85%以上是high，40%-85%是medium，40%以下是low）
- tensorflow_api 字段的值一定要是真实存在的 TensorFlow API 名称，不能自己编造不存在的 API。
- reason 简要说明映射的理由(一两句话即可，不要太长)
"""
    return prompt


def parse_llm_response(response: str) -> Tuple[str, str, str]:
    """
    解析 LLM 的 JSON 响应
    
    Returns:
        (tensorflow_api, confidence, reason)
    """
    try:
        # 尝试提取 JSON 块
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            tf_api = data.get("tensorflow_api", "无对应实现").strip()
            confidence = data.get("confidence", "unknown").strip()
            reason = data.get("reason", "").strip()
            return tf_api, confidence, reason
    except json.JSONDecodeError:
        pass
    
    # 如果解析失败，尝试简单提取
    if "无对应实现" in response:
        return "无对应实现", "unknown", "解析失败，但检测到无对应实现"
    
    # 尝试查找 tf. 开头的 API
    import re
    tf_pattern = r'(tf\.[a-zA-Z_][a-zA-Z0-9_\.]*)'
    matches = re.findall(tf_pattern, response)
    if matches:
        return matches[0], "unknown", "从响应文本中提取"
    
    return "无对应实现", "unknown", "解析失败"


def query_llm_for_api(
    client,
    pytorch_api: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.8,
    max_retries: int = 3,
) -> Tuple[str, str]:
    """
    调用 LLM 获取对应的 TensorFlow API
    
    Args:
        client: LLM 客户端
        pytorch_api: PyTorch API 名称
        model: LLM 模型名称
        temperature: 模型温度参数，越低输出越确定性（0.0-1.0）
        max_retries: 最大重试次数
    
    Returns:
        (tensorflow_api, full_response)
    """
    api_level = determine_api_level(pytorch_api)
    prompt = build_prompt_for_api(pytorch_api, api_level)
    
    for attempt in range(max_retries):
        try:
            if hasattr(client, "chat"):
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1024,
                )
                full_response = resp.choices[0].message.content.strip()
            else:
                resp = client.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1024,
                )
                full_response = resp.choices[0].message.content.strip()
            
            tf_api, confidence, reason = parse_llm_response(full_response)
            return tf_api, full_response
            
        except Exception as e:
            print(f"[WARN] {pytorch_api} API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            continue
    
    return "无对应实现", f"[ERROR] 调用失败，已重试 {max_retries} 次"


def process_single_api(
    client,
    pt_api: str,
    index: int,
    total: int,
    model: str,
    temperature: float,
    print_lock: Lock,
) -> Tuple[int, str, str, str, str]:
    """
    处理单个 API 的映射（用于并发执行）
    
    这个函数会被多个线程同时调用，每个线程处理一个 PyTorch API。
    
    Args:
        client: LLM 客户端（多线程共享，客户端本身是线程安全的）
        pt_api: PyTorch API 名称
        index: API 索引（从 0 开始）
        total: 总 API 数量
        model: LLM 模型名称
        temperature: 温度参数
        print_lock: 打印锁，用于线程安全的输出（避免多线程打印混乱）
    
    Returns:
        (index, pytorch_api, api_level, tensorflow_api, llm_response)
    """
    # 1. 判断 API 级别（函数级别 or 类级别）
    api_level = determine_api_level(pt_api)
    
    # 2. 使用锁保护打印操作，避免多线程输出混乱
    with print_lock:
        print(f"[INFO] 处理 [{index + 1}/{total}] {pt_api} (级别: {api_level})")
    
    # 3. 调用 LLM 获取 TensorFlow API 映射
    #    这是耗时操作，多个线程会同时执行这一步（并发的核心）
    tf_api, llm_response = query_llm_for_api(
        client,
        pt_api,
        model=model,
        temperature=temperature,
    )
    
    # 4. 再次使用锁保护打印操作
    with print_lock:
        print(f"       [{index + 1}/{total}] {pt_api} -> {tf_api}")
    
    # 5. 返回结果（包含索引，用于后续按顺序排列）
    return index, pt_api, api_level, tf_api, llm_response


def save_llm_log(
    log_entries: List[dict],
    log_path: Path,
) -> None:
    """保存 LLM 日志到文件"""
    with log_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch to TensorFlow API Mapping - LLM Log\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总条数: {len(log_entries)}\n")
        f.write("=" * 80 + "\n\n")
        
        for entry in log_entries:
            f.write("-" * 60 + "\n")
            f.write(f"序号: {entry['index']}\n")
            f.write(f"PyTorch API: {entry['pytorch_api']}\n")
            f.write(f"API 级别: {entry['api_level']}\n")
            f.write(f"TensorFlow API (提取结果): {entry['tensorflow_api']}\n")
            f.write(f"\n【LLM 完整输出】\n{entry['llm_response']}\n")
            f.write("-" * 60 + "\n\n")


def save_updated_csv(
    csv_path: Path,
    output_path: Path,
    api_mappings: List[Tuple[str, str]],
) -> None:
    """保存更新后的 CSV 文件"""
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pytorch-api", "tensorflow-api"])
        for pt_api, tf_api in api_mappings:
            writer.writerow([pt_api, tf_api])


def main():
    """命令行入口：批量提取 PyTorch API 对应的 TensorFlow API"""
    parser = argparse.ArgumentParser(
        description="基于 LLM 提取 PyTorch API 对应的 TensorFlow API 映射"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=str(ROOT / "component" / "data" / "api_mappings.csv"),
        help="输入的 api_mappings.csv 文件路径",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(ROOT / "component" / "data" / "api_mappings.csv"),
        help="输出的 CSV 文件路径（默认覆盖原文件）",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"LLM 模型名称（默认 {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--key-path",
        "-k",
        default=DEFAULT_KEY_PATH,
        help="API key 文件路径（默认 aliyun.key）",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="从第几个 API 开始处理（0-indexed，用于断点续传）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多处理多少个 API（默认全部）",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="每次 API 调用之间的延迟秒数（默认 0.5），并发模式下无作用",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.8,
        help="LLM 温度参数，越低输出越确定性，范围 0.0-1.0（默认 0.8）",
    )
    parser.add_argument(
        "--log-dir",
        default=str(LOG_DIR),
        help="LLM 日志输出目录",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=5,
        help="并发线程数（默认 5）",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}")
        return

    print(f"[INFO] 正在加载 PyTorch API 列表: {input_path}")
    pytorch_apis = load_pytorch_apis(input_path)
    
    if not pytorch_apis:
        print("[ERROR] 未从 CSV 中读取到任何 API")
        return

    # 处理 start 和 limit 参数
    total_apis = len(pytorch_apis)
    start_idx = args.start
    end_idx = total_apis if args.limit is None else min(start_idx + args.limit, total_apis)
    
    apis_to_process = pytorch_apis[start_idx:end_idx]
    print(f"[INFO] 共有 {total_apis} 个 API，本次处理范围: [{start_idx}, {end_idx})，共 {len(apis_to_process)} 个")

    try:
        client = get_qwen_client(args.key_path)
        print(f"[INFO] LLM 客户端初始化成功，使用模型: {args.model}")
        print(f"[INFO] 并发线程数: {args.workers}")
    except Exception as e:
        print(f"[ERROR] 无法初始化 LLM 客户端: {e}")
        return

    # 存储结果（使用字典以便按索引存储，保证最终顺序正确）
    results_dict = {}
    # 创建打印锁，用于多线程环境下的安全打印
    print_lock = Lock()

    # 如果从中间开始，先填充前面的 API（保持为空或从已有文件读取）
    if start_idx > 0:
        for i in range(start_idx):
            results_dict[i] = (pytorch_apis[i], "", "", "", "")

    # ==================== 并发处理核心逻辑 ====================
    print(f"\n[INFO] 开始并发处理，使用 {args.workers} 个线程...")
    start_time = time.time()
    
    # 创建线程池，max_workers 指定并发线程数（默认5个）
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 步骤1：提交所有任务到线程池
        # future_to_index 用于追踪每个任务对应的 API 索引
        future_to_index = {}
        
        for i, pt_api in enumerate(apis_to_process, start=start_idx):
            # executor.submit() 立即返回一个 Future 对象，不会阻塞
            # 实际的 process_single_api() 会在线程池中异步执行
            future = executor.submit(
                process_single_api,  # 要执行的函数
                client,              # 参数1: LLM 客户端（所有线程共享）
                pt_api,              # 参数2: 当前要处理的 PyTorch API
                i,                   # 参数3: 索引
                total_apis,          # 参数4: 总数
                args.model,          # 参数5: 模型名称
                args.temperature,    # 参数6: 温度参数
                print_lock,          # 参数7: 打印锁
            )
            # 记录 Future 对象和索引的映射关系
            future_to_index[future] = i
        
        # 此时所有任务已提交，线程池中的5个线程会并发执行这些任务
        # 例如：如果有100个API，5个线程会同时处理前5个，完成一个就处理下一个
        
        # 步骤2：收集结果
        # as_completed() 会按任务完成的顺序返回 Future 对象（不是提交顺序）
        completed = 0
        for future in as_completed(future_to_index):
            try:
                # future.result() 获取任务的返回值（会阻塞直到该任务完成）
                index, pt_api, api_level, tf_api, llm_response = future.result()
                
                # 将结果存入字典，使用索引作为 key（保证最终顺序正确）
                results_dict[index] = (pt_api, api_level, tf_api, llm_response, index)
                completed += 1
                
                # 每完成10个任务，显示一次进度和预计剩余时间
                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = len(apis_to_process) - completed
                    eta = avg_time * remaining
                    with print_lock:
                        print(f"\n[PROGRESS] 已完成 {completed}/{len(apis_to_process)}，"
                              f"耗时 {elapsed:.1f}s，预计剩余 {eta:.1f}s\n")
            except Exception as e:
                # 如果某个任务失败，记录错误但不影响其他任务
                index = future_to_index[future]
                pt_api = pytorch_apis[index]
                with print_lock:
                    print(f"[ERROR] 处理 {pt_api} 时发生异常: {e}")
                results_dict[index] = (pt_api, "unknown", "无对应实现", f"[ERROR] {e}", index)

    # 所有任务完成，统计总耗时
    elapsed_time = time.time() - start_time
    print(f"\n[INFO] 并发处理完成，总耗时: {elapsed_time:.1f}s")
    # ==================== 并发处理结束 ====================

    # 如果有剩余的 API（limit 导致未处理完），填充空值
    if end_idx < total_apis:
        for i in range(end_idx, total_apis):
            results_dict[i] = (pytorch_apis[i], "", "", "", i)

    # 按索引顺序构建最终结果
    api_mappings: List[Tuple[str, str]] = []
    log_entries: List[dict] = []
    
    for i in range(total_apis):
        if i in results_dict:
            pt_api, api_level, tf_api, llm_response, _ = results_dict[i]
            api_mappings.append((pt_api, tf_api))
            
            # 只为实际处理的 API 添加日志
            if start_idx <= i < end_idx:
                log_entries.append({
                    "index": i + 1,
                    "pytorch_api": pt_api,
                    "api_level": api_level if api_level else determine_api_level(pt_api),
                    "tensorflow_api": tf_api,
                    "llm_response": llm_response,
                })
        else:
            api_mappings.append((pytorch_apis[i], ""))

    # 保存结果
    save_updated_csv(input_path, output_path, api_mappings)
    print(f"[SUCCESS] API 映射已保存到: {output_path}")

    # 保存 LLM 日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"pt_tf_mapping_log_{timestamp}.txt"
    log_path = log_dir / log_filename
    save_llm_log(log_entries, log_path)
    print(f"[SUCCESS] LLM 日志已保存到: {log_path}")

    # 统计信息
    mapped_count = sum(1 for _, tf in api_mappings if tf and tf != "无对应实现")
    no_impl_count = sum(1 for _, tf in api_mappings if tf == "无对应实现")
    empty_count = sum(1 for _, tf in api_mappings if not tf)
    
    print("\n" + "=" * 50)
    print("【统计信息】")
    print(f"  总 API 数: {total_apis}")
    print(f"  本次处理: {len(apis_to_process)}")
    print(f"  成功映射: {mapped_count}")
    print(f"  无对应实现: {no_impl_count}")
    print(f"  未处理: {empty_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
