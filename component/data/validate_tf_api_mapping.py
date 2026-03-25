# ./component/data/validate_tf_api_mapping.py
"""基于 LLM 验证 PyTorch API 与 TensorFlow API 映射的正确性"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import sys

# 添加项目根目录到路径，保证可以导入 component 下的模块
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.migration.migrate_generate_tests import get_qwen_client
from component.doc.doc_crawler_factory import get_doc_content

DEFAULT_MODEL = "qwen-plus"
DEFAULT_KEY_PATH = "aliyun.key"

# 日志文件路径
LOG_DIR = ROOT / "component" / "data" / "llm_logs"


def load_api_mappings(csv_path: Path) -> List[Dict[str, str]]:
    """从 api_mappings.csv 中加载 API 映射列表"""
    mappings: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pt_api = row.get("pytorch-api", "").strip()
            tf_api = row.get("tensorflow-api", "").strip()
            if pt_api:
                mappings.append({
                    "pytorch_api": pt_api,
                    "tensorflow_api": tf_api,
                })
    return mappings


def fetch_api_docs(pytorch_api: str, tensorflow_api: str) -> Tuple[str, str]:
    """
    爬取 PyTorch 和 TensorFlow API 的官方文档
    
    Returns:
        (pytorch_doc, tensorflow_doc)
    """
    pt_doc = ""
    tf_doc = ""
    
    # 爬取 PyTorch 文档
    if pytorch_api:
        try:
            doc_text = get_doc_content(pytorch_api, "pytorch")
            if doc_text and "无法获取" not in doc_text and "不支持" not in doc_text:
                pt_doc = doc_text
            else:
                print(f"[WARN] 未能获取 PyTorch 文档: {pytorch_api}")
        except Exception as e:
            print(f"[WARN] 获取 PyTorch 文档失败 {pytorch_api}: {e}")
    
    # 爬取 TensorFlow 文档（如果不是"无对应实现"）
    if tensorflow_api and tensorflow_api != "无对应实现":
        try:
            doc_text = get_doc_content(tensorflow_api, "tensorflow")
            if doc_text and "无法获取" not in doc_text and "不支持" not in doc_text:
                tf_doc = doc_text
            else:
                print(f"[WARN] 未能获取 TensorFlow 文档: {tensorflow_api}")
        except Exception as e:
            print(f"[WARN] 获取 TensorFlow 文档失败 {tensorflow_api}: {e}")
    
    return pt_doc, tf_doc


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


def build_validation_prompt(
    pytorch_api: str,
    tensorflow_api: str,
    pt_doc: str,
    tf_doc: str,
) -> str:
    """为验证 API 映射构建提示词"""
    
    pt_doc_text = pt_doc if pt_doc else "未找到相关 PyTorch 文档"
    tf_doc_text = tf_doc if tf_doc else "未找到相关 TensorFlow 文档"
    
    # 情况1：原映射为"无对应实现"，需要 LLM 重新寻找
    if tensorflow_api == "无对应实现" or not tensorflow_api:
        # 需要 LLM 重新寻找对应 API
        prompt = f"""你是一个精通 PyTorch 和 TensorFlow 的深度学习框架专家。

【任务】
当前记录显示 PyTorch API "{pytorch_api}" 在 TensorFlow 中"无对应实现"。
请你根据下面提供的 PyTorch 官方文档，判断 TensorFlow 中是否真的没有功能等价的 API。
如果你认为存在对应的 TensorFlow API，请给出具体的 API 名称。

【PyTorch API】
{pytorch_api}

【PyTorch 官方文档】
{pt_doc_text}

【TensorFlow API 命名空间参考】
- 基础数学运算：tf.xxx（如 tf.abs, tf.add, tf.matmul）
- 神经网络层（类）：tf.keras.layers.XXX（如 tf.keras.layers.Dense, tf.keras.layers.Conv2D）
- 神经网络函数：tf.nn.xxx（如 tf.nn.relu, tf.nn.softmax）
- 损失函数（类）：tf.keras.losses.XXX
- 损失函数（函数）：tf.keras.losses.xxx
- 线性代数：tf.linalg.xxx
- 随机数：tf.random.xxx
- 信号处理：tf.signal.xxx
- 图像处理：tf.image.xxx

【要求】
1. 仔细阅读 PyTorch 文档，理解该 API 的功能、参数和返回值。
2. 根据你对 TensorFlow 的了解，判断是否存在功能等价的 API。
3. 如果存在，给出具体的 TensorFlow API 名称；如果确实不存在，返回"无对应实现"。
4. 返回的 API 级别要与原 API 一致（函数对函数，类对类）。

【输出格式】
请严格按照以下 JSON 格式输出，不要包含任何其他内容：

```json
{{
    "pytorch_api": "{pytorch_api}",
    "tensorflow_api": "<找到的对应 TensorFlow API 或 '无对应实现'>",
    "confidence": "<high/medium/low>",
    "reason": "<用一两句话简要说明找到的 API 为何匹配，或为何确实无对应实现，回答不要太长>"
}}
```

注意：
- tensorflow_api 字段只填写 API 全名（如 tf.abs 或 tf.keras.layers.Conv1D），或 "无对应实现"
- confidence 表示你对这个映射的信心程度（85%以上是high，40%-85%是medium，40%以下是low）
- tensorflow_api 字段的值一定要是真实存在的 TensorFlow API 名称，不能自己编造不存在的 API。
- reason 简要说明映射的理由(一两句话即可，不要太长)
"""
    # 情况2：有 TensorFlow API 但文档为空（说明该 API 可能不存在/杜撰）
    elif not tf_doc:
        api_level = determine_api_level(pytorch_api)
        level_desc = "函数" if api_level == "function" else "类"
        
        prompt = f"""你是一个精通 PyTorch 和 TensorFlow 的深度学习框架专家。

【任务】
当前记录显示 PyTorch API "{pytorch_api}" 映射到 TensorFlow API "{tensorflow_api}"。
但是，我们**无法获取到 "{tensorflow_api}" 的官方文档**，这很可能意味着该 TensorFlow API 名称是错误的或不存在的。

请你根据下面提供的 PyTorch 官方文档，重新判断 TensorFlow 中是否存在功能等价的 API，并给出正确的 API 名称。

【PyTorch API】
{pytorch_api}

【原映射的 TensorFlow API（可能不存在）】
{tensorflow_api}

【PyTorch 官方文档】
{pt_doc_text}

【API 级别】
这是一个 **{level_desc}级别** 的 API。返回的 TensorFlow API 必须与原 API 级别一致。

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

【要求】
1. 仔细阅读 PyTorch 文档，理解该 API 的功能、参数和返回值。
2. 根据你对 TensorFlow 的了解，找到功能等价的 API。
3. **必须返回真实存在的 TensorFlow API 名称**，不能编造不存在的 API。
4. 如果确实不存在对应 API，返回"无对应实现"。

【输出格式】
请严格按照以下 JSON 格式输出，不要包含任何其他内容：

```json
{{
    "pytorch_api": "{pytorch_api}",
    "tensorflow_api": "<正确的 TensorFlow API 或 '无对应实现'>",
    "confidence": "<high/medium/low>",
    "reason": "<说明为何原 API 错误以及新 API 为何正确，或为何确实无对应实现>"
}}
```

注意：
- tensorflow_api 字段只填写 API 全名（如 tf.abs 或 tf.keras.layers.Conv1D），或 "无对应实现"
- confidence 表示你对这个映射的信心程度（85%以上是high，40%-85%是medium，40%以下是low）
- tensorflow_api 字段的值一定要是真实存在的 TensorFlow API 名称，不能自己编造不存在的 API。
- reason 简要说明映射的理由(一两句话即可，不要太长)
"""
    # 情况3：有 TensorFlow API 且文档存在，验证映射是否正确
    else:
        prompt = f"""你是一个精通 PyTorch 和 TensorFlow 的深度学习框架专家。

【任务】
请验证以下 PyTorch API 与 TensorFlow API 的映射是否正确。
你需要根据给出的文档信息判断这两个 API 在功能上是否等价或极高度相似。

【当前映射】
- PyTorch API: {pytorch_api}
- TensorFlow API: {tensorflow_api}

【PyTorch 官方文档】
{pt_doc_text}

【TensorFlow 官方文档】
{tf_doc_text}

【验证要点】
1. **功能一致性**：两个 API 的核心功能是否相同？
2. **参数对应性**：主要参数是否能够对应？
3. **返回值兼容性**：返回值的类型和含义是否兼容？
4. **API 级别**：是否都是函数级别或都是类级别？

【判断标准】
- 如果两个 API 功能等价，参数可以对应，认为映射**正确**。
- 如果功能有显著差异，或参数无法对应，认为映射**错误**，并给出更合适的 TensorFlow API（如果有的话）。
- 如果当前映射错误且 TensorFlow 中确实没有对应 API，返回"无对应实现"。

【输出格式】
请严格按照以下 JSON 格式输出，不要包含任何其他内容：

```json
{{
    "pytorch_api": "{pytorch_api}",
    "tensorflow_api": "<验证后的 TensorFlow API：如果原映射正确则保持不变，如果错误则给出正确的 API 或 '无对应实现'>",
    "confidence": "<high/medium/low>",
    "reason": "<用一两句话简要说明映射正确/错误的理由，如果修改了映射请说明原因，回答不要太长>"
}}
```

注意：
- tensorflow_api 字段只填写 API 全名（如 tf.abs 或 tf.keras.layers.Conv1D），或 "无对应实现"
- confidence 表示你对这个映射的信心程度（85%以上是high，40%-85%是medium，40%以下是low）
- tensorflow_api 字段的值一定要是真实存在的 TensorFlow API 名称，不能自己编造不存在的 API。
- reason 简要说明映射的理由(一两句话即可，不要太长)
"""
    
    return prompt


def parse_llm_response(response: str, original_tf_api: str) -> Tuple[str, str, str]:
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
            tf_api = data.get("tensorflow_api", original_tf_api).strip()
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
    
    # 返回原始值
    return original_tf_api, "unknown", "解析失败，保持原映射"


def validate_mapping_with_llm(
    client,
    pytorch_api: str,
    tensorflow_api: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> Tuple[str, str, str, str, bool, bool]:
    """
    使用 LLM 验证 API 映射
    
    Returns:
        (validated_tf_api, confidence, reason, full_response, pt_doc_empty, tf_doc_empty)
    """
    # 爬取文档
    pt_doc, tf_doc = fetch_api_docs(pytorch_api, tensorflow_api)
    
    # 记录文档是否为空
    pt_doc_empty = not pt_doc
    tf_doc_empty = not tf_doc and tensorflow_api and tensorflow_api != "无对应实现"
    
    # 构建提示词
    prompt = build_validation_prompt(pytorch_api, tensorflow_api, pt_doc, tf_doc)
    
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
            
            tf_api, confidence, reason = parse_llm_response(full_response, tensorflow_api)
            return tf_api, confidence, reason, full_response, pt_doc_empty, tf_doc_empty
            
        except Exception as e:
            print(f"[WARN] API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            continue
    
    return tensorflow_api, "unknown", f"调用失败，已重试 {max_retries} 次", "[ERROR] LLM 调用失败", pt_doc_empty, tf_doc_empty


def save_validation_log(
    log_entries: List[dict],
    log_path: Path,
) -> None:
    """保存验证日志到文件"""
    with log_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch to TensorFlow API Mapping Validation - LLM Log\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总条数: {len(log_entries)}\n")
        f.write("=" * 80 + "\n\n")
        
        for entry in log_entries:
            f.write("-" * 60 + "\n")
            f.write(f"序号: {entry['index']}\n")
            f.write(f"PyTorch API: {entry['pytorch_api']}\n")
            f.write(f"原 TensorFlow API: {entry['original_tf_api']}\n")
            f.write(f"验证后 TensorFlow API: {entry['validated_tf_api']}\n")
            f.write(f"置信度: {entry['confidence']}\n")
            f.write(f"是否修改: {'是' if entry['changed'] else '否'}\n")
            f.write(f"理由: {entry['reason']}\n")
            f.write(f"\n【LLM 完整输出】\n{entry['llm_response']}\n")
            f.write("-" * 60 + "\n\n")


def save_validated_csv(
    output_path: Path,
    validated_mappings: List[Dict[str, str]],
) -> None:
    """保存验证后的 CSV 文件"""
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pytorch-api", "tensorflow-api", "confidence", "changed"])
        for mapping in validated_mappings:
            writer.writerow([
                mapping["pytorch_api"],
                mapping["tensorflow_api"],
                mapping.get("confidence", ""),
                "Y" if mapping.get("changed", False) else "N",
            ])


def main():
    """命令行入口：批量验证 PyTorch API 与 TensorFlow API 的映射"""
    parser = argparse.ArgumentParser(
        description="基于 LLM 验证 PyTorch API 与 TensorFlow API 映射的正确性"
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
        default=str(ROOT / "component" / "data" / "api_mappings_validated.csv"),
        help="输出的验证后 CSV 文件路径",
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
        default=1.0,
        help="每次 API 调用之间的延迟秒数（默认 1.0，因为需要爬取文档）",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.1,
        help="LLM 温度参数，越低输出越确定性，范围 0.0-1.0（默认 0.1）",
    )
    parser.add_argument(
        "--log-dir",
        default=str(LOG_DIR),
        help="LLM 日志输出目录",
    )
    parser.add_argument(
        "--only-no-impl",
        action="store_true",
        help="仅处理'无对应实现'的记录",
    )
    parser.add_argument(
        "--only-has-impl",
        action="store_true",
        help="仅处理有对应实现的记录（验证现有映射）",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}")
        return

    print(f"[INFO] 正在加载 API 映射: {input_path}")
    all_mappings = load_api_mappings(input_path)
    
    if not all_mappings:
        print("[ERROR] 未从 CSV 中读取到任何 API 映射")
        return

    # 根据过滤条件筛选
    if args.only_no_impl:
        mappings = [m for m in all_mappings if m["tensorflow_api"] == "无对应实现" or not m["tensorflow_api"]]
        print(f"[INFO] 筛选'无对应实现'记录: {len(mappings)} 条")
    elif args.only_has_impl:
        mappings = [m for m in all_mappings if m["tensorflow_api"] and m["tensorflow_api"] != "无对应实现"]
        print(f"[INFO] 筛选有对应实现的记录: {len(mappings)} 条")
    else:
        mappings = all_mappings

    # 处理 start 和 limit 参数
    total_mappings = len(mappings)
    start_idx = args.start
    end_idx = total_mappings if args.limit is None else min(start_idx + args.limit, total_mappings)
    
    mappings_to_process = mappings[start_idx:end_idx]
    print(f"[INFO] 共有 {total_mappings} 条记录，本次处理范围: [{start_idx}, {end_idx})，共 {len(mappings_to_process)} 条")

    try:
        client = get_qwen_client(args.key_path)
        print(f"[INFO] LLM 客户端初始化成功，使用模型: {args.model}")
    except Exception as e:
        print(f"[ERROR] 无法初始化 LLM 客户端: {e}")
        return

    # 存储结果
    validated_mappings: List[Dict[str, str]] = []
    log_entries: List[dict] = []

    # 统计变量
    unchanged_count = 0
    changed_count = 0
    found_new_count = 0  # 原来无对应实现，现在找到了
    
    # 文档爬取失败的记录
    pt_doc_empty_apis: List[str] = []  # PyTorch 文档为空的 API
    tf_doc_empty_apis: List[str] = []  # TensorFlow 文档为空的 API

    # 逐个处理映射
    for i, mapping in enumerate(mappings_to_process, start=start_idx):
        pt_api = mapping["pytorch_api"]
        original_tf_api = mapping["tensorflow_api"]
        
        status_desc = "无对应实现" if (original_tf_api == "无对应实现" or not original_tf_api) else original_tf_api
        print(f"[INFO] 处理 [{i + 1}/{total_mappings}] {pt_api} -> {status_desc}")
        
        validated_tf_api, confidence, reason, llm_response, pt_doc_empty, tf_doc_empty = validate_mapping_with_llm(
            client,
            pt_api,
            original_tf_api,
            model=args.model,
            temperature=args.temperature,
        )
        
        # 记录文档爬取失败的 API
        if pt_doc_empty:
            pt_doc_empty_apis.append(pt_api)
        if tf_doc_empty:
            tf_doc_empty_apis.append(f"{pt_api} -> {original_tf_api}")
        
        # 判断是否有变化
        changed = validated_tf_api != original_tf_api
        if changed:
            changed_count += 1
            if (original_tf_api == "无对应实现" or not original_tf_api) and validated_tf_api != "无对应实现":
                found_new_count += 1
                print(f"       -> [新发现] {validated_tf_api}")
            else:
                print(f"       -> [修改] {original_tf_api} => {validated_tf_api}")
        else:
            unchanged_count += 1
            print(f"       -> [确认] {validated_tf_api}")
        
        validated_mappings.append({
            "pytorch_api": pt_api,
            "tensorflow_api": validated_tf_api,
            "confidence": confidence,
            "changed": changed,
        })
        
        log_entries.append({
            "index": i + 1,
            "pytorch_api": pt_api,
            "original_tf_api": original_tf_api,
            "validated_tf_api": validated_tf_api,
            "confidence": confidence,
            "reason": reason,
            "changed": changed,
            "llm_response": llm_response,
        })
        
        # 延迟以避免 API 限流
        if args.delay > 0 and i < end_idx - 1:
            time.sleep(args.delay)

    # 保存验证后的 CSV
    save_validated_csv(output_path, validated_mappings)
    print(f"[SUCCESS] 验证结果已保存到: {output_path}")

    # 保存 LLM 日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"pt_tf_validation_log_{timestamp}.txt"
    log_path = log_dir / log_filename
    save_validation_log(log_entries, log_path)
    print(f"[SUCCESS] LLM 日志已保存到: {log_path}")

    # 统计信息
    print("\n" + "=" * 50)
    print("【验证统计信息】")
    print(f"  总记录数: {total_mappings}")
    print(f"  本次处理: {len(mappings_to_process)}")
    print(f"  映射确认（无变化）: {unchanged_count}")
    print(f"  映射修改: {changed_count}")
    print(f"  新发现对应 API: {found_new_count}")
    print("=" * 50)
    
    # 打印文档爬取失败的统计
    print("\n" + "=" * 50)
    print("【文档爬取失败统计】")
    print(f"  PyTorch 文档为空: {len(pt_doc_empty_apis)} 条")
    print(f"  TensorFlow 文档为空: {len(tf_doc_empty_apis)} 条")
    print("=" * 50)
    
    if pt_doc_empty_apis:
        print("\n【PyTorch 文档为空的 API 列表】")
        for api in pt_doc_empty_apis:
            print(f"  - {api}")
    
    if tf_doc_empty_apis:
        print("\n【TensorFlow 文档为空的 API 列表】")
        for api in tf_doc_empty_apis:
            print(f"  - {api}")


if __name__ == "__main__":
    main()
