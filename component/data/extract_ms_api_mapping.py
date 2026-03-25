# ./component/data/extract_ms_api_mapping.py
"""基于 LLM 提取 PyTorch API 对应的 MindSpore API 映射"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

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


def load_csv_data(csv_path: Path) -> Tuple[List[str], List[dict]]:
    """
    从 api_mappings.csv 中加载完整数据
    
    Returns:
        (fieldnames, rows) - 字段名列表和所有行数据
    """
    rows: List[dict] = []
    fieldnames: List[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        for row in reader:
            rows.append(dict(row))
    return fieldnames, rows


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
    level_example_ms = "mindspore.ops.abs" if api_level == "function" else "mindspore.nn.Conv1d"
    
    prompt = f"""你是一个精通 PyTorch 和 MindSpore 的深度学习框架专家。

【任务】
请为以下 PyTorch API 找到 MindSpore 中功能等价的对应 API。

【PyTorch API】
{pytorch_api}

【API 级别】
这是一个 **{level_desc}级别** 的 API。
- 如果原 API 是函数（如 {level_example_pt}），请返回 MindSpore 中对应的函数（如 {level_example_ms}）。
- 如果原 API 是类（如 torch.nn.Conv1d），请返回 MindSpore 中对应的类（如 mindspore.nn.Conv1d）。

【要求】（请重点阅读！！！）
1. 返回的 MindSpore API 必须与原 PyTorch API 级别一致：
   - 函数对应函数（如 torch.abs -> mindspore.ops.abs）
   - 类对应类（如 torch.nn.Conv1d -> mindspore.nn.Conv1d）
2. 优先选择功能和参数最接近的 API。
3. 我目前是在 CPU 平台上运行，请不要返回仅在特定硬件（如 Ascend、GPU）上可用的 API，如果该 API 没有 CPU 版本的对应 API，请返回 "无对应实现"。
4. 如果 MindSpore 中确实没有功能等价的对应 API，请返回 "无对应实现"。
5. 只返回一个最合适的 API，不要返回多个候选。

【MindSpore API 命名空间参考】
- 基础数学运算/张量操作：mindspore.ops.xxx（如 mindspore.ops.abs, mindspore.ops.add, mindspore.ops.matmul）
- 也可使用 mindspore.Tensor 的方法（如 Tensor.abs(), Tensor.add()）
- 神经网络层（类）：mindspore.nn.XXX（如 mindspore.nn.Dense, mindspore.nn.Conv2d, mindspore.nn.ReLU）
- 损失函数（类）：mindspore.nn.XXX（如 mindspore.nn.CrossEntropyLoss, mindspore.nn.MSELoss）
- 线性代数：mindspore.ops.xxx 或 mindspore.scipy.linalg.xxx
- 随机数：mindspore.ops.standard_normal, mindspore.ops.uniform 等
- 张量创建：mindspore.ops.zeros, mindspore.ops.ones, mindspore.Tensor 等
- Numpy 兼容：mindspore.numpy.xxx（如 mindspore.numpy.array, mindspore.numpy.zeros）
- 数据处理：mindspore.dataset.xxx
- 注意：MindSpore 2.0+ 版本中部分 API 从 mindspore.ops 移至 mindspore（如 mindspore.abs）

【输出格式】
请严格按照以下 JSON 格式输出，不要包含任何其他内容：

```json
{{
    "pytorch_api": "{pytorch_api}",
    "mindspore_api": "<对应的MindSpore API 名称或'无对应实现'>",
    "confidence": "<high/medium/low>",
    "reason": "<简要说明映射理由或为何无对应实现>"
}}
```

注意：
- mindspore_api 字段只填写 API 全名（如 mindspore.ops.abs 或 mindspore.nn.Conv1d），或 "无对应实现"
- confidence 表示你对这个映射的信心程度（85%以上是high，40%-85%是medium，40%以下是low）
- mindspore_api 字段的值一定要是真实存在的 MindSpore API 名称，不能自己编造不存在的 API。
- reason 简要说明映射的理由(一两句话即可，不要太长)
"""
    return prompt


def parse_llm_response(response: str) -> Tuple[str, str, str]:
    """
    解析 LLM 的 JSON 响应
    
    Returns:
        (mindspore_api, confidence, reason)
    """
    try:
        # 尝试提取 JSON 块
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
    
    # 如果解析失败，尝试简单提取
    if "无对应实现" in response:
        return "无对应实现", "unknown", "解析失败，但检测到无对应实现"
    
    # 尝试查找 mindspore. 开头的 API
    import re
    ms_pattern = r'(mindspore\.[a-zA-Z_][a-zA-Z0-9_\.]*)'
    matches = re.findall(ms_pattern, response)
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
    调用 LLM 获取对应的 MindSpore API
    
    Args:
        client: LLM 客户端
        pytorch_api: PyTorch API 名称
        model: LLM 模型名称
        temperature: 模型温度参数，越低输出越确定性（0.0-1.0）
        max_retries: 最大重试次数
    
    Returns:
        (mindspore_api, full_response)
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
            
            ms_api, confidence, reason = parse_llm_response(full_response)
            return ms_api, full_response
            
        except Exception as e:
            print(f"[WARN] API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            continue
    
    return "无对应实现", f"[ERROR] 调用失败，已重试 {max_retries} 次"


def save_llm_log(
    log_entries: List[dict],
    log_path: Path,
) -> None:
    """保存 LLM 日志到文件"""
    with log_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PyTorch to MindSpore API Mapping - LLM Log\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总条数: {len(log_entries)}\n")
        f.write("=" * 80 + "\n\n")
        
        for entry in log_entries:
            f.write("-" * 60 + "\n")
            f.write(f"序号: {entry['index']}\n")
            f.write(f"PyTorch API: {entry['pytorch_api']}\n")
            f.write(f"API 级别: {entry['api_level']}\n")
            f.write(f"MindSpore API (提取结果): {entry['mindspore_api']}\n")
            f.write(f"\n【LLM 完整输出】\n{entry['llm_response']}\n")
            f.write("-" * 60 + "\n\n")


def save_updated_csv(
    csv_path: Path,
    output_path: Path,
    api_mappings: List[Tuple[str, str]],
) -> None:
    """
    保存更新后的 CSV 文件，保留原有列，仅新增/更新 mindspore-api 列
    """
    # 读取原文件的完整数据
    fieldnames, rows = load_csv_data(csv_path)
    
    # 确保 mindspore-api 列存在
    target_col = "mindspore-api"
    if target_col not in fieldnames:
        fieldnames.append(target_col)
    
    # 构建 pytorch-api -> mindspore-api 的映射字典
    mapping_dict = {pt_api: ms_api for pt_api, ms_api in api_mappings}
    
    # 更新每行数据
    for row in rows:
        pt_api = row.get("pytorch-api", "").strip()
        if pt_api in mapping_dict:
            row[target_col] = mapping_dict[pt_api]
    
    # 写入更新后的数据
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """命令行入口：批量提取 PyTorch API 对应的 MindSpore API"""
    parser = argparse.ArgumentParser(
        description="基于 LLM 提取 PyTorch API 对应的 MindSpore API 映射"
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
        help="每次 API 调用之间的延迟秒数（默认 0.5）",
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
    except Exception as e:
        print(f"[ERROR] 无法初始化 LLM 客户端: {e}")
        return

    # 存储结果
    api_mappings: List[Tuple[str, str]] = []
    log_entries: List[dict] = []

    # 如果从中间开始，先填充前面的 API（保持为空或从已有文件读取）
    if start_idx > 0:
        for i in range(start_idx):
            api_mappings.append((pytorch_apis[i], ""))  # 前面的暂时留空

    # 逐个处理 API
    for i, pt_api in enumerate(apis_to_process, start=start_idx):
        api_level = determine_api_level(pt_api)
        print(f"[INFO] 处理 [{i + 1}/{total_apis}] {pt_api} (级别: {api_level})")
        
        ms_api, llm_response = query_llm_for_api(
            client,
            pt_api,
            model=args.model,
            temperature=args.temperature,
        )
        
        api_mappings.append((pt_api, ms_api))
        log_entries.append({
            "index": i + 1,
            "pytorch_api": pt_api,
            "api_level": api_level,
            "mindspore_api": ms_api,
            "llm_response": llm_response,
        })
        
        print(f"       -> {ms_api}")
        
        # 延迟以避免 API 限流
        if args.delay > 0 and i < end_idx - 1:
            time.sleep(args.delay)

    # 如果有剩余的 API（limit 导致未处理完），填充空值
    if end_idx < total_apis:
        for i in range(end_idx, total_apis):
            api_mappings.append((pytorch_apis[i], ""))

    # 保存结果
    save_updated_csv(input_path, output_path, api_mappings)
    print(f"[SUCCESS] API 映射已保存到: {output_path}")

    # 保存 LLM 日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"pt_ms_mapping_log_{timestamp}.txt"
    log_path = log_dir / log_filename
    save_llm_log(log_entries, log_path)
    print(f"[SUCCESS] LLM 日志已保存到: {log_path}")

    # 统计信息
    mapped_count = sum(1 for _, ms in api_mappings if ms and ms != "无对应实现")
    no_impl_count = sum(1 for _, ms in api_mappings if ms == "无对应实现")
    empty_count = sum(1 for _, ms in api_mappings if not ms)
    
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
