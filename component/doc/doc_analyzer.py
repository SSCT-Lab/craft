# ./component/doc_analyzer.py
"""使用大模型分析测试问题，结合官方文档"""
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from component.doc.doc_crawler_factory import get_doc_content, detect_framework
from component.migration.migrate_generate_tests import get_qwen_client, load_api_key

DEFAULT_MODEL = "qwen-flash"
DEFAULT_KEY_PATH = "aliyun.key"


def build_analysis_prompt(
    error_message: str,
    tf_code: str,
    pt_code: str,
    tf_docs: List[str],
    pt_docs: List[str],
    tf_output: Optional[str] = None,
    pt_output: Optional[str] = None,
    context: Optional[str] = None,
) -> str:
    """构建分析提示词（结构化为：任务描述 / TF 信息 / PT 信息 / 评估要求）"""
    
    tf_docs_text = "\n\n".join(tf_docs) if tf_docs else "未找到相关 TensorFlow 文档"
    pt_docs_text = "\n\n".join(pt_docs) if pt_docs else "未找到相关 PyTorch 文档"
    
    prompt = f"""你是一个资深的深度学习框架专家，擅长分析 TensorFlow 和 PyTorch 之间的差异。

【任务描述】
我们有一组从 TensorFlow 迁移到 PyTorch 的单元测试，请你根据给出的代码、输入/输出信息以及官方文档，
判断当前的异常或行为差异是否“正常”（例如框架行为本身不同、数值精度差异、边界行为定义不同），
还是因为迁移实现本身存在问题。

【整体错误信息 / 元信息】
下面是本次 case 对应的总体状态与对比元信息（精简后的摘要）：
{error_message}

----------------------------------------
【TensorFlow 测试信息】

1. 代码（TF 侧测试逻辑）：
```python
{tf_code}
```

2. 典型输入 / 使用方式（如果能从代码或文档中推断，请你在分析中明确说明）：
- 根据上面的 TF 代码和文档，总结该测试大致使用了什么输入（形状、数据类型、数值范围等）。

3. 官方文档（TensorFlow）：
{tf_docs_text}

4. 实际输出 / 行为（TensorFlow）：
```text
{tf_output or "（未提供 TF 执行日志，请根据代码和文档自行推断期望行为）"}
```

----------------------------------------
【PyTorch 测试信息】

1. 代码（迁移后的 PT 测试逻辑）：
```python
{pt_code}
```

2. 典型输入 / 使用方式：
- 根据上面的 PyTorch 代码和文档，总结该测试在 PyTorch 中使用了什么输入（形状、数据类型、数值范围等），
  并说明是否与 TensorFlow 侧保持一致。

3. 官方文档（PyTorch）：
{pt_docs_text}

4. 实际输出 / 行为（PyTorch）：
```text
{pt_output or "（未提供 PT 执行日志，请根据代码和文档自行推断行为）"}
```
"""
    
    if context:
        prompt += f"""
----------------------------------------
【额外上下文】
{context}
"""
    
    prompt += """
----------------------------------------
【评估要求】
请你基于以上信息进行严谨的技术评估，并回答下面几个问题：

1. 行为是否正常：
   - 当前 TF 与 PT 的差异（或者 PT 的异常）是否可以视为“正常的框架行为差异”（例如 API 语义差异、默认参数不同等）？
   - 如果是“正常差异”，请清楚说明差异点、引用相应文档段落（用自然语言概述即可），并说明是否需要在测试中显式容忍这种差异。

2. 是否存在迁移错误：
   - 如果你认为是迁移实现本身有问题（例如输入构造不一致、维度/dtype 错误、断言写法不当），
     请指出具体问题位置（引用关键代码片段即可），并说明正确的迁移方式应该是什么。

3. 结论标签（请在答案中明确写出）：
   - 用一句话给出整体判断，例如：
     - 「结论：正常差异，无需修改实现，只需在对比/报告中解释原因」
     - 「结论：迁移实现有问题，需要修改 PyTorch 侧代码」
     - 「结论：信息不足，无法判断，需要补充更多日志或文档」
"""
    
    return prompt


def analyze_with_llm(
    client,
    error_message: str,
    tf_code: str,
    pt_code: str,
    tf_apis: List[str],
    pt_apis: List[str],
    tf_output: Optional[str] = None,
    pt_output: Optional[str] = None,
    context: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Optional[str]:
    """使用 LLM 分析问题"""
    
    # 爬取相关文档
    print(f"[INFO] 正在爬取文档...")
    tf_docs = []
    pt_docs = []
    
    for api in tf_apis[:5]:  # 最多爬取5个相关 API 的文档
        try:
            doc_content = get_doc_content(api, "tensorflow")
            if doc_content and "无法获取" not in doc_content:
                tf_docs.append(doc_content)
        except Exception as e:
            print(f"[WARN] 爬取 TF 文档失败 {api}: {e}")
    
    for api in pt_apis[:5]:
        try:
            doc_content = get_doc_content(api, "pytorch")
            if doc_content and "无法获取" not in doc_content:
                pt_docs.append(doc_content)
        except Exception as e:
            print(f"[WARN] 爬取 PT 文档失败 {api}: {e}")
    
    print(f"[INFO] 已获取 {len(tf_docs)} 个 TF 文档，{len(pt_docs)} 个 PT 文档")
    
    # 构建提示词（TF / PT 输出分开给进去）
    prompt = build_analysis_prompt(
        error_message=error_message,
        tf_code=tf_code,
        pt_code=pt_code,
        tf_docs=tf_docs,
        pt_docs=pt_docs,
        tf_output=tf_output,
        pt_output=pt_output,
        context=context,
    )
    
    # 调用 LLM
    try:
        if hasattr(client, 'chat'):
            # 新版本
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048
            )
            analysis = resp.choices[0].message.content.strip()
        else:
            # 旧版本
            resp = client.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048
            )
            analysis = resp.choices[0].message.content.strip()
        
        return analysis
    except Exception as e:
        print(f"[ERROR] LLM 调用失败: {e}")
        return None


def analyze_test_error(
    error_message: str,
    test_file: str,
    tf_apis: Optional[List[str]] = None,
    pt_apis: Optional[List[str]] = None,
    tf_output: Optional[str] = None,
    pt_output: Optional[str] = None,
    context: Optional[str] = None,
) -> Optional[str]:
    """分析测试错误"""
    
    # 读取测试文件
    test_path = Path(test_file)
    if not test_path.exists():
        print(f"[ERROR] 测试文件不存在: {test_file}")
        return None
    
    test_content = test_path.read_text(encoding='utf-8')
    
    # 提取 TF 和 PT 代码（简单提取，可以根据需要改进）
    import re
    
    # 提取 TF 测试函数
    tf_match = re.search(r'def\s+(test\w+)\(\):.*?(?=def\s+test.*?_pt|# ===== PyTorch|if __name__)', 
                         test_content, re.DOTALL)
    tf_code = tf_match.group(0) if tf_match else ""
    
    # 提取 PT 测试函数
    pt_match = re.search(r'def\s+(test.*?_pt)\(\):.*?(?=def\s+test.*?|# ===== Main|if __name__)', 
                         test_content, re.DOTALL)
    pt_code = pt_match.group(0) if pt_match else ""
    
    # 如果没有提供 API 列表，尝试从代码中提取
    if not tf_apis:
        tf_apis = re.findall(r'tf\.\w+(?:\.\w+)*', tf_code)
    if not pt_apis:
        pt_apis = re.findall(r'torch\.\w+(?:\.\w+)*', pt_code)
    
    # 初始化 LLM 客户端
    try:
        client = get_qwen_client(DEFAULT_KEY_PATH)
    except Exception as e:
        print(f"[ERROR] 无法初始化 LLM 客户端: {e}")
        return None
    
    # 调用分析（带上 TF / PT 输出片段）
    return analyze_with_llm(
        client=client,
        error_message=error_message,
        tf_code=tf_code,
        pt_code=pt_code,
        tf_apis=tf_apis,
        pt_apis=pt_apis,
        tf_output=tf_output,
        pt_output=pt_output,
        context=context,
    )


def main():
    """命令行工具"""
    parser = argparse.ArgumentParser(description="分析测试迁移问题")
    parser.add_argument("error", help="错误信息")
    parser.add_argument("--test-file", "-t", required=True, help="测试文件路径")
    parser.add_argument("--tf-apis", nargs="+", help="TensorFlow API 列表")
    parser.add_argument("--pt-apis", nargs="+", help="PyTorch API 列表")
    parser.add_argument("--context", "-c", help="额外上下文信息")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="LLM 模型名称")
    parser.add_argument("--key-path", "-k", default=DEFAULT_KEY_PATH, help="API key 路径")
    parser.add_argument("--output", "-o", help="输出文件路径")
    
    args = parser.parse_args()
    
    # 分析问题
    analysis = analyze_test_error(
        args.error,
        args.test_file,
        args.tf_apis,
        args.pt_apis,
        args.context
    )
    
    if analysis:
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(analysis)
            print(f"[SUCCESS] 分析结果已保存到: {args.output}")
        else:
            print("\n" + "=" * 80)
            print("分析结果")
            print("=" * 80)
            print(analysis)
    else:
        print("[ERROR] 分析失败")


if __name__ == "__main__":
    main()

