# TF↔PT 差分测试工作流

基于 LLM 的 TensorFlow 与 PyTorch 算子差分测试框架，从 TensorFlow 官方测试文件出发，提取测试用例并与 PyTorch 进行跨框架对等性验证。

---

## 目录

- [0. 环境准备](#0-环境准备)
- [1. Step 1 - 提取 TF API 列表](#1-step-1---提取-tf-api-列表)
- [1.5 Step 1.5 - 过滤不存在的 TF API](#15-step-15---过滤不存在的-tf-api)
- [2. Step 2 - 用 LLM 提取测试用例](#2-step-2---用-llm-提取测试用例)
- [3. Step 3 - 生成 TF→PT 映射](#3-step-3---生成-tfpt-映射)
- [3.5 Step 3.5 - 映射后筛选与验证](#35-step-35---映射后筛选与验证)
- [4. Step 4 - 差分测试主流程](#4-step-4---差分测试主流程)
- [5. Step 5 - 结果分析](#5-step-5---结果分析)
- [6. 完整快速开始](#6-完整快速开始)
- [7. 目录结构](#7-目录结构)

---

## 0. 环境准备

### 0.1 激活 Python 环境

```bash
conda activate tf_env
```

### 0.2 确保 API Key 文件

在项目根目录确保 `aliyun.key` 存在，包含阿里云 DashScope API Key：

```
sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 0.3 确保依赖

```bash
pip install openai pymongo tqdm numpy torch tensorflow
```

### 0.4 确保测试文件目录

确保项目根目录下有 `tf_testcases/` 目录，包含从 TensorFlow 官方仓库下载的测试文件。

---

## 1. Step 1 - 提取 TF API 列表

**脚本**: `tf_pt_test/extract_tf_apis.py`  
**功能**: 从 `tf_testcases/` 的测试文件中提取被测试的 TF 公开 API 列表  
**方式**: 预定义映射 + AST 辅助（无需 LLM，速度快）

默认会自动发现 `tf_testcases/` 下的所有子目录进行扫描（排除 `__pycache__` 和 `v1_compat_tests`）。

### 基本用法

```bash
python tf_pt_test/extract_tf_apis.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tf-dir` | `tf_testcases` | TF 测试文件目录 |
| `--output` / `-o` | `tf_pt_test/data/tf_apis_new.json` | 输出路径 |
| `--categories` / `-c` | 自动发现所有子目录 | 指定扫描的子目录 |

### 示例

```bash
# 只扫描 nn_ops 和 math_ops
python tf_pt_test/extract_tf_apis.py -c nn_ops math_ops

# 指定输出路径
python tf_pt_test/extract_tf_apis.py -o tf_pt_test/data/my_apis.json
```

### 输出

`tf_pt_test/data/tf_apis_new.json`，格式：
```json
{
  "total_apis": 150,
  "apis": [
    {
      "tf_api": "tf.nn.relu",
      "source_file": "nn_ops/relu_op_test.py",
      "category": "nn_ops",
      "extraction_method": "predefined"
    }
  ]
}
```

---

## 1.5 Step 1.5 - 过滤不存在的 TF API

**脚本**: `tf_pt_test/filter_existing_tf_apis.py`  
**功能**: 访问 TensorFlow 官方文档页面，过滤掉不存在或重定向异常的 API  
**输出**: 仅保留真实存在的 API 列表

### 基本用法

```bash
python tf_pt_test/filter_existing_tf_apis.py \
    --input tf_pt_test/data/tf_apis_new.json \
    --output tf_pt_test/data/tf_apis_existing.json
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` / `-i` | `tf_pt_test/data/tf_apis_new.json` | Step 1 输出文件 |
| `--output` / `-o` | `tf_pt_test/data/tf_apis_existing.json` | 过滤后输出文件 |
| `--delay` | `0.5` | 每次请求延迟（秒） |
| `--min-page-chars` | `2000` | 页面最小字符数阈值 |
| `--min-desc-chars` | `50` | 描述最小字符数阈值 |

### 输出

`tf_pt_test/data/tf_apis_existing.json`，格式：
```json
{
  "total_apis": 120,
  "filtered_out": 30,
  "apis": [
    {
      "tf_api": "tf.nn.relu",
      "source_file": "nn_ops/relu_op_test.py",
      "category": "nn_ops",
      "extraction_method": "predefined"
    }
  ],
  "invalid_apis": [
    {
      "api": "tf.some_missing_api",
      "reason": "non_200_status"
    }
  ]
}
```

---

## 2. Step 2 - 用 LLM 提取测试用例

**脚本**: `tf_pt_test/extract_tf_test_cases.py`  
**功能**: 对每个 TF API，读取对应测试文件内容，调用 LLM 生成标准化测试用例  
**支持**: 并发调用、断点续传

### 基本用法

```bash
python tf_pt_test/extract_tf_test_cases.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` / `-i` | `tf_pt_test/data/tf_apis_existing.json` | Step 1.5 的输出文件 |
| `--output` / `-o` | `tf_pt_test/data/tf_test_cases.json` | 输出路径 |
| `--workers` / `-w` | `4` | LLM 并发线程数 |
| `--num-cases` / `-n` | `5` | 每个 API 提取的用例数 |
| `--model` / `-m` | `qwen-plus` | LLM 模型 |
| `--key-path` / `-k` | `aliyun.key` | API Key 路径 |
| `--start` | `0` | 开始位置（断点续传） |
| `--limit` | `None` | 最多处理数量 |
| `--delay` | `0.5` | 调用间隔（秒） |

### 示例

```bash
# 4线程并发，每个API生成5个用例
python tf_pt_test/extract_tf_test_cases.py -w 4 -n 5 \
  --input tf_pt_test/data/tf_apis_existing.json

# 从第50个API开始，处理100个
python tf_pt_test/extract_tf_test_cases.py --start 50 --limit 100

# 使用更强的模型
python tf_pt_test/extract_tf_test_cases.py -m qwen-max
```

### 输出

`tf_pt_test/data/tf_test_cases.json`，格式：
```json
{
  "total_apis": 150,
  "test_cases": {
    "tf.nn.relu": {
      "api": "tf.nn.relu",
      "is_class_api": false,
      "test_cases": [
        {
          "description": "基本功能测试",
          "inputs": {
            "x": {"shape": [2, 3], "dtype": "float32"}
          }
        }
      ],
      "source_file": "nn_ops/relu_op_test.py",
      "category": "nn_ops"
    }
  }
}
```

---

## 3. Step 3 - 生成 TF→PT 映射

**脚本**: `tf_pt_test/extract_tf_pt_mapping.py`  
**功能**: 对每个 TF API，调用 LLM 查找功能等价的 PyTorch API  
**支持**: 并发调用、断点续传

### 基本用法

```bash
python tf_pt_test/extract_tf_pt_mapping.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` / `-i` | `tf_pt_test/data/tf_apis_existing.json` | Step 1.5 的输出文件 |
| `--output` / `-o` | `tf_pt_test/data/tf_pt_mapping.csv` | 输出 CSV 路径 |
| `--workers` / `-w` | `4` | LLM 并发线程数 |
| `--model` / `-m` | `qwen-plus` | LLM 模型 |
| `--temperature` / `-t` | `0.1` | 温度参数（越低越确定） |
| `--start` | `0` | 开始位置 |
| `--limit` | `None` | 最多处理数量 |
| `--delay` | `0.5` | 调用间隔 |

### 示例

```bash
# 4线程并发
python tf_pt_test/extract_tf_pt_mapping.py -w 4

# 低温度获得更确定的结果
python tf_pt_test/extract_tf_pt_mapping.py -t 0.05

# 分批处理：前100个
python tf_pt_test/extract_tf_pt_mapping.py --limit 100
# 分批处理：100-200
python tf_pt_test/extract_tf_pt_mapping.py --start 100 --limit 100
```

### 输出

`tf_pt_test/data/tf_pt_mapping.csv`，格式：
```csv
tensorflow-api,pytorch-api,confidence,reason
tf.nn.relu,torch.nn.functional.relu,high,功能完全等价
tf.math.abs,torch.abs,high,功能完全等价
tf.nn.ctc_loss,无对应实现,low,PyTorch中ctc_loss参数差异较大
```

---

## 3.5 Step 3.5 - 映射后筛选与验证

**脚本 1**: `tf_pt_test/filter_high_confidence_mapping.py`  
**功能**: 仅保留置信度为 high 的映射记录

```bash
python tf_pt_test/filter_high_confidence_mapping.py \
  --input tf_pt_test/data/tf_pt_mapping.csv \
  --output tf_pt_test/data/tf_pt_mapping_high.csv
```

**脚本 2**: `tf_pt_test/validate_pt_api_docs.py`  
**功能**: 访问 PyTorch 官方文档，验证映射的 API 是否真实存在，不存在则置为“无对应实现”

```bash
python tf_pt_test/validate_pt_api_docs.py \
  --input tf_pt_test/data/tf_pt_mapping_high.csv \
  --output tf_pt_test/data/tf_pt_mapping_validated.csv
```

### 输出

- `tf_pt_test/data/tf_pt_mapping_high.csv`：仅保留 high 置信度的映射
- `tf_pt_test/data/tf_pt_mapping_validated.csv`：通过 PyTorch 文档验证后的映射

---

## 4. Step 4 - 差分测试主流程

**脚本**: `tf_pt_test/llm_enhanced_compare.py`  
**功能**: 核心差分测试框架。对每对等价算子执行 TF 和 PT，比较结果，用 LLM 进行修复/变异/跳过  
**支持**: 并发测试、批量处理、结果保存

### 基本用法

```bash
python tf_pt_test/llm_enhanced_compare.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-iterations` / `-m` | `3` | 每个用例的最大迭代次数 |
| `--num-cases` / `-n` | `3` | 每个算子测试的用例数 |
| `--start` | `1` | 起始算子索引（从1开始） |
| `--end` | `None` | 结束算子索引（包含） |
| `--operators` / `-o` | `None` | 指定测试的算子名称列表 |
| `--workers` / `-w` | `4` | 并发线程数 |
| `--model` | `qwen-plus` | LLM 模型 |
| `--key-path` / `-k` | `aliyun.key` | API Key 路径 |
| `--test-cases-file` | `data/tf_test_cases.json` | 测试用例文件 |
| `--mapping-file` | `data/tf_pt_mapping.csv` | 映射文件（推荐使用验证后的 `tf_pt_mapping_validated.csv`） |

### 示例

```bash
# 测试前10个算子，每个3个用例，3次迭代
python tf_pt_test/llm_enhanced_compare.py --start 1 --end 10 -n 3 -m 3

# 测试指定算子
python tf_pt_test/llm_enhanced_compare.py -o tf.nn.relu tf.math.abs tf.concat

# 4线程并发，使用 qwen-max 模型
python tf_pt_test/llm_enhanced_compare.py -w 4 --model qwen-max

# 测试所有算子（可能耗时较长）
python tf_pt_test/llm_enhanced_compare.py -w 4 -n 3 -m 3
```

### 核心迭代逻辑

```
初始测试用例（从文件提取）
    │
    ▼
┌─ 执行 TF 和 PT ──────────────────────────────┐
│  使用相同的 numpy 数据作为输入                │
│  分别在 TF 和 PT 中执行算子                   │
│  比较输出结果（allclose, tolerance=1e-5）     │
└───────────────────────────────────────────────┘
    │
    ▼
┌─ 调用 LLM 分析结果 ──────────────────────────┐
│  1. 一致 → 变异（mutation）：探索边界值       │
│  2. 出错 → 修复（repair）：调整参数          │
│  3. 不等价 → 跳过（skip）：终止迭代          │
└───────────────────────────────────────────────┘
    │
    ▼
  更新测试用例 → 回到"执行"步骤（最多 N 轮）
```

### 输出

- `tf_pt_test/tf_pt_log_1/llm_enhanced_<api>_<timestamp>.json` - 每个算子的详细结果
- `tf_pt_test/tf_pt_log_1/batch_test_log_<timestamp>.txt` - 批量测试日志
- `tf_pt_test/tf_pt_log_1/batch_test_summary_<timestamp>.json` - JSON 摘要

---

## 5. Step 5 - 结果分析

**脚本**: `tf_pt_test/analyze_results.py`  
**功能**: 分析 Step 4 的测试结果，生成统计报告、CSV 和 JSON

### 基本用法

```bash
python tf_pt_test/analyze_results.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--result-dir` / `-r` | `tf_pt_test/tf_pt_log_1` | 结果目录 |
| `--output` / `-o` | `tf_pt_test/analysis` | 分析输出目录 |

### 输出

- `tf_pt_test/analysis/analysis_report_<timestamp>.txt` - 详细文字报告
- `tf_pt_test/analysis/analysis_summary_<timestamp>.csv` - CSV 摘要
- `tf_pt_test/analysis/analysis_data_<timestamp>.json` - JSON 数据

---

## 6. 完整快速开始

以下是从零开始的完整命令序列：

```bash
# 0. 激活环境
conda activate tf_env

# 1. 提取 TF API 列表（无需 LLM，秒级完成）
python tf_pt_test/extract_tf_apis.py

# 1.5 过滤不存在的 TF API（基于官方文档）
python tf_pt_test/filter_existing_tf_apis.py \
  --input tf_pt_test/data/tf_apis_new.json \
  --output tf_pt_test/data/tf_apis_existing.json

# 2. 用 LLM 提取测试用例（需要 LLM，4线程并发）
python tf_pt_test/extract_tf_test_cases.py -w 4 -n 5 \
  --input tf_pt_test/data/tf_apis_existing.json

# 3. 生成 TF→PT 映射（需要 LLM，4线程并发）
python tf_pt_test/extract_tf_pt_mapping.py -w 4

# 3.5 筛选 high 置信度映射 + 验证 PyTorch 文档存在性
python tf_pt_test/filter_high_confidence_mapping.py \
  --input tf_pt_test/data/tf_pt_mapping.csv \
  --output tf_pt_test/data/tf_pt_mapping_high.csv
python tf_pt_test/validate_pt_api_docs.py \
  --input tf_pt_test/data/tf_pt_mapping_high.csv \
  --output tf_pt_test/data/tf_pt_mapping_validated.csv

# 4. 运行差分测试（核心步骤，需要 LLM）
#    建议先测试少量算子验证流程
python tf_pt_test/llm_enhanced_compare.py --start 1 --end 5 -n 3 -m 3 -w 4

#    确认无误后测试全部
python tf_pt_test/llm_enhanced_compare.py -w 4 -n 3 -m 3

# 5. 分析结果
python tf_pt_test/analyze_results.py
```

### 分批处理大量 API（推荐）

```bash
# 第一批：前50个
python tf_pt_test/llm_enhanced_compare.py --start 1 --end 50 -w 4

# 第二批：51-100
python tf_pt_test/llm_enhanced_compare.py --start 51 --end 100 -w 4

# 第三批：101-150
python tf_pt_test/llm_enhanced_compare.py --start 101 --end 150 -w 4

# 最后分析全部结果
python tf_pt_test/analyze_results.py
```

---

## 7. 目录结构

```
tf_pt_test/
├── extract_tf_apis.py           # Step 1: 提取 TF API 列表
├── extract_tf_test_cases.py     # Step 2: LLM 提取测试用例
├── extract_tf_pt_mapping.py     # Step 3: LLM 生成 TF→PT 映射
├── llm_enhanced_compare.py      # Step 4: 差分测试主框架
├── analyze_results.py           # Step 5: 结果分析
├── WORKFLOW.md                  # 本文档
├── data/                        # 数据文件
│   ├── tf_apis_new.json             # Step 1 输出
│   ├── tf_apis_existing.json        # Step 1.5 输出
│   ├── tf_test_cases.json       # Step 2 输出
│   ├── tf_pt_mapping.csv        # Step 3 输出
│   ├── tf_pt_mapping_high.csv       # Step 3.5 输出（高置信度）
│   └── tf_pt_mapping_validated.csv  # Step 3.5 输出（文档验证）
├── tf_pt_log_1/                 # Step 4 输出（测试结果）
│   ├── llm_enhanced_*.json      # 每个算子的详细结果
│   ├── batch_test_log_*.txt     # 批量日志
│   └── batch_test_summary_*.json # JSON 摘要
└── analysis/                    # Step 5 输出（分析报告）
    ├── analysis_report_*.txt    # 文字报告
    ├── analysis_summary_*.csv   # CSV 摘要
    └── analysis_data_*.json     # JSON 数据
```

---

## 8. 常见问题

### Q1: Step 2 太慢，如何加速？

- 增加并发线程数：`-w 8`
- 减少每个 API 的用例数：`-n 3`
- 减小 delay：`--delay 0.2`

### Q2: 如何处理 LLM 调用失败？

所有脚本（Step 2/3/4）都内置了重试机制和断点续传：
- 重试：默认 3 次指数退避重试
- 续传：Step 2/3 会自动跳过已处理的 API，重新运行相同命令即可

### Q3: 如何选择合适的算子范围？

建议从核心算子开始：
```bash
# 只测试 nn_ops 中的算子
python tf_pt_test/extract_tf_apis.py -c nn_ops
python tf_pt_test/extract_tf_test_cases.py
python tf_pt_test/extract_tf_pt_mapping.py
python tf_pt_test/llm_enhanced_compare.py
```

### Q4: 如何指定测试特定算子？

```bash
python tf_pt_test/llm_enhanced_compare.py -o tf.nn.relu tf.math.abs tf.concat
```

### Q5: Step 2 和 Step 3 可以并行运行吗？

可以！Step 2（提取测试用例）和 Step 3（生成映射）都只依赖 Step 1 的输出，互不依赖，可以同时在两个终端中运行。
