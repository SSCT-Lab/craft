# PD↔PT 差分测试工作流

基于 LLM 的 PaddlePaddle 与 PyTorch 算子差分测试框架，从 PaddlePaddle 官方测试文件出发，提取测试用例并与 PyTorch 进行跨框架对等性验证。

---

## 目录

- [0. 环境准备](#0-环境准备)
- [1. Step 1 - 提取 Paddle API 列表](#1-step-1---提取-paddle-api-列表)
- [1.5 Step 1.5 - 过滤不存在的 Paddle API](#15-step-15---过滤不存在的-paddle-api)
- [2. Step 2 - 用 LLM 提取测试用例](#2-step-2---用-llm-提取测试用例)
- [3. Step 3 - 生成 PD→PT 映射](#3-step-3---生成-pdpt-映射)
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
pip install openai pymongo tqdm numpy torch paddlepaddle
```

### 0.4 确保测试文件目录

确保项目根目录下有 `testcases_pd/` 目录，包含从 PaddlePaddle 官方仓库下载的测试文件（475+ 个 `test_*_op.py` 文件）。

---

## 1. Step 1 - 提取 Paddle API 列表

**脚本**: `pd_pt_test/extract_pd_apis.py`  
**功能**: 从 `testcases_pd/` 的测试文件中提取被测试的 Paddle 公开 API 列表  
**方式**: 预定义映射 + 正则提取 `self.python_api` + 文件名推断（无需 LLM，速度快）

### 基本用法

```bash
python pd_pt_test/extract_pd_apis.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pd-dir` | `testcases_pd` | Paddle 测试文件目录 |
| `--output` / `-o` | `pd_pt_test/data/pd_apis_new.json` | 输出路径 |

### 输出

`pd_pt_test/data/pd_apis_new.json`，格式：
```json
{
  "total_apis": 200,
  "apis": [
    {
      "pd_api": "paddle.abs",
      "source_file": "test_abs_op.py",
      "extraction_method": "predefined"
    }
  ]
}
```

---

## 1.5 Step 1.5 - 过滤不存在的 Paddle API

**脚本**: `pd_pt_test/filter_existing_pd_apis.py`  
**功能**: 通过 PaddlePaddle 官方文档验证每个 API 是否真实存在  
**输出**: 仅保留真实存在的 API 列表

### 基本用法

```bash
python pd_pt_test/filter_existing_pd_apis.py \
    --input pd_pt_test/data/pd_apis_new.json \
    --output pd_pt_test/data/pd_apis_existing.json
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` / `-i` | `pd_pt_test/data/pd_apis_new.json` | Step 1 输出文件 |
| `--output` / `-o` | `pd_pt_test/data/pd_apis_existing.json` | 过滤后输出文件 |
| `--delay` | `0.5` | 每次请求延迟（秒） |

### 输出

`pd_pt_test/data/pd_apis_existing.json`，格式：
```json
{
  "total_apis": 160,
  "filtered_out": 40,
  "apis": [
    {
      "pd_api": "paddle.abs",
      "source_file": "test_abs_op.py",
      "extraction_method": "predefined"
    }
  ],
  "invalid_apis": [
    {
      "api": "paddle.some_missing_api",
      "reason": "doc_not_found"
    }
  ]
}
```

---

## 2. Step 2 - 用 LLM 提取测试用例

**脚本**: `pd_pt_test/extract_pd_test_cases.py`  
**功能**: 对每个 Paddle API，读取对应测试文件内容，调用 LLM 生成标准化测试用例  
**支持**: 并发调用、断点续传

Paddle 测试文件的两种典型模式：
- **OpTest 模式**: 继承 `OpTest`，使用 `self.op_type`、`self.python_api`、`self.inputs`、`self.outputs`、`self.attrs`
- **unittest 模式**: 继承 `unittest.TestCase`，直接调用 `paddle.xxx()`

### 基本用法

```bash
python pd_pt_test/extract_pd_test_cases.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` / `-i` | `pd_pt_test/data/pd_apis_existing.json` | Step 1.5 的输出文件 |
| `--output` / `-o` | `pd_pt_test/data/pd_test_cases.json` | 输出路径 |
| `--workers` / `-w` | `4` | LLM 并发线程数 |
| `--num-cases` / `-n` | `5` | 每个 API 提取的用例数 |
| `--model` / `-m` | `qwen-plus` | LLM 模型 |
| `--key-path` / `-k` | `aliyun.key` | API Key 路径 |
| `--start` | `0` | 开始位置（断点续传） |
| `--limit` | `None` | 最多处理数量 |

### 示例

```bash
# 4线程并发，每个API生成5个用例
python pd_pt_test/extract_pd_test_cases.py -w 4 -n 5 \
  --input pd_pt_test/data/pd_apis_existing.json

# 从第50个API开始，处理100个
python pd_pt_test/extract_pd_test_cases.py --start 50 --limit 100
```

### 输出

`pd_pt_test/data/pd_test_cases.json`，格式：
```json
{
  "total_apis": 160,
  "test_cases": {
    "paddle.abs": {
      "api": "paddle.abs",
      "is_class_api": false,
      "test_cases": [
        {
          "description": "基本功能测试",
          "inputs": {
            "x": {"shape": [2, 3], "dtype": "float32"}
          }
        }
      ],
      "source_file": "test_abs_op.py"
    }
  }
}
```

---

## 3. Step 3 - 生成 PD→PT 映射

**脚本**: `pd_pt_test/extract_pd_pt_mapping.py`  
**功能**: 对每个 Paddle API，调用 LLM 查找功能等价的 PyTorch API  
**支持**: 并发调用、断点续传

### 基本用法

```bash
python pd_pt_test/extract_pd_pt_mapping.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` / `-i` | `pd_pt_test/data/pd_apis_existing.json` | Step 1.5 的输出文件 |
| `--output` / `-o` | `pd_pt_test/data/pd_pt_mapping.csv` | 输出 CSV 路径 |
| `--workers` / `-w` | `4` | LLM 并发线程数 |
| `--model` / `-m` | `qwen-plus` | LLM 模型 |
| `--start` | `0` | 开始位置 |
| `--limit` | `None` | 最多处理数量 |

### 示例

```bash
# 4线程并发
python pd_pt_test/extract_pd_pt_mapping.py -w 4

# 分批处理：前100个
python pd_pt_test/extract_pd_pt_mapping.py --limit 100
# 分批处理：100-200
python pd_pt_test/extract_pd_pt_mapping.py --start 100 --limit 100
```

### 输出

`pd_pt_test/data/pd_pt_mapping.csv`，格式：
```csv
paddle-api,pytorch-api,confidence,reason
paddle.abs,torch.abs,high,功能完全等价
paddle.nn.ReLU,torch.nn.ReLU,high,功能完全等价
paddle.nn.functional.ctc_loss,无对应实现,low,PyTorch中ctc_loss参数差异较大
```

---

## 3.5 Step 3.5 - 映射后筛选与验证

**脚本 1**: `pd_pt_test/filter_high_confidence_mapping.py`  
**功能**: 仅保留置信度为 high 的映射记录

```bash
python pd_pt_test/filter_high_confidence_mapping.py \
  --input pd_pt_test/data/pd_pt_mapping.csv \
  --output pd_pt_test/data/pd_pt_mapping_high.csv
```

**脚本 2**: `pd_pt_test/validate_pt_api_docs.py`  
**功能**: 访问 PyTorch 官方文档，验证映射的 API 是否真实存在，不存在则置为"无对应实现"

```bash
python pd_pt_test/validate_pt_api_docs.py \
  --input pd_pt_test/data/pd_pt_mapping_high.csv \
  --output pd_pt_test/data/pd_pt_mapping_validated.csv
```

### 输出

- `pd_pt_test/data/pd_pt_mapping_high.csv`：仅保留 high 置信度的映射
- `pd_pt_test/data/pd_pt_mapping_validated.csv`：通过 PyTorch 文档验证后的映射

---

## 4. Step 4 - 差分测试主流程

**脚本**: `pd_pt_test/llm_enhanced_compare.py`  
**功能**: 核心差分测试框架。对每对等价算子执行 Paddle 和 PT，比较结果，用 LLM 进行修复/变异/跳过  
**支持**: 并发测试、批量处理、结果保存

### 基本用法

```bash
python pd_pt_test/llm_enhanced_compare.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-iterations` / `-m` | `3` | 每个用例的最大迭代次数 |
| `--num-cases` / `-n` | `5` | 每个算子测试的用例数 |
| `--start` | `1` | 起始算子索引（从1开始） |
| `--end` | `None` | 结束算子索引（包含） |
| `--operators` / `-o` | `None` | 指定测试的算子名称列表 |
| `--workers` / `-w` | `6` | 并发线程数 |
| `--model` | `qwen-plus` | LLM 模型 |
| `--key-path` / `-k` | `aliyun.key` | API Key 路径 |
| `--test-cases-file` | `data/pd_test_cases.json` | 测试用例文件 |
| `--mapping-file` | `data/pd_pt_mapping_validated.csv` | 映射文件 |

### 示例

```bash
# 测试前10个算子，每个5个用例，3次迭代
python pd_pt_test/llm_enhanced_compare.py --start 1 --end 10 -n 5 -m 3

# 测试指定算子
python pd_pt_test/llm_enhanced_compare.py -o paddle.abs paddle.nn.ReLU paddle.concat

# 6线程并发，使用 qwen-max 模型
python pd_pt_test/llm_enhanced_compare.py -w 6 --model qwen-max

# 测试所有算子（可能耗时较长）
python pd_pt_test/llm_enhanced_compare.py -w 6 -n 5 -m 3
```

### 核心迭代逻辑

```
初始测试用例（从文件提取）
    │
    ▼
┌─ 执行 Paddle 和 PT ──────────────────────────┐
│  使用相同的 numpy 数据作为输入                │
│  分别在 Paddle 和 PT 中执行算子               │
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

- `pd_pt_test/pd_pt_log_1/llm_enhanced_<api>_<timestamp>.json` - 每个算子的详细结果
- `pd_pt_test/pd_pt_log_1/batch_test_log_<timestamp>.txt` - 批量测试日志
- `pd_pt_test/pd_pt_log_1/batch_test_summary_<timestamp>.json` - JSON 摘要

---

## 5. Step 5 - 结果分析

**脚本**: `pd_pt_test/analyze_results.py`  
**功能**: 分析 Step 4 的测试结果，生成统计报告、CSV 和 JSON

### 基本用法

```bash
python pd_pt_test/analyze_results.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--result-dir` / `-r` | `pd_pt_test/pd_pt_log_1` | 结果目录 |
| `--output` / `-o` | `pd_pt_test/analysis` | 分析输出目录 |

### 输出

- `pd_pt_test/analysis/analysis_report_<timestamp>.txt` - 详细文字报告
- `pd_pt_test/analysis/analysis_summary_<timestamp>.csv` - CSV 摘要
- `pd_pt_test/analysis/analysis_data_<timestamp>.json` - JSON 数据

---

## 6. 完整快速开始

以下是从零开始的完整命令序列：

```bash
# 0. 激活环境
conda activate tf_env

# 1. 提取 Paddle API 列表（无需 LLM，秒级完成）
python pd_pt_test/extract_pd_apis.py

# 1.5 过滤不存在的 Paddle API（基于官方文档）
python pd_pt_test/filter_existing_pd_apis.py \
  --input pd_pt_test/data/pd_apis_new.json \
  --output pd_pt_test/data/pd_apis_existing.json

# 2. 用 LLM 提取测试用例（需要 LLM，4线程并发）
python pd_pt_test/extract_pd_test_cases.py -w 4 -n 5 \
  --input pd_pt_test/data/pd_apis_existing.json

# 3. 生成 PD→PT 映射（需要 LLM，4线程并发）
python pd_pt_test/extract_pd_pt_mapping.py -w 4

# 3.5 筛选 high 置信度映射 + 验证 PyTorch 文档存在性
python pd_pt_test/filter_high_confidence_mapping.py \
  --input pd_pt_test/data/pd_pt_mapping.csv \
  --output pd_pt_test/data/pd_pt_mapping_high.csv
python pd_pt_test/validate_pt_api_docs.py \
  --input pd_pt_test/data/pd_pt_mapping_high.csv \
  --output pd_pt_test/data/pd_pt_mapping_validated.csv

# 4. 运行差分测试（核心步骤，需要 LLM）
#    建议先测试少量算子验证流程
python pd_pt_test/llm_enhanced_compare.py --start 1 --end 5 -n 5 -m 3 -w 6

#    确认无误后测试全部
python pd_pt_test/llm_enhanced_compare.py -w 6 -n 5 -m 3

# 5. 分析结果
python pd_pt_test/analyze_results.py
```

### 分批处理大量 API（推荐）

```bash
# 第一批：前50个
python pd_pt_test/llm_enhanced_compare.py --start 1 --end 50 -w 6

# 第二批：51-100
python pd_pt_test/llm_enhanced_compare.py --start 51 --end 100 -w 6

# 第三批：101-150
python pd_pt_test/llm_enhanced_compare.py --start 101 --end 150 -w 6

# 最后分析全部结果
python pd_pt_test/analyze_results.py
```

---

## 7. 目录结构

```
pd_pt_test/
├── extract_pd_apis.py               # Step 1: 提取 Paddle API 列表
├── filter_existing_pd_apis.py       # Step 1.5: 过滤不存在的 API
├── extract_pd_test_cases.py         # Step 2: LLM 提取测试用例
├── extract_pd_pt_mapping.py         # Step 3: LLM 生成 PD→PT 映射
├── filter_high_confidence_mapping.py # Step 3.5a: 筛选高置信度映射
├── validate_pt_api_docs.py          # Step 3.5b: 验证 PT 文档存在性
├── llm_enhanced_compare.py          # Step 4: 差分测试主框架
├── analyze_results.py               # Step 5: 结果分析
├── WORKFLOW.md                      # 本文档
├── data/                            # 数据文件
│   ├── pd_apis_new.json                 # Step 1 输出
│   ├── pd_apis_existing.json            # Step 1.5 输出
│   ├── pd_test_cases.json           # Step 2 输出
│   ├── pd_pt_mapping.csv            # Step 3 输出
│   ├── pd_pt_mapping_high.csv           # Step 3.5a 输出（高置信度）
│   └── pd_pt_mapping_validated.csv      # Step 3.5b 输出（文档验证）
├── pd_pt_log_1/                     # Step 4 输出（测试结果）
│   ├── llm_enhanced_*.json          # 每个算子的详细结果
│   ├── batch_test_log_*.txt         # 批量日志
│   └── batch_test_summary_*.json    # JSON 摘要
└── analysis/                        # Step 5 输出（分析报告）
    ├── analysis_report_*.txt        # 文字报告
    ├── analysis_summary_*.csv       # CSV 摘要
    └── analysis_data_*.json         # JSON 数据
```

---

## 8. 与 TF↔PT 框架的差异

| 对比项 | TF↔PT 框架 | PD↔PT 框架 |
|--------|------------|------------|
| 源框架 | TensorFlow | PaddlePaddle |
| 测试文件目录 | `tf_testcases/`（按 category 分子目录） | `testcases_pd/`（所有文件平铺） |
| API 提取方式 | AST + 预定义 | 预定义 + `self.python_api` 正则 + 文件名推断 |
| 张量创建 | `tf.constant()` | `paddle.to_tensor()` |
| 随机种子 | `tf.random.set_seed()` | `paddle.seed()` |
| 数据格式 | TF 默认 NHWC，需做转换 | Paddle 默认 NCHW，与 PyTorch 一致 |
| CSV 列名 | `tensorflow-api,pytorch-api,...` | `paddle-api,pytorch-api,...` |
| 文档爬取 | `get_doc_content(api, "tensorflow")` | `get_doc_content(api, "paddle")` |
| 结果目录 | `tf_pt_log_1/` | `pd_pt_log_1/` |

---

## 9. 常见问题

### Q1: Step 2 太慢，如何加速？

- 增加并发线程数：`-w 8`
- 减少每个 API 的用例数：`-n 3`

### Q2: 如何处理 LLM 调用失败？

所有脚本（Step 2/3/4）都内置了重试机制和断点续传：
- 续传：Step 2/3 会自动跳过已处理的 API，重新运行相同命令即可

### Q3: Paddle 和 PyTorch 默认数据格式一样吗？

是的，PaddlePaddle 和 PyTorch 默认都使用 **NCHW** 格式（与 TensorFlow 的 NHWC 不同），因此不需要做 NHWC→NCHW 的转换，这比 TF↔PT 测试更简单。

### Q4: 如何指定测试特定算子？

```bash
python pd_pt_test/llm_enhanced_compare.py -o paddle.abs paddle.nn.ReLU paddle.concat
```

### Q5: Step 2 和 Step 3 可以并行运行吗？

可以！Step 2（提取测试用例）和 Step 3（生成映射）都只依赖 Step 1 的输出，互不依赖，可以同时在两个终端中运行。
