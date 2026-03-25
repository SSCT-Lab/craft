# PyTorch to TensorFlow API 映射工具使用指南

本文档介绍如何使用 API 映射相关工具进行 PyTorch 到 TensorFlow 的 API 映射提取、验证与修复。

---

## 目录

- [1. 环境准备](#1-环境准备)
- [2. extract_tf_api_mapping.py - API 映射提取工具](#2-extract_tf_api_mappingpy---api-映射提取工具)
- [3. validate_tf_api_mapping.py - API 映射验证工具](#3-validate_tf_api_mappingpy---api-映射验证工具)
- [4. extract_new_found_apis.py - 新发现映射提取工具](#4-extract_new_found_apispy---新发现映射提取工具)
- [5. extract_changed_apis.py - 变更映射提取工具（含文档验证）](#5-extract_changed_apispy---变更映射提取工具含文档验证)
- [6. 完整工作流程示例](#6-完整工作流程示例)

---

## 1. 环境准备

### 1.1 确保 API Key 文件存在

在项目根目录下创建 `aliyun.key` 文件，写入阿里云 DashScope API Key：

```
sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 1.2 激活 Python 环境

```bash
conda activate tf_env
```

---

## 2. extract_tf_api_mapping.py - API 映射提取工具

该工具基于 LLM 从 PyTorch API 列表中提取对应的 TensorFlow API 映射。

### 2.1 参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | `component/data/api_mappings.csv` | 输入的 CSV 文件路径 |
| `--output` | `-o` | 同 input | 输出的 CSV 文件路径 |
| `--model` | `-m` | `qwen-plus` | LLM 模型名称 |
| `--key-path` | `-k` | `aliyun.key` | API Key 文件路径 |
| `--start` | - | `0` | 从第几个 API 开始处理（0-indexed） |
| `--limit` | - | `None` | 最多处理多少个 API |
| `--delay` | - | `0.5` | API 调用间隔（秒） |
| `--temperature` | `-t` | `0.8` | LLM 温度参数（0.0-1.0） |
| `--log-dir` | - | `component/data/llm_logs` | 日志输出目录 |

### 2.2 使用示例

#### 基本用法 - 处理所有 API

```bash
python component/data/extract_tf_api_mapping.py
```

#### 指定输入输出文件

```bash
python component/data/extract_tf_api_mapping.py \
    -i "component/data/api_mappings.csv" \
    -o "component/data/api_mappings_new.csv"
```

#### 断点续传 - 从第 100 个 API 开始

```bash
python component/data/extract_tf_api_mapping.py --start 100
```

#### 限制处理数量 - 只处理前 50 个

```bash
python component/data/extract_tf_api_mapping.py --limit 50
```

#### 处理指定范围 - 从第 100 个开始，处理 50 个

```bash
python component/data/extract_tf_api_mapping.py --start 100 --limit 50
```

#### 调整 LLM 温度参数（更确定性的输出）

```bash
python component/data/extract_tf_api_mapping.py -t 0.1
```

#### 调整 API 调用间隔（避免限流）

```bash
python component/data/extract_tf_api_mapping.py --delay 1.0
```

#### 使用不同的模型

```bash
python component/data/extract_tf_api_mapping.py -m "qwen-max"
```

#### 完整参数示例

```bash
python component/data/extract_tf_api_mapping.py \
    -i "component/data/api_mappings.csv" \
    -o "component/data/api_mappings_extracted.csv" \
    -m "qwen-plus" \
    -t 0.5 \
    --start 0 \
    --limit 100 \
    --delay 0.5 \
    --log-dir "component/data/llm_logs"
```

---

## 3. validate_tf_api_mapping.py - API 映射验证工具

该工具基于 LLM 和官方文档爬取验证已有的 PyTorch-TensorFlow API 映射是否正确。

### 3.1 参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | `component/data/api_mappings.csv` | 输入的 CSV 文件路径 |
| `--output` | `-o` | `component/data/api_mappings_validated.csv` | 输出的验证后 CSV 文件路径 |
| `--model` | `-m` | `qwen-plus` | LLM 模型名称 |
| `--key-path` | `-k` | `aliyun.key` | API Key 文件路径 |
| `--start` | - | `0` | 从第几条开始处理（0-indexed） |
| `--limit` | - | `None` | 最多处理多少条 |
| `--delay` | - | `1.0` | API 调用间隔（秒，因需爬取文档建议 ≥1） |
| `--temperature` | `-t` | `0.1` | LLM 温度参数（验证任务建议低温度） |
| `--log-dir` | - | `component/data/llm_logs` | 日志输出目录 |
| `--only-no-impl` | - | `False` | 仅处理"无对应实现"的记录 |
| `--only-has-impl` | - | `False` | 仅处理有对应实现的记录 |

### 3.2 使用示例

#### 基本用法 - 验证所有映射

```bash
python component/data/validate_tf_api_mapping.py
```

#### 指定输入输出文件

```bash
python component/data/validate_tf_api_mapping.py `
    -i "component/data/api_mappings.csv" `
    -o "component/data/api_mappings_validated.csv"
```

#### 仅验证"无对应实现"的记录（尝试找到新映射）

```bash
python component/data/validate_tf_api_mapping.py --only-no-impl
```

#### 仅验证已有映射的记录（检查映射是否正确）

```bash
python component/data/validate_tf_api_mapping.py --only-has-impl
```

#### 断点续传 - 从第 200 条开始

```bash
python component/data/validate_tf_api_mapping.py --start 200
```

#### 限制处理数量

```bash
python component/data/validate_tf_api_mapping.py --limit 100
```

#### 处理指定范围

```bash
python component/data/validate_tf_api_mapping.py --start 50 --limit 100
```

#### 调整温度参数

```bash
# 更确定性的输出（推荐用于验证任务）
python component/data/validate_tf_api_mapping.py -t 0.1

# 更随机的输出（可能发现更多候选）
python component/data/validate_tf_api_mapping.py -t 0.6
```

#### 调整延迟（因需爬取文档，建议不低于 1 秒）

```bash
python component/data/validate_tf_api_mapping.py --delay 1.5
```

#### 完整参数示例

```bash
python component/data/validate_tf_api_mapping.py \
    -i "component/data/api_mappings.csv" \
    -o "component/data/api_mappings_validated.csv" \
    -m "qwen-plus" \
    -t 0.1 \
    --start 0 \
    --limit 500 \
    --delay 1.0 \
    --only-has-impl \
    --log-dir "component/data/llm_logs"
```

---

## 4. extract_new_found_apis.py - 新发现映射提取工具

该工具从验证日志中提取新发现的高置信度映射（原来是"无对应实现"，现在找到了映射）。

### 4.1 参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--log-file` | `-l` | 必填 | 验证日志文件路径 |
| `--update-csv` | - | `False` | 是否更新 CSV 文件 |
| `--csv-file` | - | `component/data/api_mappings.csv` | 要更新的 CSV 文件路径 |
| `--output-csv` | - | 同 csv-file | 输出 CSV 文件路径 |

### 4.2 使用示例

```bash
# 仅查看新发现的映射（不更新文件）
python component/data/extract_new_found_apis.py \
    -l "component/data/llm_logs/pt_tf_validation_log_YYYYMMDD_HHMMSS.txt"

# 查看并更新 CSV 文件
python component/data/extract_new_found_apis.py \
    -l "component/data/llm_logs/pt_tf_validation_log_YYYYMMDD_HHMMSS.txt" \
    --update-csv

# 输出到新文件（不覆盖原文件）
python component/data/extract_new_found_apis.py `
    -l "component/data/llm_logs/pt_tf_validation_log_YYYYMMDD_HHMMSS.txt" `
    --update-csv `
    --output-csv "component/data/api_mappings_new.csv"
```

---

## 5. extract_changed_apis.py - 变更映射提取工具

该工具从验证日志中提取需要修改的映射（原映射错误或需要标记为"无对应实现"）。

**新功能**：支持 `--verify-doc` 参数，在更新 CSV 时会验证 TensorFlow 文档是否存在。如果文档不存在，则自动将映射标记为"无对应实现"，无需再单独运行 fix_empty_doc_apis.py。

### 5.1 参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--log-file` | `-l` | 必填 | 验证日志文件路径 |
| `--update-csv` | - | `False` | 是否更新 CSV 文件 |
| `--csv-file` | - | `component/data/api_mappings.csv` | 要更新的 CSV 文件路径 |
| `--output-csv` | - | 同 csv-file | 输出 CSV 文件路径 |
| `--verify-doc` | - | `False` | 验证 TensorFlow 文档是否存在（文档不存在则标记为"无对应实现"） |

### 5.2 使用示例

```bash
# 仅查看需要修改的映射
python component/data/extract_changed_apis.py \
    -l "component/data/llm_logs/pt_tf_validation_log_YYYYMMDD_HHMMSS.txt"

# 查看并更新 CSV 文件
python component/data/extract_changed_apis.py \
    -l "component/data/llm_logs/pt_tf_validation_log_YYYYMMDD_HHMMSS.txt" \
    --update-csv

# 【推荐】启用文档验证，自动处理文档为空的 API
python component/data/extract_changed_apis.py \
    -l "component/data/llm_logs/pt_tf_validation_log_YYYYMMDD_HHMMSS.txt" \
    --update-csv \
    --verify-doc

# 输出到新文件（不覆盖原文件）
python component/data/extract_changed_apis.py \
    -l "component/data/llm_logs/pt_tf_validation_log_YYYYMMDD_HHMMSS.txt" \
    --update-csv \
    --verify-doc \
    --output-csv "component/data/api_mappings_updated.csv"
```

### 5.3 输出说明

启用 `--verify-doc` 后，脚本会在更新时额外输出：
- 类型1（直接改 validated 值）的更新数
- 类型2（标记为"无对应实现"）的更新数
- **文档验证回退数**：因 TensorFlow 文档不存在而被回退为"无对应实现"的数量

---

## 6. 完整工作流程示例

### 6.1 首次提取 API 映射

```bash
# 步骤 1: 提取 PyTorch API 对应的 TensorFlow API
python component/data/extract_tf_api_mapping.py \
    -i "component/data/api_mappings.csv" \
    -o "component/data/api_mappings.csv" \
    -t 0.8

# 输出文件: component/data/api_mappings.csv
# 日志文件: component/data/llm_logs/pt_tf_mapping_log_YYYYMMDD_HHMMSS.txt
```

### 6.2 验证已有映射

```bash
# 步骤 2: 验证提取的映射是否正确
python component/data/validate_tf_api_mapping.py \
    -i "component/data/api_mappings.csv" \
    -o "component/data/api_mappings_validated.csv" \
    -t 0.1

# 输出文件: component/data/api_mappings_validated.csv
# 日志文件: component/data/llm_logs/pt_tf_validation_log_YYYYMMDD_HHMMSS.txt
```

### 6.3 从验证日志中提取需要更新的记录

```bash
# 步骤 3a: 提取新发现的高置信度映射（原来是"无对应实现"）
python component/data/extract_new_found_apis.py \
    -l "component/data/llm_logs/pt_tf_validation_log_YYYYMMDD_HHMMSS.txt" \
    --update-csv \
    --output-csv "component/data/api_mappings_step3a.csv"

# 步骤 3b: 提取需要修改的映射（原映射错误），同时启用文档验证
# 使用 --verify-doc 参数，自动处理文档为空的 API（无需单独运行 fix_empty_doc_apis.py）
python component/data/extract_changed_apis.py \
    -l "component/data/llm_logs/pt_tf_validation_log_YYYYMMDD_HHMMSS.txt" \
    --update-csv \
    --verify-doc \
    --output-csv "component/data/api_mappings_final.csv"
```

### 6.4 分批处理大量 API（避免超时/限流）

```bash
# 第一批: 0-199
python component/data/validate_tf_api_mapping.py --start 0 --limit 200

# 第二批: 200-399
python component/data/validate_tf_api_mapping.py --start 200 --limit 200

# 第三批: 400-465
python component/data/validate_tf_api_mapping.py --start 400
```

---

## 7. 输出文件说明

### 7.1 CSV 文件格式

**api_mappings.csv** (提取结果):
```csv
pytorch-api,tensorflow-api
torch.abs,tf.abs
torch.nn.Conv1d,tf.keras.layers.Conv1D
torch.nn.functional.relu,tf.nn.relu
```

**api_mappings_validated.csv** (验证结果):
```csv
pytorch-api,tensorflow-api,confidence,changed
torch.abs,tf.abs,high,N
torch.nn.Conv1d,tf.keras.layers.Conv1D,high,N
torch.gather,tf.gather_nd,high,Y
```

**api_mappings_final.csv** (最终结果):
```csv
pytorch-api,tensorflow-api
torch.abs,tf.abs
torch.lerp,无对应实现
torch.nn.functional.layer_norm,tf.keras.layers.LayerNormalization
```

### 7.2 日志文件

日志文件保存在 `component/data/llm_logs/` 目录下，包含：
- 每条记录的序号、API 名称、映射结果
- LLM 的完整输出（包括置信度和理由）
- 处理时间戳

---

## 8. 常见问题

### Q1: API 调用失败怎么办？

工具内置重试机制（默认 3 次），如果仍然失败：
- 检查 `aliyun.key` 文件是否正确
- 增加 `--delay` 参数值（如 2.0 秒）
- 使用 `--start` 参数从失败位置继续

### Q2: 如何提高映射准确性？

- 使用 `validate_tf_api_mapping.py` 进行二次验证
- 降低温度参数（`-t 0.1`）获得更确定的输出
- 使用更强的模型（如 `qwen-max`）

### Q3: 处理速度太慢？

- 减小 `--delay` 参数（注意不要触发限流）
- 分批处理，使用 `--start` 和 `--limit` 参数

### Q4: 如何处理"TensorFlow 文档为空"的 API？

在运行 `extract_changed_apis.py` 时添加 `--verify-doc` 参数即可自动处理：
```bash
python component/data/extract_changed_apis.py \
    -l "component/data/llm_logs/pt_tf_validation_log_YYYYMMDD_HHMMSS.txt" \
    --update-csv \
    --verify-doc
```

启用 `--verify-doc` 后，工具会自动验证每个 TensorFlow API 的文档是否存在，如果文档不存在则自动将映射标记为"无对应实现"。

---

## 9. 相关文件

| 文件 | 说明 |
|------|------|
| `component/data/extract_tf_api_mapping.py` | API 映射提取工具 |
| `component/data/validate_tf_api_mapping.py` | API 映射验证工具 |
| `component/data/extract_new_found_apis.py` | 从日志提取新发现的映射 |
| `component/data/extract_changed_apis.py` | 从日志提取需要修改的映射（支持 `--verify-doc` 文档验证） |
| `component/data/api_mappings.csv` | 原始 API 映射表 |
| `component/data/api_mappings_validated.csv` | 验证后的 API 映射表 |
| `component/data/api_mappings_final.csv` | 最终 API 映射表 |
| `component/data/llm_logs/` | LLM 日志目录 |
