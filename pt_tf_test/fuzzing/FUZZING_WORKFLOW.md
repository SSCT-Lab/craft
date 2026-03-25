# PyTorch-TensorFlow Fuzzing 差分测试工作流

## 📋 概述

本工作流用于对 PyTorch 和 TensorFlow 框架的等价算子进行基于 LLM 的 Fuzzing 差分测试，旨在发现两框架在边界条件、极端值、特殊数据类型等场景下的行为不一致问题。

## 📁 目录结构

```
pt_tf_test/fuzzing/
├── FUZZING_WORKFLOW.md              # 本工作流文档
├── extract_success_by_operator.py   # Step 1: 提取成功用例
├── llm_fuzzing_diff_test.py         # Step 2: LLM Fuzzing 差分测试
├── analyze_fuzzing_results.py       # Step 3: 结果分析
├── success_cases/                   # 按算子分类的成功用例 (187个算子)
│   ├── torch_abs_success_cases.json
│   ├── torch_add_success_cases.json
│   └── ...
├── result/                          # Fuzzing 测试结果
│   ├── torch_abs_fuzzing_result_YYYYMMDD_HHMMSS.json
│   ├── torch_add_fuzzing_result_YYYYMMDD_HHMMSS.json
│   └── ...
├── success_cases_data.json          # 所有成功用例汇总
├── success_cases_report.txt         # 成功用例提取报告
└── fuzzing_analysis_report.txt      # Fuzzing 结果分析报告
```

---

## 🔄 执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    原始测试日志 (pt_tf_log_1/)                    │
│              batch_test_log_*.txt + llm_enhanced_*.json          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│          Step 1: extract_success_by_operator.py                  │
│              提取无错误的成功测试用例，按算子分组                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    success_cases/ 目录                           │
│                  187 个算子的成功用例文件                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│            Step 2: llm_fuzzing_diff_test.py                      │
│     LLM 生成变异用例 → 执行差分测试 → 检测不一致行为                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      result/ 目录                                │
│              带时间戳的 Fuzzing 结果 JSON 文件                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│           Step 3: analyze_fuzzing_results.py                     │
│                 分析结果，生成统计报告                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 详细步骤

### Step 1: 提取成功测试用例

**脚本**: `extract_success_by_operator.py`

**功能说明**:
- 读取 `pt_tf_log_1/` 目录下的测试日志
- 筛选出完全无错误（PyTorch 和 TensorFlow 均执行成功）的测试用例
- 按算子名称分组，为每个算子生成独立的 JSON 文件
- 输出统计报告

**执行命令**:
```powershell
# 激活环境
conda activate tf_env

# 执行提取
python pt_tf_test/fuzzing/extract_success_by_operator.py
```

**输出文件**:
- `success_cases/` - 按算子分类的 JSON 文件 (187 个)
- `success_cases_data.json` - 所有成功用例汇总
- `success_cases_report.txt` - 详细提取报告
- `success_cases_summary.txt` - 统计摘要

**输出示例**:
```
提取完成！
成功用例数: 1036 (占比: 34.44%)
失败用例数: 1972 (占比: 65.56%)
涉及算子数: 187
```

---

### Step 2: LLM Fuzzing 差分测试

**脚本**: `llm_fuzzing_diff_test.py`

**功能说明**:
1. 读取 `success_cases/` 中的成功用例
2. 爬取 PyTorch 和 TensorFlow 官方文档
3. 使用 LLM (qwen-plus) 生成变异测试用例，每个用例执行 3 轮变异：
   - **Round 1**: 极端数值变异 (inf, -inf, nan, 1e38, 1e-38, 0, -0.0)
   - **Round 2**: 边界形状变异 (空张量, 标量, 超高维, 单元素)
   - **Round 3**: 复杂类型变异 (float16/32/64, int32/64, bool, complex)
4. 在两个框架上执行变异后的用例
5. 比较输出结果，检测不一致行为
6. 生成带时间戳的结果文件

**命令行参数**:
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--operators` | 指定测试的算子 (空格分隔) | 全部算子 |
| `--max-cases` | 每个算子最大测试用例数 | 不限制 |
| `--limit` | 最多处理的算子数量（按文件名排序） | 不限制 |
| `--model` | LLM 模型名称 | qwen-plus |
| `--key-path` | API Key 文件路径 | aliyun.key |

**执行命令**:

```powershell
# 激活环境
conda activate tf_env

# === 常用执行方式 ===

# 1. 测试单个算子（推荐用于调试）
python pt_tf_test/fuzzing/llm_fuzzing_diff_test.py --operators torch_abs --max-cases 1

# 2. 测试多个指定算子
python pt_tf_test/fuzzing/llm_fuzzing_diff_test.py --operators torch_abs torch_add torch_mul --max-cases 2

# 3. 测试所有算子，每个算子最多 3 个用例
python pt_tf_test/fuzzing/llm_fuzzing_diff_test.py --max-cases 3

# 4. 完整测试所有算子的所有用例（耗时较长）
python pt_tf_test/fuzzing/llm_fuzzing_diff_test.py

# 5. 使用不同的 LLM 模型
python pt_tf_test/fuzzing/llm_fuzzing_diff_test.py --model qwen-max --operators torch_add

# 6. 测试前 N 个算子（按文件名字母顺序）
python pt_tf_test/fuzzing/llm_fuzzing_diff_test.py --limit 10 --max-cases 2

# 7. 手动指定前 10 个算子（按字母序）
python pt_tf_test/fuzzing/llm_fuzzing_diff_test.py --operators torch_abs torch_acos torch_acosh torch_add torch_allclose torch_amax torch_amin torch_angle torch_argmax torch_argmin --max-cases 2
```

**输出文件**:
- `result/<operator>_fuzzing_result_YYYYMMDD_HHMMSS.json`

**结果 JSON 结构**:
```json
{
  "operator": "torch_add",
  "torch_api": "torch.add",
  "tensorflow_api": "tf.add",
  "total_cases": 5,
  "total_fuzzing_rounds": 3,
  "bug_candidates": 2,
  "timestamp": "2026-01-26T17:59:52.761913",
  "results": [
    {
      "original_case_info": {...},
      "original_torch_test_case": {...},
      "original_tensorflow_test_case": {...},
      "fuzzing_results": [
        {
          "round": 1,
          "success": true,
          "mutation_strategy": "极端数值变异...",
          "mutation_reason": "...",
          "torch_test_case": {...},
          "tensorflow_test_case": {...},
          "execution_result": {
            "torch_success": true,
            "tensorflow_success": true,
            "results_match": false,
            "comparison_error": "数值不一致，最大差异: 5.78"
          },
          "is_bug_candidate": true
        },
        ...
      ]
    }
  ]
}
```

**关键特性**:
- ✅ 支持多输入算子 (如 `torch.add` 的 `input` 和 `other`)
- ✅ 支持额外参数 (如 `alpha`, `kernel_size`, `stride`)
- ✅ 自动重试机制 (最多 2 次重试)
- ✅ JSON 截断修复 (处理 LLM 输出被截断的情况)
- ✅ 特殊值解析 (inf, -inf, nan 字符串)

---

### Step 3: 分析 Fuzzing 结果

**脚本**: `analyze_fuzzing_results.py`

**功能说明**:
- 读取 `result/` 目录下所有 Fuzzing 结果
- 统计各类问题的分布
- 识别 Bug Candidates (潜在问题)
- 生成分析报告

**执行命令**:
```powershell
# 激活环境
conda activate tf_env

# 分析所有结果
python pt_tf_test/fuzzing/analyze_fuzzing_results.py
```

**输出文件**:
- `fuzzing_analysis_report.txt` - 详细分析报告

**报告内容**:
- 总体统计 (测试算子数、用例数、Bug Candidates 数)
- 问题分类 (数值不一致、执行失败、形状不匹配等)
- 各算子问题详情
- 需要人工复核的用例列表

---

## 🚀 快速开始

### 完整执行流程（一键运行）

```powershell
# 1. 激活环境
conda activate tf_env

# 2. 进入项目目录
cd D:\graduate\DFrameworkTest

# 3. Step 1: 提取成功用例（如果已执行过可跳过）
python pt_tf_test/fuzzing/extract_success_by_operator.py

# 4. Step 2: 执行 Fuzzing 测试（推荐先测试少量算子）
python pt_tf_test/fuzzing/llm_fuzzing_diff_test.py --operators torch_abs torch_add torch_mul --max-cases 2

# 5. Step 3: 分析结果
python pt_tf_test/fuzzing/analyze_fuzzing_results.py
```

### 单算子调试流程

```powershell
# 测试单个算子的单个用例，便于调试
conda activate tf_env
python pt_tf_test/fuzzing/llm_fuzzing_diff_test.py --operators torch_add --max-cases 1
```

---

## 📊 已发现的典型问题

### 1. API 语义差异

**示例**: `torch.add` vs `tf.add`
- `torch.add(input, other, alpha=2.0)` 语义: `input + alpha * other`
- `tf.add(x, y)` 语义: `x + y`（无 `alpha` 参数）
- **结果**: 数值差异 5.78+

### 2. 数据类型支持差异

**示例**: `float16` 处理
- TensorFlow 在某些平台不支持 `float16` 运算
- 报错: `oneDNN supports DT_HALF only on platforms with AVX-512`

### 3. 边界值处理差异

**示例**: 空张量处理
- 某些算子对 `shape=[0, ...]` 的处理方式不同

---

## ⚠️ 注意事项

1. **环境要求**:
   - Python 3.8+
   - PyTorch 2.0+
   - TensorFlow 2.10+
   - 阿里云 API Key (用于 qwen-plus)

2. **API Key 配置**:
   - 将阿里云 API Key 保存到项目根目录的 `aliyun.key` 文件中

3. **执行时间**:
   - 单个算子单个用例约需 30-60 秒
   - 完整测试 187 个算子可能需要数小时

4. **结果解读**:
   - `is_bug_candidate: true` 表示发现潜在问题，需人工复核
   - 数值差异 < 1e-5 通常为浮点精度误差，可忽略
   - 数值差异 > 1.0 通常为真实的语义差异

---

## 📈 统计数据

| 指标 | 数值 |
|------|------|
| 成功测试用例总数 | 1036 |
| 涉及算子数 | 187 |
| 成功率 | 34.44% |
| 每用例 Fuzzing 轮数 | 3 |

---

## 🔧 扩展开发

### 添加新的变异策略

修改 `llm_fuzzing_diff_test.py` 中的 `mutation_strategies` 字典：

```python
mutation_strategies = {
    1: "极端数值变异...",
    2: "边界形状变异...",
    3: "复杂类型变异...",
    4: "你的新策略..."  # 添加新策略
}
```

同时修改 `FUZZING_ROUNDS` 常量。

### 支持新的框架

可参考现有代码结构，添加对 MindSpore、PaddlePaddle 等框架的支持。

---

*文档创建时间: 2026-01-26*
*维护者: DFramework Test Team*
