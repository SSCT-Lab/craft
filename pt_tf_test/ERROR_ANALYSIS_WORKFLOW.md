# PyTorch-TensorFlow 测试错误分析工作流程

本文档介绍在生成 API 测试用例日志文件后，如何使用错误分析工具进行结果分析和样例提取。

---

## 目录

- [1. 工作流程概览](#1-工作流程概览)
- [2. 工具说明](#2-工具说明)
- [3. 完整使用流程](#3-完整使用流程)
- [4. 输出文件说明](#4-输出文件说明)
- [5. 常见问题](#5-常见问题)

---

## 1. 工作流程概览

```
测试日志文件 (pt_tf_log_1/*.json)
    ↓
[步骤1] analyze_errors.py → 错误分析报告 (error_analysis_report_new.txt)
    ↓
[步骤2] 提取各类错误样例 (4个工具并行)
    ├─ extract_torch_error_cases.py → 仅PyTorch错误样例
    ├─ extract_tensorflow_error_cases.py → 仅TensorFlow错误样例
    ├─ extract_both_error_cases.py → 两个框架都报错样例
    └─ extract_comparison_error_cases.py → 结果不一致样例
    ↓ (补充)
[可选] analyze_log.py → 批量测试成功率统计
```

---

## 2. 工具说明

### 2.1 analyze_errors.py - 错误统计分析工具

**功能**：统计测试日志中的所有错误类型，生成总体错误分析报告。

**错误分类**：
- **仅 PyTorch 报错**：PyTorch 执行失败，TensorFlow 正常
- **仅 TensorFlow 报错**：TensorFlow 执行失败，PyTorch 正常
- **两个框架都报错**：PyTorch 和 TensorFlow 都执行失败
- **comparison_error**：两者都成功执行但结果不一致

**输出**：
- 总体错误统计（各类错误的总数）
- 每个文件的详细错误信息（包含错误数量和对应的迭代次数）

**位置**：`pt_tf_test/analyze_errors.py`

---

### 2.2 样例提取工具（4个）

这四个工具从错误分析报告中提取具体的测试用例代码和错误信息，便于人工审查。

#### 2.2.1 extract_torch_error_cases.py

**功能**：提取仅 PyTorch 报错的测试样例（TensorFlow 正常执行）

**适用场景**：
- PyTorch API 实现有问题
- PyTorch API 参数验证更严格
- PyTorch 特有的限制

**位置**：`pt_tf_test/analysis/extract_torch_error_cases.py`

#### 2.2.2 extract_tensorflow_error_cases.py

**功能**：提取仅 TensorFlow 报错的测试样例（PyTorch 正常执行）

**适用场景**：
- TensorFlow API 实现有问题
- API 映射不正确
- TensorFlow API 参数不兼容

**位置**：`pt_tf_test/analysis/extract_tensorflow_error_cases.py`

#### 2.2.3 extract_both_error_cases.py

**功能**：提取两个框架都报错的测试样例

**适用场景**：
- 测试用例生成错误（如缺少必需参数）
- 两个框架都不支持的操作
- 共同的边界条件限制

**位置**：`pt_tf_test/analysis/extract_both_error_cases.py`

#### 2.2.4 extract_comparison_error_cases.py

**功能**：提取结果不一致的测试样例（两者都成功执行但输出不同）

**适用场景**：
- API 行为差异
- 默认参数不同
- 数值精度问题
- 算法实现差异

**位置**：`pt_tf_test/analysis/extract_comparison_error_cases.py`

---

### 2.3 analyze_log.py - 批量测试成功率统计

**功能**：分析批量测试日志文件，统计 LLM 生成用例数和成功执行用例数。

**输出**：
- 测试算子总数
- LLM 生成用例总数
- 成功执行用例总数
- 总体成功率
- 成功率分布（0%, 1-25%, 26-50%, 51-75%, 76-99%, 100%）
- 成功率排名（Top 10）

**位置**：`pt_tf_test/analyze_log.py`

---

## 3. 完整使用流程

### 步骤 1：生成错误分析报告

```bash
# 激活环境
conda activate tf_env

# 运行错误分析工具
python pt_tf_test\analyze_errors.py
```

**输出文件**：
- `pt_tf_test/pt_tf_log_1/error_analysis_report_new.txt`

**报告内容示例**：
```
【总体错误统计】
仅 PyTorch 报错的样例数: 180
仅 TensorFlow 报错的样例数: 754
两个框架都报错的样例数: 525
comparison_error 非null值总数: 513
包含错误的文件总数: 305
```

---

### 步骤 2：提取各类错误样例

根据需要运行以下一个或多个工具：

#### 2.1 提取仅 PyTorch 错误样例

```bash
python pt_tf_test\analysis\extract_torch_error_cases.py
```

**输出文件**：`pt_tf_test/analysis/torch_error_samples_report_new.txt`

#### 2.2 提取仅 TensorFlow 错误样例

```bash
python pt_tf_test\analysis\extract_tensorflow_error_cases.py
```

**输出文件**：`pt_tf_test/analysis/tensorflow_error_samples_report_new.txt`

#### 2.3 提取两个框架都报错样例

```bash
python pt_tf_test\analysis\extract_both_error_cases.py
```

**输出文件**：`pt_tf_test/analysis/both_error_samples_report_new.txt`

#### 2.4 提取结果不一致样例

```bash
python pt_tf_test\analysis\extract_comparison_error_cases.py
```

**输出文件**：`pt_tf_test/analysis/comparison_error_samples_report_new.txt`

---

### 步骤 3（可选）：统计批量测试成功率

```bash
python pt_tf_test\analyze_log.py
```

**输出内容**：
- 控制台直接输出统计结果
- 包含成功率分布、Top 10 算子、失败算子列表

---

## 4. 输出文件说明

### 4.1 错误分析报告格式

**文件**：`error_analysis_report_new.txt`

```
1. 文件名: llm_enhanced_torch_xxx.json
--------------------------------------------------------------------------------
   仅 PyTorch 报错的样例数: 2
   对应用例的iteration值: 1, 3
   仅 TensorFlow 报错的样例数: 1
   对应用例的iteration值: 2
   两个框架都报错的样例数: 3
   对应用例的iteration值: 4, 5, 6
   comparison_error 非null值个数: 2
   对应用例的iteration值: 7, 8
```

---

### 4.2 样例报告格式

**文件**：`*_error_samples_report_new.txt`

```
================================================================================
文件: llm_enhanced_torch_unique_20260123_214510.json
--------------------------------------------------------------------------------
样例 1:
torch_error: _return_output() missing 1 required positional argument: 'input'
tensorflow_error: Missing required positional argument
torch_test_case:
{
  "api": "torch.unique",
  "*args": [
    {
      "shape": [4],
      "dtype": "torch.float64"
    }
  ],
  "**kwargs": {}
}
tensorflow_test_case:
{
  "api": "tf.unique",
  "*args": [
    {
      "shape": [4],
      "dtype": "tf.float64"
    }
  ],
  "**kwargs": {}
}
```

---

## 5. 常见问题

### Q1: 如何快速定位问题最严重的 API？

运行 `analyze_errors.py` 后查看报告中"包含错误的文件总数"最多的文件。

### Q2: 如何判断是 API 映射问题还是测试用例生成问题？

- **两个框架都报错**：很可能是测试用例生成问题
- **仅 TensorFlow 报错**：可能是 API 映射不正确
- **comparison_error**：可能是 API 行为差异

### Q3: 样例提取工具的逻辑是互斥的吗？

是的，四个样例提取工具的逻辑完全互斥：
- `torch_error != null` **且** `tensorflow_error == null` → 仅 PyTorch 错误
- `tensorflow_error != null` **且** `torch_error == null` → 仅 TensorFlow 错误
- `torch_error != null` **且** `tensorflow_error != null` → 两个框架都错误
- `comparison_error != null` → 结果不一致（独立维度）

### Q4: 为什么需要查看 iteration 值？

`iteration` 值标识了具体是哪个测试用例失败，便于：
- 在 JSON 文件中定位具体的测试用例
- 追踪 LLM 修复尝试的过程
- 分析同一算子不同用例的失败模式

### Q5: 如何修改分析的日志目录？

修改 `analyze_errors.py` 中的 `main()` 函数：

```python
def main():
    # 修改这两个路径
    log_dir = r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1"  # 日志目录
    output_file = r"d:\graduate\DFrameworkTest\pt_tf_test\pt_tf_log_1\error_analysis_report_new.txt"  # 输出报告
```

样例提取工具会自动读取 `error_analysis_report_new.txt` 中指定的文件。

---

## 6. 工作流程建议

### 6.1 首次分析

1. 运行 `analyze_errors.py` 查看总体错误分布
2. 根据错误类型优先级提取样例：
   - 优先处理"两个框架都报错"（可能是用例生成问题）
   - 其次处理"仅 TensorFlow 报错"（可能是映射问题）
   - 最后处理"comparison_error"（需要深入分析 API 差异）

### 6.2 批量测试分析

1. 运行 `analyze_log.py` 查看总体成功率
2. 针对成功率为 0% 的算子，运行 `analyze_errors.py` 查看详细错误
3. 提取对应类型的样例进行人工审查

### 6.3 问题修复流程

1. 从样例报告中识别问题模式
2. 修复问题（如调整 API 映射、改进测试用例生成逻辑）
3. 重新运行测试
4. 再次运行分析工具验证修复效果

---

## 7. 相关文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `pt_tf_test/analyze_errors.py` | 工具 | 错误统计分析 |
| `pt_tf_test/analyze_log.py` | 工具 | 批量测试成功率统计 |
| `pt_tf_test/analysis/extract_torch_error_cases.py` | 工具 | 提取仅 PyTorch 错误样例 |
| `pt_tf_test/analysis/extract_tensorflow_error_cases.py` | 工具 | 提取仅 TensorFlow 错误样例 |
| `pt_tf_test/analysis/extract_both_error_cases.py` | 工具 | 提取两个框架都报错样例 |
| `pt_tf_test/analysis/extract_comparison_error_cases.py` | 工具 | 提取结果不一致样例 |
| `pt_tf_test/pt_tf_log_1/*.json` | 数据 | 测试结果 JSON 文件 |
| `pt_tf_test/pt_tf_log_1/error_analysis_report_new.txt` | 输出 | 错误分析报告 |
| `pt_tf_test/analysis/*_error_samples_report_new.txt` | 输出 | 各类错误样例报告 |

---

**更新日期**：2026-01-24  
**版本**：v1.0
