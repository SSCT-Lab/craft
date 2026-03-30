# MindSpore → PyTorch 算子差分测试工作流

## 概述
本工作流实现了从 MindSpore 官方测试用例出发，自动化地进行 MindSpore 与 PyTorch 对等算子的差分测试。
通过 LLM 辅助完成 API 映射、测试用例生成、修复和变异，最终输出系统化的一致性/不一致性报告。

---

## 流程图

```
testcases_ms/                  (333 个来自官方仓库的 MindSpore 算子测试文件)
      │
      ▼
┌─────────────────────┐
│ Step 1: 提取 MS API  │     extract_ms_apis.py
│ 输出: ms_apis.json   │
└─────────┬───────────┘
          │
          ▼
┌───────────────────────────┐
│ Step 1.5: 验证MS API的存在性│   filter_existing_ms_apis.py
│ 输出: ms_apis_existing.json│
└─────────┬─────────────────┘
          │
          ▼
┌─────────────────────────────┐
│ Step 2: 提取测试用例 (LLM)    │   extract_ms_test_cases.py
│ 输出: ms_test_cases.json     │
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ Step 3: 生成 MS→PT 映射 (LLM)    │   extract_ms_pt_mapping.py
│ 输出: ms_pt_mapping.csv          │
└─────────┬───────────────────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
┌──────────┐  ┌───────────────┐
│ Step 3.5a│  │ Step 3.5b     │
│ 过滤高    │  │ 验证 PT 文档   │  validate_pt_api_docs.py
│ 置信度    │  │ 映射有效性     │
│ 映射      │  │               │
└────┬─────┘  └───────┬───────┘
     │                │
     ▼                ▼
  ms_pt_mapping_high.csv / ms_pt_mapping_validated.csv
          │
          ▼
┌───────────────────────────────────────┐
│ Step 4: 差分测试主框架 (LLM + 执行)     │   llm_enhanced_compare.py
│ - LLM 并发调用                         │
│ - 算子串行执行                          │
│ - 两框架共享输入张量                     │
│ - LLM 进行修复/变异/跳过                │
│ 输出: ms_pt_log_1/*.json               │
└─────────┬─────────────────────────────┘
          │
          ▼
┌───────────────────────────┐
│ Step 5: 结果分析            │   analyze_results.py
│ 输出: analysis/ 目录        │
│ - TXT 报告                 │
│ - CSV 报告                 │
│ - JSON 报告                │
└───────────────────────────┘
```

---

## 详细步骤

### Step 1: 提取 MindSpore API 列表

**脚本**: `extract_ms_apis.py`

**功能**: 扫描 `testcases_ms/` 目录下的 333 个测试文件，通过正则表达式 + AST 分析提取被测试的 MindSpore API 名称。

**支持的 API 模式**:
| 模式 | 示例 | 提取后的 API 名称 |
|------|------|-------------------|
| Primitive (P.Xxx) | `P.Abs()` | `mindspore.ops.Abs` |
| ops 类 (ops.Xxx) | `ops.Argmax()` | `mindspore.ops.Argmax` |
| Functional (F.xxx) | `F.argmax()` | `mindspore.ops.argmax` |
| NN 层 (nn.Xxx) | `nn.Conv2d()` | `mindspore.nn.Conv2d` |
| Tensor 方法 | `tensor.add()` | `mindspore.Tensor.add` |
| 直接导入 | `from mindspore.ops import Abs` | `mindspore.ops.Abs` |
| 文件名推断 | `test_conv_op.py` | `mindspore.ops.Conv2D` (回退) |

**运行命令**:
```bash
conda activate tf_env
python ms_pt_test/extract_ms_apis.py
```

**输出**: `ms_pt_test/data/ms_apis.json`

---

### Step 1.5: 验证 MindSpore API 是否存在

**脚本**: `filter_existing_ms_apis.py`

**功能**: 访问 MindSpore 官方文档页面，过滤掉不存在或无效的 API。

**验证策略**:
- 校验页面标题与内容是否包含 API 名称
- 验证页面内容长度，防止误判短页面

**运行命令**:
```bash
conda activate tf_env
python ms_pt_test/filter_existing_ms_apis.py \
    --input ms_pt_test/data/ms_apis.json \
    --output ms_pt_test/data/ms_apis_existing.json
```

**输出**: `ms_pt_test/data/ms_apis_existing.json`

---

### Step 2: 提取测试用例（LLM 辅助）

**脚本**: `extract_ms_test_cases.py`

**功能**: 对每个提取到的 MS API，读取其对应的测试文件，由 LLM 分析并提取结构化的测试用例。

**LLM Prompt 设计要点**:
- 说明 MindSpore 四种调用模式（Primitive/functional/NN/Tensor method）
- 输出 JSON 包含 `is_class_api`、`init_params`、`test_cases` 字段
- 自动截断过长的文件内容（≤ 8000 字符），防止 Token 超限

**运行命令**:
```bash
conda activate tf_env
python ms_pt_test/extract_ms_test_cases.py [--workers 4]
```

**输出**: `ms_pt_test/data/ms_test_cases.json`

---

### Step 3: 生成 MS→PT API 映射（LLM 辅助）

**脚本**: `extract_ms_pt_mapping.py`

**功能**: 对每个 MS API，让 LLM 判断其在 PyTorch 中的等价 API。

**LLM Prompt 包含的映射示例**:
| MindSpore API | PyTorch API |
|---------------|-------------|
| `mindspore.ops.Abs` | `torch.abs` |
| `mindspore.ops.Add` | `torch.add` |
| `mindspore.nn.Dense` | `torch.nn.Linear` |
| `mindspore.nn.BatchNorm2d` | `torch.nn.BatchNorm2d` |
| `mindspore.ops.Softmax` | `torch.nn.functional.softmax` |

**运行命令**:
```bash
conda activate tf_env
python ms_pt_test/extract_ms_pt_mapping.py [--workers 4]
```

**输出**: `ms_pt_test/data/ms_pt_mapping.csv`（字段：mindspore-api, pytorch-api, confidence, reason）

---

### Step 3.5a: 过滤高置信度映射

**脚本**: `filter_high_confidence_mapping.py`

**功能**: 从映射表中筛选 confidence=high 的条目。

**运行命令**:
```bash
conda activate tf_env
python ms_pt_test/filter_high_confidence_mapping.py
```

**输出**: `ms_pt_test/data/ms_pt_mapping_high.csv`

---

### Step 3.5b: 验证 PyTorch API 文档

**脚本**: `validate_pt_api_docs.py`

**功能**: 通过爬取 PyTorch 官方文档验证映射中的 PT API 是否真实存在。

**验证策略**:
- 检查官方文档 HTML 页面是否存在且长度足够（≥ 300 字符）
- 对未通过验证的 API 标记 `doc_valid=False`

**运行命令**:
```bash
conda activate tf_env
python ms_pt_test/validate_pt_api_docs.py
```

**输出**: `ms_pt_test/data/ms_pt_mapping_validated.csv`

---

### Step 4: 差分测试主框架

**脚本**: `llm_enhanced_compare.py`（核心脚本）

**功能**: 对映射有效的每对 (MS API, PT API)，执行差分测试。

**关键设计**:
1. **共享张量**: 保证 MindSpore 和 PyTorch 使用完全相同的 numpy 输入数据
2. **LLM 并发**: 多线程并发调用 LLM（qwen-plus），加速用例生成/修复
3. **执行串行**: 使用 RLock 保护算子执行，避免并发引起的框架内部状态问题
4. **多轮迭代**: 对每个用例进行多轮 LLM 修复/变异循环（默认 3 轮）
5. **MindSpore 适配**:
   - `context.set_context(mode=PYNATIVE_MODE, device_target='CPU')`
   - 类式 API 需先实例化再调用
   - 使用 `result.asnumpy()` 获取 numpy 结果

**运行命令**:
```bash
conda activate tf_env

# 测试全部算子
python ms_pt_test/llm_enhanced_compare.py --max-iterations 3 --num-cases 3 --workers 4

# 测试指定范围
python ms_pt_test/llm_enhanced_compare.py --start 1 --end 10

# 测试指定算子
python ms_pt_test/llm_enhanced_compare.py --operators mindspore.ops.Abs mindspore.ops.Add
```

**输出**: `ms_pt_test/ms_pt_log_1/` 下的 JSON 文件

---

### Step 5: 结果分析

**脚本**: `analyze_results.py`

**功能**: 汇总分析所有差分测试结果，生成多格式报告。

**报告内容**:
- 算子级别：一致/不一致/错误的分类
- 迭代级别：每次迭代的详细结果
- 错误汇总：最常见的错误类型和原因

**运行命令**:
```bash
conda activate tf_env
python ms_pt_test/analyze_results.py
```

**输出**: `ms_pt_test/analysis/` 目录下的 TXT / CSV / JSON 报告

---

## 目录结构

```
ms_pt_test/
├── extract_ms_apis.py              # Step 1: 提取 MS API 列表
├── filter_existing_ms_apis.py      # Step 1.5: 验证 MS API 是否存在
├── extract_ms_test_cases.py        # Step 2: 提取测试用例
├── extract_ms_pt_mapping.py        # Step 3: 生成 MS→PT 映射
├── filter_high_confidence_mapping.py  # Step 3.5a: 过滤高置信度
├── validate_pt_api_docs.py         # Step 3.5b: 验证 PT 文档
├── llm_enhanced_compare.py         # Step 4: 差分测试主框架 (核心)
├── analyze_results.py              # Step 5: 结果分析
├── WORKFLOW.md                     # 本文档
├── data/                           # 中间数据存储
│   ├── ms_apis.json
│   ├── ms_apis_existing.json
│   ├── ms_test_cases.json
│   ├── ms_pt_mapping.csv
│   ├── ms_pt_mapping_high.csv
│   └── ms_pt_mapping_validated.csv
├── ms_pt_log_1/                    # 差分测试结果
│   ├── llm_enhanced_*.json
│   ├── batch_test_log_*.txt
│   └── batch_test_summary_*.json
└── analysis/                       # 分析报告
    ├── analysis_report_*.txt
    ├── analysis_report_*.csv
    └── analysis_report_*.json
```

---

## 一键运行（完整流程）

```bash
conda activate tf_env

# Step 1: 提取 API
python ms_pt_test/extract_ms_apis.py

# Step 1.5: 验证 API 是否存在
python ms_pt_test/filter_existing_ms_apis.py

# Step 2: 提取测试用例
python ms_pt_test/extract_ms_test_cases.py --workers 4

# Step 3: 生成映射
python ms_pt_test/extract_ms_pt_mapping.py --workers 4

# Step 3.5a: 过滤映射
python ms_pt_test/filter_high_confidence_mapping.py

# Step 3.5b: 验证 PT 文档
python ms_pt_test/validate_pt_api_docs.py

# Step 4: 差分测试（可分批运行）
python ms_pt_test/llm_enhanced_compare.py --start 1 --end 50 --workers 4
python ms_pt_test/llm_enhanced_compare.py --start 51 --end 100 --workers 4
# ...

# Step 5: 分析结果
python ms_pt_test/analyze_results.py
```

---

## MindSpore API 特点（与 TensorFlow 的区别）

| 特征 | MindSpore | TensorFlow |
|------|-----------|------------|
| 测试文件组织 | 扁平目录 (`testcases_ms/test_*_op.py`) | 分类子目录 |
| 类式算子 | `P.Abs()` / `ops.Abs()` → 实例化后调用 | 无等价概念 |
| Functional API | `F.abs()` / `ops.abs()` | `tf.math.abs()` |
| NN 层 | `nn.Conv2d()` | `tf.keras.layers.Conv2D()` |
| 执行模式 | PyNative 模式 (eager) | Eager 模式 |
| API 命名分隔符 | 点号 (`.`) | 点号 (`.`) |
| 张量创建 | `mindspore.Tensor(np_array)` | `tf.constant(np_array)` |
| 结果提取 | `result.asnumpy()` | `result.numpy()` |

---

## 注意事项

1. **环境依赖**: 需要 `mindspore`、`torch`、`openai`、`numpy` 等包
2. **LLM API**: 使用阿里云 DashScope API（qwen-plus），需提供 `aliyun.key`
3. **设备**: 默认使用 CPU 执行，防止 GPU 资源竞争
4. **断点续传**: Step 2 和 Step 3 支持 checkpoint 机制，中断后可恢复
5. **并发安全**: LLM 调用并发 + 算子执行串行，保证结果可复现
