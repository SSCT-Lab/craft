# TensorFlow与PaddlePaddle差分测试工具

基于LLM的TensorFlow与PaddlePaddle算子比较测试框架。

## 核心思路

1. **以PyTorch测试用例作为原始数据集**：从MongoDB数据库读取PyTorch算子的测试用例
2. **双向迁移**：将测试用例同时迁移到TensorFlow和PaddlePaddle两个框架
3. **跨框架比较**：执行TensorFlow和PaddlePaddle的测试，比较两个框架的执行结果（不涉及PyTorch）
4. **LLM增强策略**：使用LLM进行测试用例的修复、变异或跳过决策

## 目录结构

```
tf_pd_test/
├── llm_enhanced_compare.py    # 主测试框架（类似pt_tf_test版本）
├── tf_pd_log_1/               # 测试结果存储目录（运行时自动创建）
└── README.md
```

## 使用方法

### 1. 运行差分测试

```bash
# 激活环境
conda activate tf_env

# 运行批量测试（默认配置）
python tf_pd_test/llm_enhanced_compare.py
```

### 2. 配置测试参数

编辑 `llm_enhanced_compare.py` 中 `main()` 函数的配置部分：

```python
# ==================== 测试参数配置 ====================
max_iterations = 3  # 每个测试用例的最大迭代次数
num_test_cases = 3  # 每个算子要测试的用例数量

# 批量测试范围配置
# 设置为 None 表示测试所有算子
# 设置为 (start, end) 表示测试第 start 到第 end 个算子
operator_range = None  # 修改这里以设置测试范围
# ====================================================
```

### 3. 单个算子测试

在 `main()` 函数中取消注释"模式1"的代码块，并注释掉"模式2"的代码：

```python
pytorch_operator_name = "torch.abs"  # 待测试的算子（PyTorch格式）
```

## 工作流程

```
MongoDB (PyTorch测试用例)
         ↓
    加载API映射表 (unified_api_mappings.csv)
         ↓
    过滤：只保留TF和Paddle都有对应实现的算子
         ↓
    准备测试用例（共享numpy数据）
         ↓
    ┌────────────────┬────────────────┐
    │   迁移到TF     │   迁移到Paddle  │
    │   执行TF测试   │   执行Paddle测试│
    └────────────────┴────────────────┘
                    ↓
            比较TF和Paddle结果
                    ↓
    ┌─────────────────────────────────────┐
    │          调用LLM分析                 │
    │  - 结果一致 → 变异(fuzzing)          │
    │  - 执行出错 → 修复(repair)/跳过(skip)│
    │  - 结果不一致 → 分析原因并决策       │
    └─────────────────────────────────────┘
                    ↓
            保存结果到JSON文件
```

## 统计信息

基于 `unified_api_mappings.csv`：
- 总映射数：465 条
- TensorFlow有效映射：366 个（78.7%）
- PaddlePaddle有效映射：431 个（92.7%）
- **两框架都有映射：357 个**（可用于TF-PD差分测试）

## 输出格式

测试结果保存在 `tf_pd_log_1/` 目录下：
- `llm_enhanced_<算子名>_<时间戳>.json`：每个算子的详细测试结果
- `batch_test_log_<时间戳>.txt`：批量测试日志
- `batch_test_summary_<时间戳>.json`：批量测试摘要

## 依赖

- TensorFlow 2.x
- PaddlePaddle 2.x
- pymongo（连接MongoDB）
- openai（调用阿里千问LLM）
- pandas, numpy

## 注意事项

1. 确保MongoDB服务正在运行，且包含 `freefuzz-torch` 数据库
2. 确保 `aliyun.key` 文件存在且包含有效的API密钥
3. TensorFlow和PaddlePaddle的数据格式可能有差异（如NHWC vs NCHW），LLM会尝试处理这些差异
