# LLM Fuzzing 差分测试工具 - 并发版本说明

## 文件对比

- **原始版本**: `llm_fuzzing_diff_test.py` - 顺序执行
- **并发版本**: `llm_fuzzing_diff_test_concurrent.py` - 多线程并发执行

## 主要改进

### 1. 并发架构设计

并发版本采用**三层并发架构**：

```
算子级别（顺序）
  └─> 用例级别（并发，默认7线程）
       └─> Fuzzing轮次级别（并发，最多3线程）
```

#### 第一层：算子级别（顺序处理）
- 按顺序处理每个算子文件
- 原因：避免过多并发导致资源竞争

#### 第二层：用例级别（并发处理）
- 每个算子的多个测试用例并发处理
- 使用 `ThreadPoolExecutor` 管理线程池
- 默认 7 个并发线程

#### 第三层：Fuzzing 轮次级别（并发处理）
- 每个用例的 3 轮 fuzzing 并发执行
- 使用独立的线程池
- 最多 3 个并发线程（对应 3 轮）

### 2. 核心并发函数

#### `process_single_fuzzing_round()`
```python
def process_single_fuzzing_round(
    client, original_case, torch_doc, tf_doc, 
    round_num, model, print_lock
) -> Dict[str, Any]:
```
- **作用**: 处理单轮 fuzzing（可被多线程调用）
- **并发点**: 3 轮 fuzzing 可同时执行
- **线程安全**: 使用 `print_lock` 保护输出

#### `run_fuzzing_for_case()`
```python
def run_fuzzing_for_case(
    client, original_case, torch_doc, tf_doc, 
    model, print_lock, workers
) -> List[Dict[str, Any]]:
```
- **作用**: 对单个用例进行多轮 fuzzing
- **并发实现**: 使用 `ThreadPoolExecutor` 并发执行 3 轮
- **结果排序**: 按轮次排序保证输出顺序

#### `process_single_case()`
```python
def process_single_case(
    client, case, case_idx, total_cases,
    torch_doc, tf_doc, model, print_lock, workers
) -> Dict[str, Any]:
```
- **作用**: 处理单个测试用例（可被多线程调用）
- **并发点**: 多个用例可同时处理
- **内部并发**: 调用 `run_fuzzing_for_case()` 实现轮次级并发

#### `process_operator()`
```python
def process_operator(
    operator_file, client, model, max_cases,
    print_lock, workers
) -> Dict[str, Any]:
```
- **作用**: 处理单个算子的所有用例
- **并发实现**: 使用 `ThreadPoolExecutor` 并发处理用例
- **结果收集**: 使用字典按索引存储，保证顺序

### 3. 线程安全机制

#### 打印锁 (`Lock`)
```python
print_lock = Lock()

with print_lock:
    print(f"[INFO] 处理中...")
```
- **作用**: 防止多线程打印混乱
- **使用场景**: 所有打印操作都需要获取锁

#### 结果字典
```python
results_dict = {}
for future in as_completed(future_to_idx):
    idx = future_to_idx[future]
    results_dict[idx] = future.result()
```
- **作用**: 按索引存储结果，保证最终顺序正确
- **原因**: `as_completed()` 按完成顺序返回，不是提交顺序

### 4. 进度跟踪

```python
elapsed = time.time() - start_time
avg_time = elapsed / completed
remaining = total - completed
eta = avg_time * remaining
print(f"已完成 {completed}/{total}，耗时 {elapsed:.1f}s，预计剩余 {eta:.1f}s")
```

- 每完成一个算子显示进度
- 计算平均耗时和预计剩余时间
- 帮助用户了解处理进度

## 使用方法

### 基本用法（与原版相同）

```bash
# 测试所有算子（使用默认7个线程）
python llm_fuzzing_diff_test_concurrent.py

# 测试指定算子
python llm_fuzzing_diff_test_concurrent.py --operators torch_abs torch_add

# 限制每个算子的用例数
python llm_fuzzing_diff_test_concurrent.py --max-cases 5

# 只处理前3个算子
python llm_fuzzing_diff_test_concurrent.py --limit 3
```

### 并发控制（新增）

```bash
# 使用 10 个并发线程（更快，但需要更多资源）
python llm_fuzzing_diff_test_concurrent.py --workers 10

# 使用 3 个并发线程（较慢，但资源占用少）
python llm_fuzzing_diff_test_concurrent.py --workers 3

# 使用 1 个线程（等同于顺序执行）
python llm_fuzzing_diff_test_concurrent.py --workers 1
```

### 完整示例

```bash
# 测试 torch_abs 和 torch_add，每个算子最多5个用例，使用5个线程
python llm_fuzzing_diff_test_concurrent.py \
    --operators torch_abs torch_add \
    --max-cases 5 \
    --workers 5 \
    --model qwen-plus \
    --key-path aliyun.key
```

## 性能对比

### 理论加速比

假设：
- 每个算子有 N 个用例
- 每个用例 3 轮 fuzzing
- 每轮 LLM 调用耗时 T 秒

**原始版本总耗时**: `N × 3 × T`

**并发版本总耗时**（7线程）:
- 用例级并发: `(N / 7) × 3 × T`（假设用例间独立）
- 轮次级并发: `(N / 7) × T`（3轮并发执行）
- **理论加速**: 约 **21倍**（7 × 3）

### 实际性能

实际加速比会受以下因素影响：
1. **LLM API 限流**: 并发请求可能触发限流
2. **网络延迟**: 多线程共享网络带宽
3. **CPU 资源**: 测试执行需要 CPU 计算
4. **GIL 限制**: Python 的全局解释器锁

**预期实际加速**: 约 **5-10倍**

## 注意事项

### 1. 资源消耗

- **内存**: 并发线程会同时持有多个测试用例数据
- **网络**: 多个 LLM 请求同时发送
- **CPU**: 多个测试同时执行

**建议**: 根据机器配置调整 `--workers` 参数

### 2. API 限流

如果遇到 LLM API 限流错误：
```
[WARN] Round X LLM 调用异常: Rate limit exceeded
```

**解决方案**:
- 减少并发线程数: `--workers 3`
- 增加重试延迟（代码中已实现指数退避）

### 3. 输出顺序

- 并发执行时，日志输出可能交错
- 使用 `print_lock` 保证单行输出完整性
- 最终结果文件中的顺序是正确的

### 4. 错误处理

- 单个任务失败不影响其他任务
- 错误信息会记录在结果文件中
- 使用 `try-except` 捕获所有异常

## 代码结构对比

### 原始版本
```python
for operator in operators:
    for case in cases:
        for round in [1, 2, 3]:
            # 顺序执行
            result = process_round(round)
```

### 并发版本
```python
for operator in operators:  # 顺序
    with ThreadPoolExecutor(workers=7) as executor:
        # 并发处理用例
        futures = [executor.submit(process_case, case) for case in cases]
        
        # 每个 process_case 内部:
        with ThreadPoolExecutor(workers=3) as inner_executor:
            # 并发处理 3 轮 fuzzing
            round_futures = [inner_executor.submit(process_round, r) for r in [1,2,3]]
```

## 调试建议

### 1. 测试单个算子
```bash
python llm_fuzzing_diff_test_concurrent.py --operators torch_abs --max-cases 2 --workers 2
```

### 2. 查看详细日志
- 观察 `[PROGRESS]` 输出了解进度
- 检查 `[WARN]` 和 `[ERROR]` 消息
- 查看结果文件中的详细信息

### 3. 性能分析
```bash
# 记录开始时间
time python llm_fuzzing_diff_test_concurrent.py --limit 5 --workers 7

# 对比不同线程数的性能
time python llm_fuzzing_diff_test_concurrent.py --limit 5 --workers 1
time python llm_fuzzing_diff_test_concurrent.py --limit 5 --workers 3
time python llm_fuzzing_diff_test_concurrent.py --limit 5 --workers 7
```

## 总结

并发版本通过三层并发架构，在保持代码可维护性的同时，显著提升了执行效率。主要优势：

1. ✅ **性能提升**: 理论加速 21 倍，实际 5-10 倍
2. ✅ **资源利用**: 充分利用多核 CPU 和网络带宽
3. ✅ **灵活配置**: 可通过 `--workers` 调整并发度
4. ✅ **向后兼容**: 参数和输出格式与原版一致
5. ✅ **错误隔离**: 单个任务失败不影响整体执行

**推荐使用场景**: 需要处理大量算子和用例时，使用并发版本可大幅缩短总耗时。
