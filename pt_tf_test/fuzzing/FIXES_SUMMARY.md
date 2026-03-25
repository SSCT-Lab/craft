# LLM Fuzzing 差分测试修复总结

## 修复的问题

### 1. 逻辑运算算子的 dtype 错误

**问题描述：**
- `torch.logical_or` 等逻辑运算算子要求输入必须是 `bool` 类型
- LLM 生成的测试用例使用了 `float32` 等其他类型
- 导致 TensorFlow 报错：`InvalidArgumentError: cannot compute LogicalOr as input #0(zero-based) was expected to be a bool tensor but is a float tensor`

**解决方案：**
1. 在 `build_fuzzing_prompt()` 中检测逻辑运算算子
2. 为逻辑运算添加特殊的 dtype 约束提示
3. 明确告知 LLM 必须使用 `bool` 类型

**代码位置：**
```python
# 检测是否是逻辑运算算子
is_logical_op = any(keyword in torch_api.lower() for keyword in ["logical_", "bitwise_"])

# 为逻辑运算添加特殊提示
dtype_constraint = """
【重要：数据类型约束】
**此算子是逻辑运算，必须使用 bool 类型！**
- PyTorch 和 TensorFlow 的逻辑运算（如 logical_and, logical_or, logical_xor）都要求输入为 bool 类型
- 变异时必须保持 dtype 为 "bool"
- 不要使用 float/int 等其他类型，否则会导致执行失败
"""
```

---

### 2. NaN 比较误报

**问题描述：**
- 当 PyTorch 和 TensorFlow 都返回 `nan` 时，被误报为"结果不一致"
- 原因：原代码只处理了数组中的 NaN，没有正确处理标量 NaN
- 错误信息：`"comparison_error": "结果不一致: torch=nan, tf=nan"`

**解决方案：**
1. 改进标量 NaN 的比较逻辑
2. 明确区分三种情况：
   - 两个都是 NaN → 一致 ✓
   - 一个是 NaN 另一个不是 → 不一致 ✗
   - 都不是 NaN → 使用数值比较

**代码位置：**
```python
# 处理标量的情况（包括 NaN）
if isinstance(torch_res, (float, np.floating)) and isinstance(tf_res, (float, np.floating)):
    # 两个都是 NaN：认为一致
    if np.isnan(torch_res) and np.isnan(tf_res):
        comparison["results_match"] = True
        return comparison
    # 一个是 NaN 另一个不是：不一致
    elif np.isnan(torch_res) or np.isnan(tf_res):
        comparison["comparison_error"] = f"NaN 不一致: torch={torch_res}, tf={tf_res}"
        return comparison
    # 都是普通数值：使用 allclose
    elif np.allclose(torch_res, tf_res, rtol=rtol, atol=atol):
        comparison["results_match"] = True
        return comparison
```

**测试验证：**
```python
# 测试1: 两个标量 NaN 应该被认为一致
torch_result = {"result": np.nan, ...}
tf_result = {"result": np.nan, ...}
comparison = compare_results(torch_result, tf_result)
assert comparison["results_match"]  # ✓ 通过

# 测试2: 一个 NaN 一个普通值应该不一致
tf_result["result"] = 1.0
comparison = compare_results(torch_result, tf_result)
assert not comparison["results_match"]  # ✓ 通过
assert "NaN 不一致" in comparison["comparison_error"]  # ✓ 通过
```

---

### 3. 数据格式差异（NCHW vs NHWC）

**问题描述：**
- PyTorch 使用 NCHW（channels-first）格式
- TensorFlow 默认使用 NHWC（channels-last）格式
- 卷积、池化等算子对输入形状有不同要求
- 输出形状也会不同，导致误报

**解决方案：**
1. 识别需要数据格式转换的算子（卷积、池化、归一化等）
2. 在执行 TensorFlow 测试前，将输入从 NCHW 转为 NHWC
3. 在比较结果前，将输出从 NHWC 转回 NCHW
4. 自动添加 `data_format` 参数

**支持的算子：**
```python
CONV_OPS_NEED_TRANSPOSE = {
    # 卷积相关
    "torch.nn.Conv1d", "torch.nn.Conv2d", "torch.nn.Conv3d",
    "torch.nn.ConvTranspose1d", "torch.nn.ConvTranspose2d", "torch.nn.ConvTranspose3d",
    # 池化相关
    "torch.nn.MaxPool1d", "torch.nn.MaxPool2d", "torch.nn.MaxPool3d",
    "torch.nn.AvgPool1d", "torch.nn.AvgPool2d", "torch.nn.AvgPool3d",
    # 归一化相关
    "torch.nn.BatchNorm1d", "torch.nn.BatchNorm2d", "torch.nn.BatchNorm3d",
    ...
}
```

**转换函数：**
```python
def convert_nchw_to_nhwc(tensor: np.ndarray) -> np.ndarray:
    """将 NCHW 格式转换为 NHWC 格式"""
    ndim = tensor.ndim
    if ndim == 4:  # 2D 卷积/池化: (N, C, H, W) -> (N, H, W, C)
        return np.transpose(tensor, (0, 2, 3, 1))
    elif ndim == 3:  # 1D 卷积/池化: (N, C, L) -> (N, L, C)
        return np.transpose(tensor, (0, 2, 1))
    elif ndim == 5:  # 3D 卷积/池化: (N, C, D, H, W) -> (N, D, H, W, C)
        return np.transpose(tensor, (0, 2, 3, 4, 1))
    else:
        return tensor

def convert_nhwc_to_nchw(tensor: np.ndarray) -> np.ndarray:
    """将 NHWC 格式转换为 NCHW 格式"""
    ndim = tensor.ndim
    if ndim == 4:  # 2D 卷积/池化: (N, H, W, C) -> (N, C, H, W)
        return np.transpose(tensor, (0, 3, 1, 2))
    elif ndim == 3:  # 1D 卷积/池化: (N, L, C) -> (N, C, L)
        return np.transpose(tensor, (0, 2, 1))
    elif ndim == 5:  # 3D 卷积/池化: (N, D, H, W, C) -> (N, C, D, H, W)
        return np.transpose(tensor, (0, 4, 1, 2, 3))
    else:
        return tensor
```

**执行流程：**
```python
# 在 execute_tensorflow_test() 中
needs_transpose = needs_data_format_conversion(api_name)

# 1. 转换输入
if needs_transpose and isinstance(input_tensor, tf.Tensor):
    input_np = input_tensor.numpy()
    input_np = convert_nchw_to_nhwc(input_np)  # NCHW -> NHWC
    input_tensor = tf.constant(input_np, dtype=input_tensor.dtype)

# 2. 添加 data_format 参数
if needs_transpose and "data_format" not in kwargs:
    if ndim == 4:
        kwargs["data_format"] = "NHWC"
    elif ndim == 3:
        kwargs["data_format"] = "NWC"
    elif ndim == 5:
        kwargs["data_format"] = "NDHWC"

# 3. 执行测试
result = func(*args, **kwargs)

# 4. 转换输出
if needs_transpose:
    result_np = convert_nhwc_to_nchw(result_np)  # NHWC -> NCHW
```

**测试验证：**
```python
# 测试 NCHW -> NHWC (4D)
nchw = np.random.randn(2, 3, 4, 5)  # (N, C, H, W)
nhwc = convert_nchw_to_nhwc(nchw)
assert nhwc.shape == (2, 4, 5, 3)  # ✓ 通过

# 测试 NHWC -> NCHW (4D)
nchw_back = convert_nhwc_to_nchw(nhwc)
assert nchw_back.shape == nchw.shape  # ✓ 通过
assert np.allclose(nchw_back, nchw)  # ✓ 通过
```

---

## 测试结果

运行 `test_fixes.py` 验证所有修复：

```bash
python pt_tf_test/fuzzing/test_fixes.py
```

**输出：**
```
======================================================================
运行 Fuzzing 修复测试
======================================================================
测试 NaN 比较...
  ✓ 标量 NaN 比较正确
  ✓ NaN 与普通值比较正确
  ✓ 数组 NaN 比较正确

测试 Inf 比较...
  ✓ 正无穷比较正确
  ✓ 正负无穷比较正确

测试数据格式转换...
  ✓ 算子检测正确
  ✓ NCHW -> NHWC 转换正确
  ✓ NHWC -> NCHW 转换正确
  ✓ 3D 转换正确
  ✓ 5D 转换正确

测试普通数值比较...
  ✓ 相同值比较正确
  ✓ 容差范围内比较正确
  ✓ 超出容差比较正确

======================================================================
✓ 所有测试通过！
======================================================================
```

---

## 影响范围

### 修复前的问题
1. **逻辑运算算子**：100% 失败（dtype 错误）
2. **返回 NaN 的算子**：误报为不一致
3. **卷积/池化算子**：形状不匹配导致误报

### 修复后的改进
1. **逻辑运算算子**：LLM 会生成正确的 bool 类型测试用例
2. **NaN 比较**：正确识别两个框架都返回 NaN 的情况
3. **数据格式**：自动处理 NCHW/NHWC 转换，消除假阳性

---

## 使用建议

1. **重新运行 fuzzing 测试**：
   ```bash
   python pt_tf_test/fuzzing/llm_fuzzing_diff_test_concurrent.py --operators torch_logical_or torch_logical_and torch_max
   ```

2. **检查结果**：
   - 逻辑运算算子应该不再报 dtype 错误
   - NaN 比较应该正确识别一致性
   - 卷积/池化算子应该正确比较结果

3. **扩展支持**：
   - 如果发现其他需要数据格式转换的算子，添加到 `CONV_OPS_NEED_TRANSPOSE`
   - 如果发现其他特殊的 dtype 约束，在 `build_fuzzing_prompt()` 中添加检测

---

## 相关文件

- **主程序**：`pt_tf_test/fuzzing/llm_fuzzing_diff_test_concurrent.py`
- **测试文件**：`pt_tf_test/fuzzing/test_fixes.py`
- **文档**：`pt_tf_test/fuzzing/FIXES_SUMMARY.md`（本文件）

---

## 后续优化

1. **更多算子支持**：
   - 添加更多需要数据格式转换的算子
   - 支持更复杂的参数映射（如 `dim` vs `axis`）

2. **更智能的 LLM 提示**：
   - 根据算子类型动态调整变异策略
   - 添加更多约束检查（如形状兼容性）

3. **更完善的比较逻辑**：
   - 支持更多特殊值（如 `-0.0`）
   - 支持复数比较
   - 支持稀疏张量比较
