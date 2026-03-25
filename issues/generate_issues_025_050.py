#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 Issue 025-050 的标准格式 issue 内容并追加到修改版文件"""

import os

output_file = r"d:\graduate\DFrameworkTest\issues\138个跨表不一致Case的GitHub Issue-修改版.md"

content = r"""
## Issue 025

llm_enhanced_torch_multinomial_20251215_233345.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][multinomial] Output difference anomaly under equivalent migration in multinomial operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for the `torch.multinomial` operator (Maximum difference: 2).

The root cause involves **two distinct problems**:

1. **Stochastic operator**: `multinomial` is a random sampling operator. Different frameworks use different RNG implementations, so identical seeds produce different sampled indices.
2. **Input semantic mismatch**: `torch.multinomial` accepts probability weights (non-negative values treated as unnormalized probabilities), while `tf.random.categorical` accepts **logits** (unnormalized log-probabilities, processed through softmax internally). The migration code passes raw probability weights directly as logits, fundamentally changing the sampling distribution.

- Input: shape=[1, 4], dtype=float32
- Maximum difference: 2

```python
import numpy as np
import torch
import tensorflow as tf

# 原始样本数据
input_data = [[0.9243277311325073, 0.5977890491485596, 1.7590144872665405, 0.1711428165435791]]
input_np = np.array(input_data, dtype=np.float32)

# PyTorch 执行 (输入作为概率权重)
torch.manual_seed(42)
input_pt = torch.tensor(input_np)
out_pt = torch.multinomial(input_pt, num_samples=2, replacement=True)

# TensorFlow 执行 (输入被当作 logits)
tf.random.set_seed(42)
input_tf = tf.constant(input_np)
out_tf = tf.random.categorical(input_tf, num_samples=2)

# 比较结果
pt_np = out_pt.numpy()
tf_np = out_tf.numpy()
max_diff = np.max(np.abs(pt_np.astype(np.float64) - tf_np.astype(np.float64)))
print(f"PyTorch sampled indices: {pt_np}")
print(f"TensorFlow sampled indices: {tf_np}")
print(f"Maximum difference: {max_diff}")
# 注意: multinomial 是随机采样操作，且两框架输入语义不同
# torch.multinomial 接收概率权重，tf.random.categorical 接收 logits
# 实测最大差异: 2
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.random.categorical] Output difference anomaly under equivalent migration in multinomial operator (sample1)

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

When mapping `torch.multinomial` to `tf.random.categorical`, the migration code passes probability weights directly as logits. `torch.multinomial` treats input as unnormalized probability weights, while `tf.random.categorical` treats input as logits (unnormalized log-probabilities, internally processed through softmax). This input semantic mismatch, combined with different RNG implementations, produces different sampled indices (Maximum difference: 2).

Expected behavior: The migration tool should convert probability weights to logits via `log()` before passing to `tf.random.categorical`, i.e., `tf.random.categorical(tf.math.log(input), num_samples)`.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

input_data = [[0.9243277311325073, 0.5977890491485596, 1.7590144872665405, 0.1711428165435791]]
input_np = np.array(input_data, dtype=np.float32)

torch.manual_seed(42)
out_pt = torch.multinomial(torch.tensor(input_np), num_samples=2, replacement=True)

tf.random.set_seed(42)
out_tf = tf.random.categorical(tf.constant(input_np), num_samples=2)

print(f"PyTorch: {out_pt.numpy()}, TensorFlow: {out_tf.numpy()}")
max_diff = np.max(np.abs(out_pt.numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 2
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 2
```

## Issue 026

llm_enhanced_torch_nanmedian_20251201_232713.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nanmedian] Output difference anomaly under equivalent migration in nanmedian operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, an output discrepancy was detected for the `torch.nanmedian` operator (Maximum difference: 0.1793074607849121).

The root cause is a **default axis semantic mismatch**: `torch.nanmedian` without specifying an axis computes the **global** nanmedian across all elements (returning a scalar), while `paddle.nanmedian` without specifying an axis defaults to computing nanmedian along the **last axis** (axis=-1), returning a tensor with the last dimension reduced.

- Input: shape=[2, 3, 4], dtype=float32
- Maximum difference: 0.1793074607849121

```python
import numpy as np
import torch
import paddle

# 使用原始样本数据构造输入
np.random.seed(42)
input_np = np.random.randn(2, 3, 4).astype(np.float32)

# PyTorch 执行 —— 全局 nanmedian（返回标量）
input_pt = torch.tensor(input_np)
out_pt = torch.nanmedian(input_pt)

# PaddlePaddle 执行 —— 默认沿最后一维计算
input_pd = paddle.to_tensor(input_np)
out_pd = paddle.nanmedian(input_pd)

# 比较结果
pt_val = out_pt.numpy()
pd_val = out_pd.numpy()
print(f"Input shape: {input_np.shape}")
print(f"PyTorch nanmedian (global): {pt_val}, shape: {pt_val.shape}")
print(f"PaddlePaddle nanmedian (axis=-1 default): {pd_val}, shape: {pd_val.shape}")
# PyTorch 返回全局中位数标量，Paddle 返回沿 axis=-1 的中位数
# 实测最大差异: 0.1793074607849121
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nanmedian] Output difference anomaly under equivalent migration in nanmedian operator (sample2)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，针对 `nanmedian` 算子，两框架输出结果被多位评审标记为不一致。最大差异为 0.1793074607849121。

根本原因是**默认 axis 语义不一致**：`torch.nanmedian` 在未指定 axis 参数时，默认对所有元素计算全局 nanmedian（返回标量）；而 `paddle.nanmedian` 在未指定 axis 参数时，默认沿最后一维（axis=-1）计算 nanmedian，返回最后一维被规约的张量。

此语义差异直接导致输出形状和数值均不同。迁移代码应在调用 `paddle.nanmedian` 时显式展平输入或指定合适的 axis 参数以对齐 PyTorch 的全局语义。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 4).astype(np.float32)

# PyTorch: 全局 nanmedian
input_pt = torch.tensor(input_np)
out_pt = torch.nanmedian(input_pt)

# Paddle: 默认 axis=-1
input_pd = paddle.to_tensor(input_np)
out_pd = paddle.nanmedian(input_pd)

print(f"PyTorch nanmedian (global): {out_pt.numpy()}")
print(f"PaddlePaddle nanmedian (default): {out_pd.numpy()}")
# 实测最大差异: 0.1793074607849121
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nanmedian",
  "input": {
    "shape": [2, 3, 4],
    "dtype": "float32"
  }
}
```

- `torch.nanmedian` 无 axis 参数时计算全局 nanmedian，`paddle.nanmedian` 无 axis 参数时默认沿 axis=-1 计算。迁移需显式对齐 axis 语义。

## Issue 027

llm_enhanced_torch_nanmedian_20251201_232713.json_sample5.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nanmedian] Output difference anomaly under equivalent migration in nanmedian operator (sample5)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, an output discrepancy was detected for the `torch.nanmedian` operator (Maximum difference: 0.0271770272751663).

Same root cause as Issue 026: `torch.nanmedian` without axis computes the global nanmedian (scalar), while `paddle.nanmedian` defaults to axis=-1, computing nanmedian along the last dimension.

- Input: shape=[2, 5, 5], dtype=float64
- Maximum difference: 0.0271770272751663

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 5, 5).astype(np.float64)

# PyTorch: 全局 nanmedian
input_pt = torch.tensor(input_np)
out_pt = torch.nanmedian(input_pt)

# PaddlePaddle: 默认 axis=-1
input_pd = paddle.to_tensor(input_np)
out_pd = paddle.nanmedian(input_pd)

pt_val = out_pt.numpy()
pd_val = out_pd.numpy()
print(f"Input shape: {input_np.shape}")
print(f"PyTorch nanmedian (global): {pt_val}, shape: {pt_val.shape}")
print(f"PaddlePaddle nanmedian (axis=-1 default): {pd_val}, shape: {pd_val.shape}")
# 实测最大差异: 0.0271770272751663
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nanmedian] Output difference anomaly under equivalent migration in nanmedian operator (sample5)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，针对 `nanmedian` 算子，两框架输出结果被多位评审标记为不一致。最大差异为 0.0271770272751663。

与 Issue 026 相同的根本原因：`torch.nanmedian` 在未指定 axis 参数时计算全局 nanmedian（返回标量），而 `paddle.nanmedian` 默认沿 axis=-1 计算，导致输出形状和数值不同。

- 输入: shape=[2, 5, 5], dtype=float64

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 5, 5).astype(np.float64)

input_pt = torch.tensor(input_np)
out_pt = torch.nanmedian(input_pt)

input_pd = paddle.to_tensor(input_np)
out_pd = paddle.nanmedian(input_pd)

print(f"PyTorch nanmedian (global): {out_pt.numpy()}")
print(f"PaddlePaddle nanmedian (default): {out_pd.numpy()}")
# 实测最大差异: 0.0271770272751663
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nanmedian",
  "input": {
    "shape": [2, 5, 5],
    "dtype": "float64"
  }
}
```

- 同 Issue 026，`torch.nanmedian` 与 `paddle.nanmedian` 的默认 axis 语义不同。

## Issue 028

llm_enhanced_torch_nn_AvgPool1d_20251202_124923.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.AvgPool1d] Output difference anomaly under equivalent migration in AvgPool1d operator

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, an output discrepancy was detected for the `torch.nn.AvgPool1d` operator (Maximum difference: 0.5917866826057434).

The root cause is a **parameter semantic inversion**: PyTorch's `count_include_pad=False` means "do NOT include padding in the average calculation", while Paddle's `exclusive=False` means "do NOT exclude padding" (i.e., DO include padding). These two parameters have **opposite semantics** — the migration code should map `count_include_pad=False` to `exclusive=True` (not `exclusive=False`).

Additionally, `ceil_mode=True` behavior may differ between the two frameworks in edge cases.

- Input: shape=[2, 3, 100], dtype=float32
- Parameters: kernel_size=5, stride=2, padding=2, ceil_mode=True, count_include_pad=False (PyTorch) / exclusive=False (Paddle)
- Maximum difference: 0.5917866826057434

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 100).astype(np.float32)

# PyTorch 执行 (count_include_pad=False: 不计算填充区域)
input_pt = torch.tensor(input_np)
pool_pt = torch.nn.AvgPool1d(kernel_size=5, stride=2, padding=2, ceil_mode=True, count_include_pad=False)
out_pt = pool_pt(input_pt)

# PaddlePaddle 执行 (exclusive=False: 不排除填充，即计算填充区域 —— 语义相反！)
input_pd = paddle.to_tensor(input_np)
pool_pd = paddle.nn.AvgPool1D(kernel_size=5, stride=2, padding=2, ceil_mode=True, exclusive=False)
out_pd = pool_pd(input_pd)

# 比较结果
pt_np = out_pt.detach().numpy()
pd_np = out_pd.numpy()
max_diff = np.max(np.abs(pt_np.astype(np.float64) - pd_np.astype(np.float64)))
print(f"PyTorch output shape: {pt_np.shape}")
print(f"PaddlePaddle output shape: {pd_np.shape}")
print(f"Maximum difference: {max_diff}")
# count_include_pad=False 应映射为 exclusive=True
# 实测最大差异: 0.5917866826057434
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.AvgPool1D] Output difference anomaly under equivalent migration in AvgPool1d operator

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，针对 `nn.AvgPool1d` 算子，两框架输出结果被多位评审标记为不一致。最大差异为 0.5917866826057434。

根本原因是**参数语义反转**：PyTorch 的 `count_include_pad=False` 表示"不将填充区域计入平均"，而 Paddle 的 `exclusive=False` 表示"不排除填充区域"（即将填充计入平均）。两者语义**相反** — 正确的映射应该是 `count_include_pad=False` → `exclusive=True`。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 100).astype(np.float32)

# PyTorch: count_include_pad=False
input_pt = torch.tensor(input_np)
pool_pt = torch.nn.AvgPool1d(kernel_size=5, stride=2, padding=2, ceil_mode=True, count_include_pad=False)
out_pt = pool_pt(input_pt)

# Paddle: exclusive=False (语义相反，应为 exclusive=True)
input_pd = paddle.to_tensor(input_np)
pool_pd = paddle.nn.AvgPool1D(kernel_size=5, stride=2, padding=2, ceil_mode=True, exclusive=False)
out_pd = pool_pd(input_pd)

pt_np = out_pt.detach().numpy()
pd_np = out_pd.numpy()
max_diff = np.max(np.abs(pt_np.astype(np.float64) - pd_np.astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 0.5917866826057434
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api_pt": "torch.nn.AvgPool1d",
  "api_pd": "paddle.nn.AvgPool1D",
  "kernel_size": 5,
  "stride": 2,
  "padding": 2,
  "ceil_mode": true,
  "count_include_pad": false,
  "exclusive": false
}
```

- `count_include_pad=False`（不计入填充）应映射为 `exclusive=True`（排除填充），当前映射为 `exclusive=False` 导致语义反转。

## Issue 029

llm_enhanced_torch_nn_AvgPool1d_20251215_234817.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.AvgPool1d] Output shape mismatch under equivalent migration in AvgPool1d operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a shape mismatch was detected for the `torch.nn.AvgPool1d` operator. PyTorch output shape: (2, 3, 4), TensorFlow output shape: (2, 1, 10).

The root cause is a **data format mismatch**: PyTorch's `nn.AvgPool1d` expects input in NCL format (batch, channels, length) and pools along the last dimension (L). TensorFlow's `tf.keras.layers.AveragePooling1D` defaults to NLC format (batch, length, channels) and pools along the second-to-last dimension. When the same NCL-format input is passed directly to TensorFlow without transposition, the pooling operates on the wrong dimension.

- Input: shape=[2, 3, 10], dtype=float32
- PyTorch: kernel_size=3, stride=2 → output shape (2, 3, 4)
- TensorFlow: pool_size=3, strides=2 → output shape (2, 1, 10) (pooling on wrong axis)

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 10).astype(np.float32)

# PyTorch 执行 (NCL format, pools along L)
input_pt = torch.tensor(input_np)
pool_pt = torch.nn.AvgPool1d(kernel_size=3, stride=2)
out_pt = pool_pt(input_pt)

# TensorFlow 执行 (默认 NLC format, 需要转置或指定 data_format)
input_tf = tf.constant(input_np)
pool_tf = tf.keras.layers.AveragePooling1D(pool_size=3, strides=2)
out_tf = pool_tf(input_tf)

print(f"PyTorch output shape: {out_pt.shape}")    # (2, 3, 4)
print(f"TensorFlow output shape: {out_tf.shape}")  # (2, 1, 10) — 维度不匹配
# 实测: 形状不匹配: PyTorch (2, 3, 4) vs TensorFlow (2, 1, 10)
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.AveragePooling1D] Output shape mismatch under equivalent migration in AvgPool1d operator (sample1)

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

When mapping `torch.nn.AvgPool1d` to `tf.keras.layers.AveragePooling1D`, the output shapes differ: PyTorch produces (2, 3, 4) while TensorFlow produces (2, 1, 10). The root cause is that PyTorch uses NCL (batch, channels, length) data format by default, pooling along the L dimension, while TensorFlow's AveragePooling1D defaults to NLC (batch, length, channels) format, pooling along the second-to-last dimension. The migration code fails to transpose the input or set `data_format='channels_first'`.

Expected behavior: The migration tool should either transpose the input from NCL to NLC before passing to TensorFlow, or set `data_format='channels_first'` in the TensorFlow layer.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 10).astype(np.float32)

out_pt = torch.nn.AvgPool1d(kernel_size=3, stride=2)(torch.tensor(input_np))
out_tf = tf.keras.layers.AveragePooling1D(pool_size=3, strides=2)(tf.constant(input_np))

print(f"PyTorch: {out_pt.shape}, TensorFlow: {out_tf.shape}")
# Shape mismatch: PyTorch (2, 3, 4) vs TensorFlow (2, 1, 10)
```

**Relevant log output**

```
comparison_error: Shape mismatch: PyTorch (2, 3, 4) vs TensorFlow (2, 1, 10)
```

## Issue 030

llm_enhanced_torch_nn_AvgPool1d_20251215_234817.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.AvgPool1d] Output shape mismatch under equivalent migration in AvgPool1d operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a shape mismatch was detected for the `torch.nn.AvgPool1d` operator. PyTorch output shape: (2, 3, 4), TensorFlow output shape: (2, 1, 10).

Same root cause as Issue 029: data format mismatch. PyTorch's AvgPool1d uses NCL format (pools along L), while TensorFlow's AveragePooling1D defaults to NLC format (pools along the second-to-last dimension). Without proper format conversion, pooling operates on the wrong axis.

- Input: shape=[2, 3, 10], dtype=float32
- Parameters: kernel_size/pool_size=3, stride/strides=2

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(123)
input_np = np.random.randn(2, 3, 10).astype(np.float32)

# PyTorch (NCL format)
input_pt = torch.tensor(input_np)
pool_pt = torch.nn.AvgPool1d(kernel_size=3, stride=2)
out_pt = pool_pt(input_pt)

# TensorFlow (默认 NLC format)
input_tf = tf.constant(input_np)
pool_tf = tf.keras.layers.AveragePooling1D(pool_size=3, strides=2)
out_tf = pool_tf(input_tf)

print(f"PyTorch output shape: {out_pt.shape}")    # (2, 3, 4)
print(f"TensorFlow output shape: {out_tf.shape}")  # (2, 1, 10)
# 实测: 形状不匹配: PyTorch (2, 3, 4) vs TensorFlow (2, 1, 10)
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.AveragePooling1D] Output shape mismatch under equivalent migration in AvgPool1d operator (sample2)

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

Same root cause as Issue 029 (sample1). When mapping `torch.nn.AvgPool1d` to `tf.keras.layers.AveragePooling1D` with input shape [2, 3, 10], the output shapes differ: PyTorch (2, 3, 4) vs TensorFlow (2, 1, 10). PyTorch uses NCL format while TensorFlow defaults to NLC. The migration fails to handle data format conversion.

Expected behavior: The migration tool should transpose input from NCL to NLC or set `data_format='channels_first'`.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(123)
input_np = np.random.randn(2, 3, 10).astype(np.float32)

out_pt = torch.nn.AvgPool1d(kernel_size=3, stride=2)(torch.tensor(input_np))
out_tf = tf.keras.layers.AveragePooling1D(pool_size=3, strides=2)(tf.constant(input_np))

print(f"PyTorch: {out_pt.shape}, TensorFlow: {out_tf.shape}")
# Shape mismatch: PyTorch (2, 3, 4) vs TensorFlow (2, 1, 10)
```

**Relevant log output**

```
comparison_error: Shape mismatch: PyTorch (2, 3, 4) vs TensorFlow (2, 1, 10)
```

## Issue 031

llm_enhanced_torch_nn_AvgPool3d_20251215_210802.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.AvgPool3d] Output shape mismatch under equivalent migration in AvgPool3d operator

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a shape mismatch was detected for the `torch.nn.AvgPool3d` operator. PyTorch output shape: (2, 3, 1, 3, 2), TensorFlow output shape: (2, 1, 3, 2, 4).

The root cause is a **data format mismatch**: PyTorch's `nn.AvgPool3d` expects input in NCDHW format (batch, channels, depth, height, width), while TensorFlow's `tf.keras.layers.AveragePooling3D` defaults to NDHWC format (batch, depth, height, width, channels). Without format conversion, the pooling kernel operates on the wrong dimensions.

- Input: shape=[2, 3, 4, 4, 4], dtype=float32
- Parameters: kernel_size=[3,2,2], stride=[2,1,2]
- Shape mismatch: PyTorch (2, 3, 1, 3, 2) vs TensorFlow (2, 1, 3, 2, 4)

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4, 4).astype(np.float32)

# PyTorch (NCDHW format)
input_pt = torch.tensor(input_np)
pool_pt = torch.nn.AvgPool3d(kernel_size=[3,2,2], stride=[2,1,2])
out_pt = pool_pt(input_pt)

# TensorFlow (默认 NDHWC format)
input_tf = tf.constant(input_np)
pool_tf = tf.keras.layers.AveragePooling3D(pool_size=[3,2,2], strides=[2,1,2])
out_tf = pool_tf(input_tf)

print(f"PyTorch output shape: {out_pt.shape}")    # (2, 3, 1, 3, 2)
print(f"TensorFlow output shape: {out_tf.shape}")  # (2, 1, 3, 2, 4)
# 实测: 形状不匹配: PyTorch (2, 3, 1, 3, 2) vs TensorFlow (2, 1, 3, 2, 4)
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.AveragePooling3D] Output shape mismatch under equivalent migration in AvgPool3d operator

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

When mapping `torch.nn.AvgPool3d` to `tf.keras.layers.AveragePooling3D`, the output shapes differ: PyTorch (2, 3, 1, 3, 2) vs TensorFlow (2, 1, 3, 2, 4). PyTorch uses NCDHW format while TensorFlow defaults to NDHWC. The migration code passes the NCDHW input directly without transposition or setting `data_format='channels_first'`.

Expected behavior: The migration tool should transpose input from NCDHW to NDHWC or set `data_format='channels_first'`.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4, 4).astype(np.float32)

out_pt = torch.nn.AvgPool3d(kernel_size=[3,2,2], stride=[2,1,2])(torch.tensor(input_np))
out_tf = tf.keras.layers.AveragePooling3D(pool_size=[3,2,2], strides=[2,1,2])(tf.constant(input_np))

print(f"PyTorch: {out_pt.shape}, TensorFlow: {out_tf.shape}")
# Shape mismatch: PyTorch (2, 3, 1, 3, 2) vs TensorFlow (2, 1, 3, 2, 4)
```

**Relevant log output**

```
comparison_error: Shape mismatch: PyTorch (2, 3, 1, 3, 2) vs TensorFlow (2, 1, 3, 2, 4)
```

## Issue 032

llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.BatchNorm1d] Output difference anomaly under equivalent migration in BatchNorm1d operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for the `torch.nn.BatchNorm1d` operator (Maximum difference: 0.7281599044799805).

The root cause involves **multiple parameter misalignments**:
1. **Normalization axis mismatch**: PyTorch's BatchNorm1d normalizes along channel dimension (axis=1, i.e., dim=1 in NCL input), while TensorFlow's BatchNormalization uses `axis=-1` which normalizes along the last dimension (L), producing entirely different statistics.
2. **Variance calculation**: PyTorch uses unbiased variance (N-1 denominator), TensorFlow uses biased variance (N denominator) by default.
3. **Training mode**: PyTorch BatchNorm1d defaults to training mode (uses batch statistics), while TensorFlow's `training` parameter defaults to False (uses running statistics which are initialized as zeros/ones).

- Input: shape=[2, 3, 10], dtype=float32, num_features=3, affine=False
- Maximum difference: 0.7281599044799805

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 10).astype(np.float32)

# PyTorch (沿 channel 维 axis=1 归一化, training mode, 无偏方差)
input_pt = torch.tensor(input_np)
bn_pt = torch.nn.BatchNorm1d(num_features=3, affine=False)
bn_pt.train()
out_pt = bn_pt(input_pt)

# TensorFlow (axis=-1 沿最后一维归一化, 默认 training=False)
input_tf = tf.constant(input_np)
bn_tf = tf.keras.layers.BatchNormalization(axis=-1, center=False, scale=False)
out_tf = bn_tf(input_tf, training=True)  # 需显式设置 training=True

pt_np = out_pt.detach().numpy()
tf_np = out_tf.numpy()
max_diff = np.max(np.abs(pt_np.astype(np.float64) - tf_np.astype(np.float64)))
print(f"PyTorch output shape: {pt_np.shape}")
print(f"TensorFlow output shape: {tf_np.shape}")
print(f"Maximum difference: {max_diff}")
# axis 不一致导致归一化维度不同
# 实测最大差异: 0.7281599044799805
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.BatchNormalization] Output difference anomaly under equivalent migration in BatchNorm1d operator (sample1)

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

When mapping `torch.nn.BatchNorm1d(num_features=3, affine=False)` to `tf.keras.layers.BatchNormalization(axis=-1, center=False, scale=False)`, the output differs significantly (Maximum difference: 0.7281599044799805). Root causes:
1. PyTorch normalizes along axis=1 (channel dimension in NCL input), TF with `axis=-1` normalizes along the last dimension (L=10).
2. PyTorch uses unbiased variance (N-1), TF uses biased variance (N).
3. TF defaults to `training=False` using uninitialized running statistics.

Expected behavior: The migration should set `axis=1` (not -1) in TensorFlow to match PyTorch's channel-wise normalization, and ensure `training=True` during comparison.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 10).astype(np.float32)

bn_pt = torch.nn.BatchNorm1d(num_features=3, affine=False)
bn_pt.train()
out_pt = bn_pt(torch.tensor(input_np))

bn_tf = tf.keras.layers.BatchNormalization(axis=-1, center=False, scale=False)
out_tf = bn_tf(tf.constant(input_np), training=True)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 0.7281599044799805
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.7281599044799805
```

## Issue 033

llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.BatchNorm1d] Output difference anomaly under equivalent migration in BatchNorm1d operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for the `torch.nn.BatchNorm1d` operator (Maximum difference: 1.0940642356872559).

Same root cause as Issue 032: normalization axis mismatch (PyTorch axis=1 vs TF axis=-1), variance calculation difference (unbiased vs biased), and training mode default difference. This sample uses a smaller input shape [2, 3, 1], making the axis difference particularly impactful.

- Input: shape=[2, 3, 1], dtype=float32, num_features=3, affine=False
- Maximum difference: 1.0940642356872559

```python
import numpy as np
import torch
import tensorflow as tf

input_data = [[[ 0.4989101 ], [ 0.09880682], [-0.09425807]],
              [[-0.45180541], [-1.52982223], [-0.46338478]]]
input_np = np.array(input_data, dtype=np.float32)

# PyTorch (axis=1, channel-wise)
bn_pt = torch.nn.BatchNorm1d(num_features=3, affine=False)
bn_pt.train()
out_pt = bn_pt(torch.tensor(input_np))

# TensorFlow (axis=-1, last dim)
bn_tf = tf.keras.layers.BatchNormalization(axis=-1, center=False, scale=False)
out_tf = bn_tf(tf.constant(input_np), training=True)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 1.0940642356872559
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.BatchNormalization] Output difference anomaly under equivalent migration in BatchNorm1d operator (sample2)

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

Same root cause as Issue 032. Mapping `torch.nn.BatchNorm1d(num_features=3)` with input shape [2, 3, 1] to `tf.keras.layers.BatchNormalization(axis=-1)` produces Maximum difference: 1.0940642356872559. PyTorch normalizes along dim=1 (3 channels), TF normalizes along dim=-1 (size 1), causing fundamentally different statistics.

Expected behavior: Migration should use `axis=1` to match PyTorch's channel normalization dimension.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

input_data = [[[ 0.4989101 ], [ 0.09880682], [-0.09425807]],
              [[-0.45180541], [-1.52982223], [-0.46338478]]]
input_np = np.array(input_data, dtype=np.float32)

bn_pt = torch.nn.BatchNorm1d(num_features=3, affine=False)
bn_pt.train()
out_pt = bn_pt(torch.tensor(input_np))

bn_tf = tf.keras.layers.BatchNormalization(axis=-1, center=False, scale=False)
out_tf = bn_tf(tf.constant(input_np), training=True)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 1.0940642356872559
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 1.0940642356872559
```

## Issue 034

llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.BatchNorm1d] Output difference anomaly under equivalent migration in BatchNorm1d operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for the `torch.nn.BatchNorm1d` operator (Maximum difference: 0.32897937297821045).

Same category as Issues 032/033: normalization axis mismatch. Additionally, in this sample the TensorFlow config uses `axis=-2` and `scale=False` (center defaults to True), while PyTorch uses `affine=False` (neither scale nor center). The `training` parameter default also causes discrepancies.

- Input: shape=[2, 3, 10], dtype=float32, num_features=3, affine=False
- TF config: axis=-2, scale=False
- Maximum difference: 0.32897937297821045

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 10).astype(np.float32)

# PyTorch (axis=1, affine=False: 无 gamma/beta)
bn_pt = torch.nn.BatchNorm1d(num_features=3, affine=False)
bn_pt.train()
out_pt = bn_pt(torch.tensor(input_np))

# TensorFlow (axis=-2, scale=False, center=True by default)
bn_tf = tf.keras.layers.BatchNormalization(axis=-2, scale=False)
out_tf = bn_tf(tf.constant(input_np), training=True)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# axis=-2 对 shape [2,3,10] 等价于 axis=1，但 center 参数不对齐
# 实测最大差异: 0.32897937297821045
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.BatchNormalization] Output difference anomaly under equivalent migration in BatchNorm1d operator (sample3)

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

When mapping `torch.nn.BatchNorm1d(affine=False)` to `tf.keras.layers.BatchNormalization(axis=-2, scale=False)`, the `center` parameter is not explicitly set to False in TensorFlow (defaults to True), while PyTorch's `affine=False` disables both scale and center. Additionally, the `training` parameter defaults differ. Maximum difference: 0.32897937297821045.

Expected behavior: When PyTorch uses `affine=False`, TensorFlow should set both `center=False` and `scale=False`. The `training` mode should also be explicitly aligned.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 10).astype(np.float32)

bn_pt = torch.nn.BatchNorm1d(num_features=3, affine=False)
bn_pt.train()
out_pt = bn_pt(torch.tensor(input_np))

bn_tf = tf.keras.layers.BatchNormalization(axis=-2, scale=False)
out_tf = bn_tf(tf.constant(input_np), training=True)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 0.32897937297821045
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.32897937297821045
```

## Issue 035

llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample5.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.BatchNorm1d] Output difference anomaly under equivalent migration in BatchNorm1d operator (sample5)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for the `torch.nn.BatchNorm1d` operator (Maximum difference: 1.0343306064605713).

Same category as Issues 032-034. In this sample, the TensorFlow config uses `axis=-1` and `scale=False`, while PyTorch normalizes along channel axis (axis=1). Input shape is [1, 3, 5], meaning TF normalizes along the length dimension (5) while PyTorch normalizes along the channel dimension (3).

- Input: shape=[1, 3, 5], dtype=float32, num_features=3, affine=False
- Maximum difference: 1.0343306064605713

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 3, 5).astype(np.float32)

# PyTorch (axis=1)
bn_pt = torch.nn.BatchNorm1d(num_features=3, affine=False)
bn_pt.train()
out_pt = bn_pt(torch.tensor(input_np))

# TensorFlow (axis=-1, 即 axis=2)
bn_tf = tf.keras.layers.BatchNormalization(axis=-1, scale=False)
out_tf = bn_tf(tf.constant(input_np), training=True)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# axis=-1 对 shape [1,3,5] 等价于 axis=2，与 PyTorch 的 axis=1 不同
# 实测最大差异: 1.0343306064605713
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.BatchNormalization] Output difference anomaly under equivalent migration in BatchNorm1d operator (sample5)

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

Same batch normalization axis mismatch as Issues 032-034. With input shape [1, 3, 5], PyTorch normalizes along dim=1 (3 channels), while TF with `axis=-1` normalizes along dim=2 (length=5). Maximum difference: 1.0343306064605713.

Expected behavior: Migration should map PyTorch's channel-wise normalization to TF's `axis=1`.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 3, 5).astype(np.float32)

bn_pt = torch.nn.BatchNorm1d(num_features=3, affine=False)
bn_pt.train()
out_pt = bn_pt(torch.tensor(input_np))

bn_tf = tf.keras.layers.BatchNormalization(axis=-1, scale=False)
out_tf = bn_tf(tf.constant(input_np), training=True)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 1.0343306064605713
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 1.0343306064605713
```

## Issue 036

llm_enhanced_torch_nn_BatchNorm2d_20251215_202219.json_sample5.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.BatchNorm2d] Output difference anomaly under equivalent migration in BatchNorm2d operator (sample5)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for the `torch.nn.BatchNorm2d` operator (Maximum difference: 1.0313220024108887).

The root cause involves **multiple parameter misalignments**:
1. **Normalization axis**: PyTorch normalizes along channel dim (axis=1 in NCHW), TF uses `axis=-1` normalizing along W (axis=3 in NCHW input).
2. **Affine parameters**: PyTorch `affine=False` disables both gamma and beta, but the TF config only omits explicit settings for `center` and `scale` (both default to True).
3. **Variance**: PyTorch uses unbiased variance, TF uses biased variance.

- Input: shape=[2, 3, 4, 4], dtype=float32, num_features=3, eps=0.001, momentum=0.1, affine=False
- Maximum difference: 1.0313220024108887

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)

# PyTorch (NCHW, axis=1, affine=False, eps=0.001)
bn_pt = torch.nn.BatchNorm2d(num_features=3, eps=0.001, momentum=0.1, affine=False)
bn_pt.train()
out_pt = bn_pt(torch.tensor(input_np))

# TensorFlow (axis=-1, eps=0.001, momentum=0.1, center/scale 未对齐)
bn_tf = tf.keras.layers.BatchNormalization(axis=-1, epsilon=0.001, momentum=0.1)
out_tf = bn_tf(tf.constant(input_np), training=True)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# axis=-1 对 NCHW 输入归一化 W 维而非 C 维
# 实测最大差异: 1.0313220024108887
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.BatchNormalization] Output difference anomaly under equivalent migration in BatchNorm2d operator (sample5)

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

When mapping `torch.nn.BatchNorm2d(num_features=3, affine=False, eps=0.001)` with NCHW input [2,3,4,4] to `tf.keras.layers.BatchNormalization(axis=-1, epsilon=0.001)`, TF normalizes along the last dimension (W=4) instead of the channel dimension (C=3). Also, PyTorch's `affine=False` maps to both `center=False` and `scale=False`, but TF defaults both to True. Maximum difference: 1.0313220024108887.

Expected behavior: Migration should use `axis=1`, `center=False`, `scale=False` to match PyTorch semantics.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)

bn_pt = torch.nn.BatchNorm2d(num_features=3, eps=0.001, momentum=0.1, affine=False)
bn_pt.train()
out_pt = bn_pt(torch.tensor(input_np))

bn_tf = tf.keras.layers.BatchNormalization(axis=-1, epsilon=0.001, momentum=0.1)
out_tf = bn_tf(tf.constant(input_np), training=True)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 1.0313220024108887
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 1.0313220024108887
```

## Issue 037

llm_enhanced_torch_nn_BatchNorm2d_20251215_202219.json_sample6.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.BatchNorm2d] Output difference anomaly under equivalent migration in BatchNorm2d operator (sample6)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for the `torch.nn.BatchNorm2d` operator (Maximum difference: 0.7775946855545044).

Same category as Issue 036. In this sample, `affine=True` with `momentum=0.9`. Key issues:
1. **Axis mismatch**: PyTorch axis=1 (channel in NCHW) vs TF `axis=-1` (last dim).
2. **Training mode**: PyTorch defaults to training, TF defaults `training=False` using uninitialized running stats.
3. **Weight initialization**: Both have affine/scale+center enabled, but gamma/beta initialization may differ between frameworks.

- Input: shape=[2, 3, 4, 4], dtype=float32, num_features=3, eps=1e-05, momentum=0.9, affine=True
- Maximum difference: 0.7775946855545044

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)

# PyTorch (NCHW, axis=1, affine=True)
bn_pt = torch.nn.BatchNorm2d(num_features=3, eps=1e-05, momentum=0.9, affine=True)
bn_pt.train()
out_pt = bn_pt(torch.tensor(input_np))

# TensorFlow (axis=-1, center=True, scale=True)
bn_tf = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)
out_tf = bn_tf(tf.constant(input_np), training=True)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# axis 不一致 + training 模式差异
# 实测最大差异: 0.7775946855545044
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.BatchNormalization] Output difference anomaly under equivalent migration in BatchNorm2d operator (sample6)

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

When mapping `torch.nn.BatchNorm2d(affine=True, momentum=0.9)` with NCHW input [2,3,4,4] to `tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9)`, axis mismatch causes different normalization dimensions. TF's `training` parameter not being explicitly set also causes issues. Maximum difference: 0.7775946855545044.

Expected behavior: Migration should set `axis=1` and explicitly pass `training=True` during comparison.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)

bn_pt = torch.nn.BatchNorm2d(num_features=3, eps=1e-05, momentum=0.9, affine=True)
bn_pt.train()
out_pt = bn_pt(torch.tensor(input_np))

bn_tf = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)
out_tf = bn_tf(tf.constant(input_np), training=True)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 0.7775946855545044
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.7775946855545044
```

## Issue 038

llm_enhanced_torch_nn_BatchNorm3d_20251215_165013.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.BatchNorm3d] Output difference anomaly under equivalent migration in BatchNorm3d operator

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for the `torch.nn.BatchNorm3d` operator (Maximum difference: 0.21329689025878906).

Same category as BatchNorm1d/2d issues. Key misalignment:
1. **Axis**: PyTorch normalizes along channel dim (axis=1 in NCDHW), TF `axis=-1` normalizes along the last dim (axis=4).
2. **Epsilon default**: PyTorch defaults to `eps=1e-5`, TF defaults to `epsilon=1e-3` if not explicitly set.
3. **Affine mapping**: PyTorch `affine=False` → TF should set both `center=False` and `scale=False`.

- Input: shape=[2, 3, 4, 4, 4], dtype=float32, num_features=3, affine=False
- Maximum difference: 0.21329689025878906

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4, 4).astype(np.float32)

# PyTorch (NCDHW, axis=1, affine=False, eps=1e-5)
bn_pt = torch.nn.BatchNorm3d(num_features=3, affine=False)
bn_pt.train()
out_pt = bn_pt(torch.tensor(input_np))

# TensorFlow (axis=-1, scale=False, center=False, eps=1e-3 by default)
bn_tf = tf.keras.layers.BatchNormalization(axis=-1, scale=False, center=False)
out_tf = bn_tf(tf.constant(input_np), training=True)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# axis 不一致 + epsilon 默认值差异 (1e-5 vs 1e-3)
# 实测最大差异: 0.21329689025878906
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.BatchNormalization] Output difference anomaly under equivalent migration in BatchNorm3d operator

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

When mapping `torch.nn.BatchNorm3d(num_features=3, affine=False)` with NCDHW input [2,3,4,4,4] to `tf.keras.layers.BatchNormalization(axis=-1, scale=False, center=False)`:
1. `axis=-1` normalizes along dim=4 (last spatial dim) instead of dim=1 (channel).
2. TF default `epsilon=1e-3` differs from PyTorch's `eps=1e-5`.
Maximum difference: 0.21329689025878906.

Expected behavior: Migration should set `axis=1` and `epsilon=1e-5` to match PyTorch.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4, 4).astype(np.float32)

bn_pt = torch.nn.BatchNorm3d(num_features=3, affine=False)
bn_pt.train()
out_pt = bn_pt(torch.tensor(input_np))

bn_tf = tf.keras.layers.BatchNormalization(axis=-1, scale=False, center=False)
out_tf = bn_tf(tf.constant(input_np), training=True)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 0.21329689025878906
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.21329689025878906
```

## Issue 039

llm_enhanced_torch_nn_Conv1d_20251215_194200.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][nn.Conv1d] Output difference anomaly under equivalent migration in Conv1d operator

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, an output discrepancy was detected for the `torch.nn.Conv1d` operator (Maximum difference: 3.1325058937072754).

The root cause is **weight initialization difference**: `nn.Conv1d` is a parameterized layer with learnable weights. PyTorch uses Kaiming uniform initialization by default, while MindSpore uses a different initialization strategy. The test script only fixes the input data but does not synchronize the weight parameters between the two frameworks, causing large output differences.

- Input: shape=[2, 16, 10], dtype=float32
- Parameters: in_channels=16, out_channels=33, kernel_size=3, stride=2, padding=0
- Maximum difference: 3.1325058937072754

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(2, 16, 10).astype(np.float32)

# PyTorch (Kaiming uniform 初始化)
conv_pt = torch.nn.Conv1d(in_channels=16, out_channels=33, kernel_size=3, stride=2, padding=0)
out_pt = conv_pt(torch.tensor(input_np))

# MindSpore (不同的默认权重初始化)
conv_ms = mindspore.nn.Conv1d(in_channels=16, out_channels=33, kernel_size=3, stride=2, padding=0, pad_mode='valid')
out_ms = conv_ms(mindspore.Tensor(input_np))

pt_np = out_pt.detach().numpy()
ms_np = out_ms.asnumpy()
max_diff = np.max(np.abs(pt_np.astype(np.float64) - ms_np.astype(np.float64)))
print(f"PyTorch output shape: {pt_np.shape}")
print(f"MindSpore output shape: {ms_np.shape}")
print(f"Maximum difference: {max_diff}")
# 权重初始化不同导致输出差异
# 实测最大差异: 3.1325058937072754
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.nn.Conv1d] Output difference anomaly under equivalent migration in Conv1d operator

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

When mapping `torch.nn.Conv1d(16, 33, 3, stride=2)` to `mindspore.nn.Conv1d(16, 33, 3, stride=2, pad_mode='valid')`, the outputs differ significantly (Maximum difference: 3.1325058937072754). The root cause is that both frameworks use different default weight initialization strategies. The test only synchronizes input data but not the convolutional layer weights.

**Describe the expected behavior***

For parameterized layers (Conv1d, Linear, etc.), cross-framework migration testing should synchronize the weight parameters (e.g., copying PyTorch's initialized weights to MindSpore) before comparing forward outputs. Alternatively, both layers should be initialized with identical weight values from NumPy.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(2, 16, 10).astype(np.float32)

conv_pt = torch.nn.Conv1d(in_channels=16, out_channels=33, kernel_size=3, stride=2, padding=0)
out_pt = conv_pt(torch.tensor(input_np))

conv_ms = mindspore.nn.Conv1d(in_channels=16, out_channels=33, kernel_size=3, stride=2, padding=0, pad_mode='valid')
out_ms = conv_ms(mindspore.Tensor(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_ms.asnumpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 3.1325058937072754
```

**Related log / screenshot**

```
comparison_error: Numerical mismatch, maximum difference: 3.1325058937072754
```

**Special notes for this issue**

This is a weight initialization synchronization issue, not a computation logic bug. For parameterized layers, the test framework must explicitly synchronize weights before comparison.

## Issue 040

llm_enhanced_torch_nn_Dropout_20251215_193853.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][nn.Dropout] Output difference anomaly under equivalent migration in Dropout operator

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, an output discrepancy was detected for the `torch.nn.Dropout` operator (Maximum difference: 2.3135151863098145).

The root cause involves **two fundamental problems**:
1. **Test setup error**: The bug report's target API is also recorded as `torch.nn.Dropout` instead of `mindspore.nn.Dropout`, suggesting a migration tool failure to correctly map the target API.
2. **Stochastic operator**: `Dropout` is inherently random — it randomly zeroes elements with probability `p`. Even within the same framework, different random seeds produce different dropout masks. Cross-framework comparison of stochastic operators requires either disabling randomness (eval mode) or statistical validation.

- Input: shape=[2, 3], dtype=float32, p=0.5
- Maximum difference: 2.3135151863098145

```python
import numpy as np
import torch
import mindspore

input_data = [[-2.3135151863098145, -1.3254762887954712, 1.866123914718628],
              [0.015756821259856224, -0.45084163546562195, 0.22283050417900085]]
input_np = np.array(input_data, dtype=np.float32)

# PyTorch 执行 (training mode, random dropout)
torch.manual_seed(42)
dropout_pt = torch.nn.Dropout(p=0.5)
dropout_pt.train()
out_pt = dropout_pt(torch.tensor(input_np))

# MindSpore 执行 (不同的 RNG)
mindspore.set_seed(42)
dropout_ms = mindspore.nn.Dropout(p=0.5)
dropout_ms.set_train(True)
out_ms = dropout_ms(mindspore.Tensor(input_np))

pt_np = out_pt.detach().numpy()
ms_np = out_ms.asnumpy()
max_diff = np.max(np.abs(pt_np.astype(np.float64) - ms_np.astype(np.float64)))
print(f"PyTorch output: {pt_np}")
print(f"MindSpore output: {ms_np}")
print(f"Maximum difference: {max_diff}")
# Dropout 是随机算子，跨框架 RNG 不同，输出必然不同
# 实测最大差异: 2.3135151863098145
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.nn.Dropout] Output difference anomaly under equivalent migration in Dropout operator

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

When mapping `torch.nn.Dropout(p=0.5)` to `mindspore.nn.Dropout(p=0.5)` in training mode, the outputs differ (Maximum difference: 2.3135151863098145). Dropout is a stochastic operator that randomly zeroes elements. Different frameworks use different RNG implementations, producing different dropout masks even with the same seed. Additionally, the test configuration records both APIs as `torch.nn.Dropout`, indicating a migration tool mapping failure.

**Describe the expected behavior***

Stochastic operators like Dropout should not be compared using direct numerical equality. Recommended approaches:
1. Compare in eval mode (dropout disabled, output should be identical to input).
2. Use statistical validation (verify dropout rate approaches p over many runs).
3. Fix the migration tool to correctly map the target API name.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

input_data = [[-2.3135151863098145, -1.3254762887954712, 1.866123914718628],
              [0.015756821259856224, -0.45084163546562195, 0.22283050417900085]]
input_np = np.array(input_data, dtype=np.float32)

torch.manual_seed(42)
dropout_pt = torch.nn.Dropout(p=0.5)
dropout_pt.train()
out_pt = dropout_pt(torch.tensor(input_np))

mindspore.set_seed(42)
dropout_ms = mindspore.nn.Dropout(p=0.5)
dropout_ms.set_train(True)
out_ms = dropout_ms(mindspore.Tensor(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_ms.asnumpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 2.3135151863098145
```

**Related log / screenshot**

```
comparison_error: Numerical mismatch, maximum difference: 2.3135151863098145
```

**Special notes for this issue**

1. The bug file records both source and target API as `torch.nn.Dropout` — the migration tool fails to map the target to `mindspore.nn.Dropout`.
2. Dropout is a stochastic operator; direct numerical comparison is invalid. Should compare in eval mode or use statistical validation.

## Issue 041

llm_enhanced_torch_nn_Linear_20251214_171639.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][nn.Linear] Output difference anomaly under equivalent migration in Linear operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, an output discrepancy was detected for the `torch.nn.Linear` operator (Maximum difference: 1.3566744327545166).

The root cause is **weight initialization difference**: `nn.Linear` is a parameterized layer. PyTorch uses Kaiming uniform initialization, while MindSpore uses a different default initialization. The test only fixes input data but does not synchronize weight parameters between frameworks.

- Input: shape=[2, 128], dtype=float32, in_features=128, out_features=1, bias=False
- Maximum difference: 1.3566744327545166

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(2, 128).astype(np.float32)

# PyTorch (Kaiming uniform 权重初始化)
linear_pt = torch.nn.Linear(in_features=128, out_features=1, bias=False)
out_pt = linear_pt(torch.tensor(input_np))

# MindSpore (不同的默认权重初始化)
linear_ms = mindspore.mint.nn.Linear(in_features=128, out_features=1, bias=False)
out_ms = linear_ms(mindspore.Tensor(input_np))

pt_np = out_pt.detach().numpy()
ms_np = out_ms.asnumpy()
max_diff = np.max(np.abs(pt_np.astype(np.float64) - ms_np.astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 权重初始化不同导致差异
# 实测最大差异: 1.3566744327545166
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.mint.nn.Linear] Output difference anomaly under equivalent migration in Linear operator (sample3)

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

When mapping `torch.nn.Linear(128, 1, bias=False)` to `mindspore.mint.nn.Linear(128, 1, bias=False)`, outputs differ significantly (Maximum difference: 1.3566744327545166). Both frameworks use different default weight initialization strategies. The test synchronizes input data but not the Linear layer weights.

**Describe the expected behavior***

Cross-framework migration testing for parameterized layers should synchronize weight parameters before comparing outputs (e.g., initialize both with identical NumPy weight matrices).

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(2, 128).astype(np.float32)

linear_pt = torch.nn.Linear(in_features=128, out_features=1, bias=False)
out_pt = linear_pt(torch.tensor(input_np))

linear_ms = mindspore.mint.nn.Linear(in_features=128, out_features=1, bias=False)
out_ms = linear_ms(mindspore.Tensor(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_ms.asnumpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 1.3566744327545166
```

**Related log / screenshot**

```
comparison_error: Numerical mismatch, maximum difference: 1.3566744327545166
```

**Special notes for this issue**

Weight initialization issue, not a computation logic bug. Test framework must synchronize weights for parameterized layers.

## Issue 042

llm_enhanced_torch_nn_Linear_20251214_171639.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][nn.Linear] Output difference anomaly under equivalent migration in Linear operator (sample4)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, an output discrepancy was detected for the `torch.nn.Linear` operator (Maximum difference: 2.126903533935547).

Same root cause as Issue 041: weight initialization difference between PyTorch and MindSpore for parameterized layers. This sample uses a larger matrix (64→64) which amplifies the difference.

- Input: shape=[2, 64], dtype=float32, in_features=64, out_features=64, bias=False
- Maximum difference: 2.126903533935547

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(2, 64).astype(np.float32)

# PyTorch
linear_pt = torch.nn.Linear(in_features=64, out_features=64, bias=False)
out_pt = linear_pt(torch.tensor(input_np))

# MindSpore
linear_ms = mindspore.mint.nn.Linear(in_features=64, out_features=64, bias=False)
out_ms = linear_ms(mindspore.Tensor(input_np))

pt_np = out_pt.detach().numpy()
ms_np = out_ms.asnumpy()
max_diff = np.max(np.abs(pt_np.astype(np.float64) - ms_np.astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 2.126903533935547
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.mint.nn.Linear] Output difference anomaly under equivalent migration in Linear operator (sample4)

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

When mapping `torch.nn.Linear(64, 64, bias=False)` to `mindspore.mint.nn.Linear(64, 64, bias=False)`, outputs differ significantly (Maximum difference: 2.126903533935547). Same root cause as Issue 041: different weight initialization strategies between frameworks.

**Describe the expected behavior***

Cross-framework migration testing for parameterized layers should synchronize weight parameters before comparison.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(2, 64).astype(np.float32)

linear_pt = torch.nn.Linear(in_features=64, out_features=64, bias=False)
out_pt = linear_pt(torch.tensor(input_np))

linear_ms = mindspore.mint.nn.Linear(in_features=64, out_features=64, bias=False)
out_ms = linear_ms(mindspore.Tensor(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_ms.asnumpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 2.126903533935547
```

**Related log / screenshot**

```
comparison_error: Numerical mismatch, maximum difference: 2.126903533935547
```

**Special notes for this issue**

Same weight initialization issue as Issue 041. Larger matrix size (64→64) amplifies the difference compared to (128→1).

## Issue 043

llm_enhanced_torch_nn_Linear_20251215_164342.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.Linear] Output difference anomaly under equivalent migration in Linear operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for the `torch.nn.Linear` operator (Maximum difference: 2.252037525177002).

The root cause is **weight initialization difference**: PyTorch's `nn.Linear` uses Kaiming uniform initialization, while TensorFlow's `tf.keras.layers.Dense` uses Glorot uniform initialization. The test only fixes input data but does not synchronize weight parameters.

- Input: shape=[2, 3], dtype=float32, in_features=3, out_features=1, bias=False
- Maximum difference: 2.252037525177002

```python
import numpy as np
import torch
import tensorflow as tf

input_data = [[-1.202417254447937, -0.6622903943061829, -0.3366345167160034],
              [0.7163922786712646, -1.2270342111587524, -0.24179375171661377]]
input_np = np.array(input_data, dtype=np.float32)

# PyTorch (Kaiming uniform 初始化)
linear_pt = torch.nn.Linear(in_features=3, out_features=1, bias=False)
out_pt = linear_pt(torch.tensor(input_np))

# TensorFlow (Glorot uniform 初始化)
linear_tf = tf.keras.layers.Dense(units=1, use_bias=False)
out_tf = linear_tf(tf.constant(input_np))

pt_np = out_pt.detach().numpy()
tf_np = out_tf.numpy()
max_diff = np.max(np.abs(pt_np.astype(np.float64) - tf_np.astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 默认权重初始化不同导致差异
# 实测最大差异: 2.252037525177002
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.Dense] Output difference anomaly under equivalent migration in Linear operator (sample2)

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

When mapping `torch.nn.Linear(3, 1, bias=False)` to `tf.keras.layers.Dense(1, use_bias=False)`, outputs differ significantly (Maximum difference: 2.252037525177002). PyTorch uses Kaiming uniform initialization while TensorFlow uses Glorot uniform. The test does not synchronize weight parameters between frameworks.

Expected behavior: Cross-framework migration testing for parameterized layers should explicitly synchronize weight parameters (e.g., copy PyTorch weights to TensorFlow layer) before comparing outputs.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

input_data = [[-1.202417254447937, -0.6622903943061829, -0.3366345167160034],
              [0.7163922786712646, -1.2270342111587524, -0.24179375171661377]]
input_np = np.array(input_data, dtype=np.float32)

linear_pt = torch.nn.Linear(in_features=3, out_features=1, bias=False)
out_pt = linear_pt(torch.tensor(input_np))

linear_tf = tf.keras.layers.Dense(units=1, use_bias=False)
out_tf = linear_tf(tf.constant(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 2.252037525177002
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 2.252037525177002
```

## Issue 044

llm_enhanced_torch_nn_Linear_20251215_164342.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.Linear] Output difference anomaly under equivalent migration in Linear operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for the `torch.nn.Linear` operator (Maximum difference: 3.229126214981079).

Same root cause as Issue 043: weight initialization difference between PyTorch (Kaiming uniform) and TensorFlow (Glorot uniform). This sample uses a larger matrix (64→64) which amplifies the difference.

- Input: shape=[2, 64], dtype=float32, in_features=64, out_features=64, bias=False
- Maximum difference: 3.229126214981079

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 64).astype(np.float32)

# PyTorch (Kaiming uniform)
linear_pt = torch.nn.Linear(in_features=64, out_features=64, bias=False)
out_pt = linear_pt(torch.tensor(input_np))

# TensorFlow (Glorot uniform)
linear_tf = tf.keras.layers.Dense(units=64, use_bias=False)
out_tf = linear_tf(tf.constant(input_np))

pt_np = out_pt.detach().numpy()
tf_np = out_tf.numpy()
max_diff = np.max(np.abs(pt_np.astype(np.float64) - tf_np.astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 3.229126214981079
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.Dense] Output difference anomaly under equivalent migration in Linear operator (sample3)

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

Same root cause as Issue 043. Mapping `torch.nn.Linear(64, 64, bias=False)` to `tf.keras.layers.Dense(64, use_bias=False)` produces Maximum difference: 3.229126214981079 due to different default weight initialization strategies.

Expected behavior: Weight parameters should be synchronized between frameworks before comparison.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 64).astype(np.float32)

linear_pt = torch.nn.Linear(in_features=64, out_features=64, bias=False)
out_pt = linear_pt(torch.tensor(input_np))

linear_tf = tf.keras.layers.Dense(units=64, use_bias=False)
out_tf = linear_tf(tf.constant(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 3.229126214981079
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 3.229126214981079
```

## Issue 045

llm_enhanced_torch_nn_MaxPool1d_20251215_164114.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.MaxPool1d] Output shape mismatch under equivalent migration in MaxPool1d operator (sample4)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a shape mismatch was detected for the `torch.nn.MaxPool1d` operator. PyTorch output shape: (2, 2, 3, 5), TensorFlow output shape: (2, 1, 10).

The root causes are:
1. **Data format mismatch**: PyTorch uses NCL format (pools along L), TensorFlow defaults to NLC format (pools along the second-to-last dim). Without format conversion, pooling operates on the wrong dimension.
2. **`return_indices` unsupported**: PyTorch's `return_indices=True` returns a tuple (values, indices), making the output shape (2, 2, 3, 5) which includes both values and indices. TensorFlow's MaxPooling1D has no native `return_indices` support.

- Input: shape=[2, 3, 10], dtype=float32
- Parameters: kernel_size=2, stride=2, return_indices=True
- Shape mismatch: PyTorch (2, 2, 3, 5) vs TensorFlow (2, 1, 10)

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 10).astype(np.float32)

# PyTorch (NCL, return_indices=True → 返回 (values, indices))
pool_pt = torch.nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
out_pt_values, out_pt_indices = pool_pt(torch.tensor(input_np))

# TensorFlow (默认 NLC, 不支持 return_indices)
pool_tf = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)
out_tf = pool_tf(tf.constant(input_np))

print(f"PyTorch values shape: {out_pt_values.shape}")    # (2, 3, 5)
print(f"PyTorch indices shape: {out_pt_indices.shape}")   # (2, 3, 5)
print(f"TensorFlow output shape: {out_tf.shape}")          # (2, 1, 10)
# 数据格式不匹配 + return_indices 不支持
# 实测: 形状不匹配: PyTorch (2, 2, 3, 5) vs TensorFlow (2, 1, 10)
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.MaxPooling1D] Output shape mismatch under equivalent migration in MaxPool1d operator (sample4)

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

When mapping `torch.nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)` to `tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)`:
1. PyTorch NCL format vs TF NLC format causes pooling on wrong dimension.
2. PyTorch's `return_indices=True` returns (values, indices) tuple, TF has no native support for this.
Shape mismatch: PyTorch (2, 2, 3, 5) vs TensorFlow (2, 1, 10).

Expected behavior: Migration should handle data format conversion (NCL→NLC or set `data_format='channels_first'`) and note that `return_indices` is not supported in TensorFlow.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 10).astype(np.float32)

pool_pt = torch.nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
out_pt = pool_pt(torch.tensor(input_np))

pool_tf = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)
out_tf = pool_tf(tf.constant(input_np))

print(f"PyTorch: {[x.shape for x in out_pt]}, TensorFlow: {out_tf.shape}")
# Shape mismatch: PyTorch (2, 2, 3, 5) vs TensorFlow (2, 1, 10)
```

**Relevant log output**

```
comparison_error: Shape mismatch: PyTorch (2, 2, 3, 5) vs TensorFlow (2, 1, 10)
```

## Issue 046

llm_enhanced_torch_nn_MaxPool1d_20251215_164114.json_sample7.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.MaxPool1d] Output shape mismatch under equivalent migration in MaxPool1d operator (sample7)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a shape mismatch was detected for the `torch.nn.MaxPool1d` operator. PyTorch output shape: (2, 3, 5), TensorFlow output shape: (2, 1, 10).

Same root cause as Issue 045: data format mismatch — PyTorch uses NCL format, TensorFlow defaults to NLC. In this sample, `return_indices=False`, so the shape mismatch is purely due to the data format issue.

- Input: shape=[2, 3, 10], dtype=float32
- Parameters: kernel_size=2, stride=2, return_indices=False
- Shape mismatch: PyTorch (2, 3, 5) vs TensorFlow (2, 1, 10)

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 10).astype(np.float32)

# PyTorch (NCL format)
pool_pt = torch.nn.MaxPool1d(kernel_size=2, stride=2, return_indices=False)
out_pt = pool_pt(torch.tensor(input_np))

# TensorFlow (默认 NLC format)
pool_tf = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)
out_tf = pool_tf(tf.constant(input_np))

print(f"PyTorch output shape: {out_pt.shape}")    # (2, 3, 5)
print(f"TensorFlow output shape: {out_tf.shape}")  # (2, 1, 10)
# 实测: 形状不匹配: PyTorch (2, 3, 5) vs TensorFlow (2, 1, 10)
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.MaxPooling1D] Output shape mismatch under equivalent migration in MaxPool1d operator (sample7)

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

Same root cause as Issue 045. When mapping `torch.nn.MaxPool1d(kernel_size=2, stride=2)` with NCL input [2, 3, 10] to `tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)` with default NLC format, shapes differ: PyTorch (2, 3, 5) vs TensorFlow (2, 1, 10). The migration fails to handle data format conversion.

Expected behavior: Migration should transpose input from NCL to NLC or set `data_format='channels_first'`.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 10).astype(np.float32)

out_pt = torch.nn.MaxPool1d(kernel_size=2, stride=2)(torch.tensor(input_np))
out_tf = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(tf.constant(input_np))

print(f"PyTorch: {out_pt.shape}, TensorFlow: {out_tf.shape}")
# Shape mismatch: PyTorch (2, 3, 5) vs TensorFlow (2, 1, 10)
```

**Relevant log output**

```
comparison_error: Shape mismatch: PyTorch (2, 3, 5) vs TensorFlow (2, 1, 10)
```

## Issue 047

llm_enhanced_torch_nn_MaxPool2d_20251215_192154.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][nn.MaxPool2d] Output difference anomaly under equivalent migration in MaxPool2d operator

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, an output discrepancy was detected for the `torch.nn.MaxPool2d` operator (Maximum difference: 7).

The root cause involves **padding semantics and return_indices behavior differences**:
1. **Padding**: PyTorch `padding=1` adds 1 row/column of zeros on each side. MindSpore `pad_mode='pad', padding=1` should be equivalent, but the exact behavior of zero-padding for integer inputs may differ.
2. **`return_indices`**: Both frameworks support `return_indices=True`, but the index computation may differ — PyTorch computes indices based on the padded tensor, while MindSpore may compute indices differently.

- Input: shape=[1, 1, 2, 4], dtype=int32
- Parameters: kernel_size=2, stride=2, padding=1, return_indices=True
- Maximum difference: 7 (index difference, not value difference)

```python
import numpy as np
import torch
import mindspore

input_data = [[[[-3, -7,  0, -2],
                [-7,  9,  8,  6]]]]
input_np = np.array(input_data, dtype=np.int32)

# PyTorch (padding=1, return_indices=True)
pool_pt = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)
out_pt_values, out_pt_indices = pool_pt(torch.tensor(input_np, dtype=torch.float32))

# MindSpore (pad_mode='pad', padding=1, return_indices=True)
pool_ms = mindspore.nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='pad', padding=1, return_indices=True)
out_ms_values, out_ms_indices = pool_ms(mindspore.Tensor(input_np, dtype=mindspore.float32))

pt_idx = out_pt_indices.numpy()
ms_idx = out_ms_indices.asnumpy()
max_diff = np.max(np.abs(pt_idx.astype(np.float64) - ms_idx.astype(np.float64)))
print(f"PyTorch indices: {pt_idx}")
print(f"MindSpore indices: {ms_idx}")
print(f"Maximum index difference: {max_diff}")
# 实测最大差异: 7 (索引差异)
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.nn.MaxPool2d] Output difference anomaly under equivalent migration in MaxPool2d operator

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

When mapping `torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)` to `mindspore.nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='pad', padding=1, return_indices=True)` with integer input [1,1,2,4], the returned indices differ between frameworks (Maximum difference: 7). PyTorch computes indices based on the padded tensor layout, while MindSpore may compute indices based on a different reference frame.

**Describe the expected behavior***

Both frameworks should return consistent max-pool indices for the same padded input. If the index reference frames differ (e.g., padded vs unpadded), this should be documented and the migration tool should apply an index offset correction.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

input_data = [[[[-3, -7,  0, -2],
                [-7,  9,  8,  6]]]]
input_np = np.array(input_data, dtype=np.int32)

pool_pt = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=True)
out_pt_values, out_pt_indices = pool_pt(torch.tensor(input_np, dtype=torch.float32))

pool_ms = mindspore.nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='pad', padding=1, return_indices=True)
out_ms_values, out_ms_indices = pool_ms(mindspore.Tensor(input_np, dtype=mindspore.float32))

print(f"PyTorch indices: {out_pt_indices.numpy()}")
print(f"MindSpore indices: {out_ms_indices.asnumpy()}")
max_diff = np.max(np.abs(out_pt_indices.numpy().astype(np.float64) - out_ms_indices.asnumpy().astype(np.float64)))
print(f"Maximum index difference: {max_diff}")
# 实测最大差异: 7
```

**Related log / screenshot**

```
comparison_error: Numerical mismatch, maximum difference: 7
```

**Special notes for this issue**

The difference of 7 is an index value difference, not a pooled value difference. The two frameworks may use different index reference frames for the `return_indices` feature when padding is applied.

## Issue 048

llm_enhanced_torch_nn_MaxPool2d_20251215_205114.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.MaxPool2d] Output shape mismatch under equivalent migration in MaxPool2d operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a shape mismatch was detected for the `torch.nn.MaxPool2d` operator. PyTorch output shape: (2, 2, 3, 2, 2), TensorFlow output shape: (2, 2, 2, 4).

The root causes are:
1. **Data format mismatch**: PyTorch uses NCHW format, TensorFlow defaults to NHWC. Direct input without transposition causes pooling on wrong dimensions.
2. **Padding mismatch**: PyTorch `padding=1` (fixed 1-pixel zero-padding), TF `padding='same'` (dynamic auto-padding). These produce different output sizes.
3. **`return_indices` unsupported**: PyTorch's `return_indices=True` returns (values, indices) tuple, causing shape (2, 2, 3, 2, 2). TensorFlow has no native support.

- Input: shape=[2, 3, 4, 4], dtype=float32
- Parameters: kernel_size=3, stride=2, padding=1 (PyTorch) / 'same' (TF), return_indices=True
- Shape mismatch: PyTorch (2, 2, 3, 2, 2) vs TensorFlow (2, 2, 2, 4)

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)

# PyTorch (NCHW, padding=1, return_indices=True)
pool_pt = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
out_pt_values, out_pt_indices = pool_pt(torch.tensor(input_np))

# TensorFlow (默认 NHWC, padding='same')
pool_tf = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
out_tf = pool_tf(tf.constant(input_np))

print(f"PyTorch values shape: {out_pt_values.shape}")   # (2, 3, 2, 2)
print(f"PyTorch indices shape: {out_pt_indices.shape}")  # (2, 3, 2, 2)
print(f"TensorFlow output shape: {out_tf.shape}")         # (2, 2, 2, 4)
# 实测: 形状不匹配: PyTorch (2, 2, 3, 2, 2) vs TensorFlow (2, 2, 2, 4)
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.MaxPooling2D] Output shape mismatch under equivalent migration in MaxPool2d operator (sample1)

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

When mapping `torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)` to `tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')`:
1. NCHW vs NHWC format mismatch causes pooling on wrong dimensions.
2. `padding=1` (fixed) vs `padding='same'` (dynamic) produce different output sizes.
3. `return_indices` is not supported in TF, causing tuple vs tensor shape difference.
Shape mismatch: PyTorch (2, 2, 3, 2, 2) vs TensorFlow (2, 2, 2, 4).

Expected behavior: Migration should handle data format conversion, map padding correctly, and document `return_indices` limitation.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)

pool_pt = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
out_pt = pool_pt(torch.tensor(input_np))

pool_tf = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
out_tf = pool_tf(tf.constant(input_np))

print(f"PyTorch: {[x.shape for x in out_pt]}, TensorFlow: {out_tf.shape}")
# Shape mismatch: PyTorch (2, 2, 3, 2, 2) vs TensorFlow (2, 2, 2, 4)
```

**Relevant log output**

```
comparison_error: Shape mismatch: PyTorch (2, 2, 3, 2, 2) vs TensorFlow (2, 2, 2, 4)
```

## Issue 049

llm_enhanced_torch_nn_MaxPool2d_20251215_205114.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.MaxPool2d] Output difference anomaly under equivalent migration in MaxPool2d operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for the `torch.nn.MaxPool2d` operator (Maximum difference: 0.819435715675354).

In this sample, TensorFlow correctly uses `data_format='channels_first'`, resolving the NCHW/NHWC format issue. However, the **padding semantics** still differ:
- PyTorch `padding=1`: fixed 1-pixel zero-padding on each side (explicit padding).
- TensorFlow `padding='same'`: dynamic auto-padding to ensure output_size = ceil(input_size / stride). The padding placement (asymmetric) and amount differ from fixed padding.

- Input: shape=[2, 3, 4, 4], dtype=float32
- Parameters: kernel_size=3, stride=2, padding=1 (PT) / 'same' (TF), data_format='channels_first'
- Maximum difference: 0.819435715675354

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)

# PyTorch (padding=1, fixed zero-padding)
pool_pt = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=False)
out_pt = pool_pt(torch.tensor(input_np))

# TensorFlow (padding='same', dynamic auto-padding, data_format correct)
pool_tf = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', data_format='channels_first')
out_tf = pool_tf(tf.constant(input_np))

pt_np = out_pt.detach().numpy()
tf_np = out_tf.numpy()
max_diff = np.max(np.abs(pt_np.astype(np.float64) - tf_np.astype(np.float64)))
print(f"PyTorch output shape: {pt_np.shape}")
print(f"TensorFlow output shape: {tf_np.shape}")
print(f"Maximum difference: {max_diff}")
# padding=1 vs padding='same' 产生不同的填充策略
# 实测最大差异: 0.819435715675354
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.MaxPooling2D] Output difference anomaly under equivalent migration in MaxPool2d operator (sample2)

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

When mapping `torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)` to `tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', data_format='channels_first')`, the data format is correctly handled but padding semantics differ. PyTorch `padding=1` applies symmetric fixed 1-pixel zero-padding on all sides, while TF `padding='same'` uses asymmetric dynamic auto-padding. Maximum difference: 0.819435715675354.

Expected behavior: Migration should use explicit padding (e.g., `tf.pad` + `padding='valid'`) instead of `padding='same'` to replicate PyTorch's fixed padding behavior.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)

pool_pt = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
out_pt = pool_pt(torch.tensor(input_np))

pool_tf = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', data_format='channels_first')
out_tf = pool_tf(tf.constant(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 0.819435715675354
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.819435715675354
```

## Issue 050

llm_enhanced_torch_nn_MaxPool2d_20251215_205114.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.MaxPool2d] Output shape mismatch under equivalent migration in MaxPool2d operator (sample4)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a shape mismatch was detected for the `torch.nn.MaxPool2d` operator. PyTorch output shape: (2, 3, 3, 3), TensorFlow output shape: (2, 3, 2, 2).

In this sample, TensorFlow correctly uses `data_format='channels_first'`. However, the **padding semantics** differ:
- PyTorch `padding=1` with `kernel_size=2, stride=2`: explicit 1-pixel zero-padding → output size = $\lfloor(4 + 2 \times 1 - 2) / 2\rfloor + 1 = 3$.
- TensorFlow `padding='same'` with `pool_size=2, strides=2`: auto-padding → output size = $\lceil 4 / 2 \rceil = 2$.

The different padding strategies produce different output spatial sizes: (3, 3) vs (2, 2).

- Input: shape=[2, 3, 4, 4], dtype=float32
- Parameters: kernel_size=2, stride=2, padding=1 (PT) / 'same' (TF), data_format='channels_first'
- Shape mismatch: PyTorch (2, 3, 3, 3) vs TensorFlow (2, 3, 2, 2)

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)

# PyTorch (padding=1, kernel_size=2, stride=2 → output 3x3)
pool_pt = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1, return_indices=False)
out_pt = pool_pt(torch.tensor(input_np))

# TensorFlow (padding='same', pool_size=2, strides=2 → output 2x2)
pool_tf = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first')
out_tf = pool_tf(tf.constant(input_np))

print(f"PyTorch output shape: {out_pt.shape}")    # (2, 3, 3, 3)
print(f"TensorFlow output shape: {out_tf.shape}")  # (2, 3, 2, 2)
# padding=1 导致 PyTorch 输出 3x3，padding='same' 导致 TF 输出 2x2
# 实测: 形状不匹配: PyTorch (2, 3, 3, 3) vs TensorFlow (2, 3, 2, 2)
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.MaxPooling2D] Output shape mismatch under equivalent migration in MaxPool2d operator (sample4)

**Have you reproduced the bug with TensorFlow Nightly?*** 

Yes

**Source*** 

source

**TensorFlow version*** 

tf 2.19

**Custom code*** 

Yes

**OS platform and distribution** 

Windows 11

**Mobile device** 

*No response*

**Python version** 

3.10.18

**Bazel version** 

*No response*

**GCC/compiler version** 

*No response*

**CUDA/cuDNN version** 

*No response*

**GPU model and memory** 

*No response*

**Current behavior?*** 

When mapping `torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)` to `tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first')`, the output shapes differ: PyTorch (2, 3, 3, 3) vs TensorFlow (2, 3, 2, 2).

- PyTorch `padding=1` adds explicit 1-pixel padding: output_size = floor((4 + 2*1 - 2) / 2) + 1 = 3.
- TensorFlow `padding='same'` uses auto-padding: output_size = ceil(4 / 2) = 2.

Expected behavior: Migration should use explicit padding via `tf.pad` before pooling with `padding='valid'` to replicate PyTorch's fixed padding behavior, rather than using `padding='same'`.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)

pool_pt = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
out_pt = pool_pt(torch.tensor(input_np))

pool_tf = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first')
out_tf = pool_tf(tf.constant(input_np))

print(f"PyTorch: {out_pt.shape}, TensorFlow: {out_tf.shape}")
# Shape mismatch: PyTorch (2, 3, 3, 3) vs TensorFlow (2, 3, 2, 2)
```

**Relevant log output**

```
comparison_error: Shape mismatch: PyTorch (2, 3, 3, 3) vs TensorFlow (2, 3, 2, 2)
```
"""

# 追加到文件
with open(output_file, 'a', encoding='utf-8') as f:
    f.write(content)

print(f"Issues 025-050 已成功追加到 {output_file}")
print(f"追加内容长度: {len(content)} 字符")
