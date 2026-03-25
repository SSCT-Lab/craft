#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 Issue 111-138 的标准格式内容并追加到修改版文件中"""

output_path = r"d:\graduate\DFrameworkTest\issues\138个跨表不一致Case的GitHub Issue-修改版.md"

content = r"""
## Issue 111

llm_enhanced_torch_randperm_20251202_133813.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][randperm] Output difference anomaly under equivalent migration in randperm operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.randperm` (Maximum difference: 11136).

The root cause is **RNG implementation difference**: `torch.randperm(n)` and `paddle.randperm(n)` both generate random permutations of [0, n-1], but they use different internal random number generation algorithms. Even with the same seed value, the underlying RNG implementations (Philox in PyTorch vs MT19937 in Paddle) produce completely different sequences. For n=11339, the maximum possible difference is 11338, and the observed 11136 is consistent with two independent random permutations.

- n: 11339
- Maximum difference: 11136

```python
import numpy as np
import torch
import paddle

torch.manual_seed(42)
out_pt = torch.randperm(11339)

paddle.seed(42)
out_pd = paddle.randperm(11339)

max_diff = np.max(np.abs(out_pt.numpy().astype(np.int64) - out_pd.numpy().astype(np.int64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 11136
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][randperm] Output difference anomaly under equivalent migration in randperm operator (sample2)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`randperm` 算子的最大输出差异为 11136。

根本原因是**随机数生成器（RNG）底层实现不同**：`torch.randperm(n)` 和 `paddle.randperm(n)` 都生成 [0, n-1] 的随机排列，但使用不同的内部 RNG 算法（PyTorch 使用 Philox，Paddle 使用 MT19937）。即使设置相同的种子值，两个框架也会产生完全不同的排列序列。

- n: 11339

```python
import numpy as np
import torch
import paddle

torch.manual_seed(42)
out_pt = torch.randperm(11339)

paddle.seed(42)
out_pd = paddle.randperm(11339)

max_diff = np.max(np.abs(out_pt.numpy().astype(np.int64) - out_pd.numpy().astype(np.int64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 11136
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.randperm",
  "n": 11339
}
```

- 随机排列生成算子，RNG 底层实现差异导致排列完全不同。

## Issue 112

llm_enhanced_torch_randperm_20251202_133813.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][randperm] Output difference anomaly under equivalent migration in randperm operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.randperm` (Maximum difference: 1990).

Same root cause as Issue 111: RNG implementation difference. `torch.randperm(n)` and `paddle.randperm(n)` use different internal RNG algorithms, producing completely different permutations even with the same seed. For n=2024, the observed maximum difference of 1990 is consistent with two independent random permutations.

- n: 2024
- Maximum difference: 1990

```python
import numpy as np
import torch
import paddle

torch.manual_seed(42)
out_pt = torch.randperm(2024)

paddle.seed(42)
out_pd = paddle.randperm(2024)

max_diff = np.max(np.abs(out_pt.numpy().astype(np.int64) - out_pd.numpy().astype(np.int64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 1990
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][randperm] Output difference anomaly under equivalent migration in randperm operator (sample3)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`randperm` 算子的最大输出差异为 1990。

与 Issue 111 相同原因：RNG 底层实现不同。对于 n=2024，观察到的最大差异 1990 符合两个独立随机排列的预期。

- n: 2024

```python
import numpy as np
import torch
import paddle

torch.manual_seed(42)
out_pt = torch.randperm(2024)

paddle.seed(42)
out_pd = paddle.randperm(2024)

max_diff = np.max(np.abs(out_pt.numpy().astype(np.int64) - out_pd.numpy().astype(np.int64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 1990
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.randperm",
  "n": 2024
}
```

## Issue 113

llm_enhanced_torch_randperm_20251202_133813.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][randperm] Output difference anomaly under equivalent migration in randperm operator (sample4)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, an output discrepancy was detected for `torch.randperm` (Maximum difference: 3).

Same root cause as Issue 111: RNG implementation difference. For n=5, the maximum possible difference is 4, and the observed 3 confirms two independent random permutations.

- n: 5
- Maximum difference: 3

```python
import numpy as np
import torch
import paddle

torch.manual_seed(42)
out_pt = torch.randperm(5)

paddle.seed(42)
out_pd = paddle.randperm(5)

max_diff = np.max(np.abs(out_pt.numpy().astype(np.int64) - out_pd.numpy().astype(np.int64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 3
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][randperm] Output difference anomaly under equivalent migration in randperm operator (sample4)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`randperm` 算子的最大输出差异为 3。

与 Issue 111 相同原因：RNG 底层实现不同。对于 n=5，最大可能差异为 4，观察到 3 符合预期。

- n: 5

```python
import numpy as np
import torch
import paddle

torch.manual_seed(42)
out_pt = torch.randperm(5)

paddle.seed(42)
out_pd = paddle.randperm(5)

max_diff = np.max(np.abs(out_pt.numpy().astype(np.int64) - out_pd.numpy().astype(np.int64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 3
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.randperm",
  "n": 5
}
```

## Issue 114

llm_enhanced_torch_range_20251202_132606.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][range] Shape mismatch under equivalent migration in range operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a **shape mismatch** was detected for `torch.range`: PyTorch output shape (20,) vs PaddlePaddle output shape (19,).

The root cause is **closed vs half-open interval semantics**: PyTorch's `torch.range(start, end)` generates values in the **closed interval** [start, end], including the endpoint. PaddlePaddle's `paddle.arange(start, end)` generates values in the **half-open interval** [start, end), excluding the endpoint. This is a fundamental API semantic difference.

- start: 1, end: 20
- PyTorch output: [1, 2, ..., 20] (20 elements, closed interval)
- Paddle output: [1, 2, ..., 19] (19 elements, half-open interval)
- Error: Shape mismatch: (20,) vs (19,)

```python
import numpy as np
import torch
import paddle

# PyTorch: torch.range 是闭区间 [start, end]
out_pt = torch.range(1, 20)

# PaddlePaddle: paddle.arange 是左闭右开 [start, end)
out_pd = paddle.arange(1, 20)

print(f"PyTorch shape: {out_pt.shape}, values: {out_pt}")
print(f"Paddle shape: {out_pd.shape}, values: {out_pd}")
# Shape mismatch: PyTorch (20,) vs PaddlePaddle (19,)

# 正确迁移：paddle.arange(start, end + step, step)
out_pd_correct = paddle.arange(1, 21)
max_diff = np.max(np.abs(out_pt.numpy().astype(np.float64) - out_pd_correct.numpy().astype(np.float64)))
print(f"Corrected max_diff: {max_diff}")
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][paddle.arange] Shape mismatch under equivalent migration in range operator (sample1)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`range`/`arange` 算子出现**形状不匹配**：PyTorch 输出 (20,) vs Paddle 输出 (19,)。

根本原因是**闭区间 vs 左闭右开语义差异**：PyTorch 的 `torch.range(start, end)` 生成 [start, end] 闭区间序列（包含 end），而 Paddle 的 `paddle.arange(start, end)` 生成 [start, end) 左闭右开序列（不包含 end）。

迁移时应使用 `paddle.arange(start, end + step, step)` 来对齐行为。

- start: 1, end: 20
- PyTorch: [1, 2, ..., 20]，共 20 个元素
- Paddle: [1, 2, ..., 19]，共 19 个元素

```python
import numpy as np
import torch
import paddle

# PyTorch: 闭区间
out_pt = torch.range(1, 20)

# Paddle: 左闭右开
out_pd = paddle.arange(1, 20)

print(f"PyTorch shape: {out_pt.shape}")  # (20,)
print(f"Paddle shape: {out_pd.shape}")   # (19,)
# Shape mismatch: (20,) vs (19,)

# 正确迁移
out_pd_correct = paddle.arange(1, 21)
max_diff = np.max(np.abs(out_pt.numpy().astype(np.float64) - out_pd_correct.numpy().astype(np.float64)))
print(f"Corrected max_diff: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.arange",
  "start": 1,
  "end": 20
}
```

- `torch.range` 闭区间 vs `paddle.arange` 左闭右开，迁移时需 end+step。

## Issue 115

llm_enhanced_torch_range_20251202_132606.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][range] Shape mismatch under equivalent migration in range operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a **shape mismatch** was detected for `torch.range`: PyTorch output shape (43,) vs PaddlePaddle output shape (42,).

Same root cause as Issue 114: **closed vs half-open interval semantics**. PyTorch's `torch.range(start=-10.5, end=10.5, step=0.5)` includes the endpoint 10.5, producing 43 elements. PaddlePaddle's `paddle.arange(start=-10.5, end=10.5, step=0.5)` excludes the endpoint, producing 42 elements.

- start: -10.5, end: 10.5, step: 0.5
- PyTorch: [-10.5, -10.0, ..., 10.0, 10.5] (43 elements)
- Paddle: [-10.5, -10.0, ..., 10.0] (42 elements)
- Error: Shape mismatch: (43,) vs (42,)

```python
import numpy as np
import torch
import paddle

# PyTorch: 闭区间
out_pt = torch.range(-10.5, 10.5, 0.5)

# Paddle: 左闭右开
out_pd = paddle.arange(-10.5, 10.5, 0.5)

print(f"PyTorch shape: {out_pt.shape}")  # (43,)
print(f"Paddle shape: {out_pd.shape}")   # (42,)

# 正确迁移：end + step
out_pd_correct = paddle.arange(-10.5, 11.0, 0.5)
max_diff = np.max(np.abs(out_pt.numpy().astype(np.float64) - out_pd_correct.numpy().astype(np.float64)))
print(f"Corrected max_diff: {max_diff}")
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][paddle.arange] Shape mismatch under equivalent migration in range operator (sample2)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`range`/`arange` 算子出现**形状不匹配**：PyTorch 输出 (43,) vs Paddle 输出 (42,)。

与 Issue 114 相同原因：`torch.range` 闭区间 vs `paddle.arange` 左闭右开。参数 start=-10.5, end=10.5, step=0.5，PyTorch 包含 end=10.5 生成 43 个元素，Paddle 不包含 end 生成 42 个元素。

```python
import numpy as np
import torch
import paddle

out_pt = torch.range(-10.5, 10.5, 0.5)
out_pd = paddle.arange(-10.5, 10.5, 0.5)

print(f"PyTorch shape: {out_pt.shape}")  # (43,)
print(f"Paddle shape: {out_pd.shape}")   # (42,)

# 正确迁移
out_pd_correct = paddle.arange(-10.5, 11.0, 0.5)
max_diff = np.max(np.abs(out_pt.numpy().astype(np.float64) - out_pd_correct.numpy().astype(np.float64)))
print(f"Corrected max_diff: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.arange",
  "start": -10.5,
  "end": 10.5,
  "step": 0.5
}
```

## Issue 116

llm_enhanced_torch_std_20251215_210017.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][std] Output difference anomaly under equivalent migration in std operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for `torch.std` (Maximum difference: nan).

The root cause is **ddof (degrees of freedom) default value mismatch**:

1. **PyTorch** `torch.std` defaults to Bessel's correction (`correction=1`, i.e., divides by N-1), computing the **sample standard deviation** (unbiased).
2. **TensorFlow** `tf.math.reduce_std` defaults to `ddof=0`, computing the **population standard deviation** (biased, divides by N).

When `dim=[0, 2]` is applied to a shape [1, 2048, 1] tensor, the reduced dimensions have size 1, making N=1. PyTorch divides by N-1=0, producing **NaN**, while TensorFlow divides by N=1, producing **0.0**. This creates a NaN-based mismatch.

- Input: shape=[1, 2048, 1], dtype=float64
- PT params: dim=[0, 2] (default correction=1)
- TF params: axis=[0, 2] (default ddof=0)
- Maximum difference: nan

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 2048, 1).astype(np.float64)

# PyTorch: 默认 correction=1 (无偏估计，除以 N-1)
out_pt = torch.std(torch.tensor(input_np), dim=[0, 2])

# TensorFlow: 默认除以 N (有偏估计)
out_tf = tf.math.reduce_std(tf.constant(input_np), axis=[0, 2])

pt_np = out_pt.detach().numpy()
tf_np = out_tf.numpy()
max_diff = np.max(np.abs(pt_np - tf_np))
print(f"PyTorch output (first 5): {pt_np[:5]}")
print(f"TensorFlow output (first 5): {tf_np[:5]}")
print(f"Maximum difference: {max_diff}")
# dim=[0,2] 对 shape[1,2048,1] 做归约，N=1*1=1
# PyTorch: 除以 N-1=0 -> NaN; TF: 除以 N=1 -> 0.0
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.math.reduce_std] Output difference anomaly under equivalent migration in std operator (sample2)

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

When mapping `torch.std(input, dim=[0, 2])` to `tf.math.reduce_std(input, axis=[0, 2])`, the outputs differ significantly (Maximum difference: nan). PyTorch's `torch.std` defaults to Bessel's correction (divides by N-1), while TensorFlow's `tf.math.reduce_std` defaults to population std (divides by N). When the reduced dimension size is 1, PyTorch produces NaN (division by 0) while TensorFlow produces 0.

Expected behavior: Migration should explicitly compute unbiased std in TF: `tf.math.reduce_std(...) * sqrt(N/(N-1))` or use manual calculation with ddof=1 alignment.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 2048, 1).astype(np.float64)

out_pt = torch.std(torch.tensor(input_np), dim=[0, 2])
out_tf = tf.math.reduce_std(tf.constant(input_np), axis=[0, 2])

max_diff = np.max(np.abs(out_pt.detach().numpy() - out_tf.numpy()))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: nan
```

## Issue 117

llm_enhanced_torch_std_20251215_210017.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][std] Output difference anomaly under equivalent migration in std operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for `torch.std` (Maximum difference: 0.0005023479461669922).

Same root cause as Issue 116: **ddof default value mismatch**. PyTorch's `torch.std` defaults to Bessel's correction (divides by N-1), TensorFlow's `tf.math.reduce_std` defaults to population std (divides by N). For input shape [1024] with dim=0, N=1024, the bias ratio is $\sqrt{\frac{N}{N-1}} = \sqrt{\frac{1024}{1023}} \approx 1.000489$, producing a small but systematic difference.

- Input: shape=[1024], dtype=float32
- PT params: dim=0 (default correction=1)
- TF params: axis=0 (default ddof=0)
- Maximum difference: 0.0005023479461669922

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1024).astype(np.float32)

out_pt = torch.std(torch.tensor(input_np), dim=0)
out_tf = tf.math.reduce_std(tf.constant(input_np), axis=0)

max_diff = np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64))
print(f"PyTorch std: {out_pt.item()}")
print(f"TensorFlow std: {out_tf.numpy()}")
print(f"Maximum difference: {max_diff}")
# ddof=1 vs ddof=0 导致系统性差异
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.math.reduce_std] Output difference anomaly under equivalent migration in std operator (sample3)

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

Same as Issue 116. PyTorch's `torch.std` defaults to unbiased estimation (divides by N-1), while TensorFlow's `tf.math.reduce_std` defaults to biased estimation (divides by N). For shape [1024] with dim=0, the difference is 0.0005023479461669922.

Expected behavior: Same as Issue 116 — migration should account for the ddof difference.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1024).astype(np.float32)

out_pt = torch.std(torch.tensor(input_np), dim=0)
out_tf = tf.math.reduce_std(tf.constant(input_np), axis=0)

max_diff = np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.0005023479461669922
```

## Issue 118

llm_enhanced_torch_std_20251215_210017.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][std] Output difference anomaly under equivalent migration in std operator (sample4)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for `torch.std` (Maximum difference: 0.0005277395248413086).

Same root cause as Issue 116: **ddof default value mismatch**. For input shape [512, 1024] with dim=1, N=1024, the difference magnitude is consistent with the bias ratio $\sqrt{N/(N-1)}$.

- Input: shape=[512, 1024], dtype=float32
- PT params: dim=1 (default correction=1)
- TF params: axis=1 (default ddof=0)
- Maximum difference: 0.0005277395248413086

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(512, 1024).astype(np.float32)

out_pt = torch.std(torch.tensor(input_np), dim=1)
out_tf = tf.math.reduce_std(tf.constant(input_np), axis=1)

pt_np = out_pt.detach().numpy().astype(np.float64)
tf_np = out_tf.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - tf_np))
print(f"Maximum difference: {max_diff}")
# ddof=1 vs ddof=0
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.math.reduce_std] Output difference anomaly under equivalent migration in std operator (sample4)

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

Same as Issue 116. PyTorch's `torch.std` defaults to unbiased estimation (divides by N-1), while TensorFlow's `tf.math.reduce_std` defaults to biased estimation (divides by N). For shape [512, 1024] with dim=1, the maximum difference is 0.0005277395248413086.

Expected behavior: Same as Issue 116.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(512, 1024).astype(np.float32)

out_pt = torch.std(torch.tensor(input_np), dim=1)
out_tf = tf.math.reduce_std(tf.constant(input_np), axis=1)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.0005277395248413086
```

## Issue 119

llm_enhanced_torch_sub_20251215_165148.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][sub] Output difference anomaly under equivalent migration in sub operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a severe output discrepancy was detected for `torch.sub` (Maximum difference: 4.318345546722412).

**Two distinct problems:**

1. **Input data not synchronized**: The PyTorch and TensorFlow test inputs contain completely different values (different random seeds or data sources). PyTorch input starts with [0.0409, -2.2117, ...] while TensorFlow input starts with [-3.3240, -0.9614, ...].
2. **alpha parameter not migrated**: PyTorch's `torch.sub(input, other, alpha=0.5)` computes `input - alpha * other`, while TensorFlow's `tf.subtract(x, y)` only computes `x - y` without alpha support. The alpha scaling must be applied manually: `tf.subtract(x, alpha * y)`.

- Input: shape=[3, 1, 5], other: shape=[5, 5], dtype=float32
- PT params: alpha=0.5
- TF params: no alpha support
- Maximum difference: 4.318345546722412

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(3, 1, 5).astype(np.float32)
other_np = np.random.randn(5, 5).astype(np.float32)
alpha = 0.5

# PyTorch: input - alpha * other
out_pt = torch.sub(torch.tensor(input_np), torch.tensor(other_np), alpha=alpha)

# TensorFlow: 需要手动实现 alpha 缩放
# 错误做法: tf.subtract(x, y) 不支持 alpha
out_tf_wrong = tf.subtract(tf.constant(input_np), tf.constant(other_np))

# 正确做法: tf.subtract(x, alpha * y)
out_tf_correct = tf.subtract(tf.constant(input_np), alpha * tf.constant(other_np))

pt_np = out_pt.detach().numpy().astype(np.float64)
tf_wrong_np = out_tf_wrong.numpy().astype(np.float64)
tf_correct_np = out_tf_correct.numpy().astype(np.float64)

print(f"Wrong migration max_diff: {np.max(np.abs(pt_np - tf_wrong_np))}")
print(f"Correct migration max_diff: {np.max(np.abs(pt_np - tf_correct_np))}")
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.subtract] Output difference anomaly under equivalent migration in sub operator (sample3)

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

When mapping `torch.sub(input, other, alpha=0.5)` to `tf.subtract(x, y)`, the outputs differ significantly (Maximum difference: 4.318345546722412). Two issues:
1. The input data was not synchronized between frameworks.
2. PyTorch's `alpha` parameter (computing `input - alpha * other`) has no equivalent in `tf.subtract`, which only computes `x - y`.

Expected behavior: Migration should synchronize inputs and manually apply alpha: `tf.subtract(x, alpha * y)`.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(3, 1, 5).astype(np.float32)
other_np = np.random.randn(5, 5).astype(np.float32)
alpha = 0.5

out_pt = torch.sub(torch.tensor(input_np), torch.tensor(other_np), alpha=alpha)
out_tf = tf.subtract(tf.constant(input_np), tf.constant(other_np))  # 缺少 alpha

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 4.318345546722412
```

## Issue 120

llm_enhanced_torch_sub_20251215_165148.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][sub] Output difference anomaly under equivalent migration in sub operator (sample4)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a severe output discrepancy was detected for `torch.sub` (Maximum difference: 9.868673502993001).

Same root cause as Issue 119:

1. **Input data not synchronized**: PyTorch and TensorFlow received completely different input tensors.
2. **alpha parameter not migrated**: PyTorch computes `input - alpha * other` with `alpha=2`, while `tf.subtract` only computes `x - y`.

- Input: shape=[5, 5, 5], other: shape=[5, 5], dtype=float64
- PT params: alpha=2
- Maximum difference: 9.868673502993001

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(5, 5, 5).astype(np.float64)
other_np = np.random.randn(5, 5).astype(np.float64)
alpha = 2

# PyTorch: input - alpha * other
out_pt = torch.sub(torch.tensor(input_np), torch.tensor(other_np), alpha=alpha)

# TensorFlow: 正确迁移需手动实现 alpha
out_tf_correct = tf.subtract(tf.constant(input_np), alpha * tf.constant(other_np))

pt_np = out_pt.detach().numpy()
tf_np = out_tf_correct.numpy()
max_diff = np.max(np.abs(pt_np - tf_np))
print(f"Maximum difference: {max_diff}")
# 输入同步后应为 0.0
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.subtract] Output difference anomaly under equivalent migration in sub operator (sample4)

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

Same as Issue 119. PyTorch's `torch.sub(input, other, alpha=2)` computes `input - 2 * other`, but `tf.subtract(x, y)` only computes `x - y`. Combined with unsynchronized input data, the maximum difference reaches 9.868673502993001.

Expected behavior: Same as Issue 119 — synchronize inputs and apply alpha manually.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(5, 5, 5).astype(np.float64)
other_np = np.random.randn(5, 5).astype(np.float64)
alpha = 2

out_pt = torch.sub(torch.tensor(input_np), torch.tensor(other_np), alpha=alpha)
out_tf = tf.subtract(tf.constant(input_np), tf.constant(other_np))  # 缺少 alpha

max_diff = np.max(np.abs(out_pt.detach().numpy() - out_tf.numpy()))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 9.868673502993001
```

## Issue 121

llm_enhanced_torch_svd_20251215_205358.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][svd] Comparison error under equivalent migration in svd operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a **comparison process error** was detected for `torch.svd`: "setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (3, 2) + inhomogeneous part."

The root cause is **different SVD return structure**:

1. **PyTorch** `torch.svd(input)` returns a namedtuple `(U, S, V)` where U: (2, 2), S: (2,), V: (2, 2) — all tensors of different shapes.
2. **TensorFlow** `tf.linalg.svd(input)` returns `(S, U, V)` by default — note the different order (S first). Also, when `compute_uv=True`, it returns `(s, u, v)`.

The comparison framework tries to stack these heterogeneous return values into a single array, causing the inhomogeneous shape error.

- Input: shape=[2, 2], dtype=float32
- Error: comparison process error (inhomogeneous shape)

```python
import numpy as np
import torch
import tensorflow as tf

input_data = np.array([[-0.6722479462623596, -0.11334283649921417],
                        [-1.4957332611083984, -0.48832809925079346]], dtype=np.float32)

# PyTorch: 返回 (U, S, V)
U_pt, S_pt, V_pt = torch.svd(torch.tensor(input_data))

# TensorFlow: 返回 (S, U, V) — 注意顺序不同
S_tf, U_tf, V_tf = tf.linalg.svd(tf.constant(input_data))

# 逐项比较奇异值
max_diff_S = np.max(np.abs(S_pt.numpy().astype(np.float64) - S_tf.numpy().astype(np.float64)))
print(f"Singular values max diff: {max_diff_S}")
# 注意：U 和 V 可能存在符号翻转，需要比较 U @ diag(S) @ V^T
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.linalg.svd] Comparison error under equivalent migration in svd operator (sample3)

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

When mapping `torch.svd(input)` to `tf.linalg.svd(input)`, the comparison process fails with "setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions."

Two issues:
1. **Return order differs**: PyTorch returns (U, S, V), TensorFlow returns (S, U, V).
2. **Inhomogeneous shapes**: U is (2,2), S is (2,), V is (2,2) — the comparison framework cannot stack these into a single array.

Expected behavior: Compare SVD results component-wise, accounting for return order difference and potential sign ambiguity in U/V.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

input_data = np.array([[-0.6722479462623596, -0.11334283649921417],
                        [-1.4957332611083984, -0.48832809925079346]], dtype=np.float32)

U_pt, S_pt, V_pt = torch.svd(torch.tensor(input_data))
S_tf, U_tf, V_tf = tf.linalg.svd(tf.constant(input_data))

# 尝试将不同形状的返回值放入同一数组会报错
try:
    pt_result = np.array([U_pt.numpy(), S_pt.numpy(), V_pt.numpy()])
except ValueError as e:
    print(f"Error: {e}")
```

**Relevant log output**

```
comparison_error: setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (3, 2) + inhomogeneous part.
```

## Issue 122

llm_enhanced_torch_svd_20251215_205358.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][svd] Comparison error under equivalent migration in svd operator (sample4)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a **comparison process error** was detected for `torch.svd`, identical to Issue 121.

Same root cause: **different SVD return structure** (PyTorch returns (U, S, V), TensorFlow returns (S, U, V)) and inhomogeneous tensor shapes causing array construction failure.

- Input: shape=[2, 2], dtype=float32, values=[0.7734, -1.7030, -0.9357, 1.0658]
- Error: comparison process error (inhomogeneous shape)

```python
import numpy as np
import torch
import tensorflow as tf

input_data = np.array([[0.773365318775177, -1.7030305862426758],
                        [-0.935749351978302, 1.0658057928085327]], dtype=np.float32)

U_pt, S_pt, V_pt = torch.svd(torch.tensor(input_data))
S_tf, U_tf, V_tf = tf.linalg.svd(tf.constant(input_data))

max_diff_S = np.max(np.abs(S_pt.numpy().astype(np.float64) - S_tf.numpy().astype(np.float64)))
print(f"Singular values max diff: {max_diff_S}")
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.linalg.svd] Comparison error under equivalent migration in svd operator (sample4)

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

Same as Issue 121. `torch.svd` returns (U, S, V) while `tf.linalg.svd` returns (S, U, V). The comparison framework fails to handle the inhomogeneous return shape.

Expected behavior: Same as Issue 121.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

input_data = np.array([[0.773365318775177, -1.7030305862426758],
                        [-0.935749351978302, 1.0658057928085327]], dtype=np.float32)

U_pt, S_pt, V_pt = torch.svd(torch.tensor(input_data))
S_tf, U_tf, V_tf = tf.linalg.svd(tf.constant(input_data))

print(f"PT: U={U_pt.shape}, S={S_pt.shape}, V={V_pt.shape}")
print(f"TF: S={S_tf.shape}, U={U_tf.shape}, V={V_tf.shape}")
```

**Relevant log output**

```
comparison_error: setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (3, 2) + inhomogeneous part.
```

## Issue 123

llm_enhanced_torch_tanh_20251216_010955.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][tanh] Output difference anomaly under equivalent migration in tanh operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, a small output discrepancy was detected for `torch.tanh` (Maximum difference: 4.5359134674072266e-05).

The root cause is **float32 computation precision difference**: Both frameworks implement the same mathematical function $\tanh(x) = \frac{e^{2x}-1}{e^{2x}+1}$, but the difference (~4.5e-5) arises from different underlying implementations (SIMD vectorization paths, intermediate rounding strategies, or library-specific optimizations in cuDNN/MKL vs MindSpore's own kernel).

- Input: shape=[100, 3, 28, 28], dtype=float32
- Maximum difference: 4.5359134674072266e-05

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(100, 3, 28, 28).astype(np.float32)

out_pt = torch.tanh(torch.tensor(input_np))
out_ms = mindspore.ops.tanh(mindspore.Tensor(input_np))

pt_np = out_pt.detach().numpy().astype(np.float64)
ms_np = out_ms.asnumpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - ms_np))
print(f"Maximum difference: {max_diff}")
# Float32 精度误差，属于底层实现差异
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.ops.tanh] Output difference anomaly under equivalent migration in tanh operator (sample2)

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

When mapping `torch.tanh(input)` to `mindspore.ops.tanh(input)` with identical float32 input (shape=[100, 3, 28, 28]), the outputs differ by up to 4.5359134674072266e-05. This is a float32 computation precision difference caused by different underlying kernel implementations.

**Describe the expected behavior***

Both frameworks implement the same mathematical tanh function. The ~4.5e-5 difference is within typical float32 precision bounds and reflects inherent floating-point non-determinism across different framework backends.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(100, 3, 28, 28).astype(np.float32)

out_pt = torch.tanh(torch.tensor(input_np))
out_ms = mindspore.ops.tanh(mindspore.Tensor(input_np))

pt_np = out_pt.detach().numpy().astype(np.float64)
ms_np = out_ms.asnumpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - ms_np))
print(f"Maximum difference: {max_diff}")
```

**Related log / screenshot**

```
comparison_error: Numerical mismatch, maximum difference: 4.5359134674072266e-05
```

**Special notes for this issue**

Float32 tanh 精度误差，差异量级 ~$10^{-5}$，属于底层实现差异，非功能性 bug。

## Issue 124

llm_enhanced_torch_tensordot_20251125_141922.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][tensordot] Output difference anomaly under equivalent migration in tensordot operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.tensordot` (Maximum difference: 16.279010772705078).

The root cause is **input data not synchronized**: The bug file shows PyTorch's input tensor `a` starts with [0.2494, 1.5775, ...] while PaddlePaddle's input tensor `x` starts with [0.6328, 2.2707, ...] — completely different values. This indicates the test harness did not fix random seeds, causing different random inputs to be fed to each framework.

- Input a: shape=[5, 6, 7], b: shape=[6, 5, 4], dtype=float32
- PT params: dims=[[1, 0], [0, 1]]
- PD params: axes=[[1, 0], [0, 1]]
- Maximum difference: 16.279010772705078

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
a_np = np.random.randn(5, 6, 7).astype(np.float32)
b_np = np.random.randn(6, 5, 4).astype(np.float32)

out_pt = torch.tensordot(torch.tensor(a_np), torch.tensor(b_np), dims=[[1, 0], [0, 1]])
out_pd = paddle.tensordot(paddle.to_tensor(a_np), paddle.to_tensor(b_np), axes=[[1, 0], [0, 1]])

pt_np = out_pt.detach().numpy().astype(np.float64)
pd_np = out_pd.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"Maximum difference: {max_diff}")
# 输入同步后差异应在 float32 精度范围内
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][paddle.tensordot] Output difference anomaly under equivalent migration in tensordot operator (sample2)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`tensordot` 算子的最大输出差异为 16.279010772705078。

根本原因是**输入数据未同步**：PyTorch 和 Paddle 的输入张量包含完全不同的随机数值，说明测试框架未固定随机种子。当输入对齐后，两框架的 tensordot 输出应在 float32 精度范围内一致。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
a_np = np.random.randn(5, 6, 7).astype(np.float32)
b_np = np.random.randn(6, 5, 4).astype(np.float32)

out_pt = torch.tensordot(torch.tensor(a_np), torch.tensor(b_np), dims=[[1, 0], [0, 1]])
out_pd = paddle.tensordot(paddle.to_tensor(a_np), paddle.to_tensor(b_np), axes=[[1, 0], [0, 1]])

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.tensordot",
  "x": {"shape": [5, 6, 7], "dtype": "float32"},
  "y": {"shape": [6, 5, 4], "dtype": "float32"},
  "axes": [[1, 0], [0, 1]]
}
```

- 输入数据未对齐导致大数值差异，同步输入后应无功能性差异。

## Issue 125

llm_enhanced_torch_topk_20251215_164432.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][topk] Output difference anomaly under equivalent migration in topk operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a severe output discrepancy was detected for `torch.topk` (Maximum difference: 1020.0).

The root cause is **`largest` parameter not supported by TensorFlow**:

1. **PyTorch** `torch.topk(input, k=1, dim=-1, largest=False, sorted=True)` returns the k **smallest** values (because `largest=False`).
2. **TensorFlow** `tf.math.top_k(input, k=1, sorted=True)` always returns the k **largest** values — it has no `largest` parameter.

To achieve `largest=False` semantics in TensorFlow, one must negate the input: `tf.math.top_k(-input, k)`, then negate the result values back.

- Input: shape=[3, 5000], dtype=float64
- PT params: k=1, dim=-1, largest=False, sorted=True
- TF params: k=1, sorted=True (always largest)
- Maximum difference: 1020.0

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(3, 5000).astype(np.float64)

# PyTorch: largest=False 返回最小值
values_pt, indices_pt = torch.topk(torch.tensor(input_np), k=1, dim=-1, largest=False, sorted=True)

# TensorFlow: 只支持最大值
values_tf, indices_tf = tf.math.top_k(tf.constant(input_np), k=1, sorted=True)

# 正确迁移: 取反后取 topk，再取反
values_tf_correct, indices_tf_correct = tf.math.top_k(-tf.constant(input_np), k=1, sorted=True)
values_tf_correct = -values_tf_correct

print(f"PT smallest: {values_pt.numpy()}")
print(f"TF largest (wrong): {values_tf.numpy()}")
print(f"TF smallest (correct): {values_tf_correct.numpy()}")
max_diff_wrong = np.max(np.abs(values_pt.numpy() - values_tf.numpy()))
max_diff_correct = np.max(np.abs(values_pt.numpy() - values_tf_correct.numpy()))
print(f"Wrong migration max_diff: {max_diff_wrong}")
print(f"Correct migration max_diff: {max_diff_correct}")
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.math.top_k] Output difference anomaly under equivalent migration in topk operator (sample2)

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

When mapping `torch.topk(input, k=1, dim=-1, largest=False, sorted=True)` to `tf.math.top_k(input, k=1, sorted=True)`, the outputs differ by 1020.0. PyTorch's `largest=False` returns the smallest k values, while `tf.math.top_k` always returns the largest k values — it does not support `largest=False`.

Expected behavior: Migration should negate the input before `tf.math.top_k`, then negate the result: `values, indices = tf.math.top_k(-input, k); values = -values`.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(3, 5000).astype(np.float64)

values_pt, _ = torch.topk(torch.tensor(input_np), k=1, dim=-1, largest=False, sorted=True)
values_tf, _ = tf.math.top_k(tf.constant(input_np), k=1, sorted=True)

max_diff = np.max(np.abs(values_pt.numpy() - values_tf.numpy()))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 1020.0
```

## Issue 126

llm_enhanced_torch_unique_20251202_135202.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][unique] Output difference anomaly under equivalent migration in unique operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.unique` (Maximum difference: 1.5436841316623915).

The root cause is **input data not synchronized**: The bug file shows the input tensors for PyTorch and Paddle contain different random values (random seeds not fixed). Both `torch.unique` and `paddle.unique` should produce the same sorted unique values when given identical input.

- Input: shape=[4], dtype=float64
- Maximum difference: 1.5436841316623915

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(4).astype(np.float64)

out_pt = torch.unique(torch.tensor(input_np))
out_pd = paddle.unique(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 输入同步后应为 0.0
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][paddle.unique] Output difference anomaly under equivalent migration in unique operator (sample3)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`unique` 算子的最大输出差异为 1.5436841316623915。

根本原因是**输入数据未同步**：PyTorch 和 Paddle 接收了不同的随机输入数据（未固定随机种子）。当输入对齐后，两框架的 unique 输出应完全一致。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(4).astype(np.float64)

out_pt = torch.unique(torch.tensor(input_np))
out_pd = paddle.unique(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.unique",
  "*args": [{"shape": [4], "dtype": "float64"}]
}
```

- 输入未对齐导致差异，同步后无功能性差异。

## Issue 127

llm_enhanced_torch_unique_20251202_135202.json_sample6.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][unique] Output difference anomaly under equivalent migration in unique operator (sample6)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.unique` (Maximum difference: 1.112005352973938).

Same root cause as Issue 126: **input data not synchronized**. The test harness did not fix random seeds.

- Input: shape=[2, 3, 4], dtype=float32
- PT params: sorted=True, return_inverse=False, return_counts=False
- PD params: return_index=False, return_inverse=False, return_counts=False, axis=None
- Maximum difference: 1.112005352973938

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 4).astype(np.float32)

out_pt = torch.unique(torch.tensor(input_np), sorted=True, return_inverse=False, return_counts=False)
out_pd = paddle.unique(paddle.to_tensor(input_np), return_index=False, return_inverse=False, return_counts=False)

pt_np = out_pt.numpy().astype(np.float64)
pd_np = out_pd.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"Maximum difference: {max_diff}")
# 输入同步后应接近 0
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][paddle.unique] Output difference anomaly under equivalent migration in unique operator (sample6)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`unique` 算子的最大输出差异为 1.112005352973938。

与 Issue 126 相同原因：输入数据未同步（随机种子未固定）。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 4).astype(np.float32)

out_pt = torch.unique(torch.tensor(input_np), sorted=True, return_inverse=False, return_counts=False)
out_pd = paddle.unique(paddle.to_tensor(input_np), return_index=False, return_inverse=False, return_counts=False)

max_diff = np.max(np.abs(out_pt.numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.unique",
  "*args": [{"shape": [2, 3, 4], "dtype": "float32"}],
  "**kwargs": {
    "return_index": false,
    "return_inverse": false,
    "return_counts": false,
    "axis": null
  }
}
```

## Issue 128

llm_enhanced_torch_unique_20251216_003709.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][unique] Shape mismatch under equivalent migration in unique operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a **shape mismatch** was detected for `torch.unique`: PyTorch output shape (4,) vs TensorFlow output shape (2, 4).

The root cause is **different return structure**:

1. **PyTorch** `torch.unique(input)` by default returns only the unique values tensor, shape (N,).
2. **TensorFlow** `tf.unique(input)` always returns a tuple `(values, indices)`, which when naively compared as a single array gives shape (2, 4) — the values tensor and the indices tensor stacked together.

The comparison framework needs to extract only the values component from TensorFlow's output.

- Input: shape=[4], dtype=float64
- Error: Shape mismatch: (4,) vs (2, 4)

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(4).astype(np.float64)

# PyTorch: 默认只返回 unique values
out_pt = torch.unique(torch.tensor(input_np))

# TensorFlow: 始终返回 (values, indices)
out_tf = tf.unique(tf.constant(input_np))
values_tf = out_tf.y   # unique values
indices_tf = out_tf.idx  # indices

print(f"PT output shape: {out_pt.shape}")           # (4,)
print(f"TF values shape: {values_tf.shape}")         # (N,)
print(f"TF indices shape: {indices_tf.shape}")        # (4,)

# 正确比较方式：仅比较 values
pt_sorted = np.sort(out_pt.numpy())
tf_sorted = np.sort(values_tf.numpy())
if pt_sorted.shape == tf_sorted.shape:
    max_diff = np.max(np.abs(pt_sorted - tf_sorted))
    print(f"Maximum difference: {max_diff}")
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.unique] Shape mismatch under equivalent migration in unique operator (sample1)

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

When mapping `torch.unique(input)` to `tf.unique(input)`, the comparison reports a shape mismatch: PyTorch (4,) vs TensorFlow (2, 4). PyTorch's `torch.unique` by default returns only the unique values tensor, while TensorFlow's `tf.unique` always returns a namedtuple `(y, idx)` — the values and the inverse indices. The comparison framework incorrectly stacks both into one array.

Expected behavior: Compare only the values component (`tf.unique(input).y`) against PyTorch's output. Note that `torch.unique` returns sorted values by default, while `tf.unique` preserves input order — sorting may be needed for alignment.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(4).astype(np.float64)

out_pt = torch.unique(torch.tensor(input_np))
out_tf = tf.unique(tf.constant(input_np))

print(f"PT shape: {out_pt.shape}")
print(f"TF y shape: {out_tf.y.shape}, TF idx shape: {out_tf.idx.shape}")
# Shape mismatch when comparing directly
```

**Relevant log output**

```
comparison_error: Shape mismatch: PyTorch (4,) vs TensorFlow (2, 4)
```

## Issue 129

llm_enhanced_torch_unique_20251216_003709.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][unique] Shape mismatch under equivalent migration in unique operator (sample2)

**🐛 Describe the bug*** 

Same issue as Issue 128. `torch.unique` returns shape (4,) while `tf.unique` returns (2, 4) due to the (values, indices) tuple being compared as a single array.

- Input: shape=[4], dtype=float64
- Error: Shape mismatch: (4,) vs (2, 4)

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(4).astype(np.float64)

out_pt = torch.unique(torch.tensor(input_np))
out_tf_values = tf.unique(tf.constant(input_np)).y

pt_sorted = np.sort(out_pt.numpy())
tf_sorted = np.sort(out_tf_values.numpy())
max_diff = np.max(np.abs(pt_sorted - tf_sorted))
print(f"Maximum difference: {max_diff}")
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.unique] Shape mismatch under equivalent migration in unique operator (sample2)

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

Same as Issue 128. `torch.unique` returns only unique values (shape (4,)), while `tf.unique` returns `(y, idx)` tuple causing shape mismatch (2, 4) in direct comparison.

Expected behavior: Same as Issue 128.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(4).astype(np.float64)

out_pt = torch.unique(torch.tensor(input_np))
out_tf = tf.unique(tf.constant(input_np))

print(f"PT: {out_pt.shape}, TF stacked: {(2, 4)}")
```

**Relevant log output**

```
comparison_error: Shape mismatch: PyTorch (4,) vs TensorFlow (2, 4)
```

## Issue 130

llm_enhanced_torch_unique_20251216_003709.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][unique] Shape mismatch under equivalent migration in unique operator (sample3)

**🐛 Describe the bug*** 

Same issue as Issue 128. `torch.unique` returns shape (4,) while `tf.unique` returns (2, 4) due to the (values, indices) tuple being compared as a single array.

- Input: shape=[4], dtype=float64
- Error: Shape mismatch: (4,) vs (2, 4)

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(4).astype(np.float64)

out_pt = torch.unique(torch.tensor(input_np))
out_tf_values = tf.unique(tf.constant(input_np)).y

pt_sorted = np.sort(out_pt.numpy())
tf_sorted = np.sort(out_tf_values.numpy())
max_diff = np.max(np.abs(pt_sorted - tf_sorted))
print(f"Maximum difference: {max_diff}")
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.unique] Shape mismatch under equivalent migration in unique operator (sample3)

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

Same as Issue 128. `torch.unique` returns only unique values, while `tf.unique` returns `(y, idx)` tuple causing shape mismatch in direct comparison.

Expected behavior: Same as Issue 128.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(4).astype(np.float64)

out_pt = torch.unique(torch.tensor(input_np))
out_tf = tf.unique(tf.constant(input_np))
print(f"PT: {out_pt.shape}, TF (y, idx): ({out_tf.y.shape}, {out_tf.idx.shape})")
```

**Relevant log output**

```
comparison_error: Shape mismatch: PyTorch (4,) vs TensorFlow (2, 4)
```

## Issue 131

llm_enhanced_torch_unique_20251216_003709.json_sample8.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][unique] Shape mismatch under equivalent migration in unique operator (sample8)

**🐛 Describe the bug*** 

Same issue as Issue 128. `torch.unique` returns shape (4,) while `tf.unique` returns (2, 4) due to the (values, indices) tuple being compared as a single array.

- Input: shape=[4], dtype=float64
- Error: Shape mismatch: (4,) vs (2, 4)

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(4).astype(np.float64)

out_pt = torch.unique(torch.tensor(input_np))
out_tf_values = tf.unique(tf.constant(input_np)).y

pt_sorted = np.sort(out_pt.numpy())
tf_sorted = np.sort(out_tf_values.numpy())
max_diff = np.max(np.abs(pt_sorted - tf_sorted))
print(f"Maximum difference: {max_diff}")
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.unique] Shape mismatch under equivalent migration in unique operator (sample8)

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

Same as Issue 128. `torch.unique` returns only unique values, while `tf.unique` returns `(y, idx)` tuple causing shape mismatch in direct comparison.

Expected behavior: Same as Issue 128.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(4).astype(np.float64)

out_pt = torch.unique(torch.tensor(input_np))
out_tf = tf.unique(tf.constant(input_np))
print(f"PT: {out_pt.shape}, TF (y, idx): ({out_tf.y.shape}, {out_tf.idx.shape})")
```

**Relevant log output**

```
comparison_error: Shape mismatch: PyTorch (4,) vs TensorFlow (2, 4)
```

## Issue 132

llm_enhanced_torch_unique_consecutive_20251125_142753.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][unique_consecutive] Output difference anomaly under equivalent migration in unique_consecutive operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.unique_consecutive` (Maximum difference: 2.4568031275212663).

The root cause is **input data not synchronized**: The test harness did not fix random seeds, causing each framework to receive different random inputs. `torch.unique_consecutive` and `paddle.unique_consecutive` implement the same semantics (compressing consecutive equal elements), and should produce identical results with identical input.

- Input: shape=[4], dtype=float64
- Maximum difference: 2.4568031275212663

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(4).astype(np.float64)

out_pt = torch.unique_consecutive(torch.tensor(input_np))
out_pd = paddle.unique_consecutive(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 输入同步后应为 0.0
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][paddle.unique_consecutive] Output difference anomaly under equivalent migration in unique_consecutive operator (sample1)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`unique_consecutive` 算子的最大输出差异为 2.4568031275212663。

根本原因是**输入数据未同步**（随机种子未固定）。`torch.unique_consecutive` 和 `paddle.unique_consecutive` 语义完全一致（压缩连续相同元素），输入对齐后应输出相同结果。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(4).astype(np.float64)

out_pt = torch.unique_consecutive(torch.tensor(input_np))
out_pd = paddle.unique_consecutive(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.unique_consecutive",
  "*args": [{"shape": [4], "dtype": "float64"}]
}
```

- 输入未对齐导致差异，同步后无功能性差异。

## Issue 133

llm_enhanced_torch_unique_consecutive_20251125_142753.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][unique_consecutive] Output difference anomaly under equivalent migration in unique_consecutive operator (sample3)

**🐛 Describe the bug*** 

Same issue as Issue 132. Input data not synchronized (Maximum difference: 2.4779274439244947).

- Input: shape=[4], dtype=float64
- Maximum difference: 2.4779274439244947

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(4).astype(np.float64)

out_pt = torch.unique_consecutive(torch.tensor(input_np))
out_pd = paddle.unique_consecutive(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][paddle.unique_consecutive] Output difference anomaly under equivalent migration in unique_consecutive operator (sample3)

**bug描述 Describe the Bug*** 

与 Issue 132 相同：输入数据未同步，最大差异为 2.4779274439244947。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(4).astype(np.float64)

out_pt = torch.unique_consecutive(torch.tensor(input_np))
out_pd = paddle.unique_consecutive(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.unique_consecutive",
  "*args": [{"shape": [4], "dtype": "float64"}]
}
```

## Issue 134

llm_enhanced_torch_var_20251216_000004.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][var] Output difference anomaly under equivalent migration in var operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for `torch.var` (Maximum difference: 0.7173954599889627).

The root cause is **ddof (degrees of freedom) default value mismatch**:

1. **PyTorch** `torch.var` defaults to Bessel's correction (`correction=1`), computing the **sample variance** (divides by N-1).
2. **TensorFlow** `tf.math.reduce_variance` defaults to `ddof=0`, computing the **population variance** (divides by N).

For input shape [3, 1, 4, 1] with dim=2, N=4. The ratio is $\frac{N}{N-1} = \frac{4}{3} \approx 1.333$, creating a significant systematic difference.

- Input: shape=[3, 1, 4, 1], dtype=float64
- PT params: dim=2 (default correction=1)
- TF params: axis=2 (default ddof=0)
- Maximum difference: 0.7173954599889627

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(3, 1, 4, 1).astype(np.float64)

# PyTorch: 默认 correction=1 (无偏方差，除以 N-1)
out_pt = torch.var(torch.tensor(input_np), dim=2)

# TensorFlow: 默认除以 N (有偏方差)
out_tf = tf.math.reduce_variance(tf.constant(input_np), axis=2)

pt_np = out_pt.detach().numpy()
tf_np = out_tf.numpy()
max_diff = np.max(np.abs(pt_np - tf_np))
print(f"Maximum difference: {max_diff}")
# N=4: PyTorch 除以 3, TF 除以 4
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.math.reduce_variance] Output difference anomaly under equivalent migration in var operator (sample2)

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

When mapping `torch.var(input, dim=2)` to `tf.math.reduce_variance(input, axis=2)`, the outputs differ by up to 0.7173954599889627. PyTorch's `torch.var` defaults to unbiased variance (divides by N-1=3), while TensorFlow's `tf.math.reduce_variance` defaults to biased variance (divides by N=4).

Expected behavior: Migration should manually correct for the ddof difference, e.g., `tf_var * N / (N-1)`.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(3, 1, 4, 1).astype(np.float64)

out_pt = torch.var(torch.tensor(input_np), dim=2)
out_tf = tf.math.reduce_variance(tf.constant(input_np), axis=2)

max_diff = np.max(np.abs(out_pt.detach().numpy() - out_tf.numpy()))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.7173954599889627
```

## Issue 135

llm_enhanced_torch_var_20251216_000004.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][var] Output difference anomaly under equivalent migration in var operator (sample4)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for `torch.var` (Maximum difference: nan).

The root cause is **ddof default mismatch** combined with **dimension size 1**:

1. **PyTorch** `torch.var(input, dim=2)` with default `correction=1` divides by N-1. When dim=2 has size 1 (N=1), it divides by 0, producing **NaN**.
2. **TensorFlow** `tf.math.reduce_variance(input, axis=2)` with default ddof=0 divides by N=1, producing **0.0**.

This NaN vs 0.0 difference results in max_diff=nan.

- Input: shape=[2, 5, 1], dtype=float32
- PT params: dim=2 (N=1, correction=1 → divide by 0 → NaN)
- TF params: axis=2 (N=1, ddof=0 → divide by 1 → 0.0)
- Maximum difference: nan

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 5, 1).astype(np.float32)

# PyTorch: dim=2 size=1, correction=1 → N-1=0 → NaN
out_pt = torch.var(torch.tensor(input_np), dim=2)

# TensorFlow: dim=2 size=1, ddof=0 → N=1 → 0.0
out_tf = tf.math.reduce_variance(tf.constant(input_np), axis=2)

print(f"PyTorch output: {out_pt}")   # 全 NaN
print(f"TensorFlow output: {out_tf}")  # 全 0.0
max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")  # nan
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.math.reduce_variance] Output difference anomaly under equivalent migration in var operator (sample4)

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

When mapping `torch.var(input, dim=2)` to `tf.math.reduce_variance(input, axis=2)` with dim size 1, PyTorch outputs NaN (divides by N-1=0) while TensorFlow outputs 0.0 (divides by N=1). Maximum difference: nan.

Expected behavior: Migration must account for the ddof difference. When N=1 and correction=1, the variance is undefined (NaN in PyTorch), while TF returns 0.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 5, 1).astype(np.float32)

out_pt = torch.var(torch.tensor(input_np), dim=2)
out_tf = tf.math.reduce_variance(tf.constant(input_np), axis=2)

print(f"PT: {out_pt}")   # NaN
print(f"TF: {out_tf}")   # 0.0
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: nan
```

## Issue 136

llm_enhanced_torch_var_mean_20251215_173951.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][var_mean] Output difference anomaly under equivalent migration in var_mean operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for `torch.var_mean` (Maximum difference: 3.0460211903100234).

The root cause is **variance computation method mismatch**:

1. **PyTorch** `torch.var_mean(input, dim=[2, 4], unbiased=True)` computes the **unbiased (sample) variance** (divides by N-1) and mean.
2. **TensorFlow** `tf.nn.moments(input, axes=[2, 4])` computes the **biased (population) variance** (second central moment, divides by N) and mean.

For dim=[2, 4] on shape [3, 4, 5, 6, 2, 3], the reduced size is 5×2=10, so the ratio is $\frac{N}{N-1} = \frac{10}{9} \approx 1.111$, producing significant differences.

- Input: shape=[3, 4, 5, 6, 2, 3], dtype=float64
- PT params: dim=[2, 4], unbiased=True, keepdim=False
- TF params: axes=[2, 4], keepdims=False
- Maximum difference: 3.0460211903100234

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(3, 4, 5, 6, 2, 3).astype(np.float64)

# PyTorch: unbiased=True (除以 N-1)
var_pt, mean_pt = torch.var_mean(torch.tensor(input_np), dim=[2, 4], unbiased=True, keepdim=False)

# TensorFlow: tf.nn.moments 默认有偏方差 (除以 N)
mean_tf, var_tf = tf.nn.moments(tf.constant(input_np), axes=[2, 4], keepdims=False)

max_diff_var = np.max(np.abs(var_pt.detach().numpy() - var_tf.numpy()))
max_diff_mean = np.max(np.abs(mean_pt.detach().numpy() - mean_tf.numpy()))
print(f"Variance max diff: {max_diff_var}")
print(f"Mean max diff: {max_diff_mean}")
# 注意：tf.nn.moments 返回 (mean, variance)，顺序与 PyTorch 不同
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.nn.moments] Output difference anomaly under equivalent migration in var_mean operator (sample1)

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

When mapping `torch.var_mean(input, dim=[2,4], unbiased=True)` to `tf.nn.moments(input, axes=[2,4])`, the variance outputs differ by up to 3.0460211903100234. PyTorch with `unbiased=True` computes sample variance (divides by N-1), while `tf.nn.moments` computes the second central moment (biased variance, divides by N). Additionally, return order differs: PyTorch returns (variance, mean), TF returns (mean, variance).

Expected behavior: Migration should correct the variance: `tf_var * N / (N-1)` and account for return order.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(3, 4, 5, 6, 2, 3).astype(np.float64)

var_pt, mean_pt = torch.var_mean(torch.tensor(input_np), dim=[2, 4], unbiased=True, keepdim=False)
mean_tf, var_tf = tf.nn.moments(tf.constant(input_np), axes=[2, 4], keepdims=False)

max_diff = np.max(np.abs(var_pt.detach().numpy() - var_tf.numpy()))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 3.0460211903100234
```

## Issue 137

llm_enhanced_torch_var_mean_20251215_173951.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][var_mean] Output difference anomaly under equivalent migration in var_mean operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for `torch.var_mean` (Maximum difference: 2.605342993166393).

Same root cause as Issue 136: **unbiased vs biased variance**. PyTorch's `torch.var_mean(dim=0, unbiased=True, keepdim=True)` divides by N-1, while `tf.nn.moments(axes=0, keepdims=True)` divides by N.

For input shape [5, 5, 5] with dim=0, N=5, the ratio is $\frac{5}{4} = 1.25$.

- Input: shape=[5, 5, 5], dtype=float64
- PT params: dim=0, unbiased=True, keepdim=True
- TF params: axes=0, keepdims=True
- Maximum difference: 2.605342993166393

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(5, 5, 5).astype(np.float64)

var_pt, mean_pt = torch.var_mean(torch.tensor(input_np), dim=0, unbiased=True, keepdim=True)
mean_tf, var_tf = tf.nn.moments(tf.constant(input_np), axes=[0], keepdims=True)

max_diff_var = np.max(np.abs(var_pt.detach().numpy() - var_tf.numpy()))
max_diff_mean = np.max(np.abs(mean_pt.detach().numpy() - mean_tf.numpy()))
print(f"Variance max diff: {max_diff_var}")
print(f"Mean max diff: {max_diff_mean}")
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.nn.moments] Output difference anomaly under equivalent migration in var_mean operator (sample2)

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

Same as Issue 136. PyTorch's `torch.var_mean(unbiased=True)` computes sample variance (divides by N-1=4), while `tf.nn.moments` computes biased variance (divides by N=5). For shape [5, 5, 5] with dim=0, maximum difference reaches 2.605342993166393.

Expected behavior: Same as Issue 136.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(5, 5, 5).astype(np.float64)

var_pt, mean_pt = torch.var_mean(torch.tensor(input_np), dim=0, unbiased=True, keepdim=True)
mean_tf, var_tf = tf.nn.moments(tf.constant(input_np), axes=[0], keepdims=True)

max_diff = np.max(np.abs(var_pt.detach().numpy() - var_tf.numpy()))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 2.605342993166393
```

## Issue 138

llm_enhanced_torch_var_mean_20251215_173951.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][var_mean] Output difference anomaly under equivalent migration in var_mean operator (sample4)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for `torch.var_mean` (Maximum difference: 1.9016787153849102).

Same root cause as Issue 136: **unbiased vs biased variance**. PyTorch divides by N-1, TensorFlow divides by N.

For input shape [3, 4] with dim=0, N=3, the ratio is $\frac{3}{2} = 1.5$.

- Input: shape=[3, 4], dtype=float64
- PT params: dim=0, unbiased=True, keepdim=True
- TF params: axes=[0], keepdims=True
- Maximum difference: 1.9016787153849102

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(3, 4).astype(np.float64)

var_pt, mean_pt = torch.var_mean(torch.tensor(input_np), dim=0, unbiased=True, keepdim=True)
mean_tf, var_tf = tf.nn.moments(tf.constant(input_np), axes=[0], keepdims=True)

max_diff_var = np.max(np.abs(var_pt.detach().numpy() - var_tf.numpy()))
max_diff_mean = np.max(np.abs(mean_pt.detach().numpy() - mean_tf.numpy()))
print(f"Variance max diff: {max_diff_var}")
print(f"Mean max diff: {max_diff_mean}")
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.nn.moments] Output difference anomaly under equivalent migration in var_mean operator (sample4)

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

Same as Issue 136. PyTorch's `torch.var_mean(unbiased=True)` computes sample variance (divides by N-1=2), while `tf.nn.moments` computes biased variance (divides by N=3). For shape [3, 4] with dim=0, the ratio is 3/2=1.5, producing maximum difference 1.9016787153849102.

Expected behavior: Same as Issue 136.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(3, 4).astype(np.float64)

var_pt, mean_pt = torch.var_mean(torch.tensor(input_np), dim=0, unbiased=True, keepdim=True)
mean_tf, var_tf = tf.nn.moments(tf.constant(input_np), axes=[0], keepdims=True)

max_diff = np.max(np.abs(var_pt.detach().numpy() - var_tf.numpy()))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 1.9016787153849102
```
"""

with open(output_path, "a", encoding="utf-8") as f:
    f.write(content)

print("Done! Appended Issues 111-138 to the file.")
