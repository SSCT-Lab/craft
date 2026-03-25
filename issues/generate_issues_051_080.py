#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 Issue 051-080 的标准格式 issue 内容并追加到修改版文件"""

import os

output_file = r"d:\graduate\DFrameworkTest\issues\138个跨表不一致Case的GitHub Issue-修改版.md"

content = r"""
## Issue 051

llm_enhanced_torch_nn_MultiLabelSoftMarginLoss_20251202_000132.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.MultiLabelSoftMarginLoss] Output crash anomaly under equivalent migration in MultiLabelSoftMarginLoss operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, the `torch.nn.MultiLabelSoftMarginLoss` operator encountered a runtime crash: `ufunc 'isfinite' not supported for the input types`. The comparison framework failed because the Loss function requires both `input` and `target` tensors, but the test JSON only provides `input`, causing the frameworks to raise errors or return non-Tensor objects.

**Two distinct problems exist:**

1. **Missing required parameter**: `MultiLabelSoftMarginLoss` requires both `input` and `target` arguments. The test configuration only provides `input`, causing the forward pass to fail.
2. **No TF equivalent**: TensorFlow has no direct native equivalent of `torch.nn.MultiLabelSoftMarginLoss`. The PyTorch implementation computes multi-label soft margin loss as: `loss = -sum(y_i * log(sigmoid(x_i)) + (1-y_i) * log(1-sigmoid(x_i)))`.

- Input: shape=[2, 3], dtype=float32
- Error: `ufunc 'isfinite' not supported for the input types`

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3).astype(np.float32)
# MultiLabelSoftMarginLoss 需要 target 参数，此处手动补充
target_np = np.random.randint(0, 2, size=(2, 3)).astype(np.float32)

# PyTorch 执行
input_pt = torch.tensor(input_np)
target_pt = torch.tensor(target_np)
loss_pt = torch.nn.MultiLabelSoftMarginLoss()
out_pt = loss_pt(input_pt, target_pt)

# PaddlePaddle 执行
input_pd = paddle.to_tensor(input_np)
target_pd = paddle.to_tensor(target_np)
loss_pd = paddle.nn.MultiLabelSoftMarginLoss()
out_pd = loss_pd(input_pd, target_pd)

# 比较结果
pt_val = out_pt.detach().numpy()
pd_val = out_pd.numpy()
max_diff = np.abs(pt_val - pd_val)
print(f"PyTorch loss: {pt_val}")
print(f"PaddlePaddle loss: {pd_val}")
print(f"Maximum difference: {max_diff}")
# 注意：原始测试配置缺少 target 参数，导致比较框架崩溃
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][MultiLabelSoftMarginLoss] Output crash anomaly under equivalent migration in MultiLabelSoftMarginLoss operator (sample2)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`MultiLabelSoftMarginLoss` 算子出现运行时崩溃：`ufunc 'isfinite' not supported for the input types`。

根本原因是**测试配置缺少必要参数**：`MultiLabelSoftMarginLoss` 是损失函数，需要同时提供 `input` 和 `target` 两个张量。原始 JSON 中仅提供了 `input`，导致框架在调用 `forward` 时抛出异常或返回非 Tensor 对象，比较器在对其进行 `isfinite` 检查时崩溃。

此外，TensorFlow 中没有直接等价于 `torch.nn.MultiLabelSoftMarginLoss` 的原生损失函数。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3).astype(np.float32)
target_np = np.random.randint(0, 2, size=(2, 3)).astype(np.float32)

input_pt = torch.tensor(input_np)
target_pt = torch.tensor(target_np)
loss_pt = torch.nn.MultiLabelSoftMarginLoss()
out_pt = loss_pt(input_pt, target_pt)

input_pd = paddle.to_tensor(input_np)
target_pd = paddle.to_tensor(target_np)
loss_pd = paddle.nn.MultiLabelSoftMarginLoss()
out_pd = loss_pd(input_pd, target_pd)

print(f"PyTorch loss: {out_pt.detach().numpy()}")
print(f"PaddlePaddle loss: {out_pd.numpy()}")
# 原始测试缺少 target 参数导致崩溃
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.MultiLabelSoftMarginLoss",
  "input": {
    "shape": [2, 3],
    "dtype": "float32"
  }
}
```

- 缺少 `target` 参数是崩溃的直接原因。此文件来自 `comparison_d` 目录，表明该 case 在比较阶段即出错。

## Issue 052

llm_enhanced_torch_nn_PReLU_20251215_233126.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.PReLU] Output difference anomaly under equivalent migration in PReLU operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a severe output discrepancy was detected for the `torch.nn.PReLU` operator (Maximum difference: 0.44696375727653503).

The root cause is a **learnable parameter initialization and mode mismatch**:

1. **PyTorch** `torch.nn.PReLU(num_parameters=1)` uses a single shared learnable alpha (default=0.25) applied uniformly across all channels.
2. **TensorFlow** `tf.keras.layers.PReLU` uses per-channel (or per-element) learnable alpha initialized to zeros by default.

These different initialization strategies and parameter sharing modes produce different negative-slope behaviors.

- Input: shape=[1, 3, 4, 4], dtype=float32
- Maximum difference: 0.44696375727653503

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 3, 4, 4).astype(np.float32)

# PyTorch: 单个共享 alpha=0.25
input_pt = torch.tensor(input_np)
prelu_pt = torch.nn.PReLU(num_parameters=1)
out_pt = prelu_pt(input_pt)

# TensorFlow: 需要 NHWC 格式，且 alpha 初始化为 0
input_tf_nhwc = np.transpose(input_np, (0, 2, 3, 1))
input_tf = tf.constant(input_tf_nhwc)
prelu_tf = tf.keras.layers.PReLU(alpha_initializer='zeros')
out_tf_nhwc = prelu_tf(input_tf)
out_tf = tf.transpose(out_tf_nhwc, perm=[0, 3, 1, 2])

pt_np = out_pt.detach().numpy().astype(np.float64)
tf_np = out_tf.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - tf_np))
print(f"PyTorch output shape: {out_pt.shape}")
print(f"TensorFlow output shape: {out_tf.shape}")
print(f"Maximum difference: {max_diff}")
# PT alpha=0.25 (shared), TF alpha=0.0 (per-element) => 对负值区域输出不同
# 实测最大差异: 0.44696375727653503
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.PReLU] Output difference anomaly under equivalent migration in PReLU operator (sample2)

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

When mapping `torch.nn.PReLU(num_parameters=1)` to `tf.keras.layers.PReLU`, the alpha initialization and sharing mode differ: PyTorch uses a single shared learnable alpha (default=0.25), while TensorFlow uses per-element alpha initialized to zeros. This causes different behaviors for negative input values (Maximum difference: 0.44696375727653503).

Expected behavior: Migration should ensure alpha initialization and parameter sharing mode are aligned, e.g., by setting `alpha_initializer=tf.constant_initializer(0.25)` and `shared_axes=[1,2]` in TensorFlow to match PyTorch's shared alpha behavior.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 3, 4, 4).astype(np.float32)

prelu_pt = torch.nn.PReLU(num_parameters=1)
out_pt = prelu_pt(torch.tensor(input_np))

input_tf = tf.constant(np.transpose(input_np, (0, 2, 3, 1)))
prelu_tf = tf.keras.layers.PReLU(alpha_initializer='zeros')
out_tf = tf.transpose(prelu_tf(input_tf), perm=[0, 3, 1, 2])

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 0.44696375727653503
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.44696375727653503
```

## Issue 053

llm_enhanced_torch_nn_PReLU_20251215_233126.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.PReLU] Output difference anomaly under equivalent migration in PReLU operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a severe output discrepancy was detected for the `torch.nn.PReLU` operator (Maximum difference: 0.5739413499832153).

Same root cause as Issue 052: PyTorch's `torch.nn.PReLU` uses a single shared learnable alpha (default=0.25) while TensorFlow's `tf.keras.layers.PReLU` initializes alpha to zeros per-element. Additionally, when `num_parameters` is set in PyTorch to match channel count, the weight initialization still differs from TensorFlow's default zeros.

- Input: shape=[2, 5, 1, 3], dtype=float32
- Maximum difference: 0.5739413499832153

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 5, 1, 3).astype(np.float32)

# PyTorch: alpha 初始化为 0.25
input_pt = torch.tensor(input_np)
prelu_pt = torch.nn.PReLU(num_parameters=1)
out_pt = prelu_pt(input_pt)

# TensorFlow: alpha 初始化为 0
input_tf = tf.constant(np.transpose(input_np, (0, 2, 3, 1)))
prelu_tf = tf.keras.layers.PReLU(alpha_initializer='zeros')
out_tf_nhwc = prelu_tf(input_tf)
out_tf = tf.transpose(out_tf_nhwc, perm=[0, 3, 1, 2])

pt_np = out_pt.detach().numpy().astype(np.float64)
tf_np = out_tf.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - tf_np))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 0.5739413499832153
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.keras.layers.PReLU] Output difference anomaly under equivalent migration in PReLU operator (sample3)

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

Same as Issue 052. When mapping `torch.nn.PReLU` to `tf.keras.layers.PReLU`, the alpha initialization differs: PyTorch default alpha=0.25 (shared), TensorFlow default alpha=0.0 (per-element). This produces different outputs for negative input values (Maximum difference: 0.5739413499832153).

Expected behavior: Migration should align alpha initialization and parameter sharing mode between the two frameworks.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 5, 1, 3).astype(np.float32)

prelu_pt = torch.nn.PReLU(num_parameters=1)
out_pt = prelu_pt(torch.tensor(input_np))

input_tf = tf.constant(np.transpose(input_np, (0, 2, 3, 1)))
prelu_tf = tf.keras.layers.PReLU(alpha_initializer='zeros')
out_tf = tf.transpose(prelu_tf(input_tf), perm=[0, 3, 1, 2])

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 0.5739413499832153
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.5739413499832153
```

## Issue 054

llm_enhanced_torch_nn_RReLU_20251202_010732.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.RReLU] Output difference anomaly under equivalent migration in RReLU operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, an output discrepancy was detected for the `torch.nn.RReLU` operator (Maximum difference: 0.25186100602149963).

The root cause is **stochastic operator behavior**: RReLU (Randomized Leaky ReLU) randomly samples a negative slope from `Uniform(lower, upper)` during training. Different frameworks use different RNG implementations, so even with identical inputs and parameters (`lower=0.1, upper=0.3`), the sampled slopes differ, producing different outputs for negative input values.

- Input: shape=[2, 3], dtype=float32
- Parameters: lower=0.1, upper=0.3
- Maximum difference: 0.25186100602149963

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3).astype(np.float32)

# PyTorch 执行
torch.manual_seed(42)
input_pt = torch.tensor(input_np)
rrelu_pt = torch.nn.RReLU(lower=0.1, upper=0.3)
rrelu_pt.train()
out_pt = rrelu_pt(input_pt)

# PaddlePaddle 执行
paddle.seed(42)
input_pd = paddle.to_tensor(input_np)
rrelu_pd = paddle.nn.RReLU(lower=0.1, upper=0.3)
rrelu_pd.train()
out_pd = rrelu_pd(input_pd)

pt_np = out_pt.detach().numpy().astype(np.float64)
pd_np = out_pd.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"PyTorch output: {out_pt.detach().numpy()}")
print(f"PaddlePaddle output: {out_pd.numpy()}")
print(f"Maximum difference: {max_diff}")
# RReLU 是随机算子，不同框架 RNG 实现不同，输出差异是预期的
# 实测最大差异: 0.25186100602149963
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.RReLU] Output difference anomaly under equivalent migration in RReLU operator (sample1)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`nn.RReLU` 算子在训练模式下的输出差异为 0.25186100602149963。

根本原因是**随机算子行为**：RReLU（Randomized Leaky ReLU）在训练模式下会从 `Uniform(lower, upper)` 中随机采样负斜率。不同框架使用不同的随机数生成器（RNG）实现，即使输入和参数（`lower=0.1, upper=0.3`）完全相同，采样到的斜率也不同，导致对负值输入的输出不一致。

迁移比较框架应将 RReLU 在训练模式下标记为"不可比随机算子"，或在 eval 模式下比较（eval 模式使用固定斜率 `(lower+upper)/2`）。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3).astype(np.float32)

torch.manual_seed(42)
rrelu_pt = torch.nn.RReLU(lower=0.1, upper=0.3)
rrelu_pt.train()
out_pt = rrelu_pt(torch.tensor(input_np))

paddle.seed(42)
rrelu_pd = paddle.nn.RReLU(lower=0.1, upper=0.3)
rrelu_pd.train()
out_pd = rrelu_pd(paddle.to_tensor(input_np))

print(f"Maximum difference: {np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))}")
# 实测最大差异: 0.25186100602149963
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.RReLU",
  "input": {
    "shape": [2, 3],
    "dtype": "float32"
  },
  "lower": 0.1,
  "upper": 0.3
}
```

- RReLU 训练模式输出具有随机性，跨框架 RNG 实现不同导致差异。eval 模式下使用固定中值斜率应可对齐。

## Issue 055

llm_enhanced_torch_nn_RReLU_20251216_010240.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][nn.RReLU] Output difference anomaly under equivalent migration in RReLU operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, an output discrepancy was detected for the `torch.nn.RReLU` operator (Maximum difference: 0.05685841292142868).

Same root cause as Issue 054: RReLU randomly samples negative slopes from `Uniform(lower, upper)` during training. Different frameworks have different RNG implementations, producing different sampled slopes. Additionally, TensorFlow has no native equivalent of RReLU.

- Input: shape=[2, 3], dtype=float32
- Parameters: lower=0.1, upper=0.3
- Maximum difference: 0.05685841292142868

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(2, 3).astype(np.float32)

# PyTorch 执行
torch.manual_seed(42)
input_pt = torch.tensor(input_np)
rrelu_pt = torch.nn.RReLU(lower=0.1, upper=0.3)
rrelu_pt.train()
out_pt = rrelu_pt(input_pt)

# MindSpore 执行
mindspore.set_seed(42)
input_ms = mindspore.Tensor(input_np)
rrelu_ms = mindspore.nn.RReLU(lower=0.1, upper=0.3)
rrelu_ms.set_train(True)
out_ms = rrelu_ms(input_ms)

pt_np = out_pt.detach().numpy().astype(np.float64)
ms_np = out_ms.asnumpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - ms_np))
print(f"Maximum difference: {max_diff}")
# RReLU 随机算子，跨框架 RNG 不同导致差异
# 实测最大差异: 0.05685841292142868
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.nn.RReLU] Output difference anomaly under equivalent migration in RReLU operator (sample1)

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

During cross-framework equivalent migration testing, `torch.nn.RReLU` and `mindspore.nn.RReLU` produce different outputs (Maximum difference: 0.05685841292142868) when applied to the same input (shape=[2,3], dtype=float32, lower=0.1, upper=0.3). RReLU randomly samples negative slopes from Uniform(lower, upper) during training, and each framework's RNG implementation differs.

**Describe the expected behavior***

The migration comparison framework should classify RReLU in training mode as a "stochastic operator" exempt from strict numerical comparison. In eval mode (using fixed slope `(lower+upper)/2 = 0.2`), both frameworks should produce identical outputs.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(2, 3).astype(np.float32)

torch.manual_seed(42)
rrelu_pt = torch.nn.RReLU(lower=0.1, upper=0.3)
rrelu_pt.train()
out_pt = rrelu_pt(torch.tensor(input_np))

mindspore.set_seed(42)
rrelu_ms = mindspore.nn.RReLU(lower=0.1, upper=0.3)
rrelu_ms.set_train(True)
out_ms = rrelu_ms(mindspore.Tensor(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_ms.asnumpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 0.05685841292142868
```

**Related log / screenshot**

```
comparison_error: Numerical mismatch, maximum difference: 0.05685841292142868
```

**Special notes for this issue**

RReLU is a stochastic operator. In training mode, different frameworks sample different random slopes. This is expected behavior, not a framework bug.

## Issue 056

llm_enhanced_torch_nn_ReplicationPad2d_20251202_004340.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.ReplicationPad2d] Output difference anomaly under equivalent migration in ReplicationPad2d operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for the `torch.nn.ReplicationPad2d` operator (Maximum difference: 2.1693625450134277).

The root cause is **input data not synchronized**: Both `torch.nn.ReplicationPad2d` and `paddle.nn.Pad2D(mode='replicate')` are deterministic copy-padding operators. If inputs are identical, outputs must be identical. The large discrepancy (2.17) indicates the two frameworks received different random input data (random seed not fixed across frameworks in the test harness).

- Input: shape=[2, 3, 4, 4], dtype=float32
- Parameters: padding=2
- Maximum difference: 2.1693625450134277

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)

# PyTorch 执行
input_pt = torch.tensor(input_np)
pad_pt = torch.nn.ReplicationPad2d(padding=2)
out_pt = pad_pt(input_pt)

# PaddlePaddle 执行
input_pd = paddle.to_tensor(input_np)
pad_pd = paddle.nn.Pad2D(padding=2, mode='replicate')
out_pd = pad_pd(input_pd)

pt_np = out_pt.detach().numpy().astype(np.float64)
pd_np = out_pd.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"PyTorch output shape: {out_pt.shape}")
print(f"PaddlePaddle output shape: {out_pd.shape}")
print(f"Maximum difference: {max_diff}")
# 若输入相同，确定性复制填充应输出相同。差异说明原始测试中输入未同步。
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.ReplicationPad2d] Output difference anomaly under equivalent migration in ReplicationPad2d operator (sample1)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`ReplicationPad2d` 算子的最大输出差异为 2.1693625450134277。

根本原因是**输入数据未同步**：`torch.nn.ReplicationPad2d` 和 `paddle.nn.Pad2D(mode='replicate')` 均为确定性复制填充算子，若输入一致，输出必一致。差异值 2.17 远超精度误差，表明原始测试中两端使用了不同的随机输入（未固定随机种子）。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)

pad_pt = torch.nn.ReplicationPad2d(padding=2)
out_pt = pad_pt(torch.tensor(input_np))

pad_pd = paddle.nn.Pad2D(padding=2, mode='replicate')
out_pd = pad_pd(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 输入同步后差异应为 0
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.Pad2D",
  "input": {
    "shape": [2, 3, 4, 4],
    "dtype": "float32"
  },
  "padding": 2,
  "mode": "replicate"
}
```

- 确定性算子，输入同步后差异应为 0。原始测试框架需确保两端使用相同的随机种子或预生成的输入数据。

## Issue 057

llm_enhanced_torch_nn_ReplicationPad2d_20251202_004340.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.ReplicationPad2d] Output difference anomaly under equivalent migration in ReplicationPad2d operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for the `torch.nn.ReplicationPad2d` operator (Maximum difference: 2.2087299823760986).

Same root cause as Issue 056: input data not synchronized across frameworks. ReplicationPad2d is a deterministic operator — if inputs match, outputs must match. Additionally, the padding parameter format may differ: PyTorch uses `[left, right, top, bottom]` order, while Paddle may expect a different order.

- Input: shape=[2, 3, 4, 4], dtype=float32
- Parameters: padding=[1, 1, 2, 0]
- Maximum difference: 2.2087299823760986

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)

# PyTorch: padding = [left, right, top, bottom]
input_pt = torch.tensor(input_np)
pad_pt = torch.nn.ReplicationPad2d(padding=[1, 1, 2, 0])
out_pt = pad_pt(input_pt)

# PaddlePaddle: padding = [left, right, top, bottom] (same order)
input_pd = paddle.to_tensor(input_np)
pad_pd = paddle.nn.Pad2D(padding=[1, 1, 2, 0], mode='replicate')
out_pd = pad_pd(input_pd)

pt_np = out_pt.detach().numpy().astype(np.float64)
pd_np = out_pd.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"PyTorch output shape: {out_pt.shape}")
print(f"PaddlePaddle output shape: {out_pd.shape}")
print(f"Maximum difference: {max_diff}")
# 输入同步后，确定性复制填充应输出相同
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.ReplicationPad2d] Output difference anomaly under equivalent migration in ReplicationPad2d operator (sample2)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`ReplicationPad2d` 算子（非对称 padding=[1,1,2,0]）的最大输出差异为 2.2087299823760986。

与 Issue 056 相同的根本原因：输入数据未同步。此外需注意 padding 参数顺序：PyTorch 的 `ReplicationPad2d` 接受 `[left, right, top, bottom]`，Paddle 的 `Pad2D` 也接受 `[left, right, top, bottom]`，顺序一致但需确认迁移代码是否正确映射。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4).astype(np.float32)

pad_pt = torch.nn.ReplicationPad2d(padding=[1, 1, 2, 0])
out_pt = pad_pt(torch.tensor(input_np))

pad_pd = paddle.nn.Pad2D(padding=[1, 1, 2, 0], mode='replicate')
out_pd = pad_pd(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.Pad2D",
  "input": {
    "shape": [2, 3, 4, 4],
    "dtype": "float32"
  },
  "padding": [1, 1, 2, 0],
  "mode": "replicate"
}
```

- 确定性算子，输入同步后差异应为 0。

## Issue 058

llm_enhanced_torch_nn_ReplicationPad3d_20251202_132259.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.ReplicationPad3d] Output difference anomaly under equivalent migration in ReplicationPad3d operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for the `torch.nn.ReplicationPad3d` operator (Maximum difference: 3.3501651287078857).

The root cause is **input data not synchronized**: Both operators are deterministic replicate-padding operators. If inputs are identical, outputs must be identical. The large difference (3.35) confirms the test harness fed different random inputs to each framework.

Additionally, `torch.nn.ReplicationPad3d` uses 'replicate' mode by default, while `paddle.nn.Pad3D` defaults to 'constant' mode (filled with zeros). The migration code must explicitly set `mode='replicate'` in Paddle.

- Input: shape=[2, 3, 4, 4, 4], dtype=float32
- Parameters: padding=3
- Maximum difference: 3.3501651287078857

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4, 4).astype(np.float32)

# PyTorch: replicate padding
input_pt = torch.tensor(input_np)
pad_pt = torch.nn.ReplicationPad3d(padding=3)
out_pt = pad_pt(input_pt)

# PaddlePaddle: 必须显式指定 mode='replicate'
input_pd = paddle.to_tensor(input_np)
pad_pd = paddle.nn.Pad3D(padding=3, mode='replicate')
out_pd = pad_pd(input_pd)

pt_np = out_pt.detach().numpy().astype(np.float64)
pd_np = out_pd.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"PyTorch output shape: {out_pt.shape}")
print(f"PaddlePaddle output shape: {out_pd.shape}")
print(f"Maximum difference: {max_diff}")
# 输入同步后应为 0。注意 Paddle Pad3D 默认 mode='constant'，需显式设 'replicate'
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.ReplicationPad3d] Output difference anomaly under equivalent migration in ReplicationPad3d operator (sample1)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`ReplicationPad3d` 算子的最大输出差异为 3.3501651287078857。

**两个问题**：
1. **输入数据未同步**：确定性算子在输入相同时输出必相同，差异 3.35 说明测试中两端输入不同。
2. **默认 mode 不一致**：`torch.nn.ReplicationPad3d` 自动使用 replicate 模式，而 `paddle.nn.Pad3D` 默认使用 constant 模式（填充 0）。迁移代码需显式设置 `mode='replicate'`。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 4, 4).astype(np.float32)

pad_pt = torch.nn.ReplicationPad3d(padding=3)
out_pt = pad_pt(torch.tensor(input_np))

pad_pd = paddle.nn.Pad3D(padding=3, mode='replicate')
out_pd = pad_pd(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.Pad3D",
  "input": {
    "shape": [2, 3, 4, 4, 4],
    "dtype": "float32"
  },
  "padding": 3,
  "mode": "replicate"
}
```

- `paddle.nn.Pad3D` 默认 `mode='constant'`（填充 0），需显式设为 `mode='replicate'` 以匹配 PyTorch 的 ReplicationPad3d。

## Issue 059

llm_enhanced_torch_nn_ReplicationPad3d_20251202_132259.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.ReplicationPad3d] Output difference anomaly under equivalent migration in ReplicationPad3d operator (sample4)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, an output discrepancy was detected for the `torch.nn.ReplicationPad3d` operator (Maximum difference: 0.1199876537563763).

**Two issues contribute:**

1. **Padding mode mismatch**: PyTorch's `ReplicationPad3d` uses replicate padding (edge value copying), while Paddle's `Pad3D` defaults to constant padding (zero-fill). Migration must explicitly specify `mode='replicate'`.
2. **Padding order**: PyTorch uses `[left, right, top, bottom, front, back]` order, while Paddle also accepts this format but implementation details may differ.

- Input: shape=[1, 1, 1, 1, 1], dtype=float64
- Parameters: padding=[0, 1, 0, 1, 0, 1]
- Maximum difference: 0.1199876537563763

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(1, 1, 1, 1, 1).astype(np.float64)

# PyTorch: replicate padding
input_pt = torch.tensor(input_np)
pad_pt = torch.nn.ReplicationPad3d(padding=[0, 1, 0, 1, 0, 1])
out_pt = pad_pt(input_pt)

# PaddlePaddle: 显式指定 mode='replicate'
input_pd = paddle.to_tensor(input_np)
pad_pd = paddle.nn.Pad3D(padding=[0, 1, 0, 1, 0, 1], mode='replicate')
out_pd = pad_pd(input_pd)

pt_np = out_pt.detach().numpy().astype(np.float64)
pd_np = out_pd.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"PyTorch output shape: {out_pt.shape}, output: {out_pt.detach().numpy()}")
print(f"PaddlePaddle output shape: {out_pd.shape}, output: {out_pd.numpy()}")
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 0.1199876537563763
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.ReplicationPad3d] Output difference anomaly under equivalent migration in ReplicationPad3d operator (sample4)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`ReplicationPad3d` 算子的最大输出差异为 0.1199876537563763。

根本原因：
1. **填充模式不匹配**：PyTorch `ReplicationPad3d` 严格使用复制填充规则，Paddle `Pad3D` 默认使用常数填充（constant，填充值默认为 0）。迁移代码需显式指定 `mode='replicate'`。
2. **Padding 顺序**：PyTorch 的 padding 参数顺序为 `(left, right, top, bottom, front, back)`，对应空间维度的前后左右上下。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(1, 1, 1, 1, 1).astype(np.float64)

pad_pt = torch.nn.ReplicationPad3d(padding=[0, 1, 0, 1, 0, 1])
out_pt = pad_pt(torch.tensor(input_np))

pad_pd = paddle.nn.Pad3D(padding=[0, 1, 0, 1, 0, 1], mode='replicate')
out_pd = pad_pd(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 0.1199876537563763
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.Pad3D",
  "input": {
    "shape": [1, 1, 1, 1, 1],
    "dtype": "float64"
  },
  "padding": [0, 1, 0, 1, 0, 1],
  "mode": "replicate"
}
```

- `paddle.nn.Pad3D` 默认 `mode='constant'`，需显式指定 `mode='replicate'`。

## Issue 060

llm_enhanced_torch_nn_TransformerEncoderLayer_20251215_184808.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][nn.TransformerEncoderLayer] Output difference anomaly under equivalent migration in TransformerEncoderLayer operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, a severe output discrepancy was detected for `torch.nn.TransformerEncoderLayer` (Maximum difference: 1.5200990438461304).

The root causes include:

1. **Default parameter misalignment**: `activation` and `norm_first` have different defaults between frameworks, affecting the forward computation path.
2. **Weight initialization**: Different frameworks use different weight initialization strategies (e.g., kaiming_uniform_ in PyTorch vs Xavier or other defaults in MindSpore).
3. **No direct TF equivalent**: TensorFlow lacks a direct equivalent high-level API; composed implementations differ in subtle ways.

- Input: shape=[2, 512], dtype=float32
- Parameters: d_model=512, nhead=8
- Maximum difference: 1.5200990438461304

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
torch.manual_seed(42)
mindspore.set_seed(42)

input_np = np.random.randn(2, 512).astype(np.float32)

# PyTorch 执行
input_pt = torch.tensor(input_np).unsqueeze(0)  # [1, 2, 512]
layer_pt = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
layer_pt.eval()
out_pt = layer_pt(input_pt)

# MindSpore 执行
input_ms = mindspore.Tensor(input_np).unsqueeze(0)
layer_ms = mindspore.nn.TransformerEncoderLayer(d_model=512, nhead=8)
layer_ms.set_train(False)
out_ms = layer_ms(input_ms)

pt_np = out_pt.detach().numpy().astype(np.float64)
ms_np = out_ms.asnumpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - ms_np))
print(f"Maximum difference: {max_diff}")
# 权重初始化和默认参数不同导致大差异
# 实测最大差异: 1.5200990438461304
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.nn.TransformerEncoderLayer] Output difference anomaly under equivalent migration in TransformerEncoderLayer operator (sample1)

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

When mapping `torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)` to `mindspore.nn.TransformerEncoderLayer(d_model=512, nhead=8)`, the outputs differ significantly (Maximum difference: 1.5200990438461304) even with the same input. The root causes are weight initialization differences and potential default parameter misalignment (activation, norm_first, LayerNorm eps).

**Describe the expected behavior***

After explicitly aligning all default parameters (activation, norm_first, dropout, eps) and synchronizing weight initialization between frameworks, the outputs should be numerically close within floating-point precision.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(1, 2, 512).astype(np.float32)

torch.manual_seed(42)
layer_pt = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
layer_pt.eval()
out_pt = layer_pt(torch.tensor(input_np))

mindspore.set_seed(42)
layer_ms = mindspore.nn.TransformerEncoderLayer(d_model=512, nhead=8)
layer_ms.set_train(False)
out_ms = layer_ms(mindspore.Tensor(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_ms.asnumpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 1.5200990438461304
```

**Related log / screenshot**

```
comparison_error: Numerical mismatch, maximum difference: 1.5200990438461304
```

**Special notes for this issue**

TransformerEncoderLayer is a complex composite operator. Weight initialization alignment and default parameter matching are critical for cross-framework equivalence.

## Issue 061

llm_enhanced_torch_nn_TransformerEncoderLayer_20251215_184808.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][nn.TransformerEncoderLayer] Output difference anomaly under equivalent migration in TransformerEncoderLayer operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, a severe output discrepancy was detected for `torch.nn.TransformerEncoderLayer` (Maximum difference: 2.1482882499694824).

Same root causes as Issue 060: weight initialization differences, default parameter misalignment (activation, norm_first), and LayerNorm eps. Additionally, TensorFlow has no direct high-level TransformerEncoderLayer API.

- Input: shape=[4, 128, 512], dtype=float32
- Parameters: d_model=512, nhead=8
- Maximum difference: 2.1482882499694824

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(4, 128, 512).astype(np.float32)

torch.manual_seed(42)
layer_pt = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
layer_pt.eval()
out_pt = layer_pt(torch.tensor(input_np))

mindspore.set_seed(42)
layer_ms = mindspore.nn.TransformerEncoderLayer(d_model=512, nhead=8)
layer_ms.set_train(False)
out_ms = layer_ms(mindspore.Tensor(input_np))

pt_np = out_pt.detach().numpy().astype(np.float64)
ms_np = out_ms.asnumpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - ms_np))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 2.1482882499694824
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.nn.TransformerEncoderLayer] Output difference anomaly under equivalent migration in TransformerEncoderLayer operator (sample2)

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

Same as Issue 060. `torch.nn.TransformerEncoderLayer` and `mindspore.nn.TransformerEncoderLayer` produce significantly different outputs (Maximum difference: 2.1482882499694824) with the same input (shape=[4, 128, 512], d_model=512, nhead=8) due to weight initialization and default parameter differences.

**Describe the expected behavior***

After aligning weight initialization, default parameters (activation, norm_first, dropout, LayerNorm eps), outputs should be numerically close.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(4, 128, 512).astype(np.float32)

torch.manual_seed(42)
layer_pt = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
layer_pt.eval()
out_pt = layer_pt(torch.tensor(input_np))

mindspore.set_seed(42)
layer_ms = mindspore.nn.TransformerEncoderLayer(d_model=512, nhead=8)
layer_ms.set_train(False)
out_ms = layer_ms(mindspore.Tensor(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_ms.asnumpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 2.1482882499694824
```

**Related log / screenshot**

```
comparison_error: Numerical mismatch, maximum difference: 2.1482882499694824
```

**Special notes for this issue**

Same as Issue 060. TransformerEncoderLayer requires careful weight initialization alignment for cross-framework comparison.

## Issue 062

llm_enhanced_torch_nn_TransformerEncoderLayer_20251215_184808.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][nn.TransformerEncoderLayer] Output difference anomaly under equivalent migration in TransformerEncoderLayer operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, a severe output discrepancy was detected for `torch.nn.TransformerEncoderLayer` (Maximum difference: 1.3077116012573242).

The root causes include weight initialization differences, LayerNorm eps mismatch, and framework-internal implementation variations of the Transformer encoder layer. Specifically, `d_model=256, nhead=4, dim_feedforward=1024, dropout=0.0, activation='relu', batch_first=True` were explicitly provided.

- Input: shape=[2, 64, 256], dtype=float32
- Parameters: d_model=256, nhead=4, dim_feedforward=1024, dropout=0.0, activation=relu, batch_first=true
- Maximum difference: 1.3077116012573242

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(2, 64, 256).astype(np.float32)

torch.manual_seed(42)
layer_pt = torch.nn.TransformerEncoderLayer(
    d_model=256, nhead=4, dim_feedforward=1024,
    dropout=0.0, activation='relu', batch_first=True
)
layer_pt.eval()
out_pt = layer_pt(torch.tensor(input_np))

mindspore.set_seed(42)
layer_ms = mindspore.nn.TransformerEncoderLayer(
    d_model=256, nhead=4, dim_feedforward=1024,
    dropout=0.0, activation='relu', batch_first=True
)
layer_ms.set_train(False)
out_ms = layer_ms(mindspore.Tensor(input_np))

pt_np = out_pt.detach().numpy().astype(np.float64)
ms_np = out_ms.asnumpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - ms_np))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 1.3077116012573242
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.nn.TransformerEncoderLayer] Output difference anomaly under equivalent migration in TransformerEncoderLayer operator (sample3)

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

When mapping `torch.nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, dropout=0.0, activation='relu', batch_first=True)` to the MindSpore equivalent, outputs differ significantly (Maximum difference: 1.3077116012573242). Root causes: parameter initialization, LayerNorm eps, and internal implementation differences.

**Describe the expected behavior***

After aligning weight initialization and all configuration parameters, outputs should be numerically close.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(2, 64, 256).astype(np.float32)

torch.manual_seed(42)
layer_pt = torch.nn.TransformerEncoderLayer(
    d_model=256, nhead=4, dim_feedforward=1024,
    dropout=0.0, activation='relu', batch_first=True
)
layer_pt.eval()
out_pt = layer_pt(torch.tensor(input_np))

mindspore.set_seed(42)
layer_ms = mindspore.nn.TransformerEncoderLayer(
    d_model=256, nhead=4, dim_feedforward=1024,
    dropout=0.0, activation='relu', batch_first=True
)
layer_ms.set_train(False)
out_ms = layer_ms(mindspore.Tensor(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_ms.asnumpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 1.3077116012573242
```

**Related log / screenshot**

```
comparison_error: Numerical mismatch, maximum difference: 1.3077116012573242
```

**Special notes for this issue**

Same composite operator issue. Even with explicit parameter alignment, weight initialization differences cause significant output divergence.

## Issue 063

llm_enhanced_torch_nn_functional_adaptive_max_pool1d_20251125_151911.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.functional.adaptive_max_pool1d] Output difference anomaly under equivalent migration in adaptive_max_pool1d operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.nn.functional.adaptive_max_pool1d` (Maximum difference: 2.9891507625579834).

The root cause is **input data not synchronized**: `adaptive_max_pool1d` is a deterministic pooling operator. If the input data is identical, the output must be identical. The large difference (2.99) indicates the test harness did not fix random seeds, resulting in completely different input data being fed to each framework.

- Input: shape=[1, 64, 8], dtype=torch.float32
- Parameters: output_size=5
- Maximum difference: 2.9891507625579834

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(1, 64, 8).astype(np.float32)

# PyTorch 执行
input_pt = torch.tensor(input_np)
out_pt = torch.nn.functional.adaptive_max_pool1d(input_pt, output_size=5)

# PaddlePaddle 执行
input_pd = paddle.to_tensor(input_np)
out_pd = paddle.nn.functional.adaptive_max_pool1d(input_pd, output_size=5)

pt_np = out_pt.detach().numpy().astype(np.float64)
pd_np = out_pd.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"PyTorch output shape: {out_pt.shape}")
print(f"PaddlePaddle output shape: {out_pd.shape}")
print(f"Maximum difference: {max_diff}")
# 确定性算子，输入同步后差异应为 0
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.functional.adaptive_max_pool1d] Output difference anomaly under equivalent migration in adaptive_max_pool1d operator (sample1)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`adaptive_max_pool1d` 算子的最大输出差异为 2.9891507625579834。

根本原因是**输入数据未同步**：`adaptive_max_pool1d` 是确定性池化算子，若输入一致，输出必一致。差异值 2.99 远超精度误差，且原始 JSON 中仅有 Shape 信息无 Sample Values，表明两端使用了不同的随机输入（未固定随机种子）。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(1, 64, 8).astype(np.float32)

out_pt = torch.nn.functional.adaptive_max_pool1d(torch.tensor(input_np), output_size=5)
out_pd = paddle.nn.functional.adaptive_max_pool1d(paddle.to_tensor(input_np), output_size=5)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 输入同步后差异应为 0
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.functional.adaptive_max_pool1d",
  "input": {
    "shape": [1, 64, 8],
    "dtype": "float32"
  },
  "output_size": 5
}
```

- 确定性算子，输入未同步是差异的唯一原因。

## Issue 064

llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.functional.adaptive_max_pool2d] Output difference anomaly under equivalent migration in adaptive_max_pool2d operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.nn.functional.adaptive_max_pool2d` (Maximum difference: 20.0).

The root cause is **input data not synchronized**: `adaptive_max_pool2d` is a deterministic pooling operator. The extremely large difference (20.0) confirms the two frameworks received completely different random input data.

- Input: shape=[1, 64, 8, 9], dtype=float32 (no sample values in original JSON)
- Parameters: output_size=[5, 7]
- Maximum difference: 20.0

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(1, 64, 8, 9).astype(np.float32)

out_pt = torch.nn.functional.adaptive_max_pool2d(torch.tensor(input_np), output_size=(5, 7))
out_pd = paddle.nn.functional.adaptive_max_pool2d(paddle.to_tensor(input_np), output_size=(5, 7))

pt_np = out_pt.detach().numpy().astype(np.float64)
pd_np = out_pd.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"Maximum difference: {max_diff}")
# 确定性算子，输入同步后差异应为 0
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.functional.adaptive_max_pool2d] Output difference anomaly under equivalent migration in adaptive_max_pool2d operator (sample1)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`adaptive_max_pool2d` 算子的最大输出差异为 20.0。

根本原因是**输入数据未同步**（未固定随机种子）。`adaptive_max_pool2d` 是确定性池化算子，若输入一致，输出必一致。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(1, 64, 8, 9).astype(np.float32)

out_pt = torch.nn.functional.adaptive_max_pool2d(torch.tensor(input_np), output_size=(5, 7))
out_pd = paddle.nn.functional.adaptive_max_pool2d(paddle.to_tensor(input_np), output_size=(5, 7))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.functional.adaptive_max_pool2d",
  "input": {
    "shape": [1, 64, 8, 9],
    "dtype": "float32"
  },
  "output_size": [5, 7]
}
```

- 确定性算子，输入未同步是差异的唯一原因。

## Issue 065

llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.functional.adaptive_max_pool2d] Output difference anomaly under equivalent migration in adaptive_max_pool2d operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.nn.functional.adaptive_max_pool2d` (Maximum difference: 57.0).

Same root cause as Issue 064: input data not synchronized across frameworks.

- Input: shape=[2, 64, 16, 18], dtype=float32
- Parameters: output_size=[7, 9]
- Maximum difference: 57.0

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 64, 16, 18).astype(np.float32)

out_pt = torch.nn.functional.adaptive_max_pool2d(torch.tensor(input_np), output_size=(7, 9))
out_pd = paddle.nn.functional.adaptive_max_pool2d(paddle.to_tensor(input_np), output_size=(7, 9))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 确定性算子，输入同步后差异应为 0
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.functional.adaptive_max_pool2d] Output difference anomaly under equivalent migration in adaptive_max_pool2d operator (sample2)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`adaptive_max_pool2d` 算子的最大输出差异为 57.0。

与 Issue 064 相同的根本原因：输入数据未同步。确定性池化算子，输入一致时输出必一致。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 64, 16, 18).astype(np.float32)

out_pt = torch.nn.functional.adaptive_max_pool2d(torch.tensor(input_np), output_size=(7, 9))
out_pd = paddle.nn.functional.adaptive_max_pool2d(paddle.to_tensor(input_np), output_size=(7, 9))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.functional.adaptive_max_pool2d",
  "input": {
    "shape": [2, 64, 16, 18],
    "dtype": "float32"
  },
  "output_size": [7, 9]
}
```

## Issue 066

llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample5.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.functional.adaptive_max_pool2d] Output difference anomaly under equivalent migration in adaptive_max_pool2d operator (sample5)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, an output discrepancy was detected for `torch.nn.functional.adaptive_max_pool2d` (Maximum difference: 6.0).

The root cause is **input data not synchronized**: The pooling operator is deterministic. The JSON only contained Shape info without sample values, indicating different random inputs were used. Additionally, TensorFlow has no direct equivalent of PyTorch's `adaptive_max_pool2d`.

- Input: shape=[1, 32, 8, 5], dtype=float64
- Parameters: output_size=[6, 4]
- Maximum difference: 6.0

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(1, 32, 8, 5).astype(np.float64)

out_pt = torch.nn.functional.adaptive_max_pool2d(torch.tensor(input_np), output_size=(6, 4))
out_pd = paddle.nn.functional.adaptive_max_pool2d(paddle.to_tensor(input_np), output_size=(6, 4))

max_diff = np.max(np.abs(out_pt.detach().numpy() - out_pd.numpy()))
print(f"Maximum difference: {max_diff}")
# 确定性算子，输入同步后差异应为 0
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.functional.adaptive_max_pool2d] Output difference anomaly under equivalent migration in adaptive_max_pool2d operator (sample5)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`adaptive_max_pool2d` 算子的最大输出差异为 6.0。

根本原因是**输入数据未同步**。池化算子是确定性的，若输入一致，输出必一致。差异值（6.0）远超精度误差，且 JSON 中仅有 Shape 信息无 Sample Values，表明两端使用了不同的随机输入。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(1, 32, 8, 5).astype(np.float64)

out_pt = torch.nn.functional.adaptive_max_pool2d(torch.tensor(input_np), output_size=(6, 4))
out_pd = paddle.nn.functional.adaptive_max_pool2d(paddle.to_tensor(input_np), output_size=(6, 4))

max_diff = np.max(np.abs(out_pt.detach().numpy() - out_pd.numpy()))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.functional.adaptive_max_pool2d",
  "input": {
    "shape": [1, 32, 8, 5],
    "dtype": "float64"
  },
  "output_size": [6, 4]
}
```

## Issue 067

llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample6.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.functional.adaptive_max_pool2d] Output difference anomaly under equivalent migration in adaptive_max_pool2d operator (sample6)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, an output discrepancy was detected for `torch.nn.functional.adaptive_max_pool2d` (Maximum difference: 1.0).

Uniquely, this case has **both inputs and default parameters identical**. The input shape is [1, 1, 4, 4] with output_size=[1, 1]. Despite aligned inputs, a difference of 1.0 was observed, suggesting a potential numerical or implementation-level difference in how the two frameworks handle adaptive pooling for this specific edge case (reducing a 4x4 input to 1x1 with global max pooling).

- Input: shape=[1, 1, 4, 4], dtype=float32
- Parameters: output_size=[1, 1]
- Maximum difference: 1.0

```python
import numpy as np
import torch
import paddle

# 使用原始样本数据
input_data = np.array([[[[-0.07065916061401367, -0.8627572059631348,
                           -0.9025261402130127, 0.3025325536727905,
                           -0.9171720743179321, -1.0291898250579834,
                           -0.5254021286964417, 0.2561033368110657,
                           2.014113664627075, 0.5817925930023193,
                           -2.044620990753174, -0.07028037309646606,
                           0.43254658579826355, 0.21523572504520416,
                           0.4683907926082611, -0.0023083686828613]]]], dtype=np.float32).reshape(1, 1, 4, 4)

out_pt = torch.nn.functional.adaptive_max_pool2d(torch.tensor(input_data), output_size=(1, 1))
out_pd = paddle.nn.functional.adaptive_max_pool2d(paddle.to_tensor(input_data), output_size=(1, 1))

pt_np = out_pt.detach().numpy().astype(np.float64)
pd_np = out_pd.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"PyTorch output: {out_pt.detach().numpy()}")
print(f"PaddlePaddle output: {out_pd.numpy()}")
print(f"Maximum difference: {max_diff}")
# 输入和参数均已对齐，仍存在差异 1.0
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.functional.adaptive_max_pool2d] Output difference anomaly under equivalent migration in adaptive_max_pool2d operator (sample6)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`adaptive_max_pool2d` 算子在输入内容和默认值均一致的情况下，仍出现数值为 1.0 的差异。

输入 shape=[1, 1, 4, 4]，output_size=[1, 1]（全局最大池化）。这是一种边界情况，两框架对 4x4→1x1 的自适应池化实现可能存在细微差异。

```python
import numpy as np
import torch
import paddle

input_data = np.array([[[[-0.07065916, -0.86275721,
                           -0.90252614, 0.30253255,
                           -0.91717207, -1.02918983,
                           -0.52540213, 0.25610334,
                           2.01411366, 0.58179259,
                           -2.04462099, -0.07028037,
                           0.43254659, 0.21523573,
                           0.46839079, -0.00230837]]]], dtype=np.float32).reshape(1, 1, 4, 4)

out_pt = torch.nn.functional.adaptive_max_pool2d(torch.tensor(input_data), output_size=(1, 1))
out_pd = paddle.nn.functional.adaptive_max_pool2d(paddle.to_tensor(input_data), output_size=(1, 1))

print(f"PyTorch: {out_pt.detach().numpy()}, Paddle: {out_pd.numpy()}")
print(f"Maximum difference: {np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.functional.adaptive_max_pool2d",
  "input": {
    "shape": [1, 1, 4, 4],
    "dtype": "float32"
  },
  "output_size": [1, 1]
}
```

- 输入内容和默认值一致但仍有差异 1.0，疑似实现层面的差异。

## Issue 068

llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample7.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.functional.adaptive_max_pool2d] Output difference anomaly under equivalent migration in adaptive_max_pool2d operator (sample7)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.nn.functional.adaptive_max_pool2d` (Maximum difference: 20.0).

Same root cause as Issue 064: input data not synchronized.

- Input: shape=[1, 64, 10, 9], dtype=float32
- Parameters: output_size=[7, 7]
- Maximum difference: 20.0

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(1, 64, 10, 9).astype(np.float32)

out_pt = torch.nn.functional.adaptive_max_pool2d(torch.tensor(input_np), output_size=(7, 7))
out_pd = paddle.nn.functional.adaptive_max_pool2d(paddle.to_tensor(input_np), output_size=(7, 7))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 确定性算子，输入同步后差异应为 0
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.functional.adaptive_max_pool2d] Output difference anomaly under equivalent migration in adaptive_max_pool2d operator (sample7)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`adaptive_max_pool2d` 算子的最大输出差异为 20.0。与 Issue 064 相同原因：输入数据未同步。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(1, 64, 10, 9).astype(np.float32)

out_pt = torch.nn.functional.adaptive_max_pool2d(torch.tensor(input_np), output_size=(7, 7))
out_pd = paddle.nn.functional.adaptive_max_pool2d(paddle.to_tensor(input_np), output_size=(7, 7))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.functional.adaptive_max_pool2d",
  "input": {
    "shape": [1, 64, 10, 9],
    "dtype": "float32"
  },
  "output_size": [7, 7]
}
```

## Issue 069

llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample8.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.functional.adaptive_max_pool2d] Output difference anomaly under equivalent migration in adaptive_max_pool2d operator (sample8)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.nn.functional.adaptive_max_pool2d` (Maximum difference: 14.0).

Same root cause as Issue 064: input data not synchronized.

- Input: shape=[1, 64, 5, 3], dtype=float32
- Parameters: output_size=[1, 1]
- Maximum difference: 14.0

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(1, 64, 5, 3).astype(np.float32)

out_pt = torch.nn.functional.adaptive_max_pool2d(torch.tensor(input_np), output_size=(1, 1))
out_pd = paddle.nn.functional.adaptive_max_pool2d(paddle.to_tensor(input_np), output_size=(1, 1))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 确定性算子，输入同步后差异应为 0
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.functional.adaptive_max_pool2d] Output difference anomaly under equivalent migration in adaptive_max_pool2d operator (sample8)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`adaptive_max_pool2d` 算子的最大输出差异为 14.0。与 Issue 064 相同原因：输入数据未同步。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(1, 64, 5, 3).astype(np.float32)

out_pt = torch.nn.functional.adaptive_max_pool2d(torch.tensor(input_np), output_size=(1, 1))
out_pd = paddle.nn.functional.adaptive_max_pool2d(paddle.to_tensor(input_np), output_size=(1, 1))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.functional.adaptive_max_pool2d",
  "input": {
    "shape": [1, 64, 5, 3],
    "dtype": "float32"
  },
  "output_size": [1, 1]
}
```

## Issue 070

llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample9.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.functional.adaptive_max_pool2d] Output difference anomaly under equivalent migration in adaptive_max_pool2d operator (sample9)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, an output discrepancy was detected for `torch.nn.functional.adaptive_max_pool2d` (Maximum difference: 9.0).

This case has **input parameters and defaults aligned**, yet a difference of 9.0 was observed. The input shape is [2, 64, 6, 4] with output_size=[2, 2]. This suggests a potential implementation difference in how adaptive pooling computes window boundaries when input dimensions are not evenly divisible by output dimensions.

- Input: shape=[2, 64, 6, 4], dtype=float32
- Parameters: output_size=[2, 2]
- Maximum difference: 9.0

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 64, 6, 4).astype(np.float32)

out_pt = torch.nn.functional.adaptive_max_pool2d(torch.tensor(input_np), output_size=(2, 2))
out_pd = paddle.nn.functional.adaptive_max_pool2d(paddle.to_tensor(input_np), output_size=(2, 2))

pt_np = out_pt.detach().numpy().astype(np.float64)
pd_np = out_pd.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"Maximum difference: {max_diff}")
# 输入和参数对齐后仍有差异，疑似实现层面差异
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.functional.adaptive_max_pool2d] Output difference anomaly under equivalent migration in adaptive_max_pool2d operator (sample9)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`adaptive_max_pool2d` 算子在输入参数和默认值一致的情况下，仍出现数值差异 9.0。

输入 shape=[2, 64, 6, 4]，output_size=[2, 2]。两框架对 adaptive pooling 的窗口划分策略（当输入维度不能被输出维度整除时）可能存在实现差异。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 64, 6, 4).astype(np.float32)

out_pt = torch.nn.functional.adaptive_max_pool2d(torch.tensor(input_np), output_size=(2, 2))
out_pd = paddle.nn.functional.adaptive_max_pool2d(paddle.to_tensor(input_np), output_size=(2, 2))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.functional.adaptive_max_pool2d",
  "input": {
    "shape": [2, 64, 6, 4],
    "dtype": "float32"
  },
  "output_size": [2, 2]
}
```

- 输入参数一致但仍有差异，疑似 adaptive pooling 的窗口划分策略不同。

## Issue 071

llm_enhanced_torch_nn_functional_avg_pool2d_20251215_172318.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.functional.avg_pool2d] Shape mismatch under equivalent migration in avg_pool2d operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a **shape mismatch** was detected for `torch.nn.functional.avg_pool2d`: PyTorch output shape (128, 24, 16, 16) vs TensorFlow output shape (128, 24, 15, 15).

The root cause is **padding semantic difference**: PyTorch uses `padding=1` (explicit symmetric zero-padding of 1 pixel on each side), while TensorFlow uses `padding="VALID"` (no padding at all). With kernel_size=3, stride=2, input 32x32:
- PyTorch: `floor((32 + 2*1 - 3) / 2) + 1 = 16`
- TensorFlow VALID: `floor((32 - 3) / 2) + 1 = 15`

- Input: shape=[128, 24, 32, 32], dtype=float32
- PT params: kernel_size=3, stride=2, padding=1, count_include_pad=true
- TF params: ksize=3, strides=2, padding="VALID", data_format="NCHW"
- Error: Shape mismatch: (128,24,16,16) vs (128,24,15,15)

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(128, 24, 32, 32).astype(np.float32)

# PyTorch: padding=1 (显式填充)
out_pt = torch.nn.functional.avg_pool2d(
    torch.tensor(input_np), kernel_size=3, stride=2, padding=1, count_include_pad=True
)

# TensorFlow: padding="VALID" (无填充)
out_tf = tf.nn.avg_pool2d(
    tf.constant(input_np), ksize=3, strides=2, padding="VALID", data_format="NCHW"
)

print(f"PyTorch output shape: {out_pt.shape}")
print(f"TensorFlow output shape: {out_tf.shape}")
# Shape mismatch: (128, 24, 16, 16) vs (128, 24, 15, 15)
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.nn.avg_pool2d] Shape mismatch under equivalent migration in avg_pool2d operator (sample2)

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

When mapping `torch.nn.functional.avg_pool2d(kernel_size=3, stride=2, padding=1)` to `tf.nn.avg_pool2d(ksize=3, strides=2, padding="VALID")`, the output shapes differ: PyTorch (128, 24, 16, 16) vs TensorFlow (128, 24, 15, 15).

- PyTorch `padding=1` explicitly adds 1-pixel zero-padding: output = floor((32+2-3)/2)+1 = 16.
- TensorFlow `padding="VALID"` uses no padding: output = floor((32-3)/2)+1 = 15.

Expected behavior: Migration should use `tf.pad` for explicit padding before `tf.nn.avg_pool2d(padding="VALID")`, or correctly compute the equivalent `padding="SAME"` configuration.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(128, 24, 32, 32).astype(np.float32)

out_pt = torch.nn.functional.avg_pool2d(torch.tensor(input_np), kernel_size=3, stride=2, padding=1)
out_tf = tf.nn.avg_pool2d(tf.constant(input_np), ksize=3, strides=2, padding="VALID", data_format="NCHW")

print(f"PyTorch: {out_pt.shape}, TensorFlow: {out_tf.shape}")
# Shape mismatch: (128, 24, 16, 16) vs (128, 24, 15, 15)
```

**Relevant log output**

```
comparison_error: Shape mismatch: PyTorch (128, 24, 16, 16) vs TensorFlow (128, 24, 15, 15)
```

## Issue 072

llm_enhanced_torch_nn_functional_avg_pool2d_20251215_172318.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.functional.avg_pool2d] Output difference anomaly under equivalent migration in avg_pool2d operator (sample4)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a severe output discrepancy was detected for `torch.nn.functional.avg_pool2d` (Maximum difference: 1.793502688407898).

The root cause is **padding semantic mismatch**: PyTorch `padding=1` adds symmetric 1-pixel zero-padding, while TensorFlow `padding="SAME"` uses asymmetric auto-padding with different counting behavior. The `count_include_pad` default in PyTorch includes padded zeros in the average denominator, while TensorFlow's SAME padding normalizes only by the actual overlapping window size.

- Input: shape=[128, 240, 16, 16], dtype=float32
- PT params: kernel_size=3, stride=2, padding=1
- TF params: ksize=[1,1,3,3], strides=[1,1,2,2], padding="SAME", data_format="NCHW"
- Maximum difference: 1.793502688407898

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 240, 16, 16).astype(np.float32)  # 缩小 batch 节省内存

# PyTorch: padding=1, count_include_pad=True (default)
out_pt = torch.nn.functional.avg_pool2d(
    torch.tensor(input_np), kernel_size=3, stride=2, padding=1
)

# TensorFlow: padding="SAME"
out_tf = tf.nn.avg_pool2d(
    tf.constant(input_np), ksize=[1, 1, 3, 3], strides=[1, 1, 2, 2],
    padding="SAME", data_format="NCHW"
)

pt_np = out_pt.detach().numpy().astype(np.float64)
tf_np = out_tf.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - tf_np))
print(f"PyTorch output shape: {out_pt.shape}")
print(f"TensorFlow output shape: {out_tf.shape}")
print(f"Maximum difference: {max_diff}")
# padding=1 vs "SAME" 语义不同导致数值差异
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.nn.avg_pool2d] Output difference anomaly under equivalent migration in avg_pool2d operator (sample4)

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

When mapping `torch.nn.functional.avg_pool2d(kernel_size=3, stride=2, padding=1)` to `tf.nn.avg_pool2d(ksize=[1,1,3,3], strides=[1,1,2,2], padding="SAME")`, the outputs differ by up to 1.793502688407898.

PyTorch's `padding=1` + `count_include_pad=True` counts zero-padded elements in the average, while TensorFlow's `padding="SAME"` uses auto-padding where the average division only considers valid (non-padded) elements.

Expected behavior: Migration should explicitly implement PyTorch's padding behavior using `tf.pad` + `tf.nn.avg_pool2d(padding="VALID")` to match count_include_pad semantics.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 240, 16, 16).astype(np.float32)

out_pt = torch.nn.functional.avg_pool2d(torch.tensor(input_np), kernel_size=3, stride=2, padding=1)
out_tf = tf.nn.avg_pool2d(tf.constant(input_np), ksize=[1,1,3,3], strides=[1,1,2,2], padding="SAME", data_format="NCHW")

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 1.793502688407898
```

## Issue 073

llm_enhanced_torch_nn_functional_avg_pool2d_20251215_172318.json_sample6.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.functional.avg_pool2d] Output difference anomaly under equivalent migration in avg_pool2d operator (sample6)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a severe output discrepancy was detected for `torch.nn.functional.avg_pool2d` (Maximum difference: 1.6940934658050537).

Same root cause as Issue 072: PyTorch's explicit `padding=1` + `count_include_pad=True` semantics differ from TensorFlow's `padding="SAME"` auto-padding. PyTorch includes zero-padded elements in the average denominator, while TensorFlow only counts valid elements.

- Input: shape=[128, 480, 8, 8], dtype=float32
- PT params: kernel_size=3, stride=2, padding=1, count_include_pad=true
- TF params: ksize=3, strides=2, padding="SAME", data_format="NCHW"
- Maximum difference: 1.6940934658050537

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 480, 8, 8).astype(np.float32)  # 缩小 batch

out_pt = torch.nn.functional.avg_pool2d(
    torch.tensor(input_np), kernel_size=3, stride=2, padding=1, count_include_pad=True
)
out_tf = tf.nn.avg_pool2d(
    tf.constant(input_np), ksize=3, strides=2, padding="SAME", data_format="NCHW"
)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"PyTorch shape: {out_pt.shape}, TF shape: {out_tf.shape}")
print(f"Maximum difference: {max_diff}")
# padding 语义不等价
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.nn.avg_pool2d] Output difference anomaly under equivalent migration in avg_pool2d operator (sample6)

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

Same as Issue 072. PyTorch's `padding=1` + `count_include_pad=True` vs TensorFlow's `padding="SAME"` produces different averages at boundary positions (Maximum difference: 1.6940934658050537).

Expected behavior: Same as Issue 072 — use explicit `tf.pad` + `padding="VALID"` to match PyTorch's fixed padding and count_include_pad semantics.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 480, 8, 8).astype(np.float32)

out_pt = torch.nn.functional.avg_pool2d(torch.tensor(input_np), kernel_size=3, stride=2, padding=1, count_include_pad=True)
out_tf = tf.nn.avg_pool2d(tf.constant(input_np), ksize=3, strides=2, padding="SAME", data_format="NCHW")

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 1.6940934658050537
```

## Issue 074

llm_enhanced_torch_nn_functional_avg_pool3d_20251215_201404.json_sample5.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.functional.avg_pool3d] Output difference anomaly under equivalent migration in avg_pool3d operator (sample5)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for `torch.nn.functional.avg_pool3d` (Maximum difference: 0.30140891671180725).

The root causes include:

1. **count_include_pad**: PyTorch's `count_include_pad=True` includes padded zeros in the average denominator. TensorFlow's `tf.nn.avg_pool3d` does not support this parameter — it only counts valid (non-padded) elements.
2. **Padding format**: PyTorch uses `padding=1` (symmetric per-dimension), while TensorFlow requires 5D ksize/strides lists matching `[N, C, D, H, W]` for NCDHW format. `padding="SAME"` has different auto-padding behavior.

- Input: shape=[1, 3, 4, 4, 4], dtype=float32
- PT params: kernel_size=[3,3,3], stride=[2,2,2], padding=1, count_include_pad=true
- TF params: ksize=[3,3,3], strides=[2,2,2], padding="SAME", data_format="NCDHW"
- Maximum difference: 0.30140891671180725

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 3, 4, 4, 4).astype(np.float32)

# PyTorch: count_include_pad=True
out_pt = torch.nn.functional.avg_pool3d(
    torch.tensor(input_np), kernel_size=[3,3,3], stride=[2,2,2],
    padding=1, count_include_pad=True
)

# TensorFlow: padding="SAME", 不支持 count_include_pad
out_tf = tf.nn.avg_pool3d(
    tf.constant(input_np), ksize=[1, 1, 3, 3, 3], strides=[1, 1, 2, 2, 2],
    padding="SAME", data_format="NCDHW"
)

pt_np = out_pt.detach().numpy().astype(np.float64)
tf_np = out_tf.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - tf_np))
print(f"PyTorch shape: {out_pt.shape}, TF shape: {out_tf.shape}")
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 0.30140891671180725
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.nn.avg_pool3d] Output difference anomaly under equivalent migration in avg_pool3d operator (sample5)

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

When mapping `torch.nn.functional.avg_pool3d(kernel_size=[3,3,3], stride=[2,2,2], padding=1, count_include_pad=True)` to `tf.nn.avg_pool3d(ksize=[3,3,3], strides=[2,2,2], padding="SAME")`, outputs differ by up to 0.30140891671180725.

TensorFlow's `tf.nn.avg_pool3d` does not support `count_include_pad=True`. PyTorch's `padding=1` is fixed symmetric padding, while TensorFlow's `padding="SAME"` is dynamic auto-padding. The ksize/strides for NCDHW format need to be 5D lists.

Expected behavior: Migration should use explicit padding (`tf.pad`) before `tf.nn.avg_pool3d(padding="VALID")` and manually implement count_include_pad logic.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 3, 4, 4, 4).astype(np.float32)

out_pt = torch.nn.functional.avg_pool3d(torch.tensor(input_np), kernel_size=[3,3,3], stride=[2,2,2], padding=1, count_include_pad=True)
out_tf = tf.nn.avg_pool3d(tf.constant(input_np), ksize=[1,1,3,3,3], strides=[1,1,2,2,2], padding="SAME", data_format="NCDHW")

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.30140891671180725
```

## Issue 075

llm_enhanced_torch_nn_functional_batch_norm_20251202_005716.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.functional.batch_norm] Output difference anomaly under equivalent migration in batch_norm operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.nn.functional.batch_norm` (Maximum difference: 0.4121027693343535).

The root cause is **eps default value mismatch**: PyTorch's test configuration uses `eps=1e-7`, while Paddle's `paddle.nn.functional.batch_norm` defaults to `eps=1e-5`. Since batch normalization divides by `sqrt(var + eps)`, a 100x difference in eps significantly affects the output, especially when variance values are small.

- Input: shape=[2, 1024], dtype=float64
- PT params: training=true, momentum=0.1, eps=1e-7
- PD params: training=true, momentum=0.1, eps=1e-5 (default, not explicitly set)
- Maximum difference: 0.4121027693343535

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 1024).astype(np.float64)
running_mean = np.random.randn(1024).astype(np.float64)
running_var = np.abs(np.random.randn(1024).astype(np.float64))
weight = np.random.randn(1024).astype(np.float64)
bias = np.random.randn(1024).astype(np.float64)

# PyTorch: eps=1e-7
out_pt = torch.nn.functional.batch_norm(
    torch.tensor(input_np), torch.tensor(running_mean), torch.tensor(running_var),
    weight=torch.tensor(weight), bias=torch.tensor(bias),
    training=True, momentum=0.1, eps=1e-7
)

# PaddlePaddle: 默认 eps=1e-5
out_pd = paddle.nn.functional.batch_norm(
    paddle.to_tensor(input_np), paddle.to_tensor(running_mean), paddle.to_tensor(running_var),
    weight=paddle.to_tensor(weight), bias=paddle.to_tensor(bias),
    training=True, momentum=0.1
)

max_diff = np.max(np.abs(out_pt.detach().numpy() - out_pd.numpy()))
print(f"Maximum difference: {max_diff}")
# eps=1e-7 vs eps=1e-5 导致数值差异
# 实测最大差异: 0.4121027693343535
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.functional.batch_norm] Output difference anomaly under equivalent migration in batch_norm operator (sample1)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`batch_norm` 算子的最大输出差异为 0.4121027693343535。

根本原因是 **eps 默认值不一致**：PyTorch 的测试配置使用 `eps=1e-7`，而 Paddle 的 `paddle.nn.functional.batch_norm` 未显式指定 eps 时默认值为 `1e-5`。由于 batch normalization 计算中除以 `sqrt(var + eps)`，eps 相差 100 倍会显著影响输出。

迁移代码应显式传递 `eps=1e-7` 给 Paddle 以对齐行为。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 1024).astype(np.float64)
running_mean = np.random.randn(1024).astype(np.float64)
running_var = np.abs(np.random.randn(1024).astype(np.float64))
weight = np.random.randn(1024).astype(np.float64)
bias = np.random.randn(1024).astype(np.float64)

# PyTorch: eps=1e-7
out_pt = torch.nn.functional.batch_norm(
    torch.tensor(input_np), torch.tensor(running_mean), torch.tensor(running_var),
    weight=torch.tensor(weight), bias=torch.tensor(bias),
    training=True, momentum=0.1, eps=1e-7
)

# Paddle: 显式 eps=1e-7 以对齐
out_pd = paddle.nn.functional.batch_norm(
    paddle.to_tensor(input_np), paddle.to_tensor(running_mean), paddle.to_tensor(running_var),
    weight=paddle.to_tensor(weight), bias=paddle.to_tensor(bias),
    training=True, momentum=0.1, epsilon=1e-7
)

max_diff = np.max(np.abs(out_pt.detach().numpy() - out_pd.numpy()))
print(f"Maximum difference (with aligned eps): {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.functional.batch_norm",
  "input": {"shape": [2, 1024], "dtype": "float64"},
  "running_mean": {"shape": [1024], "dtype": "float64"},
  "running_var": {"shape": [1024], "dtype": "float64"},
  "weight": {"shape": [1024], "dtype": "float64"},
  "bias": {"shape": [1024], "dtype": "float64"},
  "training": true,
  "momentum": 0.1
}
```

- Paddle 未显式指定 eps，默认 1e-5 与 PyTorch 的 1e-7 不同。

## Issue 076

llm_enhanced_torch_nn_functional_conv2d_20251202_132551.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.functional.conv2d] Output difference anomaly under equivalent migration in conv2d operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a small but consistent output discrepancy was detected for `torch.nn.functional.conv2d` (Maximum difference: 4.1961669921875e-05).

The root cause is **float32 accumulation precision error**: The difference magnitude (~4e-5) is characteristic of float32 convolution operations where different backends (cuDNN, MKL, OpenBLAS) or different algorithms (Winograd vs GEMM) perform multiply-accumulate operations in different orders, leading to slightly different floating-point rounding.

- Input: shape=[8, 16, 128, 256], dtype=float32
- Weight: shape=[48, 16, 3, 3], Bias: shape=[48]
- Parameters: stride=1, padding=0, dilation=[1,1], groups=1
- Maximum difference: 4.1961669921875e-05

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(8, 16, 128, 256).astype(np.float32)
weight_np = np.random.randn(48, 16, 3, 3).astype(np.float32)
bias_np = np.random.randn(48).astype(np.float32)

out_pt = torch.nn.functional.conv2d(
    torch.tensor(input_np), torch.tensor(weight_np), torch.tensor(bias_np),
    stride=1, padding=0, dilation=(1, 1), groups=1
)
out_pd = paddle.nn.functional.conv2d(
    paddle.to_tensor(input_np), paddle.to_tensor(weight_np), paddle.to_tensor(bias_np),
    stride=1, padding=0, dilation=(1, 1), groups=1
)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# Float32 卷积精度误差，属于底层库实现差异
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.functional.conv2d] Output difference anomaly under equivalent migration in conv2d operator (sample1)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`conv2d` 算子的最大输出差异为 4.1961669921875e-05。

差异值在 $3 \times 10^{-5}$ 到 $4 \times 10^{-5}$ 之间，属于 Float32 卷积运算在不同后端（如 cuDNN vs MKL/OpenBLAS）或不同算法（Winograd vs GEMM）实现下的精度波动。这是浮点计算的固有属性，不是功能性 bug。

建议将此类 $< 10^{-4}$ 的差异标记为"可接受的精度差异"。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(8, 16, 128, 256).astype(np.float32)
weight_np = np.random.randn(48, 16, 3, 3).astype(np.float32)
bias_np = np.random.randn(48).astype(np.float32)

out_pt = torch.nn.functional.conv2d(
    torch.tensor(input_np), torch.tensor(weight_np), torch.tensor(bias_np),
    stride=1, padding=0, dilation=(1, 1), groups=1
)
out_pd = paddle.nn.functional.conv2d(
    paddle.to_tensor(input_np), paddle.to_tensor(weight_np), paddle.to_tensor(bias_np),
    stride=1, padding=0, dilation=(1, 1), groups=1
)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.functional.conv2d",
  "input": {"shape": [8, 16, 128, 256], "dtype": "float32"},
  "weight": {"shape": [48, 16, 3, 3], "dtype": "float32"},
  "bias": {"shape": [48], "dtype": "float32"},
  "stride": 1,
  "padding": 0,
  "dilation": [1, 1],
  "groups": 1
}
```

- Float32 卷积精度误差，差异量级 ~$10^{-5}$，属于正常浮点精度波动。

## Issue 077

llm_enhanced_torch_nn_functional_conv2d_20251202_132551.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.functional.conv2d] Output difference anomaly under equivalent migration in conv2d operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a small output discrepancy was detected for `torch.nn.functional.conv2d` (Maximum difference: 3.4332275390625e-05).

Same root cause as Issue 076: float32 accumulation precision differences across different backend implementations.

- Input: shape=[8, 64, 64, 128], dtype=float32
- Weight: shape=[64, 64, 3, 1], Bias: shape=[64]
- Parameters: stride=1, padding=0, dilation=[1,1], groups=1
- Maximum difference: 3.4332275390625e-05

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(8, 64, 64, 128).astype(np.float32)
weight_np = np.random.randn(64, 64, 3, 1).astype(np.float32)
bias_np = np.random.randn(64).astype(np.float32)

out_pt = torch.nn.functional.conv2d(
    torch.tensor(input_np), torch.tensor(weight_np), torch.tensor(bias_np),
    stride=1, padding=0, dilation=(1, 1), groups=1
)
out_pd = paddle.nn.functional.conv2d(
    paddle.to_tensor(input_np), paddle.to_tensor(weight_np), paddle.to_tensor(bias_np),
    stride=1, padding=0, dilation=(1, 1), groups=1
)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# Float32 精度误差
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.functional.conv2d] Output difference anomaly under equivalent migration in conv2d operator (sample2)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`conv2d` 算子的最大输出差异为 3.4332275390625e-05。

与 Issue 076 相同原因：Float32 卷积运算在不同后端实现下的精度波动。差异量级 ~$10^{-5}$，属于正常浮点精度误差。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(8, 64, 64, 128).astype(np.float32)
weight_np = np.random.randn(64, 64, 3, 1).astype(np.float32)
bias_np = np.random.randn(64).astype(np.float32)

out_pt = torch.nn.functional.conv2d(
    torch.tensor(input_np), torch.tensor(weight_np), torch.tensor(bias_np),
    stride=1, padding=0, dilation=(1, 1), groups=1
)
out_pd = paddle.nn.functional.conv2d(
    paddle.to_tensor(input_np), paddle.to_tensor(weight_np), paddle.to_tensor(bias_np),
    stride=1, padding=0, dilation=(1, 1), groups=1
)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.functional.conv2d",
  "input": {"shape": [8, 64, 64, 128], "dtype": "float32"},
  "weight": {"shape": [64, 64, 3, 1], "dtype": "float32"},
  "bias": {"shape": [64], "dtype": "float32"},
  "stride": 1,
  "padding": 0,
  "dilation": [1, 1],
  "groups": 1
}
```

## Issue 078

llm_enhanced_torch_nn_functional_conv3d_20251201_233532.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.functional.conv3d] Output difference anomaly under equivalent migration in conv3d operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a small output discrepancy was detected for `torch.nn.functional.conv3d` (Maximum difference: 9.1552734375e-05).

The root cause is **float32 accumulation precision error**: 3D convolution involves extensive multiply-accumulate operations. Different frameworks use different underlying libraries (cuDNN, MKL, OpenBLAS) with different computation orders, causing $10^{-5}$ level precision differences.

- Input: shape=[20, 16, 10, 50, 100], dtype=float32
- Weight: shape=[33, 16, 3, 5, 2], Bias: shape=[33]
- Parameters: stride=[2,1,1], padding=[4,2,0], dilation=[1,1,1], groups=1
- Maximum difference: 9.1552734375e-05

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(20, 16, 10, 50, 100).astype(np.float32)
weight_np = np.random.randn(33, 16, 3, 5, 2).astype(np.float32)
bias_np = np.random.randn(33).astype(np.float32)

out_pt = torch.nn.functional.conv3d(
    torch.tensor(input_np), torch.tensor(weight_np), torch.tensor(bias_np),
    stride=[2, 1, 1], padding=[4, 2, 0], dilation=[1, 1, 1], groups=1
)
out_pd = paddle.nn.functional.conv3d(
    paddle.to_tensor(input_np), paddle.to_tensor(weight_np), paddle.to_tensor(bias_np),
    stride=[2, 1, 1], padding=[4, 2, 0], dilation=[1, 1, 1], groups=1
)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 3D 卷积浮点累加误差
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.functional.conv3d] Output difference anomaly under equivalent migration in conv3d operator (sample1)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`conv3d` 算子的最大输出差异为 9.1552734375e-05。

根本原因是浮点累加误差：3D 卷积涉及大量的乘加运算，不同框架调用的底层库（如 cuDNN, MKL, OpenBLAS）计算顺序不同，导致 $10^{-5}$ 级别的精度差异。这是浮点计算的固有属性。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(20, 16, 10, 50, 100).astype(np.float32)
weight_np = np.random.randn(33, 16, 3, 5, 2).astype(np.float32)
bias_np = np.random.randn(33).astype(np.float32)

out_pt = torch.nn.functional.conv3d(
    torch.tensor(input_np), torch.tensor(weight_np), torch.tensor(bias_np),
    stride=[2, 1, 1], padding=[4, 2, 0], dilation=[1, 1, 1], groups=1
)
out_pd = paddle.nn.functional.conv3d(
    paddle.to_tensor(input_np), paddle.to_tensor(weight_np), paddle.to_tensor(bias_np),
    stride=[2, 1, 1], padding=[4, 2, 0], dilation=[1, 1, 1], groups=1
)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.functional.conv3d",
  "input": {"shape": [20, 16, 10, 50, 100], "dtype": "float32"},
  "weight": {"shape": [33, 16, 3, 5, 2], "dtype": "float32"},
  "bias": {"shape": [33], "dtype": "float32"},
  "stride": [2, 1, 1],
  "padding": [4, 2, 0],
  "dilation": [1, 1, 1],
  "groups": 1
}
```

- 3D 卷积浮点累加误差，差异量级 ~$10^{-5}$。

## Issue 079

llm_enhanced_torch_nn_functional_gelu_20251216_003428.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.functional.gelu] Output difference anomaly under equivalent migration in GELU operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a severe output discrepancy was detected for `torch.nn.functional.gelu` (Maximum difference: 0.7142203739121105).

**Two distinct problems:**

1. **Input data not synchronized**: The bug txt shows PyTorch input value (`-0.5839`) differs from TensorFlow input value (`0.7207`), indicating the test harness did not fix random seeds.
2. **GELU tanh approximation formula difference**: Even with aligned inputs, PyTorch uses $0.5 \cdot x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 x^3)))$ while TensorFlow may use a slightly different approximation formula or precision, though this should produce much smaller differences (~$10^{-7}$).

- Input: shape=[1], dtype=float64
- Parameters: approximate="tanh" (PyTorch) / approximate=True (TensorFlow)
- Maximum difference: 0.7142203739121105

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1).astype(np.float64)

# PyTorch
out_pt = torch.nn.functional.gelu(torch.tensor(input_np), approximate='tanh')

# TensorFlow
out_tf = tf.nn.gelu(tf.constant(input_np), approximate=True)

pt_np = out_pt.detach().numpy()
tf_np = out_tf.numpy()
max_diff = np.max(np.abs(pt_np - tf_np))
print(f"Input: {input_np}")
print(f"PyTorch GELU: {pt_np}")
print(f"TensorFlow GELU: {tf_np}")
print(f"Maximum difference: {max_diff}")
# 输入同步后两框架的 tanh 近似 GELU 输出应非常接近
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.nn.gelu] Output difference anomaly under equivalent migration in GELU operator (sample2)

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

When mapping `torch.nn.functional.gelu(input, approximate='tanh')` to `tf.nn.gelu(x, approximate=True)`, the outputs differ significantly (Maximum difference: 0.7142203739121105). The primary cause is input data not being synchronized (PyTorch input: -0.5839, TF input: 0.7207). Additionally, the two frameworks may use slightly different GELU tanh approximation formulas.

Expected behavior: With synchronized inputs, both frameworks' tanh-approximate GELU should produce nearly identical results.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1).astype(np.float64)

out_pt = torch.nn.functional.gelu(torch.tensor(input_np), approximate='tanh')
out_tf = tf.nn.gelu(tf.constant(input_np), approximate=True)

max_diff = np.max(np.abs(out_pt.detach().numpy() - out_tf.numpy()))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.7142203739121105
```

## Issue 080

llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.functional.interpolate] Shape mismatch under equivalent migration in interpolate operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a **shape mismatch** was detected for `torch.nn.functional.interpolate`: PyTorch output (1, 512, 56, 56) vs TensorFlow output (1, 56, 56, 7).

The root cause is **data format mismatch (NCHW vs NHWC)**:

1. **PyTorch** `torch.nn.functional.interpolate` operates on NCHW format: input (1, 512, 7, 7) → output (1, 512, 56, 56).
2. **TensorFlow** `tf.image.resize` strictly requires NHWC format: it interprets the last dimension (7) as channels, so input (1, 512, 7, 7) is treated as (batch=1, H=512, W=7, C=7), producing output (1, 56, 56, 7).

The migration code must transpose NCHW→NHWC before calling `tf.image.resize`, then transpose NHWC→NCHW afterward.

- Input: shape=[1, 512, 7, 7], dtype=float32
- Parameters: size=[56, 56], mode="bilinear", align_corners=True (PyTorch) / method="bilinear" (TensorFlow)
- Error: Shape mismatch: (1, 512, 56, 56) vs (1, 56, 56, 7)

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 512, 7, 7).astype(np.float32)

# PyTorch: NCHW 格式
out_pt = torch.nn.functional.interpolate(
    torch.tensor(input_np), size=(56, 56), mode='bilinear', align_corners=True
)

# TensorFlow: 需要先转 NHWC
input_nhwc = np.transpose(input_np, (0, 2, 3, 1))  # NCHW -> NHWC
out_tf_nhwc = tf.image.resize(tf.constant(input_nhwc), size=[56, 56], method='bilinear')
out_tf = tf.transpose(out_tf_nhwc, perm=[0, 3, 1, 2])  # NHWC -> NCHW

pt_np = out_pt.detach().numpy().astype(np.float64)
tf_np = out_tf.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - tf_np))
print(f"PyTorch output shape: {out_pt.shape}")
print(f"TensorFlow output shape (after transpose): {out_tf.shape}")
print(f"Maximum difference: {max_diff}")
# 正确做法：NCHW -> NHWC -> tf.image.resize -> NHWC -> NCHW
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.image.resize] Shape mismatch under equivalent migration in interpolate operator (sample1)

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

When mapping `torch.nn.functional.interpolate(input, size=(56,56), mode='bilinear', align_corners=True)` to `tf.image.resize(input, size=[56,56], method='bilinear')`, the output shapes differ: PyTorch (1, 512, 56, 56) vs TensorFlow (1, 56, 56, 7).

PyTorch's `interpolate` uses NCHW format (channels-first), while `tf.image.resize` requires NHWC format (channels-last). Without format conversion, TensorFlow interprets the input dimensions incorrectly.

Expected behavior: Migration should transpose NCHW→NHWC before `tf.image.resize`, then NHWC→NCHW afterward. Additionally, `align_corners=True` has no direct equivalent in `tf.image.resize` — TensorFlow 2.x uses `half_pixel_centers` parameter instead.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 512, 7, 7).astype(np.float32)

out_pt = torch.nn.functional.interpolate(torch.tensor(input_np), size=(56, 56), mode='bilinear', align_corners=True)

# 错误做法：直接传 NCHW 给 tf.image.resize
out_tf_wrong = tf.image.resize(tf.constant(input_np), size=[56, 56], method='bilinear')
print(f"PyTorch: {out_pt.shape}, TF (wrong): {out_tf_wrong.shape}")
# Shape mismatch: (1, 512, 56, 56) vs (1, 56, 56, 7)

# 正确做法：NCHW -> NHWC -> resize -> NCHW
input_nhwc = tf.transpose(tf.constant(input_np), perm=[0, 2, 3, 1])
out_tf_nhwc = tf.image.resize(input_nhwc, size=[56, 56], method='bilinear')
out_tf_correct = tf.transpose(out_tf_nhwc, perm=[0, 3, 1, 2])
print(f"TF (correct): {out_tf_correct.shape}")
```

**Relevant log output**

```
comparison_error: Shape mismatch: PyTorch (1, 512, 56, 56) vs TensorFlow (1, 56, 56, 7)
```
"""

# 追加到文件
with open(output_file, 'a', encoding='utf-8') as f:
    f.write(content)

print(f"Issues 051-080 已成功追加到 {output_file}")
print(f"追加内容长度: {len(content)} 字符")
