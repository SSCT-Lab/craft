# -*- coding: utf-8 -*-
"""
生成 Issue 081-110 的标准格式 issue 内容，
追加写入 138个跨表不一致Case的GitHub Issue-修改版.md
"""

VERSIONS_REF = "同1"

issues_content = r"""
## Issue 081

llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.functional.interpolate] Shape mismatch under equivalent migration in interpolate operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a **shape mismatch** was detected for `torch.nn.functional.interpolate`: PyTorch output (1, 512, 56, 56) vs TensorFlow output (1, 56, 56, 7).

The root cause is **data format mismatch (NCHW vs NHWC)**:

1. **PyTorch** `torch.nn.functional.interpolate` operates on NCHW format: input (1, 512, 7, 7) → output (1, 512, 56, 56).
2. **TensorFlow** `tf.image.resize` strictly requires NHWC format: it interprets the last dimension (7) as channels, so input (1, 512, 7, 7) is treated as (batch=1, H=512, W=7, C=7), producing output (1, 56, 56, 7).

The migration code must transpose NCHW→NHWC before calling `tf.image.resize`, then transpose NHWC→NCHW afterward. Additionally, PyTorch's `align_corners=True` has no direct equivalent in `tf.image.resize`.

- Input: shape=[1, 512, 7, 7], dtype=float32
- PT params: size=[56, 56], mode="bilinear", align_corners=True
- TF params: size=[56, 56], method="bilinear"
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

if pt_np.shape != tf_np.shape:
    print(f"Shape mismatch: PyTorch {pt_np.shape} vs TensorFlow {tf_np.shape}")
else:
    max_diff = np.max(np.abs(pt_np - tf_np))
    print(f"Maximum difference: {max_diff}")
# 正确做法：NCHW -> NHWC -> tf.image.resize -> NHWC -> NCHW
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.image.resize] Shape mismatch under equivalent migration in interpolate operator (sample2)

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

PyTorch's `interpolate` uses NCHW format (channels-first), while `tf.image.resize` requires NHWC format (channels-last). Without format conversion, TensorFlow interprets the input dimensions incorrectly. Additionally, `align_corners=True` has no direct equivalent in `tf.image.resize`.

Expected behavior: Migration should transpose NCHW→NHWC before `tf.image.resize`, then NHWC→NCHW afterward.

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

## Issue 082

llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.functional.interpolate] Shape mismatch under equivalent migration in interpolate operator (sample4)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a **shape mismatch** was detected for `torch.nn.functional.interpolate`: PyTorch output (1, 256, 56, 56) vs TensorFlow output (1, 56, 56, 14).

Same root cause as Issue 081: **data format mismatch (NCHW vs NHWC)**. PyTorch operates on NCHW, while `tf.image.resize` requires NHWC. Without format conversion, TensorFlow misinterprets the channel dimension (14) as height/width.

- Input: shape=[1, 256, 14, 14], dtype=float32
- PT params: size=[56, 56], mode="bilinear", align_corners=True
- TF params: size=[56, 56], method="bilinear"
- Error: Shape mismatch: (1, 256, 56, 56) vs (1, 56, 56, 14)

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 256, 14, 14).astype(np.float32)

# PyTorch: NCHW 格式
out_pt = torch.nn.functional.interpolate(
    torch.tensor(input_np), size=(56, 56), mode='bilinear', align_corners=True
)

# TensorFlow: 需要先转 NHWC
input_nhwc = np.transpose(input_np, (0, 2, 3, 1))
out_tf_nhwc = tf.image.resize(tf.constant(input_nhwc), size=[56, 56], method='bilinear')
out_tf = tf.transpose(out_tf_nhwc, perm=[0, 3, 1, 2])

pt_np = out_pt.detach().numpy().astype(np.float64)
tf_np = out_tf.numpy().astype(np.float64)

if pt_np.shape != tf_np.shape:
    print(f"Shape mismatch: PyTorch {pt_np.shape} vs TensorFlow {tf_np.shape}")
else:
    max_diff = np.max(np.abs(pt_np - tf_np))
    print(f"Maximum difference: {max_diff}")
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.image.resize] Shape mismatch under equivalent migration in interpolate operator (sample4)

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

Same as Issue 081. When mapping `torch.nn.functional.interpolate` to `tf.image.resize` without NCHW→NHWC conversion, shape mismatch occurs: PyTorch (1, 256, 56, 56) vs TensorFlow (1, 56, 56, 14).

Expected behavior: Migration should transpose NCHW→NHWC before `tf.image.resize`, then NHWC→NCHW afterward.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 256, 14, 14).astype(np.float32)

out_pt = torch.nn.functional.interpolate(torch.tensor(input_np), size=(56, 56), mode='bilinear', align_corners=True)
out_tf_wrong = tf.image.resize(tf.constant(input_np), size=[56, 56], method='bilinear')
print(f"PyTorch: {out_pt.shape}, TF (wrong): {out_tf_wrong.shape}")
```

**Relevant log output**

```
comparison_error: Shape mismatch: PyTorch (1, 256, 56, 56) vs TensorFlow (1, 56, 56, 14)
```

## Issue 083

llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample5.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.functional.interpolate] Shape mismatch under equivalent migration in interpolate operator (sample5)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a **shape mismatch** was detected for `torch.nn.functional.interpolate`: PyTorch output (1, 128, 56, 56) vs TensorFlow output (1, 56, 56, 28).

Same root cause as Issue 081: **data format mismatch (NCHW vs NHWC)**. Additionally, PyTorch's `align_corners=True` affects interpolation grid alignment, which needs special handling in TensorFlow.

- Input: shape=[1, 128, 28, 28], dtype=float32
- PT params: size=[56, 56], mode="bilinear", align_corners=True
- TF params: size=[56, 56], method="bilinear"
- Error: Shape mismatch: (1, 128, 56, 56) vs (1, 56, 56, 28)

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 128, 28, 28).astype(np.float32)

# PyTorch: NCHW 格式
out_pt = torch.nn.functional.interpolate(
    torch.tensor(input_np), size=(56, 56), mode='bilinear', align_corners=True
)

# TensorFlow: 需要先转 NHWC
input_nhwc = np.transpose(input_np, (0, 2, 3, 1))
out_tf_nhwc = tf.image.resize(tf.constant(input_nhwc), size=[56, 56], method='bilinear')
out_tf = tf.transpose(out_tf_nhwc, perm=[0, 3, 1, 2])

pt_np = out_pt.detach().numpy().astype(np.float64)
tf_np = out_tf.numpy().astype(np.float64)

if pt_np.shape != tf_np.shape:
    print(f"Shape mismatch: PyTorch {pt_np.shape} vs TensorFlow {tf_np.shape}")
else:
    max_diff = np.max(np.abs(pt_np - tf_np))
    print(f"Maximum difference: {max_diff}")
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.image.resize] Shape mismatch under equivalent migration in interpolate operator (sample5)

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

Same as Issue 081. When mapping `torch.nn.functional.interpolate` to `tf.image.resize` without NCHW→NHWC conversion, shape mismatch occurs: PyTorch (1, 128, 56, 56) vs TensorFlow (1, 56, 56, 28). Additionally, `align_corners=True` is not directly supported by `tf.image.resize`.

Expected behavior: Migration should transpose NCHW→NHWC before `tf.image.resize`, then NHWC→NCHW afterward.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 128, 28, 28).astype(np.float32)

out_pt = torch.nn.functional.interpolate(torch.tensor(input_np), size=(56, 56), mode='bilinear', align_corners=True)
out_tf_wrong = tf.image.resize(tf.constant(input_np), size=[56, 56], method='bilinear')
print(f"PyTorch: {out_pt.shape}, TF (wrong): {out_tf_wrong.shape}")
```

**Relevant log output**

```
comparison_error: Shape mismatch: PyTorch (1, 128, 56, 56) vs TensorFlow (1, 56, 56, 28)
```

## Issue 084

llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample6.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.functional.interpolate] Shape mismatch under equivalent migration in interpolate operator (sample6)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a **shape mismatch** was detected for `torch.nn.functional.interpolate`: PyTorch output (1, 128, 56, 56) vs TensorFlow output (1, 56, 56, 28).

Same root cause as Issue 081: **data format mismatch (NCHW vs NHWC)**. The input shape is [1, 128, 28, 28] and the migration code fails to transpose NCHW→NHWC before calling `tf.image.resize`.

- Input: shape=[1, 128, 28, 28], dtype=float32
- PT params: size=[56, 56], mode="bilinear", align_corners=True
- TF params: size=[56, 56], method="bilinear"
- Error: Shape mismatch: (1, 128, 56, 56) vs (1, 56, 56, 28)

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 128, 28, 28).astype(np.float32)

# PyTorch: NCHW 格式
out_pt = torch.nn.functional.interpolate(
    torch.tensor(input_np), size=(56, 56), mode='bilinear', align_corners=True
)

# TensorFlow: 需要先转 NHWC
input_nhwc = np.transpose(input_np, (0, 2, 3, 1))
out_tf_nhwc = tf.image.resize(tf.constant(input_nhwc), size=[56, 56], method='bilinear')
out_tf = tf.transpose(out_tf_nhwc, perm=[0, 3, 1, 2])

pt_np = out_pt.detach().numpy().astype(np.float64)
tf_np = out_tf.numpy().astype(np.float64)

if pt_np.shape != tf_np.shape:
    print(f"Shape mismatch: PyTorch {pt_np.shape} vs TensorFlow {tf_np.shape}")
else:
    max_diff = np.max(np.abs(pt_np - tf_np))
    print(f"Maximum difference: {max_diff}")
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.image.resize] Shape mismatch under equivalent migration in interpolate operator (sample6)

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

Same as Issue 081. When mapping `torch.nn.functional.interpolate` to `tf.image.resize` without NCHW→NHWC conversion, shape mismatch occurs: PyTorch (1, 128, 56, 56) vs TensorFlow (1, 56, 56, 28).

Expected behavior: Migration should transpose NCHW→NHWC before `tf.image.resize`, then NHWC→NCHW afterward.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 128, 28, 28).astype(np.float32)

out_pt = torch.nn.functional.interpolate(torch.tensor(input_np), size=(56, 56), mode='bilinear', align_corners=True)
out_tf_wrong = tf.image.resize(tf.constant(input_np), size=[56, 56], method='bilinear')
print(f"PyTorch: {out_pt.shape}, TF (wrong): {out_tf_wrong.shape}")
```

**Relevant log output**

```
comparison_error: Shape mismatch: PyTorch (1, 128, 56, 56) vs TensorFlow (1, 56, 56, 28)
```

## Issue 085

llm_enhanced_torch_nn_functional_leaky_relu_20251215_173249.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.functional.leaky_relu] Output difference anomaly under equivalent migration in leaky_relu operator (sample4)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for `torch.nn.functional.leaky_relu` (Maximum difference: 0.00042096355031003355).

The root cause is **input tensor data not synchronized**: The bug txt shows PyTorch input sample value is -1.3410084255671362 while TensorFlow input sample value is -1.2989120995690806. The input tensors fed to the two frameworks are numerically different. Since LeakyReLU is a deterministic element-wise operation, different inputs will produce different outputs.

- Input: shape=[1, 1, 1, 1], dtype=float64
- PT params: negative_slope=0.01, inplace=false
- TF params: alpha=0.01
- PT input sample: -1.3410084255671362, TF input sample: -1.2989120995690806
- Maximum difference: 0.00042096355031003355

```python
import numpy as np
import torch
import tensorflow as tf

# 使用统一的 numpy 随机种子确保输入一致
np.random.seed(42)
input_np = np.random.randn(1, 1, 1, 1).astype(np.float64)

# PyTorch
out_pt = torch.nn.functional.leaky_relu(torch.tensor(input_np), negative_slope=0.01)

# TensorFlow
out_tf = tf.nn.leaky_relu(tf.constant(input_np), alpha=0.01)

max_diff = np.max(np.abs(out_pt.detach().numpy() - out_tf.numpy()))
print(f"Input: {input_np.flatten()}")
print(f"PyTorch output: {out_pt.detach().numpy().flatten()}")
print(f"TensorFlow output: {out_tf.numpy().flatten()}")
print(f"Maximum difference: {max_diff}")
# 输入同步后，两框架的 LeakyReLU 输出应完全一致
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.nn.leaky_relu] Output difference anomaly under equivalent migration in leaky_relu operator (sample4)

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

When mapping `torch.nn.functional.leaky_relu(input, negative_slope=0.01)` to `tf.nn.leaky_relu(x, alpha=0.01)`, the outputs differ by 0.00042096355031003355. The primary cause is that the input tensors fed to each framework are numerically different (PT input: -1.341, TF input: -1.299), indicating the test harness did not properly synchronize inputs.

Expected behavior: With synchronized inputs, both frameworks' LeakyReLU should produce identical results since it is a simple deterministic element-wise operation.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(1, 1, 1, 1).astype(np.float64)

out_pt = torch.nn.functional.leaky_relu(torch.tensor(input_np), negative_slope=0.01)
out_tf = tf.nn.leaky_relu(tf.constant(input_np), alpha=0.01)

max_diff = np.max(np.abs(out_pt.detach().numpy() - out_tf.numpy()))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.00042096355031003355
```

## Issue 086

llm_enhanced_torch_nn_functional_local_response_norm_20251215_231102.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.functional.local_response_norm] Output difference anomaly under equivalent migration in local_response_norm operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for `torch.nn.functional.local_response_norm` (Maximum difference: 0.004879951477050781).

The root cause is **`size` vs `depth_radius` parameter semantic mismatch**:

- **PyTorch**: `size=2` means the normalization window covers `size` channels on each side of the current channel. The total window is `2 * floor(size/2) + 1 = 2 * 1 + 1 = 3` channels.
- **TensorFlow**: `depth_radius=2` means the window extends 2 channels in each direction. The total window is `2 * depth_radius + 1 = 5` channels.

So `size=2` in PyTorch corresponds to `depth_radius=1` in TensorFlow (both giving a 3-channel window), not `depth_radius=2` (which gives a 5-channel window).

- Input: shape=[32, 5, 24, 24], dtype=float32
- PT params: size=2, alpha=0.0001, beta=0.75, k=1.0
- TF params: depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0
- Maximum difference: 0.004879951477050781

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(32, 5, 24, 24).astype(np.float32)

# PyTorch: size=2, 窗口 = 2*floor(2/2)+1 = 3 channels
out_pt = torch.nn.functional.local_response_norm(
    torch.tensor(input_np), size=2, alpha=0.0001, beta=0.75, k=1.0
)

# TensorFlow 需要 NHWC 格式, depth_radius 应为 floor(size/2)=1
input_nhwc = np.transpose(input_np, (0, 2, 3, 1))
out_tf_nhwc = tf.nn.local_response_normalization(
    tf.constant(input_nhwc), depth_radius=1, alpha=0.0001, beta=0.75, bias=1.0
)
out_tf = tf.transpose(out_tf_nhwc, perm=[0, 3, 1, 2])

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# depth_radius=1 (而非2) 才能对齐 PyTorch 的 size=2
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.nn.local_response_normalization] Output difference anomaly under equivalent migration in local_response_norm operator (sample1)

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

When mapping `torch.nn.functional.local_response_norm(input, size=2, alpha=0.0001, beta=0.75, k=1.0)` to `tf.nn.local_response_normalization(input, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0)`, outputs differ by up to 0.004879951477050781.

PyTorch's `size=2` gives a window of 3 channels (`2*floor(2/2)+1`), while TensorFlow's `depth_radius=2` gives a window of 5 channels (`2*2+1`). The correct mapping should be `depth_radius=floor(size/2)=1`. Additionally, TensorFlow requires NHWC format input, not NCHW.

Expected behavior: Use `depth_radius=1` and convert NCHW→NHWC to align with PyTorch's `size=2` semantics.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(32, 5, 24, 24).astype(np.float32)

out_pt = torch.nn.functional.local_response_norm(torch.tensor(input_np), size=2, alpha=0.0001, beta=0.75, k=1.0)

# 错误做法：depth_radius=2（窗口5通道），应为 depth_radius=1（窗口3通道）
input_nhwc = np.transpose(input_np, (0, 2, 3, 1))
out_tf_wrong = tf.nn.local_response_normalization(tf.constant(input_nhwc), depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0)
out_tf_wrong_nchw = tf.transpose(out_tf_wrong, perm=[0, 3, 1, 2])

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf_wrong_nchw.numpy().astype(np.float64)))
print(f"Maximum difference (wrong depth_radius=2): {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.004879951477050781
```

## Issue 087

llm_enhanced_torch_nn_functional_local_response_norm_20251215_231102.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.functional.local_response_norm] Output difference anomaly under equivalent migration in local_response_norm operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for `torch.nn.functional.local_response_norm` (Maximum difference: 0.005212545394897461).

Same root cause as Issue 086: **`size` vs `depth_radius` parameter semantic mismatch**.

- **PyTorch**: `size=3` → window = `2 * floor(3/2) + 1 = 3` channels.
- **TensorFlow**: `depth_radius=3` → window = `2 * 3 + 1 = 7` channels.

The correct mapping: `depth_radius = floor(size/2) = 1`.

- Input: shape=[8, 3, 224, 224], dtype=float32
- PT params: size=3, alpha=0.0002, beta=0.7, k=2.0
- TF params: depth_radius=3, alpha=0.0002, beta=0.7, bias=2.0
- Maximum difference: 0.005212545394897461

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(8, 3, 224, 224).astype(np.float32)

# PyTorch: size=3, 窗口 = 2*floor(3/2)+1 = 3 channels
out_pt = torch.nn.functional.local_response_norm(
    torch.tensor(input_np), size=3, alpha=0.0002, beta=0.7, k=2.0
)

# TensorFlow: depth_radius 应为 floor(3/2)=1
input_nhwc = np.transpose(input_np, (0, 2, 3, 1))
out_tf_nhwc = tf.nn.local_response_normalization(
    tf.constant(input_nhwc), depth_radius=1, alpha=0.0002, beta=0.7, bias=2.0
)
out_tf = tf.transpose(out_tf_nhwc, perm=[0, 3, 1, 2])

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# depth_radius=1 (而非3) 才能对齐 PyTorch 的 size=3
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.nn.local_response_normalization] Output difference anomaly under equivalent migration in local_response_norm operator (sample3)

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

Same as Issue 086. PyTorch `size=3` gives a 3-channel window, while TensorFlow `depth_radius=3` gives a 7-channel window. The correct mapping is `depth_radius=floor(size/2)=1` (Maximum difference: 0.005212545394897461).

Expected behavior: Use `depth_radius=1` and NHWC conversion to match PyTorch's `size=3` semantics.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(8, 3, 224, 224).astype(np.float32)

out_pt = torch.nn.functional.local_response_norm(torch.tensor(input_np), size=3, alpha=0.0002, beta=0.7, k=2.0)

input_nhwc = np.transpose(input_np, (0, 2, 3, 1))
out_tf_wrong = tf.nn.local_response_normalization(tf.constant(input_nhwc), depth_radius=3, alpha=0.0002, beta=0.7, bias=2.0)
out_tf_wrong_nchw = tf.transpose(out_tf_wrong, perm=[0, 3, 1, 2])

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf_wrong_nchw.numpy().astype(np.float64)))
print(f"Maximum difference (wrong depth_radius=3): {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.005212545394897461
```

## Issue 088

llm_enhanced_torch_nn_functional_max_pool1d_20251202_130600.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.functional.max_pool1d] Output difference anomaly under equivalent migration in max_pool1d operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.nn.functional.max_pool1d` (Maximum difference: 1.762655258178711).

Max pooling is a deterministic operator — given identical inputs and parameters, the output should be identical. The large difference (1.76) strongly indicates that the **input tensors fed to the two frameworks are not synchronized** (different random data).

- Input: shape=[1, 1, 8], dtype=float32
- PT params: kernel_size=2
- PD params: kernel_size=2
- Maximum difference: 1.762655258178711

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(1, 1, 8).astype(np.float32)

# PyTorch
out_pt = torch.nn.functional.max_pool1d(torch.tensor(input_np), kernel_size=2)

# PaddlePaddle
out_pd = paddle.nn.functional.max_pool1d(paddle.to_tensor(input_np), kernel_size=2)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"PyTorch shape: {out_pt.shape}, Paddle shape: {out_pd.shape}")
print(f"Maximum difference: {max_diff}")
# 输入同步后，max_pool1d 为确定性算子，差异应为 0
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.functional.max_pool1d] Output difference anomaly under equivalent migration in max_pool1d operator (sample1)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`max_pool1d` 算子的最大输出差异为 1.762655258178711。

最大池化是确定性算子，在相同输入和参数下输出应完全一致。差异值高达 1.76，强烈表明**两端输入张量数据未同步**（使用了不同的随机数据）。

- 输入形状: [1, 1, 8], dtype=float32
- 参数: kernel_size=2

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(1, 1, 8).astype(np.float32)

# PyTorch
out_pt = torch.nn.functional.max_pool1d(torch.tensor(input_np), kernel_size=2)

# PaddlePaddle
out_pd = paddle.nn.functional.max_pool1d(paddle.to_tensor(input_np), kernel_size=2)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"PyTorch shape: {out_pt.shape}, Paddle shape: {out_pd.shape}")
print(f"Maximum difference: {max_diff}")
# 输入同步后，max_pool1d 为确定性算子，差异应为 0
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.functional.max_pool1d",
  "*args": [{"shape": [1, 1, 8], "dtype": "float32"}, 2],
  "**kwargs": {}
}
```

- max_pool1d 为确定性算子，怀疑两端输入数据未同步。

## Issue 089

llm_enhanced_torch_nn_functional_max_pool1d_20251202_130600.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.functional.max_pool1d] Output difference anomaly under equivalent migration in max_pool1d operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.nn.functional.max_pool1d` (Maximum difference: 4.071681976318359).

Max pooling is a deterministic operator. The large difference (4.07) strongly indicates the **input tensors are not synchronized**. Additionally, the `dilation=2` parameter may have different implementation behaviors across frameworks for 1D pooling.

- Input: shape=[2, 3, 10], dtype=float32
- PT params: kernel_size=3, stride=2, padding=1, dilation=2, return_indices=false, ceil_mode=true
- PD params: kernel_size=3, stride=2, padding=1, dilation=2, return_indices=false, ceil_mode=true
- Maximum difference: 4.071681976318359

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 10).astype(np.float32)

# PyTorch
out_pt = torch.nn.functional.max_pool1d(
    torch.tensor(input_np), kernel_size=3, stride=2, padding=1,
    dilation=2, return_indices=False, ceil_mode=True
)

# PaddlePaddle: 注意 Paddle 可能不支持 dilation 参数
try:
    out_pd = paddle.nn.functional.max_pool1d(
        paddle.to_tensor(input_np), kernel_size=3, stride=2, padding=1,
        ceil_mode=True, return_mask=False
    )
    max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
    print(f"Maximum difference: {max_diff}")
except Exception as e:
    print(f"Paddle error: {e}")
# dilation 参数在 Paddle max_pool1d 中可能不被支持
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.functional.max_pool1d] Output difference anomaly under equivalent migration in max_pool1d operator (sample2)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`max_pool1d` 算子的最大输出差异为 4.071681976318359。

最大池化是确定性算子，差异值 4.07 表明两端输入数据不同。此外，PyTorch 支持 `dilation=2` 参数，而 Paddle 的 `max_pool1d` 可能不支持 dilation 参数，导致窗口覆盖范围不同。

- 输入形状: [2, 3, 10], dtype=float32
- 参数: kernel_size=3, stride=2, padding=1, dilation=2, ceil_mode=true

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 10).astype(np.float32)

# PyTorch: 支持 dilation=2
out_pt = torch.nn.functional.max_pool1d(
    torch.tensor(input_np), kernel_size=3, stride=2, padding=1,
    dilation=2, return_indices=False, ceil_mode=True
)

# PaddlePaddle: Paddle 的 max_pool1d 可能不支持 dilation
try:
    out_pd = paddle.nn.functional.max_pool1d(
        paddle.to_tensor(input_np), kernel_size=3, stride=2, padding=1,
        ceil_mode=True, return_mask=False
    )
    max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
    print(f"Maximum difference: {max_diff}")
except Exception as e:
    print(f"Paddle error: {e}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.functional.max_pool1d",
  "*args": [{"shape": [2, 3, 10], "dtype": "float32"}, 3],
  "**kwargs": {"stride": 2, "padding": 1, "dilation": 2, "return_indices": false, "ceil_mode": true}
}
```

- Paddle 的 `max_pool1d` 可能不支持 `dilation` 参数。

## Issue 090

llm_enhanced_torch_nn_functional_max_pool1d_20251202_130600.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.functional.max_pool1d] Output difference anomaly under equivalent migration in max_pool1d operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.nn.functional.max_pool1d` (Maximum difference: 2.8710241317749023).

Same root cause as Issue 089: the `dilation=2` parameter may not be supported by Paddle's `max_pool1d`, and the input tensors may not be synchronized. Additionally, Paddle's `max_pool1d` does not support `return_indices` parameter (uses `return_mask` instead).

- Input: shape=[2, 3, 12], dtype=float32
- PT params: kernel_size=3, stride=2, padding=1, dilation=2, return_indices=false, ceil_mode=true
- PD params: kernel_size=3, stride=2, padding=1, dilation=2, return_indices=false, ceil_mode=true
- Maximum difference: 2.8710241317749023

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 12).astype(np.float32)

# PyTorch: 支持 dilation=2
out_pt = torch.nn.functional.max_pool1d(
    torch.tensor(input_np), kernel_size=3, stride=2, padding=1,
    dilation=2, return_indices=False, ceil_mode=True
)

# PaddlePaddle
try:
    out_pd = paddle.nn.functional.max_pool1d(
        paddle.to_tensor(input_np), kernel_size=3, stride=2, padding=1,
        ceil_mode=True, return_mask=False
    )
    max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
    print(f"Maximum difference: {max_diff}")
except Exception as e:
    print(f"Paddle error: {e}")
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.functional.max_pool1d] Output difference anomaly under equivalent migration in max_pool1d operator (sample3)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`max_pool1d` 算子的最大输出差异为 2.8710241317749023。

与 Issue 089 相同原因：PyTorch 支持 `dilation=2`，Paddle 的 `max_pool1d` 可能不支持 dilation，且 `return_indices` 参数在 Paddle 中对应 `return_mask`。

- 输入形状: [2, 3, 12], dtype=float32
- 参数: kernel_size=3, stride=2, padding=1, dilation=2, ceil_mode=true

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 12).astype(np.float32)

out_pt = torch.nn.functional.max_pool1d(
    torch.tensor(input_np), kernel_size=3, stride=2, padding=1,
    dilation=2, return_indices=False, ceil_mode=True
)

try:
    out_pd = paddle.nn.functional.max_pool1d(
        paddle.to_tensor(input_np), kernel_size=3, stride=2, padding=1,
        ceil_mode=True, return_mask=False
    )
    max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
    print(f"Maximum difference: {max_diff}")
except Exception as e:
    print(f"Paddle error: {e}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.functional.max_pool1d",
  "*args": [{"shape": [2, 3, 12], "dtype": "float32"}, 3],
  "**kwargs": {"stride": 2, "padding": 1, "dilation": 2, "return_indices": false, "ceil_mode": true}
}
```

- Paddle 不支持 `dilation` 和 `return_indices`（使用 `return_mask` 代替）。

## Issue 091

llm_enhanced_torch_nn_functional_rrelu_20251202_004720.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nn.functional.rrelu] Output difference anomaly under equivalent migration in rrelu operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, an output discrepancy was detected for `torch.nn.functional.rrelu` (Maximum difference: 0.015284627676010132).

The root cause is **different random number generators (RNG)**: `rrelu` in `training=True` mode randomly samples a slope from `Uniform(lower, upper)` for each negative element. PyTorch uses MT19937 RNG while Paddle uses a different RNG algorithm (Philox-based). Even with the same random seed, the sampled slopes differ between frameworks.

- Input: shape=[2], dtype=float32, sample_values=[0.295, -0.160]
- PT params: lower=0.1, upper=0.3, training=true
- PD params: lower=0.1, upper=0.3, training=true
- Maximum difference: 0.015284627676010132

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.array([0.29499974846839905, -0.16003058850765228], dtype=np.float32)

# PyTorch
torch.manual_seed(42)
out_pt = torch.nn.functional.rrelu(torch.tensor(input_np), lower=0.1, upper=0.3, training=True)

# PaddlePaddle
paddle.seed(42)
out_pd = paddle.nn.functional.rrelu(paddle.to_tensor(input_np), lower=0.1, upper=0.3, training=True)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"PyTorch output: {out_pt.detach().numpy()}")
print(f"Paddle output: {out_pd.numpy()}")
print(f"Maximum difference: {max_diff}")
# RNG 底层实现不同（MT19937 vs Philox），采样斜率值不同
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.functional.rrelu] Output difference anomaly under equivalent migration in rrelu operator (sample1)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`rrelu` 算子的最大输出差异为 0.015284627676010132。

根本原因是**随机数生成器（RNG）底层实现不同**：`rrelu` 在 `training=True` 时对每个负元素从 `Uniform(lower, upper)` 采样斜率。PyTorch 使用 MT19937，Paddle 使用不同的 RNG 算法。即使设置相同种子，采样的斜率值也不同。

- 输入: shape=[2], dtype=float32
- 参数: lower=0.1, upper=0.3, training=true

```python
import numpy as np
import torch
import paddle

input_np = np.array([0.29499974846839905, -0.16003058850765228], dtype=np.float32)

# PyTorch
torch.manual_seed(42)
out_pt = torch.nn.functional.rrelu(torch.tensor(input_np), lower=0.1, upper=0.3, training=True)

# PaddlePaddle
paddle.seed(42)
out_pd = paddle.nn.functional.rrelu(paddle.to_tensor(input_np), lower=0.1, upper=0.3, training=True)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# RNG 底层实现不同，采样斜率值不同
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.nn.functional.rrelu",
  "input": {"shape": [2], "dtype": "float32"},
  "lower": 0.1,
  "upper": 0.3,
  "training": true
}
```

- `rrelu` 的 training 模式依赖随机采样，RNG 底层不同导致数值差异。

## Issue 092

llm_enhanced_torch_nn_functional_softplus_20251216_002857.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.functional.softplus] Output difference anomaly under equivalent migration in softplus operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a significant output discrepancy was detected for `torch.nn.functional.softplus` (Maximum difference: 0.41588732733444334).

The root cause is **`beta` parameter not supported by TensorFlow**: PyTorch's `softplus` supports a `beta` parameter: $\text{softplus}(x) = \frac{1}{\beta} \cdot \log(1 + e^{\beta \cdot x})$. TensorFlow's `tf.nn.softplus` only supports the basic formula with $\beta = 1$: $\text{softplus}(x) = \log(1 + e^x)$. When `beta=2.5`, the difference is significant.

- Input: shape=[2, 3, 4, 5, 6], dtype=float64
- PT params: beta=2.5
- TF params: (no beta support, fixed at 1)
- Maximum difference: 0.41588732733444334

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 5, 6).astype(np.float64)

# PyTorch: beta=2.5
out_pt = torch.nn.functional.softplus(torch.tensor(input_np), beta=2.5)

# TensorFlow: 不支持 beta，固定为 1
out_tf = tf.nn.softplus(tf.constant(input_np))

# 手动实现 beta=2.5 的 softplus 以对齐 PyTorch
out_tf_aligned = (1.0 / 2.5) * tf.nn.softplus(tf.constant(input_np * 2.5))

max_diff_wrong = np.max(np.abs(out_pt.detach().numpy() - out_tf.numpy()))
max_diff_aligned = np.max(np.abs(out_pt.detach().numpy() - out_tf_aligned.numpy()))
print(f"Maximum difference (wrong, beta=1): {max_diff_wrong}")
print(f"Maximum difference (aligned, beta=2.5): {max_diff_aligned}")
# TF 不支持 beta 参数，需手动实现
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.nn.softplus] Output difference anomaly under equivalent migration in softplus operator (sample1)

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

When mapping `torch.nn.functional.softplus(input, beta=2.5)` to `tf.nn.softplus(input)`, the outputs differ by up to 0.41588732733444334. TensorFlow's `tf.nn.softplus` does not support the `beta` parameter and is fixed at `beta=1`.

PyTorch formula: $\frac{1}{\beta} \cdot \log(1 + e^{\beta \cdot x})$ with `beta=2.5`. TensorFlow formula: $\log(1 + e^x)$ (fixed `beta=1`).

Expected behavior: Migration should manually implement `(1/beta) * tf.nn.softplus(beta * x)` to match PyTorch's `beta` parameter.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 5, 6).astype(np.float64)

out_pt = torch.nn.functional.softplus(torch.tensor(input_np), beta=2.5)
out_tf = tf.nn.softplus(tf.constant(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy() - out_tf.numpy()))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.41588732733444334
```

## Issue 093

llm_enhanced_torch_nn_functional_softplus_20251216_002857.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][nn.functional.softplus] Output difference anomaly under equivalent migration in softplus operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a significant output discrepancy was detected for `torch.nn.functional.softplus` (Maximum difference: 0.3465735614299774).

Same root cause as Issue 092: **TensorFlow's `tf.nn.softplus` does not support `beta` or `threshold` parameters**. PyTorch uses `beta=2` and `threshold=20` (when `beta * x > threshold`, output switches to linear). TensorFlow has no such feature.

- Input: shape=[2, 3, 400], dtype=float32
- PT params: beta=2, threshold=20
- TF params: (no beta/threshold support)
- Maximum difference: 0.3465735614299774

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 400).astype(np.float32)

# PyTorch: beta=2, threshold=20
out_pt = torch.nn.functional.softplus(torch.tensor(input_np), beta=2, threshold=20)

# TensorFlow: 不支持 beta/threshold
out_tf = tf.nn.softplus(tf.constant(input_np))

# 手动实现对齐版本
beta = 2.0
threshold = 20.0
scaled = input_np * beta
out_tf_aligned = tf.where(
    tf.constant(scaled) > threshold,
    tf.constant(input_np),
    (1.0 / beta) * tf.nn.softplus(tf.constant(scaled))
)

max_diff_wrong = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
max_diff_aligned = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf_aligned.numpy().astype(np.float64)))
print(f"Maximum difference (wrong): {max_diff_wrong}")
print(f"Maximum difference (aligned): {max_diff_aligned}")
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.nn.softplus] Output difference anomaly under equivalent migration in softplus operator (sample2)

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

Same as Issue 092. PyTorch's `softplus(beta=2, threshold=20)` uses a generalized formula with linear fallback, while TensorFlow's `tf.nn.softplus` is fixed at `beta=1` with no threshold support (Maximum difference: 0.3465735614299774).

Expected behavior: Migration should manually implement `(1/beta) * tf.nn.softplus(beta * x)` with `tf.where` for the threshold mechanism.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 400).astype(np.float32)

out_pt = torch.nn.functional.softplus(torch.tensor(input_np), beta=2, threshold=20)
out_tf = tf.nn.softplus(tf.constant(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.3465735614299774
```

## Issue 094

llm_enhanced_torch_nonzero_20251216_010320.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][nonzero] Output format mismatch under equivalent migration in nonzero operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, a **shape mismatch** was detected for `torch.nonzero`: PyTorch output shape (1, 5) vs MindSpore output shape (5, 1).

The root cause is **`as_tuple=True` return format difference**:

- **PyTorch** `torch.nonzero(input, as_tuple=True)` returns a tuple of 1-D tensors, one for each dimension. For a 1-D input of shape [5] with all non-zero elements, it returns `(tensor([0, 1, 2, 3, 4]),)` — a tuple of one tensor with shape (5,).
- **MindSpore** `ops.NonZero` returns a 2-D tensor of shape (N, ndim) where N is the number of non-zero elements. For the same input, it returns a tensor of shape (5, 1).

The transposed shape relationship (1, 5) vs (5, 1) reflects this fundamental format difference.

- Input: shape=[5], dtype=int64, values=[-9, 9, -9, -10, -4]
- PT params: as_tuple=True
- MS API: mindspore.ops.NonZero
- Error: Shape mismatch: (1, 5) vs (5, 1)

```python
import numpy as np
import torch
import mindspore
from mindspore import ops

input_data = np.array([-9, 9, -9, -10, -4], dtype=np.int64)

# PyTorch: as_tuple=True 返回元组
out_pt = torch.nonzero(torch.tensor(input_data), as_tuple=True)
pt_stacked = torch.stack(out_pt)  # shape: (1, 5)

# MindSpore: 返回二维张量
out_ms = ops.NonZero()(mindspore.Tensor(input_data))  # shape: (5, 1)

print(f"PyTorch stacked shape: {pt_stacked.shape}")
print(f"MindSpore shape: {out_ms.shape}")
print(f"PyTorch values: {pt_stacked.numpy()}")
print(f"MindSpore values: {out_ms.asnumpy()}")
# as_tuple=True 返回的元组格式 vs 二维张量格式
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.ops.NonZero] Output format mismatch under equivalent migration in nonzero operator (sample1)

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

When mapping `torch.nonzero(input, as_tuple=True)` to `mindspore.ops.NonZero`, the output shapes differ: PyTorch returns (1, 5) (tuple of 1-D tensors stacked) while MindSpore returns (5, 1) (2-D tensor).

PyTorch's `as_tuple=True` returns a tuple of 1-D index tensors (one per dimension), while MindSpore's `ops.NonZero` returns a single 2-D tensor of shape (N, ndim).

**Describe the expected behavior***

Migration should handle the format conversion: either transpose MindSpore's output or convert PyTorch's tuple format to match. The underlying index values are equivalent, just in different shapes/formats.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore
from mindspore import ops

input_data = np.array([-9, 9, -9, -10, -4], dtype=np.int64)

out_pt = torch.nonzero(torch.tensor(input_data), as_tuple=True)
pt_stacked = torch.stack(out_pt)

out_ms = ops.NonZero()(mindspore.Tensor(input_data))

print(f"PyTorch stacked shape: {pt_stacked.shape}")  # (1, 5)
print(f"MindSpore shape: {out_ms.shape}")  # (5, 1)
```

**Related log / screenshot**

```
comparison_error: Shape mismatch: PyTorch (1, 5) vs MindSpore (5, 1)
```

**Special notes for this issue**

`as_tuple=True` 的元组格式与 MindSpore 的二维张量格式存在转置关系，数据内容等价。

## Issue 095

llm_enhanced_torch_nonzero_20251216_010320.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][nonzero] Output format mismatch under equivalent migration in nonzero operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, a **shape mismatch** was detected for `torch.nonzero`: PyTorch output shape (1, 4) vs MindSpore output shape (4, 1).

Same root cause as Issue 094: PyTorch `as_tuple=True` returns a tuple of 1-D tensors, MindSpore `mint.nonzero` returns a 2-D tensor. The input [2, -6, -8, 2, 0] has 4 non-zero elements.

- Input: shape=[5], dtype=int64, values=[2, -6, -8, 2, 0]
- PT params: as_tuple=True
- MS API: mindspore.mint.nonzero
- Error: Shape mismatch: (1, 4) vs (4, 1)

```python
import numpy as np
import torch
import mindspore

input_data = np.array([2, -6, -8, 2, 0], dtype=np.int64)

# PyTorch: as_tuple=True 返回元组
out_pt = torch.nonzero(torch.tensor(input_data), as_tuple=True)
pt_stacked = torch.stack(out_pt)  # shape: (1, 4)

# MindSpore: 返回二维张量
out_ms = mindspore.mint.nonzero(mindspore.Tensor(input_data))  # shape: (4, 1)

print(f"PyTorch stacked shape: {pt_stacked.shape}")
print(f"MindSpore shape: {out_ms.shape}")
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.mint.nonzero] Output format mismatch under equivalent migration in nonzero operator (sample2)

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

Same as Issue 094. PyTorch `nonzero(as_tuple=True)` returns (1, 4), MindSpore `mint.nonzero` returns (4, 1). Input [2, -6, -8, 2, 0] has 4 non-zero elements.

**Describe the expected behavior***

Migration should handle the tuple-to-tensor format conversion or transpose.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

input_data = np.array([2, -6, -8, 2, 0], dtype=np.int64)

out_pt = torch.nonzero(torch.tensor(input_data), as_tuple=True)
pt_stacked = torch.stack(out_pt)

out_ms = mindspore.mint.nonzero(mindspore.Tensor(input_data))

print(f"PyTorch stacked shape: {pt_stacked.shape}")  # (1, 4)
print(f"MindSpore shape: {out_ms.shape}")  # (4, 1)
```

**Related log / screenshot**

```
comparison_error: Shape mismatch: PyTorch (1, 4) vs MindSpore (4, 1)
```

**Special notes for this issue**

`as_tuple=True` 的元组格式与 MindSpore 的二维张量格式存在转置关系。

## Issue 096

llm_enhanced_torch_nonzero_20251216_010320.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][nonzero] Output format mismatch under equivalent migration in nonzero operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, a **shape mismatch** was detected for `torch.nonzero`: PyTorch output shape (3, 0) vs MindSpore output shape (0, 3).

The input is a 3-D tensor of shape [2, 3, 0] with zero elements (empty tensor). With `as_tuple=True`, PyTorch returns a tuple of 3 empty tensors (one per dimension, each of shape (0,)), which when stacked gives shape (3, 0). MindSpore returns an empty 2-D tensor of shape (0, 3).

- Input: shape=[2, 3, 0], dtype=int64 (empty tensor)
- PT params: as_tuple=True
- MS API: mindspore.mint.nonzero
- Error: Shape mismatch: (3, 0) vs (0, 3)

```python
import numpy as np
import torch
import mindspore

input_data = np.zeros((2, 3, 0), dtype=np.int64)

# PyTorch: as_tuple=True，返回3个空张量的元组
out_pt = torch.nonzero(torch.tensor(input_data), as_tuple=True)
pt_stacked = torch.stack(out_pt)  # shape: (3, 0)

# MindSpore: 返回空的二维张量
out_ms = mindspore.mint.nonzero(mindspore.Tensor(input_data))  # shape: (0, 3)

print(f"PyTorch stacked shape: {pt_stacked.shape}")
print(f"MindSpore shape: {out_ms.shape}")
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.mint.nonzero] Output format mismatch under equivalent migration in nonzero operator (sample3)

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

For an empty 3-D input tensor of shape [2, 3, 0], PyTorch `nonzero(as_tuple=True)` returns 3 empty tensors (stacked: (3, 0)), while MindSpore `mint.nonzero` returns an empty 2-D tensor of shape (0, 3).

**Describe the expected behavior***

Both represent the same semantic: zero non-zero elements in a 3-D tensor. The formats are transposed. Migration should handle the conversion accordingly.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

input_data = np.zeros((2, 3, 0), dtype=np.int64)

out_pt = torch.nonzero(torch.tensor(input_data), as_tuple=True)
pt_stacked = torch.stack(out_pt)

out_ms = mindspore.mint.nonzero(mindspore.Tensor(input_data))

print(f"PyTorch stacked shape: {pt_stacked.shape}")  # (3, 0)
print(f"MindSpore shape: {out_ms.shape}")  # (0, 3)
```

**Related log / screenshot**

```
comparison_error: Shape mismatch: PyTorch (3, 0) vs MindSpore (0, 3)
```

**Special notes for this issue**

空张量的 `nonzero` 结果，两种形状 (3,0) 和 (0,3) 在语义上等价。

## Issue 097

llm_enhanced_torch_nonzero_20251216_010320.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][nonzero] Output format mismatch under equivalent migration in nonzero operator (sample4)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, a **shape mismatch** was detected for `torch.nonzero`: PyTorch output shape (1, 5) vs MindSpore output shape (5, 1).

Same root cause as Issue 094: PyTorch `as_tuple=True` returns tuple format, MindSpore `ops.NonZero` returns 2-D tensor format. Input [8, -3, 2, 8, 5] has 5 non-zero elements.

- Input: shape=[5], dtype=int64, values=[8, -3, 2, 8, 5]
- PT params: as_tuple=True
- MS API: mindspore.ops.NonZero
- Error: Shape mismatch: (1, 5) vs (5, 1)

```python
import numpy as np
import torch
import mindspore
from mindspore import ops

input_data = np.array([8, -3, 2, 8, 5], dtype=np.int64)

# PyTorch: as_tuple=True
out_pt = torch.nonzero(torch.tensor(input_data), as_tuple=True)
pt_stacked = torch.stack(out_pt)  # shape: (1, 5)

# MindSpore
out_ms = ops.NonZero()(mindspore.Tensor(input_data))  # shape: (5, 1)

print(f"PyTorch stacked shape: {pt_stacked.shape}")
print(f"MindSpore shape: {out_ms.shape}")
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.ops.NonZero] Output format mismatch under equivalent migration in nonzero operator (sample4)

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

Same as Issue 094. PyTorch `nonzero(as_tuple=True)` stacked shape (1, 5), MindSpore `ops.NonZero` shape (5, 1). Input [8, -3, 2, 8, 5] has 5 non-zero elements.

**Describe the expected behavior***

Migration should handle tuple-to-tensor format conversion or transpose.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore
from mindspore import ops

input_data = np.array([8, -3, 2, 8, 5], dtype=np.int64)

out_pt = torch.nonzero(torch.tensor(input_data), as_tuple=True)
pt_stacked = torch.stack(out_pt)

out_ms = ops.NonZero()(mindspore.Tensor(input_data))

print(f"PyTorch stacked shape: {pt_stacked.shape}")  # (1, 5)
print(f"MindSpore shape: {out_ms.shape}")  # (5, 1)
```

**Related log / screenshot**

```
comparison_error: Shape mismatch: PyTorch (1, 5) vs MindSpore (5, 1)
```

**Special notes for this issue**

元组格式与二维张量格式的转置关系。

## Issue 098

llm_enhanced_torch_nonzero_20251216_010320.json_sample5.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][nonzero] Output format mismatch under equivalent migration in nonzero operator (sample5)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, a **shape mismatch** was detected for `torch.nonzero`: PyTorch output shape (2, 5) vs MindSpore output shape (5, 2).

The input is a 2-D tensor of shape [2, 3] with values [-5, 0, -6, 2, 5, 8]. It has 5 non-zero elements. With `as_tuple=True`, PyTorch returns a tuple of 2 tensors (one per dimension), each of shape (5,), which when stacked gives (2, 5). MindSpore returns a 2-D tensor of shape (5, 2).

- Input: shape=[2, 3], dtype=int64, values=[-5, 0, -6, 2, 5, 8]
- PT params: as_tuple=True
- MS API: mindspore.mint.nonzero
- Error: Shape mismatch: (2, 5) vs (5, 2)

```python
import numpy as np
import torch
import mindspore

input_data = np.array([[-5, 0, -6], [2, 5, 8]], dtype=np.int64)

# PyTorch: as_tuple=True 返回2个张量的元组（每维一个）
out_pt = torch.nonzero(torch.tensor(input_data), as_tuple=True)
pt_stacked = torch.stack(out_pt)  # shape: (2, 5)

# MindSpore
out_ms = mindspore.mint.nonzero(mindspore.Tensor(input_data))  # shape: (5, 2)

print(f"PyTorch stacked shape: {pt_stacked.shape}")
print(f"MindSpore shape: {out_ms.shape}")
print(f"PyTorch values:\n{pt_stacked.numpy()}")
print(f"MindSpore values:\n{out_ms.asnumpy()}")
# 转置关系：pt_stacked.T == out_ms
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.mint.nonzero] Output format mismatch under equivalent migration in nonzero operator (sample5)

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

For a 2-D input tensor of shape [2, 3] with 5 non-zero elements, PyTorch `nonzero(as_tuple=True)` stacked shape is (2, 5), while MindSpore `mint.nonzero` returns (5, 2). This is a transposed relationship.

**Describe the expected behavior***

Both contain the same index data. The shapes are transposed. Migration should handle this format conversion, e.g., `ms_output.T` should equal the stacked PyTorch tuple output.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

input_data = np.array([[-5, 0, -6], [2, 5, 8]], dtype=np.int64)

out_pt = torch.nonzero(torch.tensor(input_data), as_tuple=True)
pt_stacked = torch.stack(out_pt)

out_ms = mindspore.mint.nonzero(mindspore.Tensor(input_data))

print(f"PyTorch stacked shape: {pt_stacked.shape}")  # (2, 5)
print(f"MindSpore shape: {out_ms.shape}")  # (5, 2)
```

**Related log / screenshot**

```
comparison_error: Shape mismatch: PyTorch (2, 5) vs MindSpore (5, 2)
```

**Special notes for this issue**

二维输入的 `nonzero` 结果，元组格式 stacked 后与二维张量格式呈转置关系。

## Issue 099

llm_enhanced_torch_pow_20251215_204420.json_sample6.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][pow] Output difference anomaly with NaN under equivalent migration in pow operator (sample6)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a **NaN-based output discrepancy** was detected for `torch.pow` (Maximum difference: nan).

Two distinct problems:

1. **Input data not synchronized**: The bug txt shows PyTorch input sample values [2.660, 0.072, -1.629, ...] while TensorFlow input sample values [0.448, 0.876, 0.637, ...] are completely different. The inputs were not aligned.
2. **NaN from negative base with fractional exponent**: `pow(x, -0.5)` computes $x^{-0.5} = \frac{1}{\sqrt{x}}$. When $x < 0$, the result is NaN in real arithmetic. Both frameworks have negative values in input, inevitably producing NaN.

- Input: shape=[20], dtype=float32
- Exponent: -0.5
- PT input samples: [2.660, 0.072, -1.629, ...]
- TF input samples: [0.448, 0.876, 0.637, ...]
- Maximum difference: nan

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(20).astype(np.float32)

# PyTorch
out_pt = torch.pow(torch.tensor(input_np), -0.5)

# TensorFlow
out_tf = tf.pow(tf.constant(input_np), -0.5)

pt_np = out_pt.detach().numpy().astype(np.float64)
tf_np = out_tf.numpy().astype(np.float64)

# 过滤 NaN 后比较
valid_mask = ~(np.isnan(pt_np) | np.isnan(tf_np))
if valid_mask.any():
    max_diff = np.max(np.abs(pt_np[valid_mask] - tf_np[valid_mask]))
    print(f"Max diff (excluding NaN): {max_diff}")
else:
    print("All values are NaN")
print(f"NaN count PT: {np.isnan(pt_np).sum()}, TF: {np.isnan(tf_np).sum()}")
# 负数的 -0.5 次方产生 NaN
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.pow] Output difference anomaly with NaN under equivalent migration in pow operator (sample6)

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

When mapping `torch.pow(input, -0.5)` to `tf.pow(x, -0.5)`, the maximum difference is NaN. Two problems: (1) Input data was not synchronized between frameworks (PT samples: [2.660, 0.072, -1.629, ...] vs TF samples: [0.448, 0.876, 0.637, ...]). (2) Negative values raised to fractional power -0.5 produce NaN.

Expected behavior: With synchronized inputs, both frameworks should produce identical results (including matching NaN positions for negative inputs).

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(20).astype(np.float32)

out_pt = torch.pow(torch.tensor(input_np), -0.5)
out_tf = tf.pow(tf.constant(input_np), -0.5)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: nan
```

## Issue 100

llm_enhanced_torch_pow_20251215_204420.json_sample7.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][pow] Output difference anomaly with NaN under equivalent migration in pow operator (sample7)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a **NaN-based output discrepancy** was detected for `torch.pow` (Maximum difference: nan).

Same root cause as Issue 099: **input data not synchronized** and **negative base with fractional exponent produces NaN**. PyTorch input samples: [0.947, -1.568, -0.790, ...], TF input samples: [0.316, -0.352, -0.273, ...] — completely different values.

- Input: shape=[20], dtype=float32
- Exponent: -0.5
- Maximum difference: nan

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(20).astype(np.float32)

# PyTorch
out_pt = torch.pow(torch.tensor(input_np), -0.5)

# TensorFlow
out_tf = tf.pow(tf.constant(input_np), -0.5)

pt_np = out_pt.detach().numpy().astype(np.float64)
tf_np = out_tf.numpy().astype(np.float64)

valid_mask = ~(np.isnan(pt_np) | np.isnan(tf_np))
if valid_mask.any():
    max_diff = np.max(np.abs(pt_np[valid_mask] - tf_np[valid_mask]))
    print(f"Max diff (excluding NaN): {max_diff}")
print(f"NaN count PT: {np.isnan(pt_np).sum()}, TF: {np.isnan(tf_np).sum()}")
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.pow] Output difference anomaly with NaN under equivalent migration in pow operator (sample7)

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

Same as Issue 099. Input data not synchronized between frameworks, and negative values raised to -0.5 produce NaN (Maximum difference: nan).

Expected behavior: With synchronized inputs, both frameworks should produce identical results.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(20).astype(np.float32)

out_pt = torch.pow(torch.tensor(input_np), -0.5)
out_tf = tf.pow(tf.constant(input_np), -0.5)

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_tf.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: nan
```

## Issue 101

llm_enhanced_torch_quantile_20251202_123958.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][quantile] Shape mismatch under equivalent migration in quantile operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a **shape mismatch** was detected for `torch.quantile`: PyTorch output shape (1, 5, 5) vs PaddlePaddle output shape (1, 1, 1).

The root cause is **`dim` parameter not mapped**: PyTorch `torch.quantile(input, q=0.5, dim=0, keepdim=True)` computes the quantile along dimension 0 of a (5,5,5) tensor, producing output shape (1,5,5). Paddle's `paddle.quantile(input, q=0.5, keepdim=True)` computes the **global** quantile (no `dim` specified), producing output shape (1,1,1).

The migration code must explicitly pass `axis=0` to `paddle.quantile`.

- Input: shape=[5, 5, 5], dtype=float64
- PT params: q=0.5, dim=0, keepdim=True → output (1, 5, 5)
- PD params: q=0.5, keepdim=True (no axis) → output (1, 1, 1)
- Error: Shape mismatch: (1, 5, 5) vs (1, 1, 1)

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(5, 5, 5).astype(np.float64)

# PyTorch: dim=0
out_pt = torch.quantile(torch.tensor(input_np), q=0.5, dim=0, keepdim=True)

# PaddlePaddle: 未指定 axis，默认全局计算
out_pd_wrong = paddle.quantile(paddle.to_tensor(input_np), q=0.5, keepdim=True)

# 正确做法：显式指定 axis=0
out_pd_correct = paddle.quantile(paddle.to_tensor(input_np), q=0.5, axis=0, keepdim=True)

print(f"PyTorch shape: {out_pt.shape}")           # (1, 5, 5)
print(f"Paddle (wrong) shape: {out_pd_wrong.shape}")  # (1, 1, 1)
print(f"Paddle (correct) shape: {out_pd_correct.shape}")  # (1, 5, 5)

max_diff = np.max(np.abs(out_pt.detach().numpy() - out_pd_correct.numpy()))
print(f"Maximum difference (correct): {max_diff}")
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][quantile] Shape mismatch under equivalent migration in quantile operator (sample3)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`quantile` 算子出现形状不匹配：PyTorch 输出 (1, 5, 5)，Paddle 输出 (1, 1, 1)。

根本原因是 **`dim` 参数未正确映射**：PyTorch 使用 `dim=0` 沿第0维计算分位数，输出 (1,5,5)。Paddle 的迁移代码未指定 `axis` 参数，默认对全部元素计算全局分位数，输出 (1,1,1)。

迁移代码应显式传递 `axis=0` 给 `paddle.quantile`。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(5, 5, 5).astype(np.float64)

# PyTorch: dim=0
out_pt = torch.quantile(torch.tensor(input_np), q=0.5, dim=0, keepdim=True)

# Paddle: 未指定 axis（错误）
out_pd_wrong = paddle.quantile(paddle.to_tensor(input_np), q=0.5, keepdim=True)

# Paddle: 显式 axis=0（正确）
out_pd_correct = paddle.quantile(paddle.to_tensor(input_np), q=0.5, axis=0, keepdim=True)

print(f"PyTorch shape: {out_pt.shape}")
print(f"Paddle (wrong) shape: {out_pd_wrong.shape}")
print(f"Paddle (correct) shape: {out_pd_correct.shape}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.quantile",
  "input": {"shape": [5, 5, 5], "dtype": "float64"},
  "q": 0.5,
  "keepdim": true
}
```

- Paddle 的 `axis` 参数等价于 PyTorch 的 `dim` 参数，迁移代码中缺失了此映射。

## Issue 102

llm_enhanced_torch_rand_20251215_231651.json_sample6.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][rand] Output difference anomaly under equivalent migration in rand operator (sample6)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a significant output discrepancy was detected for `torch.rand` (Maximum difference: 0.9108648790938367).

The root cause is **fundamentally different PRNG implementations**: PyTorch uses MT19937 (Mersenne Twister) while TensorFlow uses Philox as the default RNG. Even with the same seed (1234), the generated uniform random sequences are completely different. This is an inherent framework-level difference that cannot be resolved through parameter alignment.

- Shape: [2, 3, 4, 5], dtype=float64
- PT seed: 1234, TF seed: 1234
- Maximum difference: 0.9108648790938367

```python
import numpy as np
import torch
import tensorflow as tf

# PyTorch: MT19937
torch.manual_seed(1234)
out_pt = torch.rand(2, 3, 4, 5, dtype=torch.float64)

# TensorFlow: Philox
tf.random.set_seed(1234)
out_tf = tf.random.uniform(shape=[2, 3, 4, 5], minval=0.0, maxval=1.0, dtype=tf.float64)

max_diff = np.max(np.abs(out_pt.numpy() - out_tf.numpy()))
print(f"Maximum difference: {max_diff}")
# PRNG 算法不同（MT19937 vs Philox），即使种子相同也无法对齐
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.random.uniform] Output difference anomaly under equivalent migration in rand operator (sample6)

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

When mapping `torch.rand(size, seed=1234)` to `tf.random.uniform(shape, seed=1234)`, the outputs differ by up to 0.9108648790938367. PyTorch uses MT19937 RNG while TensorFlow uses Philox RNG. Even with identical seeds, the generated sequences are fundamentally different.

Expected behavior: This is an inherent framework-level difference. Random number generation operators should be excluded from numerical comparison in cross-framework migration testing, or a shared numpy-generated random array should be used as input instead.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

torch.manual_seed(1234)
out_pt = torch.rand(2, 3, 4, 5, dtype=torch.float64)

tf.random.set_seed(1234)
out_tf = tf.random.uniform(shape=[2, 3, 4, 5], minval=0.0, maxval=1.0, dtype=tf.float64)

max_diff = np.max(np.abs(out_pt.numpy() - out_tf.numpy()))
print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.9108648790938367
```

## Issue 103

llm_enhanced_torch_rand_like_20251125_143509.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][rand_like] Output difference anomaly under equivalent migration in rand_like operator (sample4)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, an output discrepancy was detected for `torch.rand_like` (Maximum difference: 0.5080589814591261).

The root cause is **RNG implementation difference**: `rand_like` generates uniform random numbers in [0, 1). Both frameworks use different PRNG algorithms internally, so even with the same seed, the generated values differ. The input is a scalar tensor (shape=[], dtype=float64).

- Input: shape=[], dtype=float64, value=1.0
- Maximum difference: 0.5080589814591261

```python
import numpy as np
import torch
import paddle

input_np = np.array(1.0, dtype=np.float64)

# PyTorch
torch.manual_seed(42)
out_pt = torch.rand_like(torch.tensor(input_np))

# PaddlePaddle
paddle.seed(42)
out_pd = paddle.rand_like(paddle.to_tensor(input_np))

max_diff = np.abs(out_pt.numpy() - out_pd.numpy())
print(f"PyTorch: {out_pt.numpy()}, Paddle: {out_pd.numpy()}")
print(f"Maximum difference: {max_diff}")
# RNG 实现不同，随机数值无法对齐
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][rand_like] Output difference anomaly under equivalent migration in rand_like operator (sample4)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`rand_like` 算子的最大输出差异为 0.5080589814591261。

根本原因是**随机数生成器（RNG）底层实现不同**：`rand_like` 生成 [0, 1) 均匀分布随机数。两个框架使用不同的 PRNG 算法，即使使用相同种子也无法产生一致的随机序列。

- 输入: 标量张量, shape=[], dtype=float64

```python
import numpy as np
import torch
import paddle

input_np = np.array(1.0, dtype=np.float64)

torch.manual_seed(42)
out_pt = torch.rand_like(torch.tensor(input_np))

paddle.seed(42)
out_pd = paddle.rand_like(paddle.to_tensor(input_np))

max_diff = np.abs(out_pt.numpy() - out_pd.numpy())
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "torch.rand_like",
  "input": {"shape": [], "dtype": "float64", "sample_values": [1.0]}
}
```

- 随机数生成算子，RNG 实现差异导致数值不可对齐。

## Issue 104

llm_enhanced_torch_rand_like_20251125_143509.json_sample5.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][rand_like] Output difference anomaly under equivalent migration in rand_like operator (sample5)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, an output discrepancy was detected for `torch.rand_like` (Maximum difference: 0.6013766649573999).

Same root cause as Issue 103: **RNG algorithm difference** (MT19937 vs Philox). The difference magnitude (~0.6) is expected for U[0,1) random outputs.

- Input: shape=[2, 3, 4], dtype=float64
- Maximum difference: 0.6013766649573999

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 4).astype(np.float64)

# PyTorch
torch.manual_seed(42)
out_pt = torch.rand_like(torch.tensor(input_np))

# PaddlePaddle
paddle.seed(42)
out_pd = paddle.rand_like(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.numpy() - out_pd.numpy()))
print(f"Maximum difference: {max_diff}")
# MT19937 vs Philox, 随机数无法对齐
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][rand_like] Output difference anomaly under equivalent migration in rand_like operator (sample5)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`rand_like` 算子的最大输出差异为 0.6013766649573999。

与 Issue 103 相同原因：底层 RNG 算法不同（MT19937 vs Philox），即使种子相同也无法产生一致的随机序列。

- 输入形状: [2, 3, 4], dtype=float64

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 4).astype(np.float64)

torch.manual_seed(42)
out_pt = torch.rand_like(torch.tensor(input_np))

paddle.seed(42)
out_pd = paddle.rand_like(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.numpy() - out_pd.numpy()))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.rand_like",
  "input": {"shape": [2, 3, 4], "dtype": "float64"}
}
```

- RNG 底层实现差异（MT19937 vs Philox）。

## Issue 105

llm_enhanced_torch_randn_20251202_005904.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][randn] Output difference anomaly under equivalent migration in randn operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.randn` (Maximum difference: 5.521675109863281).

The root cause is **RNG algorithm fundamentally different**: `torch.randn` generates standard normal N(0,1) random tensors using MT19937 + Box-Muller transform, while `paddle.randn` uses a different RNG implementation. Even with the same seed, the generated normal sequences differ entirely.

- Shape: [1, 512, 4, 4]
- Maximum difference: 5.521675109863281

```python
import numpy as np
import torch
import paddle

# PyTorch
torch.manual_seed(42)
out_pt = torch.randn(1, 512, 4, 4)

# PaddlePaddle
paddle.seed(42)
out_pd = paddle.randn([1, 512, 4, 4])

max_diff = np.max(np.abs(out_pt.numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# RNG 算法不同，正态分布随机数无法对齐
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][randn] Output difference anomaly under equivalent migration in randn operator (sample3)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`randn` 算子的最大输出差异为 5.521675109863281。

根本原因是**随机数生成器（RNG）算法不同**：`torch.randn` 使用 MT19937 + Box-Muller 变换生成标准正态分布 N(0,1) 随机张量，`paddle.randn` 使用不同的 RNG 实现。即使种子相同，生成的正态分布序列也完全不同。

- 形状: [1, 512, 4, 4]

```python
import numpy as np
import torch
import paddle

torch.manual_seed(42)
out_pt = torch.randn(1, 512, 4, 4)

paddle.seed(42)
out_pd = paddle.randn([1, 512, 4, 4])

max_diff = np.max(np.abs(out_pt.numpy().astype(np.float64) - out_pd.numpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "torch.randn",
  "*size": [1, 512, 4, 4],
  "requires_grad": true
}
```

- 随机数生成算子，RNG 底层实现差异导致数值不可对齐。

## Issue 106

llm_enhanced_torch_randn_20251202_005904.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][randn] Output difference anomaly under equivalent migration in randn operator (sample4)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a significant output discrepancy was detected for `torch.randn` (Maximum difference: 2.5926952362060547).

Same root cause as Issue 105: **RNG algorithm difference** (MT19937 vs Philox-based). This sample uses float64 dtype.

- Shape: [2, 1, 4, 1, 4], dtype=float64
- Maximum difference: 2.5926952362060547

```python
import numpy as np
import torch
import paddle

# PyTorch
torch.manual_seed(42)
out_pt = torch.randn(2, 1, 4, 1, 4, dtype=torch.float64)

# PaddlePaddle
paddle.seed(42)
out_pd = paddle.randn([2, 1, 4, 1, 4], dtype='float64')

max_diff = np.max(np.abs(out_pt.numpy() - out_pd.numpy()))
print(f"Maximum difference: {max_diff}")
# RNG 算法不同，即使种子同步也无法一致
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][randn] Output difference anomaly under equivalent migration in randn operator (sample4)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`randn` 算子的最大输出差异为 2.5926952362060547。

与 Issue 105 相同原因：RNG 算法不同。此样例使用 float64 数据类型。

- 形状: [2, 1, 4, 1, 4], dtype=float64

```python
import numpy as np
import torch
import paddle

torch.manual_seed(42)
out_pt = torch.randn(2, 1, 4, 1, 4, dtype=torch.float64)

paddle.seed(42)
out_pd = paddle.randn([2, 1, 4, 1, 4], dtype='float64')

max_diff = np.max(np.abs(out_pt.numpy() - out_pd.numpy()))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.randn",
  "*size": [2, 1, 4, 1, 4],
  "dtype": "float64"
}
```

- RNG 底层实现差异导致数值不一致。

## Issue 107

llm_enhanced_torch_randn_like_20251125_141142.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][randn_like] Output difference anomaly under equivalent migration in randn_like operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a significant output discrepancy was detected for `torch.randn_like` (Maximum difference: 2.0844786981984345).

The root cause is **RNG implementation difference**: `randn_like` generates N(0,1) random tensors with the same shape and dtype as the input. Different frameworks use different PRNG algorithms internally, producing different random sequences even with the same seed.

- Input: shape=[1, 2, 3], dtype=float64
- Maximum difference: 2.0844786981984345

```python
import numpy as np
import torch
import paddle

input_np = np.array([[[-0.04563242953276263, 1.087673202944817, 0.45814348682490913],
                       [-1.0433938175339883, 0.4971076744967649, -0.04430917673972046]]], dtype=np.float64)

# PyTorch
torch.manual_seed(42)
out_pt = torch.randn_like(torch.tensor(input_np))

# PaddlePaddle
paddle.seed(42)
out_pd = paddle.randn_like(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.numpy() - out_pd.numpy()))
print(f"Maximum difference: {max_diff}")
# RNG 实现不同，随机数值无法对齐
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][randn_like] Output difference anomaly under equivalent migration in randn_like operator (sample2)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`randn_like` 算子的最大输出差异为 2.0844786981984345。

根本原因是**随机数生成器（RNG）底层实现不同**：`randn_like` 生成与输入相同 shape 和 dtype 的 N(0,1) 随机张量。不同框架使用不同的 PRNG 算法。

- 输入形状: [1, 2, 3], dtype=float64

```python
import numpy as np
import torch
import paddle

input_np = np.array([[[-0.04563242953276263, 1.087673202944817, 0.45814348682490913],
                       [-1.0433938175339883, 0.4971076744967649, -0.04430917673972046]]], dtype=np.float64)

torch.manual_seed(42)
out_pt = torch.randn_like(torch.tensor(input_np))

paddle.seed(42)
out_pd = paddle.randn_like(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.numpy() - out_pd.numpy()))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.randn_like",
  "input": {"shape": [1, 2, 3], "dtype": "float64"}
}
```

- RNG 底层实现差异导致数值不可对齐。

## Issue 108

llm_enhanced_torch_randn_like_20251125_141142.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][randn_like] Output difference anomaly under equivalent migration in randn_like operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a significant output discrepancy was detected for `torch.randn_like` (Maximum difference: 2.9182678404442495).

Same root cause as Issue 107: **RNG implementation difference** (MT19937 + Box-Muller vs Philox-based). This sample has a 1-D input of shape [25].

- Input: shape=[25], dtype=float64
- Maximum difference: 2.9182678404442495

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(25).astype(np.float64)

# PyTorch
torch.manual_seed(42)
out_pt = torch.randn_like(torch.tensor(input_np))

# PaddlePaddle
paddle.seed(42)
out_pd = paddle.randn_like(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.numpy() - out_pd.numpy()))
print(f"Maximum difference: {max_diff}")
# RNG 底层算法不同（Philox+Box-Muller等）
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][randn_like] Output difference anomaly under equivalent migration in randn_like operator (sample3)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`randn_like` 算子的最大输出差异为 2.9182678404442495。

与 Issue 107 相同原因：RNG 底层算法不同（MT19937 + Box-Muller vs Philox 等）。

- 输入形状: [25], dtype=float64

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(25).astype(np.float64)

torch.manual_seed(42)
out_pt = torch.randn_like(torch.tensor(input_np))

paddle.seed(42)
out_pd = paddle.randn_like(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.numpy() - out_pd.numpy()))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.randn_like",
  "input": {"shape": [25], "dtype": "float64"}
}
```

- RNG 底层实现差异（Philox + Box-Muller 等）。

## Issue 109

llm_enhanced_torch_randn_like_20251125_141142.json_sample5.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][randn_like] Output difference anomaly under equivalent migration in randn_like operator (sample5)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a significant output discrepancy was detected for `torch.randn_like` (Maximum difference: 1.7193031781428427).

Same root cause as Issue 107: **RNG implementation difference**. This sample has a 4-D input of shape [1, 2, 1, 5].

- Input: shape=[1, 2, 1, 5], dtype=float64
- Maximum difference: 1.7193031781428427

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(1, 2, 1, 5).astype(np.float64)

# PyTorch
torch.manual_seed(42)
out_pt = torch.randn_like(torch.tensor(input_np))

# PaddlePaddle
paddle.seed(42)
out_pd = paddle.randn_like(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.numpy() - out_pd.numpy()))
print(f"Maximum difference: {max_diff}")
# RNG 不可复现——完全随机不可对齐
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][randn_like] Output difference anomaly under equivalent migration in randn_like operator (sample5)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`randn_like` 算子的最大输出差异为 1.7193031781428427。

与 Issue 107 相同原因：RNG 底层实现差异，随机数不可对齐。

- 输入形状: [1, 2, 1, 5], dtype=float64

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(1, 2, 1, 5).astype(np.float64)

torch.manual_seed(42)
out_pt = torch.randn_like(torch.tensor(input_np))

paddle.seed(42)
out_pd = paddle.randn_like(paddle.to_tensor(input_np))

max_diff = np.max(np.abs(out_pt.numpy() - out_pd.numpy()))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.randn_like",
  "input": {"shape": [1, 2, 1, 5], "dtype": "float64"}
}
```

- RNG 底层实现差异，随机数完全不可复现。

## Issue 110

llm_enhanced_torch_randperm_20251202_133813.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][randperm] Output difference anomaly under equivalent migration in randperm operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for `torch.randperm` (Maximum difference: 59864).

The root cause is **fundamentally different PRNG implementations**: `torch.randperm(n)` and `paddle.randperm(n)` both generate a random permutation of integers [0, n-1], but use different internal RNG algorithms. Even with the same seed, the generated permutations are completely different. For n=60000, the maximum possible difference is 59999, and the observed 59864 is consistent with two independent random permutations.

- n: 60000
- Maximum difference: 59864

```python
import numpy as np
import torch
import paddle

# PyTorch
torch.manual_seed(42)
out_pt = torch.randperm(60000)

# PaddlePaddle
paddle.seed(42)
out_pd = paddle.randperm(60000)

max_diff = np.max(np.abs(out_pt.numpy().astype(np.int64) - out_pd.numpy().astype(np.int64)))
print(f"Maximum difference: {max_diff}")
print(f"PyTorch first 10: {out_pt[:10].numpy()}")
print(f"Paddle first 10: {out_pd[:10].numpy()}")
# RNG 算法不同，生成的排列完全不同
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][randperm] Output difference anomaly under equivalent migration in randperm operator (sample1)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`randperm` 算子的最大输出差异为 59864。

根本原因是**随机数生成器（RNG）底层实现不同**：`torch.randperm(n)` 和 `paddle.randperm(n)` 都生成 [0, n-1] 的随机排列，但使用不同的内部 RNG 算法。对于 n=60000，最大可能差异为 59999，观察到 59864 符合两个独立随机排列的预期差异。

- n: 60000

```python
import numpy as np
import torch
import paddle

torch.manual_seed(42)
out_pt = torch.randperm(60000)

paddle.seed(42)
out_pd = paddle.randperm(60000)

max_diff = np.max(np.abs(out_pt.numpy().astype(np.int64) - out_pd.numpy().astype(np.int64)))
print(f"Maximum difference: {max_diff}")
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.randperm",
  "n": 60000
}
```

- 随机排列生成算子，RNG 底层实现差异导致排列完全不同。
"""

def main():
    output_file = r"d:\graduate\DFrameworkTest\issues\138个跨表不一致Case的GitHub Issue-修改版.md"
    
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(issues_content.strip())
        f.write("\n")
    
    print(f"Issues 081-110 已追加写入: {output_file}")

if __name__ == "__main__":
    main()
