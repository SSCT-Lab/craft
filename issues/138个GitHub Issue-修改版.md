



# 138个跨表不一致Case的GitHub Issue提交稿

> 用法：每个 `## Issue` 段落可直接复制到 GitHub 新建 Issue。
> 口径：同一 `file_name` 在至少两张分配表中出现“不一致/有问题”信号。

- 总Issue数：**138**
- 来源文件：`分配-朱婷.xlsx`、`分配-林哲远.xlsx`、`分配-陈建军.xlsx`、`分配-陈桂学.xlsx`

## Issue 001（输入未统一）

llm_enhanced_torch_atleast_1d_20251202_132406.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][atleast_1d] Output difference anomaly under equivalent migration in atleast_1d operator

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a severe output discrepancy was detected for the `torch.atleast_1d` operator. The mismatch has been repeatedly flagged as inconsistent by multiple reviewers.

The phenomenon exceeds the pure numerical error range (Maximum difference: 2.4143106937408447). It is highly suspected to be a defect in parameter mapping, input alignment, or semantic adaptation between the frameworks. This directly affects the credibility of this operator in cross-framework migration verification and causes persistent false positives/negatives in regression tests.

Python

```
import numpy as np
import torch
import paddle

# 使用 numpy 构造相同的输入数据，确保两框架输入完全一致
np.random.seed(42)
data_float32 = np.random.randn(1).astype(np.float32)
data_int64 = np.random.randint(0, 10, size=(1,)).astype(np.int64)
data_bool = np.array([True])

# PyTorch 执行
t1_pt = torch.tensor(data_float32)
t2_pt = torch.tensor(data_int64)
t3_pt = torch.tensor(data_bool)
out_pt = torch.atleast_1d(t1_pt, t2_pt, t3_pt)

# PaddlePaddle 执行
t1_pd = paddle.to_tensor(data_float32)
t2_pd = paddle.to_tensor(data_int64)
t3_pd = paddle.to_tensor(data_bool)
out_pd = paddle.atleast_1d(t1_pd, t2_pd, t3_pd)

# 比较结果
for i, (pt_out, pd_out) in enumerate(zip(out_pt, out_pd)):
    pt_np = pt_out.numpy().astype(np.float64)
    pd_np = pd_out.numpy().astype(np.float64)
    max_diff = np.max(np.abs(pt_np - pd_np))
    print(f"Tensor {i}: PT shape={pt_out.shape}, PD shape={pd_out.shape}, max_diff={max_diff}")
# 实测最大差异: 2.4143106937408447
```

**Versions*** 

Collecting environment information...
PyTorch version: 2.7.1+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Microsoft Windows 11 家庭中文版 (10.0.26100 64 位)
GCC version: (x86_64-win32-seh-rev0, Built by MinGW-W64 project) 8.1.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: N/A

Python version: 3.10.18 | packaged by Anaconda, Inc. | (main, Jun  5 2025, 13:08:55) [MSC v.1929 64 
bit (AMD64)] (64-bit runtime)                                                                       Python platform: Windows-10-10.0.26100-SP0
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True
Caching allocator config: N/A

CPU:
Name: AMD Ryzen 7 8845H w/ Radeon 780M Graphics
Manufacturer: AuthenticAMD
Family: 107
Architecture: 9
ProcessorType: 3
DeviceID: CPU0
CurrentClockSpeed: 3801
MaxClockSpeed: 3801
L2CacheSize: 8192
L2CacheSpeed: None
Revision: 29954

Versions of relevant libraries:
[pip3] numpy==1.26.4
[pip3] onnx==1.19.0
[pip3] onnx_graphsurgeon==0.5.8
[pip3] onnx2tf==1.28.8
[pip3] onnxruntime==1.22.1
[pip3] optree==0.16.0
[pip3] torch==2.7.1
[pip3] torchaudio==2.7.1
[pip3] torchvision==0.22.1
[conda] numpy                     1.26.4                   pypi_0    pypi
[conda] optree                    0.16.0                   pypi_0    pypi
[conda] torch                     2.7.1                    pypi_0    pypi
[conda] torchaudio                2.7.1                    pypi_0    pypi
[conda] torchvision               0.22.1                   pypi_0    pypi

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][atleast_1d] Output difference anomaly under equivalent migration in atleast_1d operator

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，针对 `atleast_1d` 算子，A/B 框架的输出结果被多位评审重复标记为不一致。当前现象已超过单纯数值误差范围（最大差异: 2.4143106937408447），疑似存在参数映射、输入对齐或语义适配缺陷。

此问题直接影响该算子在跨框架迁移验证中的可信度，并可能导致回归测试出现持续误报或漏报。

Python

```
import numpy as np
import torch
import paddle

# 使用 numpy 构造相同的输入数据，确保两框架输入完全一致
np.random.seed(42)
data_float32 = np.random.randn(1).astype(np.float32)
data_int64 = np.random.randint(0, 10, size=(1,)).astype(np.int64)
data_bool = np.array([True])

# PyTorch 执行
t1_pt = torch.tensor(data_float32)
t2_pt = torch.tensor(data_int64)
t3_pt = torch.tensor(data_bool)
out_pt = torch.atleast_1d(t1_pt, t2_pt, t3_pt)

# PaddlePaddle 执行
t1_pd = paddle.to_tensor(data_float32)
t2_pd = paddle.to_tensor(data_int64)
t3_pd = paddle.to_tensor(data_bool)
out_pd = paddle.atleast_1d(t1_pd, t2_pd, t3_pd)

# 比较结果
for i, (pt_out, pd_out) in enumerate(zip(out_pt, out_pd)):
    pt_np = pt_out.numpy().astype(np.float64)
    pd_np = pd_out.numpy().astype(np.float64)
    max_diff = np.max(np.abs(pt_np - pd_np))
    print(f"Tensor {i}: PT shape={pt_out.shape}, PD shape={pd_out.shape}, max_diff={max_diff}")
# 实测最大差异: 2.4143106937408447
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

JSON

```
{
  "api": "paddle.atleast_1d",
  "*tensors": [
    {
      "shape": [1],
      "dtype": "float32"
    },
    {
      "shape": [1],
      "dtype": "int64"
    },
    {
      "shape": [1],
      "dtype": "bool"
    }
  ]
}
```

## Issue 002
llm_enhanced_torch_bmm_20251215_230705.json_sample1.txt

### PyTorch Issue

**Add a title*** 

[PyTorch -> TensorFlow][bmm] Output difference anomaly under equivalent migration in bmm operator

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for the `torch.bmm` operator.

The maximum difference observed is 9.5367431640625e-07. While this falls within standard float32 numerical precision margins, it has been flagged as inconsistent during strict cross-framework alignment verification. This affects the strict credibility of this operator in cross-framework migration verification and causes false positives in regression tests demanding exact bit-level equivalence.

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(8, 131072, 3).astype(np.float32)
mat2_np = np.random.randn(8, 3, 3).astype(np.float32)

input_pt = torch.tensor(input_np)
mat2_pt = torch.tensor(mat2_np)
out_pt = torch.bmm(input_pt, mat2_pt)

input_tf = tf.constant(input_np)
mat2_tf = tf.constant(mat2_np)
out_tf = tf.linalg.matmul(input_tf, mat2_tf)

pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
max_diff = np.max(np.abs(pt_np_out - tf_np_out))
print(f"PyTorch output shape: {pt_np_out.shape}")
print(f"TensorFlow output shape: {tf_np_out.shape}")
print(f"Maximum difference: {max_diff}")
```

output：

```
Maximum difference: 9.5367431640625e-07
```

**Versions***

同1

### TensorFlow Issue（√）

**Add a title*** 

[PyTorch -> TensorFlow][tf.linalg.matmul] Output difference anomaly under equivalent migration in bmm operator

**Issue type*** 

Bug

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

When executing `tf.linalg.matmul(a, b)` with inputs equivalent to PyTorch's `torch.bmm`, the output results demonstrate a numerical discrepancy. The maximum difference between the outputs of the two frameworks reaches 9.5367431640625e-07. While this represents a standard floating-point variance, it triggers failures in strict cross-framework migration verification tests for large batch shapes (e.g., `[8, 131072, 3]`).

Expected behavior: The numerical output of `tf.linalg.matmul` should align with `torch.bmm` behaviors or provide configuration options to eliminate non-deterministic floating-point discrepancies under strict equivalent inputs.

**Standalone code to reproduce the issue***

Python

```
import numpy as np
import torch
import tensorflow as tf

# 使用相同的随机种子生成输入数据，确保两框架输入完全一致
np.random.seed(42)
input_np = np.random.randn(8, 131072, 3).astype(np.float32)
mat2_np = np.random.randn(8, 3, 3).astype(np.float32)

# PyTorch 执行
input_pt = torch.tensor(input_np)
mat2_pt = torch.tensor(mat2_np)
out_pt = torch.bmm(input_pt, mat2_pt)

# TensorFlow 执行
input_tf = tf.constant(input_np)
mat2_tf = tf.constant(mat2_np)
out_tf = tf.linalg.matmul(input_tf, mat2_tf)

# 比较结果
pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
max_diff = np.max(np.abs(pt_np_out - tf_np_out))
print(f"PyTorch output shape: {pt_np_out.shape}")
print(f"TensorFlow output shape: {tf_np_out.shape}")
print(f"Maximum difference: {max_diff}")
# 实测最大差异:9.5367431640625e-07
```

**Relevant log output**

Plaintext

```
Maximum difference: 9.5367431640625e-07
```

## Issue 003
llm_enhanced_torch_diagflat_20251215_234731.json_sample4.txt

### PyTorch Issue

**Add a title*** 

[PyTorch -> TensorFlow][diagflat] Output difference anomaly under equivalent migration in llm_enhanced_torch_diagflat_20251215_234731.json_sample4.txt

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a shape mismatch discrepancy was detected for the `torch.diagflat` operator. The mismatch has been repeatedly flagged as inconsistent by multiple reviewers.

The issue stems from a parameter mapping or semantic adaptation defect. Specifically, when `torch.diagflat` is called with an `offset=1`, it returns a matrix of shape `(5, 5)` for a 1D input of size `4`. The migrated TensorFlow equivalent lacks the translation of this offset parameter, resulting in a standard diagonal matrix of shape `(4, 4)`. This directly affects the credibility of this operator in cross-framework migration verification and causes persistent false positives/negatives in regression tests.

```python
import numpy as np
import torch
import tensorflow as tf

input_data = [0.23347382247447968, -0.032445747405290604, -0.03567954897880554, 1.7368338108062744]
input_np = np.array(input_data, dtype=np.float32)
offset = 1

input_pt = torch.tensor(input_np)
out_pt = torch.diagflat(input_pt, offset=offset)

input_tf = tf.constant(input_np)
out_tf = tf.linalg.diag(input_tf)

pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
print(f"PyTorch output shape: {pt_np_out.shape}")   # (5, 5)
print(f"TensorFlow output shape: {tf_np_out.shape}") # (4, 4)
if pt_np_out.shape != tf_np_out.shape:
    print(f"Shape mismatch: PyTorch {pt_np_out.shape} vs TensorFlow {tf_np_out.shape}")
else:
    max_diff = np.max(np.abs(pt_np_out - tf_np_out))
    print(f"Maximum difference: {max_diff}")
```

```
PyTorch output shape: (5, 5)
TensorFlow output shape: (4, 4)
Shape mismatch: PyTorch (5, 5) vs TensorFlow (4, 4)
```

**Versions***

同1

### TensorFlow Issue（√）

**Add a title*** 

[PyTorch -> TensorFlow][tf.linalg.diag] Output difference anomaly under equivalent migration in  diagflat operator

**Issue type*** 

Bug

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

During cross-framework equivalent migration testing, mapping PyTorch's `torch.diagflat` to TensorFlow's `tf.linalg.diag` results in a structural output discrepancy.

In the failing sample, the PyTorch implementation includes an `offset=1` parameter, placing the 1D input array of size `4` on the first super-diagonal, thereby yielding a `(5, 5)` tensor. The generated equivalent code for TensorFlow fails to map this offset (which should correspond to the `k` argument in `tf.linalg.diag`), resulting in a `(4, 4)` tensor where the input values populate the main diagonal. This indicates a defect in parameter mapping and semantic adaptation for the `diagflat` operator.

Expected behavior: The generated TensorFlow code should map the `offset` parameter to the `k` parameter in `tf.linalg.diag` to produce an output shape of `(5, 5)` that matches PyTorch's semantic behavior.

**Standalone code to reproduce the issue***

Python

```
import numpy as np
import torch
import tensorflow as tf

# 使用 JSON 中的原始数据构造相同输入
input_data = [0.23347382247447968, -0.032445747405290604, -0.03567954897880554, 1.7368338108062744]
input_np = np.array(input_data, dtype=np.float32)
offset = 1

# PyTorch 执行
input_pt = torch.tensor(input_np)
out_pt = torch.diagflat(input_pt, offset=offset)

# TensorFlow 执行
# 注意：torch.diagflat 的 offset 参数应对应 tf.linalg.diag 的 k 参数
# 当前迁移代码缺失了 k 参数映射，导致形状不匹配
input_tf = tf.constant(input_np)
out_tf = tf.linalg.diag(input_tf)  # 缺少 k=1 参数映射

# 比较结果
pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
print(f"PyTorch output shape: {pt_np_out.shape}")   # (5, 5)
print(f"TensorFlow output shape: {tf_np_out.shape}") # (4, 4)
if pt_np_out.shape != tf_np_out.shape:
    print(f"Shape mismatch: PyTorch {pt_np_out.shape} vs TensorFlow {tf_np_out.shape}")
else:
    max_diff = np.max(np.abs(pt_np_out - tf_np_out))
    print(f"Maximum difference: {max_diff}")
```

**Relevant log output**

Plaintext

```
PyTorch output shape: (5, 5)
TensorFlow output shape: (4, 4)
Shape mismatch: PyTorch (5, 5) vs TensorFlow (4, 4)
```


## Issue 004（未初始化内存）

llm_enhanced_torch_empty_like_20251214_172128.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][empty_like] Output difference anomaly under equivalent migration in empty_like operator

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, an extremely large output discrepancy was detected for the `torch.empty_like` operator (Maximum difference: 4607182418800017408). The mismatch has been flagged as inconsistent by multiple reviewers.

The root cause is that `empty_like` allocates uninitialized memory in both frameworks. The output values depend entirely on residual garbage data in memory, which inherently differs between frameworks and even between successive calls within the same framework. This is a comparison methodology issue, not a framework defect.

```python
import numpy as np
import torch
import mindspore

# 使用原始样本数据构造输入
data = np.array([6, 8, 9], dtype=np.int64)

# PyTorch 执行
input_pt = torch.tensor(data)
out_pt = torch.empty_like(input_pt)

# MindSpore 执行
input_ms = mindspore.Tensor(data)
out_ms = mindspore.mint.empty_like(input_ms)

# 比较结果
pt_np = out_pt.numpy().astype(np.float64)
ms_np = out_ms.asnumpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - ms_np))
print(f"PyTorch output shape: {out_pt.shape}, values: {out_pt.numpy()}")
print(f"MindSpore output shape: {out_ms.shape}, values: {out_ms.asnumpy()}")
print(f"Maximum difference: {max_diff}")
# 注意: empty_like 返回未初始化内存，不同框架甚至同框架不同调用结果必然不同
# 实测最大差异: 4607182418800017408
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.mint.empty_like] Output difference anomaly under equivalent migration in empty_like operator

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

During cross-framework equivalent migration testing, `torch.empty_like` and `mindspore.mint.empty_like` produce dramatically different outputs (Maximum difference: 4607182418800017408) when applied to the same input tensor (shape=[3], dtype=int64). Both functions return uninitialized memory, and the output values depend entirely on residual garbage data, which inherently differs across frameworks.

**Describe the expected behavior***

Both frameworks correctly implement `empty_like` semantics (returning uninitialized memory). The migration comparison framework should explicitly exclude `empty_like` from numerical comparison, as this operator's semantics guarantee non-deterministic output.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

data = np.array([6, 8, 9], dtype=np.int64)

input_pt = torch.tensor(data)
out_pt = torch.empty_like(input_pt)

input_ms = mindspore.Tensor(data)
out_ms = mindspore.mint.empty_like(input_ms)

pt_np = out_pt.numpy().astype(np.float64)
ms_np = out_ms.asnumpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - ms_np))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 4607182418800017408
```

**Related log / screenshot**

```
comparison_error: Numerical mismatch, maximum difference: 4607182418800017408
```

**Special notes for this issue**

`empty_like` allocates uninitialized memory. The output values are undefined and framework-dependent. This is expected behavior, not a framework bug.

## Issue 005

llm_enhanced_torch_erfinv_20251125_145454.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][erfinv] Output difference anomaly under equivalent migration in erfinv operator

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, an output discrepancy was detected for the `torch.erfinv` operator.

The root cause is that the input data contains values outside the mathematical domain of `erfinv`, which is strictly `(-1, 1)`. For the out-of-domain input value `-1.4654309`, PyTorch produces `nan`, whereas PaddlePaddle produces `-inf`. This divergence in handling out-of-domain values leads to an inconsistent comparison result (Maximum difference: nan) in cross-framework validations.

```python
import numpy as np
import torch
import paddle

input_data = [0.7778396010398865, -0.9116541147232056, 0.43586117029190063,
              -1.4654308557510376, -0.7387789487838745]
input_np = np.array(input_data, dtype=np.float32)

input_pt = torch.tensor(input_np)
out_pt = torch.erfinv(input_pt)

input_pd = paddle.to_tensor(input_np)
out_pd = paddle.erfinv(input_pd)

pt_np = out_pt.numpy().astype(np.float64)
pd_np = out_pd.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"PyTorch erfinv output: {out_pt.numpy()}")
print(f"PaddlePaddle erfinv output: {out_pd.numpy()}")
print(f"Maximum difference: {max_diff}")

# output:
Input values: [ 0.7778396  -0.9116541   0.43586117 -1.4654309  -0.73877895]
Out-of-domain indices: [3]
PyTorch erfinv output: [ 0.8632385  -1.2050432   0.40779194         nan -0.7944414 ]
PaddlePaddle erfinv output: [ 0.8632385  -1.2050431   0.40779188        -inf -0.79444134]
Maximum difference: nan
```

**Versions***

同1

### PaddlePaddle Issue（√）

**Title*** 

 [PyTorch -> Paddle][erfinv] Output difference anomaly under equivalent migration in erfinv operator

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，针对 `erfinv` 算子（`shape=[5], dtype=float32`），两框架对于定义域外输入的数据处理逻辑不一致，导致输出结果存在差异（最大差异为 `nan`）。

根本原因：`erfinv` 的数学定义域为 `(-1, 1)`。输入数据中包含超出该定义域的负数值（如 `-1.4654309`）。对于此超出范围的输入，PyTorch 返回 `nan`，而 PaddlePaddle 返回 `-inf`。这种处理策略的差异直接导致了跨框架校验的失败。

```python
import numpy as np
import torch
import paddle

input_data = [0.7778396010398865, -0.9116541147232056, 0.43586117029190063,
              -1.4654308557510376, -0.7387789487838745]
input_np = np.array(input_data, dtype=np.float32)

input_pt = torch.tensor(input_np)
out_pt = torch.erfinv(input_pt)

input_pd = paddle.to_tensor(input_np)
out_pd = paddle.erfinv(input_pd)

pt_np = out_pt.numpy().astype(np.float64)
pd_np = out_pd.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"PyTorch erfinv output: {out_pt.numpy()}")
print(f"PaddlePaddle erfinv output: {out_pd.numpy()}")
print(f"Maximum difference: {max_diff}")
# 实测最大差异: nan
# Input values: [ 0.7778396  -0.9116541   0.43586117 -1.4654309  -0.73877895]
# Out-of-domain indices: [3]
# PyTorch erfinv output: [ 0.8632385  -1.2050432   0.40779194         nan -0.7944414 ]
# PaddlePaddle erfinv output: [ 0.8632385  -1.2050431   0.40779188        -inf -0.79444134]
# Maximum difference: nan
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.erfinv",
  "input": {
    "shape": [5],
    "dtype": "float32"
  }
}
```

- `erfinv` 的数学定义域严格为 `(-1, 1)`。当输入小于 -1 时，由于底层实现或异常捕获机制的不同，PyTorch 返回 `nan`，而 PaddlePaddle 返回 `-inf`，建议框架统一越界输入的异常值输出标准，或在文档中明确说明越界行为。

## Issue 006（同5）

llm_enhanced_torch_erfinv_20251125_145454.json_sample3.txt

### PyTorch Issue

Title* 

[torch.erfinv] Inconsistent cross-framework behavior (nan vs +/-inf) for float64 out-of-domain inputs

🐛 Describe the bug* 

During cross-framework equivalent migration testing, a discrepancy was identified in the `torch.erfinv` operator when processing `float64` 1D tensors containing both positive and negative out-of-domain values.

The mathematical domain for the inverse error function is (-1, 1). However, when the input array includes values strictly outside this domain (e.g., `1.4295`, `-1.4130`, `1.8013`, `1.0087`), PyTorch uniformly returns `nan` for all out-of-bounds elements. In contrast, PaddlePaddle returns `inf` for positive out-of-bounds inputs and `-inf` for negative ones. This difference in exception handling semantics results in a validation failure (Maximum difference: nan).

Python

```
import numpy as np
import torch
import paddle

input_data = [-0.877115400943947, -0.17205383221963766, 0.14039304124475271,
              0.9606750841273599, 1.4295494246249612, -0.45974536025361523,
              -1.4130413570449318, 1.8013200780067657, 1.0087162320309664,
              -0.02971757573285859]
input_np = np.array(input_data, dtype=np.float64)

input_pt = torch.tensor(input_np)
out_pt = torch.erfinv(input_pt)

input_pd = paddle.to_tensor(input_np)
out_pd = paddle.erfinv(input_pd)

pt_np = out_pt.numpy()
pd_np = out_pd.numpy()
oob_mask = np.abs(input_np) >= 1
print(f"PyTorch erfinv output: {pt_np}")
print(f"PaddlePaddle erfinv output: {pd_np}")
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"Maximum difference: {max_diff}")

# 实际输出
# PyTorch erfinv output: [-1.09090893 -0.15368006  0.12506917  1.45718497         nan -0.43305197
#         nan         nan         nan -0.02634261]
# PaddlePaddle erfinv output: [-1.09090893 -0.15368006  0.12506917  1.45718497         inf -0.43305197
#        -inf         inf         inf -0.02634261]
# Maximum difference: nan
```

Versions*

同1

### PaddlePaddle Issue（√）

Title* 

paddle.erfinv 在 float64 一维张量双向越界输入时跨框架对齐失败 (nan vs +/-inf)

bug描述 Describe the Bug* 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，发现在处理 `float64` 类型的 1D 数据时，两框架对于 `erfinv` 算子的正负双向越界输入存在底层异常返回值的差异。

输入数组（shape=[10], dtype=float64）中同时包含正向越界值（如 `1.4295`, `1.8013`, `1.0087`）和负向越界值（如 `-1.4130`）。对于偏离数学定义域 (-1, 1) 的数据，PaddlePaddle 根据符号分别返回了 `inf` 和 `-inf`，而 PyTorch 统一返回 `nan`。这种异常处理策略的差异导致跨框架测试结果判定失败（最大差异为 nan）。

Python

```
import numpy as np
import torch
import paddle

input_data = [-0.877115400943947, -0.17205383221963766, 0.14039304124475271,
              0.9606750841273599, 1.4295494246249612, -0.45974536025361523,
              -1.4130413570449318, 1.8013200780067657, 1.0087162320309664,
              -0.02971757573285859]
input_np = np.array(input_data, dtype=np.float64)

input_pt = torch.tensor(input_np)
out_pt = torch.erfinv(input_pt)

input_pd = paddle.to_tensor(input_np)
out_pd = paddle.erfinv(input_pd)

pt_np = out_pt.numpy()
pd_np = out_pd.numpy()
oob_mask = np.abs(input_np) >= 1
print(f"PyTorch erfinv output: {pt_np}")
print(f"PaddlePaddle erfinv output: {pd_np}")
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"Maximum difference: {max_diff}")

# 实际输出
# PyTorch erfinv output: [-1.09090893 -0.15368006  0.12506917  1.45718497         nan -0.43305197
#         nan         nan         nan -0.02634261]
# PaddlePaddle erfinv output: [-1.09090893 -0.15368006  0.12506917  1.45718497         inf -0.43305197
#        -inf         inf         inf -0.02634261]
# Maximum difference: nan
```

其他补充信息 Additional Supplementary Information 

原始 JSON 提取的算子配置特征:

JSON

```
{
  "api": "paddle.erfinv",
  "input": {
    "shape": [10],
    "dtype": "float64"
  }
}
```

建议底层 C++ 实现或算子层面明确针对 `|x| >= 1` 的边界行为。目前返回 `+/-inf` 虽能表示越界方向，但在主流框架对齐时会导致兼容性测试无法通过。

## Issue 007（同5）

llm_enhanced_torch_erfinv_20251125_151700.json_sample3.txt

### PyTorch Issue

Title* [torch.erfinv] Output difference on 2D float32 tensors with negative out-of-domain values

🐛 Describe the bug* A cross-framework output anomaly was detected for the `torch.erfinv` operator when evaluating equivalent code against PaddlePaddle. This specifically occurs with 2D `float32` tensors containing negative values that fall outside the permitted mathematical domain.

Given a tensor of shape `[10, 1]`, the input data includes negative out-of-bounds values such as `-1.0186`, `-1.6230`, and `-1.4968`. The `erfinv` function is only valid for strictly `(-1, 1)`. In this edge case, PyTorch populates the output tensor with `nan` for those specific indices. PaddlePaddle, however, maps these negative out-of-domain values to `-inf`. This divergent handling of undefined inputs breaks downstream comparative analysis across frameworks (Maximum difference: nan).

Python

```
import numpy as np
import torch
import paddle

input_data = [[0.2596988081932068], [-0.9840201735496521], [-0.2839934229850769],
              [0.931747317314148], [-1.0186134576797485], [-1.6229819059371948],
              [-0.2458077371120453], [-1.4968321323394775], [-0.9785251617431641],
              [0.8734411001205444]]
input_np = np.array(input_data, dtype=np.float32)

input_pt = torch.tensor(input_np)
out_pt = torch.erfinv(input_pt)

input_pd = paddle.to_tensor(input_np)
out_pd = paddle.erfinv(input_pd)

pt_np = out_pt.numpy().astype(np.float64)
pd_np = out_pd.numpy().astype(np.float64)
max_diff = np.max(np.abs(pt_np - pd_np))
print(f"PyTorch erfinv output:\n{out_pt.numpy()}")
print(f"PaddlePaddle erfinv output:\n{out_pd.numpy()}")
print(f"Maximum difference: {max_diff}")

# 实测最大差异: nan
# PyTorch erfinv output:
# [[ 0.23437373]
#  [-1.7036861 ]
#  [-0.2572462 ]
#  [ 1.2892925 ]
#  [        nan]
#  [        nan]
#  [-0.2214067 ]
#  [        nan]
#  [-1.6260135 ]
#  [ 1.0803272 ]]
# PaddlePaddle erfinv output:
# [[ 0.2343738 ]
#  [-1.7036862 ]
#  [-0.2572462 ]
#  [ 1.2892925 ]
#  [       -inf]
#  [       -inf]
#  [-0.22140673]
#  [       -inf]
#  [-1.6260134 ]
#  [ 1.0803269 ]]
# Maximum difference: nan
```

Versions*

同1

### PaddlePaddle Issue（√）

- Title* paddle.erfinv 针对二维 float32 张量负向越界输入的异常处理跨框架不一致 (nan vs -inf)

  bug描述 Describe the Bug* 在算子等价迁移校验中，发现 PaddlePaddle 处理带有负向越界值的 `float32` 二维张量（shape=[10, 1]）时，其 `erfinv` 算子的输出表现与 PyTorch 不一致，已被标记为迁移阻塞项。

  输入数据中包含了位于 `erfinv` 数学定义域 (-1, 1) 左侧的值（例如 `-1.0186`、`-1.6230`、`-1.4968`）。对于二维张量中的这些非法输入点，PyTorch 会在对应位置输出 `nan`，而 PaddlePaddle 此时的底层逻辑将其处理为 `-inf`。这一差异使得逐元素的 `np.abs(pt_np - pd_np)` 比较失败，计算得出的最大差异呈现为 nan。

  Python

  ```
  import numpy as np
  import torch
  import paddle
  
  input_data = [[0.2596988081932068], [-0.9840201735496521], [-0.2839934229850769],
                [0.931747317314148], [-1.0186134576797485], [-1.6229819059371948],
                [-0.2458077371120453], [-1.4968321323394775], [-0.9785251617431641],
                [0.8734411001205444]]
  input_np = np.array(input_data, dtype=np.float32)
  
  input_pt = torch.tensor(input_np)
  out_pt = torch.erfinv(input_pt)
  
  input_pd = paddle.to_tensor(input_np)
  out_pd = paddle.erfinv(input_pd)
  
  pt_np = out_pt.numpy().astype(np.float64)
  pd_np = out_pd.numpy().astype(np.float64)
  max_diff = np.max(np.abs(pt_np - pd_np))
  print(f"Maximum difference: {max_diff}")
  
  # 实测最大差异: nan
  # PyTorch erfinv output:
  # [[ 0.23437373]
  #  [-1.7036861 ]
  #  [-0.2572462 ]
  #  [ 1.2892925 ]
  #  [        nan]
  #  [        nan]
  #  [-0.2214067 ]
  #  [        nan]
  #  [-1.6260135 ]
  #  [ 1.0803272 ]]
  # PaddlePaddle erfinv output:
  # [[ 0.2343738 ]
  #  [-1.7036862 ]
  #  [-0.2572462 ]
  #  [ 1.2892925 ]
  #  [       -inf]
  #  [       -inf]
  #  [-0.22140673]
  #  [       -inf]
  #  [-1.6260134 ]
  #  [ 1.0803269 ]]
  # Maximum difference: nan
  ```

  其他补充信息 Additional Supplementary Information 原始 JSON 提取的算子配置特征:

  JSON

  ```
  {
    "api": "paddle.erfinv",
    "input": {
      "shape": [10, 1],
      "dtype": "float32"
    }
  }
  ```

  越界输入的处理在张量结构变化（如从1D升维到2D）时依然稳定复现。建议在 API 文档中补充针对 `x <= -1` 及 `x >= 1` 的边界行为说明，或在迁移工具中考虑加入对此类 `nan` 与 `-inf` 映射的处理机制。

## Issue 008（输出含nan，实际输出一致）

llm_enhanced_torch_erfinv_20251215_184626.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][erfinv] Output difference anomaly under equivalent migration in erfinv operator

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, a NaN-based output discrepancy was detected for the `torch.erfinv` operator. The mismatch has been flagged as inconsistent by multiple reviewers.

The input data (shape=[2, 3, 4, 5], dtype=float64) contains values outside the mathematical domain of `erfinv` (-1, 1), such as -1.2202 and -1.7247. The `erfinv` function is mathematically undefined for |x| >= 1. PyTorch returns NaN for such inputs, while MindSpore may handle them differently, leading to comparison failure (Maximum difference: nan).

```python
import numpy as np
import torch
import mindspore

# 使用随机种子生成与原始测试一致的输入
np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 5).astype(np.float64)

# PyTorch 执行
input_pt = torch.tensor(input_np)
out_pt = torch.erfinv(input_pt)

# MindSpore 执行
input_ms = mindspore.Tensor(input_np)
out_ms = mindspore.mint.erfinv(input_ms)

# 比较结果
pt_np = out_pt.numpy()
ms_np = out_ms.asnumpy()
oob_count = np.sum(np.abs(input_np) >= 1)
max_diff = np.max(np.abs(pt_np - ms_np))
print(f"Input shape: {input_np.shape}")
print(f"Number of out-of-domain values (|x| >= 1): {oob_count}")
print(f"Maximum difference: {max_diff}")
# 注意: erfinv 定义域为 (-1, 1)，输入含超出定义域的值如 -1.2202, -1.7247
# 实测最大差异: nan
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.mint.erfinv] Output difference anomaly under equivalent migration in erfinv operator

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

During cross-framework equivalent migration testing, `torch.erfinv` and `mindspore.mint.erfinv` produce different outputs when input contains values outside the domain (-1, 1). PyTorch returns NaN for out-of-domain values (e.g., -1.2202, -1.7247), while MindSpore may handle them differently, causing comparison failure (Maximum difference: nan).

**Describe the expected behavior***

Both frameworks should produce consistent handling of out-of-domain inputs for `erfinv`. For |x| >= 1, the expected mathematical result is undefined, and both frameworks should return NaN consistently.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(2, 3, 4, 5).astype(np.float64)

input_pt = torch.tensor(input_np)
out_pt = torch.erfinv(input_pt)

input_ms = mindspore.Tensor(input_np)
out_ms = mindspore.mint.erfinv(input_ms)

pt_np = out_pt.numpy()
ms_np = out_ms.asnumpy()
max_diff = np.max(np.abs(pt_np - ms_np))
print(f"Number of out-of-domain values: {np.sum(np.abs(input_np) >= 1)}")
print(f"Maximum difference: {max_diff}")
# 实测最大差异: nan
```

**Related log / screenshot**

```
comparison_error: Numerical mismatch, maximum difference: nan
```

**Special notes for this issue**

The `erfinv` function domain is (-1, 1). Input values outside this range produce mathematically undefined results (NaN). The discrepancy stems from different NaN handling across frameworks, not from a computation error.

## Issue 009（算子不完全等价）

llm_enhanced_torch_fmod_20251215_201658.json_sample10.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][fmod] Output difference anomaly under equivalent migration in fmod operator (sample10)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a severe output discrepancy was detected for the `torch.fmod` operator. The mismatch has been flagged as inconsistent by multiple reviewers.

The root cause is a semantic mismatch between `torch.fmod` and `tf.math.floormod`. `torch.fmod` implements truncated division remainder (C-style), where the result's sign follows the dividend. `tf.math.floormod` implements floor division remainder, where the result's sign follows the divisor. When negative numbers are involved, these two definitions yield different results.

- Input: shape=[6, 1, 5], dtype=float64
- Other: shape=[1, 5], dtype=float64
- Maximum difference: 2.1085654639330538

```python
import numpy as np
import torch
import tensorflow as tf

# 使用原始样本数据构造输入
np.random.seed(42)
input_np = np.random.randn(6, 1, 5).astype(np.float64)
other_np = np.array([[-0.6051631246422225, 0.8991063368663581, -1.1361974496746359,
                      0.040088999963003996, -2.1085654639330538]], dtype=np.float64)

# PyTorch 执行 (truncated division remainder)
input_pt = torch.tensor(input_np)
other_pt = torch.tensor(other_np)
out_pt = torch.fmod(input_pt, other_pt)

# TensorFlow 执行 (floor division remainder)
input_tf = tf.constant(input_np)
other_tf = tf.constant(other_np)
out_tf = tf.math.floormod(input_tf, other_tf)

# 比较结果
pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
max_diff = np.max(np.abs(pt_np_out - tf_np_out))
print(f"PyTorch fmod output (first row): {pt_np_out[0]}")
print(f"TensorFlow floormod output (first row): {tf_np_out[0]}")
print(f"Maximum difference: {max_diff}")
# Maximum difference: 2.1085654639330538
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.math.floormod] Output difference anomaly under equivalent migration in fmod operator (sample10)

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

When mapping `torch.fmod` to `tf.math.floormod` during cross-framework migration testing, a severe output discrepancy occurs (Maximum difference: 2.1085654639330538). The root cause is a fundamental semantic difference: `torch.fmod` computes the truncated remainder (sign follows dividend), while `tf.math.floormod` computes the floor remainder (sign follows divisor). For negative dividends or divisors, this produces different results.

Expected behavior: The migration tool should map `torch.fmod` to `tf.math.truncatemod` (or an equivalent implementation using `tf.math.truncatediv`) instead of `tf.math.floormod`, as the latter has fundamentally different semantics.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(6, 1, 5).astype(np.float64)
other_np = np.array([[-0.6051631246422225, 0.8991063368663581, -1.1361974496746359,
                      0.040088999963003996, -2.1085654639330538]], dtype=np.float64)

input_pt = torch.tensor(input_np)
other_pt = torch.tensor(other_np)
out_pt = torch.fmod(input_pt, other_pt)

input_tf = tf.constant(input_np)
other_tf = tf.constant(other_np)
out_tf = tf.math.floormod(input_tf, other_tf)

pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
max_diff = np.max(np.abs(pt_np_out - tf_np_out))
print(f"Maximum difference: {max_diff}")
# Maximum difference: 2.1085654639330538
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 2.1085654639330538
```

## Issue 010（同9）

llm_enhanced_torch_fmod_20251215_201658.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][fmod] Output difference anomaly under equivalent migration in fmod operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a severe output discrepancy was detected for the `torch.fmod` operator against a scalar divisor (Maximum difference: 1.5).

The root cause is the semantic mismatch between `torch.fmod` (truncated division remainder) and `tf.math.floormod` (floor division remainder). For negative dividends with a negative divisor (-1.5), the sign convention differs, causing systematic output differences.

- Input: shape=[6, 3, 2], dtype=float64
- Other: scalar -1.5

```python
import numpy as np
import torch
import tensorflow as tf

# 使用原始样本数据
np.random.seed(42)
input_np = np.random.randn(6, 3, 2).astype(np.float64)
other = -1.5

# PyTorch 执行 (truncated division remainder)
input_pt = torch.tensor(input_np)
out_pt = torch.fmod(input_pt, other)

# TensorFlow 执行 (floor division remainder)
input_tf = tf.constant(input_np)
out_tf = tf.math.floormod(input_tf, other)

# 比较结果
pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
max_diff = np.max(np.abs(pt_np_out - tf_np_out))
print(f"PyTorch fmod output (sample): {pt_np_out.flatten()[:5]}")
print(f"TensorFlow floormod output (sample): {tf_np_out.flatten()[:5]}")
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 1.5
# 根因: torch.fmod 采用截断除法，tf.math.floormod 采用地板除法
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.math.floormod] Output difference anomaly under equivalent migration in fmod operator (sample2)

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

When mapping `torch.fmod(input, -1.5)` to `tf.math.floormod(input, -1.5)` during cross-framework migration testing, the outputs differ significantly (Maximum difference: 1.5). `torch.fmod` uses truncated division (remainder sign follows dividend), while `tf.math.floormod` uses floor division (remainder sign follows divisor). This is a well-known semantic difference.

Expected behavior: The migration tool should map `torch.fmod` to `tf.math.truncatemod` or an equivalent truncated remainder implementation instead of `tf.math.floormod`.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(6, 3, 2).astype(np.float64)
other = -1.5

input_pt = torch.tensor(input_np)
out_pt = torch.fmod(input_pt, other)

input_tf = tf.constant(input_np)
out_tf = tf.math.floormod(input_tf, other)

max_diff = np.max(np.abs(out_pt.numpy() - out_tf.numpy()))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 1.5
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 1.5
```

## Issue 011（同9）

llm_enhanced_torch_fmod_20251215_201658.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][fmod] Output difference anomaly under equivalent migration in fmod operator (sample3)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a severe output discrepancy was detected for the `torch.fmod` operator (Maximum difference: 1.5).

The root cause remains the semantic mismatch: `torch.fmod` implements truncated remainder while `tf.math.floormod` implements floor remainder. For negative input values with a positive divisor (1.5), the results diverge.

- Input: shape=[5, 1], dtype=float32
- Other: scalar 1.5

```python
import numpy as np
import torch
import tensorflow as tf

# 使用原始样本数据
input_data = [[-0.8602762222290039], [-0.8983254432678223], [0.014818377792835236],
              [0.47603628039360046], [2.090144395828247]]
input_np = np.array(input_data, dtype=np.float32)
other = 1.5

# PyTorch 执行 (truncated division remainder)
input_pt = torch.tensor(input_np)
out_pt = torch.fmod(input_pt, other)

# TensorFlow 执行 (floor division remainder)
input_tf = tf.constant(input_np)
out_tf = tf.math.floormod(input_tf, other)

# 比较结果
pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
max_diff = np.max(np.abs(pt_np_out - tf_np_out))
print(f"PyTorch fmod output: {pt_np_out.flatten()}")
print(f"TensorFlow floormod output: {tf_np_out.flatten()}")
print(f"Maximum difference: {max_diff}")
# 例: fmod(-0.86, 1.5) = -0.86, floormod(-0.86, 1.5) = 0.64
# 实测最大差异: 1.5
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.math.floormod] Output difference anomaly under equivalent migration in fmod operator (sample3)

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

When mapping `torch.fmod(input, 1.5)` to `tf.math.floormod(input, 1.5)` with input shape=[5, 1] and dtype=float32, the outputs differ by up to 1.5. For negative inputs like -0.86, `torch.fmod` returns -0.86 (sign follows dividend), while `tf.math.floormod` returns 0.64 (sign follows divisor: -0.86 + 1.5 = 0.64). This is a semantic mismatch, not a numerical precision issue.

Expected behavior: The migration tool should use `tf.math.truncatemod` or equivalent instead of `tf.math.floormod` to match `torch.fmod` semantics.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

input_data = [[-0.8602762222290039], [-0.8983254432678223], [0.014818377792835236],
              [0.47603628039360046], [2.090144395828247]]
input_np = np.array(input_data, dtype=np.float32)
other = 1.5

input_pt = torch.tensor(input_np)
out_pt = torch.fmod(input_pt, other)

input_tf = tf.constant(input_np)
out_tf = tf.math.floormod(input_tf, other)

max_diff = np.max(np.abs(out_pt.numpy() - out_tf.numpy()))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 1.5
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 1.5
```

## Issue 012（同9）

llm_enhanced_torch_fmod_20251215_201658.json_sample6.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][fmod] Output difference anomaly under equivalent migration in fmod operator (sample6)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a severe output discrepancy was detected for the `torch.fmod` operator (Maximum difference: 3.7).

Same root cause as other fmod issues: `torch.fmod` (truncated remainder) vs `tf.math.floormod` (floor remainder) semantic mismatch. With a negative divisor of -3.7, positive input values produce opposite-sign remainders.

- Input: shape=[4, 1], dtype=float64
- Other: scalar -3.7

```python
import numpy as np
import torch
import tensorflow as tf

# 使用原始样本数据
input_data = [[-0.49545063101844433], [0.27046169092537087],
              [1.5556632190621569], [0.13487971268907317]]
input_np = np.array(input_data, dtype=np.float64)
other = -3.7

# PyTorch 执行 (truncated division remainder)
input_pt = torch.tensor(input_np)
out_pt = torch.fmod(input_pt, other)

# TensorFlow 执行 (floor division remainder)
input_tf = tf.constant(input_np)
out_tf = tf.math.floormod(input_tf, other)

# 比较结果
pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
max_diff = np.max(np.abs(pt_np_out - tf_np_out))
print(f"PyTorch fmod output: {pt_np_out.flatten()}")
print(f"TensorFlow floormod output: {tf_np_out.flatten()}")
print(f"Maximum difference: {max_diff}")
# 例: fmod(0.27, -3.7) = 0.27, floormod(0.27, -3.7) = -3.43
# 实测最大差异: 3.7
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.math.floormod] Output difference anomaly under equivalent migration in fmod operator (sample6)

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

When mapping `torch.fmod(input, -3.7)` to `tf.math.floormod(input, -3.7)` with input shape=[4, 1] and dtype=float64, the outputs differ by up to 3.7. For positive inputs with a negative divisor, `torch.fmod` produces a positive result (sign follows dividend), while `tf.math.floormod` produces a negative result (sign follows divisor).

Expected behavior: The migration tool should use `tf.math.truncatemod` or equivalent instead of `tf.math.floormod` to match `torch.fmod` truncated division semantics.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

input_data = [[-0.49545063101844433], [0.27046169092537087],
              [1.5556632190621569], [0.13487971268907317]]
input_np = np.array(input_data, dtype=np.float64)
other = -3.7

input_pt = torch.tensor(input_np)
out_pt = torch.fmod(input_pt, other)

input_tf = tf.constant(input_np)
out_tf = tf.math.floormod(input_tf, other)

max_diff = np.max(np.abs(out_pt.numpy() - out_tf.numpy()))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 3.7
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 3.7
```

## Issue 013（同9）

llm_enhanced_torch_fmod_20251215_201658.json_sample7.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][fmod] Output difference anomaly under equivalent migration in fmod operator (sample7)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a severe output discrepancy was detected for the `torch.fmod` operator (Maximum difference: 3.7).

Same root cause as other fmod issues: `torch.fmod` (truncated remainder) vs `tf.math.floormod` (floor remainder) semantic mismatch. With a positive divisor of 3.7, negative input values produce different-sign remainders.

- Input: shape=[2, 3, 1], dtype=float64
- Other: scalar 3.7

```python
import numpy as np
import torch
import tensorflow as tf

# 使用原始样本数据
input_data = [[[-1.0119007581356982]], [[0.45037909433016066]], [[1.0490606706325265]],
              [[2.1848027911224572]], [[0.9355703202929162]], [[0.356326455897288]]]
input_np = np.array(input_data, dtype=np.float64).reshape(2, 3, 1)
other = 3.7

# PyTorch 执行 (truncated division remainder)
input_pt = torch.tensor(input_np)
out_pt = torch.fmod(input_pt, other)

# TensorFlow 执行 (floor division remainder)
input_tf = tf.constant(input_np)
out_tf = tf.math.floormod(input_tf, other)

# 比较结果
pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
max_diff = np.max(np.abs(pt_np_out - tf_np_out))
print(f"PyTorch fmod output: {pt_np_out.flatten()}")
print(f"TensorFlow floormod output: {tf_np_out.flatten()}")
print(f"Maximum difference: {max_diff}")
# 例: fmod(-1.012, 3.7) = -1.012, floormod(-1.012, 3.7) = 2.688
# 实测最大差异: 3.7
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.math.floormod] Output difference anomaly under equivalent migration in fmod operator (sample7)

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

When mapping `torch.fmod(input, 3.7)` to `tf.math.floormod(input, 3.7)` with input shape=[2, 3, 1] containing negative values, the outputs differ by up to 3.7. For example, `torch.fmod(-1.012, 3.7) = -1.012` while `tf.math.floormod(-1.012, 3.7) = 2.688`. This is caused by the fundamental semantic difference between truncated and floor division remainder operations.

Expected behavior: The migration tool should use `tf.math.truncatemod` or equivalent instead of `tf.math.floormod` to preserve `torch.fmod` semantics.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

input_data = [[[-1.0119007581356982]], [[0.45037909433016066]], [[1.0490606706325265]],
              [[2.1848027911224572]], [[0.9355703202929162]], [[0.356326455897288]]]
input_np = np.array(input_data, dtype=np.float64).reshape(2, 3, 1)
other = 3.7

input_pt = torch.tensor(input_np)
out_pt = torch.fmod(input_pt, other)

input_tf = tf.constant(input_np)
out_tf = tf.math.floormod(input_tf, other)

max_diff = np.max(np.abs(out_pt.numpy() - out_tf.numpy()))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 3.7
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 3.7
```

## Issue 014

llm_enhanced_torch_logdet_20251215_203929.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][logdet] Output difference anomaly under equivalent migration in logdet operator (sample1)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, an output discrepancy was detected for the `torch.logdet` operator. The mismatch has been flagged as inconsistent by multiple reviewers.

The root cause stems from how the two frameworks handle non-symmetric or randomly generated matrices. `torch.logdet` calculates the log determinant successfully if the determinant is positive (likely using LU decomposition), returning `NaN` only when the determinant is negative or zero. In contrast, TensorFlow's `tf.linalg.logdet` assumes symmetric positive-definite input (internally utilizing Cholesky decomposition) and returns `NaN` even for matrices with a valid positive determinant that do not meet its stricter requirements.

For the same generated input tensor of shape `[2, 3, 3]`, PyTorch successfully computes the `logdet` for the first matrix and returns `NaN` for the second, while TensorFlow returns `NaN` for both.

```python
import numpy as np
import torch
import tensorflow as tf

# 使用原始样本数据构造输入矩阵
input_data = np.array([
    [[1.3255056142807007, 1.6319661140441895, -0.31763410568237305],
     [-0.7044814825057983, -0.11586584150791168, -0.595336377620697],
     [-1.935481071472168, -0.11229047179222107, 0.2102297693490982]],
    [[-0.3306174874305725, 0.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0]]
], dtype=np.float32)
# 注意: 第二个矩阵用零填充以演示，实际使用随机数据
np.random.seed(42)
input_np = np.random.randn(2, 3, 3).astype(np.float32)

# PyTorch 执行
input_pt = torch.tensor(input_np)
out_pt = torch.logdet(input_pt)

# TensorFlow 执行
input_tf = tf.constant(input_np)
out_tf = tf.linalg.logdet(input_tf)

# 比较结果
pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
max_diff = np.max(np.abs(pt_np_out - tf_np_out))
print(f"PyTorch logdet output: {pt_np_out}")
print(f"TensorFlow logdet output: {tf_np_out}")
print(f"Maximum difference: {max_diff}")
# 注意: 随机矩阵行列式可能为负，log(负数) 在实数域无定义
# 两框架对此边界情况的处理方式不同
# 实测最大差异: nan
PyTorch logdet output: [0.08860854        nan]
TensorFlow logdet output: [nan nan]
Maximum difference: nan
```

**Versions***

同1

### TensorFlow Issue（√）

**Add a title*** 

[PyTorch -> TensorFlow][tf.linalg.logdet] Output difference anomaly under equivalent migration in logdet operator

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

When mapping `torch.logdet` to `tf.linalg.logdet` during cross-framework migration testing with random 3x3 matrices (batch shape=`[2, 3, 3]`, dtype=`float32`), the outputs differ. For identical input matrices, PyTorch evaluates to `[0.08860854 nan]` while TensorFlow evaluates to `[nan nan]`, resulting in a maximum difference of `nan`.

The root cause is that `torch.logdet` successfully processes non-symmetric matrices that have a valid positive determinant (using LU decomposition), failing only when `det(A) <= 0`. Conversely, `tf.linalg.logdet` assumes symmetric positive-definite input (internally utilizing Cholesky decomposition). As a result, it returns `NaN` for matrices that do not meet this strict condition, even if their determinants are mathematically positive.

Expected behavior: `tf.linalg.logdet` should either compute the log determinant for valid matrices with positive determinants (aligning with `torch.logdet`), or clearly raise an exception rather than silently outputting `NaN` for non-symmetric but valid inputs.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(2, 3, 3).astype(np.float32)

input_pt = torch.tensor(input_np)
out_pt = torch.logdet(input_pt)

input_tf = tf.constant(input_np)
out_tf = tf.linalg.logdet(input_tf)

pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
max_diff = np.max(np.abs(pt_np_out - tf_np_out))
print(f"PyTorch logdet: {pt_np_out}")
print(f"TensorFlow logdet: {tf_np_out}")
print(f"Maximum difference: {max_diff}")
# 实测最大差异: nan
```

**Relevant log output**

```
PyTorch logdet output: [0.08860854        nan]
TensorFlow logdet output: [nan nan]
Maximum difference: nan
```

## Issue 015（同14）

llm_enhanced_torch_logdet_20251215_203929.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][logdet] Output difference anomaly under equivalent migration in logdet operator

**🐛 Describe the bug*** 

During cross-framework testing, we identified a discrepancy between PyTorch's `torch.logdet` and TensorFlow's `tf.linalg.logdet` when processing a specific 3x3 `float64` matrix.

Despite the matrix having a positive determinant (`0.2633489202527668`), the frameworks return vastly different results. PyTorch successfully computes the log determinant (`-1.3342754331075959`), presumably relying on LU decomposition which can handle non-symmetric matrices with positive determinants. Conversely, TensorFlow returns `NaN`. This indicates a significant mathematical divergence when translating operations for non-symmetric positive-determinant matrices.

```python
import numpy as np
import torch
import tensorflow as tf

input_data = [
    [-0.44332579318682674, -0.14522185536296073, 0.14717903848811734],
    [0.08460438987005668, -0.6951234296799749, 0.24125915763781158],
    [0.34045975782091975, 0.6165841633753035, 0.5205773942639326]
]
input_np = np.array(input_data, dtype=np.float64)

input_pt = torch.tensor(input_np)
out_pt = torch.logdet(input_pt)

input_tf = tf.constant(input_np)
out_tf = tf.linalg.logdet(input_tf)

pt_val = out_pt.numpy()
tf_val = out_tf.numpy()
det_val = np.linalg.det(input_np)
print(f"Matrix determinant: {det_val}")
print(f"PyTorch logdet output: {pt_val}")
print(f"TensorFlow logdet output: {tf_val}")
print(f"Difference: {np.abs(pt_val - tf_val)}")
# output:
Matrix determinant: 0.2633489202527668
PyTorch logdet output: -1.3342754331075959
TensorFlow logdet output: nan
Difference: nan
```

**Versions***

同1

### TensorFlow Issue(√)

**Add a title*** 

tf.linalg.logdet returns NaN for 3x3 float64 matrix with valid positive determinant compared to PyTorch

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

When calculating `tf.linalg.logdet` on a 3x3 `float64` matrix with a strictly positive determinant (`0.2633...`), TensorFlow returns `NaN`. In contrast, PyTorch's `torch.logdet` evaluates this correctly to `-1.334275...`.

The discrepancy occurs because `tf.linalg.logdet` relies on Cholesky decomposition, strictly assuming a symmetric positive-definite input matrix. When a matrix has a positive determinant but fails the SPD criteria, TensorFlow silently outputs `NaN`.

Expected behavior: TensorFlow should ideally compute the log determinant correctly for matrices with positive determinants (similar to PyTorch's LU-based implementation) or raise an explicit `InvalidArgumentError` instead of propagating `NaN`s silently, which breaks downstream cross-framework migration validation.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

input_data = [
    [-0.44332579318682674, -0.14522185536296073, 0.14717903848811734],
    [0.08460438987005668, -0.6951234296799749, 0.24125915763781158],
    [0.34045975782091975, 0.6165841633753035, 0.5205773942639326]
]
input_np = np.array(input_data, dtype=np.float64)

input_pt = torch.tensor(input_np)
out_pt = torch.logdet(input_pt)

input_tf = tf.constant(input_np)
out_tf = tf.linalg.logdet(input_tf)

pt_val = out_pt.numpy()
tf_val = out_tf.numpy()
det_val = np.linalg.det(input_np)
print(f"Matrix determinant: {det_val}")
print(f"PyTorch logdet output: {pt_val}")
print(f"TensorFlow logdet output: {tf_val}")
print(f"Difference: {np.abs(pt_val - tf_val)}")
# output:
Matrix determinant: 0.2633489202527668
PyTorch logdet output: -1.3342754331075959
TensorFlow logdet output: nan
Difference: nan
```

## Issue 016（同14）

llm_enhanced_torch_logdet_20251215_203929.json_sample3.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][logdet] Output difference anomaly under equivalent migration in logdet operator

**🐛 Describe the bug*** 

We observed a severe NaN-masking inconsistency between `torch.logdet` and TensorFlow when processing 4D batched matrices (`shape=[4, 4, 2, 2]`, `float32`).

For a batch containing 9 matrices with non-positive determinants, PyTorch handles the valid matrices, computing float values while naturally returning `NaN` only for the non-positive boundaries. TensorFlow, however, returns significantly more `NaN` values across the batch tensor, leading to an output mismatch where the maximum difference is `nan`. This discrepancy in handling batched LU factorization vs Cholesky decomposition creates unpredictable numerical mappings in translation pipelines.

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(4, 4, 2, 2).astype(np.float32)

input_pt = torch.tensor(input_np)
out_pt = torch.logdet(input_pt)

input_tf = tf.constant(input_np)
out_tf = tf.linalg.logdet(input_tf)

pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
max_diff = np.max(np.abs(pt_np_out - tf_np_out))
neg_det_count = np.sum(np.linalg.det(input_np) <= 0)
print(f"Number of matrices with non-positive determinant: {neg_det_count}")
print(f"PyTorch logdet output: {pt_np_out}")
print(f"TensorFlow logdet output: {tf_np_out}")
print(f"Maximum difference: {max_diff}")
# output:
Number of matrices with non-positive determinant: 9
PyTorch logdet output: [[-0.16716182 -1.6604435  -0.7548522          nan]
 [ 0.5398615          nan         nan         nan]
 [-0.12048236         nan         nan         nan]
 [-0.8231379          nan         nan -1.0002422 ]]
TensorFlow logdet output: [[-1.0876427        nan        nan        nan]
 [       nan        nan        nan        nan]
 [       nan        nan        nan        nan]
 [       nan        nan        nan        nan]]
Maximum difference: nan
```

**Versions***

同1

### TensorFlow Issue(√)

**Add a title*** 

Batched tf.linalg.logdet produces inconsistent NaN mask on 2x2 float32 matrices vs PyTorch

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

When executing `tf.linalg.logdet` on batched 2x2 float32 tensors (shape `[4, 4, 2, 2]`), the function resolves almost the entire batch to `NaN`, heavily mismatching PyTorch's output. While both frameworks properly fail on the 9 internal matrices that possess non-positive determinants, TensorFlow additionally outputs `NaN` for matrices that PyTorch evaluates mathematically without issue.

This happens because TensorFlow imposes strict symmetric positive-definite constraints on the entire batch. If matrices within the batch are merely non-symmetric (yet possess valid positive determinants), TF's batched Cholesky solver fails and outputs `NaN`.

Expected behavior: For matrices with positive determinants, `tf.linalg.logdet` should successfully compute the scalar values akin to PyTorch, rather than populating the output tensor with unpredictable `NaN` evaluations.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(4, 4, 2, 2).astype(np.float32)

input_pt = torch.tensor(input_np)
out_pt = torch.logdet(input_pt)

input_tf = tf.constant(input_np)
out_tf = tf.linalg.logdet(input_tf)

pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
max_diff = np.max(np.abs(pt_np_out - tf_np_out))
neg_det_count = np.sum(np.linalg.det(input_np) <= 0)
print(f"Number of matrices with non-positive determinant: {neg_det_count}")
print(f"PyTorch logdet output: {pt_np_out}")
print(f"TensorFlow logdet output: {tf_np_out}")
print(f"Maximum difference: {max_diff}")
# output:
Number of matrices with non-positive determinant: 9
PyTorch logdet output: [[-0.16716182 -1.6604435  -0.7548522          nan]
 [ 0.5398615          nan         nan         nan]
 [-0.12048236         nan         nan         nan]
 [-0.8231379          nan         nan -1.0002422 ]]
TensorFlow logdet output: [[-1.0876427        nan        nan        nan]
 [       nan        nan        nan        nan]
 [       nan        nan        nan        nan]
 [       nan        nan        nan        nan]]
Maximum difference: nan
```

## Issue 017（同14）

llm_enhanced_torch_logdet_20251215_203929.json_sample4.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][logdet] Output difference anomaly under equivalent migration in logdet operator (sample4)

**🐛 Describe the bug*** 

- During an automated equivalence evaluation between `torch.logdet` and TensorFlow on multi-dimensional batched 4x4 inputs (`shape=[3, 3, 4, 4]`, `dtype=float64`), we identified divergent validation boundaries resulting in numerical collapse on the TensorFlow side.

  PyTorch successfully evaluates several randomly generated 4x4 sub-matrices within the batch, accurately computing results like `0.99923636` and `1.9053382`. However, TensorFlow outputs entirely `NaN` across the same target slice. This stems from PyTorch's robust LU-decomposition implementation which continues execution for valid non-symmetric inputs, while TensorFlow's underlying routine fails completely.

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(3, 3, 4, 4).astype(np.float64)

input_pt = torch.tensor(input_np)
out_pt = torch.logdet(input_pt)

input_tf = tf.constant(input_np)
out_tf = tf.linalg.logdet(input_tf)

pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
max_diff = np.max(np.abs(pt_np_out - tf_np_out))
neg_det_count = np.sum(np.linalg.det(input_np) <= 0)
print(f"Number of matrices with non-positive determinant: {neg_det_count}")
print(f"PyTorch logdet output: {pt_np_out}")
print(f"TensorFlow logdet output: {tf_np_out}")
print(f"Maximum difference: {max_diff}")
# output:
Number of matrices with non-positive determinant: 5
PyTorch logdet output: [[        nan         nan         nan]
 [        nan         nan  0.99923636]
 [-0.94512092  1.10761909  1.9053382 ]]
TensorFlow logdet output: [[nan nan nan]
 [nan nan nan]
 [nan nan nan]]
Maximum difference: nan
```

**Versions***

同1

### TensorFlow Issue(√)

**Add a title*** 

tf.linalg.logdet aggressively outputs NaNs on batched 4x4 float64 inputs during framework translation

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

When operating on a `[3, 3, 4, 4]` batch of `float64` random matrices, `tf.linalg.logdet` resolves entirely to `NaN` in multidimensional blocks, failing to compute valid determinants. PyTorch evaluates the same inputs and correctly provides float responses for the valid matrices.

This behavior originates from TensorFlow's architectural requirement of symmetric positive-definiteness for `logdet` via Cholesky decomposition. Because randomly generated batched 4x4 matrices fail the SPD criteria (despite containing valid positive determinants in some elements), TF uniformly blankets the output with `NaN`.

Expected behavior: `tf.linalg.logdet` should align with standard LU-decomposition functionality for general non-symmetric positive-determinant inputs. Alternatively, it should throw an explicit constraint exception instead of silently invalidating batched output spaces with `NaN` arrays.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

np.random.seed(42)
input_np = np.random.randn(3, 3, 4, 4).astype(np.float64)

input_pt = torch.tensor(input_np)
out_pt = torch.logdet(input_pt)

input_tf = tf.constant(input_np)
out_tf = tf.linalg.logdet(input_tf)

pt_np_out = out_pt.numpy()
tf_np_out = out_tf.numpy()
max_diff = np.max(np.abs(pt_np_out - tf_np_out))
neg_det_count = np.sum(np.linalg.det(input_np) <= 0)
print(f"Number of matrices with non-positive determinant: {neg_det_count}")
print(f"PyTorch logdet output: {pt_np_out}")
print(f"TensorFlow logdet output: {tf_np_out}")
print(f"Maximum difference: {max_diff}")
# output:
Number of matrices with non-positive determinant: 5
PyTorch logdet output: [[        nan         nan         nan]
 [        nan         nan  0.99923636]
 [-0.94512092  1.10761909  1.9053382 ]]
TensorFlow logdet output: [[nan nan nan]
 [nan nan nan]
 [nan nan nan]]
Maximum difference: nan
```

## Issue 018（base=2参数没传，传了就对了）

llm_enhanced_torch_logspace_20251215_203651.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> TensorFlow][logspace] Output difference anomaly under equivalent migration in logspace operator

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and TensorFlow, a severe output discrepancy was detected for the `torch.logspace` operator (Maximum difference: 1014.0). The mismatch has been flagged as inconsistent by multiple reviewers.

The root cause is a parameter mapping defect: `torch.logspace` accepts a `base` parameter (set to 2 in this test), generating values as $2^x$. However, `tf.experimental.numpy.logspace` defaults to `base=10` and the migration code fails to pass the `base=2` parameter, generating values as $10^x$ instead. This produces dramatically different output sequences.

- Parameters: start=-10, end=10, steps=5, base=2 (PyTorch) vs base=10 (TensorFlow default)
- Maximum difference: 1014.0

```python
import numpy as np
import torch
import tensorflow as tf

# PyTorch 执行 (base=2)
out_pt = torch.logspace(-10, 10, 5, base=2)

# TensorFlow 执行 (默认 base=10，缺少 base=2 参数映射)
out_tf = tf.experimental.numpy.logspace(-10, 10, 5)

# 比较结果
pt_np = out_pt.numpy().astype(np.float64)
tf_np = np.array(out_tf).astype(np.float64)
max_diff = np.max(np.abs(pt_np - tf_np))
print(f"PyTorch logspace (base=2): {pt_np}")
print(f"TensorFlow logspace (base=10): {tf_np}")
print(f"Maximum difference: {max_diff}")
# PyTorch: [2^-10, 2^-5, 2^0, 2^5, 2^10] = [0.000977, 0.03125, 1.0, 32.0, 1024.0]
# TensorFlow: [10^-10, 10^-5, 10^0, 10^5, 10^10] = [1e-10, 1e-5, 1.0, 1e5, 1e10]
# 实测最大差异: 1014.0
```

**Versions***

同1

### TensorFlow Issue

**Add a title*** 

[PyTorch -> TensorFlow][tf.experimental.numpy.logspace] Output difference anomaly under equivalent migration in logspace operator

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

When mapping `torch.logspace(start=-10, end=10, steps=5, base=2)` to `tf.experimental.numpy.logspace(start=-10, stop=10, num=5)`, the migration code fails to pass the `base=2` parameter. `tf.experimental.numpy.logspace` defaults to `base=10`, producing a completely different sequence of values. Maximum difference: 1014.0.

- PyTorch (base=2): [0.000977, 0.03125, 1.0, 32.0, 1024.0]
- TensorFlow (base=10): [1e-10, 1e-5, 1.0, 1e5, 1e10]

Expected behavior: The migration tool should pass the `base` parameter to `tf.experimental.numpy.logspace(start, stop, num, base=2)` to match PyTorch's `torch.logspace` semantics.

**Standalone code to reproduce the issue***

```python
import numpy as np
import torch
import tensorflow as tf

out_pt = torch.logspace(-10, 10, 5, base=2)
out_tf = tf.experimental.numpy.logspace(-10, 10, 5)  # 缺少 base=2

pt_np = out_pt.numpy().astype(np.float64)
tf_np = np.array(out_tf).astype(np.float64)
max_diff = np.max(np.abs(pt_np - tf_np))
print(f"PyTorch (base=2): {pt_np}")
print(f"TensorFlow (base=10): {tf_np}")
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 1014.0
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 1014.0
```

## Issue 019（输出含nan，实际输出一致）

llm_enhanced_torch_matmul_20251216_013733.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][matmul] Output difference anomaly under equivalent migration in matmul operator with zero-dimension input

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, a NaN-based output discrepancy was detected for the `torch.matmul` operator when operating on zero-dimension tensors (Maximum difference: nan).

The input tensors have shapes [4096, 0] and [0, 4096], resulting in a matrix multiplication where the inner dimension is 0. PyTorch computes this as a [4096, 4096] all-zero matrix (sum over empty dimension yields 0). MindSpore's `mint.matmul` may handle zero-dimension inputs differently, potentially producing NaN values or raising errors.

- Input: shape=[4096, 0], dtype=float32
- Other: shape=[0, 4096], dtype=float32
- Maximum difference: nan

```python
import numpy as np
import torch
import mindspore

# 构造零维输入
input_np = np.empty((4096, 0), dtype=np.float32)
other_np = np.empty((0, 4096), dtype=np.float32)

# PyTorch 执行 —— 结果为全零矩阵 [4096, 4096]
input_pt = torch.tensor(input_np)
other_pt = torch.tensor(other_np)
out_pt = torch.matmul(input_pt, other_pt)

# MindSpore 执行
input_ms = mindspore.Tensor(input_np)
other_ms = mindspore.Tensor(other_np)
try:
    out_ms = mindspore.mint.matmul(input_ms, other_ms)
    ms_np = out_ms.asnumpy()
except Exception as e:
    print(f"MindSpore error: {e}")
    ms_np = None

# 比较结果
pt_np_out = out_pt.numpy()
print(f"PyTorch output shape: {pt_np_out.shape}")
print(f"PyTorch output (all zeros): min={pt_np_out.min()}, max={pt_np_out.max()}")
if ms_np is not None:
    max_diff = np.max(np.abs(pt_np_out.astype(np.float64) - ms_np.astype(np.float64)))
    print(f"MindSpore output shape: {ms_np.shape}")
    print(f"Maximum difference: {max_diff}")
# 实测最大差异: nan
```

**Versions***

同1

### MindSpore Issue

**Title***

 [PyTorch -> MindSpore][mindspore.mint.matmul] Output difference anomaly under equivalent migration in matmul operator with zero-dimension input

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

When performing matrix multiplication with zero-dimension tensors (shape=[4096, 0] @ [0, 4096]), PyTorch produces a [4096, 4096] all-zero matrix. MindSpore's `mindspore.mint.matmul` handles this edge case differently, producing NaN values or raising an error (Maximum difference: nan).

**Describe the expected behavior***

Both frameworks should produce consistent results for zero-dimension matrix multiplication. Mathematically, when the inner dimension is 0, the result should be a zero matrix (sum over an empty set is 0).

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

input_np = np.empty((4096, 0), dtype=np.float32)
other_np = np.empty((0, 4096), dtype=np.float32)

input_pt = torch.tensor(input_np)
other_pt = torch.tensor(other_np)
out_pt = torch.matmul(input_pt, other_pt)

input_ms = mindspore.Tensor(input_np)
other_ms = mindspore.Tensor(other_np)
out_ms = mindspore.mint.matmul(input_ms, other_ms)

print(f"PyTorch output: shape={out_pt.shape}, values all zero={torch.all(out_pt == 0).item()}")
print(f"MindSpore output: shape={out_ms.shape}")
max_diff = np.max(np.abs(out_pt.numpy().astype(np.float64) - out_ms.asnumpy().astype(np.float64)))
print(f"Maximum difference: {max_diff}")
# 实测最大差异: nan
```

**Related log / screenshot**

```
comparison_error: Numerical mismatch, maximum difference: nan
```

**Special notes for this issue**

Zero-dimension tensor multiplication is a valid edge case. PyTorch handles it by returning a zero matrix. MindSpore's handling of this boundary condition should be verified and documented.

## Issue 020

llm_enhanced_torch_mean_20251215_184512.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> MindSpore][mean] Output difference anomaly under equivalent migration in mean operator with empty tensor

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, an output discrepancy was detected for the `torch.mean` operator when applied to an empty tensor.

The input tensor has shape `[2, 0, 4, 5, 6]` with a zero-size dimension, making it an empty tensor with no elements. PyTorch's `torch.mean` returns `nan` for empty tensors (consistent with IEEE 754: 0/0 = NaN). However, MindSpore's `mint.mean` evaluates this to `inf`. This is a fundamental behavioral difference for empty tensor edge cases, leading to a maximum difference of `nan` during framework alignment checks.

```python
import numpy as np
import torch
import mindspore

# 构造空张量（含零维度）
input_np = np.empty((2, 0, 4, 5, 6), dtype=np.float64)

# PyTorch 执行 —— 返回 NaN（空张量无元素可计算均值）
input_pt = torch.tensor(input_np)
out_pt = torch.mean(input_pt)

# MindSpore 执行
input_ms = mindspore.Tensor(input_np)
try:
    out_ms = mindspore.mint.mean(input_ms)
    ms_val = out_ms.asnumpy()
except Exception as e:
    print(f"MindSpore error: {e}")
    ms_val = None

# 比较结果
pt_val = out_pt.numpy()
print(f"Input shape: {input_np.shape}, total elements: {input_np.size}")
print(f"PyTorch mean output: {pt_val}")
if ms_val is not None:
    print(f"MindSpore mean output: {ms_val}")
    print(f"Difference: {np.abs(pt_val - ms_val)}")
# PyTorch 返回 nan（IEEE 754: 0/0），MindSpore 行为不同
# 实测最大差异: nan
PyTorch mean: nan
MindSpore mean: inf
```

**Versions***

同1

### MindSpore Issue（√）

**Title***

 [PyTorch -> MindSpore][mindspore.mint.mean] Output difference anomaly under equivalent migration in mean operator with empty tensor

Environment

Hardware Environment(`Ascend`/`GPU`/`CPU`):

/device cpu

Software Environment:

- **MindSpore version (source or binary)**: 2.8.0 (binary)
- **Python version (e.g., Python 3.7.5)**: 3.10.18
- **OS platform and distribution (e.g., Linux Ubuntu 16.04)**: Windows 11
- **GCC/Compiler version (if compiled from source)**: N/A

Describe the current behavior

When computing `mindspore.mint.mean` on an empty tensor with shape `[2, 0, 4, 5, 6]` (zero elements), MindSpore returns `inf`. In equivalent cross-framework testing, PyTorch's `torch.mean` returns `nan` (following the IEEE 754 convention: mean of zero elements = 0/0 = NaN). This inconsistency results in an evaluation mismatch where the maximum difference is `nan`.

Describe the expected behavior

Both frameworks should produce consistent behavior for empty tensor mean operations. The mathematically expected result for the mean of zero elements is `nan` (undefined: 0/0), following the IEEE 754 convention.

Steps to reproduce the issue

1. Execute the following equivalent migration test script:

Python

```
import numpy as np
import torch
import mindspore

input_np = np.empty((2, 0, 4, 5, 6), dtype=np.float64)

input_pt = torch.tensor(input_np)
out_pt = torch.mean(input_pt)

input_ms = mindspore.Tensor(input_np)
out_ms = mindspore.mint.mean(input_ms)

print(f"PyTorch mean: {out_pt.numpy()}")
print(f"MindSpore mean: {out_ms.asnumpy()}")
```

1. Observe the output discrepancy between frameworks.

Related log / screenshot

Plaintext

```
PyTorch mean: nan
MindSpore mean: inf
comparison_error: Numerical mismatch, maximum difference: nan
```

Special notes for this issue

Empty tensor mean is a specific boundary case. PyTorch handles this by complying with IEEE 754 specifications (`0/0 = NaN`). MindSpore's generation of `inf` for an empty tensor reduction deviates from this standard and breaks cross-framework alignment tests. Expected to unify or document reduction logic for zero-element tensors.

## Issue 021（同20）

llm_enhanced_torch_mean_20251215_184512.json_sample2.txt

### PyTorch Issue

**Title***

PyTorch -> MindSpore Output difference anomaly under equivalent migration in mean operator with 2D empty tensor [0, 5]

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, an output discrepancy was detected for the `torch.mean` operator when applied to a 2D empty tensor with shape `[0, 5]` and `float32` datatype.

Specifically, for a matrix with zero rows, PyTorch's `torch.mean` correctly returns `nan` (as the mean of zero elements evaluates to 0/0 = NaN under IEEE 754). However, MindSpore's `mint.mean` incorrectly evaluates this to `inf`. This creates a fundamental behavioral mismatch for 2D empty dimensional cases, resulting in a maximum difference of `nan` during framework alignment checks.

```python
import numpy as np
import torch
import mindspore

input_np = np.empty((0, 5), dtype=np.float32)

input_pt = torch.tensor(input_np)
out_pt = torch.mean(input_pt)

input_ms = mindspore.Tensor(input_np)
try:
    out_ms = mindspore.mint.mean(input_ms)
    ms_val = out_ms.asnumpy()
except Exception as e:
    print(f"MindSpore error: {e}")
    ms_val = None

pt_val = out_pt.numpy()
print(f"Input shape: {input_np.shape}, total elements: {input_np.size}")
print(f"PyTorch mean: {pt_val}")
if ms_val is not None:
    print(f"MindSpore mean: {ms_val}")
# output:
Input shape: (0, 5), total elements: 0
PyTorch mean: nan
MindSpore mean: inf
```

**Versions***

同1

### MindSpore Issue（√）

**Title***

PyTorch -> MindSpore Output difference anomaly under equivalent migration in mean operator with 2D empty tensor [0, 5]

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.8.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

When computing `mindspore.mint.mean` on a 2D empty tensor with shape `[0, 5]` and `float32` datatype (total 0 elements), MindSpore evaluates the result to `inf`. In equivalent cross-framework testing, PyTorch's `torch.mean` returns `nan`. This discrepancy in handling matrices with zero rows causes a maximum difference of `nan` in output verification.

**Describe the expected behavior***

Both frameworks should produce consistent behavior for 2D empty tensor mean operations. Following the IEEE 754 convention, the mathematical result for the mean of an empty set (0 elements) is undefined (0/0) and should evaluate to `nan`, not `inf`.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

input_np = np.empty((0, 5), dtype=np.float32)

input_pt = torch.tensor(input_np)
out_pt = torch.mean(input_pt)

input_ms = mindspore.Tensor(input_np)
try:
    out_ms = mindspore.mint.mean(input_ms)
    ms_val = out_ms.asnumpy()
except Exception as e:
    print(f"MindSpore error: {e}")
    ms_val = None

pt_val = out_pt.numpy()
print(f"Input shape: {input_np.shape}, total elements: {input_np.size}")
print(f"PyTorch mean: {pt_val}")
if ms_val is not None:
    print(f"MindSpore mean: {ms_val}")
```

**Related log / screenshot**

```
Input shape: (0, 5), total elements: 0
PyTorch mean: nan
MindSpore mean: inf
```

**Special notes for this issue**

This highlights an inconsistency specific to 2D empty matrices (`float32`), frequently encountered when filtering logic results in zero valid rows but preserves feature dimensions. PyTorch strictly adheres to IEEE 754 (`nan`), whereas MindSpore yields `inf`.

## Issue 022（同20）

llm_enhanced_torch_mean_20251215_184512.json_sample5.txt

### PyTorch Issue

**Title***

PyTorch -> MindSpore Output difference anomaly under equivalent migration in mean operator with batched empty tensor [0, 64]

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, a NaN-to-Inf output discrepancy was detected for the `torch.mean` operator on a typical batched empty tensor with shape `[0, 64]`.

This specific shape represents a common edge case where the batch dimension becomes zero (e.g., after a strict filtering operation) while the feature dimension (`64`) is retained. PyTorch robustly handles this by returning `nan` (0 elements divided by 0), whereas MindSpore unexpectedly evaluates this zero-element reduction to `inf`. The equivalent migration test fails with a maximum difference of `nan`.

```python
import numpy as np
import torch
import mindspore

input_np = np.empty((0, 64), dtype=np.float32)

input_pt = torch.tensor(input_np)
out_pt = torch.mean(input_pt)

input_ms = mindspore.Tensor(input_np)
try:
    out_ms = mindspore.mint.mean(input_ms)
    ms_val = out_ms.asnumpy()
except Exception as e:
    print(f"MindSpore error: {e}")
    ms_val = None

pt_val = out_pt.numpy()
print(f"Input shape: {input_np.shape}, total elements: {input_np.size}")
print(f"PyTorch mean: {pt_val}")
if ms_val is not None:
    print(f"MindSpore mean: {ms_val}")
# output:
Input shape: (0, 64), total elements: 0
PyTorch mean: nan
MindSpore mean: inf
```

**Versions***

同1

### MindSpore Issue（√）

**Title***

PyTorch -> MindSpore Output difference anomaly under equivalent migration in mean operator with batched empty tensor [0, 64]

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:

- **MindSpore version (source or binary)**: 2.8.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

When executing `mindspore.mint.mean` on a batched empty tensor with shape `[0, 64]` (a typical shape when batch size is 0 but feature channels are 64), MindSpore computes the mean as `inf`. Correspondingly, PyTorch's `torch.mean` calculates the empty tensor mean as `nan`. This causes cross-framework parity tests to fail with a maximum difference of `nan`.

**Describe the expected behavior***

MindSpore should output `nan` for global mean reductions over empty batched tensors, aligning with PyTorch's behavior and IEEE 754 standards (0/0 = NaN).

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

input_np = np.empty((0, 64), dtype=np.float32)

input_pt = torch.tensor(input_np)
out_pt = torch.mean(input_pt)

input_ms = mindspore.Tensor(input_np)
try:
    out_ms = mindspore.mint.mean(input_ms)
    ms_val = out_ms.asnumpy()
except Exception as e:
    print(f"MindSpore error: {e}")
    ms_val = None

pt_val = out_pt.numpy()
print(f"Input shape: {input_np.shape}, total elements: {input_np.size}")
print(f"PyTorch mean: {pt_val}")
if ms_val is not None:
    print(f"MindSpore mean: {ms_val}")

```

**Related log / screenshot**

```
Input shape: (0, 64), total elements: 0
PyTorch mean: nan
MindSpore mean: inf
```

**Special notes for this issue**

The `[0, 64]` shape is highly prevalent in dynamic batching scenarios where the batch size can drop to zero during inference or masked loss calculations. Addressing this anomaly is critical for seamless model migration, as returning `inf` instead of `nan` severely impacts downstream NaN-propagation and masking logic.

## Issue 023（同20）

llm_enhanced_torch_mean_20251215_184512.json_sample6.txt

### PyTorch Issue

**Title***

PyTorch -> MindSpore Output difference anomaly under equivalent migration in mean operator with zero-sample feature tensor [0, 10]

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, a critical discrepancy was detected for the `torch.mean` operator when processing a zero-sample feature tensor with shape `[0, 10]` and `float32` datatype.

In scenarios where a 10-dimensional feature representation receives an empty batch (0 samples), PyTorch's `torch.mean` correctly adheres to IEEE 754 specifications by returning `nan` (evaluating 0 elements / 0). Conversely, MindSpore's `mint.mean` yields an unexpected `inf` result. This distinct divergence in handling empty tensor reductions causes a maximum alignment difference of `nan` during equivalent migration validation.

```python
import numpy as np
import torch
import mindspore

input_np = np.empty((0, 10), dtype=np.float32)

input_pt = torch.tensor(input_np)
out_pt = torch.mean(input_pt)

input_ms = mindspore.Tensor(input_np)
try:
    out_ms = mindspore.mint.mean(input_ms)
    ms_val = out_ms.asnumpy()
except Exception as e:
    print(f"MindSpore error: {e}")
    ms_val = None

pt_val = out_pt.numpy()
print(f"Input shape: {input_np.shape}, total elements: {input_np.size}")
print(f"PyTorch mean: {pt_val}")
if ms_val is not None:
    print(f"MindSpore mean: {ms_val}")
# output:
Input shape: (0, 10), total elements: 0
PyTorch mean: nan
MindSpore mean: inf
```

**Versions***

同1

### MindSpore Issue（√）

**Title***

PyTorch -> MindSpore Output difference anomaly under equivalent migration in mean operator with zero-sample feature tensor [0, 10]

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:
- **MindSpore version (source or binary)**: 2.8.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

When executing `mindspore.mint.mean` on a `float32` empty tensor of shape `[0, 10]` (representing 0 samples across a 10-dimensional feature space), MindSpore computes the reduction as `inf`. In contrast, the equivalent PyTorch operation evaluates to `nan`. This framework inconsistency triggers a numerical mismatch with a maximum difference of `nan` during migration testing.

**Describe the expected behavior***

MindSpore should output `nan` when performing a global mean reduction on an empty tensor (0 total elements). This mathematically aligns with the IEEE 754 standard (0/0 = NaN) and matches PyTorch's established behavior.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

input_np = np.empty((0, 10), dtype=np.float32)

input_pt = torch.tensor(input_np)
out_pt = torch.mean(input_pt)

input_ms = mindspore.Tensor(input_np)
try:
    out_ms = mindspore.mint.mean(input_ms)
    ms_val = out_ms.asnumpy()
except Exception as e:
    print(f"MindSpore error: {e}")
    ms_val = None

pt_val = out_pt.numpy()
print(f"Input shape: {input_np.shape}, total elements: {input_np.size}")
print(f"PyTorch mean: {pt_val}")
if ms_val is not None:
    print(f"MindSpore mean: {ms_val}")
```

**Related log / screenshot**

```
Input shape: (0, 10), total elements: 0
PyTorch mean: nan
MindSpore mean: inf
```

**Special notes for this issue**

Same root cause as other mean-related issues: empty tensor reduction behavior inconsistency between PyTorch and MindSpore.

## Issue 024（随机采样算子，跨框架 RNG 实现不同，数值比较无意义）

llm_enhanced_torch_multinomial_20251202_012420.json_sample1.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][multinomial] Output difference anomaly under equivalent migration in multinomial operator

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, an output discrepancy was detected for the `torch.multinomial` operator (Maximum difference: 3). The mismatch has been flagged as inconsistent by multiple reviewers.

The root cause is that `multinomial` is a stochastic sampling operator. It samples indices from a probability distribution, and different frameworks use different random number generators (RNG). Even with the same seed value, the internal RNG implementations differ, producing different sampled indices. The "difference" of 3 is the difference between sampled index values, not a numerical precision issue.

```python
import numpy as np
import torch
import paddle

# 使用原始样本数据
input_data = [1.6700464487075806, 0.6655388474464417, 0.15147598087787628, 1.5376497507095337]
input_np = np.array(input_data, dtype=np.float32)

# PyTorch 执行
torch.manual_seed(42)
input_pt = torch.tensor(input_np)
out_pt = torch.multinomial(input_pt, num_samples=2, replacement=True)

# PaddlePaddle 执行
paddle.seed(42)
input_pd = paddle.to_tensor(input_np)
out_pd = paddle.multinomial(input_pd, num_samples=2, replacement=True)

# 比较结果
pt_np = out_pt.numpy()
pd_np = out_pd.numpy()
max_diff = np.max(np.abs(pt_np.astype(np.float64) - pd_np.astype(np.float64)))
print(f"Input (unnormalized probabilities): {input_np}")
print(f"PyTorch sampled indices: {pt_np}")
print(f"PaddlePaddle sampled indices: {pd_np}")
print(f"Maximum difference: {max_diff}")
# 注意: multinomial 是随机采样操作，不同框架 RNG 实现不同
# 相同种子下采样结果必然不同，差异为索引值之差
# 实测最大差异: 3
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][multinomial] Output difference anomaly under equivalent migration in multinomial operator

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，针对 `multinomial` 算子，两框架的输出结果被多位评审标记为不一致。最大差异为 3。

根本原因：`multinomial` 是一个随机采样算子，从输入概率分布中采样索引。两个框架的随机数生成器（RNG）实现不同，即使设置相同的随机种子，采样出的索引值也必然不同。差异值 3 代表采样索引的差值，并非数值精度问题。

此类随机算子不适合直接进行数值一致性比较，建议改为分布一致性验证（如多次采样后的统计分布对比）。

```python
import numpy as np
import torch
import paddle

input_data = [1.6700464487075806, 0.6655388474464417, 0.15147598087787628, 1.5376497507095337]
input_np = np.array(input_data, dtype=np.float32)

torch.manual_seed(42)
input_pt = torch.tensor(input_np)
out_pt = torch.multinomial(input_pt, num_samples=2, replacement=True)

paddle.seed(42)
input_pd = paddle.to_tensor(input_np)
out_pd = paddle.multinomial(input_pd, num_samples=2, replacement=True)

pt_np = out_pt.numpy()
pd_np = out_pd.numpy()
max_diff = np.max(np.abs(pt_np.astype(np.float64) - pd_np.astype(np.float64)))
print(f"PyTorch sampled indices: {pt_np}")
print(f"PaddlePaddle sampled indices: {pd_np}")
print(f"Maximum difference: {max_diff}")
# 实测最大差异: 3
```

**其他补充信息 Additional Supplementary Information**

- **原始 JSON 提取的算子配置特征**:

```json
{
  "api": "paddle.multinomial",
  "input": {
    "shape": [4],
    "dtype": "float32"
  },
  "num_samples": 2,
  "replacement": true
}
```

- `multinomial` 是随机采样算子，跨框架 RNG 实现不同，数值比较无意义。建议使用分布一致性验证替代数值一致性比较。

## Issue 025（参数映射和RNG不同）

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

## Issue 026（中位数定义不同）

llm_enhanced_torch_nanmedian_20251201_232713.json_sample2.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nanmedian] Output difference anomaly under equivalent migration in nanmedian operator (sample2)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, an output discrepancy was detected for the `torch.nanmedian` operator (Maximum difference: 0.1793074607849121).

The root cause is **not an axis mismatch**. When axis is not specified, both frameworks perform global reduction. The discrepancy comes from different median definitions when the number of valid elements is even: PyTorch tends to return the lower middle value, while Paddle returns the average of the two middle values.

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

# PaddlePaddle 执行 —— 全局 nanmedian（返回标量）
input_pd = paddle.to_tensor(input_np)
out_pd = paddle.nanmedian(input_pd)

# 比较结果
pt_val = out_pt.numpy()
pd_val = out_pd.numpy()
print(f"Input shape: {input_np.shape}")
print(f"PyTorch nanmedian (global): {pt_val}, shape: {pt_val.shape}")
print(f"PaddlePaddle nanmedian (global): {pd_val}, shape: {pd_val.shape}")
# 两者都做全局规约；差异来自偶数样本下中位数定义不同（lower vs average）
# 实测最大差异: 0.1793074607849121
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nanmedian] Output difference anomaly under equivalent migration in nanmedian operator (sample2)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，针对 `nanmedian` 算子，两框架输出结果被多位评审标记为不一致。最大差异为 0.1793074607849121。

根本原因**不是默认 axis 语义不一致**：`torch.nanmedian` 与 `paddle.nanmedian` 在未指定 axis 时都执行全局规约并返回标量。

该不一致来自偶数个有效元素时的中位数定义差异：PyTorch 更偏向返回 lower median（靠左中位数），Paddle 返回两个中位数的平均值。因此即使输入和 axis 完全一致，数值仍会出现差异。

```python
import numpy as np
import torch
import paddle

np.random.seed(42)
input_np = np.random.randn(2, 3, 4).astype(np.float32)

# PyTorch: 全局 nanmedian
input_pt = torch.tensor(input_np)
out_pt = torch.nanmedian(input_pt)

# Paddle: 全局 nanmedian
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

- `torch.nanmedian` 与 `paddle.nanmedian` 在无 axis 参数时都做全局规约；不一致核心在于偶数样本时的中位数定义（lower vs average），与 axis 无关。

## Issue 027（同26）

llm_enhanced_torch_nanmedian_20251201_232713.json_sample5.txt

### PyTorch Issue

**Title***

 [PyTorch -> Paddle][nanmedian] Output difference anomaly under equivalent migration in nanmedian operator (sample5)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, an output discrepancy was detected for the `torch.nanmedian` operator (Maximum difference: 0.0271770272751663).

Same root cause as Issue 026: this is not caused by axis behavior. Both frameworks compute global nanmedian when axis is omitted, but they differ in median definition for even-sized valid elements (lower median vs average median).

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

# PaddlePaddle: 全局 nanmedian
input_pd = paddle.to_tensor(input_np)
out_pd = paddle.nanmedian(input_pd)

pt_val = out_pt.numpy()
pd_val = out_pd.numpy()
print(f"Input shape: {input_np.shape}")
print(f"PyTorch nanmedian (global): {pt_val}, shape: {pt_val.shape}")
print(f"PaddlePaddle nanmedian (global): {pd_val}, shape: {pd_val.shape}")
# 实测最大差异: 0.0271770272751663
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nanmedian] Output difference anomaly under equivalent migration in nanmedian operator (sample5)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，针对 `nanmedian` 算子，两框架输出结果被多位评审标记为不一致。最大差异为 0.0271770272751663。

与 Issue 026 相同的根本原因：两框架在未指定 axis 时都计算全局 nanmedian（返回标量），但偶数有效元素下中位数定义不同，导致数值存在差异。

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

- 同 Issue 026，差异与默认 axis 无关，核心是偶数样本下中位数定义差异（lower vs average）。

## Issue 028（参数映射不对）

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

## Issue 029（输入未对齐）

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

## Issue 030（输入未对齐）

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

## Issue 031（输入未对齐）

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

## Issue 032（同31）

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

## Issue 033（同31）

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

## Issue 034（同31）

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

## Issue 035（同31）

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

## Issue 036（同31）

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

## Issue 037（同31）

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

## Issue 038（同31）

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

## Issue 039（初始化时权重未统一）

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

## Issue 040（随机算子，随机数生成器RNG不同）

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

## Issue 041（权重未同步）

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

## Issue 042（同41）

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

## Issue 043（同41）

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

## Issue 044（同41）

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

## Issue 045（输入未统一，参数不支持）

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

## Issue 046（输入未统一，参数不支持）

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

## Issue 047（实测结果一致）

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
# 实测最大差异: 0
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
# 实测最大差异: 0
```

**Related log / screenshot**

```
comparison_error: Numerical mismatch, maximum difference: 7
```

**Special notes for this issue**

The difference of 7 is an index value difference, not a pooled value difference. The two frameworks may use different index reference frames for the `return_indices` feature when padding is applied.

## Issue 048（输入未统一，参数不支持）

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
# 实测: 形状不匹配: PyTorch (2, 3, 2, 2) vs TensorFlow (2, 2, 2, 4)
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

## Issue 049（padding参数不一致）

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
# 实测最大差异: 2.0426304638385773
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
# 实测最大差异: 2.0426304638385773
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.819435715675354
```

## Issue 050（padding参数不一致）

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

## Issue 051（输入不对）

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

## Issue 052（参数未对齐）

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
# 实测最大差异:0.489917516708374
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
# 实测最大差异: 0.489917516708374
```

**Relevant log output**

```
comparison_error: Numerical mismatch, maximum difference: 0.44696375727653503
```

## Issue 053（同52）

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

## Issue 054（训练模式下会随机采样斜率）

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
# 实测最大差异: 0.03580858372151852
```

**Versions***

同1

### PaddlePaddle Issue

**Title*** 

 [PyTorch -> Paddle][nn.RReLU] Output difference anomaly under equivalent migration in RReLU operator (sample1)

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`nn.RReLU` 算子在训练模式下的输出差异为0.03580858372151852。

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
# 实测最大差异: 0.03580858372151852
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

PyTorch -> MindSpore Output difference anomaly under equivalent migration in RReLU operator (eval mode)

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and MindSpore, an output discrepancy was detected for the `torch.nn.RReLU` operator in `eval` mode. In `eval` mode, the negative slope should be fixed to `(lower+upper)/2`. However, there is a numerical mismatch between the two frameworks.

- Input: shape=[2, 3], dtype=float32
- Parameters: lower=0.1, upper=0.3
- Maximum difference: 0.05685841292142868

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(2, 3).astype(np.float32)

torch.manual_seed(42)
rrelu_pt = torch.nn.RReLU(lower=0.1, upper=0.3)
rrelu_pt.eval()
out_pt = rrelu_pt(torch.tensor(input_np))

mindspore.set_seed(42)
rrelu_ms = mindspore.nn.RReLU(lower=0.1, upper=0.3)
rrelu_ms.set_train(False)
out_ms = rrelu_ms(mindspore.Tensor(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_ms.asnumpy().astype(np.float64)))
print("Mode: eval")
print(f"Maximum difference: {max_diff}")

Mode: eval
Maximum difference: 0.011770706623792648
```

**Versions***

同1

### MindSpore Issue（√）

**Title***

PyTorch -> MindSpore Output difference anomaly under equivalent migration in RReLU operator (eval mode)

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:

- **MindSpore version (source or binary)**: 2.8.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

During cross-framework equivalent migration testing, `torch.nn.RReLU` and `mindspore.nn.RReLU` produce different outputs (Maximum difference: 0.011770706623792648) when applied to the same input (shape=[2,3], dtype=float32, lower=0.1, upper=0.3) under `eval` mode.

**Describe the expected behavior***

In `eval` mode, both frameworks should use a deterministic fixed slope of `(lower+upper)/2 = 0.2` for negative values. Therefore, both frameworks should produce completely identical outputs for the same input tensor.

**Steps to reproduce the issue***

```python
import numpy as np
import torch
import mindspore

np.random.seed(42)
input_np = np.random.randn(2, 3).astype(np.float32)

torch.manual_seed(42)
rrelu_pt = torch.nn.RReLU(lower=0.1, upper=0.3)
rrelu_pt.eval()
out_pt = rrelu_pt(torch.tensor(input_np))

mindspore.set_seed(42)
rrelu_ms = mindspore.nn.RReLU(lower=0.1, upper=0.3)
rrelu_ms.set_train(False)
out_ms = rrelu_ms(mindspore.Tensor(input_np))

max_diff = np.max(np.abs(out_pt.detach().numpy().astype(np.float64) - out_ms.asnumpy().astype(np.float64)))
print("Mode: eval")
print(f"Maximum difference: {max_diff}")

Mode: eval
Maximum difference: 0.011770706623792648
```

**Related log / screenshot**

```
Mode: eval
Maximum difference: 0.011770706623792648
```

**Special notes for this issue**

While RReLU is a stochastic operator in training mode, it is deterministic in eval mode. The numerical mismatch of `0.01177` observed here in eval mode suggests a potential algorithmic or implementation discrepancy in computing the output for the fixed slope.



## Issue 056（输入不一致）

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

## Issue 057（输入不一致）

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

## Issue 058（输入不一致）

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

## Issue 059（输入不一致）

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
# 实测最大差异: 0
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
# 实测最大差异: 0
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

## Issue 060（权重初始化和默认参数不同）

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

## Issue 061（同60）

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

## Issue 062（同60）

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

## Issue 063（输入不一致）

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

## Issue 064（输入不一致）

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

## Issue 065（输入不一致）

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

## Issue 066（输入不一致）

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

## Issue 067（输入不一致）

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
# 输入和参数均已对齐，差异 0
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

## Issue 068（输入不一致）

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

## Issue 069（输入不一致）

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

## Issue 070（输入不一致）

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

## Issue 071（padding参数不同）

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

## Issue 072（同71）

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

## Issue 073（同71）

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

## Issue 074（同71）

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

## Issue 075（eps参数未统一）

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

## Issue 076（卷积误差，已提过https://github.com/PaddlePaddle/Paddle/issues/76819）

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
- Maximum difference: 4.57763671875e-05

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

 [PyTorch -> Paddle][nn.functional.conv2d] Output difference anomaly under equivalent migration in conv2d operator

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，`conv2d` 算子的最大输出差异为 4.57763671875e-05。

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

## Issue 077（同76）

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

 [PyTorch -> Paddle][nn.functional.conv3d] Output difference anomaly under equivalent migration in conv3d operator

**🐛 Describe the bug*** 

During cross-framework equivalent migration testing between PyTorch and PaddlePaddle, a precision discrepancy was detected for `torch.nn.functional.conv3d` (Maximum difference: 9.918212890625e-05).

The issue manifests as a float32 accumulation precision error. 3D convolution involves extensive multiply-accumulate operations, and the current underlying implementation in PyTorch yields a $\sim 10^{-5}$ level precision difference compared to PaddlePaddle under identical inputs and parameters. This precision deviation requires further investigation to align the mathematical implementations.

- Input: shape=[20, 16, 10, 50, 100], dtype=float32
- Weight: shape=[33, 16, 3, 5, 2], Bias: shape=[33]
- Parameters: stride=[2,1,1], padding=[4,2,0], dilation=[1,1,1], groups=1
- Maximum difference: 9.918212890625e-05

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
# output：
Maximum difference: 9.918212890625e-05
```

**Versions***

同1

### PaddlePaddle Issue（√）

**Title*** 

 [PyTorch -> Paddle][nn.functional.conv3d] Output difference anomaly under equivalent migration in conv3d operator

**bug描述 Describe the Bug*** 

在进行 PyTorch 到 Paddle 的算子等价迁移验证时，发现在相同输入和参数下，`paddle.nn.functional.conv3d` 算子与 PyTorch 对应算子存在明显的精度误差，最大输出差异达到 9.918212890625e-05。

该问题表现为浮点累加精度误差。3D 卷积涉及大量的乘加运算，目前底层的计算与累加实现逻辑导致了 $10^{-5}$ 级别的精度差异，建议排查底层算子实现，以对齐框架间的计算精度。

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
# output：
Maximum difference: 9.918212890625e-05
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

## Issue 079（输入不一致）

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

## Issue 080（输入不一致，参数未对齐）

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

## Issue 081（同80）

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

## Issue 082（同80）

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

## Issue 083（同80）

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

## Issue 084（同80）

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

## Issue 085（输入不一致）

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

## Issue 086（参数未对齐）

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

## Issue 087（同86）

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

## Issue 088（输入不一致）

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

## Issue 089（同88）

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

## Issue 090（同88）

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

## Issue 091（RNG不同）

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

## Issue 092（参数未对齐）

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

## Issue 093（同92）

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

## Issue 094（输出格式不同）

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

## Issue 095（同94）

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

## Issue 096（同94）

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

## Issue 097（同94）

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

## Issue 098（同94）

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

## Issue 099（输出含nan，实际一致）

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

## Issue 100（同99）

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

## Issue 101（参数未对齐）

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

## Issue 102（随机数生成器算法不同）

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

## Issue 103（RNG 实现不同）

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

## Issue 104（同103）

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

## Issue 105（RNG 实现不同）

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

## Issue 106（同105）

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

## Issue 107（同103）

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

## Issue 108（同105）

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

## Issue 109（同105）

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

## Issue 110（RNG 实现不同）

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

## Issue 111（同110）

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

## Issue 112（同110）

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

## Issue 113（同110）

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

## Issue 114（参数未对齐）

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

## Issue 115（同114）

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

## Issue 116（有偏估计与无偏估计的差别）

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

## Issue 117（同116）

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

## Issue 118（同116）

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

## Issue 119（输入不一致、参数未对齐）

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

## Issue 120（同119）

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

## Issue 121（返回形状不同）

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

## Issue 122（同121）

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
- Maximum difference: 4.7266483306884766e-05

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

### MindSpore Issue（√）

**Title***

 [PyTorch -> MindSpore][mindspore.ops.tanh] Output difference anomaly under equivalent migration in tanh operator

**Environment***

**Hardware Environment(`Ascend`/`GPU`/`CPU`)**: CPU

**Software Environment**:

- **MindSpore version (source or binary)**: 2.8.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

**Describe the current behavior***

When mapping `torch.tanh(input)` to `mindspore.ops.tanh(input)` with identical float32 input (shape=[100, 3, 28, 28]), the outputs differ by up to 4.7266483306884766e-05. This is a float32 computation precision difference caused by different underlying kernel implementations.

**Describe the expected behavior***

Both frameworks implement the same mathematical `tanh` function. The observed difference is less than 1e-5, which is within the typical float32 precision range and reflects normal floating-point rounding behavior across different framework backends.

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
aximum difference: 4.5359134674072266e-05
```

**Special notes for this issue**

Float32 tanh 精度误差，差异量级 ~$10^{-5}$，属于底层实现差异，非功能性 bug。

## Issue 124（输入不一致）

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

## Issue 125（参数未对齐）

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

## Issue 126（输入未对齐）

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

## Issue 127（同126）

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

## Issue 128（返回结构不同）

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

## Issue 129（同128）

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

## Issue 130（同128）

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

## Issue 131（同128）

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

## Issue 132（输入不一致）

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

## Issue 133（同132）

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

## Issue 134（无偏估计和有偏估计差别）

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

## Issue 135（同134）

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

## Issue 136（无偏估计和有偏估计差别）

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

## Issue 137（同136）

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

## Issue 138（同136）

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

## Issue 补1

### PyTorch Issue

**Title:** Inconsistent output between `torch.nn.LPPool1d` and MindSpore for negative inputs with fractional `norm_type`

**🐛 Describe the bug** During cross-framework differential testing between PyTorch and MindSpore, an inconsistency was found in the `LPPool1d` operator. When providing an input tensor containing negative values and setting `norm_type=1.5`, `torch.nn.LPPool1d` returns `NaN`, whereas `mindspore.nn.LPPool1d` evaluates to `0.0`.

Mathematically, raising a negative number to a fractional power (1.5) in the real domain is undefined, which explains the `NaN` in PyTorch. However, the inconsistency across frameworks may indicate a difference in underlying bounds checking, zeroing, or exception handling for such inputs.

Minimal reproducing example:

Python

```
import numpy as np
import torch
import mindspore
import mindspore.nn as ms_nn

# Minimal reproducing example trimmed down to the essential parts
input_np = np.array(
    [
        0.63573098,
        0.81810886,
        -0.96552032,
        -0.99548149,
        0.22021013,
        -0.02396944,
    ],
    dtype=np.float32,
).reshape(2, 1, 3)

# PyTorch
pt_layer = torch.nn.LPPool1d(norm_type=1.5, kernel_size=3, stride=1)
out_pt = pt_layer(torch.tensor(input_np)).detach().numpy()

# MindSpore
mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
ms_layer = ms_nn.LPPool1d(norm_type=1.5, kernel_size=3, stride=1)
out_ms = ms_layer(mindspore.Tensor(input_np)).asnumpy()

abs_diff = np.abs(out_pt - out_ms)
max_diff = np.nan if np.isnan(abs_diff).all() else np.nanmax(abs_diff)
print(f"PyTorch output: {out_pt}")
print(f"MindSpore output: {out_ms}")
print(f"NaN count (PT, MS): {np.isnan(out_pt).sum()}, {np.isnan(out_ms).sum()}")
print(f"Maximum difference: {max_diff}")
```

Observed results:

Plaintext

```
PyTorch output: [[[nan]]

 [[nan]]]
MindSpore output: [[[0.]]

 [[0.]]]
NaN count (PT, MS): 2, 0
Maximum difference: nan
```

**Versions** 

同1

------

### MindSpore Issue

**Title:** Inconsistent output behavior of `mindspore.nn.LPPool1d` compared to PyTorch for negative inputs and fractional `norm_type`

**Description:**

## Environment

### Hardware Environment(`Ascend`/`GPU`/`CPU`):

/device cpu

### Software Environment:

- **MindSpore version (source or binary)**: 2.5.0 (binary)
- **Python version**: 3.10.18
- **OS platform and distribution**: Windows 11
- **GCC/Compiler version**: N/A

## Describe the current behavior

When computing `LPPool1d` with `norm_type=1.5` on an input tensor containing negative values, `mindspore.nn.LPPool1d` outputs `0.0`.

## Describe the expected behavior

The expected behavior should either explicitly raise an error or output `NaN` (as PyTorch does), since raising a negative number to a fractional power (1.5) is undefined in the real number domain. The current output of `0.0` masks potential numerical errors.

## Steps to reproduce the issue

1. Define an input numpy array containing negative float values.
2. Initialize `mindspore.nn.LPPool1d(norm_type=1.5, kernel_size=3, stride=1)`.
3. Pass the tensor into the operator and observe the output.

Python

```
import numpy as np
import torch
import mindspore
import mindspore.nn as ms_nn

# Sample: llm_enhanced_torch_nn_LPPool1d_20251215_193350.json_sample1.txt
input_np = np.array(
    [
        0.63573098,
        0.81810886,
        -0.96552032,
        -0.99548149,
        0.22021013,
        -0.02396944,
    ],
    dtype=np.float32,
).reshape(2, 1, 3)

# PyTorch
pt_layer = torch.nn.LPPool1d(norm_type=1.5, kernel_size=3, stride=1)
out_pt = pt_layer(torch.tensor(input_np)).detach().numpy()

# MindSpore
mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
ms_layer = ms_nn.LPPool1d(norm_type=1.5, kernel_size=3, stride=1)
out_ms = ms_layer(mindspore.Tensor(input_np)).asnumpy()

abs_diff = np.abs(out_pt - out_ms)
max_diff = np.nan if np.isnan(abs_diff).all() else np.nanmax(abs_diff)
print(f"PyTorch output: {out_pt}")
print(f"MindSpore output: {out_ms}")
print(f"NaN count (PT, MS): {np.isnan(out_pt).sum()}, {np.isnan(out_ms).sum()}")
print(f"Maximum difference: {max_diff}")
```

## Related log / screenshot

Plaintext

```
PyTorch output: [[[nan]]

 [[nan]]]
MindSpore output: [[[0.]]

 [[0.]]]
NaN count (PT, MS): 2, 0
Maximum difference: nan
```

## Issue 补2

### PyTorch Issue

**Title:** Differential testing inconsistency: `torch.full_like` and `paddle.full_like` handle `int64` max `fill_value` differently

**🐛 Describe the bug*** During cross-framework differential testing between PyTorch and PaddlePaddle, an inconsistency was found when using the `full_like` operator with a `fill_value` of `9223372036854775807` (`int64` maximum value).

`torch.full_like` correctly fills the tensor with `9223372036854775807`. However, the equivalent `paddle.full_like` operator overflows and fills the tensor with `-9223372036854775808`. This issue is reported to document the differential testing findings and the correctness of the PyTorch implementation in this edge case.

Minimal reproducing example:

Python

```
import numpy as np
import torch
import paddle

# From sample: llm_enhanced_torch_full_like_20251202_004252.json_sample1.txt
input_np = np.array(
    [-2, -10, 3, -9, -4, 5, 4, -3, 6, 8],
    dtype=np.int64,
).reshape(2, 1, 1, 1, 5)

fill_value = 9223372036854775807

# PyTorch
out_pt = torch.full_like(torch.tensor(input_np), fill_value=fill_value).numpy()

# Paddle
out_pd = paddle.full_like(paddle.to_tensor(input_np), fill_value=fill_value).numpy()

max_diff = np.max(np.abs(out_pt.astype(np.int64) - out_pd.astype(np.int64)))
print(f"PyTorch output dtype: {out_pt.dtype}")
print(f"Paddle output dtype: {out_pd.dtype}")
print(f"PyTorch first value: {out_pt.flatten()[0]}")
print(f"Paddle first value: {out_pd.flatten()[0]}")
print(f"Maximum difference: {max_diff}")
```

Observed results:

Plaintext

```
PyTorch output dtype: int64
Paddle output dtype: int64
PyTorch first value: 9223372036854775807
Paddle first value: -9223372036854775808
Maximum difference: 1
```

**Versions***

同1

### Paddle Issue

**Title:** `paddle.full_like` 在 `fill_value` 为 `int64` 最大值 (9223372036854775807) 时发生数值溢出错误

**bug描述 Describe the Bug*** 在进行跨框架算子的差分测试时，发现 `paddle.full_like` 存在数值溢出行为。当输入张量的数据类型为 `int64`，且传入的 `fill_value` 为 `9223372036854775807`（即 `int64` 的最大值）时，`paddle.full_like` 无法正确填充该数值，而是溢出变成了 `-9223372036854775808`。相比之下，等价的 PyTorch 算子能够正确处理并输出预期最大值。

复现代码如下：

Python

```
# 导入所有必要的库。 All necessary imports at the beginning.
import numpy as np
import torch
import paddle

# 一个简洁的片段，能够定位到bug。 A succinct reproducing example trimmed down to the essential parts.
# 来自样例: llm_enhanced_torch_full_like_20251202_004252.json_sample1.txt
input_np = np.array(
    [-2, -10, 3, -9, -4, 5, 4, -3, 6, 8],
    dtype=np.int64,
).reshape(2, 1, 1, 1, 5)

fill_value = 9223372036854775807

# PyTorch
out_pt = torch.full_like(torch.tensor(input_np), fill_value=fill_value).numpy()

# Paddle
out_pd = paddle.full_like(paddle.to_tensor(input_np), fill_value=fill_value).numpy()

max_diff = np.max(np.abs(out_pt.astype(np.int64) - out_pd.astype(np.int64)))
print(f"PyTorch output dtype: {out_pt.dtype}")
print(f"Paddle output dtype: {out_pd.dtype}")
print(f"PyTorch first value: {out_pt.flatten()[0]}")
print(f"Paddle first value: {out_pd.flatten()[0]}") # 注意：这里发生了溢出
print(f"Maximum difference: {max_diff}")
```

实际输出结果与对比：

Plaintext

```
PyTorch output dtype: int64
Paddle output dtype: int64
PyTorch first value: 9223372036854775807
Paddle first value: -9223372036854775808
Maximum difference: 1
```

期望结果： PaddlePaddle 应与 PyTorch 一致，能够正确处理 `int64` 的边界值 `9223372036854775807`，而不应发生溢出翻转。

**其他补充信息 Additional Supplementary Information**

**PaddlePaddle version**: 3.2.0

**Python version**: 3.10.18

**OS**: Windows 11
