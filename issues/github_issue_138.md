# 138个跨表不一致Case的GitHub Issue提交稿

> 用法：每个 `## Issue` 段落可直接复制到 GitHub 新建 Issue。
> 口径：同一 `file_name` 在至少两张分配表中出现“不一致/有问题”信号。

- 总Issue数：**138**
- 来源文件：`分配-朱婷.xlsx`、`分配-林哲远.xlsx`、`分配-陈建军.xlsx`、`分配-陈桂学.xlsx`

## Issue 001

### 标题
`[PyTorch -> Paddle][atleast_1d] llm_enhanced_torch_atleast_1d_20251202_132406.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_atleast_1d_20251202_132406.json_sample3.txt`
- 算子: `atleast_1d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_atleast_1d_20251202_132406.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_atleast_1d_20251202_132406.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：两端框架输入了完全不同的随机数据（测试脚本未固定随机种子）
  - 分配-陈桂学.xlsx: 迁移=输入参数问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 002

### 标题
`[PyTorch -> TensorFlow][bmm] llm_enhanced_torch_bmm_20251215_230705.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_bmm_20251215_230705.json_sample1.txt`
- 算子: `bmm`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_bmm_20251215_230705.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_bmm_20251215_230705.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.bmm要求输入张量a.shape = (b, n, m)和b.shape = (b, m, p)，执行batch-wise matrix multiplication，结果shape为(b, n, p)。TensorFlow的tf.linalg.matmul在输入为3D张量时默认执行batch matmul（与bmm语义一致），但...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：输入数据不一致

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 003

### 标题
`[PyTorch -> TensorFlow][diagflat] llm_enhanced_torch_diagflat_20251215_234731.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_diagflat_20251215_234731.json_sample4.txt`
- 算子: `diagflat`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_diagflat_20251215_234731.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_diagflat_20251215_234731.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数未对齐; 不一致=是
    - 原因摘要：尽管两个 API 都叫“对角线构造”，但它们的行为本质不同：torch.diagflat支持任意 offset参数（包括负值和正值），并动态调整输出矩阵大小以适应偏移；tf.linalg.diag 仅支持主对角线，且输出大小固定为 n×n，不接受 offset。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：torch.diagflat(input, offset=1) 在 PyTorch 中将一维输入向量展平为二维矩阵，并在指定偏移量（offset=1）处填充对角线，其余位置补零。对于长度为 4 的输入向量，diagflat(..., offset=1) 生成一个 (5, 5) 矩阵：因为 offset=1 表示主对角线上方第一条对角线，需保证该对角线能容纳全...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 004

### 标题
`[PyTorch -> MindSpore][empty_like] llm_enhanced_torch_empty_like_20251214_172128.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_empty_like_20251214_172128.json_sample1.txt`
- 算子: `empty_like`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_empty_like_20251214_172128.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_empty_like_20251214_172128.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：empty_like 的语义是分配未初始化内存，输出值完全取决于内存中残留的随机“垃圾数据”（Undefined Values），在不同框架甚至同框架不同调用间必然不一致。
  - 分配-陈桂学.xlsx: 迁移=比较过程问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 005

### 标题
`[PyTorch -> Paddle][erfinv] llm_enhanced_torch_erfinv_20251125_145454.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_erfinv_20251125_145454.json_sample2.txt`
- 算子: `erfinv`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_erfinv_20251125_145454.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_erfinv_20251125_145454.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=是; 不一致=是
    - 原因摘要：NaN问题，感觉不是不一致
  - 分配-陈桂学.xlsx: 迁移=比较过程问题
    - 原因摘要：由于输入数据超出了算子的数学定义域导致的非正常结果。

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 006

### 标题
`[PyTorch -> Paddle][erfinv] llm_enhanced_torch_erfinv_20251125_145454.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_erfinv_20251125_145454.json_sample3.txt`
- 算子: `erfinv`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_erfinv_20251125_145454.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_erfinv_20251125_145454.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow 2.x 原生不支持 torch.erfinv（逆误差函数）的直接等价 API。tf.math.erfinv 仅在 TensorFlow 2.10+ 中引入，且要求输入值严格位于开区间 (-1, 1) 内；而 PyTorch 的 torch.erfinv 同样定义域为 (-1, 1)，但输入样本中包含超出该范围的值：1.429549424...
  - 分配-陈建军.xlsx: 迁移=是; 不一致=否
    - 原因摘要：erfinv（逆误差函数）的数学定义值域为 (-1, 1)，超出该范围无实数解，所以会输出NaN

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 007

### 标题
`[PyTorch -> Paddle][erfinv] llm_enhanced_torch_erfinv_20251125_151700.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_erfinv_20251125_151700.json_sample3.txt`
- 算子: `erfinv`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_erfinv_20251125_151700.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_erfinv_20251125_151700.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：torch.erfinv 是 inverse error function（反误差函数），其定义域严格限制在 (-1, 1) 内；输入超出此范围时，PyTorch 会返回 NaN（例如 -1.0186134576797485、-1.6229819059371948 等均 < -1）。TensorFlow 中无直接等价的 tf.erfinv 操作（tf.mat...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=否
    - 原因摘要：erfinv（逆误差函数）的数学定义值域为 (-1, 1)，超出该范围无实数解，所以会输出NaN

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 008

### 标题
`[PyTorch -> MindSpore][erfinv] llm_enhanced_torch_erfinv_20251215_184626.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_erfinv_20251215_184626.json_sample1.txt`
- 算子: `erfinv`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_erfinv_20251215_184626.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_erfinv_20251215_184626.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是（代码实现问题）; 不一致=是
    - 原因摘要：输入数据包含超出 erfinv 定义域(-1, 1)的数值（如 -1.22, -1.72），两框架对定义域外输入的处理逻辑（如返回 NaN、Inf 还是未定义值）存在差异。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.erfinv是反误差函数（inverse error function），定义域为(-1, 1)；输入值如-1.2201734282525922、-1.7247004652368227等已超出定义域（|x| >= 1），在PyTorch中会返回nan（根据PyTorch文档，torch.erfinv对|x| >= 1的输入返回na...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 009

### 标题
`[PyTorch -> TensorFlow][fmod] llm_enhanced_torch_fmod_20251215_201658.json_sample10.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_fmod_20251215_201658.json_sample10.txt`
- 算子: `fmod`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_fmod_20251215_201658.json_sample10.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_fmod_20251215_201658.json_sample10.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：两个框架均为浮点取模操作，但在负数计算结果处理上的符号规则存在差异。
  - 分配-陈桂学.xlsx: 迁移=API语义适配错误
    - 原因摘要：torch.fmod采用截断除法，而tf.math.floormod采用地板除法

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 010

### 标题
`[PyTorch -> TensorFlow][fmod] llm_enhanced_torch_fmod_20251215_201658.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_fmod_20251215_201658.json_sample2.txt`
- 算子: `fmod`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_fmod_20251215_201658.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_fmod_20251215_201658.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：两个框架均为浮点取模操作，但在负数计算结果处理上的符号规则存在差异。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：torch.fmod 和 tf.math.floormod 的数学定义不同：torch.fmod 实现的是 C/Python 风格的 remainder 运算（即 remainder = dividend - trunc(dividend / divisor) * divisor），其符号与被除数（dividend）相同；而 tf.math.floormod...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 011

### 标题
`[PyTorch -> TensorFlow][fmod] llm_enhanced_torch_fmod_20251215_201658.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_fmod_20251215_201658.json_sample3.txt`
- 算子: `fmod`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_fmod_20251215_201658.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_fmod_20251215_201658.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：torch.fmod 和 tf.math.floormod 的数学定义不同：torch.fmod 实现的是 IEEE 754 余数运算（即 'truncated division' 余数），符号与被除数（input）相同；而 tf.math.floormod 实现的是 'floor division' 余数，符号与除数（other）相同。例如，当 input...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch fmod 余数符号与被除数一致（截断除法），TF floormod 余数符号与除数一致

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 012

### 标题
`[PyTorch -> TensorFlow][fmod] llm_enhanced_torch_fmod_20251215_201658.json_sample6.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_fmod_20251215_201658.json_sample6.txt`
- 算子: `fmod`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_fmod_20251215_201658.json_sample6.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_fmod_20251215_201658.json_sample6.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：torch.fmod 和 tf.math.floormod 语义不同：torch.fmod 实现的是 C 风格的浮点取余（remainder），定义为 r = a - n*b，其中 n = trunc(a/b)（向零取整），结果与被除数 a 同号；而 tf.math.floormod 实现的是 floor division 取余，定义为 r = a - fl...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch fmod 余数符号与被除数一致（截断除法），TF floormod 余数符号与除数一致

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 013

### 标题
`[PyTorch -> TensorFlow][fmod] llm_enhanced_torch_fmod_20251215_201658.json_sample7.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_fmod_20251215_201658.json_sample7.txt`
- 算子: `fmod`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_fmod_20251215_201658.json_sample7.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_fmod_20251215_201658.json_sample7.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：两个框架均为浮点取模操作，但在负数计算结果处理上的符号规则存在差异。
  - 分配-陈桂学.xlsx: 迁移=API语义适配错误
    - 原因摘要：torch.fmod采用截断除法，而tf.math.floormod采用地板除法

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 014

### 标题
`[PyTorch -> TensorFlow][logdet] llm_enhanced_torch_logdet_20251215_203929.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_logdet_20251215_203929.json_sample1.txt`
- 算子: `logdet`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_logdet_20251215_203929.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_logdet_20251215_203929.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：尽管两个 API 语义一致，但在处理边界情况（如负行列式）时，由于底层数值算法实现差异（如 Cholesky 前的预处理、特征值提取精度控制等），导致一个框架正确返回 `NaN`，而另一个框架由于数值误差或容错策略，尝试计算并返回了一个非 `NaN` 的结果（如 `inf` 或异常值），从而引发比较错误
  - 分配-陈桂学.xlsx: 迁移=输入参数问题
    - 原因摘要：由于随机生成的输入矩阵产生了非正的行列式，导致对数运算在实数范围内无定义

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 015

### 标题
`[PyTorch -> TensorFlow][logdet] llm_enhanced_torch_logdet_20251215_203929.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_logdet_20251215_203929.json_sample2.txt`
- 算子: `logdet`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_logdet_20251215_203929.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_logdet_20251215_203929.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：尽管两个 API 语义一致，但在处理边界情况（如负行列式）时，由于底层数值算法实现差异（如 Cholesky 前的预处理、特征值提取精度控制等），导致一个框架正确返回 `NaN`，而另一个框架由于数值误差或容错策略，尝试计算并返回了一个非 `NaN` 的结果（如 `inf` 或异常值），从而引发比较错误
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.logdet要求输入矩阵为正定（或至少可逆），但对非正定/负行列式矩阵会返回实数结果（即log|det| + iπ×sign(det)的实部？实际上torch.logdet严格定义为log(|det|)当det<0时仍返回实数，但其底层实现基于LU分解并显式处理符号；而TensorFlow的tf.linalg.logdet在输入矩...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 016

### 标题
`[PyTorch -> TensorFlow][logdet] llm_enhanced_torch_logdet_20251215_203929.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_logdet_20251215_203929.json_sample3.txt`
- 算子: `logdet`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_logdet_20251215_203929.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_logdet_20251215_203929.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：尽管两个 API 语义一致，但在处理边界情况（如负行列式）时，由于底层数值算法实现差异（如 Cholesky 前的预处理、特征值提取精度控制等），导致一个框架正确返回 `NaN`，而另一个框架由于数值误差或容错策略，尝试计算并返回了一个非 `NaN` 的结果（如 `inf` 或异常值），从而引发比较错误
  - 分配-陈桂学.xlsx: 迁移=输入参数问题
    - 原因摘要：由于随机生成的输入矩阵产生了非正的行列式，导致对数运算在实数范围内无定义

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 017

### 标题
`[PyTorch -> TensorFlow][logdet] llm_enhanced_torch_logdet_20251215_203929.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_logdet_20251215_203929.json_sample4.txt`
- 算子: `logdet`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_logdet_20251215_203929.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_logdet_20251215_203929.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：torch.logdet 和 tf.linalg.logdet 虽然语义相同（均计算批处理矩阵的对数行列式），但二者在输入合法性检查和数值稳定性处理上存在关键差异：PyTorch 的 torch.logdet 要求输入矩阵必须为正定（或至少可逆且行列式 > 0），否则行为未定义（可能返回 nan 或抛出 RuntimeError）；而 TensorFlow ...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch仅支持2D方阵输入，TF则支持批量张量。因此出现NaN

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 018

### 标题
`[PyTorch -> TensorFlow][logspace] llm_enhanced_torch_logspace_20251215_203651.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_logspace_20251215_203651.json_sample1.txt`
- 算子: `logspace`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_logspace_20251215_203651.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_logspace_20251215_203651.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数未对齐; 不一致=是
    - 原因摘要：两个函数的核心语义不同：PyTorch 允许用户通过 `base` 显式设置对数底数；TensorFlow (tf.experimental.numpy.logspace) 强制使用 base=10，无法更改。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.logspace(..., base=2) 显式指定底数为2，而TensorFlow中tf.experimental.numpy.logspace默认base=10（与numpy.logspace行为一致），未传递base=2参数。因此两者生成的等比数列底数不同：PyTorch生成的是2^(-10), 2^(-5), 2^0, 2...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 019

### 标题
`[PyTorch -> MindSpore][matmul] llm_enhanced_torch_matmul_20251216_013733.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_matmul_20251216_013733.json_sample2.txt`
- 算子: `matmul`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_matmul_20251216_013733.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_matmul_20251216_013733.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch中torch.matmul允许零维（即shape中含0）的张量参与矩阵乘法，例如形状为[4096, 0]和[0, 4096]的两个张量相乘，结果为形状[4096, 4096]的全零矩阵（因内积维度为0，求和为空，故每个元素为0）。TensorFlow的tf.linalg.matmul（或等价的@运算符）在TensorFlow 2.x中**默认不...
  - 分配-陈建军.xlsx: 迁移=是; 不一致=否
    - 原因摘要：NaN问题，应该是一致的输出

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 020

### 标题
`[PyTorch -> MindSpore][mean] llm_enhanced_torch_mean_20251215_184512.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_mean_20251215_184512.json_sample1.txt`
- 算子: `mean`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_mean_20251215_184512.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_mean_20251215_184512.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：两个框架在处理“全空维度”这一边界情况时的行为不一致
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch中input的shape为[2, 0, 4, 5, 6]，即第二维尺寸为0（空维度），构成一个合法的空张量（empty tensor）。torch.mean在空张量上默认行为是：当dim未指定且输入为空时，若keepdim=False（默认），返回标量nan（因无元素可均值）；若keepdim=True，则保持维度但对应空轴上结果为nan。而T...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 021

### 标题
`[PyTorch -> MindSpore][mean] llm_enhanced_torch_mean_20251215_184512.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_mean_20251215_184512.json_sample2.txt`
- 算子: `mean`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_mean_20251215_184512.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_mean_20251215_184512.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch中torch.mean()在空张量（shape=[0, 5]）上的行为是明确定义的：当输入张量无元素时，torch.mean()默认返回NaN（符合IEEE 754规范），且不会报错。TensorFlow中tf.reduce_mean()对空张量（如shape=(0, 5)）的行为取决于版本和reduce_axis设置：在TensorFlow ...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch torch.mean 处理形状为 [0,5] 的空张量时，内核会抛出运行时错误（因无元素可计算均值）；MindSpore mint.mean 处理同形状空张量时，内核默认返回 0.0

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 022

### 标题
`[PyTorch -> MindSpore][mean] llm_enhanced_torch_mean_20251215_184512.json_sample5.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_mean_20251215_184512.json_sample5.txt`
- 算子: `mean`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_mean_20251215_184512.json_sample5.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_mean_20251215_184512.json_sample5.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=是; 不一致=是
    - 原因摘要：PyTorch：仍返回形状[64]的全 0 张量；MindSpore：仍抛出「无元素可计算」的错误，无容错机制。
  - 分配-陈桂学.xlsx: 迁移=比较过程问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 023

### 标题
`[PyTorch -> MindSpore][mean] llm_enhanced_torch_mean_20251215_184512.json_sample6.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_mean_20251215_184512.json_sample6.txt`
- 算子: `mean`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_mean_20251215_184512.json_sample6.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_mean_20251215_184512.json_sample6.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch中torch.mean()在空张量（shape=[0, 10]）上的行为是定义明确的：当输入张量无元素时，torch.mean()默认返回NaN（符合IEEE 754规范及PyTorch文档），且dtype保持为float32。TensorFlow中tf.reduce_mean()在空张量（如tf.zeros([0, 10], tf.float...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch torch.mean 处理形状为 [0,10] 的空张量时，内核会抛出运行时错误（因无元素可计算均值）；MindSpore mint.mean 处理同形状空张量时，内核默认返回 0.0

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 024

### 标题
`[PyTorch -> Paddle][multinomial] llm_enhanced_torch_multinomial_20251202_012420.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_multinomial_20251202_012420.json_sample1.txt`
- 算子: `multinomial`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_multinomial_20251202_012420.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_multinomial_20251202_012420.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：multinomial 是从分布中采样索引，属随机操作。不同框架 RNG 实现不同，采样结果必然不同（差异值为索引值的差）。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.multinomial在replacement=True时，对输入概率向量执行带放回的多项式采样，但其底层实现要求输入为非负数且通常隐式归一化（即自动除以sum(input)得到概率分布）。TensorFlow中无直接等价的tf.random.categorical或tf.random.uniform + tf.math.top_...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 025

### 标题
`[PyTorch -> TensorFlow][multinomial] llm_enhanced_torch_multinomial_20251215_233345.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_multinomial_20251215_233345.json_sample1.txt`
- 算子: `multinomial`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_multinomial_20251215_233345.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_multinomial_20251215_233345.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：torch.multinomial 和 tf.random.categorical 的输入语义不同：torch.multinomial 接收概率（非负、可未归一化，但需为概率质量，即视为离散分布的权重），而 tf.random.categorical 接收的是 logits（未归一化的对数概率），会先通过 softmax 归一化。给定的 input 值 [0...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch直接基于权重抽样，TF基于对数概率抽样；PyTorch默认有放回，TF默认无放回

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 026

### 标题
`[PyTorch -> Paddle][nanmedian] llm_enhanced_torch_nanmedian_20251201_232713.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nanmedian_20251201_232713.json_sample2.txt`
- 算子: `nanmedian`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nanmedian_20251201_232713.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nanmedian_20251201_232713.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow 核心 API 中不存在与 PyTorch torch.nanmedian 等价的原生函数。tf.experimental.numpy.nanmedian 仅在 TensorFlow 2.9+ 的 experimental.numpy 模块中提供，且其行为（如 axis 默认值、输出 dtype 推断、NaN 处理边界情况）与 PyTor...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch torch.nanmedian 内核默认对全部元素计算全局 nanmedian（忽略 NaN 后求整体中位数）；Paddle paddle.nanmedian 内核默认沿最后一维（axis=-1，即 shape 中的 4 维）计算 nanmedian。

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 027

### 标题
`[PyTorch -> Paddle][nanmedian] llm_enhanced_torch_nanmedian_20251201_232713.json_sample5.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nanmedian_20251201_232713.json_sample5.txt`
- 算子: `nanmedian`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nanmedian_20251201_232713.json_sample5.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nanmedian_20251201_232713.json_sample5.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow 2.x 原生不提供 torch.nanmedian 的等价 API。tf.experimental.numpy.nanmedian 或 tf.math.reduce_* 系列算子均不直接支持沿指定轴计算 NaN-aware 中位数，且其底层实现（如排序策略、NaN 处理逻辑、插值方式）与 PyTorch 存在差异：1) PyTorch ...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch torch.nanmedian：内核默认对全部元素计算全局 nanmedian（忽略 NaN 后求整个张量的中位数），输入为 2×5×5=50 个元素，最终输出单个标量值； Paddle paddle.nanmedian：内核默认沿最后一维（axis=-1，即 shape 中的 5 维）计算 nanmedian，输入为 2×5 行、每行 5 ...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 028

### 标题
`[PyTorch -> Paddle][nn_AvgPool1d] llm_enhanced_torch_nn_AvgPool1d_20251202_124923.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_AvgPool1d_20251202_124923.json_sample1.txt`
- 算子: `nn_AvgPool1d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_AvgPool1d_20251202_124923.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_AvgPool1d_20251202_124923.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数未对齐; 不一致=是
    - 原因摘要：PyTorch: count_include_pad=False（不计算填充区域）；Paddle: exclusive=False（含义是“不排除填充”，即计算填充区域）
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的AvgPool1d在ceil_mode=True时，输出长度计算公式为: floor((L + 2*padding - kernel_size) / stride) + 1，但当ceil_mode=True时，实际使用向上取整：output_length = ceil((L + 2*padding - kernel_size) / strid...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 029

### 标题
`[PyTorch -> TensorFlow][nn_AvgPool1d] llm_enhanced_torch_nn_AvgPool1d_20251215_234817.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_AvgPool1d_20251215_234817.json_sample1.txt`
- 算子: `nn_AvgPool1d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_AvgPool1d_20251215_234817.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_AvgPool1d_20251215_234817.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：两者默认的池化目标维度不匹配，torch.nn.AvgPool1d默认作用于最后一个维度，而tf.keras.layers.AveragePooling1D默认作用于倒数第二个维度。需显式指定data_format="channels_first"
  - 分配-陈桂学.xlsx: 迁移=输入参数问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 030

### 标题
`[PyTorch -> TensorFlow][nn_AvgPool1d] llm_enhanced_torch_nn_AvgPool1d_20251215_234817.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_AvgPool1d_20251215_234817.json_sample2.txt`
- 算子: `nn_AvgPool1d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_AvgPool1d_20251215_234817.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_AvgPool1d_20251215_234817.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：输入：PyTorch默认 `NCHW`；TensorFlow默认 `NHWC`
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的AvgPool1d默认输入格式为(N, C, L)，即[batch, channels, sequence_length]，其中池化操作在最后一个维度（L=10）上进行；而TensorFlow的AveragePooling1D默认输入格式为(N, L, C)，即[batch, sequence_length, channels]。给定输入sh...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 031

### 标题
`[PyTorch -> TensorFlow][nn_AvgPool3d] llm_enhanced_torch_nn_AvgPool3d_20251215_210802.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_AvgPool3d_20251215_210802.json_sample1.txt`
- 算子: `nn_AvgPool3d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_AvgPool3d_20251215_210802.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_AvgPool3d_20251215_210802.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：输入：PyTorch默认 `NCHW`；TensorFlow默认 `NHWC`
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.AvgPool3d默认采用'NCDHW'数据格式（batch, channels, depth, height, width），而TensorFlow的tf.keras.layers.AveragePooling3D默认采用'NDHWC'格式（batch, depth, height, width, channels）。输入...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 032

### 标题
`[PyTorch -> TensorFlow][nn_BatchNorm1d] llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample1.txt`
- 算子: `nn_BatchNorm1d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.BatchNorm1d默认对第1维（channel维，即axis=1）进行归一化，其输入形状为(N, C, L)，其中C=3是num_features，因此它在每个channel上独立计算均值和方差（即对维度[0,2]，也就是batch和sequence长度维度）；而TensorFlow配置中axis=-1等价于axis=2...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 沿通道维（C=1）归一化，TF 的axis=-1则沿最后一维（L=10）归一化。PyTorch 用无偏方差（N-1），TF 默认用有偏方差。PyTorch的eps默认1e-5，而TF是1e-3

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 033

### 标题
`[PyTorch -> TensorFlow][nn_BatchNorm1d] llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample2.txt`
- 算子: `nn_BatchNorm1d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的BatchNorm1d默认对第1维（即特征维，dim=1）进行归一化，输入形状为[2, 3, 1]时，它将3个特征通道（num_features=3）独立归一化，即在batch（dim=0）和sequence/height/width（dim=2）维度上计算均值和方差，等价于对每个特征通道在(N, L) = (2, 1)共2个样本上统计；而T...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 沿通道维（C=1）归一化，TF 默认沿最后一维（-1）归一化。PyTorch 用无偏方差（N-1），TF 默认用有偏方差（N）。

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 034

### 标题
`[PyTorch -> TensorFlow][nn_BatchNorm1d] llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample3.txt`
- 算子: `nn_BatchNorm1d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数未对齐（tf的training参数没有显式设置）; 不一致=是
    - 原因摘要：两个框架未明确指定训练状态，导致行为不一致。一个使用当前 batch 统计量并更新运行统计量，另一个却因默认 `training=False` 而使用无效的运行统计量，造成巨大误差。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.BatchNorm1d默认对通道维度（dim=1）进行归一化，即在shape=(N, C, L)中，C=3是num_features，统计和归一化沿batch和sequence维度（N和L）进行，等价于对每个特征通道独立计算均值和方差：E[x_c], Var[x_c]，其中c∈[0,2]。而TensorFlow的tf.ker...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 035

### 标题
`[PyTorch -> TensorFlow][nn_BatchNorm1d] llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample5.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample5.txt`
- 算子: `nn_BatchNorm1d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample5.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_BatchNorm1d_20251215_233430.json_sample5.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数默认值不一致; 不一致=是
    - 原因摘要：参数默认值不一致导致行为、运行时统计量的处理方式存在差异
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的BatchNorm1d默认对通道维度（dim=1）进行归一化，即对形状为[batch, num_features, seq_len]的输入，沿第1维（num_features=3）计算每个特征通道的均值和方差；而TensorFlow配置中axis=-1等价于axis=2（因输入shape为[1,3,5]，-1指最后一个维度），导致TF在长度维...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 036

### 标题
`[PyTorch -> TensorFlow][nn_BatchNorm2d] llm_enhanced_torch_nn_BatchNorm2d_20251215_202219.json_sample5.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_BatchNorm2d_20251215_202219.json_sample5.txt`
- 算子: `nn_BatchNorm2d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_BatchNorm2d_20251215_202219.json_sample5.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_BatchNorm2d_20251215_202219.json_sample5.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow的BatchNormalization默认axis=-1（即通道在最后一维），而PyTorch的nn.BatchNorm2d要求输入为[N, C, H, W]格式，其归一化沿C（通道）维度进行，对应TensorFlow中axis=1。当前TensorFlow配置使用axis=-1，将导致在形状为[2,3,4,4]的输入上沿最后一个维度（即...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 沿通道维（C=1）归一化，TF设置axis=-1沿最后一维（W=4）归一化。PyTorch 禁用affine（无 γ/β），TF 默认启用center/scale。PyTorch 用无偏方差（N-1），TF 默认用有偏方差。

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 037

### 标题
`[PyTorch -> TensorFlow][nn_BatchNorm2d] llm_enhanced_torch_nn_BatchNorm2d_20251215_202219.json_sample6.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_BatchNorm2d_20251215_202219.json_sample6.txt`
- 算子: `nn_BatchNorm2d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_BatchNorm2d_20251215_202219.json_sample6.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_BatchNorm2d_20251215_202219.json_sample6.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数未对齐（tf的training参数没有显式设置）; 不一致=是
    - 原因摘要：两个框架未明确指定训练状态，导致行为不一致。一个使用当前 batch 统计量并更新运行统计量，另一个却因默认 `training=False` 而使用无效的运行统计量，造成巨大误差。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的BatchNorm2d默认对通道维度（dim=1，即NCHW格式的第1维）进行归一化，而TensorFlow的tf.keras.layers.BatchNormalization默认axis=-1（即NHWC格式的最后一个维度），在输入shape为[2,3,4,4]（NCHW）时，TF层会错误地将axis=-1解释为第3维（即H或W维度，而非...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 038

### 标题
`[PyTorch -> TensorFlow][nn_BatchNorm3d] llm_enhanced_torch_nn_BatchNorm3d_20251215_165013.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_BatchNorm3d_20251215_165013.json_sample1.txt`
- 算子: `nn_BatchNorm3d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_BatchNorm3d_20251215_165013.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_BatchNorm3d_20251215_165013.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数默认值不一致; 不一致=是
    - 原因摘要：TensorFlow 的 BatchNormalization 的默认 epsilon 参数值是 `1e-3`，而 PyTorch 是 `1e-5`。这一差异直接导致了归一化过程中的分母偏移，尤其是在输入数据分布较广、方差较小的情况下，对结果影响显著
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.BatchNorm3d默认对通道维度（dim=1，即shape[1]）进行归一化，而TensorFlow的tf.keras.layers.BatchNormalization默认axis=-1（即最后一个维度），在输入shape=[2,3,4,4,4]下，axis=-1对应第5维（索引4），即归一化维度为4（channel ...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 039

### 标题
`[PyTorch -> MindSpore][nn_Conv1d] llm_enhanced_torch_nn_Conv1d_20251215_194200.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_Conv1d_20251215_194200.json_sample3.txt`
- 算子: `nn_Conv1d`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_Conv1d_20251215_194200.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_Conv1d_20251215_194200.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数默认值不一致; 不一致=是
    - 原因摘要：权重（Weight）和偏置（Bias）初始化不同。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow默认使用'channels_last'数据格式（NHWC），而PyTorch使用'channels_first'（NCHW）。在Conv1D中，PyTorch输入形状为[2, 16, 10]（batch, channels, length），对应TensorFlow需显式指定data_format='channels_first'；否则T...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 040

### 标题
`[PyTorch -> MindSpore][nn_Dropout] llm_enhanced_torch_nn_Dropout_20251215_193853.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_Dropout_20251215_193853.json_sample4.txt`
- 算子: `nn_Dropout`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_Dropout_20251215_193853.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_Dropout_20251215_193853.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：输入的api都是torch.nn.Dropout，而且Dropout的随机数生成器即使种子一样，也可能会生成不同的数
  - 分配-陈桂学.xlsx: 迁移=报告有问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 041

### 标题
`[PyTorch -> MindSpore][nn_Linear] llm_enhanced_torch_nn_Linear_20251214_171639.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_Linear_20251214_171639.json_sample3.txt`
- 算子: `nn_Linear`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_Linear_20251214_171639.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_Linear_20251214_171639.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数默认值不一致; 不一致=是
    - 原因摘要：权重（Weight）初始化不同。nn.Linear 是包含可学习参数的层，两端框架默认随机初始化权重，测试脚本仅固定了输入数据（Input）而未显式同步两端的权重参数，导致计算结果差异巨大。
  - 分配-陈桂学.xlsx: 迁移=初始化层时权重不一致

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 042

### 标题
`[PyTorch -> MindSpore][nn_Linear] llm_enhanced_torch_nn_Linear_20251214_171639.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_Linear_20251214_171639.json_sample4.txt`
- 算子: `nn_Linear`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_Linear_20251214_171639.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_Linear_20251214_171639.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数默认值不一致; 不一致=是
    - 原因摘要：权重（Weight）初始化不同。nn.Linear 是包含可学习参数的层，两端框架默认随机初始化权重，测试脚本仅固定了输入数据（Input）而未显式同步两端的权重参数，导致计算结果差异巨大。
  - 分配-陈桂学.xlsx: 迁移=初始化层时权重不一致

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 043

### 标题
`[PyTorch -> TensorFlow][nn_Linear] llm_enhanced_torch_nn_Linear_20251215_164342.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_Linear_20251215_164342.json_sample2.txt`
- 算子: `nn_Linear`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_Linear_20251215_164342.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_Linear_20251215_164342.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数默认值不一致; 不一致=是
    - 原因摘要：两者的默认权重初始化方式不同，导致初始权重分布完全不同，前向传播结果产生显著偏差
  - 分配-陈桂学.xlsx: 迁移=初始化层时权重不一致
    - 原因摘要：随机初始化导致两个框架使用的权重矩阵数值不同。

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 044

### 标题
`[PyTorch -> TensorFlow][nn_Linear] llm_enhanced_torch_nn_Linear_20251215_164342.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_Linear_20251215_164342.json_sample3.txt`
- 算子: `nn_Linear`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_Linear_20251215_164342.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_Linear_20251215_164342.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数默认值不一致; 不一致=是
    - 原因摘要：两者的默认权重初始化方式不同，导致初始权重分布完全不同，前向传播结果产生显著偏差
  - 分配-陈桂学.xlsx: 迁移=初始化层时权重不一致
    - 原因摘要：随机初始化导致两个框架使用的权重矩阵数值不同。

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 045

### 标题
`[PyTorch -> TensorFlow][nn_MaxPool1d] llm_enhanced_torch_nn_MaxPool1d_20251215_164114.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_MaxPool1d_20251215_164114.json_sample4.txt`
- 算子: `nn_MaxPool1d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_MaxPool1d_20251215_164114.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_MaxPool1d_20251215_164114.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.MaxPool1d期望输入形状为(N, C, L)，即[batch, channels, length]，其中池化操作在最后一个维度（L）上进行；而TensorFlow的tf.keras.layers.MaxPooling1D期望输入形状为(N, L, C)，即[batch, length, channels]，池化操作在轴...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 默认沿[N,C,L]的 L 维池化，TF 默认沿[N,L,C]的 L 维池化。PyTorch 支持return_indices=True返回索引，TF 无原生支持。

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 046

### 标题
`[PyTorch -> TensorFlow][nn_MaxPool1d] llm_enhanced_torch_nn_MaxPool1d_20251215_164114.json_sample7.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_MaxPool1d_20251215_164114.json_sample7.txt`
- 算子: `nn_MaxPool1d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_MaxPool1d_20251215_164114.json_sample7.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_MaxPool1d_20251215_164114.json_sample7.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：输入：PyTorch默认 `NCL`；TensorFlow默认 `NLC`
  - 分配-陈桂学.xlsx: 迁移=输入参数问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 047

### 标题
`[PyTorch -> MindSpore][nn_MaxPool2d] llm_enhanced_torch_nn_MaxPool2d_20251215_192154.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_MaxPool2d_20251215_192154.json_sample1.txt`
- 算子: `nn_MaxPool2d`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_MaxPool2d_20251215_192154.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_MaxPool2d_20251215_192154.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数未对齐; 不一致=是
    - 原因摘要：`pad_mode` 与 `padding` 的语义不一致
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的MaxPool2d在padding=1、kernel_size=2、stride=2下对输入[1,1,2,4]（即H=2, W=4）进行填充和池化时，实际填充行为是'implicit zero-padding'，但其索引返回（return_indices=true）基于填充后的张量计算；而TensorFlow的tf.nn.max_pool_w...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 048

### 标题
`[PyTorch -> TensorFlow][nn_MaxPool2d] llm_enhanced_torch_nn_MaxPool2d_20251215_205114.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_MaxPool2d_20251215_205114.json_sample1.txt`
- 算子: `nn_MaxPool2d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_MaxPool2d_20251215_205114.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_MaxPool2d_20251215_205114.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：1. 输入张量格式不一致：PyTorch默认使用NCHW（batch, channels, height, width），而TensorFlow默认使用NHWC（batch, height, width, channels）。但TF配置中输入shape仍写为[2, 3, 4, 4]，这在TF中会被解释为NHWC格式下的(batch=2, height=3, ...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：padding双方不同，Pytorch是1而TF是same。而且PyTorch支持return_indices=True返回索引，TF无原生支持。

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 049

### 标题
`[PyTorch -> TensorFlow][nn_MaxPool2d] llm_enhanced_torch_nn_MaxPool2d_20251215_205114.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_MaxPool2d_20251215_205114.json_sample2.txt`
- 算子: `nn_MaxPool2d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_MaxPool2d_20251215_205114.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_MaxPool2d_20251215_205114.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数未对齐; 不一致=是
    - 原因摘要：- 在无训练前提下，一个框架使用 batch 统计量，另一个却尝试使用初始化的 running stats；
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的MaxPool2d中padding=1表示在输入特征图的每一边（top/bottom/left/right）显式添加1行/列零填充，属于'explicit padding'；而TensorFlow的MaxPooling2D中padding='same'表示自动计算填充量以使输出空间尺寸满足ceil(input_size / stride)，其...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 050

### 标题
`[PyTorch -> TensorFlow][nn_MaxPool2d] llm_enhanced_torch_nn_MaxPool2d_20251215_205114.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_MaxPool2d_20251215_205114.json_sample4.txt`
- 算子: `nn_MaxPool2d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_MaxPool2d_20251215_205114.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_MaxPool2d_20251215_205114.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：pytorch输入的padding默认是1，而tf输入的是same，参数不同，前者是固定填充，后者则是动态填充
  - 分配-陈桂学.xlsx: 迁移=输入参数问题
    - 原因摘要：维度排列

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 051

### 标题
`[PyTorch -> Paddle][nn_MultiLabelSoftMarginLoss] llm_enhanced_torch_nn_MultiLabelSoftMarginLoss_20251202_000132.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_MultiLabelSoftMarginLoss_20251202_000132.json_sample2.txt`
- 算子: `nn_MultiLabelSoftMarginLoss`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_d/llm_enhanced_torch_nn_MultiLabelSoftMarginLoss_20251202_000132.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_d/llm_enhanced_torch_nn_MultiLabelSoftMarginLoss_20251202_000132.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=否; 不一致=否
    - 原因摘要：缺少必要参数导致调用失败。Loss 函数通常需要 input 和 target 两个参数。JSON 中仅提供了 input，导致框架在调用 forward 时抛出异常或返回非 Tensor 对象（如 None/Error），比较器在对其进行 isfinite 检查时崩溃。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow 中没有直接等价于 torch.nn.MultiLabelSoftMarginLoss 的原生损失函数。PyTorch 的 MultiLabelSoftMarginLoss 计算的是多标签分类的软边界损失，其数学形式为：loss = -sum(y_i * log(sigmoid(x_i)) + (1-y_i) * log(1-sigmoi...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 052

### 标题
`[PyTorch -> TensorFlow][nn_PReLU] llm_enhanced_torch_nn_PReLU_20251215_233126.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_PReLU_20251215_233126.json_sample2.txt`
- 算子: `nn_PReLU`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_PReLU_20251215_233126.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_PReLU_20251215_233126.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数未对齐; 不一致=是
    - 原因摘要：一个共享学习参数，一个按通道独立学习参数（可在 PyTorch 中设置 `num_parameters`值以匹配 TensorFlow 行为）
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.PReLU默认采用'channel-wise'（逐通道）可学习参数：每个通道共享一个独立的可学习斜率α（即num_parameters=1时，α.shape = [C]，C为通道数）；而TensorFlow的tf.keras.layers.PReLU默认采用'per-parameter'模式，但其行为取决于input ran...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 053

### 标题
`[PyTorch -> TensorFlow][nn_PReLU] llm_enhanced_torch_nn_PReLU_20251215_233126.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_PReLU_20251215_233126.json_sample3.txt`
- 算子: `nn_PReLU`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_PReLU_20251215_233126.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_PReLU_20251215_233126.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数未对齐; 不一致=是
    - 原因摘要：一个共享学习参数，一个按通道独立学习参数（可在 PyTorch 中设置 `num_parameters`值以匹配 TensorFlow 行为）
  - 分配-陈桂学.xlsx: 迁移=初始化层时权重不一致

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 054

### 标题
`[PyTorch -> Paddle][nn_RReLU] llm_enhanced_torch_nn_RReLU_20251202_010732.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_RReLU_20251202_010732.json_sample1.txt`
- 算子: `nn_RReLU`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_RReLU_20251202_010732.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_RReLU_20251202_010732.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：随机算子行为。RReLU（Randomized ReLU）在训练模式下会从给定范围内随机采样斜率。不同框架的随机数生成器（RNG）实现不同，导致采样的斜率不一致
  - 分配-陈桂学.xlsx: 迁移=报告有问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 055

### 标题
`[PyTorch -> MindSpore][nn_RReLU] llm_enhanced_torch_nn_RReLU_20251216_010240.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_RReLU_20251216_010240.json_sample1.txt`
- 算子: `nn_RReLU`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_RReLU_20251216_010240.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_RReLU_20251216_010240.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：随机性行为未对齐所致
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.RReLU是随机化Leaky ReLU，在训练模式下对每个元素独立采样斜率a ~ Uniform(lower, upper)，且该采样在每次前向传播时重新进行（即非确定性、随机抖动）；而TensorFlow官方未提供等价的RReLU层（tf.keras.layers.LeakyReLU仅支持固定负斜率，无随机采样机制）。若用...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 056

### 标题
`[PyTorch -> Paddle][nn_ReplicationPad2d] llm_enhanced_torch_nn_ReplicationPad2d_20251202_004340.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_ReplicationPad2d_20251202_004340.json_sample1.txt`
- 算子: `nn_ReplicationPad2d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_ReplicationPad2d_20251202_004340.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_ReplicationPad2d_20251202_004340.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：输入数据未同步。这两种算子均为确定性的复制填充算子（Deterministic Copy），若输入一致，输出必一致。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.ReplicationPad2d(padding=2)在2D输入上执行对称的复制填充：对每个维度（H, W）在两侧各填充padding=2个像素，且填充值为边界像素的复制（即边缘值重复）。TensorFlow中无直接等价的tf.keras.layers.ReplicationPadding2D层；若使用tf.pad(mode...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 057

### 标题
`[PyTorch -> Paddle][nn_ReplicationPad2d] llm_enhanced_torch_nn_ReplicationPad2d_20251202_004340.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_ReplicationPad2d_20251202_004340.json_sample2.txt`
- 算子: `nn_ReplicationPad2d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_ReplicationPad2d_20251202_004340.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_ReplicationPad2d_20251202_004340.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：输入数据未同步。这两种算子均为确定性的复制填充算子（Deterministic Copy），若输入一致，输出必一致。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.ReplicationPad2d接受一个长度为4的padding列表[pad_left, pad_right, pad_top, pad_bottom]（即按左右上下顺序），而TensorFlow的tf.pad默认不支持直接等价的'CONSTANT'或'REFLECT'模式下的复制填充；若使用tf.pad配合tf.exper...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 058

### 标题
`[PyTorch -> Paddle][nn_ReplicationPad3d] llm_enhanced_torch_nn_ReplicationPad3d_20251202_132259.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_ReplicationPad3d_20251202_132259.json_sample1.txt`
- 算子: `nn_ReplicationPad3d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_ReplicationPad3d_20251202_132259.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_ReplicationPad3d_20251202_132259.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：输入数据未同步。这两种算子均为确定性的复制填充算子（Deterministic Copy），若输入一致，输出必一致。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.ReplicationPad3d采用'对称填充'（即沿每个空间维度在前后分别填充指定数量的元素，且填充内容为边界值的复制），其padding参数为单个整数时，表示在每个维度（D, H, W）的前后各填充该数值，即总填充为(3,3,3,3,3,3)——对应[front, back, top, bottom, left, rig...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 059

### 标题
`[PyTorch -> Paddle][nn_ReplicationPad3d] llm_enhanced_torch_nn_ReplicationPad3d_20251202_132259.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_ReplicationPad3d_20251202_132259.json_sample4.txt`
- 算子: `nn_ReplicationPad3d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_ReplicationPad3d_20251202_132259.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_ReplicationPad3d_20251202_132259.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.ReplicationPad3d采用'replicate'（边缘复制）填充策略，其padding参数顺序为(left, right, top, bottom, front, back)，对应空间维度(d1, d2, d3)的前后左右上下（即：(depth_front, depth_back, height_top, heig...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch ReplicationPad3d：内核严格遵循复制填充规则。Paddle Pad3D：内核默认采用常数填充（constant）策略（填充值默认为 0）

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 060

### 标题
`[PyTorch -> MindSpore][nn_TransformerEncoderLayer] llm_enhanced_torch_nn_TransformerEncoderLayer_20251215_184808.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_TransformerEncoderLayer_20251215_184808.json_sample1.txt`
- 算子: `nn_TransformerEncoderLayer`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_TransformerEncoderLayer_20251215_184808.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_TransformerEncoderLayer_20251215_184808.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数默认值不一致; 不一致=是
    - 原因摘要：默认参数未对齐，尤其是 `activation` 和 `norm_first` 两个核心参数的默认值在两个框架中不同，且影响了前向传播中的计算路径。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.TransformerEncoderLayer默认使用LayerNorm在残差连接后（即Post-LN），且其LayerNorm默认应用在最后一个维度（-1），权重初始化（如Linear层的kaiming_uniform_）和dropout行为（训练/评估模式、掩码生成方式、随机种子处理）与TensorFlow的tf.ker...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 061

### 标题
`[PyTorch -> MindSpore][nn_TransformerEncoderLayer] llm_enhanced_torch_nn_TransformerEncoderLayer_20251215_184808.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_TransformerEncoderLayer_20251215_184808.json_sample2.txt`
- 算子: `nn_TransformerEncoderLayer`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_TransformerEncoderLayer_20251215_184808.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_TransformerEncoderLayer_20251215_184808.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数默认值不一致; 不一致=是
    - 原因摘要：默认参数未对齐，尤其是 `activation` 和 `norm_first` 两个核心参数的默认值在两个框架中不同，且影响了前向传播中的计算路径。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow中没有直接等价于torch.nn.TransformerEncoderLayer的官方高层API；tf.keras.layers.MultiHeadAttention与tf.keras.layers.LayerNormalization等组合虽可构建类似结构，但默认行为存在关键差异：(1) PyTorch的TransformerEncod...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 062

### 标题
`[PyTorch -> MindSpore][nn_TransformerEncoderLayer] llm_enhanced_torch_nn_TransformerEncoderLayer_20251215_184808.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_TransformerEncoderLayer_20251215_184808.json_sample3.txt`
- 算子: `nn_TransformerEncoderLayer`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_TransformerEncoderLayer_20251215_184808.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nn_TransformerEncoderLayer_20251215_184808.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TransformerEncoderLayer 的参数初始化、基础算子计算细节、层归一化 eps 等框架级实现差异
  - 分配-陈桂学.xlsx: 迁移=初始化层时权重不一致

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 063

### 标题
`[PyTorch -> Paddle][nn_functional_adaptive_max_pool1d] llm_enhanced_torch_nn_functional_adaptive_max_pool1d_20251125_151911.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_adaptive_max_pool1d_20251125_151911.json_sample1.txt`
- 算子: `nn_functional_adaptive_max_pool1d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_adaptive_max_pool1d_20251125_151911.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_adaptive_max_pool1d_20251125_151911.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：输入数据完全不同（未固定随机种子）
  - 分配-陈桂学.xlsx: 迁移=输入参数问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 064

### 标题
`[PyTorch -> Paddle][nn_functional_adaptive_max_pool2d] llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample1.txt`
- 算子: `nn_functional_adaptive_max_pool2d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：输入数据完全不同（未固定随机种子）
  - 分配-陈桂学.xlsx: 迁移=输入参数问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 065

### 标题
`[PyTorch -> Paddle][nn_functional_adaptive_max_pool2d] llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample2.txt`
- 算子: `nn_functional_adaptive_max_pool2d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：输入数据完全不同（未固定随机种子）
  - 分配-陈桂学.xlsx: 迁移=输入参数问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 066

### 标题
`[PyTorch -> Paddle][nn_functional_adaptive_max_pool2d] llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample5.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample5.txt`
- 算子: `nn_functional_adaptive_max_pool2d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample5.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample5.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：池化算子是确定性的，若输入一致，输出必一致。差异值（6.0）远超精度误差，且 JSON 中仅有 Shape 信息无 Sample Values，表明两端使用了不同的随机输入。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow 中没有与 PyTorch 的 torch.nn.functional.adaptive_max_pool2d 完全等价的原生函数。PyTorch 的 adaptive_max_pool2d 会动态计算池化核大小和步长，以精确输出指定目标尺寸（此处为 [6, 4]），并支持非整除情形下的自适应划分（如对输入 H=8→output H=6，...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 067

### 标题
`[PyTorch -> Paddle][nn_functional_adaptive_max_pool2d] llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample6.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample6.txt`
- 算子: `nn_functional_adaptive_max_pool2d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample6.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample6.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=是; 不一致=是
    - 原因摘要：输入内容和默认值一致，但出现数值为1.0的差异
  - 分配-陈桂学.xlsx: 迁移=输入参数问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 068

### 标题
`[PyTorch -> Paddle][nn_functional_adaptive_max_pool2d] llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample7.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample7.txt`
- 算子: `nn_functional_adaptive_max_pool2d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample7.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample7.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：输入数据完全不同（未固定随机种子）
  - 分配-陈桂学.xlsx: 迁移=输入参数问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 069

### 标题
`[PyTorch -> Paddle][nn_functional_adaptive_max_pool2d] llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample8.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample8.txt`
- 算子: `nn_functional_adaptive_max_pool2d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample8.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample8.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：输入数据完全不同（未固定随机种子）
  - 分配-陈桂学.xlsx: 迁移=输入参数问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 070

### 标题
`[PyTorch -> Paddle][nn_functional_adaptive_max_pool2d] llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample9.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample9.txt`
- 算子: `nn_functional_adaptive_max_pool2d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample9.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_adaptive_max_pool2d_20251201_232545.json_sample9.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=是; 不一致=是
    - 原因摘要：输入的参数和默认值一致，但出现数值差异
  - 分配-陈桂学.xlsx: 迁移=输入参数问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 071

### 标题
`[PyTorch -> TensorFlow][nn_functional_avg_pool2d] llm_enhanced_torch_nn_functional_avg_pool2d_20251215_172318.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_avg_pool2d_20251215_172318.json_sample2.txt`
- 算子: `nn_functional_avg_pool2d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_functional_avg_pool2d_20251215_172318.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_functional_avg_pool2d_20251215_172318.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数未对齐; 不一致=是
    - 原因摘要：PyTorch 使用了 `padding=1`，而 TensorFlow 明确使用了 `padding="VALID"`，两者对输入空间的边界处理完全不同。（same填充0，valid不填充）
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的avg_pool2d默认使用'padding=1'（显式指定）且padding模式为'same'-like行为（即实际执行零填充），配合stride=2、kernel_size=3，在输入尺寸32x32下，输出尺寸计算公式为：floor((H + 2*pad - kernel_size) / stride) + 1 = floor((32 +...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 072

### 标题
`[PyTorch -> TensorFlow][nn_functional_avg_pool2d] llm_enhanced_torch_nn_functional_avg_pool2d_20251215_172318.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_avg_pool2d_20251215_172318.json_sample4.txt`
- 算子: `nn_functional_avg_pool2d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_functional_avg_pool2d_20251215_172318.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_functional_avg_pool2d_20251215_172318.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数未对齐; 不一致=是
    - 原因摘要：参数`padding=1` vs `"SAME"` 的不等价性
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow的tf.nn.avg_pool2d默认使用'NHWC'数据格式，而配置中指定了'data_format': 'NCHW'，但PyTorch的torch.nn.functional.avg_pool2d原生以NCHW格式处理（channel-first）。问题在于padding行为不一致：PyTorch中padding=1表示在每个空间维度...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 073

### 标题
`[PyTorch -> TensorFlow][nn_functional_avg_pool2d] llm_enhanced_torch_nn_functional_avg_pool2d_20251215_172318.json_sample6.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_avg_pool2d_20251215_172318.json_sample6.txt`
- 算子: `nn_functional_avg_pool2d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_functional_avg_pool2d_20251215_172318.json_sample6.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_functional_avg_pool2d_20251215_172318.json_sample6.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数未对齐; 不一致=是
    - 原因摘要：PyTorch 与 TensorFlow 的“padding”处理方式上存在语义差异，PyTorch：显式对称填充（`padding=1` → 总宽+2）；TensorFlow：隐式非对称填充（`padding="same"` → 自动调整填充）
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow的tf.nn.avg_pool2d在data_format='NCHW'下不支持padding='SAME'与stride=2、kernel_size=3组合下的精确等效行为；PyTorch的F.avg_pool2d(padding=1, stride=2, kernel_size=3, count_include_pad=True)明确...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 074

### 标题
`[PyTorch -> TensorFlow][nn_functional_avg_pool3d] llm_enhanced_torch_nn_functional_avg_pool3d_20251215_201404.json_sample5.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_avg_pool3d_20251215_201404.json_sample5.txt`
- 算子: `nn_functional_avg_pool3d`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_functional_avg_pool3d_20251215_201404.json_sample5.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_functional_avg_pool3d_20251215_201404.json_sample5.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow的tf.nn.avg_pool3d不支持'count_include_pad=True'（即默认对padding区域的零值也参与平均计算），而PyTorch的torch.nn.functional.avg_pool3d在count_include_pad=True（默认值）时，会将填充位置（padding=1）的零值纳入分母计数；但Ten...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 的padding=1是 D/H/W 维度对称固定填充，TF 的padding="SAME"是动态填充 TF 的ksize/strides需传入5 维列表（对应 N/C/D/H/W），而 PyTorch 仅需指定 D/H/W 维度

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 075

### 标题
`[PyTorch -> Paddle][nn_functional_batch_norm] llm_enhanced_torch_nn_functional_batch_norm_20251202_005716.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_batch_norm_20251202_005716.json_sample1.txt`
- 算子: `nn_functional_batch_norm`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_batch_norm_20251202_005716.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_batch_norm_20251202_005716.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow 的 tf.nn.batch_normalization（或 Keras BatchNormalization 层）在 training=True 模式下默认使用当前 batch 的均值和方差进行归一化，而 PyTorch 的 torch.nn.functional.batch_norm 在 training=True 时同样使用当前 b...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：Paddle 未显式指定 eps，默认值 1e-5 与 PyTorch 的 1e-7 不同

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 076

### 标题
`[PyTorch -> Paddle][nn_functional_conv2d] llm_enhanced_torch_nn_functional_conv2d_20251202_132551.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_conv2d_20251202_132551.json_sample1.txt`
- 算子: `nn_functional_conv2d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_conv2d_20251202_132551.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_conv2d_20251202_132551.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：计算累积误差。差异值在 3x10^{-5} 到 4x10^{-5} 之间，属于 Float32 卷积运算在不同后端（如 cuDNN vs MKL/OpenBLAS）或不同算法（Winograd vs GEMM）实现下的精度波动。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow配置字段为空（'TensorFlow配置' JSON 为 {}），未提供任何等效实现信息（如 tf.nn.conv2d 调用参数、weight/bias 初始化方式、数据格式、padding 处理、channel order 等）。PyTorch 默认使用 NCHW 格式，而 TensorFlow 默认使用 NHWC 格式；若未显式指定 ...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 077

### 标题
`[PyTorch -> Paddle][nn_functional_conv2d] llm_enhanced_torch_nn_functional_conv2d_20251202_132551.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_conv2d_20251202_132551.json_sample2.txt`
- 算子: `nn_functional_conv2d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_conv2d_20251202_132551.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_conv2d_20251202_132551.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：计算累积误差。差异值在 3x10^{-5} 到 4x10^{-5} 之间，属于 Float32 卷积运算在不同后端（如 cuDNN vs MKL/OpenBLAS）或不同算法（Winograd vs GEMM）实现下的精度波动。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.functional.conv2d要求输入张量格式为[N, C_in, H, W]（即channels-first），而TensorFlow的tf.nn.conv2d默认要求输入格式为[N, H, W, C_in]（即channels-last）。本例中PyTorch输入shape为[8, 64, 64, 128]，表示N=...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 078

### 标题
`[PyTorch -> Paddle][nn_functional_conv3d] llm_enhanced_torch_nn_functional_conv3d_20251201_233532.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_conv3d_20251201_233532.json_sample1.txt`
- 算子: `nn_functional_conv3d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_conv3d_20251201_233532.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_conv3d_20251201_233532.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：浮点累加误差。3D 卷积涉及大量的乘加运算，不同框架调用的底层库（如 cuDNN, MKL, OpenBLAS）计算顺序不同，导致 10^{-5} 级别的精度差异。
  - 分配-陈桂学.xlsx: 迁移=报告有问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 079

### 标题
`[PyTorch -> TensorFlow][nn_functional_gelu] llm_enhanced_torch_nn_functional_gelu_20251216_003428.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_gelu_20251216_003428.json_sample2.txt`
- 算子: `nn_functional_gelu`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_functional_gelu_20251216_003428.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_functional_gelu_20251216_003428.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.functional.gelu(input, approximate='tanh')与TensorFlow的tf.nn.gelu(x, approximate=True)虽均使用tanh近似，但二者实现的数学公式不同：PyTorch（>=1.12）采用精确的GELU定义的tanh近似形式：0.5 * x * (1 + tan...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：输入数值不一致

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 080

### 标题
`[PyTorch -> TensorFlow][nn_functional_interpolate] llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample1.txt`
- 算子: `nn_functional_interpolate`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：TensorFlow 的 `tf.nn.elu` 不支持自定义 alpha，始终使用 α=1.0。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.functional.interpolate默认采用NCHW格式（batch, channels, height, width），输入形状为(1, 512, 7, 7)，插值后输出为(1, 512, 56, 56)；而TensorFlow的tf.image.resize严格要求输入为NHWC格式（batch, height,...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 081

### 标题
`[PyTorch -> TensorFlow][nn_functional_interpolate] llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample2.txt`
- 算子: `nn_functional_interpolate`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.functional.interpolate默认采用NCHW格式（batch, channels, height, width），输入形状(1, 512, 7, 7)表示1个样本、512通道、7×7空间尺寸；插值后目标尺寸为[56, 56]，因此输出为(1, 512, 56, 56)。而TensorFlow的tf.image...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 输入为[N,C,H,W]（通道优先），TF 要求[N,H,W,C]（通道最后）。PyTorch 的align_corners=True需匹配 TF 的antialias=False。

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 082

### 标题
`[PyTorch -> TensorFlow][nn_functional_interpolate] llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample4.txt`
- 算子: `nn_functional_interpolate`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：输入：PyTorch默认 `NCHW`；TensorFlow默认 `NHWC`
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.functional.interpolate默认采用NCHW格式（batch, channel, height, width），输入形状为(1, 256, 14, 14)，插值后输出为(1, 256, 56, 56)；而TensorFlow的tf.image.resize严格要求输入为NHWC格式（batch, height...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 083

### 标题
`[PyTorch -> TensorFlow][nn_functional_interpolate] llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample5.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample5.txt`
- 算子: `nn_functional_interpolate`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample5.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample5.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.functional.interpolate默认采用NCHW格式（batch, channels, height, width），输入形状(1, 128, 28, 28)经插值到(56, 56)后输出为(1, 128, 56, 56)；而TensorFlow的tf.image.resize默认采用NHWC格式（batch, ...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 输入为[N,C,H,W]（通道优先），TF 要求[N,H,W,C]。PyTorch 的align_corners=True需匹配 TF 的antialias=False，否则插值规则不一致

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 084

### 标题
`[PyTorch -> TensorFlow][nn_functional_interpolate] llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample6.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample6.txt`
- 算子: `nn_functional_interpolate`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample6.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_functional_interpolate_20251215_230141.json_sample6.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：输入：PyTorch默认 `NCHW`；TensorFlow默认 `NHWC`
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.functional.interpolate默认采用NCHW格式（batch, channels, height, width），输入形状为(1, 128, 28, 28)，插值到(56, 56)后输出为(1, 128, 56, 56)。而TensorFlow的tf.image.resize默认采用NHWC格式（batch,...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 085

### 标题
`[PyTorch -> TensorFlow][nn_functional_leaky_relu] llm_enhanced_torch_nn_functional_leaky_relu_20251215_173249.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_leaky_relu_20251215_173249.json_sample4.txt`
- 算子: `nn_functional_leaky_relu`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_functional_leaky_relu_20251215_173249.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_functional_leaky_relu_20251215_173249.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：输入张量的数值不一致：PyTorch输入样本值为 -1.3410084255671362，而TensorFlow输入样本值为 -1.2989120995690806，二者相差约 0.042096，远超浮点计算误差（如机器精度或舍入误差）范围。LeakyReLU 是确定性逐元素操作，输出完全由输入和 alpha 决定；当输入不同，即使 dtype（float6...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：pytorch的值为-1.3410084255671362，TF是-1.2989120995690806，数值不一样

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 086

### 标题
`[PyTorch -> TensorFlow][nn_functional_local_response_norm] llm_enhanced_torch_nn_functional_local_response_norm_20251215_231102.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_local_response_norm_20251215_231102.json_sample1.txt`
- 算子: `nn_functional_local_response_norm`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_functional_local_response_norm_20251215_231102.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_functional_local_response_norm_20251215_231102.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数未对齐; 不一致=是
    - 原因摘要：在 PyTorch：`size=2` 表示邻域包含当前通道及其前后各一个通道 → 总共 3 个通道。在 TensorFlow：`depth_radius=2` 表示左右各延伸 2 个通道 → 总共 5 个通道。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.functional.local_response_norm中参数size=2表示局部响应归一化窗口在channel维度上的**半宽（half-width）**，即实际归一化窗口覆盖的通道数为2 * size + 1 = 5个通道（中心通道±2）；而TensorFlow的tf.nn.local_response_normal...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 087

### 标题
`[PyTorch -> TensorFlow][nn_functional_local_response_norm] llm_enhanced_torch_nn_functional_local_response_norm_20251215_231102.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_local_response_norm_20251215_231102.json_sample3.txt`
- 算子: `nn_functional_local_response_norm`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_functional_local_response_norm_20251215_231102.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_functional_local_response_norm_20251215_231102.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数未对齐; 不一致=是
    - 原因摘要：PyTorch 使用 `size=3`，表示邻域大小为3；而 TensorFlow 使用了 `depth_radius=3`，这意味着它会考虑 7 个通道 的平方和，远大于 PyTorch 的 3 个。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的local_response_norm中参数k（即bias）对应TensorFlow中bias参数，但PyTorch的size参数与TensorFlow的depth_radius参数语义不同：PyTorch的size表示'window size'（即归一化窗口包含的相邻通道数），为奇数，且实际半径r = (size-1)//2；而Tensor...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 088

### 标题
`[PyTorch -> Paddle][nn_functional_max_pool1d] llm_enhanced_torch_nn_functional_max_pool1d_20251202_130600.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_max_pool1d_20251202_130600.json_sample1.txt`
- 算子: `nn_functional_max_pool1d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_max_pool1d_20251202_130600.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_max_pool1d_20251202_130600.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=是; 不一致=是
    - 原因摘要：输入的参数和默认值一致，但出现数值差异
  - 分配-陈桂学.xlsx: 迁移=输入参数问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 089

### 标题
`[PyTorch -> Paddle][nn_functional_max_pool1d] llm_enhanced_torch_nn_functional_max_pool1d_20251202_130600.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_max_pool1d_20251202_130600.json_sample2.txt`
- 算子: `nn_functional_max_pool1d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_max_pool1d_20251202_130600.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_max_pool1d_20251202_130600.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：最大池化是确定性算子，差异值（4.07）表明两端输入的 Tensor 数据不同。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow 中没有直接等价于 PyTorch `max_pool1d` 且支持 `dilation` 参数的原生1D池化层。PyTorch 的 `F.max_pool1d` 支持 `dilation`（空洞率），而 TensorFlow 的 `tf.nn.max_pool1d`（TF 2.x）**不支持 `dilation` 参数**；其 dila...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 090

### 标题
`[PyTorch -> Paddle][nn_functional_max_pool1d] llm_enhanced_torch_nn_functional_max_pool1d_20251202_130600.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_max_pool1d_20251202_130600.json_sample3.txt`
- 算子: `nn_functional_max_pool1d`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_max_pool1d_20251202_130600.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_max_pool1d_20251202_130600.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow中没有直接等价于PyTorch的torch.nn.functional.max_pool1d（尤其是当dilation > 1时）的原生1D池化操作。tf.nn.max_pool1d仅支持stride和padding，但**不支持dilation参数**；其底层实现严格对应无空洞（dilation=1）的池化。而PyTorch的max_p...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 的 return_indices=false 是默认行为，Paddle 的 max_pool1d 暂不支持 return_indices 参数（影响是否返回最大值索引，不影响池化计算）

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 091

### 标题
`[PyTorch -> Paddle][nn_functional_rrelu] llm_enhanced_torch_nn_functional_rrelu_20251202_004720.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_rrelu_20251202_004720.json_sample1.txt`
- 算子: `nn_functional_rrelu`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_rrelu_20251202_004720.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_nn_functional_rrelu_20251202_004720.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.functional.rrelu在training=True时采用随机采样：对每个元素独立地从均匀分布U(lower, upper)中采样一个斜率a，然后计算输出为x if x >= 0 else a * x；而TensorFlow官方API（tf.nn.rrelu）虽语义相近，但其随机性实现与PyTorch不一致：(1) ...
  - 分配-陈建军.xlsx: 迁移=是; 不一致=是
    - 原因摘要：底层随机数生成器（PyTorch 为 MT19937、Paddle 为 Philox）不同，会导致采样的 a 值数值差异

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 092

### 标题
`[PyTorch -> TensorFlow][nn_functional_softplus] llm_enhanced_torch_nn_functional_softplus_20251216_002857.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_softplus_20251216_002857.json_sample1.txt`
- 算子: `nn_functional_softplus`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_functional_softplus_20251216_002857.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_functional_softplus_20251216_002857.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数未对齐; 不一致=是
    - 原因摘要：PyTorch支持参数 `beta`，TensorFlow不支持，默认等价于 `beta=1` 时的 PyTorch 版本
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.functional.softplus支持beta参数（softplus(x) = 1/beta * log(1 + exp(beta * x))），默认beta=1；而TensorFlow的tf.nn.softplus不接受beta参数，其定义为固定形式softplus(x) = log(1 + exp(x))，即等价于b...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 093

### 标题
`[PyTorch -> TensorFlow][nn_functional_softplus] llm_enhanced_torch_nn_functional_softplus_20251216_002857.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nn_functional_softplus_20251216_002857.json_sample2.txt`
- 算子: `nn_functional_softplus`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_nn_functional_softplus_20251216_002857.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_nn_functional_softplus_20251216_002857.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 已标记异常
    - 原因摘要：- 这一差异直接导致了归一化过程中的分母偏移，尤其是在输入数据分布较广、方差较小的情况下，对结果影响显著。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nn.functional.softplus支持beta和threshold参数，其中beta用于缩放输入（softplus(x) = 1/beta * log(1 + exp(beta * x))），threshold用于在输入较大时切换到线性近似（避免数值溢出）。而TensorFlow的tf.nn.softplus仅实现标准s...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 支持beta=2（斜率）和threshold=20（截断阈值），TF 原生softplus仅支持基础公式（beta=1、无截断）

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 094

### 标题
`[PyTorch -> MindSpore][nonzero] llm_enhanced_torch_nonzero_20251216_010320.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nonzero_20251216_010320.json_sample1.txt`
- 算子: `nonzero`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nonzero_20251216_010320.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nonzero_20251216_010320.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 因as_tuple=True返回单元素元组格式索引，MindSpore 默认返回二维张量格式索引，仅数据结构形式不同
  - 分配-陈桂学.xlsx: 迁移=本身输出维度的问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 095

### 标题
`[PyTorch -> MindSpore][nonzero] llm_enhanced_torch_nonzero_20251216_010320.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nonzero_20251216_010320.json_sample2.txt`
- 算子: `nonzero`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nonzero_20251216_010320.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nonzero_20251216_010320.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nonzero(input, as_tuple=True)对一维输入（shape=[5]）返回一个包含单个张量的元组，该张量形状为(4, 1)，其中4是非零元素个数，1是输入维度数（即列向量形式，每行是一个非零索引的坐标）。但问题描述中错误地将PyTorch行为写为'(1, 4)'，而实际PyTorch行为是：当as_tuple=...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch torch.nonzero(as_tuple=True) 内核返回元组形式的非零元素索引（一维输入下为 (tensor([0,1,2,3,4]),)）；MindSpore ops.NonZero() 内核默认返回二维张量形式的非零元素索引（一维输入下为 [[0],[1],[2],[3],[4]]），二者返回结构的内核生成逻辑完全不同

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 096

### 标题
`[PyTorch -> MindSpore][nonzero] llm_enhanced_torch_nonzero_20251216_010320.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nonzero_20251216_010320.json_sample3.txt`
- 算子: `nonzero`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nonzero_20251216_010320.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nonzero_20251216_010320.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 因as_tuple=True返回 3 个空张量的元组，MindSpore 返回形状(0,3)的空二维张量，仅数据结构形式不同
  - 分配-陈桂学.xlsx: 迁移=本身输出维度的问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 097

### 标题
`[PyTorch -> MindSpore][nonzero] llm_enhanced_torch_nonzero_20251216_010320.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nonzero_20251216_010320.json_sample4.txt`
- 算子: `nonzero`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nonzero_20251216_010320.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nonzero_20251216_010320.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.nonzero(input, as_tuple=True)对一维输入（shape=[5]）返回一个包含单个张量的元组，该张量形状为(5,)（即每个非零元素的索引，按行优先顺序排列），而非(5, 1)；但问题中错误描述提到'MindSpore (5, 1)'，说明实际对比对象是MindSpore而非TensorFlow（题干要求分析...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch torch.nonzero(as_tuple=True) 内核返回元组形式的非零元素索引（一维输入下为 (tensor([0,1,2,3,4]),)）；MindSpore ops.NonZero() 内核默认返回二维张量形式的非零元素索引（一维输入下为 [[0],[1],[2],[3],[4]]），二者返回结构的内核生成逻辑完全不同

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 098

### 标题
`[PyTorch -> MindSpore][nonzero] llm_enhanced_torch_nonzero_20251216_010320.json_sample5.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_nonzero_20251216_010320.json_sample5.txt`
- 算子: `nonzero`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nonzero_20251216_010320.json_sample5.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_nonzero_20251216_010320.json_sample5.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 因as_tuple=True返回元组格式索引，MindSpore 默认返回二维张量格式索引，数据结构形式不同
  - 分配-陈桂学.xlsx: 迁移=本身输出维度的问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 099

### 标题
`[PyTorch -> TensorFlow][pow] llm_enhanced_torch_pow_20251215_204420.json_sample6.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_pow_20251215_204420.json_sample6.txt`
- 算子: `pow`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_pow_20251215_204420.json_sample6.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_pow_20251215_204420.json_sample6.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：NaN问题而且迁移的数据存在不一致问题
  - 分配-陈桂学.xlsx: 迁移=数据不一致

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 100

### 标题
`[PyTorch -> TensorFlow][pow] llm_enhanced_torch_pow_20251215_204420.json_sample7.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_pow_20251215_204420.json_sample7.txt`
- 算子: `pow`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_pow_20251215_204420.json_sample7.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_pow_20251215_204420.json_sample7.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch和TensorFlow的输入张量内容完全不一致：PyTorch的input.sample_values（前10个）为[0.947, -1.568, -0.790, 0.490, -0.336, -0.448, -1.378, -0.435, 2.121, -0.240]，而TensorFlow的x.sample_values（前10个）为[0....
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：输入数值不一致

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 101

### 标题
`[PyTorch -> Paddle][quantile] llm_enhanced_torch_quantile_20251202_123958.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_quantile_20251202_123958.json_sample3.txt`
- 算子: `quantile`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_quantile_20251202_123958.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_quantile_20251202_123958.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.quantile(input, q, dim=0, keepdim=True)在输入shape为(5,5,5)时，沿dim=0（即第一个维度，大小为5）计算分位数，结果shape应为(1,5,5)（因keepdim=True，dim=0被保留为长度1）。而PaddlePaddle报错显示输出为(1,1,1)，表明其实际执行了全局分...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 显式指定 dim=0（沿第 0 维计算分位数），Paddle 默认对全部元素计算全局分位数，而非沿第 0 维计算

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 102

### 标题
`[PyTorch -> TensorFlow][rand] llm_enhanced_torch_rand_20251215_231651.json_sample6.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_rand_20251215_231651.json_sample6.txt`
- 算子: `rand`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_rand_20251215_231651.json_sample6.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_rand_20251215_231651.json_sample6.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：二者在种子（seed）初始化机制和伪随机数生成器的内部实现细节上存在根本性差异，即使输入完全一致，也无法保证输出数值严格一致
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.rand生成的是[0, 1)区间内均匀分布的随机数（左闭右开，即包含0.0，不包含1.0），而TensorFlow的tf.random.uniform默认生成的是[minval, maxval)区间（也是左闭右开），因此区间定义本身一致。但关键差异在于：PyTorch的torch.rand在固定seed下使用其内部独立的伪随机数生...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 103

### 标题
`[PyTorch -> Paddle][rand_like] llm_enhanced_torch_rand_like_20251125_143509.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_rand_like_20251125_143509.json_sample4.txt`
- 算子: `rand_like`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_rand_like_20251125_143509.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_rand_like_20251125_143509.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：未固定随机种子，随机数生成器的实现差异导致生成的序列不同
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.rand_like(input) 会生成与input同形状、同dtype（此处为float64）、且在[0,1)均匀分布的随机张量。但给定input为标量（shape=[]）且sample_values=[1.0]，torch.rand_like将返回一个float64标量随机值（如0.372...）。而TensorFlow配置为...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 104

### 标题
`[PyTorch -> Paddle][rand_like] llm_enhanced_torch_rand_like_20251125_143509.json_sample5.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_rand_like_20251125_143509.json_sample5.txt`
- 算子: `rand_like`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_rand_like_20251125_143509.json_sample5.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_rand_like_20251125_143509.json_sample5.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.rand_like生成的是在[0,1)区间内均匀分布的随机浮点数，而非基于输入张量值的采样；其行为与输入张量的sample_values完全无关——sample_values仅用于调试/参考，不参与随机数生成。而错误描述中出现显著数值差异（最大差值≈0.6），表明TensorFlow端很可能误用了tf.random.uniform...
  - 分配-陈建军.xlsx: 迁移=是; 不一致=是
    - 原因摘要：底层随机数生成器（MT19937 vs Philox）不同导致随机数值差异

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 105

### 标题
`[PyTorch -> Paddle][randn] llm_enhanced_torch_randn_20251202_005904.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_randn_20251202_005904.json_sample3.txt`
- 算子: `randn`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randn_20251202_005904.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randn_20251202_005904.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow配置为空（空JSON对象），未提供任何等效实现（如tf.random.normal或tf.random.uniform）的参数信息，导致无法复现PyTorch的torch.randn行为。PyTorch配置明确指定了形状[1,512,4,4]、dtype（应为torch.float32或类似）、requires_grad=True（即需梯...
  - 分配-陈建军.xlsx: 迁移=是; 不一致=是
    - 原因摘要：输入的api是一样的，如果是Paddle和Pytorch，则随机数生成逻辑也不一致

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 106

### 标题
`[PyTorch -> Paddle][randn] llm_enhanced_torch_randn_20251202_005904.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_randn_20251202_005904.json_sample4.txt`
- 算子: `randn`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randn_20251202_005904.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randn_20251202_005904.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow配置为空（空JSON对象），未提供任何等效实现；PyTorch使用torch.randn生成标准正态分布随机张量（dtype=torch.float64，shape=[2,1,4,1,4]），而TensorFlow中缺失对应调用（如tf.random.normal(seed=..., dtype=tf.float64)）。由于未指定随机种...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：随机数生成算法（MT19937 vs Philox），即使种子同步，生成的正态分布数值也无法一致

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 107

### 标题
`[PyTorch -> Paddle][randn_like] llm_enhanced_torch_randn_like_20251125_141142.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_randn_like_20251125_141142.json_sample2.txt`
- 算子: `randn_like`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randn_like_20251125_141142.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randn_like_20251125_141142.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：未固定随机种子，随机数生成器的实现差异导致生成的序列不同
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.randn_like生成服从标准正态分布N(0,1)的随机浮点数，且dtype=float64（即float64精度）；而TensorFlow中若未显式指定dtype和随机种子，tf.random.normal默认dtype为float32（除非显式设为tf.float64），且随机数生成器的底层算法（如Philox vs. MT...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 108

### 标题
`[PyTorch -> Paddle][randn_like] llm_enhanced_torch_randn_like_20251125_141142.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_randn_like_20251125_141142.json_sample3.txt`
- 算子: `randn_like`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randn_like_20251125_141142.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randn_like_20251125_141142.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：未固定随机种子，随机数生成器的实现差异导致生成的序列不同
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.randn_like生成的是标准正态分布（均值0、标准差1）的随机张量，其输出 dtype 严格继承自输入张量的 dtype（此处为 float64），且采样算法（Philox + Box-Muller 或类似）与 TensorFlow 的 tf.random.normal 默认行为存在底层实现差异：TensorFlow 1.x/...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 109

### 标题
`[PyTorch -> Paddle][randn_like] llm_enhanced_torch_randn_like_20251125_141142.json_sample5.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_randn_like_20251125_141142.json_sample5.txt`
- 算子: `randn_like`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randn_like_20251125_141142.json_sample5.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randn_like_20251125_141142.json_sample5.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：未固定随机种子，随机数生成器的实现差异导致生成的序列不同
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.randn_like生成的是标准正态分布（均值0、方差1）的随机张量，其数值完全随机且不可复现（除非设置相同随机种子和随机数生成器状态）；而问题中TensorFlow配置为空（无对应tf.random.normal或等效调用），说明未实现任何随机采样操作，导致输出可能为零张量、未初始化内存、默认填充值或报错。此外，即使补全为tf....

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 110

### 标题
`[PyTorch -> Paddle][randperm] llm_enhanced_torch_randperm_20251202_133813.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_randperm_20251202_133813.json_sample1.txt`
- 算子: `randperm`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randperm_20251202_133813.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randperm_20251202_133813.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：生成随机排列，两端 RNG 未对齐，生成的序列完全不同。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow中没有直接对应torch.randperm的等效API；torch.randperm(n)生成0到n-1的随机排列（整数张量），而TensorFlow默认不提供原生的、行为完全一致的随机排列操作。tf.random.shuffle仅对输入张量的第0维进行随机打乱，不能直接生成[0, 1, ..., n-1]的排列；若用户误用tf.rang...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 111

### 标题
`[PyTorch -> Paddle][randperm] llm_enhanced_torch_randperm_20251202_133813.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_randperm_20251202_133813.json_sample2.txt`
- 算子: `randperm`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randperm_20251202_133813.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randperm_20251202_133813.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：生成随机排列，两端 RNG 未对齐，生成的序列完全不同。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow中没有直接等价于PyTorch的torch.randperm(n, generator=None)的内置函数。PyTorch的torch.randperm(n)生成0到n-1的随机排列，其底层使用Philox RNG（在CUDA上）或默认CPU RNG，并且当generator为None时，依赖全局随机状态（受torch.manual_s...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 112

### 标题
`[PyTorch -> Paddle][randperm] llm_enhanced_torch_randperm_20251202_133813.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_randperm_20251202_133813.json_sample3.txt`
- 算子: `randperm`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randperm_20251202_133813.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randperm_20251202_133813.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：生成随机排列，两端 RNG 未对齐，生成的序列完全不同。
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow 中没有直接等价于 PyTorch torch.randperm(n, generator=None) 的原生 API。torch.randperm 生成 0 到 n-1 的随机排列（整数序列），其随机性依赖于全局或指定的 torch.Generator，且默认使用确定性/可复现的伪随机算法（如 Philox 或 CPU backend ...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 113

### 标题
`[PyTorch -> Paddle][randperm] llm_enhanced_torch_randperm_20251202_133813.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_randperm_20251202_133813.json_sample4.txt`
- 算子: `randperm`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randperm_20251202_133813.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_randperm_20251202_133813.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow中没有直接对应torch.randperm的API；torch.randperm(n)生成0到n-1的随机排列（整数张量），而TensorFlow需使用tf.random.shuffle(tf.range(n))才能等效实现。但问题中TensorFlow配置为空（空字符串或未提供任何代码），说明未实现任何迁移逻辑，导致无输出或默认行为（如...
  - 分配-陈建军.xlsx: 迁移=是; 不一致=是
    - 原因摘要：底层随机数生成器不同导致排列结果数值差异

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 114

### 标题
`[PyTorch -> Paddle][range] llm_enhanced_torch_range_20251202_132606.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_range_20251202_132606.json_sample1.txt`
- 算子: `range`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_range_20251202_132606.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_range_20251202_132606.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.range(start=1, end=20)生成包含start到end（含）的等差序列，即[1, 2, ..., 20]，共20个元素，形状为(20,)。而PaddlePaddle（题目中误写为TensorFlow配置为空，但错误描述明确指出是PaddlePaddle (19,)）的paddle.arange(start=1, e...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：输入的api是一样的，且rangePytorch的内核计算时会遵循闭区间而Paddle是左闭右开

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 115

### 标题
`[PyTorch -> Paddle][range] llm_enhanced_torch_range_20251202_132606.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_range_20251202_132606.json_sample2.txt`
- 算子: `range`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_range_20251202_132606.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_range_20251202_132606.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.range(start, end, step)生成包含end的闭区间序列（即当end能被step整除时，末尾值恰好为end），其行为等价于torch.arange(start, end + step, step)，但实际实现中会包含所有满足start + k*step <= end的k对应值。对于start=-10.5, end=...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch torch.range 内核遵循闭区间逻辑（包含 end 值），生成的序列包含 -10.5, -10.0, ..., 10.0, 10.5（共 43 个元素）；Paddle paddle.arange 内核遵循左闭右开逻辑（不包含 end 值），生成的序列包含 -10.5, -10.0, ..., 10.0（共 42 个元素）

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 116

### 标题
`[PyTorch -> TensorFlow][std] llm_enhanced_torch_std_20251215_210017.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_std_20251215_210017.json_sample2.txt`
- 算子: `std`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_std_20251215_210017.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_std_20251215_210017.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.std默认计算样本标准差（ddof=1），即无偏估计，分母为n-1；而TensorFlow的tf.math.reduce_std默认计算总体标准差（ddof=0），即分母为n。二者在数值上存在系统性偏差。当输入张量在被约简的维度（axis=[0,2]）上存在全零方差或极小方差、或因维度退化导致有效元素数为1时，PyTorch在dd...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 默认计算无偏标准差（除以 N-1），TF 默认计算有偏标准差（除以 N）。出现NaN

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 117

### 标题
`[PyTorch -> TensorFlow][std] llm_enhanced_torch_std_20251215_210017.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_std_20251215_210017.json_sample3.txt`
- 算子: `std`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_std_20251215_210017.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_std_20251215_210017.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch torch.std：默认 ddof=1（无偏估计，除以N−1，适用于样本标准差）；TensorFlow tf.math.reduce_std：默认ddof=0（有偏估计，除以N，适用于总体标准差）。
  - 分配-陈桂学.xlsx: 迁移=API映射
    - 原因摘要：无偏估计 (Unbiased) 与 有偏估计 (Biased) 之间的算法冲突

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 118

### 标题
`[PyTorch -> TensorFlow][std] llm_enhanced_torch_std_20251215_210017.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_std_20251215_210017.json_sample4.txt`
- 算子: `std`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_std_20251215_210017.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_std_20251215_210017.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=参数默认值不一致; 不一致=是
    - 原因摘要：两框架在默认的自由度修正（correction）策略上存在关键差异
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.std默认使用Bessel's correction（无偏估计），即除以(n-1)，而TensorFlow的tf.math.reduce_std默认使用总体标准差（有偏估计），即除以n。两者在计算标准差时自由度不同，导致数值差异。该差异（最大0.0005277）符合小样本下(n-1) vs n校正引起的相对误差量级。要使迁移一致，...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 119

### 标题
`[PyTorch -> TensorFlow][sub] llm_enhanced_torch_sub_20251215_165148.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_sub_20251215_165148.json_sample3.txt`
- 算子: `sub`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_sub_20251215_165148.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_sub_20251215_165148.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：迁移的数据有问题，数据都不一一对应
  - 分配-陈桂学.xlsx: 迁移=输入参数问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 120

### 标题
`[PyTorch -> TensorFlow][sub] llm_enhanced_torch_sub_20251215_165148.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_sub_20251215_165148.json_sample4.txt`
- 算子: `sub`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_sub_20251215_165148.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_sub_20251215_165148.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.sub(input, other, alpha=2)执行的是 input - alpha * other，即逐元素减法并支持标量缩放因子alpha；而TensorFlow的tf.subtract(x, y)仅执行x - y，不支持alpha参数。本例中PyTorch实际计算为 input - 2 * other，但TensorFl...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：输入数值不一致

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 121

### 标题
`[PyTorch -> TensorFlow][svd] llm_enhanced_torch_svd_20251215_205358.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_svd_20251215_205358.json_sample3.txt`
- 算子: `svd`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_svd_20251215_205358.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_svd_20251215_205358.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=是; 不一致=是
    - 原因摘要：这个好像是由于数据形状不规整导致的运行时错误，不是结果问题
  - 分配-陈桂学.xlsx: 迁移=比较过程问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 122

### 标题
`[PyTorch -> TensorFlow][svd] llm_enhanced_torch_svd_20251215_205358.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_svd_20251215_205358.json_sample4.txt`
- 算子: `svd`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_svd_20251215_205358.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_svd_20251215_205358.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=是; 不一致=是
    - 原因摘要：这个好像是由于数据形状不规整导致的运行时错误，不是结果问题
  - 分配-陈桂学.xlsx: 迁移=比较过程问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 123

### 标题
`[PyTorch -> MindSpore][tanh] llm_enhanced_torch_tanh_20251216_010955.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_tanh_20251216_010955.json_sample2.txt`
- 算子: `tanh`
- 框架对: `PyTorch -> MindSpore`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_tanh_20251216_010955.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2ms-comparison_a_error/comparison_a/llm_enhanced_torch_tanh_20251216_010955.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：疑似框架问题
  - 分配-林哲远.xlsx: 迁移=是; 不一致=是
    - 原因摘要：PyTorch和TensorFlow的tanh实现在数值上理论上完全一致（均遵循IEEE 754 float32标准及相同数学定义），但实际观测到的最大差异4.536e-05属于典型浮点计算可接受的微小偏差范畴。该差异源于底层实现细节差异：如运算顺序（reduction order）、中间结果舍入策略、SIMD向量化路径选择、或CUDA cuBLAS/cuD...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 124

### 标题
`[PyTorch -> Paddle][tensordot] llm_enhanced_torch_tensordot_20251125_141922.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_tensordot_20251125_141922.json_sample2.txt`
- 算子: `tensordot`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_tensordot_20251125_141922.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_tensordot_20251125_141922.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：输入的样本值完全不一样
  - 分配-陈桂学.xlsx: 迁移=输入参数问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 125

### 标题
`[PyTorch -> TensorFlow][topk] llm_enhanced_torch_topk_20251215_164432.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_topk_20251215_164432.json_sample2.txt`
- 算子: `topk`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_topk_20251215_164432.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_topk_20251215_164432.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.topk(..., dim=-1, largest=False, sorted=True) 在最后一个维度（dim=-1，即每行）上查找最小的k=1个元素，并返回按值升序排列的结果（因largest=False且sorted=True，故返回最小值且索引对应升序位置）。而TensorFlow的tf.math.top_k默认行为是l...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 支持largest=False直接取最小值，TF 仅支持取最大值

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 126

### 标题
`[PyTorch -> Paddle][unique] llm_enhanced_torch_unique_20251202_135202.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_unique_20251202_135202.json_sample3.txt`
- 算子: `unique`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_unique_20251202_135202.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_unique_20251202_135202.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：输入数据完全不同（未固定随机种子）
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow中没有提供与PyTorch的torch.unique完全等价的默认API。torch.unique在PyTorch中默认返回唯一值（按输入顺序首次出现位置排序）、索引（original indices）、逆索引（inverse indices）和计数（counts），且其行为依赖于dtype（如float64）和排序语义（默认sorted...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 127

### 标题
`[PyTorch -> Paddle][unique] llm_enhanced_torch_unique_20251202_135202.json_sample6.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_unique_20251202_135202.json_sample6.txt`
- 算子: `unique`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_unique_20251202_135202.json_sample6.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_unique_20251202_135202.json_sample6.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=是; 不一致=是
    - 原因摘要：输入的参数和默认值一致，但出现数值差异
  - 分配-陈桂学.xlsx: 迁移=输入参数问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 128

### 标题
`[PyTorch -> TensorFlow][unique] llm_enhanced_torch_unique_20251216_003709.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_unique_20251216_003709.json_sample1.txt`
- 算子: `unique`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_unique_20251216_003709.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_unique_20251216_003709.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.unique返回两个张量：(values, indices)，其中values形状为(4,)（一维，长度为唯一值个数），indices形状为(4,)（原输入索引映射）；而TensorFlow的tf.unique也返回(values, indices)，但其values形状为(4,)，indices形状也为(4,)，二者输出结构一致...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 默认仅返回 “唯一值张量”。TF 始终返回(唯一值张量, 原张量索引张量)

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 129

### 标题
`[PyTorch -> TensorFlow][unique] llm_enhanced_torch_unique_20251216_003709.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_unique_20251216_003709.json_sample2.txt`
- 算子: `unique`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_unique_20251216_003709.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_unique_20251216_003709.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：两个框架在 `unique` 操作的返回值设计上存在结构性差异——PyTorch 支持灵活的返回形式（可选逆索引/计数），而 TensorFlow 强制返回 `(y, idx)` 元组，且无选项关闭
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.unique返回两个张量：(values, indices)，其中values形状为(N,)（N为唯一元素个数），indices形状为(input_shape,)（与输入同形）。而TensorFlow的tf.unique也返回(values, indices)，但其indices默认返回int32类型且形状为(input_shap...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 130

### 标题
`[PyTorch -> TensorFlow][unique] llm_enhanced_torch_unique_20251216_003709.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_unique_20251216_003709.json_sample3.txt`
- 算子: `unique`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_unique_20251216_003709.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_unique_20251216_003709.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：两个框架在 `unique` 操作的返回值设计上存在结构性差异——PyTorch 支持灵活的返回形式（可选逆索引/计数），而 TensorFlow 强制返回 `(y, idx)` 元组，且无选项关闭
  - 分配-陈桂学.xlsx: 迁移=本身输出维度的问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 131

### 标题
`[PyTorch -> TensorFlow][unique] llm_enhanced_torch_unique_20251216_003709.json_sample8.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_unique_20251216_003709.json_sample8.txt`
- 算子: `unique`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_unique_20251216_003709.json_sample8.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_unique_20251216_003709.json_sample8.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=是; 不一致=是
    - 原因摘要：两个框架在 `unique` 操作的返回值设计上存在结构性差异——PyTorch 支持灵活的返回形式（可选逆索引/计数），而 TensorFlow 强制返回 `(y, idx)` 元组，且无选项关闭
  - 分配-陈桂学.xlsx: 迁移=输入参数问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 132

### 标题
`[PyTorch -> Paddle][unique_consecutive] llm_enhanced_torch_unique_consecutive_20251125_142753.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_unique_consecutive_20251125_142753.json_sample1.txt`
- 算子: `unique_consecutive`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-林哲远.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_unique_consecutive_20251125_142753.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_unique_consecutive_20251125_142753.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：未固定随机种子，随机数生成器的实现差异导致生成的序列不同
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：TensorFlow 中没有直接等价于 PyTorch 的 torch.unique_consecutive 的原生 API。tf.unique() 仅实现全局去重（保持首次出现顺序），但不支持'连续重复元素压缩'语义；而 torch.unique_consecutive 专门针对连续相同元素进行分组去重（即只合并相邻相等元素），且保留原始顺序和连续段结构。...

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 133

### 标题
`[PyTorch -> Paddle][unique_consecutive] llm_enhanced_torch_unique_consecutive_20251125_142753.json_sample3.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_unique_consecutive_20251125_142753.json_sample3.txt`
- 算子: `unique_consecutive`
- 框架对: `PyTorch -> Paddle`
- 跨表确认: 分配-朱婷.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_unique_consecutive_20251125_142753.json_sample3.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/pt2pd-comparison_a_error/comparison_a/llm_enhanced_torch_unique_consecutive_20251125_142753.json_sample3.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-朱婷.xlsx: 迁移=输入未对齐; 不一致=是
    - 原因摘要：未固定随机种子，导致两端生成的随机输入不同，从而输出了完全不同的数值。
  - 分配-陈桂学.xlsx: 迁移=输入参数问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 134

### 标题
`[PyTorch -> TensorFlow][var] llm_enhanced_torch_var_20251216_000004.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_var_20251216_000004.json_sample2.txt`
- 算子: `var`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_var_20251216_000004.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_var_20251216_000004.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch torch.var：默认 ddof=1（无偏估计，除以 N−1，N=4时分母为3）；TensorFlow tf.math.reduce_variance：默认 ddof=0（有偏估计，除以N，N=4时分母为4）；
  - 分配-陈桂学.xlsx: 迁移=API映射
    - 原因摘要：无偏估计 (Unbiased) 与 有偏估计 (Biased) 之间的算法冲突

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 135

### 标题
`[PyTorch -> TensorFlow][var] llm_enhanced_torch_var_20251216_000004.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_var_20251216_000004.json_sample4.txt`
- 算子: `var`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_var_20251216_000004.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_var_20251216_000004.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：默认自由度ddof不同（PyTorch=1，TensorFlow=0），且本次计算维度长度为 1，导致 PyTorch 输出nan、TensorFlow 输出0
  - 分配-陈桂学.xlsx: 迁移=比较过程问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 136

### 标题
`[PyTorch -> TensorFlow][var_mean] llm_enhanced_torch_var_mean_20251215_173951.json_sample1.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_var_mean_20251215_173951.json_sample1.txt`
- 算子: `var_mean`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-陈建军.xlsx, 分配-陈桂学.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_var_mean_20251215_173951.json_sample1.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_var_mean_20251215_173951.json_sample1.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：torch.var_mean有unbiased=true指定无偏估计，而tf.nn.moments默认是计算有偏方差
  - 分配-陈桂学.xlsx: 迁移=本身输出维度的问题

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 137

### 标题
`[PyTorch -> TensorFlow][var_mean] llm_enhanced_torch_var_mean_20251215_173951.json_sample2.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_var_mean_20251215_173951.json_sample2.txt`
- 算子: `var_mean`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_var_mean_20251215_173951.json_sample2.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_var_mean_20251215_173951.json_sample2.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.var_mean(dim=0, unbiased=True, keepdim=True) 计算的是沿维度0（即batch维度，大小为5）的无偏样本方差（分母为n-1=4）和均值；而TensorFlow的tf.nn.moments(axes=0, keepdims=True) 默认计算的是二阶中心矩（即有偏方差，分母为n=5），其v...
  - 分配-陈建军.xlsx: 迁移=是; 不一致=是
    - 原因摘要：Pytorch的dim与TF的axes保持功能类似，而数据不一致

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

## Issue 138

### 标题
`[PyTorch -> TensorFlow][var_mean] llm_enhanced_torch_var_mean_20251215_173951.json_sample4.txt 在等价迁移下输出差异异常`

### 背景
- Case: `llm_enhanced_torch_var_mean_20251215_173951.json_sample4.txt`
- 算子: `var_mean`
- 框架对: `PyTorch -> TensorFlow`
- 跨表确认: 分配-林哲远.xlsx, 分配-陈建军.xlsx

### 问题描述
- 在该测试样例中，A/B框架对同一算子的结果被多位评审重复标记为不一致。
- 当前现象已超过单纯数值误差范围，疑似存在参数映射、输入对齐或语义适配缺陷。

### 影响范围
- 直接影响该算子在跨框架迁移验证中的可信度。
- 可能导致回归测试出现持续误报或漏报。

### 复现材料
- 历史测试文件(full_path):
  - `bug/pt2tf-comparison_error/comparison_a/llm_enhanced_torch_var_mean_20251215_173951.json_sample4.txt`
- 本地可回溯文件:
  - `/Users/linzheyuan/code/TransTest/filecheck/comparison_a/llm_enhanced_torch_var_mean_20251215_173951.json_sample4.txt`

### 复现步骤
1. 打开上述测试文件，读取输入、参数和目标算子配置。
2. 在A框架与B框架分别执行同一测试。
3. 记录输出shape、max diff、异常值(NaN/Inf)、索引差异（若适用）。
4. 按评审意见逐项对齐关键参数后复测，比较差异变化。

### 实际结果
- 多份评审均标记“迁移异常/存在不一致”。
- 评审摘要：
  - 分配-林哲远.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch的torch.var_mean(..., unbiased=True) 计算的是无偏样本方差（即除以 n-1），而 TensorFlow 的 tf.nn.moments 默认计算的是二阶中心矩（即有偏方差，除以 n）。tf.nn.moments 的 variance 输出等价于 torch.var(..., unbiased=False)。虽然...
  - 分配-陈建军.xlsx: 迁移=否; 不一致=是
    - 原因摘要：PyTorch 默认计算无偏方差（除以 N-1），TF 默认计算有偏方差（除以 N）

### 期望结果
- 在等价输入和等价参数下，A/B框架结果应保持数值接近或语义一致。
- 若存在已知不可对齐语义，应在迁移规则中显式标注并从数值一致性比较中剔除。

### 验收标准
- 关键参数映射和输入对齐完成后，该case的差异降至阈值内或被正确归类为“不可比语义”。
- 回归任务对该case不再重复报同类异常。

### 建议标签
`migration` `cross-framework` `consistency` `needs-triage`

---

