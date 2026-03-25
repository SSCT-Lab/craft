Add a title
*
Title
[CPU] 多个算子在 float16 输入下报 NotFound: kernel not registered（pd_ms_test_1 复现）

在向Paddle报bug之前，请先查询历史issue是否报过同样的bug。
Before submitting a bug, please make sure the issue hasn't been already addressed by searching through the existing and past issues.

bug描述 Describe the Bug
*
在 Windows CPU 环境下，多个 Paddle 算子在 `float16` 输入时直接报错：
`RuntimeError: (NotFound) The kernel with key (CPU, Undefined(AnyLayout), float16) of kernel <op> is not registered.`

同一组输入在对照框架（MindSpore）可执行成功，且本仓库最小复现脚本均出现 `REPRODUCED_PADDLE_ERROR_ONLY=True`。

涉及脚本（可逐个运行）：
- `pd_ms_test_1/simple_tests/pd_error_only_candidate_038_baddbmm.py`
- `pd_ms_test_1/simple_tests/pd_error_only_candidate_084_cumsum.py`
- `pd_ms_test_1/simple_tests/pd_error_only_candidate_150_logaddexp.py`（内部触发 `subtract` float16 kernel 缺失）
- `pd_ms_test_1/simple_tests/pd_error_only_candidate_210_multiply.py`
- `pd_ms_test_1/simple_tests/pd_error_only_candidate_305_prod.py`
- `pd_ms_test_1/simple_tests/pd_error_only_candidate_314_remainder.py`
- `pd_ms_test_1/simple_tests/pd_error_only_candidate_360_sinc.py`（内部触发 `where` float16 kernel 缺失）
- `pd_ms_test_1/simple_tests/pd_error_only_candidate_367_square.py`

Markdown Editor
Markdown input: edit mode selected.
Write
Preview
请清晰简洁的描述这个bug。A clear and concise description of what the bug is.

```python
# 最小可复现代码。Sample code to reproduce the problem.
import numpy as np
import paddle

print("paddle:", paddle.__version__)

x = paddle.to_tensor(np.ones((2, 3), dtype=np.float16))
y = paddle.to_tensor(np.ones((2, 3), dtype=np.float16))

# 示例算子之一：multiply（其它算子见上面的脚本列表）
out = paddle.multiply(x, y)
print(out)
```

```shell
带有完整回溯的报错信息。The error message you got, with the full traceback.
RuntimeError: (NotFound) The kernel with key (CPU, Undefined(AnyLayout), float16)
of kernel `multiply` is not registered. Selected wrong DataType `float16`.
Paddle support following DataTypes: complex64, bool, bfloat16, complex128,
float32, int32, float64, int64.
```

其他补充信息 Additional Supplementary Information
- 复现环境：Windows 10，Python 3.10.18，Paddle 3.2.0，CPU。
- 结果汇总文件：`pd_ms_test_1/issues/pd_error_only_candidate_run_results.json`
- 期望行为：若 CPU 不支持 float16，应在 API 文档/错误信息中统一且提前给出明确约束；若按设计应支持，则应补齐 kernel。

Markdown Editor
Markdown input: edit mode selected.
Write
Preview
Leave a comment
感谢你的贡献 🎉！Thanks for your contribution 🎉!
