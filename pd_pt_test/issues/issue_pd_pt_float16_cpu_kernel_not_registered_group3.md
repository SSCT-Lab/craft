Add a title
*
Title
[CPU] Paddle 多个算子在 float16 输入下 kernel not registered（pd_pt_test 复现）

在向Paddle报bug之前，请先查询历史issue是否报过同样的bug。
Before submitting a bug, please make sure the issue hasn't been already addressed by searching through the existing and past issues.

bug描述 Describe the Bug
*
在 `pd_pt_test` 最小复现中，多个算子在 CPU 下处理 float16 输入时报 NotFound（kernel 未注册）。
对照框架 PyTorch 对应 case 可执行，脚本均显示 `REPRODUCED_PADDLE_ERROR_ONLY=True`。

涉及脚本（部分）：
- `pd_pt_test/simple_tests/pd_error_only_candidate_024_allclose.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_037_asin.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_039_atan.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_051_baddbmm.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_065_cosh.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_073_cumprod.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_075_cumsum.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_086_fmax.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_087_fmin.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_092_gammainc.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_115_gammaincc.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_133_hypot.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_138_isclose.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_145_lerp.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_150_lgamma.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_171_maximum.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_179_multiply.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_197_nn_functional_celu.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_208_nn_functional_hardswish.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_252_reciprocal.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_259_sinc.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_264_sinh.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_270_tanh.py`

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

x = paddle.to_tensor(np.ones((2, 2), dtype=np.float16))
y = paddle.to_tensor(np.ones((2, 2), dtype=np.float16))
print(paddle.multiply(x, y))
```

```shell
带有完整回溯的报错信息。The error message you got, with the full traceback.
RuntimeError: (NotFound) The kernel with key (CPU, Undefined(AnyLayout), float16)
of kernel `multiply` is not registered.
```

其他补充信息 Additional Supplementary Information
- 复现环境：Windows 10，Python 3.10.18，Paddle 3.2.0，CPU。
- 结果汇总文件：`pd_pt_test/issues/pd_error_only_candidate_run_results.json`

Markdown Editor
Markdown input: edit mode selected.
Write
Preview
Leave a comment
感谢你的贡献 🎉！Thanks for your contribution 🎉!
