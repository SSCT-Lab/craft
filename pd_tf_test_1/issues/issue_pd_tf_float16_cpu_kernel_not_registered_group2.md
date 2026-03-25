Add a title
*
Title
[CPU] 多个 Paddle 算子在 float16 输入下报 NotFound kernel（pd_tf_test_1 复现）

在向Paddle报bug之前，请先查询历史issue是否报过同样的bug。
Before submitting a bug, please make sure the issue hasn't been already addressed by searching through the existing and past issues.

bug描述 Describe the Bug
*
在 `pd_tf_test_1` 的最小复现集中，多组 API 在 CPU + float16 输入下报错：
`RuntimeError: (NotFound) The kernel with key (CPU, Undefined(AnyLayout), float16) of kernel <op> is not registered.`

对照框架 TensorFlow 在对应 case 下成功，脚本标记 `REPRODUCED_PADDLE_ERROR_ONLY=True`。

涉及脚本：
- `pd_tf_test_1/simple_tests/pd_error_only_candidate_002_abs.py`
- `pd_tf_test_1/simple_tests/pd_error_only_candidate_008_allclose.py`
- `pd_tf_test_1/simple_tests/pd_error_only_candidate_025_cosh.py`
- `pd_tf_test_1/simple_tests/pd_error_only_candidate_050_gammainc.py`
- `pd_tf_test_1/simple_tests/pd_error_only_candidate_068_gammaincc.py`
- `pd_tf_test_1/simple_tests/pd_error_only_candidate_157_multiply.py`
- `pd_tf_test_1/simple_tests/pd_error_only_candidate_195_nn_functional_silu.py`
- `pd_tf_test_1/simple_tests/pd_error_only_candidate_203_nn_functional_softplus.py`
- `pd_tf_test_1/simple_tests/pd_error_only_candidate_204_nn_functional_softsign.py`
- `pd_tf_test_1/simple_tests/pd_error_only_candidate_238_square.py`
- `pd_tf_test_1/simple_tests/pd_error_only_candidate_241_subtract.py`
- `pd_tf_test_1/simple_tests/pd_error_only_candidate_244_tan.py`
- `pd_tf_test_1/simple_tests/pd_error_only_candidate_253_unstack.py`

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

x = paddle.to_tensor(np.ones((4, 4), dtype=np.float16))
print(paddle.abs(x))
```

```shell
带有完整回溯的报错信息。The error message you got, with the full traceback.
RuntimeError: (NotFound) The kernel with key (CPU, Undefined(AnyLayout), float16)
of kernel `abs` is not registered.
```

其他补充信息 Additional Supplementary Information
- 复现环境：Windows 10，Python 3.10.18，Paddle 3.2.0，CPU。
- 结果汇总文件：`pd_tf_test_1/issues/pd_error_only_candidate_run_results.json`
- 同目录下 `pd_error_only_candidate_140_min.py` 的空维规约报错更接近输入约束，未并入本 issue。

Markdown Editor
Markdown input: edit mode selected.
Write
Preview
Leave a comment
感谢你的贡献 🎉！Thanks for your contribution 🎉!
