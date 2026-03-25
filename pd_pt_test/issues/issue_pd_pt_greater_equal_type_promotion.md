Add a title
*
Title
paddle.greater_equal 在 int32 与 float32 比较时类型提升报错（pd_pt_test candidate 127）

在向Paddle报bug之前，请先查询历史issue是否报过同样的bug。
Before submitting a bug, please make sure the issue hasn't been already addressed by searching through the existing and past issues.

bug描述 Describe the Bug
*
`paddle.greater_equal` 在 `int32` 与 `float32` 混合输入下报 `TypeError`，提示 type promotion 不支持该组合。对照框架可执行。

对应脚本：
- `pd_pt_test/simple_tests/pd_error_only_candidate_127_greater_equal.py`

Markdown Editor
Markdown input: edit mode selected.
Write
Preview
请清晰简洁的描述这个bug。A clear and concise description of what the bug is.

```python
# 最小可复现代码。Sample code to reproduce the problem.
import numpy as np
import paddle

x = paddle.to_tensor(np.array([1, 2, 3], dtype=np.int32))
y = paddle.to_tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))

print(paddle.greater_equal(x, y))
```

```shell
带有完整回溯的报错信息。The error message you got, with the full traceback.
TypeError: (InvalidType) Type promotion only support calculations between
floating-point numbers and between complex and real numbers.
But got different data type x: int32, y: float32.
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
