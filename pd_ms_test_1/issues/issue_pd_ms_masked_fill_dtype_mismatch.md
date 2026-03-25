Add a title
*
Title
paddle.masked_fill 在整型输入场景出现 int32/int64 类型读取不一致报错（pd_ms_test_1 candidate 166）

在向Paddle报bug之前，请先查询历史issue是否报过同样的bug。
Before submitting a bug, please make sure the issue hasn't been already addressed by searching through the existing and past issues.

bug描述 Describe the Bug
*
在最小复现中，`paddle.masked_fill` 报错：
`ValueError: (InvalidArgument) The type of data we are trying to retrieve (int64) does not match the type of data (int32) currently contained in the container.`

同样的测试在对照框架可运行成功，脚本复现标记为 `REPRODUCED_PADDLE_ERROR_ONLY=True`。

对应脚本：
- `pd_ms_test_1/simple_tests/pd_error_only_candidate_166_masked_fill.py`

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

x = paddle.to_tensor(np.array([[1, 2], [3, 4]], dtype=np.int32))
mask = paddle.to_tensor(np.array([[True, False], [False, True]], dtype=np.bool_))

# 触发点
out = paddle.masked_fill(x, mask, 0)
print(out)
```

```shell
带有完整回溯的报错信息。The error message you got, with the full traceback.
ValueError: (InvalidArgument) The type of data we are trying to retrieve (int64)
does not match the type of data (int32) currently contained in the container.
```

其他补充信息 Additional Supplementary Information
- 复现环境：Windows 10，Python 3.10.18，Paddle 3.2.0，CPU。
- 结果汇总文件：`pd_ms_test_1/issues/pd_error_only_candidate_run_results.json`

Markdown Editor
Markdown input: edit mode selected.
Write
Preview
Leave a comment
感谢你的贡献 🎉！Thanks for your contribution 🎉!
