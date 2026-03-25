Add a title
*
Title
paddle.shape 访问第10维时报 Unimplemented（仅支持访问维度 0~9）（pd_ms_test_1 candidate 349）

在向Paddle报bug之前，请先查询历史issue是否报过同样的bug。
Before submitting a bug, please make sure the issue hasn't been already addressed by searching through the existing and past issues.

bug描述 Describe the Bug
*
在最小复现中，访问高维张量 `shape[10]` 时出现：
`OSError: Unimplemented error. Invalid dimension to be accessed. Now only supports access to dimension 0 to 9, but received dimension is 10.`

该限制在对照框架中未出现，脚本标记为 `REPRODUCED_PADDLE_ERROR_ONLY=True`。

对应脚本：
- `pd_ms_test_1/simple_tests/pd_error_only_candidate_349_shape.py`

Markdown Editor
Markdown input: edit mode selected.
Write
Preview
请清晰简洁的描述这个bug。A clear and concise description of what the bug is.

```python
# 最小可复现代码。Sample code to reproduce the problem.
import paddle

x = paddle.ones([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='float32')
print(x.shape[10])
```

```shell
带有完整回溯的报错信息。The error message you got, with the full traceback.
OSError: Unimplemented error. Invalid dimension to be accessed.
Now only supports access to dimension 0 to 9, but received dimension is 10.
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
