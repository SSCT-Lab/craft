Add a title
*
Title
paddle.logsumexp 对输入维度限制为 <=4（pd_pt_test candidate 163）

在向Paddle报bug之前，请先查询历史issue是否报过同样的bug。
Before submitting a bug, please make sure the issue hasn't been already addressed by searching through the existing and past issues.

bug描述 Describe the Bug
*
`paddle.logsumexp` 在高维输入上报错：
`The input tensor X's dimensions of logsumexp should be less or equal than 4.`

对照框架在该场景可执行，脚本标记 `REPRODUCED_PADDLE_ERROR_ONLY=True`。

对应脚本：
- `pd_pt_test/simple_tests/pd_error_only_candidate_163_logsumexp.py`

Markdown Editor
Markdown input: edit mode selected.
Write
Preview
请清晰简洁的描述这个bug。A clear and concise description of what the bug is.

```python
# 最小可复现代码。Sample code to reproduce the problem.
import paddle

x = paddle.ones([1, 1, 1, 1, 1], dtype='float32')
print(paddle.logsumexp(x, axis=-1))
```

```shell
带有完整回溯的报错信息。The error message you got, with the full traceback.
ValueError: (InvalidArgument) The input tensor X's dimensions of logsumexp
should be less or equal than 4.
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
