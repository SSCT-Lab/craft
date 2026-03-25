Add a title
*
Title
[CPU] bfloat16 输入下 mean/softmax kernel not registered（pd_pt_test candidates 172/230）

在向Paddle报bug之前，请先查询历史issue是否报过同样的bug。
Before submitting a bug, please make sure the issue hasn't been already addressed by searching through the existing and past issues.

bug描述 Describe the Bug
*
在 CPU 环境下，`bfloat16` 输入触发以下报错：
- `mean`：kernel `mean` bfloat16 not registered
- `nn.functional.softmax`：kernel `softmax` bfloat16 not registered

对应脚本：
- `pd_pt_test/simple_tests/pd_error_only_candidate_172_mean.py`
- `pd_pt_test/simple_tests/pd_error_only_candidate_230_nn_functional_softmax.py`

Markdown Editor
Markdown input: edit mode selected.
Write
Preview
请清晰简洁的描述这个bug。A clear and concise description of what the bug is.

```python
# 最小可复现代码。Sample code to reproduce the problem.
import paddle

x = paddle.ones([2, 3], dtype='bfloat16')
print(paddle.mean(x))
```

```shell
带有完整回溯的报错信息。The error message you got, with the full traceback.
RuntimeError: (NotFound) The kernel with key (CPU, Undefined(AnyLayout), bfloat16)
of kernel `mean` is not registered.
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
