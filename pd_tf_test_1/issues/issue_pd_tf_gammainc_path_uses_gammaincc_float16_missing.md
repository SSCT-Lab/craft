Add a title
*
Title
paddle.gammainc 在 float16 输入下走到 gammaincc 路径并报 kernel not registered（pd_tf_test_1 candidates 050/068）

在向Paddle报bug之前，请先查询历史issue是否报过同样的bug。
Before submitting a bug, please make sure the issue hasn't been already addressed by searching through the existing and past issues.

bug描述 Describe the Bug
*
在 `pd_tf_test_1` 中，`gammainc` 与 `gammaincc` 相关 case 均报同一错误：
`RuntimeError: (NotFound) ... kernel 'gammaincc' ... float16 is not registered.`

这说明 `gammainc` 场景也进入了 `gammaincc` 内核路径并因 float16 CPU 支持缺失失败。

对应脚本：
- `pd_tf_test_1/simple_tests/pd_error_only_candidate_050_gammainc.py`
- `pd_tf_test_1/simple_tests/pd_error_only_candidate_068_gammaincc.py`

Markdown Editor
Markdown input: edit mode selected.
Write
Preview
请清晰简洁的描述这个bug。A clear and concise description of what the bug is.

```python
# 最小可复现代码。Sample code to reproduce the problem.
import paddle

x = paddle.ones([2, 2], dtype='float16')
y = paddle.ones([2, 2], dtype='float16')
print(paddle.gammainc(x, y))
```

```shell
带有完整回溯的报错信息。The error message you got, with the full traceback.
RuntimeError: (NotFound) The kernel with key (CPU, Undefined(AnyLayout), float16)
of kernel `gammaincc` is not registered.
```

其他补充信息 Additional Supplementary Information
- 复现环境：Windows 10，Python 3.10.18，Paddle 3.2.0，CPU。
- 结果汇总文件：`pd_tf_test_1/issues/pd_error_only_candidate_run_results.json`

Markdown Editor
Markdown input: edit mode selected.
Write
Preview
Leave a comment
感谢你的贡献 🎉！Thanks for your contribution 🎉!
