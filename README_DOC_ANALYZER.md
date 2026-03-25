# 文档爬取和问题分析工具

## 概述

这个工具集用于：
1. **爬取文档**：从 PyTorch 和 TensorFlow 官方文档中提取 API 文档
2. **分析问题**：当测试迁移出现问题时，调用大模型结合官方文档分析问题

## 安装依赖

```bash
pip install requests beautifulsoup4
```

## 工具说明

### 1. doc_crawler.py - 文档爬取工具

爬取 PyTorch 和 TensorFlow 官方文档并缓存。

#### 使用方法

```bash
# 爬取 PyTorch 文档
# python3 component/doc_crawler.py torch.nn.Conv2d --framework pytorch
python -m component.doc.doc_crawler torch.nn.Conv2d --framework pytorch

# 爬取 TensorFlow 文档
python -m component.doc.doc_crawler tf.keras.layers.Dense --framework tensorflow

# 爬取 Paddle 文档
python -m component.doc.doc_crawler paddle.nn.Conv2D --framework paddle
python -m component.doc.doc_crawler paddle.nn.functional.conv2d --framework paddle

# 爬取 MindSpore 文档
python -m component.doc.doc_crawler mindspore.nn.Conv2d --framework mindspore

# 自动检测框架
python -m component.doc.doc_crawler tf.nn.relu

# 保存到文件
python -m component.doc.doc_crawler torch.nn.ReLU -o data/pt_relu_doc.json
```

#### 功能特点

- **自动缓存**：文档会缓存到 `data/docs_cache/` 目录，避免重复请求
- **自动检测框架**：根据 API 名称前缀自动识别框架
- **结构化提取**：提取描述、参数、返回值等信息

### 2. doc_analyzer.py - 问题分析工具

使用大模型分析测试迁移问题，结合官方文档判断差异是否正常。

#### 使用方法

```bash
# 基本用法：分析测试文件中的错误
python3 component/doc_analyzer.py \
    "IndentationError: expected an indented block" \
    --test-file migrated_tests/testArgRenames.py

# 指定相关 API（可选，会自动从代码中提取）
python3 component/doc_analyzer.py \
    "TypeError: ..." \
    --test-file migrated_tests/testRenames.py \
    --tf-apis tf.reduce_sum tf.reduce_any \
    --pt-apis torch.sum torch.any

# 添加额外上下文
python3 component/doc_analyzer.py \
    "错误信息" \
    --test-file migrated_tests/test.py \
    --context "这是在测试矩阵运算时出现的错误"

# 保存分析结果
python3 component/doc_analyzer.py \
    "错误信息" \
    --test-file migrated_tests/test.py \
    --output analysis_result.txt
```

#### 功能特点

- **自动提取代码**：从测试文件中自动提取 TF 和 PT 代码
- **自动爬取文档**：根据使用的 API 自动爬取相关文档
- **智能分析**：结合官方文档判断差异是否正常
- **详细报告**：提供错误原因、差异说明和修复建议

## 工作流程示例

### 场景：测试迁移后出现错误

1. **运行测试发现问题**：
```bash
python3 migrated_tests/testArgRenames.py
# 输出: IndentationError: expected an indented block after 'with' statement
```

2. **使用分析工具**：
```bash
python3 component/doc_analyzer.py \
    "IndentationError: expected an indented block after 'with' statement on line 145" \
    --test-file migrated_tests/testArgRenames.py \
    --output analysis.txt
```

3. **查看分析结果**：
分析工具会：
- 自动提取 TF 和 PT 代码
- 识别使用的 API
- 爬取相关文档
- 调用大模型分析问题
- 判断是否为正常差异
- 提供修复建议

## 文档缓存

文档会缓存在 `data/docs_cache/` 目录中，格式为：
- `pytorch_{api_hash}.json` - PyTorch 文档
- `tensorflow_{api_hash}.json` - TensorFlow 文档

缓存可以：
- 加快后续分析速度
- 离线使用（如果之前爬取过）
- 减少对官方文档服务器的请求

## 配置

### API Key 配置

分析工具需要 LLM API key，默认从 `aliyun.key` 文件读取：

```bash
echo "your-api-key" > aliyun.key
```

### 自定义模型

```bash
python3 component/doc_analyzer.py ... --model qwen-plus
```

## 注意事项

1. **网络访问**：文档爬取需要访问互联网
2. **请求频率**：工具会自动延迟请求，避免过快
3. **文档更新**：如果文档有更新，可以删除缓存文件重新爬取
4. **API 名称格式**：支持 `tf.function`、`torch.function`、`tf.module.function` 等格式

## 示例输出

分析工具的输出示例：

```
================================================================================
分析结果
================================================================================

【错误原因分析】
这是一个缩进错误，发生在 `with tf.compat.v1.Session():` 语句后。
...

【是否为正常差异】
否，这是代码生成时的缩进处理错误。

【框架差异说明】
TensorFlow 的 Session 需要 with 语句管理，但 PyTorch 不需要。
...

【建议的修复方案】
1. 修复缩进：确保 with 块内的代码正确缩进
2. 或者移除 with 语句（如果不需要 Session）
...
```

## 集成到工作流

可以在测试失败后自动调用分析工具：

```bash
# 运行测试并捕获错误
python3 migrated_tests/test.py 2>&1 | tee test_output.log

# 如果失败，自动分析
if [ $? -ne 0 ]; then
    ERROR=$(grep -i "error\|exception" test_output.log | head -1)
    python3 component/doc_analyzer.py "$ERROR" --test-file migrated_tests/test.py
fi
```

