# 深度学习框架 API 兼容性测试工具

## 项目简介

本项目是一个用于测试和比较不同深度学习框架之间 API 兼容性的自动化工具集。主要功能包括：

- **跨框架 API 映射**：自动提取和验证 PyTorch 与其他框架（TensorFlow、PaddlePaddle、MindSpore）之间的 API 映射关系
- **算子行为比较**：基于 MongoDB 中的测试用例，自动比较不同框架中对应算子的执行结果
- **文档智能分析**：爬取官方文档并结合 LLM 分析框架差异和迁移问题
- **差异检测与报告**：生成详细的测试报告，标识功能差异和潜在问题

## 核心功能

### 1. API 映射管理
- 基于 LLM 自动提取 PyTorch 到目标框架的 API 映射
- 验证映射准确性并标记置信度
- 支持文档验证，自动识别无对应实现的 API

### 2. 跨框架算子测试
支持以下框架组合的对比测试：
- PyTorch ↔ TensorFlow
- PyTorch ↔ PaddlePaddle  
- PyTorch ↔ MindSpore

### 3. 文档爬取与分析
- 自动爬取 PyTorch、TensorFlow、PaddlePaddle、MindSpore 官方文档
- 文档缓存机制，提高分析效率
- 结合 LLM 分析框架差异和迁移问题

## 项目结构

```
├── component/              # 核心组件
│   ├── data/              # API 映射数据和处理脚本
│   ├── doc/               # 文档爬取和分析工具
│   ├── matching/          # API 匹配相关工具
│   └── migration/         # 迁移辅助工具
├── pt_pd_test/            # PyTorch-PaddlePaddle 对比测试
├── pt_tf_test/            # PyTorch-TensorFlow 对比测试
├── pt_ms_test/            # PyTorch-MindSpore 对比测试
├── api_mapping/           # API 映射表（CSV 格式）
└── data/                  # 文档缓存和测试数据
```

## 快速开始

### 环境要求
- Python 3.8+
- MongoDB（用于存储测试用例）
- PyTorch、TensorFlow/PaddlePaddle/MindSpore（根据需要安装）

### 安装依赖
```bash
pip install -r requirements.txt
```

### 基本使用

1. **提取 API 映射**
```bash
python component/data/extract_tf_api_mapping.py -i api_mappings.csv
```

2. **验证 API 映射**
```bash
python component/data/validate_tf_api_mapping.py
```

3. **运行对比测试（基于 LLM 增强）**
```bash
# PyTorch vs TensorFlow
python pt_tf_test/llm_enhanced_compare.py

# PyTorch vs PaddlePaddle
python pt_pd_test/llm_enhanced_compare.py

# PyTorch vs MindSpore
python pt_ms_test/llm_enhanced_compare.py
```

**注意**：实际测试使用的是各框架测试目录下的 `llm_enhanced_compare.py` 脚本，这些脚本集成了 LLM 能力，可以自动修复和优化测试用例。

4. **分析测试结果**
```bash
# 错误统计分析
python pt_tf_test/analyze_errors.py

# 提取错误样例
python pt_tf_test/analysis/extract_torch_error_cases.py
```

5. **文档分析工具**
```bash
python component/doc/doc_analyzer.py "错误信息" --test-file test.py
```

## 主要特性

- ✅ **自动化测试**：基于 MongoDB 测试用例库自动生成和执行测试
- ✅ **智能映射**：利用 LLM 自动发现和验证 API 映射关系
- ✅ **文档驱动**：结合官方文档进行智能分析和问题诊断
- ✅ **详细报告**：生成包含成功率、失败原因、差异分析的完整报告
- ✅ **可扩展性**：易于添加新的框架支持

## 文档

- [API 映射工具使用指南](API_MAPPING_TOOLS_USAGE.md)
- [文档分析工具说明](README_DOC_ANALYZER.md)
- [错误分析工作流程](pt_tf_test/ERROR_ANALYSIS_WORKFLOW.md) - PyTorch-TensorFlow 测试错误分析指南
  - 类似文档也存在于 `pt_pd_test/` 和 `pt_ms_test/` 目录

## 技术栈

- **深度学习框架**：PyTorch, TensorFlow, PaddlePaddle, MindSpore
- **数据库**：MongoDB
- **LLM**：阿里云 DashScope (Qwen 系列模型)
- **文档处理**：BeautifulSoup, Requests
- **数据分析**：Pandas, NumPy

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

[待添加]
