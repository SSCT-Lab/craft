# Craft: Cross-Framework Repair-Assisted Fuzzing for Testing Equivalent Deep Learning Operators

## Project Overview

Craft is a research-oriented, automated system for evaluating API compatibility across major deep learning frameworks. Craft is designed for directional, operator-level differential testing and supports reproducible analysis of behavioral divergence.

The current implementation of Craft focuses on four framework ecosystems: PyTorch, TensorFlow, PaddlePaddle, and MindSpore.

## Core Features

### 1. API Mapping Extraction and Validation
- LLM-assisted extraction of cross-framework API correspondences
- Mapping validation with confidence-aware filtering
- Documentation-grounded verification for unsupported or partially aligned APIs

### 2. Directional Differential Operator Testing
- Execution and comparison of semantically matched operators under unified test cases
- Automatic test-case repair and mutation via LLM-based enhancement pipelines
- Standardized result logging for downstream statistical analysis

The experimental matrix is organized into 12 directional test suites:
- `ms_pd_test_1`, `ms_tf_test_1`, `ms_pt_test`
- `pd_ms_test_1`, `pd_pt_test`, `pd_tf_test_1`
- `pt_ms_test`, `pt_pd_test`, `pt_tf_test`
- `tf_ms_test_1`, `tf_pd_test_1`, `tf_pt_test`

### 3. Documentation Crawling and Migration-Oriented Analysis
- Automated crawling of official framework documentation
- Cached document processing for scalable analysis
- LLM-supported diagnosis for migration and compatibility issues

## Project Structure

```
├── component/              # Core components
│   ├── data/              # API mapping data and processing scripts
│   ├── doc/               # Documentation crawling and analysis modules
│   ├── matching/          # API matching modules
│   └── migration/         # Migration assistance modules
├── ms_pd_test_1/          # MindSpore -> PaddlePaddle differential tests
├── ms_tf_test_1/          # MindSpore -> TensorFlow differential tests
├── ms_pt_test/            # MindSpore -> PyTorch differential tests
├── pd_ms_test_1/          # PaddlePaddle -> MindSpore differential tests
├── pd_pt_test/            # PaddlePaddle -> PyTorch differential tests
├── pd_tf_test_1/          # PaddlePaddle -> TensorFlow differential tests
├── pt_ms_test/            # PyTorch -> MindSpore differential tests
├── pt_pd_test/            # PyTorch -> PaddlePaddle differential tests
├── pt_tf_test/            # PyTorch -> TensorFlow differential tests
├── tf_ms_test_1/          # TensorFlow -> MindSpore differential tests
├── tf_pd_test_1/          # TensorFlow -> PaddlePaddle differential tests
├── tf_pt_test/            # TensorFlow -> PyTorch differential tests
├── api_mapping_old/       # Historical API mapping tables (CSV)
└── data/                  # Documentation cache and auxiliary datasets
```

## Quick Start

### Requirements
- Python 3.8+
- MongoDB (for test-case storage)
- Framework dependencies installed as required by the selected directional suite

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Extract API Mappings (Example: PyTorch -> TensorFlow)**
```bash
python component/data/extract_tf_api_mapping.py -i api_mappings.csv
```

2. **Validate API Mappings (Example: PyTorch -> TensorFlow)**
```bash
python component/data/validate_tf_api_mapping.py
```

3. **Run Directional Differential Tests (LLM-Enhanced)**
```bash
# MindSpore as source framework
python ms_pd_test_1/llm_enhanced_compare.py
python ms_tf_test_1/llm_enhanced_compare.py
python ms_pt_test/llm_enhanced_compare.py

# PaddlePaddle as source framework
python pd_ms_test_1/llm_enhanced_compare.py
python pd_pt_test/llm_enhanced_compare.py
python pd_tf_test_1/llm_enhanced_compare.py

# PyTorch as source framework
python pt_ms_test/llm_enhanced_compare.py
python pt_pd_test/llm_enhanced_compare.py
python pt_tf_test/llm_enhanced_compare.py

# TensorFlow as source framework
python tf_ms_test_1/llm_enhanced_compare.py
python tf_pd_test_1/llm_enhanced_compare.py
python tf_pt_test/llm_enhanced_compare.py
```

4. **Analyze Test Results**
```bash
# Error-level analysis (example)
python pt_tf_test/analyze_errors.py

# Sample-based statistical analysis (example)
python ms_pd_test_1/analyze_results_with_samples.py
```

## Key Highlights

- ✅ **Research-Grade Automation**: End-to-end differential testing from mapping to analysis
- ✅ **Methodological Consistency**: Directional test suites with comparable execution and logging patterns
- ✅ **LLM-Augmented Robustness**: Automated repair and mutation of test cases to improve executable coverage
- ✅ **Evidence-Centered Reporting**: Structured outputs for failure attribution and cross-framework divergence analysis
- ✅ **Extensible Design**: Modular directory layout for incorporating additional framework pairs

## Tech Stack

- **Deep Learning Frameworks**: PyTorch, TensorFlow, PaddlePaddle, MindSpore
- **Database**: MongoDB
- **LLM Service**: Alibaba Cloud DashScope (Qwen series)
- **Documentation Processing**: BeautifulSoup, Requests
- **Data Analysis**: Pandas, NumPy

## Contributing

Issues and Pull Requests are welcome.
