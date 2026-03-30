# Execution-Guided Cross-Project Differential Testing for Deep Learning Frameworks

## Abstract
This project studies the semantic consistency of operators across four major deep learning frameworks: PyTorch, TensorFlow, PaddlePaddle, and MindSpore. We build a unified differential-testing pipeline that (1) mines framework APIs from official test suites, (2) generates cross-framework API mappings with LLM support, (3) executes paired operators under shared inputs, and (4) iteratively repairs or mutates test cases using LLM feedback. Across 12 pairwise directories, we report large-scale evidence of both semantic equivalence and non-equivalence. For 12 standardized pairwise campaigns with structured CSV reports, we analyze 2,888 mapped operators and 42,467 execution iterations. We observe 1,297 consistent operators, 932 inconsistent operators, 645 error-dominant operators, and 14 unknown operators. We further benchmark LLM-driven adaptation against converter-based baselines (MindConverter, X2Paddle, ONNX routes) in 3 legacy PyTorch-centric campaigns, where LLM-driven execution success is consistently higher than conversion run success. In addition, we add selected-pair ablation code for LLM vs non-LLM strategies and dedicated effectiveness-validation scripts that quantify repair/mutation gains in a unified way. Additionally, our method reveals a number of previously unreported framework bugs, with approximately 50 issues submitted to the corresponding repositories. The study demonstrates that LLM-guided iterative repair is effective for cross-framework compatibility testing, while also revealing systematic sources of disagreement (parameter semantics, dtype constraints, reduction definitions, and unsupported kernels).


## 1. Research Motivation and Questions
Cross-framework migration and model portability require more than name-level API mapping: semantic compatibility must be validated under realistic input spaces and edge cases. This project targets three questions:

1. Can LLM-guided differential testing scale to broad operator coverage across four frameworks?
2. Which inconsistency categories are true semantic differences versus test-construction artifacts?
3. How does LLM-guided adaptation compare with conversion-based baselines in practical executability?

## 2. Overall Methodology
The pipeline is implemented in each pairwise directory (e.g., `ms_pd_test_1`, `pd_tf_test_1`, `tf_pt_test`) with domain-specific variants but shared logic:

1. API extraction from official/curated test suites.
2. API existence filtering and documentation validation.
3. LLM-based API mapping generation with confidence filtering.
4. LLM-based test-case extraction from source tests.
5. Differential execution with shared input tensors and iterative case evolution.
6. Post-hoc analysis with per-operator and per-iteration metrics.

### 2.1 LLM-in-the-Loop Differential Testing
The core execution script pattern (`llm_enhanced_compare.py`) adopts an iterative strategy:

- Initial execution: run mapped APIs on both frameworks with aligned inputs.
- Decision by LLM:
  - `repair`: fix parameter/API mismatch to restore semantic alignment.
  - `mutation`: generate harder or broader fuzzing cases once a case is valid.
  - `skip`: mark non-equivalent or unsupported mappings.
  - `final_execution`: final validation of the latest generated case.
- Multi-round control: typically up to 3 rounds per case.

### 2.2 Determinism and Stability Controls
Key engineering choices include:

- Shared NumPy seed and tensor materialization for both frameworks.
- Conservative threading and environment settings (e.g., MKL/OMP/TF thread limits).
- Execution locking for critical sections to reduce concurrency-induced instability.
- Rich logging of status, errors, shapes, dtypes, and max-difference diagnostics.

### 2.3 Newly Added Evaluation Components
To strengthen the empirical argument around the LLM contribution, we added two complementary components:

1. **LLM vs non-LLM comparison scripts (selected pairs)**
   - We maintain pairwise scripts that compare LLM-guided testing against rule-based or converter-style baselines (e.g., MindConverter/X2Paddle/ONNX routes) depending on pair direction and available tooling.
2. **LLM effectiveness validation scripts (selected pairs)**
   - We introduce dedicated `llm_effectiveness_validation.py` scripts that split cases into:
     - initially failed cases (repair-focused path), and
     - initially successful cases (mutation-robustness path).
   - These scripts keep **LLM calls concurrent** while forcing **operator execution sequential** to avoid execution-layer interference.
   - They emit two real-time detail logs and two summary logs (`repair_*`, `mutate_*`) for direct paper-ready metrics.

## 3. Experimental Scope
The study covers 12 directories:

- MindSpore-first: `ms_pd_test_1`, `ms_tf_test_1`, `ms_pt_test`
- Paddle-first: `pd_ms_test_1`, `pd_pt_test`, `pd_tf_test_1`
- PyTorch-first (legacy style): `pt_ms_test`, `pt_pd_test`, `pt_tf_test`
- TensorFlow-first: `tf_ms_test_1`, `tf_pd_test_1`, `tf_pt_test`

All twelve directories now provide standardized `analysis_report_*.csv` files with unified fields. The three PyTorch-first directories additionally provide structured global comparison JSONs against converter baselines for baseline-oriented analysis.

For the newly added methodology checks, we selected representative framework pairs for code-level validation:

- **Effectiveness-validation scripts added**: `pt_tf_test`, `pt_pd_test`, `pt_ms_test`, `tf_pd_test_1`, `pd_ms_test_1`, `ms_pt_test`
- **LLM-vs-non-LLM comparison scripts available in selected pairs**: e.g., `ms_pd_test_1`, `tf_ms_test_1`, `pd_pt_test`, `pt_tf_test`, `tf_pt_test`, `pt_pd_test`, `pt_ms_test`

## 4. Quantitative Results (12 Standardized Campaigns)

### 4.1 Operator-Level Outcomes
From 12 standardized reports:

- Total mapped operators: 2,888
- `consistent`: 1,297 (44.91%)
- `inconsistent`: 932 (32.27%)
- `error`: 645 (22.33%)
- `unknown`: 14 (0.48%)

### 4.2 Iteration-Level Outcomes
- Total iterations: 42,467
- Consistent iterations: 20,175
- Inconsistent iterations: 5,524
- Both-error iterations: 5,627

### 4.3 Pairwise Breakdown

| Pair Directory | Operators | Consistent | Inconsistent | Error | Unknown | Iterations |
|---|---:|---:|---:|---:|---:|---:|
| ms_pd_test_1 | 199 | 93 | 49 | 57 | 0 | 3,840 |
| ms_tf_test_1 | 214 | 85 | 55 | 74 | 0 | 4,043 |
| ms_pt_test | 239 | 104 | 55 | 66 | 14 | 4,586 |
| pd_ms_test_1 | 221 | 101 | 99 | 21 | 0 | 4,368 |
| pd_pt_test | 277 | 195 | 69 | 13 | 0 | 5,448 |
| pd_tf_test_1 | 210 | 98 | 100 | 12 | 0 | 4,076 |
| pt_ms_test | 403 | 170 | 75 | 158 | 0 | 3,183 |
| pt_pd_test | 420 | 198 | 139 | 83 | 0 | 3,329 |
| pt_tf_test | 366 | 119 | 140 | 107 | 0 | 3,002 |
| tf_ms_test_1 | 97 | 29 | 41 | 27 | 0 | 1,822 |
| tf_pd_test_1 | 114 | 57 | 50 | 7 | 0 | 2,269 |
| tf_pt_test | 128 | 48 | 60 | 20 | 0 | 2,501 |

## 5. Legacy PyTorch-Centric Campaigns: LLM vs Converter Baselines
The three legacy directories include additional baseline comparisons:

### 5.1 pt_ms_test (LLM vs MindConverter)
- Total operators: 265
- Tested operators: 184
- LLM success: 266 / 418 cases (63.6%)
- MindConverter run success: 218 / 418 cases (52.2%)

### 5.2 pt_pd_test (LLM vs X2Paddle)
- Total operators: 249
- Tested operators: 201
- LLM success: 350 / 438 cases (79.9%)
- X2Paddle run success: 209 / 438 cases (47.7%)

### 5.3 pt_tf_test (LLM vs ONNX Route)
- Total operators: 265
- Tested operators: 189
- LLM success: 299 / 405 cases (73.8%)
- ONNX run success: 217 / 405 cases (53.6%)

Observation: LLM-guided repair/mutation consistently improves executable cross-framework cases compared with direct conversion pipelines.

### 5.4 Additional "With-LLM vs Without-LLM" Comparison Code (Selected Pairs)
Beyond the three legacy PyTorch-centric campaigns, we also include comparison scripts in several standardized or semi-standardized directories to support controlled ablations. Representative examples include:

- `ms_pd_test_1/compare_llm_vs_rulebased.py`
- `tf_ms_test_1/compare_llm_vs_rulebased.py`
- `pd_pt_test/compare_llm_vs_rulebased.py`
- `pt_tf_test/compare_llm_vs_rulebased.py`
- `tf_pt_test/compare_llm_vs_onnx.py`
- `pt_pd_test/compare_llm_vs_x2paddle.py`

These scripts provide reusable infrastructure to contrast LLM-guided case evolution with non-LLM routes under comparable operator subsets.

### 5.5 Dedicated LLM Effectiveness Validation (Selected Pairs)
We additionally implement dedicated effectiveness scripts:

- `pt_tf_test/llm_effectiveness_validation.py`
- `pt_pd_test/llm_effectiveness_validation.py`
- `pt_ms_test/llm_effectiveness_validation.py`
- `tf_pd_test_1/llm_effectiveness_validation.py`
- `pd_ms_test_1/llm_effectiveness_validation.py`
- `ms_pt_test/llm_effectiveness_validation.py`

For each selected pair, the script reports:

1. **Repair effectiveness** on initially failed cases
   - total initially failed cases
   - effective failed cases (excluding `skip`)
   - repaired-at-round-1/2/3 counts and ratios
   - post-repair mutation survivability by mutation round
2. **Mutation effectiveness** on initially successful cases
   - total initially successful cases
   - survive-after-mutation-round-1/2/3 counts and ratios

This provides a direct estimate of (a) how quickly LLM can repair failing cross-framework cases and (b) how robust mutated cases remain after iterative fuzzing.

## 6. Error Taxonomy and Representative Findings
From analysis artifacts (notably `pd_tf_test_1/analysis/inconsistent_success_samples_lt1_20260306_144330.json` and corresponding pairwise reports), inconsistencies are concentrated in several categories:

1. Parameter semantic mismatch
   - `paddle.div(rounding_mode=...)` vs `tf.math.divide` (no rounding mode).
   - `paddle.round(decimals=...)` vs `tf.math.round` (integer rounding only).
2. Statistical definition mismatch
   - `paddle.std/var` with `unbiased` or `correction` vs TF defaults.
3. Activation/operator option mismatch
   - ELU/SELU/SiLU parameter differences (`alpha`, `scale`, `beta`) across frameworks.
4. Randomness-induced false inconsistency
   - Cases where only shape/dtype were aligned but actual values differed across frameworks.
5. API surface mismatch
   - Missing kwargs (`keepdim` vs `keepdims`, unsupported keyword sets).
6. Kernel/backend support mismatch
   - Frequent `NotFound` or dtype/device kernel limitations (e.g., float16/int kernels on CPU).

These findings show that many failures are not pure numerical drift; they are often rooted in semantic/interface incompatibility.

## 7. Why the LLM Loop Helps
The iterative LLM loop improves differential quality in three ways:

1. Repairs non-semantic failures first (keyword names, dtype constraints, axis conventions).
2. Expands valid cases through mutation and fuzzing after reaching executability.
3. Produces human-readable rationales for each transformation (`repair`, `mutation`, `skip`).

This converts differential testing from a one-shot pass/fail process into a guided search over semantically comparable test space.

## 8. Threats to Validity

1. Mapping quality dependence
   - LLM mapping and confidence filtering may still include non-equivalent pairs.
2. Incomplete semantic normalization
   - Some APIs are fundamentally non-isomorphic (e.g., option sets differ by design).
3. Hardware/software environment bias
   - CPU-only kernels and version-specific behavior affect reproducibility.
4. Metric heterogeneity across analysis goals
   - Although operator-level differential reports are now standardized across 12 directories, baseline-comparison metrics (e.g., converter run success) still follow campaign-specific schemas and should be interpreted separately.

## 9. Conclusion
This project demonstrates a practical and scalable framework for semantic differential testing across PyTorch, TensorFlow, PaddlePaddle, and MindSpore. The combined evidence from 12 directories shows that:

- Large-scale consistency is achievable but far from universal.
- A substantial fraction of mismatches come from API semantics and execution constraints, not only floating-point tolerance.
- LLM-guided iterative repair/mutation is more effective than direct converter baselines in producing executable, comparable test cases.

With the newly added selected-pair ablation scripts and effectiveness-validation scripts, the study now includes both **global cross-project evidence** and **targeted mechanism-level evidence** for the utility of LLM-guided repair/mutation.

The project provides both methodological value (LLM-in-the-loop differential testing) and actionable compatibility evidence for cross-framework migration and bug discovery.

## Appendix A. Core Artifacts Used

- Standardized pairwise workflows and scripts:
   - `ms_pd_test_1`, `ms_tf_test_1`, `ms_pt_test`, `pd_ms_test_1`, `pd_pt_test`, `pd_tf_test_1`, `pt_ms_test`, `pt_pd_test`, `pt_tf_test`, `tf_ms_test_1`, `tf_pd_test_1`, `tf_pt_test`
- Standardized reports:
  - `analysis/analysis_report_*.csv`, `analysis/analysis_report_*.txt`
  - `analysis/inconsistent_success_samples*.json`, `analysis/*_error_only_samples*.json`, `analysis/issue_candidates_*.json`
- Legacy PyTorch-first baseline comparisons:
  - `pt_ms_test/llm_vs_mindconverter_result_20260206_234443.json`
  - `pt_pd_test/llm_vs_x2paddle_result_20260206_230759.json`
  - `pt_tf_test/llm_vs_onnx_result_20260206_222722.json`
- Selected "with-LLM vs without-LLM" comparison scripts:
   - `ms_pd_test_1/compare_llm_vs_rulebased.py`
   - `tf_ms_test_1/compare_llm_vs_rulebased.py`
   - `pd_pt_test/compare_llm_vs_rulebased.py`
   - `pt_tf_test/compare_llm_vs_rulebased.py`
   - `tf_pt_test/compare_llm_vs_onnx.py`
   - `pt_pd_test/compare_llm_vs_x2paddle.py`
   - `pt_ms_test/compare_llm_vs_mindconverter.py`
- Selected LLM effectiveness-validation scripts:
   - `pt_tf_test/llm_effectiveness_validation.py`
   - `pt_pd_test/llm_effectiveness_validation.py`
   - `pt_ms_test/llm_effectiveness_validation.py`
   - `tf_pd_test_1/llm_effectiveness_validation.py`
   - `pd_ms_test_1/llm_effectiveness_validation.py`
   - `ms_pt_test/llm_effectiveness_validation.py`
