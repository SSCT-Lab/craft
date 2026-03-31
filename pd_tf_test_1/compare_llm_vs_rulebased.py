#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM vs Rule-based: PaddlePaddle -> TensorFlow test case conversion success rate comparison
=======================================================================================

Compare two cross-framework test case migration approaches:
1. LLM method: run initial case -> LLM repair/mutation (fixed 1 iteration) -> run LLM-generated TF case
2. Rule-based method: convert using API mapping + dtype mapping -> run rule-generated TF case

Key metric: whether the generated TF case runs successfully (tensorflow_success).

Outputs:
- Realtime results: JSONL (one case per line)
- Summary results: JSON (global stats + per-operator details)
"""

import argparse
import copy
import json
import os
import re
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock, RLock
from typing import Any, Dict, List, Optional, Tuple

# Env vars must be set before importing Paddle/TensorFlow to avoid thread library conflicts.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf
import paddle

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pd_tf_test_1.llm_enhanced_compare import (
    DEFAULT_KEY_PATH,
    DEFAULT_MODEL,
    LLMEnhancedComparator,
)

DEFAULT_MAX_ITERATIONS = 1
DEFAULT_NUM_CASES = 3
DEFAULT_WORKERS = 4


def safe_print(msg: str, print_lock: Optional[Lock] = None, end: str = "\n") -> None:
    """Thread-safe print."""
    if print_lock:
        with print_lock:
            print(msg, end=end, flush=True)
    else:
        print(msg, end=end, flush=True)


class RuleBasedConverter:
    """
    Rule-based converter: only handles API name and dtype conversions.

    Design constraints:
    - No LLM usage.
    - No complex semantic fixes; only automated name/type conversion.
    - API name mapping is driven by the external mapping table.
    """

    def __init__(self) -> None:
        self.tf_dtype_map = {
            "paddle.float16": tf.float16,
            "paddle.float32": tf.float32,
            "paddle.float64": tf.float64,
            "paddle.bfloat16": tf.bfloat16,
            "paddle.int8": tf.int8,
            "paddle.uint8": tf.uint8,
            "paddle.int16": tf.int16,
            "paddle.int32": tf.int32,
            "paddle.int64": tf.int64,
            "paddle.bool": tf.bool,
            "tensorflow.float16": tf.float16,
            "tensorflow.float32": tf.float32,
            "tensorflow.float64": tf.float64,
            "tensorflow.bfloat16": tf.bfloat16,
            "tensorflow.int8": tf.int8,
            "tensorflow.uint8": tf.uint8,
            "tensorflow.int16": tf.int16,
            "tensorflow.int32": tf.int32,
            "tensorflow.int64": tf.int64,
            "tensorflow.bool": tf.bool,
            "tf.float16": tf.float16,
            "tf.float32": tf.float32,
            "tf.float64": tf.float64,
            "tf.bfloat16": tf.bfloat16,
            "tf.int8": tf.int8,
            "tf.uint8": tf.uint8,
            "tf.int16": tf.int16,
            "tf.int32": tf.int32,
            "tf.int64": tf.int64,
            "tf.bool": tf.bool,
            "float16": tf.float16,
            "float32": tf.float32,
            "float64": tf.float64,
            "bfloat16": tf.bfloat16,
            "int8": tf.int8,
            "uint8": tf.uint8,
            "int16": tf.int16,
            "int32": tf.int32,
            "int64": tf.int64,
            "bool": tf.bool,
            "bool_": tf.bool,
            "double": tf.float64,
            "half": tf.float16,
            "long": tf.int64,
            "int": tf.int32,
            "float": tf.float32,
        }

    @staticmethod
    def _is_tensor_desc(value: Any) -> bool:
        return isinstance(value, dict) and "shape" in value and "dtype" in value

    @staticmethod
    def _normalize_tensor_desc_dtype(dtype_value: Any) -> Any:
        if not isinstance(dtype_value, str):
            return dtype_value

        normalized = dtype_value.strip()
        for prefix in ["paddle.", "tensorflow.", "tf.", "np.", "numpy.", "torch.", "mindspore.", "ms."]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]

        aliases = {
            "double": "float64",
            "half": "float16",
            "long": "int64",
            "int": "int32",
            "float": "float32",
            "bool_": "bool",
        }
        return aliases.get(normalized, normalized)

    def _convert_dtype_param_value(self, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized in self.tf_dtype_map:
                return self.tf_dtype_map[normalized]

            for prefix in ["paddle.", "tensorflow.", "tf.", "np.", "numpy.", "torch.", "mindspore.", "ms."]:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
            return self.tf_dtype_map.get(normalized, tf.float32)

        if isinstance(value, int):
            int_map = {
                0: tf.float32,
                1: tf.float64,
                2: tf.int32,
                3: tf.uint8,
                4: tf.int16,
                5: tf.int8,
                6: tf.int64,
                7: tf.bool,
            }
            return int_map.get(value, tf.float32)

        return value

    def _is_dtype_like_key(self, key: str) -> bool:
        key_lower = key.lower()
        return ("dtype" in key_lower) or (key_lower in {"type", "dst_type", "src_type"})

    def _convert_recursive(self, value: Any) -> Any:
        if isinstance(value, dict):
            if self._is_tensor_desc(value):
                converted = dict(value)
                converted["dtype"] = self._normalize_tensor_desc_dtype(converted.get("dtype"))
                return converted

            converted_dict: Dict[str, Any] = {}
            for key, child in value.items():
                if self._is_dtype_like_key(key):
                    converted_dict[key] = self._convert_dtype_param_value(child)
                else:
                    converted_dict[key] = self._convert_recursive(child)
            return converted_dict

        if isinstance(value, list):
            return [self._convert_recursive(item) for item in value]

        if isinstance(value, tuple):
            return tuple(self._convert_recursive(item) for item in value)

        return value

    def convert_pd_case_to_tf_case(self, pd_test_case: Dict[str, Any], tf_api: str) -> Dict[str, Any]:
        converted = self._convert_recursive(copy.deepcopy(pd_test_case))
        converted["api"] = tf_api
        return converted


class LLMvsRuleBasedComparator:
    def __init__(
        self,
        key_path: str = DEFAULT_KEY_PATH,
        model: str = DEFAULT_MODEL,
        num_workers: int = DEFAULT_WORKERS,
    ) -> None:
        self.print_lock = Lock()
        self.realtime_lock = Lock()
        self.execution_lock = RLock()
        self.num_workers = max(1, int(num_workers))

        self.llm_method = LLMEnhancedComparator(
            key_path=key_path,
            model=model,
            print_lock=self.print_lock,
            llm_workers=self.num_workers,
        )
        self.rule_converter = RuleBasedConverter()

        self.result_dir = os.path.join(ROOT_DIR, "pd_tf_test_1")
        os.makedirs(self.result_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.realtime_file_path = os.path.join(
            self.result_dir, f"llm_vs_rulebased_realtime_{timestamp}.jsonl"
        )
        self.realtime_file = open(self.realtime_file_path, "a", encoding="utf-8")

    def _safe_print(self, msg: str, end: str = "\n") -> None:
        safe_print(msg, self.print_lock, end)

    def _append_realtime_record(self, record: Dict[str, Any]) -> None:
        with self.realtime_lock:
            self.realtime_file.write(json.dumps(self._make_serializable(record), ensure_ascii=False) + "\n")
            self.realtime_file.flush()

    def _build_initial_case(self, pd_api: str, case_index: int) -> Dict[str, Any]:
        api_data = self.llm_method.test_cases_data.get(pd_api, {})
        test_cases = api_data.get("test_cases", []) if isinstance(api_data, dict) else []

        if case_index < len(test_cases):
            case_data = test_cases[case_index]
            if isinstance(case_data, dict):
                if "inputs" in case_data and isinstance(case_data["inputs"], dict):
                    flat_case = dict(case_data["inputs"])
                else:
                    flat_case = {k: v for k, v in case_data.items() if k != "description"}
            else:
                flat_case = {"x": {"shape": [2, 3], "dtype": "float32"}}
        else:
            flat_case = {"x": {"shape": [2, 3], "dtype": "float32"}}

        flat_case["api"] = pd_api
        if isinstance(api_data, dict) and "init_params" in api_data:
            flat_case["init_params"] = copy.deepcopy(api_data["init_params"])
        return flat_case

    def run_comparison(
        self,
        operator_names: List[str],
        num_cases: int = DEFAULT_NUM_CASES,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> Dict[str, Any]:
        if max_iterations != 1:
            self._safe_print("⚠️ This ablation uses max_iterations=1; forced to 1")

        global_stats = {
            "total_operators": len(operator_names),
            "tested_operators": 0,
            "skipped_operators_no_tf": 0,
            "skipped_operators_all_llm_skip": 0,
            "skipped_operators_deprecated": 0,
            "llm_generated_total": 0,
            "llm_tensorflow_success": 0,
            "rule_generated_total": 0,
            "rule_tensorflow_success": 0,
        }
        operator_details: List[Dict[str, Any]] = []

        for index, pd_api in enumerate(operator_names, 1):
            self._safe_print("\n" + "=" * 72)
            self._safe_print(f"[{index}/{len(operator_names)}] Operator: {pd_api}")
            self._safe_print("=" * 72)

            _, tf_api, mapping_method = self.llm_method.convert_api_name(pd_api)
            if tf_api is None:
                self._safe_print(f"  ⏭️ No TF mapping ({mapping_method}), skipped")
                global_stats["skipped_operators_no_tf"] += 1
                operator_details.append({"operator": pd_api, "status": "skipped_no_tf", "mapping_method": mapping_method})
                continue

            total_cases = len(self.llm_method.test_cases_data.get(pd_api, {}).get("test_cases", []))
            actual_cases = min(max(1, num_cases), total_cases) if total_cases > 0 else max(1, num_cases)

            self._safe_print(f"  PD: {pd_api} -> TF: {tf_api} ({mapping_method})")
            self._safe_print("  📖 Prefetching API docs (for LLM)...")
            pd_doc, tf_doc = self.llm_method._fetch_api_docs(pd_api, tf_api)

            operator_result = self._test_single_operator(pd_api, tf_api, actual_cases, pd_doc, tf_doc)
            operator_details.append(operator_result)

            if operator_result["status"] == "skipped_deprecated":
                global_stats["skipped_operators_deprecated"] += 1
                continue
            if operator_result["status"] == "skipped_all_llm_skip":
                global_stats["skipped_operators_all_llm_skip"] += 1
                continue

            global_stats["tested_operators"] += 1
            global_stats["llm_generated_total"] += operator_result["llm_generated_total"]
            global_stats["llm_tensorflow_success"] += operator_result["llm_tensorflow_success"]
            global_stats["rule_generated_total"] += operator_result["rule_generated_total"]
            global_stats["rule_tensorflow_success"] += operator_result["rule_tensorflow_success"]

        return {"global_stats": global_stats, "operator_details": operator_details}

    def _test_single_operator(self, pd_api: str, tf_api: str, num_cases: int, pd_doc: str, tf_doc: str) -> Dict[str, Any]:
        op_result: Dict[str, Any] = {
            "operator": pd_api,
            "paddle_api": pd_api,
            "tensorflow_api": tf_api,
            "num_cases": num_cases,
            "status": "completed",
            "llm_generated_total": 0,
            "llm_tensorflow_success": 0,
            "rule_generated_total": 0,
            "rule_tensorflow_success": 0,
            "case_details": [],
        }

        initial_cases = [(idx + 1, self._build_initial_case(pd_api, idx)) for idx in range(num_cases)]

        if self.num_workers <= 1:
            case_results = [
                self._process_single_case(pd_api, tf_api, case_number, test_case, pd_doc, tf_doc)
                for case_number, test_case in initial_cases
            ]
        else:
            case_results = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(self._process_single_case, pd_api, tf_api, case_number, test_case, pd_doc, tf_doc): case_number
                    for case_number, test_case in initial_cases
                }
                for future in as_completed(futures):
                    case_results.append(future.result())

        case_results.sort(key=lambda item: item["case_number"])

        if any(item.get("deprecated_skip", False) for item in case_results):
            op_result["status"] = "skipped_deprecated"
            op_result["case_details"] = case_results
            self._safe_print("  ⏭️ Operator deprecated, skipping")
            return op_result

        if case_results and all(item.get("llm_skipped", False) for item in case_results):
            op_result["status"] = "skipped_all_llm_skip"
            op_result["case_details"] = case_results
            self._safe_print("  ⏭️ All cases skipped by LLM; excluded")
            return op_result

        for item in case_results:
            if item.get("llm_skipped", False):
                continue
            op_result["llm_generated_total"] += item.get("llm_generated_total", 0)
            op_result["llm_tensorflow_success"] += item.get("llm_tensorflow_success", 0)
            op_result["rule_generated_total"] += item.get("rule_generated_total", 0)
            op_result["rule_tensorflow_success"] += item.get("rule_tensorflow_success", 0)

        op_result["case_details"] = case_results
        return op_result

    def _process_single_case(self, pd_api: str, tf_api: str, case_number: int, initial_pd_case: Dict[str, Any], pd_doc: str, tf_doc: str) -> Dict[str, Any]:
        case_result: Dict[str, Any] = {
            "case_number": case_number,
            "llm_skipped": False,
            "deprecated_skip": False,
            "llm_generated_total": 0,
            "llm_tensorflow_success": 0,
            "rule_generated_total": 1,
            "rule_tensorflow_success": 0,
            "rule_detail": {},
            "llm_detail": {},
        }

        rule_tf_case = self.rule_converter.convert_pd_case_to_tf_case(initial_pd_case, tf_api)

        self._safe_print(f"  [Case {case_number}] Rule-based execution...", end="")
        try:
            with self.execution_lock:
                rule_exec = self.llm_method._execute_test_case_sequential(pd_api, tf_api, initial_pd_case, rule_tf_case)
            rule_tf_ok = bool(rule_exec.get("tensorflow_success", False))
            case_result["rule_tensorflow_success"] = 1 if rule_tf_ok else 0
            self._safe_print(f" TF:{'✓' if rule_tf_ok else '✗'}")
        except Exception as error:
            rule_exec = {"status": "fatal_error", "tensorflow_success": False, "error": str(error), "traceback": traceback.format_exc()}
            self._safe_print(f" TF:✗ ({str(error)[:80]})")

        case_result["rule_detail"] = {"generated_tf_test_case": rule_tf_case, "execution_result": rule_exec}

        initial_tf_case = copy.deepcopy(initial_pd_case)
        initial_tf_case["api"] = tf_api

        self._safe_print(f"  [Case {case_number}] LLM initial execution...", end="")
        try:
            with self.execution_lock:
                initial_exec = self.llm_method._execute_test_case_sequential(pd_api, tf_api, initial_pd_case, initial_tf_case)
            self._safe_print(
                f" PD:{'✓' if initial_exec.get('pd_success') else '✗'}"
                f" TF:{'✓' if initial_exec.get('tensorflow_success') else '✗'}"
            )
        except Exception as error:
            initial_exec = {
                "paddle_api": pd_api,
                "tensorflow_api": tf_api,
                "status": "fatal_error",
                "pd_success": False,
                "tensorflow_success": False,
                "results_match": False,
                "pd_error": str(error),
                "tensorflow_error": None,
                "comparison_error": None,
            }
            self._safe_print(f" PD:✗ TF:✗ ({str(error)[:80]})")

        pd_error = str(initial_exec.get("pd_error", ""))
        if (not initial_exec.get("pd_success", False)) and pd_error:
            if re.search(r"deprecated|removed", pd_error, re.IGNORECASE):
                case_result["deprecated_skip"] = True
                case_result["llm_detail"] = {"initial_exec": initial_exec, "llm_operation": "skip", "llm_reason": "Paddle operator deprecated"}
                self._append_realtime_record(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "operator": pd_api,
                        "case_number": case_number,
                        "status": "skipped_deprecated",
                        "llm_skipped": False,
                        "deprecated_skip": True,
                        "llm_tensorflow_success": False,
                        "rule_tensorflow_success": bool(case_result["rule_tensorflow_success"]),
                    }
                )
                return case_result

        self._safe_print(f"  [Case {case_number}] Calling LLM...", end="")
        llm_result = self.llm_method.call_llm_for_repair_or_mutation(initial_exec, initial_pd_case, initial_tf_case, pd_doc, tf_doc)
        operation = llm_result.get("operation", "skip")
        reason = str(llm_result.get("reason", ""))
        self._safe_print(f" {operation} - {reason[:70]}")

        if operation == "skip":
            case_result["llm_skipped"] = True
            case_result["llm_detail"] = {"initial_exec": initial_exec, "llm_operation": operation, "llm_reason": reason}
            self._append_realtime_record(
                {
                    "timestamp": datetime.now().isoformat(),
                    "operator": pd_api,
                    "case_number": case_number,
                    "status": "llm_skip",
                    "llm_skipped": True,
                    "deprecated_skip": False,
                    "llm_tensorflow_success": False,
                    "rule_tensorflow_success": bool(case_result["rule_tensorflow_success"]),
                }
            )
            return case_result

        llm_pd_case = llm_result.get("paddle_test_case", initial_pd_case)
        llm_tf_case = llm_result.get("tensorflow_test_case", initial_tf_case)

        try:
            llm_pd_case, llm_tf_case = self.llm_method._convert_llm_test_cases(llm_pd_case, llm_tf_case)
        except Exception as error:
            case_result["llm_skipped"] = True
            case_result["llm_detail"] = {"initial_exec": initial_exec, "llm_operation": "skip", "llm_reason": f"LLM case conversion failed: {error}"}
            self._safe_print(f"  [Case {case_number}] LLM case conversion failed, skipped: {str(error)[:70]}")
            self._append_realtime_record(
                {
                    "timestamp": datetime.now().isoformat(),
                    "operator": pd_api,
                    "case_number": case_number,
                    "status": "llm_case_convert_failed",
                    "llm_skipped": True,
                    "deprecated_skip": False,
                    "llm_tensorflow_success": False,
                    "rule_tensorflow_success": bool(case_result["rule_tensorflow_success"]),
                }
            )
            return case_result

        case_result["llm_generated_total"] = 1

        self._safe_print(f"  [Case {case_number}] Executing LLM-generated case...", end="")
        try:
            with self.execution_lock:
                llm_exec = self.llm_method._execute_test_case_sequential(pd_api, tf_api, llm_pd_case, llm_tf_case)
            llm_tf_ok = bool(llm_exec.get("tensorflow_success", False))
            case_result["llm_tensorflow_success"] = 1 if llm_tf_ok else 0
            self._safe_print(f" PD:{'✓' if llm_exec.get('pd_success') else '✗'} TF:{'✓' if llm_tf_ok else '✗'}")
        except Exception as error:
            llm_exec = {"status": "fatal_error", "pd_success": False, "tensorflow_success": False, "results_match": False, "error": str(error), "traceback": traceback.format_exc()}
            self._safe_print(f" PD:✗ TF:✗ ({str(error)[:80]})")

        case_result["llm_detail"] = {
            "initial_exec": initial_exec,
            "llm_operation": operation,
            "llm_reason": reason,
            "generated_pd_test_case": llm_pd_case,
            "generated_tf_test_case": llm_tf_case,
            "generated_exec": llm_exec,
        }

        self._append_realtime_record(
            {
                "timestamp": datetime.now().isoformat(),
                "operator": pd_api,
                "case_number": case_number,
                "status": "completed",
                "llm_skipped": False,
                "deprecated_skip": False,
                "llm_tensorflow_success": bool(case_result["llm_tensorflow_success"]),
                "rule_tensorflow_success": bool(case_result["rule_tensorflow_success"]),
                "llm_operation": operation,
            }
        )

        return case_result

    def print_and_save_results(self, result: Dict[str, Any]) -> str:
        stats = result["global_stats"]

        print("\n" + "=" * 80)
        print("📊 LLM vs Rule-based (mapping table + dtype auto-conversion)")
        print("=" * 80)
        print(f"Total operators: {stats['total_operators']}")
        print(f"- Skipped (no TF mapping): {stats['skipped_operators_no_tf']}")
        print(f"- Skipped (all LLM skipped): {stats['skipped_operators_all_llm_skip']}")
        print(f"- Skipped (deprecated): {stats['skipped_operators_deprecated']}")
        print(f"- Actually compared: {stats['tested_operators']}")

        llm_total = stats["llm_generated_total"]
        llm_success = stats["llm_tensorflow_success"]
        llm_rate = (llm_success / llm_total * 100.0) if llm_total > 0 else 0.0

        rule_total = stats["rule_generated_total"]
        rule_success = stats["rule_tensorflow_success"]
        rule_rate = (rule_success / rule_total * 100.0) if rule_total > 0 else 0.0

        print("\n" + "-" * 48)
        print("🤖 LLM method (generated TF cases)")
        print(f"- Generated cases: {llm_total}")
        print(f"- TF success count: {llm_success}")
        print(f"- TF success rate: {llm_rate:.2f}%" if llm_total > 0 else "- TF success rate: N/A")

        print("\n" + "-" * 48)
        print("🧪 Rule-based method (generated TF cases)")
        print(f"- Generated cases: {rule_total}")
        print(f"- TF success count: {rule_success}")
        print(f"- TF success rate: {rule_rate:.2f}%" if rule_total > 0 else "- TF success rate: N/A")

        print("\n" + "-" * 48)
        if llm_total > 0 and rule_total > 0:
            diff = llm_rate - rule_rate
            if diff > 0:
                print(f"Conclusion: LLM exceeds rule-based by {diff:.2f} percentage points")
            elif diff < 0:
                print(f"Conclusion: Rule-based exceeds LLM by {-diff:.2f} percentage points")
            else:
                print("Conclusion: tie")
        else:
            print("Conclusion: insufficient samples to compare")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.result_dir, f"llm_vs_rulebased_result_{timestamp}.json")

        result_to_save = self._make_serializable(result)
        result_to_save["summary"] = {
            "llm_tensorflow_success_rate": f"{llm_rate:.2f}%" if llm_total > 0 else "N/A",
            "rule_tensorflow_success_rate": f"{rule_rate:.2f}%" if rule_total > 0 else "N/A",
            "timestamp": datetime.now().isoformat(),
        }

        self._atomic_dump_json(result_file, result_to_save)

        print("=" * 80)
        print(f"💾 Realtime JSONL: {self.realtime_file_path}")
        print(f"💾 Summary JSON: {result_file}")

        return result_file

    @staticmethod
    def _atomic_dump_json(file_path: str, data: Dict[str, Any]) -> None:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp", prefix="tmp_result_", dir=file_dir, delete=False) as temp_file:
            temp_path = temp_file.name
            json.dump(data, temp_file, indent=2, ensure_ascii=False)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(temp_path, file_path)

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: LLMvsRuleBasedComparator._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [LLMvsRuleBasedComparator._make_serializable(item) for item in obj]
        if isinstance(obj, tuple):
            return [LLMvsRuleBasedComparator._make_serializable(item) for item in obj]
        if isinstance(obj, np.ndarray):
            return {"__type__": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, paddle.Tensor):
            return {"__type__": "paddle_tensor", "shape": list(obj.shape), "dtype": str(obj.dtype)}
        if isinstance(obj, tf.Tensor):
            return {"__type__": "tf_tensor", "shape": obj.shape.as_list() if hasattr(obj.shape, "as_list") else list(obj.shape), "dtype": str(obj.dtype)}
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="ignore")

        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

    def close(self) -> None:
        try:
            if self.realtime_file:
                self.realtime_file.flush()
                self.realtime_file.close()
        finally:
            self.llm_method.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM vs rule-based: Paddle->TensorFlow case execution success rate")
    parser.add_argument("--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES, help=f"Test cases per operator (default {DEFAULT_NUM_CASES})")
    parser.add_argument("--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS, help="LLM iterations (fixed to 1 in this script)")
    parser.add_argument("--start", type=int, default=1, help="Start operator index (1-based)")
    parser.add_argument("--end", type=int, default=None, help="End operator index (inclusive)")
    parser.add_argument("--operators", "-o", nargs="*", help="Explicit operator list")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS, help=f"Worker threads (default {DEFAULT_WORKERS})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM model (default {DEFAULT_MODEL})")
    parser.add_argument("--key-path", "-k", default=DEFAULT_KEY_PATH, help=f"API key file path (default {DEFAULT_KEY_PATH})")
    args = parser.parse_args()

    print("=" * 80)
    print("LLM vs rule-based (mapping table + dtype auto-conversion)")
    print("=" * 80)
    print(f"📌 Cases per operator: {args.num_cases}")
    print("📌 LLM iterations: 1 (fixed)")
    print(f"📌 Worker threads: {args.workers}")
    print(f"📌 LLM model: {args.model}")
    print("=" * 80)

    comparator = LLMvsRuleBasedComparator(key_path=args.key_path, model=args.model, num_workers=args.workers)
    start_time = time.time()

    try:
        all_operators = sorted(list(comparator.llm_method.test_cases_data.keys()))
        print(f"\n📋 Test set contains {len(all_operators)} Paddle operators")

        if args.operators:
            operator_names = args.operators
        else:
            start_idx = max(1, args.start) - 1
            end_idx = args.end if args.end is not None else len(all_operators)
            end_idx = min(end_idx, len(all_operators))
            if start_idx >= end_idx:
                raise ValueError(f"Start index {args.start} must be less than end index {end_idx}")
            operator_names = all_operators[start_idx:end_idx]
            print(f"📌 Test range: operators {start_idx + 1} to {end_idx}")

        print(f"📋 Operators tested: {len(operator_names)}")
        preview = ", ".join(operator_names[:10])
        print(f"📋 First 10 operators: {preview}{'...' if len(operator_names) > 10 else ''}\n")

        result = comparator.run_comparison(operator_names=operator_names, num_cases=args.num_cases, max_iterations=args.max_iterations)
        comparator.print_and_save_results(result)

        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        print(f"\n⏱️ Total time: {hours}h {minutes}m {seconds}s")

    finally:
        comparator.close()
        print("✅ Program completed")


if __name__ == "__main__":
    main()
