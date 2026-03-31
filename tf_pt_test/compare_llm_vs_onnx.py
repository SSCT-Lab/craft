#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM vs ONNX: TensorFlow → PyTorch test-case conversion success rate
==================================================================

Compare two cross-framework test-case migration approaches:
1. LLM method: based on tf_pt_test/llm_enhanced_compare.py execution, repair/mutation, and concurrency flow (iterations fixed at 1)
2. ONNX method: wrap TensorFlow operators as tf.Module, export ONNX, and run inference with onnxruntime

Metric: LLM-generated PT case success rate vs ONNX export+inference success rate
(excluding operators skipped by LLM)
"""

import os
import sys
import re
import json
import copy
import time
import argparse
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from threading import Lock, RLock
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tensorflow as tf
import onnxruntime as ort
import importlib

try:
    tf2onnx = importlib.import_module("tf2onnx")
except Exception:
    tf2onnx = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tf_pt_test.llm_enhanced_compare import (  # noqa: E402
    LLMEnhancedComparator,
    DEFAULT_MODEL,
    DEFAULT_KEY_PATH,
)

DEFAULT_MAX_ITERATIONS = 1
DEFAULT_NUM_CASES = 3
DEFAULT_WORKERS = 4


def safe_print(msg: str, print_lock: Optional[Lock] = None, end: str = "\n"):
    if print_lock:
        with print_lock:
            print(msg, end=end, flush=True)
    else:
        print(msg, end=end, flush=True)


class _FuncWrapper(tf.Module):
    """Wrap a plain function as a tf.Module."""

    def __init__(self, func):
        super().__init__()
        self._func = func

    def __call__(self, *args):
        return self._func(*args)


class OnnxConverter:
    """Simulate the traditional TensorFlow → PyTorch conversion path via ONNX."""

    def __init__(self, print_lock: Optional[Lock] = None):
        self.print_lock = print_lock or Lock()
        self.execution_lock = RLock()

    def _safe_print(self, msg: str, end: str = "\n"):
        safe_print(msg, self.print_lock, end)

    @staticmethod
    def _resolve_attr(parts: List[str]):
        try:
            obj = tf
            for part in parts[1:]:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return None

    def _wrap_as_module(
        self,
        tf_api: str,
        is_class_api: bool,
        init_kwargs: Dict[str, Any],
        extra_kwargs: Dict[str, Any],
    ) -> Optional[tf.Module]:
        parts = tf_api.split(".")

        try:
            if is_class_api:
                cls = self._resolve_attr(parts)
                if cls is None:
                    return None
                instance = cls(**init_kwargs)
                return _FuncWrapper(lambda *args, _inst=instance: _inst(*args))

            func = self._resolve_attr(parts)
            if func is None:
                return None
            if extra_kwargs:
                return _FuncWrapper(lambda *args, _f=func, _kw=extra_kwargs: _f(*args, **_kw))
            return _FuncWrapper(func)
        except Exception:
            return None

    @staticmethod
    def _to_tf_tensor(value: Any) -> Optional[tf.Tensor]:
        if isinstance(value, tf.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return tf.convert_to_tensor(value)
        if isinstance(value, dict) and "shape" in value:
            dtype_str = str(value.get("dtype", "float32"))
            for prefix in ["tf.", "torch.", "np.", "numpy."]:
                if dtype_str.startswith(prefix):
                    dtype_str = dtype_str[len(prefix):]

            dtype_map = {
                "float16": np.float16,
                "float32": np.float32,
                "float64": np.float64,
                "int8": np.int8,
                "int16": np.int16,
                "int32": np.int32,
                "int64": np.int64,
                "uint8": np.uint8,
                "bool": np.bool_,
            }
            np_dtype = dtype_map.get(dtype_str, np.float32)
            shape = value.get("shape", [])
            if isinstance(shape, list) and any(dim == 0 for dim in shape):
                array = np.empty(shape, dtype=np_dtype)
            elif np_dtype == np.bool_:
                array = np.random.choice([True, False], size=shape).astype(np.bool_)
            elif np.issubdtype(np_dtype, np.integer):
                array = np.random.randint(0, 10, size=shape).astype(np_dtype)
            else:
                array = np.random.randn(*shape).astype(np_dtype)
            return tf.convert_to_tensor(array)
        if isinstance(value, (list, tuple, int, float, bool)):
            return tf.convert_to_tensor(value)
        return None

    def _prepare_inputs(
        self,
        test_case: Dict[str, Any],
        is_class_api: bool,
    ) -> Tuple[List[tf.Tensor], Dict[str, Any], List[str], Dict[str, Any]]:
        input_tensors: List[tf.Tensor] = []
        input_names: List[str] = []
        init_kwargs: Dict[str, Any] = {}
        extra_kwargs: Dict[str, Any] = {}

        positional_tensor_params = ["condition", "input", "x", "y", "other"]
        skip_params = {"api", "layout", "requires_grad", "out"}

        varargs_key = next((key for key in test_case.keys() if key.startswith("*")), None)
        if varargs_key:
            varargs_value = test_case[varargs_key]
            if isinstance(varargs_value, list):
                for index, item in enumerate(varargs_value):
                    tensor = self._to_tf_tensor(item)
                    if tensor is not None:
                        input_tensors.append(tensor)
                        input_names.append(f"input_{index}")
            return input_tensors, init_kwargs, input_names, extra_kwargs

        for name in positional_tensor_params:
            if name not in test_case:
                continue
            tensor = self._to_tf_tensor(test_case[name])
            if tensor is not None:
                input_tensors.append(tensor)
                input_names.append(name)

        for key, value in test_case.items():
            if key in positional_tensor_params or key in skip_params or key.startswith("*"):
                continue
            if is_class_api:
                if not isinstance(value, (np.ndarray, dict, list, tuple)):
                    init_kwargs[key] = value
            else:
                if isinstance(value, (int, float, bool, str)):
                    extra_kwargs[key] = value

        return input_tensors, init_kwargs, input_names, extra_kwargs

    def convert_and_run(self, tf_api: str, test_case: Dict[str, Any], is_class_api: bool) -> Dict[str, Any]:
        result = {
            "onnx_export_success": False,
            "onnx_run_success": False,
            "tf_success": False,
            "error": None,
            "tf_shape": None,
            "onnx_shape": None,
        }

        if tf2onnx is None:
            result["error"] = "tf2onnx is not installed; cannot perform ONNX export"
            return result

        with self.execution_lock:
            try:
                input_tensors, init_kwargs, input_names, extra_kwargs = self._prepare_inputs(test_case, is_class_api)
            except Exception as error:
                result["error"] = f"Input preparation failed: {error}"
                return result

            if not input_tensors:
                result["error"] = "No valid input tensors"
                return result

            module = self._wrap_as_module(tf_api, is_class_api, init_kwargs, extra_kwargs)
            if module is None:
                result["error"] = f"Failed to wrap operator {tf_api} as tf.Module"
                return result

            try:
                tf_output = module(*input_tensors)
                result["tf_success"] = True
                if hasattr(tf_output, "shape"):
                    result["tf_shape"] = list(tf_output.shape)
            except Exception as error:
                result["error"] = f"TensorFlow forward pass failed: {error}"
                return result

            onnx_path = None
            try:
                onnx_fd, onnx_path = tempfile.mkstemp(suffix=".onnx")
                os.close(onnx_fd)

                signatures = [
                    tf.TensorSpec(shape=tensor.shape, dtype=tensor.dtype, name=input_names[index] if index < len(input_names) else f"input_{index}")
                    for index, tensor in enumerate(input_tensors)
                ]

                @tf.function(input_signature=signatures)
                def wrapped(*args):
                    return module(*args)

                tf2onnx.convert.from_function(
                    wrapped,
                    input_signature=signatures,
                    opset=14,
                    output_path=onnx_path,
                )
                result["onnx_export_success"] = True
            except Exception as error:
                result["error"] = f"ONNX export failed: {error}"
                if onnx_path and os.path.exists(onnx_path):
                    os.remove(onnx_path)
                return result

            try:
                session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                ort_inputs = session.get_inputs()
                feed = {}
                for index, tensor in enumerate(input_tensors):
                    input_name = ort_inputs[index].name if index < len(ort_inputs) else f"input_{index}"
                    feed[input_name] = tensor.numpy()
                outputs = session.run(None, feed)
                onnx_result = outputs[0]
                result["onnx_run_success"] = True
                result["onnx_shape"] = list(onnx_result.shape)
            except Exception as error:
                result["error"] = f"OnnxRuntime inference failed: {error}"
            finally:
                if onnx_path and os.path.exists(onnx_path):
                    os.remove(onnx_path)

        return result


class LLMvsONNXComparator:
    """Manage comparative testing between LLM and ONNX methods."""

    def __init__(self, key_path: str = DEFAULT_KEY_PATH, model: str = DEFAULT_MODEL, num_workers: int = DEFAULT_WORKERS):
        self.print_lock = Lock()
        self.execution_lock = RLock()
        self.realtime_lock = Lock()
        self.num_workers = max(1, int(num_workers))

        self.llm_method = LLMEnhancedComparator(
            key_path=key_path,
            model=model,
            print_lock=self.print_lock,
            llm_workers=self.num_workers,
        )
        self.onnx_converter = OnnxConverter(print_lock=self.print_lock)

        self.result_dir = os.path.join(ROOT_DIR, "tf_pt_test")
        os.makedirs(self.result_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.realtime_file_path = os.path.join(self.result_dir, f"llm_vs_onnx_realtime_{timestamp}.jsonl")
        self.realtime_file = open(self.realtime_file_path, "a", encoding="utf-8")

    def _safe_print(self, msg: str, end: str = "\n"):
        safe_print(msg, self.print_lock, end)

    def _append_realtime_record(self, record: Dict[str, Any]):
        with self.realtime_lock:
            self.realtime_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.realtime_file.flush()

    def _build_initial_case(self, tf_api: str, case_index: int) -> Dict[str, Any]:
        api_data = self.llm_method.test_cases_data.get(tf_api, {})
        test_cases = api_data.get("test_cases", [])
        if case_index < len(test_cases):
            case_data = test_cases[case_index]
            flat_case = dict(case_data["inputs"]) if "inputs" in case_data else {k: v for k, v in case_data.items() if k != "description"}
        else:
            flat_case = {"x": {"shape": [2, 3], "dtype": "float32"}}
        flat_case["api"] = tf_api
        return flat_case

    def run_comparison(
        self,
        operator_names: List[str],
        num_cases: int = DEFAULT_NUM_CASES,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> Dict[str, Any]:
        max_iterations = 1
        global_stats = {
            "total_operators": len(operator_names),
            "tested_operators": 0,
            "skipped_operators_llm": 0,
            "skipped_operators_no_target": 0,
            "skipped_operators_deprecated": 0,
            "llm_total_cases": 0,
            "llm_pt_success": 0,
            "onnx_total_cases": 0,
            "onnx_export_success": 0,
            "onnx_run_success": 0,
        }

        operator_details = []

        for index, tf_api in enumerate(operator_names, 1):
            self._safe_print(f"\n{'=' * 70}")
            self._safe_print(f"[{index}/{len(operator_names)}] Operator: {tf_api}")
            self._safe_print(f"{'=' * 70}")

            tf_name, pt_api, mapping_method = self.llm_method.convert_api_name(tf_api)
            if pt_api is None:
                self._safe_print(f"  ⏭️ Skipped (no PT mapping): {tf_api} ({mapping_method})")
                global_stats["skipped_operators_no_target"] += 1
                operator_details.append({
                    "operator": tf_api,
                    "status": "skipped_no_target",
                    "reason": mapping_method,
                })
                continue

            self._safe_print(f"  TF: {tf_name} → PT: {pt_api}")

            total_cases = len(self.llm_method.test_cases_data.get(tf_api, {}).get("test_cases", []))
            actual_cases = min(num_cases, total_cases) if total_cases > 0 else num_cases

            op_result = self._test_operator(tf_api, pt_api, actual_cases, max_iterations)
            operator_details.append(op_result)

            if op_result["status"] == "skipped_by_llm":
                global_stats["skipped_operators_llm"] += 1
            elif op_result["status"] == "skipped_deprecated":
                global_stats["skipped_operators_deprecated"] += 1
            else:
                global_stats["tested_operators"] += 1
                global_stats["llm_total_cases"] += op_result["llm_total"]
                global_stats["llm_pt_success"] += op_result["llm_pt_success"]
                global_stats["onnx_total_cases"] += op_result["onnx_total"]
                global_stats["onnx_export_success"] += op_result["onnx_export_success"]
                global_stats["onnx_run_success"] += op_result["onnx_run_success"]

        return {"global_stats": global_stats, "operator_details": operator_details}

    def _test_operator(self, tf_api: str, pt_api: str, num_cases: int, max_iterations: int) -> Dict[str, Any]:
        _ = max_iterations
        op_result = {
            "operator": tf_api,
            "tf_api": tf_api,
            "target_api": pt_api,
            "status": "completed",
            "num_cases": num_cases,
            "llm_total": 0,
            "llm_pt_success": 0,
            "onnx_total": 0,
            "onnx_export_success": 0,
            "onnx_run_success": 0,
            "case_details": [],
        }

        is_class_api = self.llm_method.is_class_based_api(tf_api)

        self._safe_print("  📖 Fetching API docs...")
        tf_doc, pt_doc = self.llm_method._fetch_api_docs(tf_api, pt_api)

        cases = []
        for case_idx in range(num_cases):
            cases.append((case_idx + 1, self._build_initial_case(tf_api, case_idx)))

        if self.num_workers <= 1:
            case_results = [
                self._process_single_case(tf_api, pt_api, case_data, case_number, is_class_api, tf_doc, pt_doc)
                for case_number, case_data in cases
            ]
        else:
            case_results = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [
                    executor.submit(
                        self._process_single_case,
                        tf_api,
                        pt_api,
                        case_data,
                        case_number,
                        is_class_api,
                        tf_doc,
                        pt_doc,
                    )
                    for case_number, case_data in cases
                ]
                for future in as_completed(futures):
                    case_results.append(future.result())

        case_results.sort(key=lambda item: item["case_number"])

        if any(item.get("deprecated_skip", False) for item in case_results):
            op_result["status"] = "skipped_deprecated"
            op_result["case_details"] = case_results
            self._safe_print("  ⏭️ This operator may be deprecated; skipped by rule")
            return op_result

        if case_results and all(item.get("llm_skipped", False) for item in case_results):
            op_result["status"] = "skipped_by_llm"
            op_result["case_details"] = case_results
            self._safe_print("  ⏭️ LLM chose to skip this operator")
            return op_result

        for item in case_results:
            if item.get("llm_skipped", False):
                continue
            op_result["llm_total"] += item["llm_total"]
            op_result["llm_pt_success"] += item["llm_pt_success"]
            op_result["onnx_total"] += item["onnx_total"]
            op_result["onnx_export_success"] += item["onnx_export_success"]
            op_result["onnx_run_success"] += item["onnx_run_success"]

        op_result["case_details"] = case_results
        return op_result

    def _process_single_case(
        self,
        tf_api: str,
        pt_api: str,
        test_case: Dict[str, Any],
        case_number: int,
        is_class_api: bool,
        tf_doc: str,
        pt_doc: str,
    ) -> Dict[str, Any]:
        case_result = {
            "case_number": case_number,
            "llm_skipped": False,
            "deprecated_skip": False,
            "llm_total": 0,
            "llm_pt_success": 0,
            "onnx_total": 0,
            "onnx_export_success": 0,
            "onnx_run_success": 0,
            "llm_detail": None,
            "onnx_detail": None,
        }

        self._safe_print(f"  [Case {case_number}] ONNX method...", end="")
        onnx_result = self.onnx_converter.convert_and_run(tf_api, test_case, is_class_api)
        case_result["onnx_total"] = 1
        if onnx_result["onnx_export_success"]:
            case_result["onnx_export_success"] = 1
        if onnx_result["onnx_run_success"]:
            case_result["onnx_run_success"] = 1
        onnx_status = "✓" if onnx_result["onnx_run_success"] else "✗"
        onnx_err = f" ({onnx_result['error'][:60]})" if onnx_result.get("error") else ""
        self._safe_print(f" ONNX:{onnx_status}{onnx_err}")
        case_result["onnx_detail"] = onnx_result

        tf_test_case = copy.deepcopy(test_case)
        pt_test_case = copy.deepcopy(test_case)
        pt_test_case["api"] = pt_api

        self._safe_print(f"  [Case {case_number}] LLM method: initial execution...", end="")
        try:
            with self.execution_lock:
                exec_result = self.llm_method._execute_test_case_sequential(
                    tf_api,
                    pt_api,
                    tf_test_case,
                    pt_test_case,
                )
            tf_s = "✓" if exec_result.get("tf_success") else "✗"
            pt_s = "✓" if exec_result.get("pytorch_success") else "✗"
            self._safe_print(f" TF:{tf_s} PT:{pt_s}")
        except Exception as error:
            self._safe_print(f" ❌ Execution failed: {str(error)[:60]}")
            exec_result = {
                "tf_api": tf_api,
                "pytorch_api": pt_api,
                "tf_success": False,
                "pytorch_success": False,
                "results_match": False,
                "status": "fatal_error",
                "tf_error": str(error),
                "pytorch_error": None,
                "comparison_error": None,
            }

        tf_error = str(exec_result.get("tf_error", ""))
        if not exec_result.get("tf_success", False) and tf_error and re.search(r"deprecated|removed", tf_error, re.IGNORECASE):
            case_result["deprecated_skip"] = True
            case_result["llm_skipped"] = True
            case_result["llm_detail"] = {
                "initial_exec": exec_result,
                "llm_operation": "skip",
                "llm_reason": f"Detected deprecation info: {tf_error[:120]}",
            }
            return case_result

        self._safe_print(f"  [Case {case_number}] Calling LLM...", end="")
        llm_result = self.llm_method.call_llm_for_repair_or_mutation(
            exec_result,
            tf_test_case,
            pt_test_case,
            tf_doc,
            pt_doc,
        )
        operation = llm_result.get("operation", "skip")
        reason = llm_result.get("reason", "")[:60]
        self._safe_print(f" {operation} - {reason}")

        if operation == "skip":
            case_result["llm_skipped"] = True
            case_result["llm_detail"] = {
                "initial_exec": exec_result,
                "llm_operation": operation,
                "llm_reason": llm_result.get("reason", ""),
            }
            self._append_realtime_record(
                {
                    "timestamp": datetime.now().isoformat(),
                    "operator": tf_api,
                    "case_number": case_number,
                    "status": "llm_skip",
                    "onnx_run_success": bool(case_result["onnx_run_success"]),
                    "llm_pt_success": False,
                    "llm_skipped": True,
                    "deprecated_skip": False,
                }
            )
            return case_result

        llm_tf_case = llm_result.get("tensorflow_test_case", tf_test_case)
        llm_pt_case = llm_result.get("pytorch_test_case", pt_test_case)

        try:
            llm_tf_case, llm_pt_case = self.llm_method._convert_llm_test_cases(llm_tf_case, llm_pt_case)
        except Exception as error:
            case_result["llm_skipped"] = True
            case_result["llm_detail"] = {
                "initial_exec": exec_result,
                "llm_operation": "skip",
                "llm_reason": f"LLM case conversion failed: {error}",
            }
            self._safe_print(f"  [Case {case_number}] LLM case conversion failed, skipped: {str(error)[:60]}")
            return case_result

        case_result["llm_total"] = 1

        self._safe_print(f"  [Case {case_number}] Executing LLM case...", end="")
        try:
            with self.execution_lock:
                llm_exec = self.llm_method._execute_test_case_sequential(
                    tf_api,
                    pt_api,
                    llm_tf_case,
                    llm_pt_case,
                )
            tf_s = "✓" if llm_exec.get("tf_success") else "✗"
            pt_s = "✓" if llm_exec.get("pytorch_success") else "✗"
            match_s = "✓" if llm_exec.get("results_match") else "✗"
            self._safe_print(f" TF:{tf_s} PT:{pt_s} Match:{match_s}")
            if llm_exec.get("pytorch_success", False):
                case_result["llm_pt_success"] = 1
        except Exception as error:
            self._safe_print(f" ❌ {str(error)[:60]}")
            llm_exec = {
                "tf_success": False,
                "pytorch_success": False,
                "results_match": False,
                "status": "fatal_error",
                "error": str(error),
            }

        case_result["llm_detail"] = {
            "initial_exec": exec_result,
            "llm_operation": operation,
            "llm_reason": llm_result.get("reason", ""),
            "llm_exec": llm_exec,
        }

        self._append_realtime_record(
            {
                "timestamp": datetime.now().isoformat(),
                "operator": tf_api,
                "case_number": case_number,
                "status": "completed",
                "onnx_run_success": bool(case_result["onnx_run_success"]),
                "llm_pt_success": bool(case_result["llm_pt_success"]),
                "llm_skipped": False,
                "deprecated_skip": False,
            }
        )
        return case_result

    def print_and_save_results(self, comparison_result: Dict[str, Any]):
        gs = comparison_result["global_stats"]

        print("\n" + "=" * 80)
        print("📊 LLM vs ONNX — TF→PT test-case conversion success rate")
        print("=" * 80)

        print(f"\n📌 Total operators: {gs['total_operators']}")
        print(f"   - Skipped (no target mapping): {gs['skipped_operators_no_target']}")
        print(f"   - Skipped by LLM: {gs['skipped_operators_llm']}")
        print(f"   - Skipped (deprecated): {gs['skipped_operators_deprecated']}")
        print(f"   - Tested: {gs['tested_operators']}")

        print(f"\n{'─' * 40}")
        print("🤖 LLM method (excluding skipped operators):")
        print(f"   LLM-generated PT cases: {gs['llm_total_cases']}")
        print(f"   PT successes: {gs['llm_pt_success']}")
        if gs["llm_total_cases"] > 0:
            llm_rate = gs["llm_pt_success"] / gs["llm_total_cases"] * 100
            print(f"   ✅ PT success rate: {llm_rate:.2f}%")
        else:
            llm_rate = 0.0
            print("   ✅ PT success rate: N/A (no LLM cases)")

        print(f"\n{'─' * 40}")
        print("🔄 ONNX method (excluding LLM-skipped operators):")
        print(f"   ONNX conversion attempts: {gs['onnx_total_cases']}")
        print(f"   ONNX export successes: {gs['onnx_export_success']}")
        print(f"   ONNX inference successes (= conversions): {gs['onnx_run_success']}")
        if gs["onnx_total_cases"] > 0:
            onnx_rate = gs["onnx_run_success"] / gs["onnx_total_cases"] * 100
            print(f"   ✅ ONNX conversion success rate: {onnx_rate:.2f}%")
        else:
            onnx_rate = 0.0
            print("   ✅ ONNX conversion success rate: N/A")

        print(f"\n{'─' * 40}")
        print("📈 Conclusion:")
        if gs["llm_total_cases"] > 0 and gs["onnx_total_cases"] > 0:
            diff = llm_rate - onnx_rate
            if diff > 0:
                print(f"   LLM outperforms ONNX by {diff:.2f} percentage points")
            elif diff < 0:
                print(f"   ONNX outperforms LLM by {abs(diff):.2f} percentage points")
            else:
                print("   Tie")
        else:
            print("   Insufficient data to compare")
        print("=" * 80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.result_dir, f"llm_vs_onnx_result_{timestamp}.json")
        save_data = self._make_serializable(comparison_result)
        save_data["summary"] = {
            "llm_pt_success_rate": f"{llm_rate:.2f}%" if gs["llm_total_cases"] > 0 else "N/A",
            "onnx_success_rate": f"{onnx_rate:.2f}%" if gs["onnx_total_cases"] > 0 else "N/A",
            "skipped_operators_deprecated": gs.get("skipped_operators_deprecated", 0),
            "timestamp": datetime.now().isoformat(),
        }

        with open(result_file, "w", encoding="utf-8") as file:
            json.dump(save_data, file, indent=2, ensure_ascii=False)

        print(f"\n💾 Detailed results saved to: {result_file}")

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(key): LLMvsONNXComparator._make_serializable(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [LLMvsONNXComparator._make_serializable(item) for item in obj]
        if isinstance(obj, tuple):
            return [LLMvsONNXComparator._make_serializable(item) for item in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="ignore")
        return obj

    def close(self):
        try:
            if self.realtime_file:
                self.realtime_file.close()
        finally:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="LLM vs ONNX: TF→PT test case conversion success rate comparison"
    )
    parser.add_argument(
        "--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES,
        help=f"Test cases per operator (default {DEFAULT_NUM_CASES})"
    )
    parser.add_argument(
        "--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS,
        help="LLM iterations (fixed at 1)"
    )
    parser.add_argument("--start", type=int, default=1, help="Start operator index (1-based)")
    parser.add_argument("--end", type=int, default=None, help="End operator index (inclusive)")
    parser.add_argument("--operators", "-o", nargs="*", help="Specify operator name list")
    parser.add_argument(
        "--workers", "-w", type=int, default=DEFAULT_WORKERS,
        help=f"LLM concurrent workers (default {DEFAULT_WORKERS})"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM model (default {DEFAULT_MODEL})")
    parser.add_argument(
        "--key-path", "-k", default=DEFAULT_KEY_PATH,
        help=f"API key path (default {DEFAULT_KEY_PATH})"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("LLM vs ONNX — TensorFlow→PyTorch test case conversion success rate comparison")
    print("=" * 80)
    print(f"📌 Cases per operator: {args.num_cases}")
    print("📌 LLM iterations: 1 (fixed)")
    print(f"📌 Workers: {args.workers}")
    print(f"📌 LLM model: {args.model}")
    print("=" * 80)

    comparator = LLMvsONNXComparator(key_path=args.key_path, model=args.model, num_workers=args.workers)
    start_time = time.time()

    try:
        all_ops = sorted(list(comparator.llm_method.test_cases_data.keys()))
        print(f"\n📋 Test set has {len(all_ops)} TF operators")

        if args.operators:
            operator_names = args.operators
        else:
            start_idx = max(1, args.start) - 1
            end_idx = args.end if args.end is not None else len(all_ops)
            end_idx = min(end_idx, len(all_ops))
            operator_names = all_ops[start_idx:end_idx]
            print(f"📌 Test range: operators {start_idx + 1} ~ {end_idx}")

        print(f"📋 Will test {len(operator_names)} operators")
        print(f"📋 First 10: {', '.join(operator_names[:10])}{'...' if len(operator_names) > 10 else ''}\n")

        result = comparator.run_comparison(
            operator_names=operator_names,
            num_cases=args.num_cases,
            max_iterations=1,
        )
        comparator.print_and_save_results(result)

        elapsed = time.time() - start_time
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
        print(f"\n⏱️ Total time: {h}h {m}m {s}s")
    finally:
        comparator.close()
        print("✅ Run completed")


if __name__ == "__main__":
    main()
