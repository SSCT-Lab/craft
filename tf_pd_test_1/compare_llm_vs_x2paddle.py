#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM方法 vs X2Paddle方法：TensorFlow → PaddlePaddle 测试用例转换成功率对比
==========================================================================
"""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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

from tf_pd_test_1.llm_enhanced_compare import LLMEnhancedComparator, DEFAULT_MODEL, DEFAULT_KEY_PATH

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
    def __init__(self, func):
        super().__init__()
        self._func = func

    def __call__(self, *args):
        return self._func(*args)


class X2PaddleConverter:
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

    def _prepare_inputs(self, test_case: Dict[str, Any], is_class_api: bool) -> Tuple[List[tf.Tensor], Dict[str, Any], List[str], Dict[str, Any]]:
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
            if name in test_case:
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

    def _wrap_as_module(self, tf_api: str, is_class_api: bool, init_kwargs: Dict[str, Any], extra_kwargs: Dict[str, Any]) -> Optional[tf.Module]:
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

    def convert_and_run(self, tf_api: str, test_case: Dict[str, Any], is_class_api: bool) -> Dict[str, Any]:
        result = {"onnx_export_success": False, "onnx_run_success": False, "tf_success": False, "error": None}
        if tf2onnx is None:
            result["error"] = "未安装 tf2onnx"
            return result

        with self.execution_lock:
            try:
                input_tensors, init_kwargs, input_names, extra_kwargs = self._prepare_inputs(test_case, is_class_api)
                if not input_tensors:
                    result["error"] = "无有效输入张量"
                    return result
                module = self._wrap_as_module(tf_api, is_class_api, init_kwargs, extra_kwargs)
                if module is None:
                    result["error"] = "无法包装TF算子"
                    return result
                _ = module(*input_tensors)
                result["tf_success"] = True
            except Exception as error:
                result["error"] = f"TF执行失败: {error}"
                return result

            onnx_path = None
            try:
                onnx_fd, onnx_path = tempfile.mkstemp(suffix=".onnx")
                os.close(onnx_fd)
                signatures = [
                    tf.TensorSpec(shape=tensor.shape, dtype=tensor.dtype, name=input_names[idx] if idx < len(input_names) else f"input_{idx}")
                    for idx, tensor in enumerate(input_tensors)
                ]

                @tf.function(input_signature=signatures)
                def wrapped(*args):
                    return module(*args)

                tf2onnx.convert.from_function(wrapped, input_signature=signatures, opset=14, output_path=onnx_path)
                result["onnx_export_success"] = True

                session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                ort_inputs = session.get_inputs()
                feed = {ort_inputs[idx].name: tensor.numpy() for idx, tensor in enumerate(input_tensors) if idx < len(ort_inputs)}
                _ = session.run(None, feed)
                result["onnx_run_success"] = True
            except Exception as error:
                result["error"] = f"ONNX路径失败: {error}"
            finally:
                if onnx_path and os.path.exists(onnx_path):
                    os.remove(onnx_path)

        return result


class LLMvsX2PaddleComparator:
    def __init__(self, key_path: str = DEFAULT_KEY_PATH, model: str = DEFAULT_MODEL, num_workers: int = DEFAULT_WORKERS):
        self.print_lock = Lock()
        self.execution_lock = RLock()
        self.realtime_lock = Lock()
        self.num_workers = max(1, int(num_workers))

        self.llm_method = LLMEnhancedComparator(key_path=key_path, model=model, print_lock=self.print_lock, llm_workers=self.num_workers)
        self.x2paddle_converter = X2PaddleConverter(print_lock=self.print_lock)

        self.result_dir = os.path.join(ROOT_DIR, "tf_pd_test_1")
        os.makedirs(self.result_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.realtime_file_path = os.path.join(self.result_dir, f"llm_vs_x2paddle_realtime_{timestamp}.jsonl")
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

    def run_comparison(self, operator_names: List[str], num_cases: int = DEFAULT_NUM_CASES, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> Dict[str, Any]:
        _ = max_iterations
        global_stats = {
            "total_operators": len(operator_names),
            "tested_operators": 0,
            "skipped_operators_llm": 0,
            "skipped_operators_no_target": 0,
            "skipped_operators_deprecated": 0,
            "llm_total_cases": 0,
            "llm_pd_success": 0,
            "onnx_total_cases": 0,
            "onnx_export_success": 0,
            "onnx_run_success": 0,
        }
        operator_details = []

        for tf_api in operator_names:
            _, pd_api, mapping_method = self.llm_method.convert_api_name(tf_api)
            if pd_api is None:
                global_stats["skipped_operators_no_target"] += 1
                operator_details.append({"operator": tf_api, "status": "skipped_no_target", "reason": mapping_method})
                continue

            total_cases = len(self.llm_method.test_cases_data.get(tf_api, {}).get("test_cases", []))
            actual_cases = min(num_cases, total_cases) if total_cases > 0 else num_cases
            op_result = self._test_operator(tf_api, pd_api, actual_cases)
            operator_details.append(op_result)

            if op_result["status"] == "skipped_by_llm":
                global_stats["skipped_operators_llm"] += 1
            elif op_result["status"] == "skipped_deprecated":
                global_stats["skipped_operators_deprecated"] += 1
            else:
                global_stats["tested_operators"] += 1
                global_stats["llm_total_cases"] += op_result["llm_total"]
                global_stats["llm_pd_success"] += op_result["llm_pd_success"]
                global_stats["onnx_total_cases"] += op_result["onnx_total"]
                global_stats["onnx_export_success"] += op_result["onnx_export_success"]
                global_stats["onnx_run_success"] += op_result["onnx_run_success"]

        return {"global_stats": global_stats, "operator_details": operator_details}

    def _test_operator(self, tf_api: str, pd_api: str, num_cases: int) -> Dict[str, Any]:
        op_result = {
            "operator": tf_api,
            "status": "completed",
            "llm_total": 0,
            "llm_pd_success": 0,
            "onnx_total": 0,
            "onnx_export_success": 0,
            "onnx_run_success": 0,
            "case_details": [],
        }
        tf_doc, pd_doc = self.llm_method._fetch_api_docs(tf_api, pd_api)
        is_class_api = self.llm_method.is_class_based_api(tf_api)

        cases = [(idx + 1, self._build_initial_case(tf_api, idx)) for idx in range(num_cases)]
        if self.num_workers <= 1:
            case_results = [self._process_single_case(tf_api, pd_api, tc, idx, is_class_api, tf_doc, pd_doc) for idx, tc in cases]
        else:
            case_results = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self._process_single_case, tf_api, pd_api, tc, idx, is_class_api, tf_doc, pd_doc) for idx, tc in cases]
                for future in as_completed(futures):
                    case_results.append(future.result())

        case_results.sort(key=lambda item: item["case_number"])

        if any(item.get("deprecated_skip", False) for item in case_results):
            op_result["status"] = "skipped_deprecated"
            op_result["case_details"] = case_results
            return op_result

        if case_results and all(item.get("llm_skipped", False) for item in case_results):
            op_result["status"] = "skipped_by_llm"
            op_result["case_details"] = case_results
            return op_result

        for item in case_results:
            if item.get("llm_skipped", False):
                continue
            op_result["llm_total"] += item["llm_total"]
            op_result["llm_pd_success"] += item["llm_pd_success"]
            op_result["onnx_total"] += item["onnx_total"]
            op_result["onnx_export_success"] += item["onnx_export_success"]
            op_result["onnx_run_success"] += item["onnx_run_success"]

        op_result["case_details"] = case_results
        return op_result

    def _process_single_case(self, tf_api: str, pd_api: str, test_case: Dict[str, Any], case_number: int, is_class_api: bool, tf_doc: str, pd_doc: str) -> Dict[str, Any]:
        case_result = {
            "case_number": case_number,
            "llm_skipped": False,
            "deprecated_skip": False,
            "llm_total": 0,
            "llm_pd_success": 0,
            "onnx_total": 0,
            "onnx_export_success": 0,
            "onnx_run_success": 0,
            "llm_detail": None,
            "onnx_detail": None,
        }

        onnx_result = self.x2paddle_converter.convert_and_run(tf_api, test_case, is_class_api)
        case_result["onnx_total"] = 1
        if onnx_result.get("onnx_export_success"):
            case_result["onnx_export_success"] = 1
        if onnx_result.get("onnx_run_success"):
            case_result["onnx_run_success"] = 1
        case_result["onnx_detail"] = onnx_result

        tf_test_case = copy.deepcopy(test_case)
        pd_test_case = copy.deepcopy(test_case)
        pd_test_case["api"] = pd_api

        try:
            with self.execution_lock:
                exec_result = self.llm_method._execute_test_case_sequential(tf_api, pd_api, tf_test_case, pd_test_case)
        except Exception as error:
            exec_result = {
                "tf_api": tf_api,
                "paddle_api": pd_api,
                "tf_success": False,
                "paddle_success": False,
                "results_match": False,
                "status": "fatal_error",
                "tf_error": str(error),
                "paddle_error": None,
                "comparison_error": None,
            }

        tf_error = str(exec_result.get("tf_error", ""))
        if not exec_result.get("tf_success", False) and tf_error and re.search(r"deprecated|removed", tf_error, re.IGNORECASE):
            case_result["deprecated_skip"] = True
            case_result["llm_skipped"] = True
            case_result["llm_detail"] = {
                "initial_exec": exec_result,
                "llm_operation": "skip",
                "llm_reason": f"检测到版本淘汰信息: {tf_error[:120]}",
            }
            return case_result

        llm_result = self.llm_method.call_llm_for_repair_or_mutation(exec_result, tf_test_case, pd_test_case, tf_doc, pd_doc)
        operation = llm_result.get("operation", "skip")
        if operation == "skip":
            case_result["llm_skipped"] = True
            case_result["llm_detail"] = {
                "initial_exec": exec_result,
                "llm_operation": operation,
                "llm_reason": llm_result.get("reason", ""),
            }
            self._append_realtime_record({
                "timestamp": datetime.now().isoformat(),
                "operator": tf_api,
                "case_number": case_number,
                "status": "llm_skip",
                "onnx_run_success": bool(case_result["onnx_run_success"]),
                "llm_pd_success": False,
                "llm_skipped": True,
                "deprecated_skip": False,
            })
            return case_result

        llm_tf_case = llm_result.get("tensorflow_test_case", tf_test_case)
        llm_pd_case = llm_result.get("paddle_test_case", pd_test_case)
        try:
            llm_tf_case, llm_pd_case = self.llm_method._convert_llm_test_cases(llm_tf_case, llm_pd_case)
        except Exception as error:
            case_result["llm_skipped"] = True
            case_result["llm_detail"] = {
                "initial_exec": exec_result,
                "llm_operation": "skip",
                "llm_reason": f"LLM用例转换失败: {error}",
            }
            return case_result

        case_result["llm_total"] = 1
        try:
            with self.execution_lock:
                llm_exec = self.llm_method._execute_test_case_sequential(tf_api, pd_api, llm_tf_case, llm_pd_case)
        except Exception as error:
            llm_exec = {
                "tf_success": False,
                "paddle_success": False,
                "results_match": False,
                "status": "fatal_error",
                "error": str(error),
            }
        if llm_exec.get("paddle_success"):
            case_result["llm_pd_success"] = 1

        case_result["llm_detail"] = {
            "initial_exec": exec_result,
            "llm_operation": operation,
            "llm_reason": llm_result.get("reason", ""),
            "llm_exec": llm_exec,
        }

        self._append_realtime_record({
            "timestamp": datetime.now().isoformat(),
            "operator": tf_api,
            "case_number": case_number,
            "status": "completed",
            "onnx_run_success": bool(case_result["onnx_run_success"]),
            "llm_pd_success": bool(case_result["llm_pd_success"]),
            "llm_skipped": False,
            "deprecated_skip": False,
        })
        return case_result

    def print_and_save_results(self, comparison_result: Dict[str, Any]):
        gs = comparison_result["global_stats"]

        print("\n" + "=" * 80)
        print("📊 LLM方法 vs X2Paddle方法 — TF→PD 测试用例转换成功率对比")
        print("=" * 80)

        print(f"\n📌 算子总数: {gs['total_operators']}")
        print(f"   - 无目标映射跳过: {gs['skipped_operators_no_target']}")
        print(f"   - LLM选择跳过: {gs['skipped_operators_llm']}")
        print(f"   - 已被版本淘汰跳过: {gs['skipped_operators_deprecated']}")
        print(f"   - 实际参与对比: {gs['tested_operators']}")

        print(f"\n{'─' * 40}")
        print("🤖 LLM 方法（剔除跳过的算子）:")
        print(f"   LLM生成的PD测试用例总数: {gs['llm_total_cases']}")
        print(f"   PD执行成功数: {gs['llm_pd_success']}")
        if gs["llm_total_cases"] > 0:
            llm_rate = gs["llm_pd_success"] / gs["llm_total_cases"] * 100
            print(f"   ✅ PD执行成功率: {llm_rate:.2f}%")
        else:
            llm_rate = 0.0
            print("   ✅ PD执行成功率: N/A（无LLM生成用例）")

        print(f"\n{'─' * 40}")
        print("🔄 X2Paddle 方法（剔除LLM跳过的算子）:")
        print(f"   X2Paddle转换尝试总数: {gs['onnx_total_cases']}")
        print(f"   X2Paddle导出成功数: {gs['onnx_export_success']}")
        print(f"   X2Paddle推理成功数（=转换成功）: {gs['onnx_run_success']}")
        if gs["onnx_total_cases"] > 0:
            x2paddle_rate = gs["onnx_run_success"] / gs["onnx_total_cases"] * 100
            print(f"   ✅ X2Paddle转换成功率: {x2paddle_rate:.2f}%")
        else:
            x2paddle_rate = 0.0
            print("   ✅ X2Paddle转换成功率: N/A")

        print(f"\n{'─' * 40}")
        print("📈 对比结论:")
        if gs["llm_total_cases"] > 0 and gs["onnx_total_cases"] > 0:
            diff = llm_rate - x2paddle_rate
            if diff > 0:
                print(f"   LLM 方法高于 X2Paddle 方法 {diff:.2f} 个百分点")
            elif diff < 0:
                print(f"   X2Paddle 方法高于 LLM 方法 {abs(diff):.2f} 个百分点")
            else:
                print("   两者持平")
        else:
            print("   数据不足，无法比较")
        print("=" * 80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.result_dir, f"llm_vs_x2paddle_result_{timestamp}.json")
        save_data = self._make_serializable(comparison_result)
        save_data["summary"] = {
            "llm_pd_success_rate": f"{llm_rate:.2f}%" if gs["llm_total_cases"] > 0 else "N/A",
            "x2paddle_success_rate": f"{x2paddle_rate:.2f}%" if gs["onnx_total_cases"] > 0 else "N/A",
            "skipped_operators_deprecated": gs.get("skipped_operators_deprecated", 0),
            "timestamp": datetime.now().isoformat(),
        }

        with open(result_file, "w", encoding="utf-8") as file:
            json.dump(save_data, file, indent=2, ensure_ascii=False)

        print(f"\n💾 详细结果已保存到: {result_file}")

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(key): LLMvsX2PaddleComparator._make_serializable(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [LLMvsX2PaddleComparator._make_serializable(item) for item in obj]
        if isinstance(obj, tuple):
            return [LLMvsX2PaddleComparator._make_serializable(item) for item in obj]
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
    parser = argparse.ArgumentParser(description="LLM方法 vs X2Paddle方法：TF->PD测试用例转换成功率对比")
    parser.add_argument("--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES, help=f"每个算子测试用例数（默认{DEFAULT_NUM_CASES}）")
    parser.add_argument("--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS, help="LLM迭代次数（固定按1执行）")
    parser.add_argument("--start", type=int, default=1, help="起始算子索引（从1开始）")
    parser.add_argument("--end", type=int, default=None, help="结束算子索引（包含）")
    parser.add_argument("--operators", "-o", nargs="*", help="指定算子名称列表")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS, help=f"LLM并发线程数（默认{DEFAULT_WORKERS}）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM模型（默认{DEFAULT_MODEL}）")
    parser.add_argument("--key-path", "-k", default=DEFAULT_KEY_PATH, help=f"API key路径（默认{DEFAULT_KEY_PATH}）")
    args = parser.parse_args()

    print("=" * 80)
    print("LLM方法 vs X2Paddle方法 — TensorFlow→PaddlePaddle 测试用例转换成功率对比")
    print("=" * 80)
    print(f"📌 每个算子用例数: {args.num_cases}")
    print("📌 LLM迭代次数: 1（固定）")
    print(f"📌 并发线程数: {args.workers}")
    print(f"📌 LLM模型: {args.model}")
    print("=" * 80)

    comparator = LLMvsX2PaddleComparator(key_path=args.key_path, model=args.model, num_workers=args.workers)
    start_time = time.time()

    try:
        all_ops = sorted(list(comparator.llm_method.test_cases_data.keys()))
        print(f"\n📋 测试集共 {len(all_ops)} 个TF算子")

        if args.operators:
            operator_names = args.operators
        else:
            start_idx = max(1, args.start) - 1
            end_idx = args.end if args.end is not None else len(all_ops)
            end_idx = min(end_idx, len(all_ops))
            operator_names = all_ops[start_idx:end_idx]
            print(f"📌 测试范围: 第 {start_idx + 1} ~ {end_idx} 个算子")

        print(f"📋 将测试 {len(operator_names)} 个算子")
        print(f"📋 前10个: {', '.join(operator_names[:10])}{'...' if len(operator_names) > 10 else ''}\n")

        result = comparator.run_comparison(operator_names, num_cases=args.num_cases, max_iterations=1)
        comparator.print_and_save_results(result)

        elapsed = time.time() - start_time
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
        print(f"\n⏱️ 总耗时: {h}h {m}m {s}s")
    finally:
        comparator.close()
        print("✅ 程序执行完成")


if __name__ == "__main__":
    main()
