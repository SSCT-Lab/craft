#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM method vs rule-based method: PyTorch -> PaddlePaddle test case conversion success rate
===============================================================================

Compare two cross-framework test-case migration strategies:
1. LLM method: run initial case -> call LLM to repair/mutate (fixed 1 iteration) -> run new PD case
2. Rule-based: only convert operator name and dtype -> run new PD case

Core metric: whether the newly generated PD case runs successfully (paddle_success).

Outputs:
- Realtime results: JSONL (one case per line)
- Summary results: JSON (global stats + per-operator details)
"""

import argparse
import copy
import json
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock, RLock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import paddle

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pt_pd_test.llm_enhanced_compare import LLMEnhancedComparator

DEFAULT_MAX_ITERATIONS = 1
DEFAULT_NUM_CASES = 3
DEFAULT_WORKERS = 4


def safe_print(msg: str, print_lock: Optional[Lock] = None, end: str = "\n") -> None:
    if print_lock:
        with print_lock:
            print(msg, end=end, flush=True)
    else:
        print(msg, end=end, flush=True)


class RuleBasedConverter:
    def __init__(self) -> None:
        self.pd_dtype_map = {
            "torch.float16": paddle.float16,
            "torch.float32": paddle.float32,
            "torch.float64": paddle.float64,
            "torch.bfloat16": paddle.bfloat16,
            "torch.int8": paddle.int8,
            "torch.uint8": paddle.uint8,
            "torch.int16": paddle.int16,
            "torch.int32": paddle.int32,
            "torch.int64": paddle.int64,
            "torch.bool": paddle.bool,
            "paddle.float16": paddle.float16,
            "paddle.float32": paddle.float32,
            "paddle.float64": paddle.float64,
            "paddle.bfloat16": paddle.bfloat16,
            "paddle.int8": paddle.int8,
            "paddle.uint8": paddle.uint8,
            "paddle.int16": paddle.int16,
            "paddle.int32": paddle.int32,
            "paddle.int64": paddle.int64,
            "paddle.bool": paddle.bool,
            "float16": paddle.float16,
            "float32": paddle.float32,
            "float64": paddle.float64,
            "bfloat16": paddle.bfloat16,
            "int8": paddle.int8,
            "uint8": paddle.uint8,
            "int16": paddle.int16,
            "int32": paddle.int32,
            "int64": paddle.int64,
            "bool": paddle.bool,
            "double": paddle.float64,
            "half": paddle.float16,
            "long": paddle.int64,
            "int": paddle.int32,
            "float": paddle.float32,
        }

    @staticmethod
    def _is_tensor_desc(value: Any) -> bool:
        return isinstance(value, dict) and "shape" in value and "dtype" in value

    @staticmethod
    def _normalize_tensor_desc_dtype(dtype_value: Any) -> Any:
        if not isinstance(dtype_value, str):
            return dtype_value

        normalized = dtype_value.strip()
        for prefix in ["torch.", "paddle.", "np.", "numpy.", "tf.", "mindspore.", "ms."]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]

        aliases = {
            "double": "float64",
            "half": "float16",
            "long": "int64",
            "int": "int32",
            "float": "float32",
        }
        return aliases.get(normalized, normalized)

    def _convert_dtype_param_value(self, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized in self.pd_dtype_map:
                return self.pd_dtype_map[normalized]

            for prefix in ["torch.", "paddle.", "np.", "numpy.", "tf.", "mindspore.", "ms."]:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
            return self.pd_dtype_map.get(normalized, paddle.float32)

        if isinstance(value, int):
            int_map = {
                0: paddle.float32,
                1: paddle.float64,
                2: paddle.int32,
                3: paddle.uint8,
                4: paddle.int16,
                5: paddle.int8,
                6: paddle.int64,
                7: paddle.bool,
            }
            return int_map.get(value, paddle.float32)

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

    def convert_torch_case_to_pd_case(self, torch_test_case: Dict[str, Any], pd_api: str) -> Dict[str, Any]:
        converted = self._convert_recursive(copy.deepcopy(torch_test_case))
        converted["api"] = pd_api
        return converted


class LLMvsRuleBasedComparator:
    def __init__(self, num_workers: int = DEFAULT_WORKERS) -> None:
        self.print_lock = Lock()
        self.realtime_lock = Lock()
        self.execution_lock = RLock()

        self.num_workers = max(1, int(num_workers))

        self.llm_method = LLMEnhancedComparator()
        self.rule_converter = RuleBasedConverter()

        self.result_dir = os.path.join(ROOT_DIR, "pt_pd_test")
        os.makedirs(self.result_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.realtime_file_path = os.path.join(self.result_dir, f"llm_vs_rulebased_realtime_{timestamp}.jsonl")
        self.realtime_file = open(self.realtime_file_path, "a", encoding="utf-8")

    def _safe_print(self, msg: str, end: str = "\n") -> None:
        safe_print(msg, self.print_lock, end)

    def _append_realtime_record(self, record: Dict[str, Any]) -> None:
        with self.realtime_lock:
            self.realtime_file.write(json.dumps(self._make_serializable(record), ensure_ascii=False) + "\n")
            self.realtime_file.flush()

    def run_comparison(
        self,
        operator_names: List[str],
        num_cases: int = DEFAULT_NUM_CASES,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> Dict[str, Any]:
        if max_iterations != 1:
            self._safe_print("This ablation keeps max_iterations=1 and forces it to 1")
        max_iterations = 1

        global_stats = {
            "total_operators": len(operator_names),
            "tested_operators": 0,
            "skipped_operators_no_pd": 0,
            "skipped_operators_all_llm_skip": 0,
            "skipped_operators_deprecated": 0,
            "llm_generated_total": 0,
            "llm_pd_success": 0,
            "rule_generated_total": 0,
            "rule_pd_success": 0,
        }

        operator_details: List[Dict[str, Any]] = []

        for index, operator_name in enumerate(operator_names, 1):
            self._safe_print("\n" + "=" * 72)
            self._safe_print(f"[{index}/{len(operator_names)}] Operator: {operator_name}")
            self._safe_print("=" * 72)

            torch_api, pd_api, mapping_method = self.llm_method.convert_api_name(operator_name)
            if pd_api is None:
                self._safe_print(f"  No PD mapping ({mapping_method}), skipping")
                global_stats["skipped_operators_no_pd"] += 1
                operator_details.append(
                    {
                        "operator": operator_name,
                        "status": "skipped_no_pd",
                        "mapping_method": mapping_method,
                    }
                )
                continue

            self._safe_print(f"  PT: {torch_api} -> PD: {pd_api}")

            document = self.llm_method.collection.find_one({"api": operator_name})
            if document is None:
                self._safe_print("  Operator not found in database, skipping")
                operator_details.append({"operator": operator_name, "status": "not_found"})
                continue

            total_cases = self.llm_method.get_num_test_cases_from_document(document)
            actual_cases = min(max(1, num_cases), total_cases)

            self._safe_print("  Prefetching API docs (for LLM)...")
            torch_doc, pd_doc = self.llm_method._fetch_api_docs(torch_api, pd_api)

            operator_result = self._test_single_operator(
                operator_name=operator_name,
                torch_api=torch_api,
                pd_api=pd_api,
                document=document,
                num_cases=actual_cases,
                torch_doc=torch_doc,
                pd_doc=pd_doc,
            )
            operator_details.append(operator_result)

            if operator_result["status"] == "skipped_deprecated":
                global_stats["skipped_operators_deprecated"] += 1
                continue
            if operator_result["status"] == "skipped_all_llm_skip":
                global_stats["skipped_operators_all_llm_skip"] += 1
                continue

            global_stats["tested_operators"] += 1
            global_stats["llm_generated_total"] += operator_result["llm_generated_total"]
            global_stats["llm_pd_success"] += operator_result["llm_pd_success"]
            global_stats["rule_generated_total"] += operator_result["rule_generated_total"]
            global_stats["rule_pd_success"] += operator_result["rule_pd_success"]

        return {"global_stats": global_stats, "operator_details": operator_details}

    def _test_single_operator(
        self,
        operator_name: str,
        torch_api: str,
        pd_api: str,
        document: Dict[str, Any],
        num_cases: int,
        torch_doc: str,
        pd_doc: str,
    ) -> Dict[str, Any]:
        op_result: Dict[str, Any] = {
            "operator": operator_name,
            "torch_api": torch_api,
            "pd_api": pd_api,
            "num_cases": num_cases,
            "status": "completed",
            "llm_generated_total": 0,
            "llm_pd_success": 0,
            "rule_generated_total": 0,
            "rule_pd_success": 0,
            "case_details": [],
        }

        initial_cases: List[Tuple[int, Dict[str, Any]]] = []
        for case_idx in range(num_cases):
            tc = self.llm_method.prepare_shared_numpy_data(document, case_index=case_idx)
            tc["api"] = torch_api
            initial_cases.append((case_idx + 1, tc))

        if self.num_workers <= 1:
            case_results = [
                self._process_single_case(torch_api, pd_api, case_number, tc, torch_doc, pd_doc)
                for case_number, tc in initial_cases
            ]
        else:
            case_results = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_single_case,
                        torch_api,
                        pd_api,
                        case_number,
                        tc,
                        torch_doc,
                        pd_doc,
                    ): case_number
                    for case_number, tc in initial_cases
                }
                for future in as_completed(futures):
                    case_results.append(future.result())

        case_results.sort(key=lambda item: item["case_number"])

        if any(item.get("deprecated_skip", False) for item in case_results):
            op_result["status"] = "skipped_deprecated"
            op_result["case_details"] = case_results
            self._safe_print("  Detected deprecated operator, skipping")
            return op_result

        if case_results and all(item.get("llm_skipped", False) for item in case_results):
            op_result["status"] = "skipped_all_llm_skip"
            op_result["case_details"] = case_results
            self._safe_print("  All cases skipped by LLM for this operator, excluded")
            return op_result

        for item in case_results:
            if item.get("llm_skipped", False):
                continue
            op_result["llm_generated_total"] += item.get("llm_generated_total", 0)
            op_result["llm_pd_success"] += item.get("llm_pd_success", 0)
            op_result["rule_generated_total"] += item.get("rule_generated_total", 0)
            op_result["rule_pd_success"] += item.get("rule_pd_success", 0)

        op_result["case_details"] = case_results
        return op_result

    def _process_single_case(
        self,
        torch_api: str,
        pd_api: str,
        case_number: int,
        initial_torch_case: Dict[str, Any],
        torch_doc: str,
        pd_doc: str,
    ) -> Dict[str, Any]:
        case_result: Dict[str, Any] = {
            "case_number": case_number,
            "llm_skipped": False,
            "deprecated_skip": False,
            "llm_generated_total": 0,
            "llm_pd_success": 0,
            "rule_generated_total": 1,
            "rule_pd_success": 0,
            "rule_detail": {},
            "llm_detail": {},
        }

        rule_pd_case = self.rule_converter.convert_torch_case_to_pd_case(initial_torch_case, pd_api)

        self._safe_print(f"  [Case {case_number}] Rule-based execution...", end="")
        try:
            with self.execution_lock:
                rule_exec = self.llm_method.execute_test_case(
                    torch_api,
                    pd_api,
                    initial_torch_case,
                    rule_pd_case,
                )
            rule_pd_ok = bool(rule_exec.get("paddle_success", False))
            case_result["rule_pd_success"] = 1 if rule_pd_ok else 0
            self._safe_print(f" PD:{'✓' if rule_pd_ok else '✗'}")
        except Exception as error:
            rule_exec = {
                "status": "fatal_error",
                "paddle_success": False,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            self._safe_print(f" PD:✗ ({str(error)[:80]})")

        case_result["rule_detail"] = {
            "generated_pd_test_case": rule_pd_case,
            "execution_result": rule_exec,
        }

        initial_pd_case = copy.deepcopy(initial_torch_case)
        initial_pd_case["api"] = pd_api

        self._safe_print(f"  [Case {case_number}] LLM initial execution...", end="")
        try:
            with self.execution_lock:
                initial_exec = self.llm_method.execute_test_case(
                    torch_api,
                    pd_api,
                    initial_torch_case,
                    initial_pd_case,
                )
            self._safe_print(
                f" PT:{'✓' if initial_exec.get('torch_success') else '✗'}"
                f" PD:{'✓' if initial_exec.get('paddle_success') else '✗'}"
            )
        except Exception as error:
            initial_exec = {
                "torch_api": torch_api,
                "paddle_api": pd_api,
                "status": "fatal_error",
                "torch_success": False,
                "paddle_success": False,
                "results_match": False,
                "torch_error": str(error),
                "paddle_error": None,
                "comparison_error": None,
            }
            self._safe_print(f" PT:✗ PD:✗ ({str(error)[:80]})")

        torch_error = str(initial_exec.get("torch_error", ""))
        if (not initial_exec.get("torch_success", False)) and torch_error:
            if any(keyword in torch_error.lower() for keyword in ["deprecated", "removed"]):
                case_result["deprecated_skip"] = True
                case_result["llm_detail"] = {
                    "initial_exec": initial_exec,
                    "llm_operation": "skip",
                    "llm_reason": "PyTorch operator is deprecated",
                }
                self._append_realtime_record(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "operator": torch_api,
                        "case_number": case_number,
                        "status": "skipped_deprecated",
                        "llm_skipped": False,
                        "deprecated_skip": True,
                        "llm_pd_success": False,
                        "rule_pd_success": bool(case_result["rule_pd_success"]),
                    }
                )
                return case_result

        self._safe_print(f"  [Case {case_number}] Calling LLM...", end="")
        llm_result = self.llm_method.call_llm_for_repair_or_mutation(
            initial_exec,
            initial_torch_case,
            initial_pd_case,
            torch_doc,
            pd_doc,
        )
        operation = llm_result.get("operation", "skip")
        reason = str(llm_result.get("reason", ""))
        self._safe_print(f" {operation} - {reason[:70]}")

        if operation == "skip":
            case_result["llm_skipped"] = True
            case_result["llm_detail"] = {
                "initial_exec": initial_exec,
                "llm_operation": operation,
                "llm_reason": reason,
            }
            self._append_realtime_record(
                {
                    "timestamp": datetime.now().isoformat(),
                    "operator": torch_api,
                    "case_number": case_number,
                    "status": "llm_skip",
                    "llm_skipped": True,
                    "deprecated_skip": False,
                    "llm_pd_success": False,
                    "rule_pd_success": bool(case_result["rule_pd_success"]),
                }
            )
            return case_result

        llm_pt_case = llm_result.get("pytorch_test_case", initial_torch_case)
        llm_pd_case = llm_result.get("paddle_test_case", initial_pd_case)

        try:
            llm_pt_case, llm_pd_case = self.llm_method._convert_llm_test_cases(llm_pt_case, llm_pd_case)
        except Exception as error:
            case_result["llm_skipped"] = True
            case_result["llm_detail"] = {
                "initial_exec": initial_exec,
                "llm_operation": "skip",
                "llm_reason": f"LLM case conversion failed: {error}",
            }
            self._safe_print(f"  [Case {case_number}] LLM case conversion failed, skipping: {str(error)[:70]}")
            self._append_realtime_record(
                {
                    "timestamp": datetime.now().isoformat(),
                    "operator": torch_api,
                    "case_number": case_number,
                    "status": "llm_case_convert_failed",
                    "llm_skipped": True,
                    "deprecated_skip": False,
                    "llm_pd_success": False,
                    "rule_pd_success": bool(case_result["rule_pd_success"]),
                }
            )
            return case_result

        case_result["llm_generated_total"] = 1

        self._safe_print(f"  [Case {case_number}] Executing LLM-generated case...", end="")
        try:
            with self.execution_lock:
                llm_exec = self.llm_method.execute_test_case(
                    torch_api,
                    pd_api,
                    llm_pt_case,
                    llm_pd_case,
                )
            llm_pd_ok = bool(llm_exec.get("paddle_success", False))
            case_result["llm_pd_success"] = 1 if llm_pd_ok else 0
            self._safe_print(
                f" PT:{'✓' if llm_exec.get('torch_success') else '✗'}"
                f" PD:{'✓' if llm_pd_ok else '✗'}"
            )
        except Exception as error:
            llm_exec = {
                "status": "fatal_error",
                "torch_success": False,
                "paddle_success": False,
                "results_match": False,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            self._safe_print(f" PT:✗ PD:✗ ({str(error)[:80]})")

        case_result["llm_detail"] = {
            "initial_exec": initial_exec,
            "llm_operation": operation,
            "llm_reason": reason,
            "generated_pt_test_case": llm_pt_case,
            "generated_pd_test_case": llm_pd_case,
            "generated_exec": llm_exec,
        }

        self._append_realtime_record(
            {
                "timestamp": datetime.now().isoformat(),
                "operator": torch_api,
                "case_number": case_number,
                "status": "completed",
                "llm_skipped": False,
                "deprecated_skip": False,
                "llm_pd_success": bool(case_result["llm_pd_success"]),
                "rule_pd_success": bool(case_result["rule_pd_success"]),
                "llm_operation": operation,
            }
        )

        return case_result

    def print_and_save_results(self, result: Dict[str, Any]) -> str:
        stats = result["global_stats"]

        print("\n" + "=" * 80)
        print("LLM method vs rule-based method (name + dtype auto-conversion only)")
        print("=" * 80)
        print(f"Total operators: {stats['total_operators']}")
        print(f"- Skipped (no PD mapping): {stats['skipped_operators_no_pd']}")
        print(f"- Skipped (all LLM skipped): {stats['skipped_operators_all_llm_skip']}")
        print(f"- Skipped (deprecated): {stats['skipped_operators_deprecated']}")
        print(f"- Actually compared: {stats['tested_operators']}")

        llm_total = stats["llm_generated_total"]
        llm_success = stats["llm_pd_success"]
        llm_rate = (llm_success / llm_total * 100.0) if llm_total > 0 else 0.0

        rule_total = stats["rule_generated_total"]
        rule_success = stats["rule_pd_success"]
        rule_rate = (rule_success / rule_total * 100.0) if rule_total > 0 else 0.0

        print("\n" + "-" * 48)
        print("LLM method (new PD cases generated)")
        print(f"- Generated cases: {llm_total}")
        print(f"- PD successful runs: {llm_success}")
        print(f"- PD success rate: {llm_rate:.2f}%" if llm_total > 0 else "- PD success rate: N/A")

        print("\n" + "-" * 48)
        print("Rule-based method (new PD cases generated)")
        print(f"- Generated cases: {rule_total}")
        print(f"- PD successful runs: {rule_success}")
        print(f"- PD success rate: {rule_rate:.2f}%" if rule_total > 0 else "- PD success rate: N/A")

        print("\n" + "-" * 48)
        if llm_total > 0 and rule_total > 0:
            diff = llm_rate - rule_rate
            if diff > 0:
                print(f"Conclusion: LLM method is higher than rule-based by {diff:.2f} percentage points")
            elif diff < 0:
                print(f"Conclusion: Rule-based method is higher than LLM by {-diff:.2f} percentage points")
            else:
                print("Conclusion: Both methods are tied")
        else:
            print("Conclusion: Insufficient samples to compare")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.result_dir, f"llm_vs_rulebased_result_{timestamp}.json")

        result_to_save = self._make_serializable(result)
        result_to_save["summary"] = {
            "llm_pd_success_rate": f"{llm_rate:.2f}%" if llm_total > 0 else "N/A",
            "rule_pd_success_rate": f"{rule_rate:.2f}%" if rule_total > 0 else "N/A",
            "timestamp": datetime.now().isoformat(),
        }

        self._atomic_dump_json(result_file, result_to_save)

        print("=" * 80)
        print(f"Realtime JSONL: {self.realtime_file_path}")
        print(f"Summary JSON: {result_file}")

        return result_file

    @staticmethod
    def _atomic_dump_json(file_path: str, data: Dict[str, Any]) -> None:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            prefix="tmp_result_",
            dir=file_dir,
            delete=False,
        ) as temp_file:
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
            return {
                "__type__": "ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, torch.Tensor):
            return {
                "__type__": "tensor",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            }
        if isinstance(obj, paddle.Tensor):
            return {
                "__type__": "pd_tensor",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            }
        return obj

    def close(self) -> None:
        try:
            if self.realtime_file:
                self.realtime_file.flush()
                self.realtime_file.close()
        finally:
            self.llm_method.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM vs rule-based: PyTorch->Paddle new-case execution success rate")
    parser.add_argument("--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES, help=f"Number of cases per operator (default {DEFAULT_NUM_CASES})")
    parser.add_argument("--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS, help="LLM iterations (fixed to 1 in this script)")
    parser.add_argument("--start", type=int, default=1, help="Start operator index (1-based)")
    parser.add_argument("--end", type=int, default=None, help="End operator index (inclusive)")
    parser.add_argument("--operators", "-o", nargs="*", help="Specify operator list")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS, help=f"Worker threads (default {DEFAULT_WORKERS})")
    args = parser.parse_args()

    print("=" * 80)
    print("LLM method vs rule-based method (name + dtype auto-conversion only)")
    print("=" * 80)
    print(f"Cases per operator: {args.num_cases}")
    print("LLM iterations: 1 (fixed)")
    print(f"Worker threads: {args.workers}")
    print("=" * 80)

    comparator = LLMvsRuleBasedComparator(num_workers=args.workers)

    start_time = time.time()

    try:
        try:
            all_docs = list(comparator.llm_method.collection.find({}, {"api": 1}))
        except Exception as error:
            print(f"Unable to connect to MongoDB or read operator list: {error}")
            print("Please ensure MongoDB is running (default mongodb://localhost:27017/) or check network/permissions.")
            return

        all_operators = [doc["api"] for doc in all_docs if "api" in doc]

        print(f"\nTotal operators in database: {len(all_operators)}")

        if args.operators:
            operator_names = args.operators
        else:
            start_idx = max(1, args.start) - 1
            end_idx = args.end if args.end is not None else len(all_operators)
            end_idx = min(end_idx, len(all_operators))
            if start_idx >= end_idx:
                raise ValueError(f"Start index {args.start} must be less than end index {end_idx}")
            operator_names = all_operators[start_idx:end_idx]
            print(f"Test range: operator {start_idx + 1} to {end_idx}")

        print(f"Operators actually tested: {len(operator_names)}")
        preview = ", ".join(operator_names[:10])
        print(f"Top 10 operators: {preview}{'...' if len(operator_names) > 10 else ''}\n")

        result = comparator.run_comparison(
            operator_names=operator_names,
            num_cases=args.num_cases,
            max_iterations=args.max_iterations,
        )

        comparator.print_and_save_results(result)

        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        print(f"\nTotal elapsed: {hours}h {minutes}m {seconds}s")

    finally:
        comparator.close()
        print("Program completed")


if __name__ == "__main__":
    main()
