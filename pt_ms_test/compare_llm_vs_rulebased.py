#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM方法 vs 规则方法：PyTorch -> MindSpore 测试用例转换成功率对比
===================================================================

对比两种跨框架测试用例迁移方案：
1. LLM方法：执行初始用例 -> 调用LLM修复/变异（迭代固定1次） -> 执行LLM生成的新MS用例
2. 规则方法：仅自动转换算子名与数据类型（dtype） -> 执行规则生成的新MS用例

核心比较指标：两种方法“生成的新MS用例”能否正常执行（mindspore_success）。

输出：
- 实时结果：JSONL（每个case一行）
- 汇总结果：JSON（全局统计 + 算子级明细）
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

import mindspore
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pt_ms_test.llm_enhanced_compare import LLMEnhancedComparator

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
        self.ms_dtype_map = {
            "torch.float16": mindspore.float16,
            "torch.float32": mindspore.float32,
            "torch.float64": mindspore.float64,
            "torch.bfloat16": mindspore.bfloat16,
            "torch.int8": mindspore.int8,
            "torch.uint8": mindspore.uint8,
            "torch.int16": mindspore.int16,
            "torch.int32": mindspore.int32,
            "torch.int64": mindspore.int64,
            "torch.bool": mindspore.bool_,
            "mindspore.float16": mindspore.float16,
            "mindspore.float32": mindspore.float32,
            "mindspore.float64": mindspore.float64,
            "mindspore.bfloat16": mindspore.bfloat16,
            "mindspore.int8": mindspore.int8,
            "mindspore.uint8": mindspore.uint8,
            "mindspore.int16": mindspore.int16,
            "mindspore.int32": mindspore.int32,
            "mindspore.int64": mindspore.int64,
            "mindspore.bool_": mindspore.bool_,
            "ms.float16": mindspore.float16,
            "ms.float32": mindspore.float32,
            "ms.float64": mindspore.float64,
            "ms.bfloat16": mindspore.bfloat16,
            "ms.int8": mindspore.int8,
            "ms.uint8": mindspore.uint8,
            "ms.int16": mindspore.int16,
            "ms.int32": mindspore.int32,
            "ms.int64": mindspore.int64,
            "ms.bool_": mindspore.bool_,
            "float16": mindspore.float16,
            "float32": mindspore.float32,
            "float64": mindspore.float64,
            "bfloat16": mindspore.bfloat16,
            "int8": mindspore.int8,
            "uint8": mindspore.uint8,
            "int16": mindspore.int16,
            "int32": mindspore.int32,
            "int64": mindspore.int64,
            "bool": mindspore.bool_,
            "bool_": mindspore.bool_,
            "double": mindspore.float64,
            "half": mindspore.float16,
            "long": mindspore.int64,
            "int": mindspore.int32,
            "float": mindspore.float32,
        }

    @staticmethod
    def _is_tensor_desc(value: Any) -> bool:
        return isinstance(value, dict) and "shape" in value and "dtype" in value

    @staticmethod
    def _normalize_tensor_desc_dtype(dtype_value: Any) -> Any:
        if not isinstance(dtype_value, str):
            return dtype_value

        normalized = dtype_value.strip()
        for prefix in ["torch.", "mindspore.", "ms.", "np.", "numpy.", "tf.", "paddle."]:
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
            if normalized in self.ms_dtype_map:
                return self.ms_dtype_map[normalized]

            for prefix in ["torch.", "mindspore.", "ms.", "np.", "numpy.", "tf.", "paddle."]:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
            return self.ms_dtype_map.get(normalized, mindspore.float32)

        if isinstance(value, int):
            int_map = {
                0: mindspore.float32,
                1: mindspore.float64,
                2: mindspore.int32,
                3: mindspore.uint8,
                4: mindspore.int16,
                5: mindspore.int8,
                6: mindspore.int64,
                7: mindspore.bool_,
            }
            return int_map.get(value, mindspore.float32)

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

    def convert_torch_case_to_ms_case(self, torch_test_case: Dict[str, Any], ms_api: str) -> Dict[str, Any]:
        converted = self._convert_recursive(copy.deepcopy(torch_test_case))
        converted["api"] = ms_api
        return converted


class LLMvsRuleBasedComparator:
    def __init__(self, num_workers: int = DEFAULT_WORKERS) -> None:
        self.print_lock = Lock()
        self.realtime_lock = Lock()
        self.execution_lock = RLock()

        self.num_workers = max(1, int(num_workers))

        self.llm_method = LLMEnhancedComparator()
        self.rule_converter = RuleBasedConverter()

        self.result_dir = os.path.join(ROOT_DIR, "pt_ms_test")
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
            self._safe_print("当前消融实验固定max_iterations=1，已强制使用1")
        max_iterations = 1

        global_stats = {
            "total_operators": len(operator_names),
            "tested_operators": 0,
            "skipped_operators_no_ms": 0,
            "skipped_operators_all_llm_skip": 0,
            "skipped_operators_deprecated": 0,
            "llm_generated_total": 0,
            "llm_ms_success": 0,
            "rule_generated_total": 0,
            "rule_ms_success": 0,
        }

        operator_details: List[Dict[str, Any]] = []

        for index, operator_name in enumerate(operator_names, 1):
            self._safe_print("\n" + "=" * 72)
            self._safe_print(f"[{index}/{len(operator_names)}] 算子: {operator_name}")
            self._safe_print("=" * 72)

            torch_api, ms_api, mapping_method = self.llm_method.convert_api_name(operator_name)
            if ms_api is None:
                self._safe_print(f"  无MS映射（{mapping_method}），跳过")
                global_stats["skipped_operators_no_ms"] += 1
                operator_details.append(
                    {
                        "operator": operator_name,
                        "status": "skipped_no_ms",
                        "mapping_method": mapping_method,
                    }
                )
                continue

            self._safe_print(f"  PT: {torch_api} -> MS: {ms_api}")

            document = self.llm_method.collection.find_one({"api": operator_name})
            if document is None:
                self._safe_print("  数据库中未找到该算子，跳过")
                operator_details.append({"operator": operator_name, "status": "not_found"})
                continue

            total_cases = self.llm_method.get_num_test_cases_from_document(document)
            actual_cases = min(max(1, num_cases), total_cases)

            self._safe_print("  预取API文档（供LLM使用）...")
            torch_doc, ms_doc = self.llm_method._fetch_api_docs(torch_api, ms_api)

            operator_result = self._test_single_operator(
                operator_name=operator_name,
                torch_api=torch_api,
                ms_api=ms_api,
                document=document,
                num_cases=actual_cases,
                torch_doc=torch_doc,
                ms_doc=ms_doc,
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
            global_stats["llm_ms_success"] += operator_result["llm_ms_success"]
            global_stats["rule_generated_total"] += operator_result["rule_generated_total"]
            global_stats["rule_ms_success"] += operator_result["rule_ms_success"]

        return {"global_stats": global_stats, "operator_details": operator_details}

    def _test_single_operator(
        self,
        operator_name: str,
        torch_api: str,
        ms_api: str,
        document: Dict[str, Any],
        num_cases: int,
        torch_doc: str,
        ms_doc: str,
    ) -> Dict[str, Any]:
        op_result: Dict[str, Any] = {
            "operator": operator_name,
            "torch_api": torch_api,
            "ms_api": ms_api,
            "num_cases": num_cases,
            "status": "completed",
            "llm_generated_total": 0,
            "llm_ms_success": 0,
            "rule_generated_total": 0,
            "rule_ms_success": 0,
            "case_details": [],
        }

        initial_cases: List[Tuple[int, Dict[str, Any]]] = []
        for case_idx in range(num_cases):
            tc = self.llm_method.prepare_shared_numpy_data(document, case_index=case_idx)
            tc["api"] = torch_api
            initial_cases.append((case_idx + 1, tc))

        if self.num_workers <= 1:
            case_results = [
                self._process_single_case(torch_api, ms_api, case_number, tc, torch_doc, ms_doc)
                for case_number, tc in initial_cases
            ]
        else:
            case_results = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_single_case,
                        torch_api,
                        ms_api,
                        case_number,
                        tc,
                        torch_doc,
                        ms_doc,
                    ): case_number
                    for case_number, tc in initial_cases
                }
                for future in as_completed(futures):
                    case_results.append(future.result())

        case_results.sort(key=lambda item: item["case_number"])

        if any(item.get("deprecated_skip", False) for item in case_results):
            op_result["status"] = "skipped_deprecated"
            op_result["case_details"] = case_results
            self._safe_print("  检测到算子版本淘汰，跳过该算子")
            return op_result

        if case_results and all(item.get("llm_skipped", False) for item in case_results):
            op_result["status"] = "skipped_all_llm_skip"
            op_result["case_details"] = case_results
            self._safe_print("  该算子所有用例均被LLM跳过，剔除")
            return op_result

        for item in case_results:
            if item.get("llm_skipped", False):
                continue
            op_result["llm_generated_total"] += item.get("llm_generated_total", 0)
            op_result["llm_ms_success"] += item.get("llm_ms_success", 0)
            op_result["rule_generated_total"] += item.get("rule_generated_total", 0)
            op_result["rule_ms_success"] += item.get("rule_ms_success", 0)

        op_result["case_details"] = case_results
        return op_result

    def _process_single_case(
        self,
        torch_api: str,
        ms_api: str,
        case_number: int,
        initial_torch_case: Dict[str, Any],
        torch_doc: str,
        ms_doc: str,
    ) -> Dict[str, Any]:
        case_result: Dict[str, Any] = {
            "case_number": case_number,
            "llm_skipped": False,
            "deprecated_skip": False,
            "llm_generated_total": 0,
            "llm_ms_success": 0,
            "rule_generated_total": 1,
            "rule_ms_success": 0,
            "rule_detail": {},
            "llm_detail": {},
        }

        rule_ms_case = self.rule_converter.convert_torch_case_to_ms_case(initial_torch_case, ms_api)

        self._safe_print(f"  [用例{case_number}] 规则方法执行...", end="")
        try:
            with self.execution_lock:
                rule_exec = self.llm_method.execute_test_case(
                    torch_api,
                    ms_api,
                    initial_torch_case,
                    rule_ms_case,
                )
            rule_ms_ok = bool(rule_exec.get("mindspore_success", False))
            case_result["rule_ms_success"] = 1 if rule_ms_ok else 0
            self._safe_print(f" MS:{'✓' if rule_ms_ok else '✗'}")
        except Exception as error:
            rule_exec = {
                "status": "fatal_error",
                "mindspore_success": False,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            self._safe_print(f" MS:✗ ({str(error)[:80]})")

        case_result["rule_detail"] = {
            "generated_ms_test_case": rule_ms_case,
            "execution_result": rule_exec,
        }

        initial_ms_case = copy.deepcopy(initial_torch_case)
        initial_ms_case["api"] = ms_api

        self._safe_print(f"  [用例{case_number}] LLM初始执行...", end="")
        try:
            with self.execution_lock:
                initial_exec = self.llm_method.execute_test_case(
                    torch_api,
                    ms_api,
                    initial_torch_case,
                    initial_ms_case,
                )
            self._safe_print(
                f" PT:{'✓' if initial_exec.get('torch_success') else '✗'}"
                f" MS:{'✓' if initial_exec.get('mindspore_success') else '✗'}"
            )
        except Exception as error:
            initial_exec = {
                "torch_api": torch_api,
                "mindspore_api": ms_api,
                "status": "fatal_error",
                "torch_success": False,
                "mindspore_success": False,
                "results_match": False,
                "torch_error": str(error),
                "mindspore_error": None,
                "comparison_error": None,
            }
            self._safe_print(f" PT:✗ MS:✗ ({str(error)[:80]})")

        torch_error = str(initial_exec.get("torch_error", ""))
        if (not initial_exec.get("torch_success", False)) and torch_error:
            if any(keyword in torch_error.lower() for keyword in ["deprecated", "removed"]):
                case_result["deprecated_skip"] = True
                case_result["llm_detail"] = {
                    "initial_exec": initial_exec,
                    "llm_operation": "skip",
                    "llm_reason": "PyTorch算子已被版本淘汰",
                }
                self._append_realtime_record(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "operator": torch_api,
                        "case_number": case_number,
                        "status": "skipped_deprecated",
                        "llm_skipped": False,
                        "deprecated_skip": True,
                        "llm_ms_success": False,
                        "rule_ms_success": bool(case_result["rule_ms_success"]),
                    }
                )
                return case_result

        self._safe_print(f"  [用例{case_number}] 调用LLM...", end="")
        llm_result = self.llm_method.call_llm_for_repair_or_mutation(
            initial_exec,
            initial_torch_case,
            initial_ms_case,
            torch_doc,
            ms_doc,
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
                    "llm_ms_success": False,
                    "rule_ms_success": bool(case_result["rule_ms_success"]),
                }
            )
            return case_result

        llm_pt_case = llm_result.get("pytorch_test_case", initial_torch_case)
        llm_ms_case = llm_result.get("mindspore_test_case", initial_ms_case)

        try:
            llm_pt_case, llm_ms_case = self.llm_method._convert_llm_test_cases(llm_pt_case, llm_ms_case)
        except Exception as error:
            case_result["llm_skipped"] = True
            case_result["llm_detail"] = {
                "initial_exec": initial_exec,
                "llm_operation": "skip",
                "llm_reason": f"LLM用例转换失败: {error}",
            }
            self._safe_print(f"  [用例{case_number}] LLM用例转换失败，跳过: {str(error)[:70]}")
            self._append_realtime_record(
                {
                    "timestamp": datetime.now().isoformat(),
                    "operator": torch_api,
                    "case_number": case_number,
                    "status": "llm_case_convert_failed",
                    "llm_skipped": True,
                    "deprecated_skip": False,
                    "llm_ms_success": False,
                    "rule_ms_success": bool(case_result["rule_ms_success"]),
                }
            )
            return case_result

        case_result["llm_generated_total"] = 1

        self._safe_print(f"  [用例{case_number}] 执行LLM生成用例...", end="")
        try:
            with self.execution_lock:
                llm_exec = self.llm_method.execute_test_case(
                    torch_api,
                    ms_api,
                    llm_pt_case,
                    llm_ms_case,
                )
            llm_ms_ok = bool(llm_exec.get("mindspore_success", False))
            case_result["llm_ms_success"] = 1 if llm_ms_ok else 0
            self._safe_print(
                f" PT:{'✓' if llm_exec.get('torch_success') else '✗'}"
                f" MS:{'✓' if llm_ms_ok else '✗'}"
            )
        except Exception as error:
            llm_exec = {
                "status": "fatal_error",
                "torch_success": False,
                "mindspore_success": False,
                "results_match": False,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            self._safe_print(f" PT:✗ MS:✗ ({str(error)[:80]})")

        case_result["llm_detail"] = {
            "initial_exec": initial_exec,
            "llm_operation": operation,
            "llm_reason": reason,
            "generated_pt_test_case": llm_pt_case,
            "generated_ms_test_case": llm_ms_case,
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
                "llm_ms_success": bool(case_result["llm_ms_success"]),
                "rule_ms_success": bool(case_result["rule_ms_success"]),
                "llm_operation": operation,
            }
        )

        return case_result

    def print_and_save_results(self, result: Dict[str, Any]) -> str:
        stats = result["global_stats"]

        print("\n" + "=" * 80)
        print("LLM方法 vs 规则方法（仅名称+dtype自动转换）")
        print("=" * 80)
        print(f"算子总数: {stats['total_operators']}")
        print(f"- 无MS映射跳过: {stats['skipped_operators_no_ms']}")
        print(f"- LLM全跳过算子: {stats['skipped_operators_all_llm_skip']}")
        print(f"- 版本淘汰跳过算子: {stats['skipped_operators_deprecated']}")
        print(f"- 实际参与对比: {stats['tested_operators']}")

        llm_total = stats["llm_generated_total"]
        llm_success = stats["llm_ms_success"]
        llm_rate = (llm_success / llm_total * 100.0) if llm_total > 0 else 0.0

        rule_total = stats["rule_generated_total"]
        rule_success = stats["rule_ms_success"]
        rule_rate = (rule_success / rule_total * 100.0) if rule_total > 0 else 0.0

        print("\n" + "-" * 48)
        print("LLM方法（生成的新MS用例）")
        print(f"- 生成用例数: {llm_total}")
        print(f"- MS执行成功数: {llm_success}")
        print(f"- MS执行成功率: {llm_rate:.2f}%" if llm_total > 0 else "- MS执行成功率: N/A")

        print("\n" + "-" * 48)
        print("规则方法（生成的新MS用例）")
        print(f"- 生成用例数: {rule_total}")
        print(f"- MS执行成功数: {rule_success}")
        print(f"- MS执行成功率: {rule_rate:.2f}%" if rule_total > 0 else "- MS执行成功率: N/A")

        print("\n" + "-" * 48)
        if llm_total > 0 and rule_total > 0:
            diff = llm_rate - rule_rate
            if diff > 0:
                print(f"结论: LLM方法高于规则方法 {diff:.2f} 个百分点")
            elif diff < 0:
                print(f"结论: 规则方法高于LLM方法 {-diff:.2f} 个百分点")
            else:
                print("结论: 两种方法持平")
        else:
            print("结论: 统计样本不足，无法比较")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.result_dir, f"llm_vs_rulebased_result_{timestamp}.json")

        result_to_save = self._make_serializable(result)
        result_to_save["summary"] = {
            "llm_ms_success_rate": f"{llm_rate:.2f}%" if llm_total > 0 else "N/A",
            "rule_ms_success_rate": f"{rule_rate:.2f}%" if rule_total > 0 else "N/A",
            "timestamp": datetime.now().isoformat(),
        }

        self._atomic_dump_json(result_file, result_to_save)

        print("=" * 80)
        print(f"实时JSONL: {self.realtime_file_path}")
        print(f"汇总JSON: {result_file}")

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
        if isinstance(obj, mindspore.Tensor):
            return {
                "__type__": "ms_tensor",
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
    parser = argparse.ArgumentParser(description="LLM方法 vs 规则方法：PyTorch->MindSpore新用例执行成功率对比")
    parser.add_argument("--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES, help=f"每个算子测试用例数（默认{DEFAULT_NUM_CASES}）")
    parser.add_argument("--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS, help="LLM迭代次数（本脚本固定为1）")
    parser.add_argument("--start", type=int, default=1, help="起始算子索引（从1开始）")
    parser.add_argument("--end", type=int, default=None, help="结束算子索引（包含）")
    parser.add_argument("--operators", "-o", nargs="*", help="指定算子列表")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS, help=f"并发线程数（默认{DEFAULT_WORKERS}）")
    args = parser.parse_args()

    print("=" * 80)
    print("LLM方法 vs 规则方法（仅名称+dtype自动转换）")
    print("=" * 80)
    print(f"每算子用例数: {args.num_cases}")
    print("LLM迭代次数: 1（固定）")
    print(f"并发线程数: {args.workers}")
    print("=" * 80)

    comparator = LLMvsRuleBasedComparator(num_workers=args.workers)

    start_time = time.time()

    try:
        try:
            all_docs = list(comparator.llm_method.collection.find({}, {"api": 1}))
        except Exception as error:
            print(f"无法连接MongoDB或读取算子列表: {error}")
            print("请确认MongoDB服务已启动（默认 mongodb://localhost:27017/），或检查网络/权限配置。")
            return

        all_operators = [doc["api"] for doc in all_docs if "api" in doc]

        print(f"\n数据库中共有 {len(all_operators)} 个算子")

        if args.operators:
            operator_names = args.operators
        else:
            start_idx = max(1, args.start) - 1
            end_idx = args.end if args.end is not None else len(all_operators)
            end_idx = min(end_idx, len(all_operators))
            if start_idx >= end_idx:
                raise ValueError(f"起始索引 {args.start} 必须小于结束索引 {end_idx}")
            operator_names = all_operators[start_idx:end_idx]
            print(f"测试范围: 第 {start_idx + 1} 到第 {end_idx} 个算子")

        print(f"实际测试算子数: {len(operator_names)}")
        preview = ", ".join(operator_names[:10])
        print(f"前10个算子: {preview}{'...' if len(operator_names) > 10 else ''}\n")

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
        print(f"\n总耗时: {hours}h {minutes}m {seconds}s")

    finally:
        comparator.close()
        print("程序执行完成")


if __name__ == "__main__":
    main()
