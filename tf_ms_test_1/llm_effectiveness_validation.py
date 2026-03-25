"""
LLM repair/mutation effectiveness validation for TensorFlow vs MindSpore.
"""

import argparse
import copy
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from llm_enhanced_compare import (
    DEFAULT_KEY_PATH,
    DEFAULT_MAPPING_FILE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MODEL,
    DEFAULT_NUM_CASES,
    DEFAULT_TEST_CASES_FILE,
    DEFAULT_WORKERS,
    LLMEnhancedComparator,
)


class LLMEffValidator(LLMEnhancedComparator):
    def __init__(
        self,
        test_cases_file: str = DEFAULT_TEST_CASES_FILE,
        mapping_file: str = DEFAULT_MAPPING_FILE,
        key_path: str = DEFAULT_KEY_PATH,
        model: str = DEFAULT_MODEL,
        llm_workers: int = DEFAULT_WORKERS,
    ):
        super().__init__(
            test_cases_file=test_cases_file,
            mapping_file=mapping_file,
            key_path=key_path,
            model=model,
            llm_workers=llm_workers,
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = os.path.join(self.result_dir, f"effectiveness_{ts}")
        os.makedirs(self.result_dir, exist_ok=True)

        self.repair_detail_path = os.path.join(self.result_dir, "repair_detail_realtime.json")
        self.repair_stats_path = os.path.join(self.result_dir, "repair_stats_summary.json")
        self.mutate_detail_path = os.path.join(self.result_dir, "mutate_detail_realtime.json")
        self.mutate_stats_path = os.path.join(self.result_dir, "mutate_stats_summary.json")

        self.log_lock = threading.Lock()

        self.repair_detail: Dict[str, Any] = {"updated_at": datetime.now().isoformat(), "records": []}
        self.mutate_detail: Dict[str, Any] = {"updated_at": datetime.now().isoformat(), "records": []}

        self.repair_stats: Dict[str, Any] = {
            "initial_failed_total": 0,
            "effective_failed_total": 0,
            "skipped_cases": 0,
            "repaired_total": 0,
            "repaired_at": {"1": 0, "2": 0, "3": 0},
            "post_repair_mutation_success": {"1": 0, "2": 0, "3": 0},
        }
        self.mutate_stats: Dict[str, Any] = {
            "initial_success_total": 0,
            "mutate_success_at": {"1": 0, "2": 0, "3": 0},
        }

        self._flush_all_logs()
        self._safe_print(f"Result dir: {self.result_dir}")

    def _fetch_api_docs(self, tf_api: str, mindspore_api: str) -> Tuple[str, str]:
        min_doc_length = 300
        tf_doc = ""
        mindspore_doc = ""

        try:
            tf_doc = self._truncate_doc(tf_api, "tensorflow", min_doc_length)
        except Exception as exc:
            tf_doc = f"Failed to fetch documentation: {exc}"

        try:
            mindspore_doc = self._truncate_doc(mindspore_api, "mindspore", min_doc_length)
        except Exception as exc:
            mindspore_doc = f"Failed to fetch documentation: {exc}"

        return tf_doc, mindspore_doc

    def _truncate_doc(self, api_name: str, framework: str, min_doc_length: int) -> str:
        from component.doc.doc_crawler_factory import get_doc_content

        doc = get_doc_content(api_name, framework)
        if not (
            doc
            and "Unable" not in doc
            and "not supported" not in doc
            and len(doc.strip()) > min_doc_length
        ):
            return f"Unable to fetch documentation for {api_name}"
        if len(doc) > 3000:
            return doc[:3000] + "\n... (doc truncated)"
        return doc

    def _to_jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_jsonable(v) for v in value]
        if isinstance(value, tuple):
            return [self._to_jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sample_values": value.flatten()[:10].tolist() if value.size > 0 else [],
            }
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _atomic_write_json(self, file_path: str, payload: Dict[str, Any]) -> None:
        temp_path = f"{file_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self._to_jsonable(payload), f, ensure_ascii=False, indent=2)
        os.replace(temp_path, file_path)

    def _flush_all_logs(self) -> None:
        self.repair_detail["updated_at"] = datetime.now().isoformat()
        self.mutate_detail["updated_at"] = datetime.now().isoformat()
        self._atomic_write_json(self.repair_detail_path, self.repair_detail)
        self._atomic_write_json(self.mutate_detail_path, self.mutate_detail)
        self._atomic_write_json(self.repair_stats_path, self._build_repair_stats_output())
        self._atomic_write_json(self.mutate_stats_path, self._build_mutate_stats_output())

    def _both_success(self, execution_result: Dict[str, Any]) -> bool:
        return bool(execution_result.get("tf_success") and execution_result.get("mindspore_success"))

    def _status_symbol(self, execution_result: Dict[str, Any]) -> str:
        return "✓" if self._both_success(execution_result) else "✗"

    def _build_repair_stats_output(self) -> Dict[str, Any]:
        effective_total = self.repair_stats["effective_failed_total"]
        repaired_total = self.repair_stats["repaired_total"]
        repaired_at_ratio = {
            k: (v / effective_total) if effective_total > 0 else 0.0
            for k, v in self.repair_stats["repaired_at"].items()
        }
        mutation_ratio = {
            k: (v / repaired_total) if repaired_total > 0 else 0.0
            for k, v in self.repair_stats["post_repair_mutation_success"].items()
        }
        return {
            "updated_at": datetime.now().isoformat(),
            "repair_stats": {
                **self.repair_stats,
                "repaired_at_ratio": repaired_at_ratio,
                "post_repair_mutation_success_ratio": mutation_ratio,
            },
        }

    def _build_mutate_stats_output(self) -> Dict[str, Any]:
        total = self.mutate_stats["initial_success_total"]
        mutate_ratio = {
            k: (v / total) if total > 0 else 0.0
            for k, v in self.mutate_stats["mutate_success_at"].items()
        }
        return {
            "updated_at": datetime.now().isoformat(),
            "mutate_stats": {**self.mutate_stats, "mutate_success_at_ratio": mutate_ratio},
        }

    def _append_repair_record(self, record: Dict[str, Any]) -> None:
        with self.log_lock:
            self.repair_detail["records"].append(record)
            self._atomic_write_json(self.repair_detail_path, self.repair_detail)
            self._atomic_write_json(self.repair_stats_path, self._build_repair_stats_output())

    def _append_mutate_record(self, record: Dict[str, Any]) -> None:
        with self.log_lock:
            self.mutate_detail["records"].append(record)
            self._atomic_write_json(self.mutate_detail_path, self.mutate_detail)
            self._atomic_write_json(self.mutate_stats_path, self._build_mutate_stats_output())

    def _update_repair_counters(self, repaired_round: Optional[int], mutation_success: Dict[int, bool], skipped: bool) -> None:
        with self.stats_lock:
            self.repair_stats["initial_failed_total"] += 1
            if skipped:
                self.repair_stats["skipped_cases"] += 1
                return
            self.repair_stats["effective_failed_total"] += 1
            if repaired_round is not None:
                self.repair_stats["repaired_total"] += 1
                self.repair_stats["repaired_at"][str(repaired_round)] += 1
            for i in (1, 2, 3):
                if mutation_success.get(i, False):
                    self.repair_stats["post_repair_mutation_success"][str(i)] += 1

    def _update_mutate_counters(self, iteration_success: Dict[int, bool]) -> None:
        with self.stats_lock:
            self.mutate_stats["initial_success_total"] += 1
            for i in (1, 2, 3):
                if iteration_success.get(i, False):
                    self.mutate_stats["mutate_success_at"][str(i)] += 1

    def _run_single_failed_case(
        self,
        tf_api: str,
        mindspore_api: str,
        case_number: int,
        tf_case: Dict[str, Any],
        mindspore_case: Dict[str, Any],
        init_result: Dict[str, Any],
        max_iterations: int,
        tf_doc: str,
        mindspore_doc: str,
    ) -> None:
        current_tf_case = tf_case
        current_mindspore_case = mindspore_case
        current_exec = init_result

        repaired_round: Optional[int] = None
        post_repair_mut_round = 0
        mutation_success = {1: False, 2: False, 3: False}
        skipped = False
        round_records: List[Dict[str, Any]] = []

        for iteration in range(1, max_iterations + 1):
            try:
                llm_result = self.call_llm_for_repair_or_mutation(
                    current_exec, current_tf_case, current_mindspore_case, tf_doc, mindspore_doc
                )
            except Exception as exc:
                llm_result = {"operation": "skip", "reason": f"llm_exception: {exc}"}

            operation = llm_result.get("operation", "skip")
            reason = llm_result.get("reason", "")

            if operation == "skip":
                skipped = True if repaired_round is None else skipped
                round_records.append({"iteration": iteration, "operation": "skip", "reason": reason, "execute_symbol": "-"})
                break

            next_tf = llm_result.get("tensorflow_test_case", current_tf_case)
            next_mindspore = llm_result.get("mindspore_test_case", current_mindspore_case)
            current_tf_case, current_mindspore_case = self._convert_llm_test_cases(next_tf, next_mindspore)

            current_exec = self._execute_test_case_sequential(tf_api, mindspore_api, current_tf_case, current_mindspore_case)
            is_success = self._both_success(current_exec)

            if repaired_round is None and is_success:
                repaired_round = iteration
            elif repaired_round is not None:
                post_repair_mut_round += 1
                if post_repair_mut_round <= 3:
                    mutation_success[post_repair_mut_round] = is_success

            round_records.append(
                {
                    "iteration": iteration,
                    "operation": operation,
                    "reason": reason,
                    "execute_symbol": self._status_symbol(current_exec),
                    "tf_success": bool(current_exec.get("tf_success")),
                    "mindspore_success": bool(current_exec.get("mindspore_success")),
                }
            )

        self._update_repair_counters(repaired_round, mutation_success, skipped)
        self._append_repair_record(
            {
                "operator": tf_api,
                "case_number": case_number,
                "initial_symbol": self._status_symbol(init_result),
                "repaired_round": repaired_round,
                "skipped": skipped,
                "post_repair_mutation_success": mutation_success,
                "rounds": round_records,
            }
        )

        repair_mark = f"R{repaired_round}" if repaired_round is not None else "R-"
        skip_mark = "S" if skipped else "N"
        self._safe_print(f"  case#{case_number:02d} init:✗ {repair_mark} skip:{skip_mark}")

    def _run_single_success_case(
        self,
        tf_api: str,
        mindspore_api: str,
        case_number: int,
        tf_case: Dict[str, Any],
        mindspore_case: Dict[str, Any],
        init_result: Dict[str, Any],
        max_iterations: int,
        tf_doc: str,
        mindspore_doc: str,
    ) -> None:
        current_tf_case = tf_case
        current_mindspore_case = mindspore_case
        current_exec = init_result

        iteration_success = {1: False, 2: False, 3: False}
        round_records: List[Dict[str, Any]] = []

        for iteration in range(1, max_iterations + 1):
            try:
                llm_result = self.call_llm_for_repair_or_mutation(
                    current_exec, current_tf_case, current_mindspore_case, tf_doc, mindspore_doc
                )
            except Exception as exc:
                llm_result = {"operation": "skip", "reason": f"llm_exception: {exc}"}

            operation = llm_result.get("operation", "skip")
            reason = llm_result.get("reason", "")

            if operation == "skip":
                round_records.append({"iteration": iteration, "operation": "skip", "reason": reason, "execute_symbol": "-"})
                break

            next_tf = llm_result.get("tensorflow_test_case", current_tf_case)
            next_mindspore = llm_result.get("mindspore_test_case", current_mindspore_case)
            current_tf_case, current_mindspore_case = self._convert_llm_test_cases(next_tf, next_mindspore)

            current_exec = self._execute_test_case_sequential(tf_api, mindspore_api, current_tf_case, current_mindspore_case)
            if iteration <= 3:
                iteration_success[iteration] = self._both_success(current_exec)

            round_records.append(
                {
                    "iteration": iteration,
                    "operation": operation,
                    "reason": reason,
                    "execute_symbol": self._status_symbol(current_exec),
                    "tf_success": bool(current_exec.get("tf_success")),
                    "mindspore_success": bool(current_exec.get("mindspore_success")),
                }
            )

        self._update_mutate_counters(iteration_success)
        self._append_mutate_record(
            {
                "operator": tf_api,
                "case_number": case_number,
                "initial_symbol": self._status_symbol(init_result),
                "mutate_success": iteration_success,
                "rounds": round_records,
            }
        )

        m1 = "✓" if iteration_success[1] else "✗"
        m2 = "✓" if iteration_success[2] else "✗"
        m3 = "✓" if iteration_success[3] else "✗"
        self._safe_print(f"  case#{case_number:02d} init:✓ m1:{m1} m2:{m2} m3:{m3}")

    def validate_operator(self, tf_api: str, max_iterations: int, num_cases: int, llm_workers: int) -> Dict[str, int]:
        if hasattr(self, "problematic_apis") and tf_api in self.problematic_apis:
            self._safe_print(f"[{tf_api}] skipped(problematic)")
            return {"processed_cases": 0, "skipped_problematic": 1}

        _, mindspore_api, _ = self.convert_api_name(tf_api)
        if mindspore_api is None:
            return {"processed_cases": 0, "skipped_no_mapping": 1}

        api_data = self.test_cases_data.get(tf_api, {})
        test_cases = api_data.get("test_cases", [])
        if not test_cases:
            return {"processed_cases": 0, "skipped_no_test_case": 1}

        use_cases = min(num_cases, len(test_cases))
        self._safe_print(f"[{tf_api}] ms:{mindspore_api} cases:{use_cases}")

        tf_doc, mindspore_doc = self._fetch_api_docs(tf_api, mindspore_api)

        initial_entries = []
        for idx in range(use_cases):
            case_num = idx + 1
            tc = test_cases[idx]
            if "inputs" in tc:
                flat = dict(tc["inputs"])
            else:
                flat = {k: v for k, v in tc.items() if k != "description"}
            tf_case = copy.deepcopy(flat)
            tf_case["api"] = tf_api
            ms_case = copy.deepcopy(flat)
            ms_case["api"] = mindspore_api

            init_exec = self._execute_test_case_sequential(tf_api, mindspore_api, tf_case, ms_case)
            init_ok = self._both_success(init_exec)
            initial_entries.append((case_num, tf_case, ms_case, init_exec, init_ok))
            self._safe_print(f"  case#{case_num:02d} init:{'✓' if init_ok else '✗'}")

        if llm_workers <= 1:
            for case_num, tf_case, ms_case, init_exec, init_ok in initial_entries:
                if init_ok:
                    self._run_single_success_case(tf_api, mindspore_api, case_num, tf_case, ms_case, init_exec, max_iterations, tf_doc, mindspore_doc)
                else:
                    self._run_single_failed_case(tf_api, mindspore_api, case_num, tf_case, ms_case, init_exec, max_iterations, tf_doc, mindspore_doc)
        else:
            with ThreadPoolExecutor(max_workers=llm_workers) as pool:
                futures = []
                for case_num, tf_case, ms_case, init_exec, init_ok in initial_entries:
                    if init_ok:
                        futures.append(pool.submit(self._run_single_success_case, tf_api, mindspore_api, case_num, tf_case, ms_case, init_exec, max_iterations, tf_doc, mindspore_doc))
                    else:
                        futures.append(pool.submit(self._run_single_failed_case, tf_api, mindspore_api, case_num, tf_case, ms_case, init_exec, max_iterations, tf_doc, mindspore_doc))
                for future in as_completed(futures):
                    future.result()

        return {"processed_cases": use_cases}


def _select_operators(validator: LLMEffValidator, start: int, end: Optional[int], explicit_ops: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    all_apis = validator.get_all_testable_apis()
    if explicit_ops:
        candidates = explicit_ops
    else:
        begin = max(1, start) - 1
        finish = end if end is not None else len(all_apis)
        finish = min(len(all_apis), finish)
        if begin >= finish:
            raise ValueError(f"Invalid range: start={start}, end={finish}")
        candidates = all_apis[begin:finish]

    valid = []
    skipped = []
    for api in candidates:
        if hasattr(validator, "problematic_apis") and api in validator.problematic_apis:
            skipped.append(f"{api} (problematic)")
            continue
        _, ms_api, reason = validator.convert_api_name(api)
        if ms_api is None:
            skipped.append(f"{api} ({reason})")
        else:
            valid.append(api)
    return valid, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM repair/mutation effectiveness validator (TF-MS)")
    parser.add_argument("--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--operators", "-o", nargs="*")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--key-path", "-k", default=DEFAULT_KEY_PATH)
    parser.add_argument("--test-cases-file", default=DEFAULT_TEST_CASES_FILE)
    parser.add_argument("--mapping-file", default=DEFAULT_MAPPING_FILE)
    parser.add_argument("--limit-operators", type=int, default=None)
    args = parser.parse_args()

    max_iterations = max(1, min(3, args.max_iterations))
    num_cases = max(1, args.num_cases)
    workers = max(1, args.workers)

    print("=" * 80)
    print("LLM Effectiveness Validation (TF-MS)")
    print("=" * 80)
    print(f"iterations={max_iterations} num_cases={num_cases} workers={workers} model={args.model}")

    validator = LLMEffValidator(
        test_cases_file=args.test_cases_file,
        mapping_file=args.mapping_file,
        key_path=args.key_path,
        model=args.model,
        llm_workers=workers,
    )

    start_time = time.time()
    tested_ops = []

    try:
        operators, skipped_ops = _select_operators(validator, args.start, args.end, args.operators)
        if args.limit_operators and args.limit_operators > 0:
            operators = operators[: args.limit_operators]

        print(f"operators_to_test={len(operators)} skipped={len(skipped_ops)}")
        if skipped_ops:
            print(f"skipped_sample={'; '.join(skipped_ops[:5])}")

        total = len(operators)
        for idx, tf_api in enumerate(operators, 1):
            print(f"\n[{idx}/{total}] {tf_api}")
            try:
                summary = validator.validate_operator(tf_api, max_iterations, num_cases, workers)
                tested_ops.append({"operator": tf_api, "status": "ok", **summary})
                print("  done: ✓")
            except Exception as exc:
                tested_ops.append({"operator": tf_api, "status": "failed", "error": str(exc)})
                print("  done: ✗")

        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print("Validation finished")
        print("=" * 80)
        print(f"tested_ops={len(tested_ops)} elapsed_sec={elapsed:.2f}")

        run_summary = {
            "updated_at": datetime.now().isoformat(),
            "config": {
                "max_iterations": max_iterations,
                "num_cases": num_cases,
                "workers": workers,
                "model": args.model,
                "start": args.start,
                "end": args.end,
                "operators": args.operators,
            },
            "tested_operators": tested_ops,
            "repair_stats": validator._build_repair_stats_output()["repair_stats"],
            "mutate_stats": validator._build_mutate_stats_output()["mutate_stats"],
        }
        validator._atomic_write_json(os.path.join(validator.result_dir, "run_summary.json"), run_summary)
        validator._flush_all_logs()

        print(f"logs: {validator.result_dir}")
        print(f"repair_detail: {validator.repair_detail_path}")
        print(f"repair_stats: {validator.repair_stats_path}")
        print(f"mutate_detail: {validator.mutate_detail_path}")
        print(f"mutate_stats: {validator.mutate_stats_path}")
    finally:
        validator.close()


if __name__ == "__main__":
    main()
