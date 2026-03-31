"""
LLM repair/mutation effectiveness validation for PyTorch vs TensorFlow.

Experimental goals: 1) For use cases that fail in initial execution, count the minimum number of repair rounds; after successful repair, continue to mutate and count the executability maintenance status. 2) For initial successfully executed use cases, perform mutations and count the executability maintenance status of each round.  Output four logs：
- repair_detail_realtime.json
- repair_stats_summary.json
- mutate_detail_realtime.json
- mutate_stats_summary.json
"""

import argparse
import copy
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path before importing component package.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from component.doc.doc_crawler_factory import get_doc_content

from llm_enhanced_compare import (
    DEFAULT_KEY_PATH,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MODEL,
    DEFAULT_NUM_CASES,
    DEFAULT_WORKERS,
    LLMEnhancedComparator,
)


class LLMEffValidator(LLMEnhancedComparator):
    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017/",
        db_name: str = "freefuzz-torch",
        key_path: str = DEFAULT_KEY_PATH,
        model: str = DEFAULT_MODEL,
        llm_workers: int = DEFAULT_WORKERS,
    ):
        super().__init__(
            mongo_uri=mongo_uri,
            db_name=db_name,
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

        self.repair_detail: Dict[str, Any] = {
            "updated_at": datetime.now().isoformat(),
            "records": [],
        }
        self.mutate_detail: Dict[str, Any] = {
            "updated_at": datetime.now().isoformat(),
            "records": [],
        }

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

    def _fetch_api_docs(self, torch_api: str, tensorflow_api: str) -> Tuple[str, str]:
        """Silently crawl documents to avoid outputting redundant information on the console。"""
        min_doc_length = 300
        torch_doc = ""
        tensorflow_doc = ""

        try:
            torch_doc = get_doc_content(torch_api, "pytorch")
            if not (
                torch_doc
                and "Unable" not in torch_doc
                and "not supported" not in torch_doc
                and len(torch_doc.strip()) > min_doc_length
            ):
                torch_doc = f"Unable to fetch documentation for {torch_api}"
            elif len(torch_doc) > 3000:
                torch_doc = torch_doc[:3000] + "\n... (doc truncated)"
        except Exception as exc:
            torch_doc = f"Failed to fetch documentation: {exc}"

        try:
            tensorflow_doc = get_doc_content(tensorflow_api, "tensorflow")
            if not (
                tensorflow_doc
                and "Unable" not in tensorflow_doc
                and "not supported" not in tensorflow_doc
                and len(tensorflow_doc.strip()) > min_doc_length
            ):
                tensorflow_doc = f"Unable to fetch documentation for {tensorflow_api}"
            elif len(tensorflow_doc) > 3000:
                tensorflow_doc = tensorflow_doc[:3000] + "\n... (doc truncated)"
        except Exception as exc:
            tensorflow_doc = f"Failed to fetch documentation: {exc}"

        return torch_doc, tensorflow_doc

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
        return bool(execution_result.get("torch_success") and execution_result.get("tensorflow_success"))

    def _status_symbol(self, execution_result: Dict[str, Any]) -> str:
        return "✓" if self._both_success(execution_result) else "✗"

    def _build_repair_stats_output(self) -> Dict[str, Any]:
        effective_total = self.repair_stats["effective_failed_total"]
        repaired_total = self.repair_stats["repaired_total"]

        repaired_at_ratio = {}
        for k, v in self.repair_stats["repaired_at"].items():
            repaired_at_ratio[k] = (v / effective_total) if effective_total > 0 else 0.0

        mutation_ratio = {}
        for k, v in self.repair_stats["post_repair_mutation_success"].items():
            mutation_ratio[k] = (v / repaired_total) if repaired_total > 0 else 0.0

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
        mutate_ratio = {}
        for k, v in self.mutate_stats["mutate_success_at"].items():
            mutate_ratio[k] = (v / total) if total > 0 else 0.0

        return {
            "updated_at": datetime.now().isoformat(),
            "mutate_stats": {
                **self.mutate_stats,
                "mutate_success_at_ratio": mutate_ratio,
            },
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

    def _update_repair_counters(
        self,
        repaired_round: Optional[int],
        mutation_success: Dict[int, bool],
        skipped: bool,
    ) -> None:
        with self.stats_lock:
            self.repair_stats["initial_failed_total"] += 1
            if skipped:
                self.repair_stats["skipped_cases"] += 1
                return

            self.repair_stats["effective_failed_total"] += 1
            if repaired_round is not None:
                self.repair_stats["repaired_total"] += 1
                self.repair_stats["repaired_at"][str(repaired_round)] += 1

            for round_idx in (1, 2, 3):
                if mutation_success.get(round_idx, False):
                    self.repair_stats["post_repair_mutation_success"][str(round_idx)] += 1

    def _update_mutate_counters(self, iteration_success: Dict[int, bool]) -> None:
        with self.stats_lock:
            self.mutate_stats["initial_success_total"] += 1
            for i in (1, 2, 3):
                if iteration_success.get(i, False):
                    self.mutate_stats["mutate_success_at"][str(i)] += 1

    def _run_single_failed_case(
        self,
        operator_name: str,
        tensorflow_api: str,
        case_number: int,
        torch_case: Dict[str, Any],
        tf_case: Dict[str, Any],
        init_result: Dict[str, Any],
        max_iterations: int,
        torch_doc: str,
        tensorflow_doc: str,
    ) -> None:
        current_torch_case = torch_case
        current_tf_case = tf_case
        current_exec = init_result

        repaired_round: Optional[int] = None
        post_repair_mut_round = 0
        mutation_success = {1: False, 2: False, 3: False}
        skipped = False

        round_records: List[Dict[str, Any]] = []

        for iteration in range(1, max_iterations + 1):
            try:
                llm_result = self.call_llm_for_repair_or_mutation(
                    current_exec,
                    current_torch_case,
                    current_tf_case,
                    torch_doc,
                    tensorflow_doc,
                )
            except Exception as exc:
                llm_result = {
                    "operation": "skip",
                    "reason": f"llm_exception: {exc}",
                }

            operation = llm_result.get("operation", "skip")
            reason = llm_result.get("reason", "")

            if operation == "skip":
                skipped = True if repaired_round is None else skipped
                round_records.append(
                    {
                        "iteration": iteration,
                        "operation": "skip",
                        "reason": reason,
                        "execute_symbol": "-",
                    }
                )
                break

            next_torch = llm_result.get("pytorch_test_case", current_torch_case)
            next_tf = llm_result.get("tensorflow_test_case", current_tf_case)
            current_torch_case, current_tf_case = self._convert_llm_test_cases(next_torch, next_tf)

            current_exec = self._execute_test_case_sequential(
                operator_name,
                tensorflow_api,
                current_torch_case,
                current_tf_case,
            )

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
                    "torch_success": bool(current_exec.get("torch_success")),
                    "tensorflow_success": bool(current_exec.get("tensorflow_success")),
                }
            )

        self._update_repair_counters(repaired_round, mutation_success, skipped)

        self._append_repair_record(
            {
                "operator": operator_name,
                "case_number": case_number,
                "initial_symbol": self._status_symbol(init_result),
                "repaired_round": repaired_round,
                "skipped": skipped,
                "post_repair_mutation_success": mutation_success,
                "rounds": round_records,
            }
        )

        with self.print_lock:
            repair_mark = f"R{repaired_round}" if repaired_round is not None else "R-"
            skip_mark = "S" if skipped else "N"
            print(f"  case#{case_number:02d} init:✗ {repair_mark} skip:{skip_mark}")

    def _run_single_success_case(
        self,
        operator_name: str,
        tensorflow_api: str,
        case_number: int,
        torch_case: Dict[str, Any],
        tf_case: Dict[str, Any],
        init_result: Dict[str, Any],
        max_iterations: int,
        torch_doc: str,
        tensorflow_doc: str,
    ) -> None:
        current_torch_case = torch_case
        current_tf_case = tf_case
        current_exec = init_result

        iteration_success = {1: False, 2: False, 3: False}
        round_records: List[Dict[str, Any]] = []

        for iteration in range(1, max_iterations + 1):
            try:
                llm_result = self.call_llm_for_repair_or_mutation(
                    current_exec,
                    current_torch_case,
                    current_tf_case,
                    torch_doc,
                    tensorflow_doc,
                )
            except Exception as exc:
                llm_result = {
                    "operation": "skip",
                    "reason": f"llm_exception: {exc}",
                }

            operation = llm_result.get("operation", "skip")
            reason = llm_result.get("reason", "")

            if operation == "skip":
                round_records.append(
                    {
                        "iteration": iteration,
                        "operation": "skip",
                        "reason": reason,
                        "execute_symbol": "-",
                    }
                )
                break

            next_torch = llm_result.get("pytorch_test_case", current_torch_case)
            next_tf = llm_result.get("tensorflow_test_case", current_tf_case)
            current_torch_case, current_tf_case = self._convert_llm_test_cases(next_torch, next_tf)

            current_exec = self._execute_test_case_sequential(
                operator_name,
                tensorflow_api,
                current_torch_case,
                current_tf_case,
            )

            is_success = self._both_success(current_exec)
            if iteration <= 3:
                iteration_success[iteration] = is_success

            round_records.append(
                {
                    "iteration": iteration,
                    "operation": operation,
                    "reason": reason,
                    "execute_symbol": self._status_symbol(current_exec),
                    "torch_success": bool(current_exec.get("torch_success")),
                    "tensorflow_success": bool(current_exec.get("tensorflow_success")),
                }
            )

        self._update_mutate_counters(iteration_success)

        self._append_mutate_record(
            {
                "operator": operator_name,
                "case_number": case_number,
                "initial_symbol": self._status_symbol(init_result),
                "mutate_success": iteration_success,
                "rounds": round_records,
            }
        )

        with self.print_lock:
            m1 = "✓" if iteration_success[1] else "✗"
            m2 = "✓" if iteration_success[2] else "✗"
            m3 = "✓" if iteration_success[3] else "✗"
            print(f"  case#{case_number:02d} init:✓ m1:{m1} m2:{m2} m3:{m3}")

    def validate_operator(
        self,
        operator_name: str,
        max_iterations: int,
        num_cases: int,
        llm_workers: int,
    ) -> Dict[str, int]:
        torch_api, tensorflow_api, _ = self.convert_api_name(operator_name)
        if tensorflow_api is None:
            return {"skipped_no_tf_mapping": 1, "processed_cases": 0}

        document = self.collection.find_one({"api": operator_name})
        if document is None:
            return {"skipped_no_doc": 1, "processed_cases": 0}

        total_cases = self.get_num_test_cases_from_document(document)
        use_cases = min(num_cases, total_cases)

        with self.print_lock:
            print(f"[{operator_name}] tf:{tensorflow_api} cases:{use_cases}")

        torch_doc, tensorflow_doc = self._fetch_api_docs(torch_api, tensorflow_api)

        initial_entries: List[Tuple[int, Dict[str, Any], Dict[str, Any], Dict[str, Any], bool]] = []

        for case_idx in range(use_cases):
            case_number = case_idx + 1
            base_case = self.prepare_shared_numpy_data(document, case_index=case_idx)
            torch_case = copy.deepcopy(base_case)
            torch_case["api"] = torch_api
            tf_case = copy.deepcopy(base_case)
            tf_case["api"] = tensorflow_api

            init_exec = self._execute_test_case_sequential(torch_api, tensorflow_api, torch_case, tf_case)
            init_ok = self._both_success(init_exec)
            initial_entries.append((case_number, torch_case, tf_case, init_exec, init_ok))

            with self.print_lock:
                init_symbol = "✓" if init_ok else "✗"
                print(f"  case#{case_number:02d} init:{init_symbol}")

        if llm_workers <= 1:
            for case_number, torch_case, tf_case, init_exec, init_ok in initial_entries:
                if init_ok:
                    self._run_single_success_case(
                        torch_api,
                        tensorflow_api,
                        case_number,
                        torch_case,
                        tf_case,
                        init_exec,
                        max_iterations,
                        torch_doc,
                        tensorflow_doc,
                    )
                else:
                    self._run_single_failed_case(
                        torch_api,
                        tensorflow_api,
                        case_number,
                        torch_case,
                        tf_case,
                        init_exec,
                        max_iterations,
                        torch_doc,
                        tensorflow_doc,
                    )
        else:
            with ThreadPoolExecutor(max_workers=llm_workers) as pool:
                futures = []
                for case_number, torch_case, tf_case, init_exec, init_ok in initial_entries:
                    if init_ok:
                        futures.append(
                            pool.submit(
                                self._run_single_success_case,
                                torch_api,
                                tensorflow_api,
                                case_number,
                                torch_case,
                                tf_case,
                                init_exec,
                                max_iterations,
                                torch_doc,
                                tensorflow_doc,
                            )
                        )
                    else:
                        futures.append(
                            pool.submit(
                                self._run_single_failed_case,
                                torch_api,
                                tensorflow_api,
                                case_number,
                                torch_case,
                                tf_case,
                                init_exec,
                                max_iterations,
                                torch_doc,
                                tensorflow_doc,
                            )
                        )

                for future in as_completed(futures):
                    future.result()

        return {"processed_cases": use_cases, "skipped_no_tf_mapping": 0, "skipped_no_doc": 0}


def _select_operators(
    validator: LLMEffValidator,
    start: int,
    end: Optional[int],
    explicit_ops: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    all_operators = list(validator.collection.find({}, {"api": 1}))
    all_operator_names = [doc["api"] for doc in all_operators if "api" in doc]

    if explicit_ops:
        candidates = explicit_ops
    else:
        total = len(all_operator_names)
        begin_idx = max(1, start) - 1
        end_idx = end if end is not None else total
        end_idx = min(total, end_idx)
        if begin_idx >= end_idx:
            raise ValueError(f"Invalid range: start={start}, end={end_idx}")
        candidates = all_operator_names[begin_idx:end_idx]

    valid_ops = []
    skipped_ops = []
    for op in candidates:
        _, tf_api, reason = validator.convert_api_name(op)
        if tf_api is None:
            skipped_ops.append(f"{op} ({reason})")
        else:
            valid_ops.append(op)

    return valid_ops, skipped_ops


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM repair/mutation effectiveness validator")
    parser.add_argument("--max-iterations", "-m", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--num-cases", "-n", type=int, default=DEFAULT_NUM_CASES)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--operators", "-o", nargs="*")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--key-path", "-k", default=DEFAULT_KEY_PATH)
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017/")
    parser.add_argument("--db-name", default="freefuzz-torch")
    parser.add_argument("--limit-operators", type=int, default=None)

    args = parser.parse_args()

    max_iterations = max(1, min(3, args.max_iterations))
    num_cases = max(1, args.num_cases)
    workers = max(1, args.workers)

    print("=" * 80)
    print("LLM Effectiveness Validation")
    print("=" * 80)
    print(f"iterations={max_iterations} num_cases={num_cases} workers={workers} model={args.model}")

    validator = LLMEffValidator(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        key_path=args.key_path,
        model=args.model,
        llm_workers=workers,
    )

    start_time = time.time()
    tested_ops = []

    try:
        operator_names, skipped_ops = _select_operators(validator, args.start, args.end, args.operators)
        if args.limit_operators is not None and args.limit_operators > 0:
            operator_names = operator_names[: args.limit_operators]

        print(f"operators_to_test={len(operator_names)} skipped_no_tf={len(skipped_ops)}")
        if skipped_ops:
            print(f"skipped_sample={'; '.join(skipped_ops[:5])}")

        total_ops = len(operator_names)
        for idx, operator_name in enumerate(operator_names, 1):
            print(f"\n[{idx}/{total_ops}] {operator_name}")
            try:
                summary = validator.validate_operator(
                    operator_name=operator_name,
                    max_iterations=max_iterations,
                    num_cases=num_cases,
                    llm_workers=workers,
                )
                tested_ops.append({"operator": operator_name, **summary, "status": "ok"})
                print("  done: ✓")
            except Exception as exc:
                tested_ops.append({"operator": operator_name, "status": "failed", "error": str(exc)})
                print("  done: ✗")

        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print("Validation finished")
        print("=" * 80)
        print(f"tested_ops={len(tested_ops)} elapsed_sec={elapsed:.2f}")

        final_summary = {
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
        validator._atomic_write_json(os.path.join(validator.result_dir, "run_summary.json"), final_summary)
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
