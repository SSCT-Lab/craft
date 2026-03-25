"""将所有测试（seed + fuzzing）迁移到 PyTorch

从 seed 列表和 fuzzing 索引中读取所有测试，使用 LLM 将它们迁移到 PyTorch。
"""
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from component.migration.migrate_generate_tests import (
    get_qwen_client,
    extract_test_function_code,
    convert_tf_to_standalone,
    extract_helper_functions,
    build_migration_prompt,
    migrate_with_llm,
    safe_filename,
    DEFAULT_MODEL,
    DEFAULT_KEY_PATH,
)

DEFAULT_SEEDS_FILE = Path("dev/tf_seeds.jsonl")
DEFAULT_FUZZ_INDEX = Path("dev/tf_fuzz/tf_fuzz_index.jsonl")
DEFAULT_OUT_DIR = Path("dev/pt_migrated")
DEFAULT_TF_ROOT = Path("framework/tensorflow-master")


def load_test_list(seeds_file: Path, fuzz_index_file: Path) -> list:
    """加载所有需要迁移的测试列表（seed + fuzzing）"""
    tests = []
    
    # 加载 seeds
    if seeds_file.exists():
        for line in seeds_file.open():
            seed = json.loads(line)
            tests.append({
                "type": "seed",
                "core_file": seed["core_file"],
                "name": seed.get("test_name", ""),
                "source_file": seed.get("source_file", ""),
                "tf_file": seed.get("tf_file", ""),
                "tf_func_name": seed.get("tf_func_name", ""),
            })
    
    # 加载 fuzzing 测试（每个变体单独处理）
    if fuzz_index_file.exists():
        for line in fuzz_index_file.open():
            fuzz_item = json.loads(line)
            fuzz_file = fuzz_item["fuzz_file"]
            
            # 直接从文件名提取 variant_idx（确保编号对应）
            import re
            match = re.search(r'_fuzz_(\d+)\.py$', fuzz_file)
            variant_idx = int(match.group(1)) if match else None
            
            tests.append({
                "type": "fuzz",
                "fuzz_file": fuzz_file,
                "seed_file": fuzz_item["seed_file"],
                "seed_name": fuzz_item["seed_name"],
                "source_file": fuzz_item.get("source_file", ""),
                "test_name": fuzz_item.get("test_name", ""),
                "variant_idx": variant_idx,  # 保存 variant_idx，但输出时直接从文件名提取确保对应
            })
    
    return tests


def extract_tf_code_from_core_file(core_file: Path) -> tuple:
    """从 TF core 文件中提取测试函数代码"""
    try:
        content = core_file.read_text(encoding='utf-8')
        
        # 查找 tf_xxx 函数
        import re
        tf_func_match = re.search(r'def\s+(tf_\w+)\s*\([^)]*\):.*?(?=def\s+main|if __name__)', content, re.DOTALL)
        if not tf_func_match:
            return None, None
        
        tf_func_name = tf_func_match.group(1)
        tf_func_code = tf_func_match.group(0)
        
        # 提取 helper 函数
        helper_code = extract_helper_functions(str(core_file)) or ""
        
        return tf_func_code, helper_code, tf_func_name
    
    except Exception as e:
        print(f"[ERROR] 提取 {core_file} 失败: {e}")
        return None, None, None


def migrate_one_test(test_item: dict, client, tf_root: Path, out_dir: Path, model: str, lock: Lock = None):
    """迁移一个测试到 PyTorch"""
    try:
        if test_item["type"] == "seed":
            # Seed 测试：从 core_file 提取
            core_file = Path(test_item["core_file"])
            if not core_file.exists():
                return {"status": "error", "error": f"文件不存在: {core_file}"}
            
            tf_code, helper_code, tf_func_name = extract_tf_code_from_core_file(core_file)
            if not tf_code:
                return {"status": "error", "error": "无法提取 TF 代码"}
            
            test_name = test_item.get("name", core_file.stem.replace("tf_core_", ""))
            source_info = f"seed: {core_file.name}"
        
        else:  # fuzz
            # Fuzzing 测试：从 fuzz_file 提取
            fuzz_file = Path(test_item["fuzz_file"])
            if not fuzz_file.exists():
                return {"status": "error", "error": f"文件不存在: {fuzz_file}"}
            
            tf_code, helper_code, tf_func_name = extract_tf_code_from_core_file(fuzz_file)
            if not tf_code:
                return {"status": "error", "error": "无法提取 TF 代码"}
            
            test_name = test_item.get("test_name", test_item.get("seed_name", ""))
            source_info = f"fuzz: {fuzz_file.name}"
        
        # 提取使用的 API（简化版，从代码中提取）
        import re
        tf_apis = list(set(re.findall(r'tf\.\w+(?:\.\w+)*', tf_code)))
        
        # 构建迁移提示词（简化版，不依赖 API mapping）
        prompt = build_migration_prompt(tf_code, tf_apis[:10], [])
        
        # 使用 LLM 迁移
        migrated_code = migrate_with_llm(client, tf_code, tf_apis[:10], [], model)
        
        if not migrated_code:
            return {"status": "error", "error": "LLM 迁移失败"}
        
        # 生成输出文件名
        safe_test_name = safe_filename(test_item.get("test_name", test_item.get("seed_name", "unknown")))
        if test_item["type"] == "fuzz":
            # 直接从 fuzz_file 文件名提取 variant_idx，确保编号对应
            import re
            fuzz_file_name = Path(test_item["fuzz_file"]).stem
            match = re.search(r'_fuzz_(\d+)$', fuzz_file_name)
            if match:
                variant_idx = int(match.group(1))
                output_file = out_dir / f"pt_{safe_test_name}_fuzz_{variant_idx}.py"
            else:
                # 如果无法提取，使用默认命名
                output_file = out_dir / f"pt_{safe_test_name}_fuzz.py"
        else:
            output_file = out_dir / f"pt_{safe_test_name}.py"
        
        # 组合最终代码
        from component.migration.migrate_generate_tests import HEADER
        
        final_code = HEADER + f"""# Auto-Migrated from TF test
# source: {source_info}
# original test function: {tf_func_name}

{helper_code if helper_code else ""}
# ===== TensorFlow Original Test =====
{tf_code}

# ===== PyTorch Migrated Test =====
{migrated_code}

# ===== Main Execution =====
if __name__ == "__main__":
    import sys
    import re
    
    # Find PyTorch test function name
    pt_match = re.search(r'def\\s+(test_\\w+_pt)\\s*\\(', '''{migrated_code}''')
    pt_test_name = pt_match.group(1) if pt_match else None
    
    print("=" * 80)
    print("执行 PyTorch 版本")
    print("=" * 80)
    
    if pt_test_name:
        exec('''{migrated_code}''')
        pt_func = eval(pt_test_name)
        pt_output = capture_output(pt_func)
        
        if pt_output["stdout"]:
            print(pt_output["stdout"], end="")
        if pt_output["stderr"]:
            print(pt_output["stderr"], end="", file=sys.stderr)
        if pt_output["exception"]:
            print(f"\\nPyTorch 执行异常: {{pt_output['exception']}}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        else:
            print("\\nPyTorch 测试执行成功")
    else:
        print("错误: 找不到 PyTorch 测试函数")
"""
        
        # 写入文件（线程安全）
        if lock:
            with lock:
                output_file.write_text(final_code, encoding='utf-8')
        else:
            output_file.write_text(final_code, encoding='utf-8')
        
        return {
            "status": "success",
            "output_file": str(output_file),
            "test_name": test_name
        }
    
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    parser = argparse.ArgumentParser(description="将所有测试（seed + fuzzing）迁移到 PyTorch")
    parser.add_argument("--seeds-file", default=str(DEFAULT_SEEDS_FILE), help="Seed 列表文件")
    parser.add_argument("--fuzz-index", default=str(DEFAULT_FUZZ_INDEX), help="Fuzzing 索引文件")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    parser.add_argument("--tf-root", default=str(DEFAULT_TF_ROOT), help="TensorFlow 源码根目录")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM 模型名称")
    parser.add_argument("--key-path", default=DEFAULT_KEY_PATH, help="API key 路径")
    parser.add_argument("--limit", type=int, default=-1, help="限制迁移数量，-1 表示全部")
    parser.add_argument("--force", action="store_true", help="强制覆盖已存在的文件")
    parser.add_argument("--workers", type=int, default=10, help="并发线程数（默认 10）")
    args = parser.parse_args()
    
    seeds_file = Path(args.seeds_file)
    fuzz_index_file = Path(args.fuzz_index)
    out_dir = Path(args.output_dir)
    tf_root = Path(args.tf_root)
    
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # 加载测试列表
    tests = load_test_list(seeds_file, fuzz_index_file)
    
    if args.limit > 0:
        tests = tests[:args.limit]
    
    print(f"[LOAD] 找到 {len(tests)} 个测试需要迁移")
    print(f"  - Seed: {sum(1 for t in tests if t['type'] == 'seed')}")
    print(f"  - Fuzzing: {sum(1 for t in tests if t['type'] == 'fuzz')}")
    
    # 初始化 LLM 客户端（为每个线程创建独立的客户端）
    print(f"[INIT] 初始化 LLM 客户端 (model: {args.model}, workers: {args.workers})...")
    from component.migration.migrate_generate_tests import initialize_llm_clients
    clients = initialize_llm_clients(args.key_path, args.workers)
    
    if not any(clients):
        print("[ERROR] 无法初始化任何 LLM 客户端，退出")
        return
    
    success_count = 0
    failed_count = 0
    lock = Lock()  # 用于文件写入的线程安全
    
    def process_test(test_item_with_idx):
        """处理单个测试的包装函数"""
        idx, test_item = test_item_with_idx
        client = clients[idx % len(clients)]  # 轮询分配客户端
        
        # 添加延迟，避免请求速率过快（每个线程错开时间）
        # 每个线程延迟 (idx % args.workers) * 0.1 秒
        time.sleep((idx % args.workers) * 0.1)
        
        # 在调用 LLM 前添加小延迟，平缓请求速率
        time.sleep(0.2)
        
        return migrate_one_test(test_item, client, tf_root, out_dir, args.model, lock)
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务
        future_to_test = {
            executor.submit(process_test, (idx, test_item)): test_item
            for idx, test_item in enumerate(tests)
        }
        
        # 使用 tqdm 显示进度
        with tqdm(total=len(tests), desc="Migrating to PyTorch") as pbar:
            for future in as_completed(future_to_test):
                test_item = future_to_test[future]
                try:
                    result = future.result()
                    
                    if result["status"] == "success":
                        success_count += 1
                        pbar.write(f"[OK] {result['test_name']} -> {Path(result['output_file']).name}")
                    else:
                        failed_count += 1
                        error_msg = result.get("error", "Unknown error")
                        pbar.write(f"[FAIL] {test_item.get('test_name', 'unknown')}: {error_msg[:100]}")
                    
                except Exception as e:
                    failed_count += 1
                    pbar.write(f"[ERROR] {test_item.get('test_name', 'unknown')}: {str(e)[:100]}")
                
                pbar.update(1)
    
    print(f"\n[SUMMARY]")
    print(f"  成功: {success_count}")
    print(f"  失败: {failed_count}")
    print(f"[OUTPUT] 迁移后的 PyTorch 测试已保存到: {out_dir}")


if __name__ == "__main__":
    main()

