# ./component/migrate_fix_names.py
# 快速修复：从原始文件中提取真实测试函数名并更新 migration_candidates_fuzzy.jsonl
import json
import ast
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def extract_real_test_names(file_path):
    """从文件中提取所有测试函数名及其行号"""
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        test_funcs = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
                test_funcs[node.lineno] = node.name
        return test_funcs
    except:
        return {}

def load_jsonl(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []

def main():
    # 输入输出文件
    Path("data/migration").mkdir(parents=True, exist_ok=True)
    in_file = Path("data/migration/migration_candidates_fuzzy.jsonl")
    out_file = Path("data/migration/migration_candidates_fuzzy_fixed.jsonl")
    tf_mapped_file = Path("data/mapping/tests_tf.mapped.jsonl")
    
    if not in_file.exists():
        print(f"[ERROR] 输入文件不存在: {in_file}")
        return
    
    # 加载现有数据
    candidates = load_jsonl(in_file)
    print(f"[LOAD] 加载了 {len(candidates)} 个候选测试")
    
    # 直接从文件中提取真实测试函数名（最可靠的方法）
    print("[INFO] 直接从原始文件中提取测试函数名...")
    tf_root = Path("framework/tensorflow-master")
    file_test_names = {}
    
    # 收集需要处理的文件
    files_to_process = set(item.get("file", "") for item in candidates)
    print(f"[INFO] 需要处理 {len(files_to_process)} 个不同的文件")
    
    for file_path in tqdm(files_to_process, desc="Extracting test names"):
        # 处理文件路径
        if file_path.startswith("framework/tensorflow-master/"):
            full_path = Path(file_path)
        else:
            full_path = tf_root / file_path
        
        if full_path.exists():
            test_funcs = extract_real_test_names(full_path)
            if test_funcs:
                # 存储为按行号排序的列表
                file_test_names[file_path] = sorted(test_funcs.items())
        else:
            file_test_names[file_path] = []
    
    print(f"[INFO] 成功提取了 {sum(1 for v in file_test_names.values() if v)} 个文件的测试函数名")
    
    # 更新候选数据
    # 按文件分组，统计每个文件有多少个 unknown_test_*
    file_unknown_counts = defaultdict(int)
    file_candidates = defaultdict(list)
    
    for idx, item in enumerate(candidates):
        file_path = item.get("file", "")
        old_name = item.get("name", "")
        if old_name.startswith("unknown_test_"):
            file_unknown_counts[file_path] += 1
            file_candidates[file_path].append((idx, item))
    
    fixed_count = 0
    results = []
    
    for item in tqdm(candidates, desc="Fixing names"):
        file_path = item.get("file", "")
        old_name = item.get("name", "")
        
        # 如果已经是真实名称（不以 unknown_test_ 开头），保持不变
        if not old_name.startswith("unknown_test_"):
            results.append(item)
            continue
        
        # 直接从文件中提取的测试函数列表中查找
        real_name = None
        
        if file_path in file_test_names:
            test_func_list = file_test_names[file_path]
            if test_func_list:
                unknown_count = file_unknown_counts.get(file_path, 0)
                
                # 策略1: 如果文件中的测试函数数量等于该文件的 unknown_test_* 数量，按顺序匹配
                if len(test_func_list) == unknown_count and unknown_count > 0:
                    file_items = file_candidates[file_path]
                    for pos, (_, candidate) in enumerate(file_items):
                        if candidate == item:
                            if pos < len(test_func_list):
                                real_name = test_func_list[pos][1]
                            break
                # 策略2: 如果只有一个候选测试，使用第一个测试函数
                elif unknown_count == 1:
                    real_name = test_func_list[0][1]
                # 策略3: 尝试从 old_name 中提取编号
                else:
                    try:
                        test_num = int(old_name.replace("unknown_test_", ""))
                        # 如果编号在范围内，使用对应的测试函数
                        if 0 <= test_num < len(test_func_list):
                            real_name = test_func_list[test_num][1]
                        # 如果编号超出范围，但只有一个测试函数，使用它
                        elif len(test_func_list) == 1:
                            real_name = test_func_list[0][1]
                    except:
                        # 如果无法解析编号，且只有一个测试函数，使用它
                        if len(test_func_list) == 1:
                            real_name = test_func_list[0][1]
                        # 否则使用第一个
                        elif test_func_list:
                            real_name = test_func_list[0][1]
        
        # 如果找到了真实名称，更新
        if real_name:
            item["name"] = real_name
            fixed_count += 1
        
        results.append(item)
    
    # 保存修复后的数据
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print("\n==== FIX SUMMARY ====")
    print(f"总测试数: {len(candidates)}")
    print(f"修复数量: {fixed_count}")
    print(f"修复后文件: {out_file}")
    print(f"\n提示: 可以重命名文件:")
    print(f"  mv {out_file} {in_file}")

if __name__ == "__main__":
    main()

