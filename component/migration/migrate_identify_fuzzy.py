# ./component/migrate_identify_fuzzy.py
import json
import re
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher


def load_jsonl(path):
    return [json.loads(line) for line in open(path)]


# -------------------------------
# Fuzzy API matcher
# -------------------------------
def normalize(api: str):
    if not api:
        return []
    api = api.lower()
    tokens = re.split(r"[._:/\-]", api)
    return [t for t in tokens if t]


def token_overlap(t1, t2):
    if not t1 or not t2:
        return 0
    A, B = set(t1), set(t2)
    return len(A & B) / max(len(A), len(B))


def fuzzy_match(tf_api, mapped_api, th=0.4):
    t1 = normalize(tf_api)
    t2 = normalize(mapped_api)

    # 规则 1：后缀一致
    if t1 and t2 and t1[-1] == t2[-1]:
        return True

    # 规则 2：token overlap
    if token_overlap(t1, t2) >= th:
        return True

    # 规则 3：字符串相似度
    if SequenceMatcher(None, tf_api.lower(), mapped_api.lower()).ratio() >= 0.55:
        return True

    return False


# -------------------------------
# Main
# -------------------------------
def main():
    f_pairs = Path("data/components/component_pairs.jsonl")
    f_test_usage = Path("data/mapping/tests_tf.mapped.jsonl")
    Path("data/migration").mkdir(parents=True, exist_ok=True)
    out_file = Path("data/migration/migration_candidates_fuzzy.jsonl")

    pairs = load_jsonl(f_pairs)
    print(f"[LOAD] API pairs = {len(pairs)}")

    # Build mapping list
    tf2pt = [(p["tf_api"], p["pt_api"], p["final_score"]) for p in pairs]

    tests = load_jsonl(f_test_usage)
    print(f"[LOAD] TF test API usage = {len(tests)}")

    results = []

    print("[MATCH] 开始模糊匹配测试用例与 API 映射...")

    for idx, t in enumerate(tqdm(tests)):
        used_apis = t.get("apis", [])
        
        # 注意：不再过滤，因为 parse_py.py 已经只提取 TensorFlow 相关的 API
        # used_apis 中应该都是 TensorFlow 相关的 API

        # ---- 修复：安全获取 name 和 file ----
        test_name = t.get("name", f"unknown_test_{idx}")
        test_file = t.get("file", "unknown_file.py")

        # 若两个关键字段均缺失，则跳过
        if not used_apis:
            continue
        
        # 过滤掉 unknown_test_* 和非测试文件
        # 这些通常是工具脚本、示例代码、测试数据生成脚本等，不是真正的测试
        if test_name.startswith("unknown_test_"):
            # 检查文件路径，如果是明显的非测试文件，跳过
            skip_paths = [
                "testing/op_tests/",  # 测试数据生成脚本
                "examples/",  # 示例代码
                "tools/",  # 工具脚本
                "compiler/mlir/tensorflow/tests/tf_saved_model/",  # 编译器测试数据
                "security/fuzzing/",  # 模糊测试
                "lite/testing/op_tests/",  # Lite 测试数据
                "lite/testing/",  # Lite 测试工具
            ]
            if any(skip_path in test_file for skip_path in skip_paths):
                continue
            # 如果文件路径包含这些关键词，也跳过
            skip_keywords = ["_fuzz.py", "_util.py", "_utils.py", "_helper.py", "_test_util.py"]
            if any(keyword in test_file for keyword in skip_keywords):
                continue

        matched = []

        for tf_api in used_apis:
            for mapped_tf, mapped_pt, score in tf2pt:
                if fuzzy_match(tf_api, mapped_tf):
                    matched.append({
                        "tf_api": tf_api,
                        "mapped_tf_api": mapped_tf,
                        "mapped_pt_api": mapped_pt,
                        "score": score
                    })

        if matched:
            results.append({
                "file": test_file,
                "name": test_name,
                "apis_used": used_apis,
                "matches": matched
            })

    # Write output
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n==== FUZZY MIGRATION SUMMARY ====")
    print(f"扫描到 TF 测试数量: {len(tests)}")
    print(f"可迁移（模糊匹配成功）的测试数量: {len(results)}")
    print(f"结果已输出到: {out_file}")


if __name__ == "__main__":
    main()
