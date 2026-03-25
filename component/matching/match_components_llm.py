# ./component/match_components_llm.py
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from llm_utils import get_qwen_client
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def load_jsonl(path):
    if not Path(path).exists(): return []
    return [json.loads(x) for x in open(path)]

PROMPT = """
你是深度学习框架 API 匹配专家。
判断下面两个 API 是否为等价或相似组件：

TF: {tf_api}
Signature: {tf_sig}

PT: {pt_api}
Signature: {pt_sig}

只输出一个 0~1 的数字。
"""

def key_from(obj):
    return obj["tf_api"] + "@@" + obj["pt_api"]

def process_one(client, c, model):
    """处理单个候选对"""
    prompt = PROMPT.format(**c)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}]
        )
        raw = resp.choices[0].message.content.strip()
        score = float(raw)
    except Exception as e:
        score = 0
    
    c["llm_score"] = score
    c["final_score"] = 0.5*c["emb_sim"] + 0.5*score
    return c

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cand", default="data/component_candidates.jsonl")
    Path("data/components").mkdir(parents=True, exist_ok=True)
    parser.add_argument("--out", default="data/components/component_pairs.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default="qwen-flash")
    parser.add_argument("--workers", type=int, default=10, help="并发线程数")
    args = parser.parse_args()

    candidates = load_jsonl(args.cand)

    # load resume
    done = {}
    if args.resume and Path(args.out).exists():
        for x in load_jsonl(args.out):
            done[key_from(x)] = True
        print(f"[resume] 已完成 {len(done)} 条")

    # 过滤已完成的候选
    todo = [c for c in candidates if key_from(c) not in done]
    print(f"[INFO] 待处理: {len(todo)} 条，使用 {args.workers} 个并发线程")

    if not todo:
        print("[DONE] 所有候选已处理完成")
        return

    # 创建线程安全的文件写入
    fout = open(args.out, "a", encoding="utf-8")
    lock = threading.Lock()

    def write_result(result):
        with lock:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    # 为每个线程创建独立的客户端
    clients = [get_qwen_client("aliyun.key") for _ in range(args.workers)]

    def process_with_client(candidate, client_idx):
        client = clients[client_idx % len(clients)]
        return process_one(client, candidate, args.model)

    # 并发处理
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_with_client, c, i % args.workers): c 
            for i, c in enumerate(todo)
        }
        
        # 处理完成的任务
        for future in tqdm(as_completed(futures), total=len(futures), desc="Matching"):
            try:
                result = future.result()
                write_result(result)
            except Exception as e:
                c = futures[future]
                print(f"[ERROR] 处理失败 {key_from(c)}: {e}")
                # 写入失败结果
                c["llm_score"] = 0
                c["final_score"] = 0.5 * c["emb_sim"]
                write_result(c)

    fout.close()
    print(f"[DONE] wrote to {args.out}")

if __name__ == "__main__":
    main()
