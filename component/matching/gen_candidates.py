# component/gen_candidates_fast.py
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

def load_jsonl(path):
    return [json.loads(x) for x in open(path)]

def worker(tf_block, pV, pt_list, topk, emb_th):
    """
    tf_block shape: [B, d]
    pV       shape: [PT, d]
    """
    # similarity matrix = tf_block @ pV.T
    sims = np.dot(tf_block, pV.T)  # shape: [B, PT]

    results = []
    B = sims.shape[0]
    PT = sims.shape[1]

    for i in range(B):
        row = sims[i]
        # top-k index
        idx = np.argpartition(row, -topk)[-topk:]
        pairs = [(row[j], j) for j in idx if row[j] >= emb_th]
        pairs.sort(reverse=True)

        results.append(pairs)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", default="data/tf_components.jsonl")
    parser.add_argument("--pt", default="data/pt_components.jsonl")
    parser.add_argument("--tf-vec", default="data/tf_vectors.npy")
    parser.add_argument("--pt-vec", default="data/pt_vectors.npy")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--emb-th", type=float, default=0.98)
    parser.add_argument("--block", type=int, default=512)
    parser.add_argument("--out", default="data/component_candidates.jsonl")
    parser.add_argument("--proc", type=int, default=8)
    args = parser.parse_args()

    tf_list = load_jsonl(args.tf)
    pt_list = load_jsonl(args.pt)

    tV = np.load(args.tf_vec)
    pV = np.load(args.pt_vec)

    Path(args.out).parent.mkdir(exist_ok=True)
    fout = open(args.out, "w", encoding="utf-8")

    N = len(tV)
    B = args.block

    pool = mp.Pool(args.proc)

    tasks = []
    for start in range(0, N, B):
        end = min(start + B, N)
        tf_block = tV[start:end]
        tasks.append(pool.apply_async(
            worker,
            (tf_block, pV, pt_list, args.topk, args.emb_th)
        ))

    for task_id, task in enumerate(tqdm(tasks)):
        res = task.get()  # list: for each tf-block row -> list of pairs
        block_start = task_id * B

        for i, pairs in enumerate(res):
            tf_idx = block_start + i
            tf_c = tf_list[tf_idx]

            for (sim, pt_idx) in pairs:
                pt_c = pt_list[pt_idx]
                obj = {
                    "tf_api": tf_c["api"],
                    "tf_sig": tf_c["sig"],
                    "pt_api": pt_c["api"],
                    "pt_sig": pt_c["sig"],
                    "emb_sim": float(sim),
                }
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    fout.close()
    pool.close()
    pool.join()
    print(f"[DONE] candidates saved to {args.out}")

if __name__ == "__main__":
    main()
