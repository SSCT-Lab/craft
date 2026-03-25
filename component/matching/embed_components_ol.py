# ./component/embed_components.py
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from llm_utils import get_qwen_client

MAX_BATCH = 25   # DashScope embedding 硬限制

def load_jsonl(path):
    return [json.loads(line) for line in open(path)]

def chunked(iterable, size=25):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

def embed_texts(client, texts, model="text-embedding-v1"):
    """
    真正按 25 条一批调用，并显示进度条。
    """
    vecs = []
    total = len(texts)

    for batch in tqdm(chunked(texts, MAX_BATCH), total=(total + MAX_BATCH - 1)//MAX_BATCH, desc="Embedding"):
        resp = client.embeddings.create(
            model=model,
            input=batch
        )
        for d in resp.data:
            vecs.append(d.embedding)

    return np.array(vecs, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", type=str, default="data/tf_components.jsonl")
    parser.add_argument("--pt", type=str, default="data/pt_components.jsonl")
    parser.add_argument("--model", type=str, default="text-embedding-v1")
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑，存在同名 .npy 则跳过")
    args = parser.parse_args()

    client = get_qwen_client("aliyun.key")

    tf = load_jsonl(args.tf)
    pt = load_jsonl(args.pt)

    tf_texts = [f"{c['api']} {c['sig']}" for c in tf]
    pt_texts = [f"{c['api']} {c['sig']}" for c in pt]

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    tf_vec_path = out / "tf_vectors.npy"
    pt_vec_path = out / "pt_vectors.npy"

    # 支持断点续跑
    if args.resume and tf_vec_path.exists():
        print("[RESUME] 发现 tf_vectors.npy，跳过 TF embedding")
        tf_vec = np.load(tf_vec_path)
    else:
        print("[EMBED] 开始 TF embedding ...")
        tf_vec = embed_texts(client, tf_texts, model=args.model)
        np.save(tf_vec_path, tf_vec)

    if args.resume and pt_vec_path.exists():
        print("[RESUME] 发现 pt_vectors.npy，跳过 PT embedding")
        pt_vec = np.load(pt_vec_path)
    else:
        print("[EMBED] 开始 PT embedding ...")
        pt_vec = embed_texts(client, pt_texts, model=args.model)
        np.save(pt_vec_path, pt_vec)

    print(f"[DONE] tf={tf_vec.shape}, pt={pt_vec.shape}")


if __name__ == "__main__":
    main()
