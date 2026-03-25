# ./component/embed_components.py
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

MAX_BATCH = 32   # bge-large 支持更大 batch，可视显存调整

def load_jsonl(path):
    return [json.loads(line) for line in open(path)]

def embed_local(model, texts, batch=MAX_BATCH):
    vecs = []
    for i in tqdm(range(0, len(texts), batch), desc="Embedding", unit="batch"):
        sub = texts[i:i+batch]
        emb = model.encode(
            sub,
            batch_size=batch,
            show_progress_bar=False,
            normalize_embeddings=True     # 建议归一化，便于后续相似度计算
        )
        vecs.append(emb)
    return np.vstack(vecs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", type=str, default="data/tf_components.jsonl")
    parser.add_argument("--pt", type=str, default="data/pt_components.jsonl")
    parser.add_argument("--model-dir", type=str, default="/Users/linzheyuan/code/TransTest/models/bge-large-en-v1.5")
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model = SentenceTransformer(args.model_dir, device="cpu")

    tf = load_jsonl(args.tf)
    pt = load_jsonl(args.pt)

    tf_texts = [f"{c['api']} {c['sig']}" for c in tf]
    pt_texts = [f"{c['api']} {c['sig']}" for c in pt]

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    tf_vec_path = out / "tf_vectors.npy"
    pt_vec_path = out / "pt_vectors.npy"

    if args.resume and tf_vec_path.exists():
        print("[RESUME] 发现 tf_vectors.npy，跳过 TF embedding")
        tf_vec = np.load(tf_vec_path)
    else:
        print("[EMBED] TF components =", len(tf_texts))
        tf_vec = embed_local(model, tf_texts)
        np.save(tf_vec_path, tf_vec)

    if args.resume and pt_vec_path.exists():
        print("[RESUME] 发现 pt_vectors.npy，跳过 PT embedding")
        pt_vec = np.load(pt_vec_path)
    else:
        print("[EMBED] PT components =", len(pt_texts))
        pt_vec = embed_local(model, pt_texts)
        np.save(pt_vec_path, pt_vec)

    print(f"[DONE] tf={tf_vec.shape}, pt={pt_vec.shape}")

if __name__ == "__main__":
    main()
