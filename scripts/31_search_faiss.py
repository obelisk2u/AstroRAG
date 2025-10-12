import json, argparse, faiss
from sentence_transformers import SentenceTransformer
import numpy as np

def load_meta(meta_path):
    ids, texts = [], []
    with open(meta_path) as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["pid"])      
            texts.append(obj["text"])       
    return ids, texts

def main(args):
    index = faiss.read_index(args.index)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    ids, _ = load_meta(args.meta)
    with open(args.queries) as qf, open(args.out, "w") as outf:
        for line in qf:
            q = json.loads(line)
            emb = model.encode([q["query"]], normalize_embeddings=True)
            D, I = index.search(np.array(emb, dtype=np.float32), args.topk)
            for rank, (sid, score) in enumerate(zip(I[0], D[0]), start=1):
                outf.write(f'{q["qid"]} Q0 {ids[sid]} {rank} {float(score):.6f} faiss\n')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=100)
    main(ap.parse_args())
