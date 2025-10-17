#!/usr/bin/env python3
import argparse, json
import numpy as np, faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

def norm_paper(x: str) -> str:
    return (x or "").replace("http://arxiv.org/abs/","").replace("https://arxiv.org/abs/","").replace("arXiv:","").strip()

def load_meta(meta_path):
    ids, texts = [], []
    with open(meta_path) as f:
        for line in f:
            o = json.loads(line)
            pid = norm_paper(o.get("paper_id",""))
            cid = int(o.get("chunk_id"))
            ids.append(f"{pid}:{cid}")
            texts.append(o.get("passage","") or "")
    return ids, texts

def main(a):
    # FAISS stage
    index = faiss.read_index(a.index)
    biencoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    docids, passages = load_meta(a.meta)
    with open(a.queries) as qf:
        queries = [json.loads(l) for l in qf]

    # Cross-encoder reranker
    reranker = CrossEncoder(a.reranker)

    with open(a.out, "w") as outf:
        for q in queries:
            qid, qtext = q["qid"], q["query"]
            qemb = biencoder.encode([qtext], normalize_embeddings=True)
            D, I = index.search(np.asarray(qemb, dtype="float32"), a.faiss_topk)
            cand_ids = [docids[i] for i in I[0]]
            cand_txt = [passages[i] for i in I[0]]

            pairs = [[qtext, t] for t in cand_txt]
            scores = reranker.predict(pairs)  # higher is better
            reranked = sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)[:a.final_topk]

            for rank, (docid, score) in enumerate(reranked, start=1):
                outf.write(f"{qid} Q0 {docid} {rank} {float(score):.6f} faiss+ce\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--reranker", required=True)   # path to outputs/reranker/minilm_ce
    ap.add_argument("--out", required=True)
    ap.add_argument("--faiss_topk", type=int, default=200)
    ap.add_argument("--final_topk", type=int, default=10)
    main(ap.parse_args())
