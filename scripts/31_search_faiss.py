import json, argparse, faiss
from sentence_transformers import SentenceTransformer
import numpy as np

def norm_paper(x: str) -> str:
    if not x: return ""
    x = x.strip()
    x = x.replace("http://arxiv.org/abs/", "").replace("https://arxiv.org/abs/", "")
    x = x.replace("arXiv:", "")
    return x

def load_meta(meta_path):
    docids, texts = [], []
    with open(meta_path) as f:
        for line in f:
            obj = json.loads(line)
            paper = norm_paper(obj.get("paper_id", ""))
            chunk_id = obj.get("chunk_id")
            text = obj.get("passage") or obj.get("text") or ""
            if paper == "" or chunk_id is None:
                continue
            docid = f"{paper}:{int(chunk_id)}"
            docids.append(docid)
            texts.append(text)
    return docids, texts

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
