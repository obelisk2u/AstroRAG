import argparse, json
from pathlib import Path
from collections import defaultdict
import torch
from sentence_transformers.cross_encoder import CrossEncoder

def load_queries(p):
    q = {}
    with open(p) as f:
        for line in f:
            obj = json.loads(line)
            q[obj["qid"]] = obj["query"]
    return q

def load_meta_docid2text(meta_path):
    m = {}
    with open(meta_path) as f:
        for line in f:
            o = json.loads(line)
            pid = (o.get("paper_id","").replace("http://arxiv.org/abs/","")
                                   .replace("https://arxiv.org/abs/","")
                                   .replace("arXiv:",""))
            docid = f"{pid}:{int(o['chunk_id'])}"
            m[docid] = o.get("passage","")
    return m

def parse_trec(run_path):
    by_q = defaultdict(list)
    with open(run_path) as f:
        for line in f:
            qid, _q0, docid, rank, score, tag = line.strip().split()
            by_q[qid].append((int(rank), float(score), docid))
    for qid in by_q:
        by_q[qid].sort(key=lambda x: x[0])  # rank asc
    return by_q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_run", required=True)          # e.g. exp/chunk_300/faiss_top100.trec
    ap.add_argument("--meta", required=True)            # e.g. exp/chunk_300/index/meta.jsonl
    ap.add_argument("--queries", required=True)         # e.g. data/queries/dev.jsonl
    ap.add_argument("--out", required=True)             # e.g. exp/chunk_300/ce_top100.trec
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    queries = load_queries(args.queries)
    docid2text = load_meta_docid2text(args.meta)
    run_by_q = parse_trec(args.in_run)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder(args.model, device=device)

    with open(args.out, "w") as outf:
        for qid, results in run_by_q.items():
            q = queries.get(qid)
            if not q:
                continue
            docids = [d for _, _, d in results]
            pairs = [(q, docid2text.get(d, "")) for d in docids]
            scores = ce.predict(pairs, batch_size=args.batch)  # higher = more relevant
            order = sorted(zip(docids, scores), key=lambda x: x[1], reverse=True)

            for rank, (docid, score) in enumerate(order, start=1):
                outf.write(f"{qid} Q0 {docid} {rank} {float(score):.6f} ce\n")

    print(f"✅ Wrote reranked run to {args.out}")

if __name__ == "__main__":
    main()