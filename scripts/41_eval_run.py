#!/usr/bin/env python3
import argparse, json, math
from collections import defaultdict

def parse_run(run_path):
    by_q = defaultdict(list)
    with open(run_path) as f:
        for line in f:
            qid, _q0, docid, rank, score, _tag = line.strip().split()
            by_q[qid].append((int(rank), float(score), docid))
    for qid in by_q:
        by_q[qid].sort(key=lambda x: x[0])
    return by_q

def build_qrels_from_meta(queries_path, meta_path):
    q2paper = {}
    with open(queries_path) as f:
        for line in f:
            o = json.loads(line); q2paper[o["qid"]] = o["paper_id"]
    paper2docids = defaultdict(set)
    with open(meta_path) as f:
        for line in f:
            m = json.loads(line)
            pid = (m.get("paper_id","").replace("http://arxiv.org/abs/","")
                                  .replace("https://arxiv.org/abs/","")
                                  .replace("arXiv:",""))
            docid = f"{pid}:{int(m['chunk_id'])}"
            paper2docids[pid].add(docid)
    return {qid: paper2docids.get(pid, set()) for qid, pid in q2paper.items()}

def dcg_at_k(rels, k=10):
    s = 0.0
    for i, r in enumerate(rels[:k], start=1):
        if r: s += 1.0 / math.log2(i + 1)
    return s

def eval_metrics(run_by_q, qrels, k=10):
    ndcg=mrr=rec=n=0.0
    for qid, results in run_by_q.items():
        gold = qrels.get(qid, set())
        if not gold: continue
        n += 1
        rels = [1 if docid in gold else 0 for _,_,docid in results[:k]]

        dcg = dcg_at_k(rels, k)
        idcg = dcg_at_k([1]*min(k,len(gold)), k) or 1.0
        ndcg += dcg/idcg

        rr = next((1.0/i for i,r in enumerate(rels, start=1) if r), 0.0)
        mrr += rr

        rec += sum(rels)/max(1,len(gold))
    if n==0: return 0,0,0
    return ndcg/n, mrr/n, rec/n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    run_by_q = parse_run(args.run)
    qrels = build_qrels_from_meta(args.queries, args.meta)
    ndcg, mrr, recall = eval_metrics(run_by_q, qrels, k=args.k)
    print(f"NDCG@{args.k}={ndcg:.4f}  MRR@{args.k}={mrr:.4f}  Recall@{args.k}={recall:.4f}")

if __name__ == "__main__":
    main()