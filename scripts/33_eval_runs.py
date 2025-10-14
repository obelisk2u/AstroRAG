import argparse, json
from collections import defaultdict
import math

def read_qrels(path):
    rels = defaultdict(set)
    with open(path) as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            if int(rel) > 0:
                rels[qid].add(docid)
    return rels

def read_run(path):
    run = defaultdict(list)
    with open(path) as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.split()
            run[qid].append((int(rank), docid, float(score)))
    for q in run: run[q].sort()
    return run

def ndcg_at_k(ranked, gold, k):
    dcg = 0.0
    for i, (_, d, _) in enumerate(ranked[:k], start=1):
        rel = 1.0 if d in gold else 0.0
        dcg += rel / math.log2(i+1)
    ideal = sum(1.0 / math.log2(i+1) for i in range(1, min(k, len(gold)) + 1))
    return dcg / ideal if ideal > 0 else 0.0

def mrr_at_k(ranked, gold, k):
    for i, (_, d, _) in enumerate(ranked[:k], start=1):
        if d in gold: return 1.0 / i
    return 0.0

def recall_at_k(ranked, gold, k):
    hit = sum(1 for _, d, _ in ranked[:k] if d in gold)
    return hit / max(1, len(gold))

def eval_one(run_path, qrels):
    run = read_run(run_path)
    qids = set(qrels.keys())
    metrics = {"NDCG@10":0,"MRR@10":0,"Recall@10":0}
    for q in qids:
        ranked = run.get(q, [])
        gold = qrels[q]
        metrics["NDCG@10"] += ndcg_at_k(ranked, gold, 10)
        metrics["MRR@10"] += mrr_at_k(ranked, gold, 10)
        metrics["Recall@10"] += recall_at_k(ranked, gold, 10)
    n = len(qids)
    for k in metrics: metrics[k] = round(metrics[k]/n, 4)
    return metrics

def main(args):
    qrels = read_qrels(args.qrels)
    out = {}
    for run in args.runs:
        out[run] = eval_one(run, qrels)
    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
