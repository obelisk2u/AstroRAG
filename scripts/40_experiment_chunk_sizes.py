import argparse, os, subprocess, json, csv, math
from pathlib import Path
from collections import defaultdict

SCRIPTS = Path(__file__).resolve().parent    
ROOT = SCRIPTS.parent                       
 
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
            obj = json.loads(line)
            q2paper[obj["qid"]] = obj["paper_id"]

    paper_to_docids = defaultdict(set)
    with open(meta_path) as f:
        for line in f:
            m = json.loads(line)
            pid_full = (m.get("paper_id") or "").strip()
            pid = (
                pid_full.replace("http://arxiv.org/abs/", "")
                .replace("https://arxiv.org/abs/", "")
                .replace("arXiv:", "")
            )
            docid = f"{pid}:{int(m['chunk_id'])}"
            paper_to_docids[pid].add(docid)

    qrels = {}
    for qid, pid in q2paper.items():
        qrels[qid] = paper_to_docids.get(pid, set())
    return qrels

def dcg_at_k(rels, k=10):
    s = 0.0
    for i, r in enumerate(rels[:k], start=1):
        if r:
            s += 1.0 / math.log2(i + 1)
    return s

def eval_run(run_by_q, qrels, k=10):
    ndcg, mrr, recall, n = 0.0, 0.0, 0.0, 0
    for qid, results in run_by_q.items():
        gold = qrels.get(qid, set())
        if not gold:
            continue
        n += 1
        rels = [1 if docid in gold else 0 for _, _, docid in results[:k]]

        dcg = dcg_at_k(rels, k)
        idcg = dcg_at_k([1] * min(k, len(gold)), k) or 1.0
        ndcg += dcg / idcg

        rr = next((1.0 / i for i, r in enumerate(rels, start=1) if r), 0.0)
        mrr += rr

        recall += sum(rels) / max(1, len(gold))

    if n == 0:
        return 0.0, 0.0, 0.0
    return ndcg / n, mrr / n, recall / n
 
def run(cmd):
    print("➤", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--queries", default="data/queries/dev.jsonl")
    ap.add_argument("--sizes", nargs="+", type=int, default=[50, 100, 200, 300])
    ap.add_argument("--overlap_frac", type=float, default=0.3)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--outdir", default="exp/chunk_sweep")
    args = ap.parse_args()

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure queries exist
    qpath = ROOT / args.queries
    if not qpath.exists():
        qpath.parent.mkdir(parents=True, exist_ok=True)
        run(["python", str(SCRIPTS / "30_build_queries.py"),
             "--raw_dir", str(ROOT / args.raw_dir),
             "--out", str(qpath),
             "--n", "500", "--seed", "42"])

    results = []
    for size in args.sizes:
        overlap = max(0, int(size * args.overlap_frac))
        exp_dir = outdir / f"chunk_{size}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        passages = exp_dir / "passages.jsonl"
        index_dir = exp_dir / "index"
        runfile = exp_dir / "faiss_top100.trec"
 
        run(["python", str(SCRIPTS / "10_chunk_passages.py"),
             "--raw_dir", str(ROOT / args.raw_dir),
             "--out", str(passages),
             "--chunk_size", str(size),
             "--overlap", str(overlap)])
 
        run(["python", str(SCRIPTS / "20_embed_and_index.py"),
             str(passages), str(index_dir),
             args.model, str(args.batch)])
 
        run(["python", str(SCRIPTS / "31_search_faiss.py"),
             "--index", str(index_dir / "index.faiss"),
             "--meta", str(index_dir / "meta.jsonl"),
             "--queries", str(qpath),
             "--out", str(runfile),
             "--topk", str(args.topk)])
 
        qrels = build_qrels_from_meta(qpath, index_dir / "meta.jsonl")
        run_by_q = parse_run(runfile)
        ndcg, mrr, rec = eval_run(run_by_q, qrels, k=10)

        results.append({
            "chunk_size": size,
            "overlap": overlap,
            "NDCG@10": round(ndcg, 4),
            "MRR@10": round(mrr, 4),
            "Recall@10": round(rec, 4),
        })
        print(f"[Chunk {size}] NDCG={ndcg:.4f}, MRR={mrr:.4f}, Recall={rec:.4f}")
 
    csv_path = outdir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_size","overlap","NDCG@10","MRR@10","Recall@10"])
        writer.writeheader()
        for row in sorted(results, key=lambda x: x["chunk_size"]):
            writer.writerow(row)

    print(f"\nResults saved to {csv_path}")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()