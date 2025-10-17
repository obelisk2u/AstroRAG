#!/usr/bin/env python3
import argparse, json, os, re
from collections import defaultdict
from typing import Dict, List, Tuple

def norm_paper(x: str) -> str:
    return (x or "").replace("http://arxiv.org/abs/","").replace("https://arxiv.org/abs/","").replace("arXiv:","").strip()

def load_qrels(path: str) -> Dict[str, set]:
    qrels = defaultdict(set)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4: continue
            qid, _, docid, rel = parts
            if int(rel) > 0:
                qrels[qid].add(docid)
    return qrels

def load_run(path: str) -> Dict[str, List[Tuple[int, str, float]]]:
    run = defaultdict(list)
    with open(path) as f:
        for line in f:
            qid, _q0, docid, rank, score, _tag = line.split()
            run[qid].append((int(rank), docid, float(score)))
    for q in run:
        run[q].sort(key=lambda x: x[0])
    return run

def load_meta(meta_path: str) -> Dict[str, str]:
    """Return docid -> passage text, where docid is '{paper_id_version}:{chunk_id}'."""
    d2t = {}
    with open(meta_path) as f:
        for line in f:
            o = json.loads(line)
            pid = norm_paper(o.get("paper_id",""))
            cid = int(o.get("chunk_id"))
            docid = f"{pid}:{cid}"
            txt = o.get("passage","") or ""
            d2t[docid] = txt
    return d2t

def load_queries(qpath: str) -> Dict[str, str]:
    q = {}
    with open(qpath) as f:
        for line in f:
            o = json.loads(line)
            q[o["qid"]] = o["query"]
    return q

def main(a):
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    qrels = load_qrels(a.qrels)
    run   = load_run(a.run)
    d2t   = load_meta(a.meta)
    qtxt  = load_queries(a.queries)

    n_written = 0
    with open(a.out, "w") as w:
        for qid, ranked in run.items():
            relset = qrels.get(qid, set())
            if not relset: continue

            # pick one positive from the ranked list if present, else any relevant present in meta
            pos_doc = None
            for _r, did, _s in ranked:
                if did in relset and did in d2t:
                    pos_doc = did
                    break
            if pos_doc is None:
                # fallback: first relevant with text
                for did in relset:
                    if did in d2t:
                        pos_doc = did
                        break
            if pos_doc is None:  # no text for positives
                continue

            # hard negatives = top ranked non-relevant docs with text
            negs = []
            for _r, did, _s in ranked:
                if did not in relset and did in d2t:
                    negs.append(did)
                if len(negs) >= a.k_neg:
                    break
            if not negs:
                continue

            rec = {
                "qid": qid,
                "query": qtxt[qid],
                "pos": d2t[pos_doc],
                "negs": [d2t[d] for d in negs]
            }
            w.write(json.dumps(rec) + "\n")
            n_written += 1

    print(f"Wrote {n_written} rows to {a.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--run", required=True)          # e.g., outputs/runs/faiss_dev_top200.trec
    ap.add_argument("--meta", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)          # e.g., data/pairs/train_pairs.jsonl
    ap.add_argument("--k_neg", type=int, default=4)
    main(ap.parse_args())
