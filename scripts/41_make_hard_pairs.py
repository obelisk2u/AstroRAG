#!/usr/bin/env python3
"""
Build hard-negative training pairs for the cross-encoder.

Inputs:
  --run     : TREC run from 31_search_faiss.py  (e.g., outputs/runs/faiss_dev.trec)
  --qrels   : TREC qrels from 34_make_qrels.py  (e.g., outputs/qrels/dev.qrels)
  --queries : JSONL with {"qid","query","paper_id"} (same file used by 31/34)
  --meta    : JSONL with {"paper_id","chunk_id","passage"| "text"} (same file used to build FAISS index)

Output (JSONL; one or more rows per query):
  {"qid": str, "query": str, "pos": str, "negs": [str, ...]}

Notes:
- Negatives are mined from the retriever's top-K (hard negatives = high-ranked but not in qrels).
- By default, we exclude negatives from the same paper_id as the query to avoid false negatives.
- You can either emit one row per query (single best positive) or one row per positive chunk.
"""

import argparse, json
from collections import defaultdict
from pathlib import Path

def norm_paper(x: str) -> str:
    return (x or "").replace("http://arxiv.org/abs/","").replace(
        "https://arxiv.org/abs/","").replace("arXiv:","").strip()

def load_queries(path):
    qtext, qpaper = {}, {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            qid = o["qid"]
            qtext[qid] = o.get("query", "")
            qpaper[qid] = norm_paper(o.get("paper_id", ""))
    return qtext, qpaper

def load_meta(meta_path):
    """Return:
       docid -> text,
       docid -> paper_id,
       paper_id -> [docid,...] (for positives lookup if needed)
    """
    dtext, dpaper, paper2docids = {}, {}, defaultdict(list)
    with open(meta_path) as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            paper = norm_paper(o.get("paper_id", ""))
            cid = o.get("chunk_id")
            if paper == "" or cid is None:
                continue
            docid = f"{paper}:{int(cid)}"
            text = o.get("passage") or o.get("text") or ""
            dtext[docid] = text
            dpaper[docid] = paper
            paper2docids[paper].append(docid)
    return dtext, dpaper, paper2docids

def load_qrels_trec(path):
    """Return qrels as: qid -> set(docid) for label>0."""
    qrels = defaultdict(set)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            # TREC qrels: qid 0 docid label
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, label = parts[0], parts[1], parts[2], parts[3]
            try:
                if int(label) > 0:
                    qrels[qid].add(docid)
            except Exception:
                continue
    return qrels

def load_run_trec(path, topk):
    """Return run as: qid -> [docid,...] limited to topk (keeps original rank order)."""
    run = defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            # TREC run: qid Q0 docid rank score tag
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, docid, rank = parts[0], parts[2], int(parts[3])
            if rank <= topk:
                run[qid].append((rank, docid))
    # sort by rank just in case
    run = {qid: [d for _r, d in sorted(items, key=lambda x: x[0])] for qid, items in run.items()}
    return run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="outputs/runs/faiss_dev.trec")
    ap.add_argument("--qrels", required=True, help="outputs/qrels/dev.qrels")
    ap.add_argument("--queries", required=True, help="data/queries/dev.jsonl")
    ap.add_argument("--meta", required=True, help="indexes/faiss_base/meta.jsonl")
    ap.add_argument("--out", required=True, help="outputs/pairs/hard_pairs.jsonl")
    ap.add_argument("--topk", type=int, default=100, help="mine negatives from top-K retriever results")
    ap.add_argument("--negs_per_row", type=int, default=4, help="cap negatives per training row")
    ap.add_argument("--exclude_same_paper", action="store_true", default=True,
                    help="skip negatives from the same paper_id as the query")
    ap.add_argument("--rows_per_query", choices=["one", "per_positive"], default="one",
                    help="emit one row per query (choose a primary positive) or one row per positive chunk")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Load inputs
    qtext, qpaper = load_queries(args.queries)
    dtext, dpaper, _paper2docids = load_meta(args.meta)
    qrels = load_qrels_trec(args.qrels)
    run = load_run_trec(args.run, args.topk)

    # Mine rows
    written, skipped_missing_pos, skipped_missing_neg = 0, 0, 0
    with open(args.out, "w") as w:
        for qid, cand_docids in run.items():
            if qid not in qtext:
                continue
            positives = list(qrels.get(qid, []))
            if not positives:
                continue

            # Build candidate negatives from top-K results
            neg_pool = []
            for docid in cand_docids:
                if docid in qrels.get(qid, set()):
                    continue
                if args.exclude_same_paper:
                    q_paper = qpaper.get(qid, "")
                    d_paper = dpaper.get(docid, "")
                    if q_paper and d_paper and q_paper == d_paper:
                        continue
                if docid in dtext:
                    neg_pool.append(docid)

            # cap negatives per row
            neg_texts = [dtext[d] for d in neg_pool[:args.negs_per_row]]

            if args.rows_per_query == "one":
                # Choose a primary positive (e.g., lowest chunk_id if available)
                def chunk_id(docid):
                    try:
                        return int(docid.split(":")[1])
                    except Exception:
                        return 1_000_000
                positives.sort(key=chunk_id)
                pos_id = positives[0]
                if pos_id not in dtext:
                    skipped_missing_pos += 1
                    continue
                row = {
                    "qid": qid,
                    "query": qtext[qid],
                    "pos": dtext[pos_id],
                    "negs": neg_texts
                }
                w.write(json.dumps(row) + "\n")
                written += 1
            else:
                # one row per positive chunk
                for pos_id in positives:
                    if pos_id not in dtext:
                        skipped_missing_pos += 1
                        continue
                    row = {
                        "qid": qid,
                        "query": qtext[qid],
                        "pos": dtext[pos_id],
                        "negs": neg_texts
                    }
                    w.write(json.dumps(row) + "\n")
                    written += 1

    print(f"[hard_pairs] wrote {written} rows → {args.out}")
    print(f"[hard_pairs] skipped_missing_pos={skipped_missing_pos} (positive text missing in meta)")

if __name__ == "__main__":
    main()