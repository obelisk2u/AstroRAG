import argparse, json, os

def norm_paper(x: str) -> str:
    if not x: return ""
    x = x.strip()
    x = x.replace("http://arxiv.org/abs/", "").replace("https://arxiv.org/abs/", "")
    x = x.replace("arXiv:", "")
    return x

def load_queries(qpath):
    q = {}
    with open(qpath) as f:
        for line in f:
            obj = json.loads(line)
            # queries already look like "2502.12248v3"
            q[obj["qid"]] = obj["paper_id"].strip()
    return q

def main(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    qid2paper = load_queries(args.queries)

    # Build paper -> [docids] where docid = "{paper}:{chunk_id}"
    paper2docids = {}
    with open(args.meta) as f:
        for line in f:
            m = json.loads(line)
            paper = norm_paper(m.get("paper_id", ""))
            chunk_id = m.get("chunk_id")
            if paper == "" or chunk_id is None:
                continue
            docid = f"{paper}:{int(chunk_id)}"
            paper2docids.setdefault(paper, []).append(docid)

    written = 0
    with open(args.out, "w") as w:
        for qid, paper in qid2paper.items():
            for docid in paper2docids.get(paper, []):
                w.write(f"{qid} 0 {docid} 1\n")
                written += 1
    print(f"Wrote {written} lines to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)  # data/queries/dev.jsonl
    ap.add_argument("--meta", required=True)     # indexes/faiss_base/meta.jsonl
    ap.add_argument("--out", required=True)      # outputs/qrels/dev.qrels
    main(ap.parse_args())
