import argparse, json, glob, random, os

def iter_raw(path_glob):
    for p in glob.glob(path_glob):
        with open(p) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    yield obj
                except:
                    continue

def main(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = random.Random(args.seed)
    rows = [r for r in iter_raw(os.path.join(args.raw_dir, "*.jsonl"))]
    rng.shuffle(rows)
    rows = rows[:args.n]

    with open(args.out, "w") as w:
        for i, r in enumerate(rows):
            qid = f"q{i:05d}"
            title = (r.get("title") or "").strip()
            paper_id = (r.get("id") or "").split("/")[-1]  # e.g., 2501.01234v1
            if not title or not paper_id:
                continue
            w.write(json.dumps({"qid": qid, "query": title, "paper_id": paper_id}) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="e.g., data/raw")
    ap.add_argument("--out", required=True, help="e.g., data/queries/dev.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
