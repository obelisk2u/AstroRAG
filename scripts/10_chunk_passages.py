import argparse, json, os, glob, re
from typing import List

def clean_text(s: str) -> str:
    if not s: return ""
    # strip LaTeX-y bits and excessive whitespace (light touch)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def chunk_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words: return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        w = words[start:start + chunk_size]
        if not w: break
        chunks.append(" ".join(w))
        if start + chunk_size >= len(words):
            break
    return chunks

def iter_raw(raw_dir_glob: str):
    for p in glob.glob(os.path.join(raw_dir_glob, "*.jsonl")):
        with open(p, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    yield obj
                except Exception:
                    continue

def main(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n_written = 0
    with open(args.out, "w") as w:
        for rec in iter_raw(args.raw_dir):
            # Inputs from Phase 1
            paper_url = rec.get("id", "")             # e.g., "http://arxiv.org/abs/2509.26611v1"
            title     = rec.get("title", "")          # keep only for metadata/display
            abstract  = rec.get("summary", "")        # abstract text (a.k.a. "content" we have)
            cats      = rec.get("categories", None)

            # --- CRITICAL CHANGE: do NOT prepend title to the passage text ---
            text = clean_text(abstract)
            if not paper_url or not text:
                continue

            chunks = chunk_words(text, args.chunk_size, args.overlap)
            for i, ch in enumerate(chunks):
                out_obj = {
                    # keep schema aligned with your meta.jsonl downstream
                    "paper_id": paper_url,        # URL form; embedding step will normalize if needed
                    "title": title,               # for display only (NOT embedded)
                    "chunk_id": i,
                    "passage": ch,
                    "category": cats if cats is not None else None
                }
                w.write(json.dumps(out_obj) + "\n")
                n_written += 1

    print(f"Wrote {n_written} passage chunks to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="e.g., data/raw")
    ap.add_argument("--out", default="data/passages.jsonl", help="output JSONL of abstract-only chunks")
    ap.add_argument("--chunk_size", type=int, default=100, help="words per chunk (default: 100)")
    ap.add_argument("--overlap", type=int, default=30, help="word overlap between chunks (default: 30)")
    main(ap.parse_args())
