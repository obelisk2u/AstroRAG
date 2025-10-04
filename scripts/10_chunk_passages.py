import json, re, sys
from pathlib import Path

IN = Path(sys.argv[1])    # e.g., data/arxiv_raw.jsonl
OUT = Path(sys.argv[2])   # e.g., data/passages.jsonl
TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 120
STRIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 100

def clean(t):
    return re.sub(r'\s+', ' ', t).strip()

def chunk_words(words, tokens=120, stride=100):
    i = 0
    while i < len(words):
        yield words[i:i+tokens]
        if i + tokens >= len(words):
            break
        i += stride

with IN.open() as fin, OUT.open("w") as fout:
    for line in fin:
        rec = json.loads(line)
        pid = rec.get("id") or rec.get("paper_id")
        title = clean(rec.get("title", ""))
        # Be robust to field name differences
        abstract = clean(rec.get("abstract") or rec.get("summary") or "")
        if not abstract:
            continue
        words = abstract.split()
        for j, seg in enumerate(chunk_words(words, TOKENS, STRIDE)):
            passage = (f"{title}. " if title else "") + " ".join(seg)
            out = {
                "paper_id": pid,
                "title": title,
                "chunk_id": j,
                "passage": passage,
                "category": rec.get("category")
            }
            fout.write(json.dumps(out) + "\n")

