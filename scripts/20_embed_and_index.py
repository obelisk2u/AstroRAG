import json, sys, numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

IN = Path(sys.argv[1])       # data/passages.jsonl
INDEX_DIR = Path(sys.argv[2])# indexes/faiss_base
MODEL = sys.argv[3] if len(sys.argv)>3 else "sentence-transformers/all-MiniLM-L6-v2"
BATCH = int(sys.argv[4]) if len(sys.argv)>4 else 512

INDEX_DIR.mkdir(parents=True, exist_ok=True)
meta_out = (INDEX_DIR / "meta.jsonl").open("w")

model = SentenceTransformer(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

# collect passages & map row->metadata
passages = []
with IN.open() as fin:
    for line in fin:
        rec = json.loads(line)
        passages.append(rec)

# embed
texts = [p["passage"] for p in passages]
embs = model.encode(texts, batch_size=BATCH, show_progress_bar=True, normalize_embeddings=True)
embs = np.asarray(embs, dtype="float32")

# FAISS IndexFlatIP (cosine via normalized vectors)
dim = embs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embs)

faiss.write_index(index, str(INDEX_DIR / "index.faiss"))

# save metadata
for p in passages:
    meta_out.write(json.dumps(p) + "\n")
meta_out.close()

# save model name for reproducibility
(INDEX_DIR / "model.txt").write_text(MODEL + "\n")
print(f"Indexed {len(passages)} passages with dim={dim} using {MODEL}.")

