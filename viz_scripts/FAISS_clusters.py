import os, json, argparse, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

p = argparse.ArgumentParser()
p.add_argument("--data", default="../data/raw/arxiv_astro.jsonl",
               help="Path to .json or .jsonl with fields: title, summary/abstract, categories")
p.add_argument("--per_tag", type=int, default=400, help="balanced sample per tag")
p.add_argument("--min_per_tag", type=int, default=150, help="drop tags with fewer than this many rows")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default="faiss_embed_pca.png")
args = p.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = args.data if os.path.isabs(args.data) else os.path.join(SCRIPT_DIR, args.data)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

def load_records(path):
    recs = []
    if path.endswith(".jsonl"):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                recs.append(json.loads(line))
    elif path.endswith(".json"):
        with open(path, "r") as f:
            obj = json.load(f)
            recs = obj if isinstance(obj, list) else obj.get("data", [])
    else:
        raise ValueError("Expected .json or .jsonl")
    return recs

records = load_records(DATA_PATH)
df = pd.DataFrame(records)

# standardize fields
if "abstract" not in df.columns and "summary" in df.columns:
    df = df.rename(columns={"summary":"abstract"})

def normalize_cat(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        if "," in x:
            return [s.strip() for s in x.split(",") if s.strip()]
        return [x.strip()]
    return []

df["categories"] = df["categories"].apply(normalize_cat)
df = df.dropna(subset=["abstract", "title"])
df = df[df["abstract"].str.len() > 0]

def pick_primary(cats):
    if not cats: return None
    astro = [c for c in cats if c.startswith("astro-ph.")]
    return astro[0] if astro else cats[0]

df["primary_tag"] = df["categories"].apply(pick_primary)
df = df.dropna(subset=["primary_tag"])

keep = {
    "astro-ph.CO","astro-ph.GA","astro-ph.EP","astro-ph.HE",
    "astro-ph.IM","astro-ph.SR","gr-qc","physics.comp-ph","physics.space-ph"
}
df = df[df["primary_tag"].isin(keep)]

counts = Counter(df["primary_tag"])
big_tags = {t for t,c in counts.items() if c >= args.min_per_tag}
df = df[df["primary_tag"].isin(big_tags)].reset_index(drop=True)

parts = []
for t, g in df.groupby("primary_tag", sort=True):
    n = min(args.per_tag, len(g))
    parts.append(g.sample(n, random_state=args.seed))
dfb = pd.concat(parts).reset_index(drop=True)

texts = (dfb["title"].fillna("") + " " + dfb["abstract"].fillna("")).tolist()
labels = dfb["primary_tag"].tolist()

print("Using tags:", sorted(set(labels)))
print("Per-tag counts:", Counter(labels))

TAG_LABELS = {
    "astro-ph.CO": "Cosmology & Nongalactic Astrophysics",
    "astro-ph.GA": "Astrophysics of Galaxies",
    "astro-ph.EP": "Earth & Planetary Astrophysics",
    "astro-ph.HE": "High-Energy Astrophysics",
    "astro-ph.IM": "Instrumentation & Methods",
    "astro-ph.SR": "Solar & Stellar Astrophysics",
    "gr-qc":       "General Relativity & Quantum Cosmology",
    "physics.comp-ph": "Computational Physics",
    "physics.space-ph": "Space Physics"
}

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(texts, normalize_embeddings=True)

X2 = PCA(n_components=2, random_state=args.seed).fit_transform(emb)

plt.figure(figsize=(8,6), dpi=140)
plt.figure(figsize=(8,6), dpi=140)
for tag in sorted(set(labels)):
    idx = [i for i, t in enumerate(labels) if t == tag]
    display_label = TAG_LABELS.get(tag, tag)   # fallback to code if not in dict
    plt.scatter(X2[idx,0], X2[idx,1], s=8, alpha=0.7, label=display_label)

plt.title("AstroRAG embeddings colored by arXiv tag (MiniLM, PCA)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend(markerscale=2, fontsize=8, ncol=2, loc="best")
plt.tight_layout()
plt.savefig(args.out)
print(f"Saved plot to {args.out}")