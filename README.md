# **AstroRAG: Transformer Reranking for Astrophysics Literature Search**

AstroRAG is a retrieval pipeline for astrophysics papers that demonstrates how **transformer bi-encoders** (for fast dense retrieval) and **cross-encoders** (for precise reranking) can improve scientific literature search.  
The project is designed to be reproducible on the **BU SCC** cluster and lightweight enough (~3 GB total) to run locally for demos.

---

## 🚀 Project Status

| Phase | Description | Status |
|:------|:-------------|:-------|
| **1. Data Collection** | Harvest astrophysics/physics abstracts from arXiv into JSONL. | ✅ Completed |
| **2. Passage Splitting & Indexing** | Chunk abstracts into overlapping ≈ 100-word passages, embed with MiniLM, and build a FAISS index. | ✅ Completed (293 k passages × 384 dims) |
| **3. Baseline Retrieval Evaluation** | Run title-based queries against FAISS to measure dense retrieval quality (NDCG/MRR/Recall). | ✅ Completed → NDCG@10 = 0.49 · MRR@10 = 0.63 · Recall@10 = 0.53 |
| **4. Cross-Encoder Reranker** | Fine-tune a transformer (MiniLM / SciBERT) on weakly-supervised (query, passage) pairs. | ⏳ In progress |
| **5. Evaluation** | Compare retrieval vs reranking using NDCG/MRR on held-out queries. | ⏳ Pending |
| **6. Visualization & Packaging** | Jupyter notebooks with quantitative plots and qualitative examples. | ⏳ Pending |

---

## 🗂 Repo Structure

```
AstroRAG/
├─ configs/          # YAML configs
├─ data/             # (gitignored) raw + processed data
│  ├─ raw/           # downloaded arXiv JSONL
│  ├─ passages.jsonl # chunked passages (Phase 2 output)
│  └─ queries/       # generated title-based queries
├─ indexes/          # FAISS indexes + metadata
│  └─ faiss_base/
│     ├─ index.faiss   # dense vector index
│     ├─ meta.jsonl    # passage metadata
│     └─ model.txt     # embedding model used
├─ outputs/          # (gitignored) models, runs, metrics
│  ├─ qrels/         # relevance labels
│  ├─ runs/          # retrieval outputs (TREC format)
│  └─ metrics/       # evaluation JSONs
├─ scripts/          # Python scripts for each phase
├─ sge/              # SCC GPU/CPU job scripts
├─ logs/             # (gitignored) cluster logs
└─ README.md
```

---

## ⚙️ Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On BU SCC (GPU nodes):
```bash
module load pytorch/1.13.1
export HF_HOME=/projectnb/aisearch/$USER/.cache/hf
```

---

## 🌌 Phase 1: Data Collection

Fetch astrophysics and physics paper metadata from the [arXiv API](https://arxiv.org/help/api/).  
Each record includes **id**, **title**, **abstract**, **authors**, **categories**, and **published date**.

- **Categories pulled:**  
  `astro-ph.CO, astro-ph.GA, astro-ph.EP, astro-ph.HE, astro-ph.IM, astro-ph.SR, gr-qc, physics.comp-ph, physics.space-ph`

- **Script:** [`scripts/00_download_arxiv.py`](scripts/00_download_arxiv.py) — polite, resume-safe harvesting with retries.

- **Run:**
  ```bash
  python scripts/00_download_arxiv.py
  ```
- **Output:** `data/raw/arxiv_astro.jsonl`

Example:
```json
{
  "id": "http://arxiv.org/abs/2501.01234v1",
  "title": "Dark matter halos in dwarf galaxies",
  "summary": "We present new results on...",
  "authors": ["A. Author", "B. Author"],
  "categories": ["astro-ph.GA"],
  "published": "2025-01-15T12:00:00Z"
}
```

---

## 🧩 Phase 2: Passage Splitting & Indexing

1. **Chunk abstracts:** [`scripts/10_chunk_passages.py`](scripts/10_chunk_passages.py)  
   → splits abstracts into overlapping ~ 100-word passages (title + text).

2. **Embed & index:** [`scripts/20_embed_and_index.py`](scripts/20_embed_and_index.py)  
   → encodes passages with `sentence-transformers/all-MiniLM-L6-v2` and builds a FAISS `IndexFlatIP`.

- **Execution (on GPU):**
  ```bash
  qsub sge/embed_index_gpu.sge
  ```
- **Result:**
  ```
  Indexed 293,969 passages with dim=384 using MiniLM-L6-v2
  ```

- **Outputs (`indexes/faiss_base/`):**
  - `index.faiss` — dense vector index  
  - `meta.jsonl` — metadata aligned with embeddings  
  - `model.txt` — embedding model used

---

## 📊 Phase 3: Baseline Retrieval Evaluation

Evaluate dense retrieval quality using title-only queries.

```bash
# generate queries
python scripts/30_build_queries.py   --raw_dir data/raw   --out data/queries/dev.jsonl --n 200

# build qrels
python scripts/34_make_qrels.py   --queries data/queries/dev.jsonl   --meta indexes/faiss_base/meta.jsonl   --out outputs/qrels/dev.qrels

# FAISS retrieval
python scripts/31_search_faiss.py   --index indexes/faiss_base/index.faiss   --meta indexes/faiss_base/meta.jsonl   --queries data/queries/dev.jsonl   --out outputs/runs/faiss_dev.trec --topk 100

# evaluation
python scripts/33_eval_runs.py   --qrels outputs/qrels/dev.qrels   --runs outputs/runs/faiss_dev.trec   --out outputs/metrics/dev_faiss.json
```

**Baseline Results (MiniLM dense retriever):**
```json
{
  "NDCG@10": 0.4905,
  "MRR@10": 0.6276,
  "Recall@10": 0.5272
}
```

---

## 📚 Next Steps (Phase 4–6)

- **Phase 4:** Generate weakly supervised (query, positive, negative) pairs and train a cross-encoder reranker (`MiniLM`, `SciBERT`).  
- **Phase 5:** Evaluate reranking performance vs FAISS baseline using NDCG/MRR.  
- **Phase 6:** Add Jupyter notebooks with side-by-side retrieval vs rerank visualization and qualitative paper examples.

---

## 🧠 Key Concepts

- **Bi-Encoder:** encodes queries and documents separately for fast approximate similarity search (FAISS).  
- **Cross-Encoder:** jointly encodes query + document pairs for higher-precision reranking.  
- **Metrics:** NDCG @ 10, MRR @ 10, Recall @ 10 quantify retrieval relevance and ordering quality.

---

## 🪐 License
MIT
