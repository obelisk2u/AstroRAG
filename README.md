# AstroRAG: Transformer Reranking for Astrophysics Literature Search

AstroRAG is a retrieval pipeline for astrophysics papers that demonstrates how **transformer bi-encoders** (for fast dense retrieval) and **cross-encoders** (for precise reranking) can improve scientific literature search.  
The project is designed to be reproducible on the BU SCC cluster, and lightweight enough (~3 GB total) to run locally for demos.  

---

## Project Status  

- ✅ **Phase 1: Data Collection** — arXiv astro/physics abstracts downloaded to JSONL format.  
- ✅ **Phase 2: Passage Splitting & Indexing** — abstracts chunked into ~100-word passages, embedded with a bi-encoder, and stored in a FAISS index.  
- ⏳ **Phase 3: Baseline Retrieval Evaluation** — run keyword/title queries against the FAISS index to measure baseline retrieval.  
- ⏳ **Phase 4: Cross-Encoder Reranker** — fine-tune a transformer reranker (MiniLM/SciBERT) on weakly supervised (query, passage) pairs.  
- ⏳ **Phase 5: Evaluation** — compare retrieval vs reranking using NDCG/MRR on held-out queries.  
- ⏳ **Phase 6: Visualization & Packaging** — Jupyter notebooks with retrieval/rerank comparisons and qualitative examples.  

---

## Repo Structure  

```
AstroRAG/
├─ configs/          # YAML configs
├─ data/             # (gitignored) raw + processed data
│  ├─ raw/           # downloaded arXiv JSONL
│  └─ passages.jsonl # chunked passages (Phase 2 output)
├─ indexes/          # FAISS indexes + metadata
│  └─ faiss_base/
│     ├─ index.faiss   # dense vector index
│     ├─ meta.jsonl    # passage metadata
│     └─ model.txt     # embedding model used
├─ scripts/          # Python scripts for each step
├─ sge/              # SCC job submission scripts (SGE)
├─ outputs/          # (gitignored) trained model checkpoints
├─ logs/             # (gitignored) job logs
└─ README.md
```

---

## Environment Setup  

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Phase 1: Data Collection  

We use the [arXiv API](https://arxiv.org/help/api/) to fetch astrophysics and physics paper metadata.  
Each record includes: **id, title, abstract (summary), authors, categories, published date**.  

- **Categories pulled**:  
  `astro-ph.CO, astro-ph.GA, astro-ph.EP, astro-ph.HE, astro-ph.IM, astro-ph.SR, gr-qc, physics.comp-ph, physics.space-ph`

- **Script**: [`scripts/00_download_arxiv.py`](scripts/00_download_arxiv.py)  
  Implements polite, resume-safe harvesting with retries and skips empty pages.  

- **Run**:  
  ```bash
  python scripts/00_download_arxiv.py
  ```

- **Output**: `data/raw/arxiv_astro.jsonl`  

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

## Phase 2: Passage Splitting & Indexing  

- **Script 1 (chunking)**: [`scripts/10_chunk_passages.py`](scripts/10_chunk_passages.py)  
  Splits abstracts into overlapping ~100-word passages (title + text).  

- **Script 2 (embedding + indexing)**: [`scripts/20_embed_and_index.py`](scripts/20_embed_and_index.py)  
  Encodes passages with `all-MiniLM-L6-v2` bi-encoder and builds a FAISS index.  

- **Outputs** (under `indexes/faiss_base/`):  
  - `index.faiss` → dense vector index for retrieval  
  - `meta.jsonl` → metadata aligned with embeddings  
  - `model.txt` → name of embedding model  

---

## License  

MIT  
