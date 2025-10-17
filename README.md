# AstroRAG: Transformer Reranking for Astrophysics Literature Search

AstroRAG is a retrieval pipeline for astrophysics papers that demonstrates how **transformer bi-encoders** (for fast dense retrieval) and **cross-encoders** (for precise reranking) can improve scientific literature search.  
The project is reproducible on the BU SCC cluster and lightweight enough (~3 GB) to run locally for demos.

---

## Project Status

- ✅ **Phase 1: Data Collection** — arXiv astrophysics/physics abstracts downloaded to JSONL format.  
- ✅ **Phase 2: Passage Splitting & Indexing** — abstracts chunked into ~100-word passages, embedded with a bi-encoder, and stored in a FAISS index.  
- ✅ **Phase 3: Baseline Retrieval Evaluation** — FAISS dense retrieval run on title-based queries.  
- ✅ **Phase 4: Cross-Encoder Reranking** — MiniLM reranker fine-tuned on weak (query, passage) pairs.  
- ⏳ **Phase 5: Extended Evaluation** — analyze per-query NDCG/MRR deltas and visualize top passages.  
- ⏳ **Phase 6: Packaging & Visualization** — clean notebooks and qualitative examples.

---

## Key Results (Phase 4)

| Metric | Dense FAISS | Cross-Encoder Reranker | Δ (Improvement) |
|:-------|-------------:|-----------------------:|----------------:|
| **NDCG@10** | 0.4905 | **0.5949** | +20 % |
| **MRR@10** | 0.6276 | **0.8025** | +27 % |
| **Recall@10** | 0.5272 | **0.5872** | +11 % |

> The reranker substantially improves ranking precision, confirming that transformer-based cross-encoding yields more relevant literature retrieval in astrophysics.

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
│     ├─ index.faiss
│     ├─ meta.jsonl
│     └─ model.txt
├─ scripts/          # Python scripts for each step
├─ sge/              # SCC job scripts
├─ outputs/          # (gitignored) trained models + metrics
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

## License

MIT
