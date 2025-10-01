# AstroReRank: Transformer Reranking for Astrophysics Literature Search

AstroReRank is a retrieval pipeline for astrophysics papers that demonstrates how **transformer bi-encoders** (for fast retrieval) and **cross-encoders** (for precise reranking) can improve scientific literature search.  
The project is designed to be fully reproducible on the BU SCC cluster, and lightweight enough (~3 GB total) to run locally for demos.

---

## Project Status

- **Phase 1: Data Collection** (arXiv astro/physics abstracts downloaded, JSONL format)
- Phase 2: Data preprocessing & pair generation
- Phase 3: Bi-encoder training & FAISS index
- Phase 4: Cross-encoder reranker
- Phase 5: Evaluation (Recall@n, nDCG)
- Phase 6: Notebooks & qualitative demos

---

## Repo Structure

```
astro-rerank/
├─ configs/          # YAML configs
├─ data/             # (gitignored) raw + processed data
├─ scripts/          # Python scripts for each step
├─ outputs/          # (gitignored) trained model checkpoints
├─ slurm/            # SCC batch scripts
├─ env.yaml          # Conda environment
├─ Makefile          # Workflow commands
└─ README.md
```

---

## Environment Setup

```bash
conda env create -f env.yaml
conda activate astro-rerank
```

---

## Phase 1: Data Collection

We use the [arXiv API](https://arxiv.org/help/api/) to fetch astrophysics and physics paper metadata.  
Each record includes: **id, title, abstract (summary), authors, categories, published date**.

- **Categories pulled**:  
  `astro-ph.CO, astro-ph.GA, astro-ph.EP, astro-ph.HE, astro-ph.IM, astro-ph.SR, gr-qc, physics.comp-ph, physics.space-ph`

- **Config file**: [`configs/data.yaml`](configs/data.yaml)

  ```yaml
  categories:
    - astro-ph.CO
    - astro-ph.GA
    - astro-ph.EP
    - astro-ph.HE
    - astro-ph.IM
    - astro-ph.SR
    - gr-qc
    - physics.comp-ph
    - physics.space-ph

  per_category_max: 15000
  out_jsonl: "data/raw/arxiv_astro.jsonl"

  page_size: 100
  delay_seconds: 4
  timeout_seconds: 30
  max_empty_skips: 10

  debug: false
  debug_per_category_max: 500
  ```

- **Script**: [`scripts/00_download_arxiv.py`](scripts/00_download_arxiv.py)  
  Implements polite, resume-safe harvesting with retries and skips empty pages.

- **Run**:

  ```bash
  # from repo root
  python scripts/00_download_arxiv.py
  ```

- **Output**:  
  JSONL file at `data/raw/arxiv_astro.jsonl`  
  Each line is one paper:

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

- **Progress so far**: 120k unique astro/physics records collected (~1.2 GB).

---

## Next Steps

1. **Phase 2**: Chunk abstracts and generate title→abstract training pairs.
2. **Phase 3**: Fine-tune a **bi-encoder retriever** (E5-base).
3. **Phase 4**: Build FAISS index and train a **cross-encoder reranker** (MiniLM).
4. **Phase 5**: Evaluate retrieval vs reranking on held-out queries.
5. **Phase 6**: Build notebooks to visualize improvements and qualitative examples.

---

## License

MIT
