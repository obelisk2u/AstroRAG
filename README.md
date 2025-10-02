# AstroRAG: Transformer Reranking for Astrophysics Literature Search

AstroRAG is a retrieval pipeline for astrophysics papers that demonstrates how **transformer bi-encoders** (for fast retrieval) and **cross-encoders** (for precise reranking) can improve scientific literature search.  
The project is designed to be fully reproducible on the BU SCC cluster, and lightweight enough (~3 GB total) to run locally for demos.

---

## Project Status

- ✅ **Phase 1: Data Collection** (arXiv astro/physics abstracts downloaded, JSONL format)
- ⏳ **Phase 2**: Chunk abstracts and generate title→abstract training pairs.
- ⏳ **Phase 3**: Fine-tune a **bi-encoder retriever** (E5-base).
- ⏳ **Phase 4**: Build FAISS index and train a **cross-encoder reranker** (MiniLM).
- ⏳**Phase 5**: Evaluate retrieval vs reranking on held-out queries.
- ⏳**Phase 6**: Build notebooks to visualize improvements and qualitative examples.

---

## Repo Structure

```
AstroRAG/
├─ configs/          # YAML configs
├─ data/             # (gitignored) raw + processed data
├─ scripts/          # Python scripts for each step
├─ outputs/          # (gitignored) trained model checkpoints
├─ slurm/            # SCC batch scripts
├─ Makefile          # Workflow commands
└─ README.md
```

---

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Phase 1: Data Collection

We use the [arXiv API](https://arxiv.org/help/api/) to fetch astrophysics and physics paper metadata.  
Each record includes: **id, title, abstract (summary), authors, categories, published date**.

- **Categories pulled**:  
  `astro-ph.CO, astro-ph.GA, astro-ph.EP, astro-ph.HE, astro-ph.IM, astro-ph.SR, gr-qc, physics.comp-ph, physics.space-ph`

- **Config file**: [`configs/data.yaml`](configs/data.yaml)

- **Script**: [`scripts/00_download_arxiv.py`](scripts/00_download_arxiv.py)  
  Implements polite, resume-safe harvesting with retries and skips empty pages.

- **Run**:

  ```bash
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

## License

MIT
