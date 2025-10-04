# Examples

This folder contains a **toy index** and a **demo notebook** that run without external models.

- `toy_passages.jsonl` – 10 sample astrophysics passages
- `toy_index/` – minimal index artifacts:
  - `embeddings.npy` – precomputed toy embeddings (keyword-based)
  - `meta.jsonl` – passage metadata
  - `model.txt` – label for the toy embedding
- `demo.ipynb` – interactive retrieval demo (cosine similarity over toy embeddings)
