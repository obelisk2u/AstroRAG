#!/bin/bash
#$ -N embed_and_index
#$ -P aisearch
#$ -cwd
#$ -pe omp 8
#$ -l h_rt=06:00:00
#$ -l mem_per_core=8G
#$ -j y
#$ -o /projectnb/aisearch/AstroRAG/logs/$JOB_NAME-$JOB_ID.log
#$ -S /bin/bash

set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

source /projectnb/aisearch/AstroRAG/.venv/bin/activate

python scripts/10_chunk_passages.py data/raw/arxiv_astro.jsonl data/passages.jsonl 120 100

if [ ! -s data/passages.jsonl ]; then
  echo "ERROR: data/passages.jsonl is missing or empty" >&2
  exit 1
fi

python scripts/20_embed_and_index.py data/passages.jsonl indexes/faiss_base sentence-transformers/all-MiniLM-L6-v2 256

