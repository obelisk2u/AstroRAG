#!/bin/bash
#$ -cwd
#$ -N rerank
#$ -P aisearch
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=01:00:00
#$ -o logs/rerank.out
#$ -e logs/rerank.err
#$ -V
set -euo pipefail

export HF_HOME=/projectnb/aisearch/$USER/.cache/hf
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$NSLOTS

python scripts/60_rerank.py \
  --index indexes/faiss_base/index.faiss \
  --meta indexes/faiss_base/meta.jsonl \
  --queries data/queries/dev.jsonl \
  --reranker outputs/reranker/minilm_ce_full \
  --out outputs/runs/faiss_ce_dev.trec \
  --faiss_topk 200 --final_topk 10
