#!/bin/bash
#$ -N embed_and_index
#$ -P aisearch
#$ -cwd
#$ -pe omp 4
#$ -l h_rt=06:00:00
#$ -l mem_per_core=8G
#$ -j y
#$ -o /projectnb/aisearch/AstroRAG/logs/$JOB_NAME-$JOB_ID.log

module load python3/3.10.12
source /projectnb/aisearch/AstroRAG/.venv/bin/activate
pip install -r requirements.txt -q

python scripts/10_chunk_passages.py data/raw/arxiv_astro.jsonl data/passages.jsonl 120 100
python scripts/20_embed_and_index.py data/passages.jsonl indexes/faiss_base sentence-transformers/all-MiniLM-L6-v2 512

