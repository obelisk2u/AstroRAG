#!/bin/bash
#$ -N arxiv_dl                   # Job name
#$ -P aisearch                   # Project account
#$ -pe omp 1                     # 1 core
#$ -l h_rt=06:00:00              # Max runtime
#$ -l mem_per_core=4G            # Memory per core
#$ -o slurm/$JOB_NAME-$JOB_ID.out  # Standard output file
#$ -e slurm/$JOB_NAME-$JOB_ID.err  # Standard error file

# --- Load modules ---
module load python3/3.10.12

# --- Activate your venv ---
source /projectnb/aisearch/AstroRAG/.venv/bin/activate

# --- Run the script ---
python /projectnb/aisearch/AstroRAG/scripts/00_download_arxiv.py

