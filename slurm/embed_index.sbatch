#!/bin/bash
#$ -cwd
#$ -N embed_index_gpu
#$ -P aisearch
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=02:00:00
#$ -l mem_per_core=6G
#$ -o logs/embed_index.out
#$ -e logs/embed_index.err
#$ -V

# Init Environment Modules (portable across nodes)
if [ -f /usr/share/Modules/init/bash ]; then
  . /usr/share/Modules/init/bash
elif [ -f /usr/share/Modules/init/sh ]; then
  . /usr/share/Modules/init/sh
fi

module purge
module load pytorch/1.13.1   # provides GPU-ready torch

# Put your user site-packages first (where you just installed ST/transformers/faiss)
# Discover the exact path programmatically to avoid version mismatches
USER_SITE=$(python -c 'import site; print(site.getusersitepackages())')
export PYTHONPATH="$USER_SITE:$PYTHONPATH"

# Keep caches out of $HOME to protect your 10GB quota
export HF_HOME=/projectnb/aisearch/$USER/.cache/hf
export TRANSFORMERS_CACHE=/projectnb/aisearch/$USER/.cache/transformers
export XDG_CACHE_HOME=/projectnb/aisearch/$USER/.cache/xdg
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$NSLOTS

echo "Node: $(hostname)"
which python || true
echo "USER_SITE=$USER_SITE"
nvidia-smi || true
python - <<'PY'
import sys, torch, sentence_transformers, numpy
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
print("SentenceTransformers:", sentence_transformers.__version__)
print("NumPy:", numpy.__version__)
PY

# Your script expects positional args
python scripts/20_embed_and_index.py \
  data/passages.jsonl \
  indexes/faiss_base \
  sentence-transformers/all-MiniLM-L6-v2 \
  512
