#!/bin/bash
#$ -cwd
#$ -N train_reranker
#$ -P aisearch
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0              # prefer newer GPUs (A40/A100/V100 class)
#$ -l h_rt=02:00:00
#$ -l mem_per_core=6G
#$ -o logs/train_reranker.out
#$ -e logs/train_reranker.err
#$ -V
set -euo pipefail

echo "==== [ENV INIT] $(hostname) ===="

# --- (Optional) Initialize modules if present; ignore if not ---
if [ -f /usr/share/lmod/lmod/init/bash ]; then
  . /usr/share/lmod/lmod/init/bash
elif [ -f /usr/share/Modules/init/bash ]; then
  . /usr/share/Modules/init/bash
elif [ -f /usr/share/Modules/init/sh ]; then
  . /usr/share/Modules/init/sh
fi

# Try to load a PyTorch module if this node has one; otherwise continue silently.
if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
  (module load pytorch/1.13.1 2>/dev/null || module load pytorch/1.12.1 2>/dev/null || true)
fi

# --- Prefer your user site-packages (sentence-transformers, faiss, etc.) ---
USER_SITE=$(python -c 'import site; print(site.getusersitepackages())' 2>/dev/null || echo "$HOME/.local/lib/python3.8/site-packages")
export PYTHONPATH="$USER_SITE:$PYTHONPATH"

# --- Put caches in project space (protect $HOME quota) ---
export HF_HOME=/projectnb/aisearch/$USER/.cache/hf
export XDG_CACHE_HOME=/projectnb/aisearch/$USER/.cache/xdg
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$NSLOTS

echo "==== [RUNTIME INFO] ===="
which python || true
nvidia-smi || true
python - <<'PY'
import sys, torch
print("Python:", sys.version.split()[0])
print("Torch:", getattr(torch,'__version__','<missing>'),
      "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
echo "========================"

# --- Train Cross-Encoder Reranker ---
python scripts/50_train_reranker.py \
  --model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --train data/pairs/train_pairs.jsonl \
  --out outputs/reranker/minilm_ce \
  --lr 2e-5 --epochs 2 --batch_size 32 --max_len 320
