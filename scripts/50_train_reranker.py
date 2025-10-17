#!/usr/bin/env python3
"""
Train a cross-encoder reranker on weakly supervised pairs.

Input: JSONL rows from 40_make_weak_pairs.py with fields:
  {
    "qid": "...",
    "query": "...",
    "pos": "positive passage text",
    "negs": ["neg passage 1", "neg passage 2", ...]
  }

This script expands each row to:
  (query, pos)  -> label 1.0
  (query, neg*) -> label 0.0
and fine-tunes a CrossEncoder.

Recommended base model:
  cross-encoder/ms-marco-MiniLM-L-6-v2
"""

import argparse
import json
import os
import random
import math
from typing import List

import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

try:
    # If sklearn is available, we’ll use it for a clean split.
    from sklearn.model_selection import train_test_split
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pairs(path: str, max_negs_per_row: int = None) -> List[InputExample]:
    """Load weakly supervised pairs and expand to InputExample list."""
    rows: List[InputExample] = []
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            q = o["query"]
            pos = o["pos"]
            negs = o.get("negs", [])
            if max_negs_per_row is not None:
                negs = negs[:max_negs_per_row]

            # Positive
            rows.append(InputExample(texts=[q, pos], label=1.0))
            # Negatives
            for n in negs:
                rows.append(InputExample(texts=[q, n], label=0.0))
    return rows


def simple_split(data: List[InputExample], test_size: float, seed: int):
    """Fallback split if sklearn isn't available."""
    rng = random.Random(seed)
    data = data[:]  # copy
    rng.shuffle(data)
    n_test = max(1, int(len(data) * test_size))
    return data[n_test:], data[:n_test]


def main(a: argparse.Namespace):
    os.makedirs(a.out, exist_ok=True)
    set_seed(a.seed)

    print(f"[cfg] model={a.model}  train={a.train}  out={a.out}")
    print(f"[cfg] epochs={a.epochs}  batch_size={a.batch_size}  lr={a.lr}  max_len={a.max_len}")
    print(f"[cfg] warmup_ratio={a.warmup_ratio}  seed={a.seed}")

    all_examples = load_pairs(a.train, max_negs_per_row=a.max_negs_per_row)
    if len(all_examples) < 10:
        raise ValueError(f"Too few training examples: {len(all_examples)}. "
                         f"Check that data/pairs/train_pairs.jsonl exists and is non-empty.")

    if HAVE_SKLEARN:
        train_rows, dev_rows = train_test_split(
            all_examples, test_size=a.dev_ratio, random_state=a.seed, shuffle=True
        )
    else:
        train_rows, dev_rows = simple_split(all_examples, test_size=a.dev_ratio, seed=a.seed)

    print(f"[data] train={len(train_rows)}  dev={len(dev_rows)}  (total={len(all_examples)})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}  cuda_available={torch.cuda.is_available()}  "
          f"gpu_count={torch.cuda.device_count()}")

    # CrossEncoder will default to BCE for 0/1 labels; no custom loss needed.
    model = CrossEncoder(a.model, max_length=a.max_len, device=device)

    train_loader = DataLoader(train_rows, shuffle=True, batch_size=a.batch_size)
    dev_loader = DataLoader(dev_rows, shuffle=False, batch_size=a.batch_size)

    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = int(steps_per_epoch * a.epochs * a.warmup_ratio)
    print(f"[train] steps/epoch={steps_per_epoch}  warmup_steps={warmup_steps}")

    model.fit(
        train_dataloader=train_loader,
        evaluator=None,
        epochs=a.epochs,
        optimizer_params={'lr': a.lr},
        warmup_steps=warmup_steps,
        output_path=a.out,
        show_progress_bar=True
    )

    # (CrossEncoder wraps a HF AutoModelForSequenceClassification + tokenizer)
    model.model.save_pretrained(a.out)
    model.tokenizer.save_pretrained(a.out)

    print(f"[done] Saved fine-tuned reranker to: {a.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    help="HuggingFace model id or local CE checkpoint.")
    ap.add_argument("--train", required=True, help="JSONL from 40_make_weak_pairs.py")
    ap.add_argument("--out", required=True, help="Output directory for the fine-tuned model")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--dev_ratio", type=float, default=0.1,
                    help="Fraction of examples to hold out for dev (only for logging).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_negs_per_row", type=int, default=4,
                    help="Cap negatives per query row to bound expansion.")
    main(ap.parse_args())
