#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to metrics_faiss_vs_ce.csv")
    ap.add_argument("--outdir", default="exp/plots", help="Directory to save plots")
    ap.add_argument("--k", default="10", help="Top-k label for titles")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv).sort_values("chunk_size")
    x = df["chunk_size"]

    # ---------- Plot 1: NDCG ----------
    plt.figure(figsize=(7,4))
    plt.plot(x, df["NDCG@10_faiss"], marker="o", label="FAISS")
    plt.plot(x, df["NDCG@10_ce"], marker="o", label="Cross Encoder")
    plt.xlabel("Chunk Size (words)")
    plt.ylabel("NDCG@10")
    plt.title(f"NDCG@10 vs Chunk Size (Top-{args.k})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "ndcg_faiss_vs_ce.png", dpi=300)

    # ---------- Plot 2: Recall ----------
    plt.figure(figsize=(7,4))
    plt.plot(x, df["Recall@10_faiss"], marker="o", label="FAISS")
    plt.plot(x, df["Recall@10_ce"], marker="o", label="Cross Encoder")
    plt.xlabel("Chunk Size (words)")
    plt.ylabel("Recall@10")
    plt.title(f"Recall@10 vs Chunk Size (Top-{args.k})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "recall_faiss_vs_ce.png", dpi=300)

    print(f"✅ Saved plots to {outdir}")

if __name__ == "__main__":
    main()