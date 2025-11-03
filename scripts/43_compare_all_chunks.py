#!/usr/bin/env python3
import json, csv, subprocess, glob, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "exp" / "chunk_sweep_fine"

def run(cmd):
    print("➤", " ".join(cmd))
    subprocess.run(cmd, check=True)

def eval_run(run_path, meta, queries, k=10):
    out = subprocess.check_output([
        "python", "scripts/41_eval_run.py",
        "--run", str(run_path),
        "--meta", str(meta),
        "--queries", str(queries),
        "--k", str(k)
    ], cwd=str(ROOT)).decode().strip()
    # expects: NDCG@10=0.6502  MRR@10=0.6054  Recall@10=0.8000
    parts = dict(x.split("=") for x in out.replace("NDCG@10","NDCG").replace("MRR@10","MRR").replace("Recall@10","Recall").split())
    return float(parts["NDCG"]), float(parts["MRR"]), float(parts["Recall"])

def main():
    rows = []
    for d in sorted(BASE.glob("chunk_*")):
        size = int(d.name.split("_")[1])
        meta = d / "index" / "meta.jsonl"
        queries = ROOT / "data" / "queries" / "dev.jsonl"
        faiss_run = d / "faiss_top100.trec"
        ce_run = d / "ce_top100.trec"

        # rerank if needed
        if not ce_run.exists():
            run([
                "python", "scripts/32_rerank_cross_encoder.py",
                "--in_run", str(faiss_run),
                "--meta", str(meta),
                "--queries", str(queries),
                "--out", str(ce_run),
                "--model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ],)

        # evaluate both
        n1,m1,r1 = eval_run(faiss_run, meta, queries, k=10)
        n2,m2,r2 = eval_run(ce_run, meta, queries, k=10)

        rows.append({
            "chunk_size": size,
            "NDCG@10_faiss": round(n1,4), "MRR@10_faiss": round(m1,4), "Recall@10_faiss": round(r1,4),
            "NDCG@10_ce":    round(n2,4), "MRR@10_ce":    round(m2,4), "Recall@10_ce":    round(r2,4),
            "ΔNDCG": round(n2-n1,4), "ΔMRR": round(m2-m1,4), "ΔRecall": round(r2-r1,4)
        })
        print(f"[chunk {size}] ΔNDCG={n2-n1:.4f} ΔMRR={m2-m1:.4f} ΔRecall={r2-r1:.4f}")

    out_csv = BASE / "metrics_faiss_vs_ce.csv"
    with open(out_csv, "w", newline="") as f:
        cols = ["chunk_size",
                "NDCG@10_faiss","MRR@10_faiss","Recall@10_faiss",
                "NDCG@10_ce","MRR@10_ce","Recall@10_ce",
                "ΔNDCG","ΔMRR","ΔRecall"]
        writer = csv.DictWriter(f, fieldnames=cols); writer.writeheader()
        for r in sorted(rows, key=lambda x: x["chunk_size"]): writer.writerow(r)
    print(f"\n✅ Wrote comparison: {out_csv}")

if __name__ == "__main__":
    main()