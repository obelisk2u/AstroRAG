import pandas as pd, matplotlib.pyplot as plt

raw_df = pd.read_csv("exp/chunk_sweep_fine/metrics.csv")

df = raw_df.copy()

for c in ["NDCG@10","MRR@10","Recall@10"]:
    df[c] = (df[c]-df[c].min())/(df[c].max()-df[c].min())

plt.figure(figsize=(7,4))
plt.plot(df["chunk_size"], df["NDCG@10"], marker='o', label="NDCG@10")
plt.plot(df["chunk_size"], df["Recall@10"], marker='o', label="Recall@10")
plt.xlabel("Chunk Size (words)")
plt.ylabel("Metric Score")
plt.title("Normalized Retrieval Performance vs Chunk Size")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("chunk_size_metrics_norm.png", dpi=300)
plt.show()