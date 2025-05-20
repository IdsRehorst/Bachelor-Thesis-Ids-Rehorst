import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <path/to/benchmark.csv>")
    sys.exit(1)
csv_path = sys.argv[1]
if not os.path.exists(csv_path):
    print(f"Error: file not found: {csv_path}")
    sys.exit(1)

print("Loading benchmark data from:", csv_path)
df = pd.read_csv(csv_path)

# --- Plot solve time vs n (sorted by n) ---
df_n = df.sort_values("n")
plt.figure(figsize=(6,4))
plt.plot(df_n["n"], df_n["t_mkl_ms"],   "o-", label="MKL")
plt.plot(df_n["n"], df_n["t_tasks_ms"], "s-", label="Tasks")
plt.xlabel("Matrix size n")
plt.ylabel("Solve time (ms)")
plt.title("SpTRSV Solve Time vs Matrix Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("time_vs_n.png", dpi=150)

# --- Plot speedup vs nnz (sorted by nnz) ---
df_nnz = df.sort_values("nnz")
plt.figure(figsize=(6,4))
plt.plot(df_nnz["nnz"], df_nnz["speedup"], "d-")
plt.xlabel("Number of nonzeros")
plt.ylabel("Speedup (MKL / Tasks)")
plt.title("Speedup vs Sparsity")
plt.grid(True)
plt.tight_layout()
plt.savefig("speedup_vs_nnz.png", dpi=150)

plt.show()
