import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------- Reload benchmark CSVs -----------------
file_threads = {
    "benchmark_1.csv": 1,
    "benchmark_16.csv": 16,
    "benchmark_32.csv": 32,
}

frames = []
for csv, thr in file_threads.items():
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    df["threads"] = thr
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

# Sort by nnz for nicer lines
data = data.sort_values("nnz")

# ------------- Figure: raw runtime vs nnz --------------
plt.figure(figsize=(10, 6))

style_map = {1: "o", 16: "s", 32: "v"}
for thr in [1, 16, 32]:
    subset = data[data["threads"] == thr]
    plt.plot(
        subset["nnz"],
        subset["t_tasks_ms"],
        marker=style_map[thr],
        linestyle="-",
        label=f"tasks, {thr}‑thr",
    )
    plt.plot(
        subset["nnz"],
        subset["t_mkl_ms"],
        marker=style_map[thr],
        linestyle="--",
        label=f"MKL, {thr}‑thr",
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Non‑zeros in matrix (nnz)")
plt.ylabel("Runtime (ms)")
plt.title("Raw runtime vs. nnz for tasks vs. MKL (all thread counts)")
plt.legend(fontsize="x-small", ncols=2, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
fig_path = Path("runtime_vs_nnz.png")
plt.savefig(fig_path, dpi=300)
plt.show()

print("Saved:", fig_path)