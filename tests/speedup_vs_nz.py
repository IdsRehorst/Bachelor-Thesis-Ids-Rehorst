import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_speedup_vs_nnz(file_threads_map, output_png="speedup_vs_nnz_threads_32.png"):
    """
    Plot speed‑up (t_mkl / t_tasks) versus nnz with one line per thread count.

    Parameters
    ----------
    file_threads_map : dict
        Mapping ``csv_path -> n_threads``.
    output_png : str
        Where to save the resulting PNG.
    """
    frames = []
    for csv, thr in file_threads_map.items():
        df = pd.read_csv(csv)
        df.columns = df.columns.str.strip()
        df["threads"] = thr
        frames.append(df)

    data = pd.concat(frames, ignore_index=True).sort_values("nnz")
    data["speedup"] = data["t_mkl_ms"] / data["t_tasks_ms"]

    plt.figure(figsize=(10, 6))
    style_map = {1: "o", 16: "s", 32: "v", 64: "^"}  # extendable
    for thr in sorted(data["threads"].unique()):
        subset = data[data["threads"] == thr]
        plt.plot(
            subset["nnz"],
            subset["speedup"],
            marker=style_map.get(thr, "o"),
            linestyle="-",
            label=f"{thr}‑thr",
        )

    plt.xscale("log")
    plt.xlabel(r"Non‑zeros in matrix ($N_{nz}$)")
    plt.ylabel("Speed‑up (MKL runtime / task runtime)")
    plt.title(r"Speed‑up vs. $N_{nz}$")
    plt.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    plt.legend(title="Thread count")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.show()
    print("Saved:", output_png)

# ------------------ Run for current benchmark set ---------------------------
file_threads = {
    #"benchmark_1.csv": 1,
    #"benchmark_16.csv": 16,
    "benchmark_32.csv": 32,
}

plot_speedup_vs_nnz(file_threads)