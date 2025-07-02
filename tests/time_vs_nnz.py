import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# 1.  Load all benchmark CSVs into a single tidy DataFrame
# ---------------------------------------------------------------------------
def load_benchmark_data(file_threads_map):
    """
    Parameters
    ----------
    file_threads_map : dict
        Maps CSV file path → thread count used in that run.
        e.g. {"results/benchmark_1.csv": 1, "results/benchmark_16.csv": 16}
    Returns
    -------
    pd.DataFrame with columns:
        matrix, nnz, t_tasks_ms, t_mkl_ms, threads
    """
    frames = []
    for csv, nthreads in file_threads_map.items():
        df = pd.read_csv(csv)
        df.columns = df.columns.str.strip()  # tidy stray spaces
        df["threads"] = nthreads
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values("nnz")

# ---------------------------------------------------------------------------
# 2.  Plot ONLY MKL lines (one per thread count)
# ---------------------------------------------------------------------------
def plot_mkl_only(data, output_png="runtime_vs_nnz_mkl_only.png"):
    marker_for = {1: "o", 16: "s", 32: "v", 64: "^"}
    plt.figure(figsize=(8, 5))
    for p in sorted(data["threads"].unique()):
        sub = data[data["threads"] == p]
        plt.plot(
            sub["nnz"], sub["t_mkl_ms"],
            marker=marker_for.get(p, "o"),
            label=f"MKL, {p}-thr"
        )
    plt.xscale("log");  plt.yscale("log")
    plt.xlabel(r"Non-zeros ($N_{nz}$)");  plt.ylabel("Runtime (ms)")
    plt.title(r"MKL runtime vs. $N_{nz}$")
    plt.legend();  plt.tight_layout()
    plt.savefig(output_png, dpi=300);  plt.show()
    print("saved →", Path(output_png).resolve())

# ---------------------------------------------------------------------------
# 3.  Plot MKL (1-thr) baseline + ALL task-solver lines (multithreaded)
# ---------------------------------------------------------------------------
def plot_mkl_and_tasks(data, output_png="runtime_vs_nnz_mkl_tasks.png"):
    plt.figure(figsize=(8, 5))

    # --- single-thread MKL baseline
    mkl1 = data[data["threads"] == 1]
    plt.plot(
        mkl1["nnz"], mkl1["t_mkl_ms"],
        marker="o", linestyle="--", color="black", label="MKL, 1-thr"
    )

    # --- multithreaded task-solver runs
    marker_for = {16: ("s", "red"), 32: ("v", "blue"), 64: ("^", "C3")}
    for p in sorted(set(data["threads"]) - {1}):
        sub = data[data["threads"] == p]
        marker, color = marker_for.get(p, ("o", None))
        plt.plot(
            sub["nnz"], sub["t_tasks_ms"],
            marker=marker, linestyle="-", color=color,
            label=f"tasks, {p}-thr"
        )

    plt.xscale("log");  plt.yscale("log")
    plt.xlabel(r"Non-zeros ($N_{nz}$)");  plt.ylabel("Runtime (ms)")
    plt.title(r"Runtime vs. $N_{nz}$ – MKL baseline vs. task solver")
    plt.legend();  plt.tight_layout()
    plt.savefig(output_png, dpi=300);  plt.show()
    print("saved →", Path(output_png).resolve())

# ---------------------------------------------------------------------------
# 4.  Example usage – EDIT THE PATHS BELOW
# ---------------------------------------------------------------------------
file_threads = {
    "benchmark_1.csv" : 1,   #   MKL & tasks, 1 thread
    "benchmark_16.csv": 16,  #   tasks, 16 threads
    "benchmark_32.csv": 32,  #   tasks, 32 threads
}

df = load_benchmark_data(file_threads)
plot_mkl_only(df)          # produces runtime_vs_nnz_mkl_only.png
plot_mkl_and_tasks(df)     # produces runtime_vs_nnz_mkl_tasks.png