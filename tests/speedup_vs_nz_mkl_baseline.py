import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_speedup_vs_nnz_baseline_mkl1(file_threads_map,
                                      output_png="speedup_vs_nnz_mkl1baseline.png"):
    """
    file_threads_map:  {csv_path : n_threads}
        Must contain an entry for the 1-thread MKL run and the task runs
        you want to compare (here 16 & 32).
    """
    # ------------------------------------------------------------------ load
    frames = []
    for csv, thr in file_threads_map.items():
        df = pd.read_csv(csv)
        df.columns = df.columns.str.strip()        # tidy headers
        df["threads"] = thr
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------ MKL 1-thread base
    mkl1 = (
        data[data["threads"] == 1]
        .set_index("matrix")["t_mkl_ms"]
        .rename("mkl1_ms")
    )
    data = data.join(mkl1, on="matrix")

    # keep only the multi-threaded task runs we care about
    data = data[data["threads"].isin([16, 32])]

    data["speedup"] = data["mkl1_ms"] / data["t_tasks_ms"]
    data = data.sort_values("nnz")                # nicer curves

    # ------------------------------------------------------------- plot
    style = {16: ("s", "C1"), 32: ("v", "C3")}
    plt.figure(figsize=(10, 6))
    for p in [16, 32]:
        sub = data[data["threads"] == p]
        plt.plot(
            sub["nnz"], sub["speedup"],
            marker=style[p][0], color=style[p][1],
            label=f"{p}-thr tasks"
        )

    plt.xscale("log")
    plt.xlabel(r"Non-zeros in matrix ($N_{nz}$)")
    plt.ylabel(r"Speed-up $=t_{\mathrm{MKL},1}/t_{\mathrm{tasks},p}$")
    plt.title("Speed-up vs. nnz (baseline: MKL, 1 thread)")
    plt.axhline(1.0, color="grey", ls=":", lw=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.show()
    print("saved →", Path(output_png).resolve())

# -------------------------------------------------------------------- usage
file_threads = {
    "benchmark_1.csv": 1,    # MKL reference run
    "benchmark_16.csv": 16,  # task solver, 16 threads
    "benchmark_32.csv": 32,  # task solver, 32 threads
}
plot_speedup_vs_nnz_baseline_mkl1(file_threads)