#!/usr/bin/env python3
# ---------------------------------------------------------------
# plot_rel_speedup_separate.py
#
#   • rel_speedup_mkl_vs_tasks.[pdf|png]
#       y = t_mkl_ms       / t_tasks_ms
#
#   • rel_speedup_kokkos_vs_tasks.[pdf|png]
#       y = t_trilinos_ms  / t_tasks_ms
#
#   X-axis = nnz (log scale, ascending)
#   One poly-line per chosen thread count.
# ---------------------------------------------------------------
from __future__ import annotations
import sys, re, matplotlib.pyplot as plt, pandas as pd
from pathlib import Path

CSV_PAT   = re.compile(r"benchmark_(\d+)\.csv$", re.I)
REQUIRED  = {"nnz", "t_tasks_ms", "t_mkl_ms", "t_trilinos_ms"}

# ----------------------------------------------------------------
def load(folder: Path = Path(".")) -> dict[int, pd.DataFrame]:
    """Return {threads : DataFrame sorted by nnz ASC}."""
    out = {}
    for f in folder.glob("benchmark_*.csv"):
        if (m := CSV_PAT.match(f.name)):
            p  = int(m.group(1))
            df = pd.read_csv(f)
            df.columns = [c.lower().strip() for c in df.columns]
            if REQUIRED.issubset(df.columns):
                out[p] = df.sort_values("nnz")
    return dict(sorted(out.items()))

# ----------------------------------------------------------------
def single_plot(kind: str, ratio_col: str, data: dict[int, pd.DataFrame],
                picks: list[int] | None = None):
    """Create one plot for *ratio_col* / tasks."""
    picks = sorted(data) if picks is None else [p for p in picks if p in data]
    if not picks:
        print(f"[{kind}]  No CSVs for the requested thread counts.")
        return

    plt.figure(figsize=(6.4, 4.0))
    for p in picks:
        df = data[p]
        plt.plot(df["nnz"], df[ratio_col] / df["t_tasks_ms"],
                 marker="o", ls="-", label=f"{p} threads")

    plt.xscale("log")
    plt.xlabel(r"non-zeros  $N_{\mathrm{nz}}$")
    plt.ylabel(rf"$t_{{\text{{{kind}}}}} / t_{{\text{{tasks}}}}$")
    plt.title(f"{kind} vs. task-based solver")
    plt.grid(True, ls="dotted")
    plt.legend(fontsize="small")
    plt.tight_layout()

    stem = f"rel_speedup_{kind.lower()}_vs_tasks"
    plt.savefig(f"{stem}.pdf")
    plt.savefig(f"{stem}.png", dpi=150)
    plt.close()
    print(f"Created {stem}.pdf / .png")

# ----------------------------------------------------------------
if __name__ == "__main__":
    # optional list of thread counts on the CLI, e.g. 6 12 24
    req = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else None

    tables = load()
    if not tables:
        sys.exit("No benchmark_<threads>.csv files found with required columns.")

    single_plot("MKL",     "t_mkl_ms",      tables, req)
    single_plot("Kokkos",  "t_trilinos_ms", tables, req)