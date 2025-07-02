#!/usr/bin/env python3
"""
runtime_vs_nnz.py  ·  v1.0
--------------------------------------------
Plot run-time [ms] versus matrix size (nnz) for
* t_mkl_ms
* t_tasks_ms
* t_trilinos_ms

Input files must be named  benchmark_<threads>.csv
and contain the columns:
  matrix, nnz, t_mkl_ms, t_tasks_ms, t_trilinos_ms
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

FILE_RE = re.compile(r"(?:benchmark|results?)_(\d+)\.csv$", re.I)
SOLVERS  = {
    "MKL"     : "t_mkl_ms",
    "Tasks"   : "t_tasks_ms",
    "Kokkos"  : "t_trilinos_ms",
}

# ────────────────────────────────────────────────────────────────────────────
def find_csv(p: int, folder: Path) -> Path:
    """Return the benchmark_<p>.csv file in *folder*, raise if missing."""
    for f in folder.glob("*.csv"):
        m = FILE_RE.search(f.name)
        if m and int(m.group(1)) == p:
            return f
    raise FileNotFoundError(f"benchmark_{p}.csv not found in {folder}")

def load_times(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    needed = {"matrix", "nnz"} | {c.lower() for c in SOLVERS.values()}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    return df

def plot_one_p(df: pd.DataFrame, p: int, out_pdf: Path, png: bool):
    plt.figure(figsize=(6.4, 4.4))

    # sort by nnz for a clean poly-line
    df = df.sort_values("nnz")

    for label, col in SOLVERS.items():
        y = df[col.lower()]
        plt.plot(df["nnz"], y,
                 marker="o", label=f"{label}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of non-zeros  $N_{\\!nz}$")
    plt.ylabel("Run-time  [ms]")
    plt.title(f"{p} threads")
    plt.grid(True, which="both", linestyle="dotted")
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_pdf)
    if png:
        plt.savefig(out_pdf.with_suffix(".png"), dpi=150)
    plt.close()
    print(f"  • wrote  {out_pdf}")

# ────────────────────────────────────────────────────────────────────────────
def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", type=Path, default=Path("."),
                    help="Folder with benchmark_<p>.csv files")
    ap.add_argument("-p", "--threads", type=int, nargs="+", default=[1,6,24],
                    metavar="P", help="Thread counts to plot")
    ap.add_argument("--png", action="store_true",
                    help="Save a .png next to each PDF")
    args = ap.parse_args(argv)

    for p in args.threads:
        print(f"[{p} threads]")
        csv_path = find_csv(p, args.dir)
        df = load_times(csv_path)
        out_pdf = args.dir / f"runtime_vs_nnz_{p}.pdf"
        plot_one_p(df, p, out_pdf, args.png)

if __name__ == "__main__":
    main()