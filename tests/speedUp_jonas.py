#!/usr/bin/env python3
"""
plot_speedup.py  ·  v2.1  ·  2025-06-27
=================================================
Generate strong-scaling speed-up curves for **any** timing
column contained in your *benchmark_<threads>.csv* files —
for example ``t_tasks_ms`` **and** ``t_mkl_ms``.

The script can now:
  • accept **one or more** timing columns via ``--cols``;
  • emit *one* PDF (and optional PNG) **per column**, so you can
    immediately spot whether MKL scales.

Typical workflow
----------------
```bash
# 1.  Task-based solver
python plot_speedup.py --cols t_tasks_ms                # default

# 2.  MKL triangular solve  (parallel check)
python plot_speedup.py --cols t_mkl_ms  --top 0         # plot all matrices

# 3.  Both at once (two PDFs)
python plot_speedup.py --cols t_tasks_ms,t_mkl_ms --savepng
```

CSV layout expected (case-insensitive)
--------------------------------------
matrix        – name/path of the test matrix (string)
nnz           – number of non-zeros            (int)
<col>         – run time in ms for a given solver & thread count

The file name must encode the thread count as in ``benchmark_16.csv``.
Only ``matrix`` and the requested timing columns are required.

Limitations
-----------
* The script still plots **one line per matrix** (not per solver).
  When you include several columns, you obtain several separate
  figures (``speedup_<col>.pdf``).  This keeps each plot readable.
* The strong-scaling ratio is always
  **S(m,p) = t(m,1) / t(m,p)**.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

FILE_RE = re.compile(r"(?:benchmark|results?)_(\d+)\.csv$", re.I)
DEFAULT_COLS = ["t_tasks_ms", "t_trilinos_ms"]

# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------

def find_csvs(directory: Path) -> List[Tuple[int, Path]]:
    """Return a list of (threads, path) pairs for all matching CSV files."""
    return [
        (int(m.group(1)), f)
        for f in directory.glob("*.csv")
        if (m := FILE_RE.search(f.name))
    ]

def read_wide_table(csv_files: List[Tuple[int, Path]], col: str) -> pd.DataFrame:
    """Return wide *times* table for *one* timing column.

    Index   = matrix names
    Columns = thread counts (int)
    Values  = run time in **ms** (float)
    """
    frames = []
    for p, path in csv_files:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if {"matrix", col}.issubset(df.columns):
            frames.append(
                df[["matrix", col]].rename(columns={col: p}).set_index("matrix")
            )
        else:
            print(
                f"Skipping {path}: missing 'matrix' or '{col}' column",
                file=sys.stderr,
            )

    if not frames:
        raise RuntimeError(
            f"No usable CSV files contained the column '{col}'."
        )

    # merge on index (matrix), keep numeric columns only
    return pd.concat(frames, axis=1).sort_index()

def largest_n_matrices(times: pd.DataFrame, nnz_ser: pd.Series, n: int) -> pd.DataFrame:
    """Keep only the *n* matrices with the largest nnz (n <= 0 ⇒ keep all)."""
    if n <= 0:
        return times
    keep = nnz_ser.sort_values(ascending=False).head(n).index
    return times.loc[keep]

# -----------------------------------------------------------------------------
# Speed-up computation & plotting
# -----------------------------------------------------------------------------

def compute_speedup(times: pd.DataFrame) -> pd.DataFrame:
    """S(m,p) = t(m,1) / t(m,p)  (baseline broadcast across columns)."""
    baseline = times[1]                      # Series
    return times.rdiv(baseline, axis=0)      # baseline / times

def plot_speedup(speedup: pd.DataFrame, out_pdf: Path, png: bool = False):
    """Write *out_pdf* (and optional PNG) with one polyline per matrix."""
    plt.figure(figsize=(6.5, 4))

    for matrix, row in speedup.iterrows():
        row = row.dropna().sort_index()
        if row.empty:
            continue
        plt.plot(row.index, row.values, marker="o", label=Path(matrix).name)

    plt.xlabel("Number of threads $p$")
    plt.ylabel(r"Speed-up $t_{1}/t_{p}$")
    plt.xticks(sorted(speedup.columns))
    plt.grid(True, linestyle="dotted", which="both")
    plt.legend(fontsize="small")
    plt.tight_layout()

    plt.savefig(out_pdf)
    if png:
        plt.savefig(out_pdf.with_suffix(".png"), dpi=150)
    plt.close()

    print(f"Plot saved as {out_pdf}")

# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--directory",
        type=Path,
        default=Path("."),
        help="Folder with benchmark_<threads>.csv files",
    )
    ap.add_argument(
        "--cols",
        type=str,
        default=",".join(DEFAULT_COLS),
        metavar="C1,C2,…",
        help="Comma-separated list of timing columns to process",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=5,
        metavar="N",
        help="Keep only the N matrices with largest nnz (0 = all)",
    )
    ap.add_argument(
        "--savepng",
        action="store_true",
        help="Save .png next to each .pdf plot",
    )
    args = ap.parse_args(argv)

    csv_files = find_csvs(args.directory)
    if not csv_files:
        ap.error(f"No benchmark_*.csv files found in {args.directory}")

    # Try to fetch nnz from ANY csv (needed for --top)
    first_csv = pd.read_csv(csv_files[0][1])
    nnz_ser = (
        first_csv[["matrix", "nnz"]].set_index("matrix")["nnz"]
        if "nnz" in map(str.lower, first_csv.columns)
        else pd.Series(dtype="int64")
    )

    for col in [c.strip().lower() for c in args.cols.split(",") if c.strip()]:
        print(f"\n>>> Processing timing column '{col}'")

        times = read_wide_table(csv_files, col)

        if not nnz_ser.empty and args.top:
            times = largest_n_matrices(times, nnz_ser, args.top)
            suffix_top = f"_top{args.top}" if args.top > 0 else ""
        else:
            suffix_top = ""

        speedup = compute_speedup(times)

        out_table = (
                args.directory
                / f"speedup_table_{col}{suffix_top}.csv"
        )
        speedup.to_csv(out_table)
        print(f"Wrote {out_table}  (rows: {len(speedup)}, cols: {len(speedup.columns)})")

        out_pdf = args.directory / f"speedup_{col}{suffix_top}.pdf"
        plot_speedup(speedup, out_pdf, png=args.savepng)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())