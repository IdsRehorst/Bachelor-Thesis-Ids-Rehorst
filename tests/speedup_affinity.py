import os
import re
import sys
import argparse
import glob

import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Plot speedup vs nnz for task-affinity experiments")
    p.add_argument("dir_no_aff", help="Directory with CSVs without affinity")
    p.add_argument("dir_aff",   help="Directory with CSVs with affinity")
    p.add_argument("out_png",   help="Output plot filename (e.g. speedup.png)")
    return p.parse_args()

def thread_count_from_filename(fname):
    # extract the first integer in the basename
    m = re.search(r'(\d+)', os.path.basename(fname))
    return int(m.group(1)) if m else None

def load_and_compute_speedup(file_no, file_af):
    df_no = pd.read_csv(file_no)
    df_af = pd.read_csv(file_af)
    # merge on common keys
    on = ['matrix','n','nnz','nzr']
    df = pd.merge(df_no, df_af, on=on, suffixes=('_no','_af'))
    # compute speedup of no-affinity over affinity
    df['speedup_tasks'] = df['t_tasks_ms_no'] / df['t_tasks_ms_af']
    return df[['nnz','speedup_tasks']].sort_values('nnz')

def main():
    args = parse_args()

    files_no = sorted(glob.glob(os.path.join(args.dir_no_aff, "*.csv")))
    files_af = sorted(glob.glob(os.path.join(args.dir_aff,   "*.csv")))

    # map thread count to dataframe
    data = {}
    for fno in files_no:
        thr = thread_count_from_filename(fno)
        faf = os.path.join(args.dir_aff, os.path.basename(fno))
        if not os.path.exists(faf):
            print(f"[warning] cannot find matching affinity file for {fno}", file=sys.stderr)
            continue
        df = load_and_compute_speedup(fno, faf)
        data[thr] = df

    # plot
    plt.figure(figsize=(8,6))
    for thr, df in sorted(data.items()):
        plt.plot(df['nnz'], df['speedup_tasks'], marker='o', label=f"{thr} threads")

    plt.xscale('log')
    plt.xlabel(r"Number of non-zeros $N_{nz}$")
    plt.ylabel("Speedup (no‐affinity / affinity)")
    plt.title(r"Task‐Affinity Speedup vs. Matrix $N_{nz}$")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    print(f"Saved plot to {args.out_png}")

if __name__ == "__main__":
    main()