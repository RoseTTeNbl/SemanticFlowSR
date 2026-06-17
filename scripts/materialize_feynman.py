#!/usr/bin/env python
"""Materialize PMLB Feynman datasets into the shared CSV layout used by formula suites:
data/materialized/feynman/<name>/seed_X_{train,test}.csv  (cols + target).

PMLB Feynman are 100k rows; we subsample to keep baseline runtimes tractable.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="external/pmlb")
    ap.add_argument("--out", default="data/materialized/feynman")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--names", nargs="+", default=None, help="subset; default = all feynman_*")
    ap.add_argument("--max_vars", type=int, default=999, help="skip datasets with more vars")
    a = ap.parse_args()

    ds_dir = Path(a.root) / "datasets"
    names = a.names or sorted(p.name for p in ds_dir.iterdir() if p.name.startswith("feynman_"))
    n = a.n_train + a.n_test
    done = 0
    for name in names:
        cols = [c for c in pd.read_csv(ds_dir / name / f"{name}.tsv.gz", sep="\t",
                compression="gzip", nrows=1).columns if c != "target"]
        if len(cols) > a.max_vars:
            continue
        df = pd.read_csv(ds_dir / name / f"{name}.tsv.gz", sep="\t", compression="gzip")
        d = Path(a.out) / name; d.mkdir(parents=True, exist_ok=True)
        for seed in a.seeds:
            rng = np.random.default_rng(seed)
            sub = df.iloc[rng.permutation(len(df))[:n]].reset_index(drop=True)
            sub.iloc[:a.n_train][cols + ["target"]].to_csv(d / f"seed_{seed}_train.csv", index=False)
            sub.iloc[a.n_train:n][cols + ["target"]].to_csv(d / f"seed_{seed}_test.csv", index=False)
        (d / "metadata.json").write_text(json.dumps(
            {"name": name, "variables": cols, "suite": "feynman", "n_vars": len(cols)}, indent=2))
        done += 1
    print(f"materialized {done} feynman datasets to {a.out}")


if __name__ == "__main__":
    main()
