#!/usr/bin/env python
"""Run the PySR baseline on materialized formula benchmarks. Run in the `pysr` env."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from semflow_sr.eval.baselines import run_pysr


def _load(d: Path, seed: int):
    tr = pd.read_csv(d / f"seed_{seed}_train.csv"); te = pd.read_csv(d / f"seed_{seed}_test.csv")
    Xtr = tr.drop(columns=["target"]).to_numpy(); ytr = tr["target"].to_numpy()
    Xte = te.drop(columns=["target"]).to_numpy(); yte = te["target"].to_numpy()
    return Xtr, ytr, Xte, yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="suite dir, e.g. data/materialized/nguyen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--niterations", type=int, default=100)
    ap.add_argument("--out", default="results/pysr")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results = {}
    for d in sorted(Path(a.data).iterdir()):
        if not (d / f"seed_{a.seed}_train.csv").exists():
            continue
        Xtr, ytr, Xte, yte = _load(d, a.seed)
        results[d.name] = run_pysr(Xtr, ytr, Xte, yte, niterations=a.niterations)
        print(d.name, results[d.name]["r2"])
    (out / f"pysr_seed{a.seed}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
