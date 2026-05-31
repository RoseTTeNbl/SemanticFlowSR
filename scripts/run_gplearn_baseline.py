#!/usr/bin/env python
"""Run the gplearn baseline on materialized formula benchmarks. Run in the `gplearn` env."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

from semflow_sr.eval.baselines import run_gplearn


def _load(d: Path, seed: int):
    tr = pd.read_csv(d / f"seed_{seed}_train.csv"); te = pd.read_csv(d / f"seed_{seed}_test.csv")
    return (tr.drop(columns=["target"]).to_numpy(), tr["target"].to_numpy(),
            te.drop(columns=["target"]).to_numpy(), te["target"].to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gplearn")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results = {}
    for d in sorted(Path(a.data).iterdir()):
        if not (d / f"seed_{a.seed}_train.csv").exists():
            continue
        results[d.name] = run_gplearn(*_load(d, a.seed))
        print(d.name, results[d.name]["r2"])
    (out / f"gplearn_seed{a.seed}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
