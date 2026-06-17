#!/usr/bin/env python
"""Run the DEAP tree-GP baseline on materialized benchmarks. Run in the `deap` env.

Iterates over one or more suite dirs (each holds <task>/seed_X_{train,test}.csv).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

from semflow_sr.eval.baselines import run_deap


def _load(d: Path, seed: int):
    tr = pd.read_csv(d / f"seed_{seed}_train.csv"); te = pd.read_csv(d / f"seed_{seed}_test.csv")
    return (tr.drop(columns=["target"]).to_numpy(), tr["target"].to_numpy(),
            te.drop(columns=["target"]).to_numpy(), te["target"].to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="suite dirs, e.g. data/materialized/nguyen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--population_size", type=int, default=1000)
    ap.add_argument("--out", default="results/deap")
    ap.add_argument("--tag", default="deap")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results = {}
    for suite in a.data:
        for d in sorted(Path(suite).iterdir()):
            if not d.is_dir() or not (d / f"seed_{a.seed}_train.csv").exists():
                continue
            Xtr, ytr, Xte, yte = _load(d, a.seed)
            results[d.name] = run_deap(Xtr, ytr, Xte, yte, seed=a.seed,
                                       generations=a.generations, population_size=a.population_size)
            print(f"{d.name:24s} r2={results[d.name]['r2']:.4f}")
    (out / f"{a.tag}_seed{a.seed}.json").write_text(json.dumps(results, indent=2))
    print(f"saved {out}/{a.tag}_seed{a.seed}.json  ({len(results)} tasks)")


if __name__ == "__main__":
    main()
