#!/usr/bin/env python
"""Run the DSR/DSO baseline on materialized benchmarks. Run in the `dso37` env
(py3.7 + tensorflow 1.14; see docs/baselines/dsr.md).

Iterates over one or more suite dirs (each holds <task>/seed_X_{train,test}.csv) and
uses the sklearn-style DeepSymbolicRegressor API.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from semflow_sr.eval.baselines import run_dso
from semflow_sr.eval.baseline_runner import collect_tasks, run_baseline_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=None, help="legacy suite dirs, e.g. data/materialized/nguyen")
    ap.add_argument("--manifest", default=None, help="unified benchmark manifest JSON")
    ap.add_argument("--suite", nargs="+", default=None, help="manifest suite filter")
    ap.add_argument("--root", default=".", help="path root for manifest-relative split files")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=100000)
    ap.add_argument("--max_tasks", "--max-tasks", type=int, default=None,
                    help="optional task cap for smoke runs")
    ap.add_argument("--out", default="results/dso")
    ap.add_argument("--tag", default="dso")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tasks = collect_tasks(data=a.data, manifest=a.manifest, suites=a.suite, root=a.root,
                          seed=a.seed, limit=a.max_tasks)
    results = run_baseline_records(
        tasks,
        run_dso,
        out_path=out / f"{a.tag}_seed{a.seed}.json",
        method="DSO",
        budget={"n_samples": a.n_samples},
        kwargs={"n_samples": a.n_samples},
    )
    for name, item in results.items():
        print(f"{name:32s} r2={item['r2']}")
    print(f"saved {out}/{a.tag}_seed{a.seed}.json  ({len(results)} tasks)")


if __name__ == "__main__":
    main()
