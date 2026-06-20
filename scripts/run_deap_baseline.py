#!/usr/bin/env python
"""Run the DEAP tree-GP baseline on materialized benchmarks. Run in the `deap` env."""
from __future__ import annotations
import argparse
from pathlib import Path

from semflow_sr.eval.baselines import run_deap
from semflow_sr.eval.baseline_runner import collect_tasks, run_baseline_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=None, help="legacy suite dirs, e.g. data/materialized/nguyen")
    ap.add_argument("--manifest", default=None, help="unified benchmark manifest JSON")
    ap.add_argument("--suite", nargs="+", default=None, help="manifest suite filter")
    ap.add_argument("--root", default=".", help="path root for manifest-relative split files")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--population_size", type=int, default=1000)
    ap.add_argument("--max_tasks", "--max-tasks", type=int, default=None,
                    help="optional task cap for smoke runs")
    ap.add_argument("--out", default="results/deap")
    ap.add_argument("--tag", default="deap")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tasks = collect_tasks(data=a.data, manifest=a.manifest, suites=a.suite, root=a.root,
                          seed=a.seed, limit=a.max_tasks)
    results = run_baseline_records(
        tasks,
        run_deap,
        out_path=out / f"{a.tag}_seed{a.seed}.json",
        method="DEAP",
        budget={"generations": a.generations, "population_size": a.population_size},
        kwargs={"seed": a.seed, "generations": a.generations, "population_size": a.population_size},
    )
    for name, item in results.items():
        print(f"{name:32s} r2={item['r2']:.4f}")
    print(f"saved {out}/{a.tag}_seed{a.seed}.json  ({len(results)} tasks)")


if __name__ == "__main__":
    main()
