#!/usr/bin/env python
"""Run the gplearn baseline on materialized formula benchmarks. Run in the `gplearn` env."""
from __future__ import annotations
import argparse
from pathlib import Path

from semflow_sr.eval.baselines import run_gplearn
from semflow_sr.eval.baseline_runner import collect_tasks, run_baseline_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=None, help="legacy suite dirs")
    ap.add_argument("--manifest", default=None, help="unified benchmark manifest JSON")
    ap.add_argument("--suite", nargs="+", default=None, help="manifest suite filter")
    ap.add_argument("--root", default=".", help="path root for manifest-relative split files")
    ap.add_argument("--legacy_87", action="store_true", help="append legacy materialized Feynman tasks")
    ap.add_argument("--feynman_root", default="data/materialized/feynman")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--population_size", type=int, default=1000)
    ap.add_argument("--max_tasks", "--max-tasks", type=int, default=None,
                    help="optional task cap for smoke runs")
    ap.add_argument("--per_task_timeout_sec", type=float, default=180.0)
    ap.add_argument("--no_resume", action="store_true", help="overwrite existing result file instead of skipping completed tasks")
    ap.add_argument("--out", default="results/gplearn")
    ap.add_argument("--tag", default="gplearn")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tasks = collect_tasks(data=a.data, manifest=a.manifest, suites=a.suite, root=a.root,
                          seed=a.seed, limit=a.max_tasks, legacy_87=a.legacy_87,
                          feynman_root=a.feynman_root)
    results = run_baseline_records(
        tasks,
        run_gplearn,
        out_path=out / f"{a.tag}_seed{a.seed}.json",
        method="gplearn",
        budget={"generations": a.generations, "population_size": a.population_size},
        kwargs={"generations": a.generations, "population_size": a.population_size},
        resume=not a.no_resume,
        timeout_sec=float(a.per_task_timeout_sec),
    )
    for name, item in results.items():
        print(f"{name:32s} r2={item['r2']:.4f}")


if __name__ == "__main__":
    main()
