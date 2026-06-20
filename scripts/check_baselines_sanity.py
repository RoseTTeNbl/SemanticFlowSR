#!/usr/bin/env python
"""Run simple sanity tasks before trusting external baseline tables."""
from __future__ import annotations

import argparse
from pathlib import Path

from semflow_sr.eval.baseline_runner import run_baseline_records
from semflow_sr.eval.baseline_sanity import make_sanity_tasks, write_sanity_outputs
from semflow_sr.eval.baselines import run_deap, run_gplearn, run_pysr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["deap", "gplearn", "pysr"], required=True)
    ap.add_argument("--out", default="results/baseline_sanity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--population_size", type=int, default=500)
    ap.add_argument("--niterations", type=int, default=40)
    args = ap.parse_args()
    fn = {"deap": run_deap, "gplearn": run_gplearn, "pysr": run_pysr}[args.baseline]
    kwargs = (
        {"seed": args.seed, "generations": args.generations, "population_size": args.population_size}
        if args.baseline in {"deap", "gplearn"}
        else {"niterations": args.niterations}
    )
    out = Path(args.out)
    records = run_baseline_records(
        make_sanity_tasks(seed=args.seed),
        fn,
        out_path=out / f"{args.baseline}_sanity_records.json",
        method=args.baseline,
        budget=kwargs,
        kwargs=kwargs,
    )
    summary, failed = write_sanity_outputs(records, out, threshold=args.threshold)
    print(f"{args.baseline}: passed={summary['passed']} failed={summary['failed']} threshold={args.threshold}")
    for item in failed:
        print(f"  FAIL {item['task_id']} r2={item['r2']:.4f}")


if __name__ == "__main__":
    main()
