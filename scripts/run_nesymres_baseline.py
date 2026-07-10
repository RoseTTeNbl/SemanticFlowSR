#!/usr/bin/env python
"""Run the local small-trained NeSymReS-style baseline adapter."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semflow_sr.eval.baseline_runner import collect_tasks, run_baseline_records
from semflow_sr.eval.small_learned_baselines import SmallBaselineConfig, make_small_baseline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/benchmark_suites/benchmark_manifest.json")
    ap.add_argument("--suite", nargs="+", default=None)
    ap.add_argument("--root", default="data/benchmark_suites")
    ap.add_argument("--out", default="results/external_baselines")
    ap.add_argument("--tag", default="nesymres_small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tasks", "--max-tasks", type=int, default=None)
    ap.add_argument("--repo_root", default="external/NeuralSymbolicRegressionThatScales")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--train_root", default="data/generated/symbolicgpt_large_2000_200_200")
    ap.add_argument("--train_splits", nargs="+", default=["train", "val"])
    ap.add_argument("--train_limit", type=int, default=None)
    ap.add_argument("--artifact", default="results/clean_benchmark_20260701/paper_complete_20260702/trained_small_models/nesymres_small_library.json")
    ap.add_argument("--candidate_limit", type=int, default=512)
    ap.add_argument("--prefilter_limit", type=int, default=2000)
    ap.add_argument("--eval_subset", type=int, default=64)
    ap.add_argument("--ridge", type=float, default=1.0e-8)
    ap.add_argument("--force_retrain", action="store_true")
    ap.add_argument("--per_task_timeout_sec", type=float, default=120.0)
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    tasks = collect_tasks(manifest=args.manifest, suites=args.suite, root=args.root, seed=int(args.seed), limit=args.max_tasks)
    cfg = SmallBaselineConfig(
        method="NeSymReS-small",
        train_root=args.train_root,
        train_splits=tuple(str(v) for v in args.train_splits),
        artifact=args.artifact,
        train_limit=args.train_limit,
        candidate_limit=int(args.candidate_limit),
        prefilter_limit=int(args.prefilter_limit),
        eval_subset=int(args.eval_subset),
        ridge=float(args.ridge),
        seed=int(args.seed),
        force_retrain=bool(args.force_retrain),
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run_baseline_records(
        tasks,
        make_small_baseline(cfg),
        out_path=out_dir / f"{args.tag}_seed{args.seed}.json",
        method="NeSymReS-small",
        budget={
            "adapter": "small_trained_nesymres_semantic_retrieval",
            "repo_root": args.repo_root,
            "official_checkpoint": args.checkpoint,
            "train_root": args.train_root,
            "train_splits": list(args.train_splits),
            "candidate_limit": int(args.candidate_limit),
            "prefilter_limit": int(args.prefilter_limit),
        },
        resume=not bool(args.no_resume),
        timeout_sec=float(args.per_task_timeout_sec),
    )
    for name, item in results.items():
        print(f"{name:40s} status={item.get('status')} r2={item.get('r2')}")


if __name__ == "__main__":
    main()
