#!/usr/bin/env python
"""Create deterministic train/eval benchmark manifests.

The split is task-level and suite-stratified: each selected suite is shuffled
with a stable seed, then divided into train/eval manifests. Data files are not
copied; the generated manifests keep the original relative paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from semflow_sr.data.benchmark_manifest import (
    BenchmarkSuiteSpec,
    load_benchmark_manifest,
    write_benchmark_manifest,
)


def _stable_rng(seed: int, suite: str) -> random.Random:
    digest = hashlib.sha256(f"{int(seed)}:{suite}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _split_suite(tasks: list, *, suite: str, seed: int, train_fraction: float, min_eval: int) -> tuple[list, list]:
    items = list(tasks)
    _stable_rng(seed, suite).shuffle(items)
    if len(items) <= 1:
        return list(items), []
    eval_count = max(int(min_eval), int(round(len(items) * (1.0 - float(train_fraction)))))
    eval_count = min(max(eval_count, 1), len(items) - 1)
    eval_tasks = sorted(items[:eval_count], key=lambda task: task.task_id)
    train_tasks = sorted(items[eval_count:], key=lambda task: task.task_id)
    return train_tasks, eval_tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/benchmark_suites/benchmark_manifest.json")
    ap.add_argument("--out-dir", default="results/benchmark_splits/formula_dev_seed0_70_30")
    ap.add_argument("--suites", nargs="+", default=["nguyen", "constant", "livermore", "jin"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-fraction", type=float, default=0.7)
    ap.add_argument("--min-eval-per-suite", type=int, default=1)
    ap.add_argument("--tag", default="formula_dev_seed0_70_30")
    args = ap.parse_args()

    source = load_benchmark_manifest(args.manifest)
    selected = set(args.suites or source.suites.keys())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_suites = {}
    eval_suites = {}
    rows = []
    for suite, tasks in source.suites.items():
        if suite not in selected:
            continue
        train_tasks, eval_tasks = _split_suite(
            tasks,
            suite=suite,
            seed=int(args.seed),
            train_fraction=float(args.train_fraction),
            min_eval=int(args.min_eval_per_suite),
        )
        train_suites[suite] = train_tasks
        eval_suites[suite] = eval_tasks
        for split_name, split_tasks in (("train", train_tasks), ("eval", eval_tasks)):
            for task in split_tasks:
                rows.append({
                    "split": split_name,
                    "suite": suite,
                    "task_id": task.task_id,
                    "num_vars": task.num_vars,
                    "ground_truth": task.ground_truth or "",
                    "train_path": task.train_path,
                    "test_path": task.test_path,
                })

    meta = dict(source.metadata)
    meta.update({
        "source_manifest": str(args.manifest),
        "split_seed": int(args.seed),
        "train_fraction": float(args.train_fraction),
        "min_eval_per_suite": int(args.min_eval_per_suite),
        "selected_suites": sorted(selected),
    })
    train_manifest = BenchmarkSuiteSpec(version=source.version, suites=train_suites, metadata={**meta, "split": "train"})
    eval_manifest = BenchmarkSuiteSpec(version=source.version, suites=eval_suites, metadata={**meta, "split": "eval"})
    train_path = out_dir / f"{args.tag}_train_manifest.json"
    eval_path = out_dir / f"{args.tag}_eval_manifest.json"
    write_benchmark_manifest(train_manifest, train_path)
    write_benchmark_manifest(eval_manifest, eval_path)

    fields = ["split", "suite", "task_id", "num_vars", "ground_truth", "train_path", "test_path"]
    with (out_dir / f"{args.tag}_tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "tag": args.tag,
        "source_manifest": str(args.manifest),
        "train_manifest": str(train_path),
        "eval_manifest": str(eval_path),
        "seed": int(args.seed),
        "train_fraction": float(args.train_fraction),
        "suites": {},
        "train_tasks": int(sum(len(v) for v in train_suites.values())),
        "eval_tasks": int(sum(len(v) for v in eval_suites.values())),
    }
    for suite in sorted(train_suites):
        summary["suites"][suite] = {
            "train": len(train_suites[suite]),
            "eval": len(eval_suites[suite]),
            "train_task_ids": [task.task_id for task in train_suites[suite]],
            "eval_task_ids": [task.task_id for task in eval_suites[suite]],
        }
    (out_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
