#!/usr/bin/env python
"""Run the E2E transformer baseline through the local TPSR/E2E wrapper."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semflow_sr.eval.baseline_runner import collect_tasks
from scripts.run_tpsr_manifest_baseline import _TPSRRunner


@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/benchmark_suites/benchmark_manifest.json")
    ap.add_argument("--suite", nargs="+", default=None)
    ap.add_argument("--root", default="data/benchmark_suites")
    ap.add_argument("--out", default="results/external_baselines")
    ap.add_argument("--tag", default="e2e")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tasks", "--max-tasks", type=int, default=None)
    ap.add_argument("--tpsr_root", default="external/TPSR")
    ap.add_argument("--model_path", default="symbolicregression/weights/model1.pt")
    ap.add_argument("--beam_size", type=int, default=1)
    ap.add_argument("--n_trees_to_refine", type=int, default=1)
    ap.add_argument("--max_input_points", type=int, default=64)
    ap.add_argument("--max_number_bags", type=int, default=1)
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--rollout", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()
    args.mode = "e2e"

    repo_root = Path.cwd()
    tpsr_root = (repo_root / args.tpsr_root).resolve()
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}_seed{args.seed}.json"
    tasks = collect_tasks(
        manifest=args.manifest,
        suites=args.suite,
        root=args.root,
        seed=int(args.seed),
        limit=args.max_tasks,
    )
    if out_path.exists() and not args.no_resume:
        results: dict[str, dict[str, Any]] = json.loads(out_path.read_text())
    else:
        results = {}

    sys.path.insert(0, str(tpsr_root))
    with _pushd(tpsr_root):
        try:
            runner = _TPSRRunner(args)
        except Exception as exc:  # noqa: BLE001
            runner = None
            init_error = (type(exc).__name__, str(exc), traceback.format_exc())
        else:
            init_error = None
        for task in tasks:
            if not args.no_resume and task.name in results and results[task.name].get("status") == "ok":
                continue
            started = time.perf_counter()
            try:
                if runner is None:
                    assert init_error is not None
                    raise RuntimeError(f"E2E runner initialization failed: {init_error[0]}: {init_error[1]}")
                row = runner.run_task(task)
                row["method"] = "E2E"
                row["runtime_sec"] = float(row.get("runtime_sec", 0.0)) + (time.perf_counter() - started)
            except Exception as exc:  # noqa: BLE001
                row = {
                    "task_id": task.name,
                    "suite": task.metadata.get("suite", _infer_suite(task.name)),
                    "method": "E2E",
                    "status": "failed",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                    "r2": 0.0,
                    "nmse": None,
                    "expression": "",
                    "ground_truth": task.expression,
                    "runtime_sec": time.perf_counter() - started,
                }
                if init_error is not None:
                    row["init_traceback"] = init_error[2]
            row.update({
                "task_id": task.name,
                "suite": task.metadata.get("suite", _infer_suite(task.name)),
                "domain": task.metadata.get("domain", "unknown"),
                "split": task.metadata.get("split", ""),
                "ground_truth": task.expression,
                "n_vars": int(task.X_train.shape[1]),
                "budget": {
                    "mode": "e2e",
                    "beam_size": args.beam_size,
                    "n_trees_to_refine": args.n_trees_to_refine,
                    "max_input_points": args.max_input_points,
                    "max_number_bags": args.max_number_bags,
                    "model_path": args.model_path,
                    "tpsr_root": str(tpsr_root),
                },
            })
            results[task.name] = row
            out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
            print(f"{task.name:32s} status={row['status']} r2={row.get('r2')}")
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"saved {out_path} ({len(results)} tasks)")


def _infer_suite(task_id: str) -> str:
    return task_id.split("/", 1)[0] if "/" in task_id else "unknown"


if __name__ == "__main__":
    main()
