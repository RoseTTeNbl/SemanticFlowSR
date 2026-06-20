"""Sanity-task helpers for external symbolic-regression baselines."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from ..data.benchmark_loader import SRTask


def make_sanity_tasks(n_train: int = 64, n_test: int = 256, seed: int = 0) -> list[SRTask]:
    rng = np.random.default_rng(seed)
    xtr = rng.uniform(-1.0, 1.0, size=(n_train, 1))
    xte = rng.uniform(-1.0, 1.0, size=(n_test, 1))
    x2tr = rng.uniform(-1.0, 1.0, size=(n_train, 2))
    x2te = rng.uniform(-1.0, 1.0, size=(n_test, 2))
    return [
        _task("sanity/y=x", xtr, xtr[:, 0], xte, xte[:, 0], "x0"),
        _task("sanity/y=x2", xtr, xtr[:, 0] ** 2, xte, xte[:, 0] ** 2, "x0**2"),
        _task("sanity/y=x+y", x2tr, x2tr[:, 0] + x2tr[:, 1], x2te, x2te[:, 0] + x2te[:, 1], "x0+x1"),
        _task("sanity/nguyen1", xtr, xtr[:, 0] ** 3 + xtr[:, 0] ** 2 + xtr[:, 0],
              xte, xte[:, 0] ** 3 + xte[:, 0] ** 2 + xte[:, 0], "x0**3+x0**2+x0"),
        _task("sanity/nguyen2", xtr, xtr[:, 0] ** 4 + xtr[:, 0] ** 3 + xtr[:, 0] ** 2 + xtr[:, 0],
              xte, xte[:, 0] ** 4 + xte[:, 0] ** 3 + xte[:, 0] ** 2 + xte[:, 0],
              "x0**4+x0**3+x0**2+x0"),
    ]


def summarize_sanity_results(records: dict[str, dict], threshold: float = 0.99) -> tuple[dict, list[dict]]:
    failed = []
    for task_id, item in records.items():
        r2 = float(item.get("r2_affine_refit", item.get("r2", 0.0)))
        if r2 < float(threshold):
            failed.append({"task_id": task_id, "r2": r2, "method": item.get("method", "")})
    summary = {
        "n_tasks": len(records),
        "threshold": float(threshold),
        "passed": len(records) - len(failed),
        "failed": len(failed),
        "pass_rate": (len(records) - len(failed)) / max(len(records), 1),
    }
    return summary, failed


def write_sanity_outputs(records: dict[str, dict], out_dir: str | Path, threshold: float = 0.99) -> tuple[dict, list[dict]]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary, failed = summarize_sanity_results(records, threshold=threshold)
    (out / "baseline_sanity_summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "baseline_failed_tasks.jsonl").open("w") as f:
        for item in failed:
            f.write(json.dumps(item) + "\n")
    return summary, failed


def _task(name, X_train, y_train, X_test, y_test, expr: str) -> SRTask:
    return SRTask(
        name=name,
        X_train=np.asarray(X_train, dtype=float),
        y_train=np.asarray(y_train, dtype=float),
        X_test=np.asarray(X_test, dtype=float),
        y_test=np.asarray(y_test, dtype=float),
        expression=expr,
        variable_names=[f"x{i}" for i in range(np.asarray(X_train).shape[1])],
        metadata={"suite": "sanity", "domain": "formula"},
    )
