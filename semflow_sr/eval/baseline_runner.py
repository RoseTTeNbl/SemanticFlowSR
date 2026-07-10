"""Shared task loading and result writing for external baseline scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import json
import signal
import time

import pandas as pd

from ..data.benchmark_loader import FeynmanCSVLoader, SRTask, load_materialized_task
from ..data.benchmark_manifest import load_benchmark_manifest


BaselineFn = Callable[[object, object, object, object], dict]


def collect_tasks(
    *,
    data: list[str | Path] | None = None,
    manifest: str | Path | None = None,
    suites: list[str] | None = None,
    root: str | Path = ".",
    seed: int = 0,
    limit: int | None = None,
    legacy_87: bool = False,
    feynman_root: str | Path = "data/materialized/feynman",
) -> list[SRTask]:
    if manifest is not None:
        tasks = _collect_manifest_tasks(manifest, suites=suites, root=root)
        if legacy_87:
            loader = FeynmanCSVLoader(feynman_root)
            tasks.extend(loader.load(name, seed=seed) for name in loader.names())
        return tasks[:int(limit)] if limit is not None else tasks
    if data:
        tasks = _collect_legacy_dir_tasks(data, seed=seed)
        return tasks[:int(limit)] if limit is not None else tasks
    raise ValueError("provide either --manifest or --data")


def run_baseline_records(
    tasks: list[SRTask],
    baseline_fn: BaselineFn,
    *,
    out_path: str | Path,
    method: str,
    budget: dict | None = None,
    kwargs: dict | None = None,
    continue_on_error: bool = True,
    resume: bool = False,
    timeout_sec: float | None = None,
) -> dict[str, dict]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and out_path.exists():
        out: dict[str, dict] = json.loads(out_path.read_text())
    else:
        out = {}
    for task in tasks:
        if resume and task.name in out and out[task.name].get("status", "ok") == "ok":
            continue
        started = time.perf_counter()
        try:
            item = _call_with_optional_timeout(
                lambda: _call_baseline(task, baseline_fn, kwargs or {}),
                timeout_sec=timeout_sec,
            )
            item = dict(item)
            item.setdefault("status", "ok")
            item.setdefault("error", "")
            item.setdefault("error_type", "")
        except Exception as exc:  # noqa: BLE001 - long benchmark runs should report all failures.
            if not continue_on_error:
                raise
            item = {
                "status": "failed",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "r2": 0.0,
                "r2_raw": 0.0,
                "r2_affine_refit": 0.0,
                "nmse": None,
                "nmse_affine_refit": None,
                "expression": "",
            }
        runtime = time.perf_counter() - started
        item = dict(item)
        item.setdefault("r2", 0.0)
        item.setdefault("nmse", 0.0)
        item.update({
            "task_id": task.name,
            "method": method,
            "suite": task.metadata.get("suite", _infer_suite(task.name)),
            "domain": task.metadata.get("domain", "unknown"),
            "split": task.metadata.get("split", ""),
            "has_dummy_vars": bool(task.metadata.get("has_dummy_vars", False)),
            "n_train": int(task.X_train.shape[0]),
            "n_test": int(task.X_test.shape[0]),
            "n_vars": int(task.X_train.shape[1]),
            "variable_names": list(task.variable_names),
            "budget": dict(budget or {}),
            "ground_truth": task.expression,
            "runtime_sec": runtime,
        })
        out[task.name] = item
        out_path.write_text(json.dumps(out, indent=2))
    out_path.write_text(json.dumps(out, indent=2))
    return out


class _TaskTimeout(Exception):
    pass


def _timeout_handler(_signum, _frame) -> None:
    raise _TaskTimeout("per-task baseline timeout")


def _call_with_optional_timeout(fn, *, timeout_sec: float | None):
    if timeout_sec is None or float(timeout_sec) <= 0:
        return fn()
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _call_baseline(task: SRTask, baseline_fn: BaselineFn, kwargs: dict) -> dict:
    if kwargs:
        try:
            return baseline_fn(task.X_train, task.y_train, task.X_test, task.y_test, **kwargs)
        except TypeError:
            return baseline_fn(task.X_train, task.y_train, task.X_test, task.y_test)
    return baseline_fn(task.X_train, task.y_train, task.X_test, task.y_test)


def _collect_manifest_tasks(
    manifest: str | Path,
    *,
    suites: list[str] | None,
    root: str | Path,
) -> list[SRTask]:
    manifest_obj = load_benchmark_manifest(manifest)
    selected = set(suites or manifest_obj.suites.keys())
    tasks: list[SRTask] = []
    for suite, specs in manifest_obj.suites.items():
        if suite not in selected:
            continue
        for spec in specs:
            tasks.append(load_materialized_task(spec, root=root))
    return tasks


def _collect_legacy_dir_tasks(data: list[str | Path], *, seed: int) -> list[SRTask]:
    tasks: list[SRTask] = []
    for suite_dir in data:
        suite_dir = Path(suite_dir)
        for task_dir in sorted(suite_dir.iterdir()):
            if not task_dir.is_dir() or not (task_dir / f"seed_{seed}_train.csv").exists():
                continue
            tr = pd.read_csv(task_dir / f"seed_{seed}_train.csv")
            te = pd.read_csv(task_dir / f"seed_{seed}_test.csv")
            meta = {}
            if (task_dir / "metadata.json").exists():
                meta = json.loads((task_dir / "metadata.json").read_text())
            cols = [c for c in tr.columns if c != "target"]
            suite = str(meta.get("suite", suite_dir.name))
            tasks.append(SRTask(
                f"{suite}/{task_dir.name}",
                tr[cols].to_numpy(float),
                tr["target"].to_numpy(float),
                te[cols].to_numpy(float),
                te["target"].to_numpy(float),
                meta.get("expression"),
                list(meta.get("variables", cols)),
                {"suite": suite, "seed": seed, **meta},
            ))
    return tasks


def _infer_suite(name: str) -> str:
    if "/" in name:
        return name.split("/", 1)[0]
    return "unknown"
