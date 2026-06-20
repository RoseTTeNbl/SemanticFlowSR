"""Validation utilities for the unified benchmark manifest."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from .benchmark_manifest import BenchmarkSuiteSpec, BenchmarkTaskSpec, load_benchmark_manifest


@dataclass(frozen=True)
class BenchmarkValidationResult:
    summary: dict[str, Any]
    suite_rows: list[dict[str, Any]]
    task_rows: list[dict[str, Any]]
    failures: list[dict[str, Any]]

    @property
    def ok(self) -> bool:
        return int(self.summary.get("n_failed", 0)) == 0


def validate_benchmark_manifest(
    manifest_path: str | Path,
    *,
    root: str | Path = ".",
    suites: list[str] | None = None,
    require_val: bool = False,
) -> BenchmarkValidationResult:
    """Validate that all manifest entries can be loaded as finite numeric tasks."""
    manifest = load_benchmark_manifest(manifest_path)
    selected = set(suites or manifest.suites.keys())
    root = Path(root)
    task_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    suite_acc: dict[str, dict[str, Any]] = {}

    for suite, specs in manifest.suites.items():
        if suite not in selected:
            continue
        suite_acc.setdefault(suite, _empty_suite_row(suite))
        for spec in specs:
            row = _validate_task(spec, root=root, require_val=require_val)
            task_rows.append(row)
            _update_suite_row(suite_acc[suite], row)
            if row["status"] != "ok":
                failures.append({
                    "task_id": spec.task_id,
                    "suite": spec.suite,
                    "error": row["error"],
                    "error_type": row["error_type"],
                })

    suite_rows = [_finalize_suite_row(row) for row in suite_acc.values()]
    summary = _build_summary(manifest, selected, task_rows, suite_rows)
    return BenchmarkValidationResult(
        summary=summary,
        suite_rows=suite_rows,
        task_rows=task_rows,
        failures=failures,
    )


def write_validation_reports(result: BenchmarkValidationResult, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest_validation_summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True)
    )
    with (out_dir / "manifest_validation_failures.jsonl").open("w") as fh:
        for item in result.failures:
            fh.write(json.dumps(item, sort_keys=True) + "\n")
    pd.DataFrame(result.task_rows).to_csv(out_dir / "manifest_validation_tasks.csv", index=False)
    pd.DataFrame(result.suite_rows).to_csv(out_dir / "manifest_validation_suites.csv", index=False)


def _validate_task(spec: BenchmarkTaskSpec, *, root: Path, require_val: bool) -> dict[str, Any]:
    base = {
        "task_id": spec.task_id,
        "suite": spec.suite,
        "domain": spec.domain,
        "split": spec.split,
        "num_vars_manifest": int(spec.num_vars),
        "num_vars_columns": len(spec.variable_names),
        "train_rows": 0,
        "val_rows": 0,
        "test_rows": 0,
        "status": "ok",
        "error": "",
        "error_type": "",
    }
    try:
        if int(spec.num_vars) != len(spec.variable_names):
            raise ValueError(
                f"num_vars={spec.num_vars} does not match variable_names length={len(spec.variable_names)}"
            )
        train = _validate_split(spec, root=root, rel_path=spec.train_path, split_name="train")
        test = _validate_split(spec, root=root, rel_path=spec.test_path, split_name="test")
        val_rows = 0
        if spec.val_path:
            val_rows = _validate_split(spec, root=root, rel_path=spec.val_path, split_name="val")
        elif require_val:
            raise FileNotFoundError(f"missing val_path for {spec.task_id}")
        base.update({
            "train_rows": train,
            "val_rows": val_rows,
            "test_rows": test,
        })
    except Exception as exc:  # noqa: BLE001 - validation must report all task failures.
        base.update({
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
        })
    return base


def _validate_split(
    spec: BenchmarkTaskSpec,
    *,
    root: Path,
    rel_path: str,
    split_name: str,
) -> int:
    path = root / rel_path
    if not path.exists():
        raise FileNotFoundError(f"{split_name} split not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{split_name} split is empty: {path}")
    missing_vars = [col for col in spec.variable_names if col not in df.columns]
    if missing_vars:
        raise ValueError(f"missing variable columns in {path}: {missing_vars}")
    if spec.target_column not in df.columns:
        raise ValueError(f"missing target column {spec.target_column!r} in {path}")
    values = df[list(spec.variable_names) + [spec.target_column]].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{split_name} split contains NaN or Inf: {path}")
    return int(len(df))


def _empty_suite_row(suite: str) -> dict[str, Any]:
    return {
        "suite": suite,
        "n_tasks": 0,
        "n_valid": 0,
        "n_failed": 0,
        "num_vars_min": None,
        "num_vars_max": None,
        "train_rows": 0,
        "val_rows": 0,
        "test_rows": 0,
    }


def _update_suite_row(row: dict[str, Any], task_row: dict[str, Any]) -> None:
    row["n_tasks"] += 1
    if task_row["status"] == "ok":
        row["n_valid"] += 1
        n_vars = int(task_row["num_vars_manifest"])
        row["num_vars_min"] = n_vars if row["num_vars_min"] is None else min(row["num_vars_min"], n_vars)
        row["num_vars_max"] = n_vars if row["num_vars_max"] is None else max(row["num_vars_max"], n_vars)
        row["train_rows"] += int(task_row["train_rows"])
        row["val_rows"] += int(task_row["val_rows"])
        row["test_rows"] += int(task_row["test_rows"])
    else:
        row["n_failed"] += 1


def _finalize_suite_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["num_vars_min"] = 0 if out["num_vars_min"] is None else int(out["num_vars_min"])
    out["num_vars_max"] = 0 if out["num_vars_max"] is None else int(out["num_vars_max"])
    return out


def _build_summary(
    manifest: BenchmarkSuiteSpec,
    selected: set[str],
    task_rows: list[dict[str, Any]],
    suite_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    n_valid = sum(1 for row in task_rows if row["status"] == "ok")
    n_failed = len(task_rows) - n_valid
    dims: dict[str, int] = {}
    for row in task_rows:
        if row["status"] == "ok":
            key = str(int(row["num_vars_manifest"]))
            dims[key] = dims.get(key, 0) + 1
    return {
        "version": manifest.version,
        "selected_suites": sorted(selected),
        "n_suites": len(suite_rows),
        "n_tasks": len(task_rows),
        "n_valid": n_valid,
        "n_failed": n_failed,
        "ok": n_failed == 0,
        "tasks_by_num_vars": dict(sorted(dims.items(), key=lambda kv: int(kv[0]))),
        "tasks_by_suite": {row["suite"]: int(row["n_tasks"]) for row in suite_rows},
    }
