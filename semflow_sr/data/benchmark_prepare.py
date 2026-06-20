"""Materialization helpers for benchmark suites."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from io import StringIO

import numpy as np
import pandas as pd

from .benchmark_manifest import BenchmarkTaskSpec


@dataclass(frozen=True)
class PMLBFilter:
    max_samples: int = 5000
    max_features: int = 20
    limit: int | None = None
    require_no_missing: bool = True


def materialize_arrays(
    *,
    task_id: str,
    suite: str,
    root: str | Path,
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    variable_names: list[str] | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    ground_truth: str | None = None,
    domain: str = "unknown",
    has_dummy_vars: bool = False,
    metrics: list[str] | None = None,
    split: str = "main",
    tags: list[str] | None = None,
    source: str | None = None,
    metadata: dict | None = None,
) -> BenchmarkTaskSpec:
    root = Path(root)
    variable_names = variable_names or [f"x{i}" for i in range(int(X_train.shape[1]))]
    task_dir = root / suite / _safe_name(name)
    task_dir.mkdir(parents=True, exist_ok=True)
    _write_split(task_dir / "train.csv", X_train, y_train, variable_names)
    _write_split(task_dir / "test.csv", X_test, y_test, variable_names)
    val_path = None
    if X_val is not None and y_val is not None:
        _write_split(task_dir / "val.csv", X_val, y_val, variable_names)
        val_path = str((task_dir / "val.csv").relative_to(root.parent))
    return BenchmarkTaskSpec(
        task_id=task_id,
        suite=suite,
        num_vars=len(variable_names),
        variable_names=list(variable_names),
        train_path=str((task_dir / "train.csv").relative_to(root.parent)),
        val_path=val_path,
        test_path=str((task_dir / "test.csv").relative_to(root.parent)),
        target_column="target",
        ground_truth=ground_truth,
        domain=domain,
        has_dummy_vars=bool(has_dummy_vars),
        metrics=metrics or ["r2", "nmse", "complexity"],
        split=split,
        tags=list(tags or []),
        source=source,
        metadata=dict(metadata or {}),
    )


def filter_pmlb_metadata(metadata: pd.DataFrame, cfg: PMLBFilter) -> list[str]:
    df = metadata.copy()
    dataset_col = _first_present(df, ["dataset", "dataset_name", "name"])
    task_col = _first_present(df, ["task", "task_type"])
    samples_col = _first_present(df, ["n_samples", "NumberOfInstances", "n_instances"])
    features_col = _first_present(df, ["n_features", "NumberOfFeatures", "n_variables"])
    missing_col = next((c for c in ["n_missing_values", "NumberOfMissingValues"] if c in df.columns), None)

    mask = df[task_col].astype(str).str.lower().eq("regression")
    mask &= df[samples_col].astype(float) <= float(cfg.max_samples)
    mask &= df[features_col].astype(float) <= float(cfg.max_features)
    if cfg.require_no_missing and missing_col is not None:
        mask &= df[missing_col].fillna(0).astype(float).eq(0)
    names = sorted(str(x) for x in df.loc[mask, dataset_col].tolist())
    if cfg.limit is not None:
        names = names[: max(int(cfg.limit), 0)]
    return names


def parse_srsd_text_table(text: str) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(StringIO(text), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        raise ValueError("SRSD text table must contain at least one input and one target column")
    return arr[:, :-1], arr[:, -1]


def srsd_problem_names_from_siblings(siblings: list[dict]) -> list[str]:
    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for item in siblings:
        filename = str(item.get("rfilename", ""))
        parts = filename.split("/", 1)
        split = "val" if len(parts) == 2 and parts[0] == "validation" else (parts[0] if len(parts) == 2 else "")
        if len(parts) != 2 or split not in by_split or not parts[1].endswith(".txt"):
            continue
        by_split[split].add(parts[1][:-4])
    return sorted(by_split["train"] & by_split["val"] & by_split["test"])


def _write_split(path: Path, X: np.ndarray, y: np.ndarray, columns: list[str]) -> None:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if X.shape[1] != len(columns):
        raise ValueError("variable_names length must match X.shape[1]")
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    df.to_csv(path, index=False)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", name).strip("_")


def _first_present(df: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"none of {names} found in metadata columns")
