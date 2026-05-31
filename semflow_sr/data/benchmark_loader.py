"""Unified SR task loader: formula-benchmark YAML and PMLB."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class SRTask:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    expression: str | None
    variable_names: list[str]
    metadata: dict = field(default_factory=dict)


def _sample(sampler: str, rng: np.random.Generator, n: int, d: int, rng_range):
    lo, hi = rng_range
    if sampler == "uniform":
        return rng.uniform(lo, hi, size=(n, d))
    if sampler == "normal":
        return rng.normal((lo + hi) / 2, (hi - lo) / 4, size=(n, d))
    raise ValueError(sampler)


def materialize_formula(entry: dict, seed: int) -> SRTask:
    """Build an SRTask from a formula-benchmark YAML entry."""
    import sympy as sp
    rng = np.random.default_rng(seed)
    variables = entry["variables"]
    d = len(variables)
    syms = sp.symbols(variables)
    if not isinstance(syms, (list, tuple)):
        syms = (syms,)
    f = sp.lambdify(syms, sp.sympify(entry["expr"]), "numpy")

    def build(split):
        cfg = entry[split]
        X = _sample(cfg.get("sampler", "uniform"), rng, cfg["n"], d, cfg["range"])
        y = np.asarray(f(*[X[:, i] for i in range(d)]), dtype=float)
        y = np.nan_to_num(np.broadcast_to(y, (X.shape[0],)).copy())
        return X, y

    Xtr, ytr = build("train")
    Xte, yte = build("test")
    return SRTask(entry["name"], Xtr, ytr, Xte, yte, entry["expr"], variables,
                  {"suite": entry.get("suite", ""), "seed": seed})


class PMLBLoader:
    """Load a PMLB dataset from a local clone (datasets/<name>/<name>.tsv.gz)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load(self, name: str, test_frac: float = 0.25, seed: int = 0) -> SRTask:
        import pandas as pd
        path = self.root / "datasets" / name / f"{name}.tsv.gz"
        df = pd.read_csv(path, sep="\t", compression="gzip")
        y = df["target"].to_numpy(dtype=float)
        X = df.drop(columns=["target"]).to_numpy(dtype=float)
        cols = [c for c in df.columns if c != "target"]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y)); cut = int(len(y) * (1 - test_frac))
        tr, te = idx[:cut], idx[cut:]
        return SRTask(name, X[tr], y[tr], X[te], y[te], None, cols, {"source": "pmlb"})
