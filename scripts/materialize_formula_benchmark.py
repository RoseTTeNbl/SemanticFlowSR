#!/usr/bin/env python
"""Materialize formula benchmarks (Nguyen/Constant/Livermore/Jin) into CSVs per seed."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from semflow_sr.data.benchmark_loader import materialize_formula

CFG_DIR = Path(__file__).resolve().parents[1] / "configs" / "data" / "formula_benchmarks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+", default=["nguyen", "constant", "livermore", "jin"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="data/materialized")
    a = ap.parse_args()

    for suite in a.suite:
        entries = yaml.safe_load((CFG_DIR / f"{suite}.yaml").read_text())
        for entry in entries:
            entry.setdefault("suite", suite)
            for seed in a.seeds:
                task = materialize_formula(entry, seed)
                d = Path(a.out) / suite / task.name
                d.mkdir(parents=True, exist_ok=True)
                cols = task.variable_names
                pd.DataFrame(np.column_stack([task.X_train, task.y_train]),
                             columns=cols + ["target"]).to_csv(d / f"seed_{seed}_train.csv", index=False)
                pd.DataFrame(np.column_stack([task.X_test, task.y_test]),
                             columns=cols + ["target"]).to_csv(d / f"seed_{seed}_test.csv", index=False)
                (d / "metadata.json").write_text(json.dumps(
                    {"name": task.name, "expression": task.expression,
                     "variables": task.variable_names, "suite": suite}, indent=2))
            print(f"materialized {suite}/{entry['name']}")
    print("done")


if __name__ == "__main__":
    main()
