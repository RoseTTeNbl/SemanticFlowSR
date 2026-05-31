#!/usr/bin/env python
"""Cache a PMLB subset (e.g. feynman) from the local external/pmlb clone into CSVs.

Reads datasets/<name>/<name>.tsv.gz; dirs without a data file are skipped.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

from semflow_sr.data.benchmark_loader import PMLBLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="external/pmlb")
    ap.add_argument("--pattern", default="feynman")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="data/pmlb/feynman")
    a = ap.parse_args()

    ds_dir = Path(a.root) / "datasets"
    names = sorted(p.name for p in ds_dir.iterdir() if p.is_dir() and a.pattern in p.name)
    if a.limit:
        names = names[: a.limit]
    loader = PMLBLoader(a.root)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for name in names:
        try:
            task = loader.load(name)
        except Exception as e:
            print(f"skip {name}: {e}"); continue
        d = out / name; d.mkdir(parents=True, exist_ok=True)
        cols = task.variable_names
        pd.DataFrame(task.X_train, columns=cols).assign(target=task.y_train).to_csv(d / "train.csv", index=False)
        pd.DataFrame(task.X_test, columns=cols).assign(target=task.y_test).to_csv(d / "test.csv", index=False)
        (d / "metadata.json").write_text(json.dumps({"name": name, "variables": cols}, indent=2))
        ok += 1
    print(f"cached {ok}/{len(names)} datasets to {out}")


if __name__ == "__main__":
    main()
