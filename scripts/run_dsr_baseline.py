#!/usr/bin/env python
"""Run the DSR/DSO baseline. DSR must be installed in its OWN conda env (see
docs/BASELINES.md); this script writes a DSR config pointing at a materialized CSV and
invokes the dso CLI. Run inside the `dso` env.
"""
from __future__ import annotations
import argparse, json, subprocess, tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to a train CSV (cols + target)")
    ap.add_argument("--out", default="results/dsr")
    ap.add_argument("--n_samples", type=int, default=200000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "task": {"task_type": "regression", "dataset": str(Path(a.csv).resolve()),
                  "function_set": ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"]},
        "training": {"n_samples": a.n_samples},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f); cfg_path = f.name
    print(f"DSR config -> {cfg_path}")
    try:
        subprocess.run(["python", "-m", "dso.run", cfg_path], check=True)
    except FileNotFoundError:
        print("dso not installed in this env. See docs/BASELINES.md to create the `dso` env.")


if __name__ == "__main__":
    main()
