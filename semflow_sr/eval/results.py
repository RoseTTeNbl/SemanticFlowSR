"""Experiment results: sample-level records + statistical summary.

Aligned with SRBench/TPSR reporting — per-task rows (r2, r2_zero, acc_tau, nmse,
complexity, simplicity, solved, energy_decrease) and aggregate stats over a benchmark.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from .evaluator import EvalReport

_AGG = ["r2", "r2_zero", "acc_tau", "nmse", "complexity", "simplicity",
        "steps", "energy_decrease", "solved"]


def summarize(records: list[dict]) -> dict:
    """Statistical metrics over sample-level records."""
    n = len(records)
    out = {"n_tasks": n}
    for k in _AGG:
        v = np.array([r[k] for r in records], dtype=float)
        out[f"{k}_mean"] = float(v.mean()) if n else 0.0
        out[f"{k}_median"] = float(np.median(v)) if n else 0.0
    out["solution_rate"] = float(np.mean([r["solved"] for r in records])) if n else 0.0
    return out


def save_results(reports: list[EvalReport], out_dir: str | Path, tag: str = "run") -> dict:
    """Write <tag>_samples.jsonl (one row per task) + <tag>_summary.json; return summary."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    records = [r.to_record() for r in reports]
    with open(out / f"{tag}_samples.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    summary = summarize(records)
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
