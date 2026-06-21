"""Experiment results: sample-level records + statistical summary.

Aligned with SRBench/TPSR reporting — per-task rows (r2, r2_zero, acc_tau, nmse,
complexity, simplicity, solved, energy_decrease) and aggregate stats over a benchmark.
"""
from __future__ import annotations
from pathlib import Path
import csv
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


def save_results(
    reports: list[EvalReport],
    out_dir: str | Path,
    tag: str = "run",
    make_plots: bool = False,
) -> dict:
    """Write <tag>_samples.jsonl (one row per task) + <tag>_summary.json; return summary."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    records = [r.to_record() for r in reports]
    with open(out / f"{tag}_samples.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    summary = summarize(records)
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    _save_metric_csv(records, out, tag)
    if make_plots:
        _save_metric_plots(records, out, tag)
    return summary


def _save_metric_csv(records: list[dict], out: Path, tag: str) -> None:
    fields = [
        "name", "r2", "r2_zero", "acc_tau", "nmse", "complexity", "simplicity",
        "steps", "energy_decrease", "solved",
    ]
    optional = [
        "dense_r2",
        "dense_nmse",
        "dense_complexity",
        "dense_num_terms",
        "num_nonzero_dense_coeffs",
        "coeff_l1_norm",
        "coeff_l2_norm",
        "max_abs_coeff",
        "coefficient_cancellation_score",
        "readout_condition_number",
        "semantic_duplicate_count",
        "canonical_duplicate_count",
        "near_duplicate_column_pairs",
    ]
    fields.extend([field for field in optional if any(field in rec for rec in records)])
    with open(out / f"{tag}_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in fields})


def _save_metric_plots(records: list[dict], out: Path, tag: str) -> None:
    if not records:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in records]
    x = np.arange(len(records))
    r2 = np.array([r["r2"] for r in records], dtype=float)
    solved = np.array([float(r["solved"]) for r in records], dtype=float)

    fig, ax = plt.subplots(figsize=(max(7, 0.32 * len(records)), 4))
    ax.plot(x, r2, "o-", label="R2")
    ax.plot(x, solved, "s--", label="solved")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("task")
    ax.set_ylabel("metric")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / f"{tag}_r2_curve.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for rec in records:
        trace = np.array(rec.get("energy_trace") or [], dtype=float)
        if trace.size == 0:
            continue
        denom = max(abs(float(trace[0])), 1e-12)
        ax.plot(np.arange(trace.size), trace / denom, alpha=0.35, linewidth=1.0)
    ax.set_xlabel("search step")
    ax.set_ylabel("normalized energy")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_energy_traces.png", dpi=130)
    plt.close(fig)
