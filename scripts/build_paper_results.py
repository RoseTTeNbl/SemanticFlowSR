#!/usr/bin/env python
"""Build paper-style result tables and compact figures.

This script intentionally avoids per-task curve plots. It aggregates baseline JSON
files and SemanticFlowSR sample JSONL files into total/suite tables and a few compact
figures suitable for paper drafts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np


@dataclass(frozen=True)
class MethodSpec:
    name: str
    group: str
    path: str
    kind: str


DEFAULT_METHODS = [
    MethodSpec("PySR", "Baselines", "results/pysr/pysr_all_seed0.json", "baseline_json"),
    MethodSpec("DEAP", "Baselines", "results/deap/deap_all_seed0.json", "baseline_json"),
    MethodSpec("DSO", "Baselines", "results/dso/dso_all_seed0.json", "baseline_json"),
    MethodSpec("Ours one-step reward", "SFSR ablations", "results/paper_runs/one_step_reward/all87_seed0_samples.jsonl", "samples_jsonl"),
    MethodSpec("Ours one-step ODE", "SFSR ablations", "results/paper_runs/one_step_ode/all87_seed0_samples.jsonl", "samples_jsonl"),
    MethodSpec("Ours future ODE (no GP)", "SFSR main", "results/paper_runs/future_ode/all87_seed0_samples.jsonl", "samples_jsonl"),
    MethodSpec("GP as rollout policy", "GP variants", "results/paper_runs/gp_rollout_policy/all87_seed0_samples.jsonl", "samples_jsonl"),
    MethodSpec("GP policy distillation", "GP variants", "results/paper_runs/gp_distill/all87_seed0_samples.jsonl", "samples_jsonl"),
]

TRAIN_CURVES = {
    "Ours one-step reward": [
        "checkpoints/train_curve_paper_onestep_d1.csv",
        "checkpoints/train_curve_paper_onestep_d2.csv",
        "checkpoints/train_curve_paper_onestep_d3.csv",
    ],
    "Ours future ODE (no GP)": [
        "checkpoints/train_curve_velocity_rollout_future_ode_d1.csv",
        "checkpoints/train_curve_velocity_rollout_future_ode_d2.csv",
        "checkpoints/train_curve_velocity_rollout_future_ode_d3.csv",
    ],
    "GP as rollout policy": [
        "checkpoints/train_curve_paper_gp_rollout_d1.csv",
        "checkpoints/train_curve_paper_gp_rollout_d2.csv",
        "checkpoints/train_curve_paper_gp_rollout_d3.csv",
    ],
}


def infer_suite(name: str, metadata: dict | None = None) -> str:
    if metadata and metadata.get("suite"):
        return str(metadata["suite"])
    for prefix, suite in [
        ("Nguyen-", "nguyen"),
        ("Constant-", "constant"),
        ("Livermore-", "livermore"),
        ("Jin-", "jin"),
    ]:
        if name.startswith(prefix):
            return suite
    return "feynman"


def load_baseline_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    records = []
    for name, item in data.items():
        r2 = max(float(item.get("r2", 0.0)), 0.0)
        records.append({
            "name": name,
            "suite": infer_suite(name),
            "r2": r2,
            "solved": bool(r2 >= 0.999),
            "diagnostics": [],
        })
    return records


def load_samples_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            meta = rec.get("task_metadata") or {}
            records.append({
                "name": rec["name"],
                "suite": infer_suite(rec["name"], meta),
                "n_vars": meta.get("n_vars"),
                "r2": float(rec.get("r2_zero", max(rec.get("r2", 0.0), 0.0))),
                "solved": bool(rec.get("solved", False)),
                "diagnostics": rec.get("diagnostics") or [],
            })
    return records


def load_records(spec: MethodSpec, root: Path) -> list[dict]:
    path = root / spec.path
    if not path.exists():
        return []
    if spec.kind == "baseline_json":
        return load_baseline_json(path)
    if spec.kind == "samples_jsonl":
        return load_samples_jsonl(path)
    raise ValueError(f"unknown method kind: {spec.kind}")


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {"coverage": 0, "r2_mean": 0.0, "r2_median": 0.0, "solution_rate": 0.0}
    r2 = np.array([r["r2"] for r in records], dtype=float)
    solved = np.array([float(r["solved"]) for r in records], dtype=float)
    return {
        "coverage": len(records),
        "r2_mean": float(r2.mean()),
        "r2_median": float(np.median(r2)),
        "solution_rate": float(solved.mean()),
    }


def diagnostic_summary(records: list[dict]) -> dict:
    values: dict[str, list[float]] = {
        "selected_reward_rank": [],
        "predicted_top1_reward_rank": [],
        "exact_semantic_fisher_top1_reward_rank": [],
        "plain_fisher_top1_reward_rank": [],
        "one_step_rollout_corr": [],
        "support_best_reward_gap": [],
    }
    for rec in records:
        for diag in rec.get("diagnostics") or []:
            for key in values:
                if key in diag:
                    value = float(diag[key])
                    if not math.isfinite(value):
                        continue
                    if key == "one_step_rollout_corr" and not (-1.0 <= value <= 1.0):
                        continue
                    values[key].append(value)
    return {f"{k}_mean": (mean(v) if v else 0.0) for k, v in values.items()}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_composition(records_by_method: list[list[dict]]) -> list[dict]:
    """Return one method-independent dataset composition table.

    Baseline result files do not carry variable counts, so prefer the first record set
    with ``n_vars`` metadata. Suite membership falls back to any covered method.
    """
    records = next((r for r in records_by_method if r and any(x.get("n_vars") is not None for x in r)), None)
    if records is None:
        records = next((r for r in records_by_method if r), [])
    if not records:
        return []

    rows = []

    def _row(suite: str, suite_records: list[dict]) -> dict:
        dims: dict[str, int] = {}
        for rec in suite_records:
            key = rec.get("n_vars")
            if key is not None:
                dims[str(key)] = dims.get(str(key), 0) + 1
        return {
            "suite": suite,
            "n_tasks": len(suite_records),
            "dims": json.dumps(dict(sorted(dims.items())), sort_keys=True),
        }

    rows.append(_row("all", records))
    for suite in sorted({r["suite"] for r in records}):
        rows.append(_row(suite, [r for r in records if r["suite"] == suite]))
    return rows


def build_tables(root: Path, methods: list[MethodSpec]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    total_rows = []
    suite_rows = []
    diagnostic_rows = []
    records_by_method = []
    for spec in methods:
        records = load_records(spec, root)
        records_by_method.append(records)
        total = aggregate(records)
        total_rows.append({"group": spec.group, "method": spec.name, **total})
        diagnostic_rows.append({"group": spec.group, "method": spec.name, **diagnostic_summary(records)})
        suites = sorted({r["suite"] for r in records})
        for suite in suites:
            suite_records = [r for r in records if r["suite"] == suite]
            suite_rows.append({"group": spec.group, "method": spec.name, "suite": suite, **aggregate(suite_records)})
    dataset_rows = _dataset_composition(records_by_method)
    return total_rows, suite_rows, dataset_rows, diagnostic_rows


def plot_totals(total_rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["method"] for r in total_rows if r["coverage"]]
    r2 = [r["r2_mean"] for r in total_rows if r["coverage"]]
    sol = [r["solution_rate"] for r in total_rows if r["coverage"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(labels)), 4.2))
    width = 0.38
    ax.bar(x - width / 2, r2, width, label="Mean R2")
    ax.bar(x + width / 2, sol, width, label="Solve rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "paper_total_r2_solution.png", dpi=160)
    plt.close(fig)


def plot_suite_solution(suite_rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [r["method"] for r in suite_rows]
    ordered_methods = list(dict.fromkeys(methods))
    suites = sorted(set(r["suite"] for r in suite_rows))
    solution_data = {(r["method"], r["suite"]): r["solution_rate"] for r in suite_rows}
    r2_data = {(r["method"], r["suite"]): r["r2_mean"] for r in suite_rows}
    fig, (ax_r2, ax_sol) = plt.subplots(2, 1, figsize=(max(8, 0.8 * len(ordered_methods)), 7.0), sharex=True)
    width = 0.8 / max(len(suites), 1)
    x = np.arange(len(ordered_methods))
    for i, suite in enumerate(suites):
        offset = x - 0.4 + width / 2 + i * width
        ax_r2.bar(offset, [r2_data.get((m, suite), 0.0) for m in ordered_methods], width, label=suite)
        ax_sol.bar(offset, [solution_data.get((m, suite), 0.0) for m in ordered_methods], width, label=suite)
    for ax, ylabel in [(ax_r2, "Mean R2"), (ax_sol, "Solve rate")]:
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    ax_sol.set_xticks(x)
    ax_sol.set_xticklabels(ordered_methods, rotation=30, ha="right")
    ax_r2.legend(ncols=min(len(suites), 5), fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "paper_suite_solution.png", dpi=160)
    plt.close(fig)


def plot_diagnostics(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.get("selected_reward_rank_mean", 0.0)]
    if not rows:
        return
    labels = [r["method"] for r in rows]
    selected = [r["selected_reward_rank_mean"] for r in rows]
    pred = [r["predicted_top1_reward_rank_mean"] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(labels)), 4.0))
    width = 0.38
    ax.bar(x - width / 2, selected, width, label="Selected reward rank")
    ax.bar(x + width / 2, pred, width, label="Pred top1 reward rank")
    ax.set_ylabel("Mean rank, lower is better")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "paper_action_ranking.png", dpi=160)
    plt.close(fig)


def load_train_curve(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        out.append({
            "step": int(float(row.get("step") or 0)),
            "loss": float(row.get("loss") or 0.0),
            "reward": None if row.get("reward") in {"", None} else float(row["reward"]),
        })
    return out


def plot_train_curves(root: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_reward) = plt.subplots(1, 2, figsize=(10, 3.8))
    wrote = False
    for method, rel_paths in TRAIN_CURVES.items():
        curves = [load_train_curve(root / p) for p in rel_paths]
        points = {}
        rewards = {}
        for curve in curves:
            for row in curve:
                points.setdefault(row["step"], []).append(row["loss"])
                if row["reward"] is not None:
                    rewards.setdefault(row["step"], []).append(row["reward"])
        if points:
            xs = sorted(points)
            ys = [mean(points[x]) for x in xs]
            ax_loss.plot(xs, ys, marker="o", label=method)
            wrote = True
        if rewards:
            xs = sorted(rewards)
            ys = [mean(rewards[x]) for x in xs]
            ax_reward.plot(xs, ys, marker="o", label=method)
            wrote = True
    if not wrote:
        plt.close(fig)
        return
    ax_loss.set_title("Training loss")
    ax_loss.set_xlabel("step")
    ax_loss.set_yscale("log")
    ax_loss.grid(alpha=0.25)
    ax_reward.set_title("Held-out reward/R2")
    ax_reward.set_xlabel("step")
    ax_reward.set_ylim(0.0, 1.05)
    ax_reward.grid(alpha=0.25)
    ax_reward.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "paper_train_loss_reward.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="results/paper")
    args = ap.parse_args()
    root = Path(args.root)
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    total_rows, suite_rows, dataset_rows, diagnostic_rows = build_tables(root, DEFAULT_METHODS)
    write_csv(out / "paper_total.csv", total_rows, ["group", "method", "coverage", "r2_mean", "r2_median", "solution_rate"])
    write_csv(out / "paper_by_suite.csv", suite_rows, ["group", "method", "suite", "coverage", "r2_mean", "r2_median", "solution_rate"])
    write_csv(out / "paper_dataset_composition.csv", dataset_rows, ["suite", "n_tasks", "dims"])
    diag_fields = ["group", "method"] + [k for k in diagnostic_rows[0].keys() if k not in {"group", "method"}]
    write_csv(out / "paper_diagnostics.csv", diagnostic_rows, diag_fields)
    plot_totals(total_rows, out)
    plot_suite_solution(suite_rows, out)
    plot_diagnostics(diagnostic_rows, out)
    plot_train_curves(root, out)
    (out / "method_groups.json").write_text(json.dumps([m.__dict__ for m in DEFAULT_METHODS], indent=2))
    print(f"saved paper results to {out}")


if __name__ == "__main__":
    main()
