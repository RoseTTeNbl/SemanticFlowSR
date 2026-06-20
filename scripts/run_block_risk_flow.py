#!/usr/bin/env python
"""Evaluate block-only SFSR RiskFlow checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from semflow_sr.blocks.inference import rollout_block_risk_flow
from semflow_sr.data.benchmark_loader import FeynmanCSVLoader, load_materialized_task
from semflow_sr.data.benchmark_manifest import load_benchmark_manifest
from semflow_sr.eval.evaluator import EvalReport
from semflow_sr.eval.metrics import accuracy_rate, accuracy_tau, energy_decrease_ratio, nmse, r2_score, r2_zero, simplicity
from semflow_sr.eval.results import save_results
from semflow_sr.models.block_flow_model import BlockFlowModel, BlockFlowModelConfig
from semflow_sr.registers.executor import evaluate_register_state
from semflow_sr.semantics.energy import ActionEnergyConfig
from semflow_sr.sr.ops import NAME_TO_ID
from semflow_sr.sr.printer import to_string


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/benchmark_suites/benchmark_manifest.json")
    ap.add_argument("--manifest_root", default="data/benchmark_suites")
    ap.add_argument("--manifest_suite", nargs="+", default=["nguyen"])
    ap.add_argument("--legacy_87", action="store_true", help="evaluate formula-dev 34 plus data/materialized/feynman 53")
    ap.add_argument("--feynman_root", default="data/materialized/feynman")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_tasks", type=int, default=None)
    ap.add_argument("--out", default="results/block_risk_flow")
    ap.add_argument("--tag", default="block_h3")
    ap.add_argument("--block_pool_budget", type=int, default=128)
    ap.add_argument("--max_blocks", type=int, default=4)
    ap.add_argument("--integration_steps", type=int, default=2)
    ap.add_argument("--max_energy_growth", type=float, default=100.0)
    ap.add_argument("--max_abs_semantic", type=float, default=1e8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg = _load_model(args.ckpt, args.device)
    gen = cfg["gen"]
    block_size = int(cfg.get("block", {}).get("size", 3))
    ops_ids = [NAME_TO_ID[o] for o in gen["ops"]]
    energy_cfg = ActionEnergyConfig(**cfg.get("energy", {}))
    if args.legacy_87:
        tasks = _load_legacy_87_tasks(
            manifest_path=args.manifest,
            manifest_root=args.manifest_root,
            feynman_root=args.feynman_root,
            seed=args.seed,
        )
        if args.limit_tasks is not None:
            tasks = tasks[: int(args.limit_tasks)]
    else:
        tasks = _load_tasks(args.manifest, args.manifest_root, args.manifest_suite, args.limit_tasks)
    reports = []
    skipped = []
    for task in tasks:
        if not _has_register_capacity(
            num_vars=int(task.X_train.shape[1]),
            K=int(gen["K"]),
            block_size=block_size,
            max_blocks=args.max_blocks,
        ):
            skipped.append({
                "task": task.name,
                "reason": "task.num_vars plus committed H-blocks requires more registers than checkpoint K",
                "num_vars": int(task.X_train.shape[1]),
                "K": int(gen["K"]),
                "block_size": int(block_size),
                "max_blocks": int(args.max_blocks),
            })
            continue
        report = _evaluate_task(
            model,
            task,
            K=int(gen["K"]),
            ops_ids=ops_ids,
            block_size=block_size,
            block_pool_budget=args.block_pool_budget,
            max_blocks=args.max_blocks,
            integration_steps=args.integration_steps,
            max_energy_growth=args.max_energy_growth,
            max_abs_semantic=args.max_abs_semantic,
            energy_cfg=energy_cfg,
        )
        reports.append(report)
        print(f"{task.name} r2={report.r2:.4f} steps={report.steps}")
    summary = save_results(reports, args.out, args.tag)
    summary["skipped"] = len(skipped)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    (Path(args.out) / f"{args.tag}_skipped.json").write_text(json.dumps(skipped, indent=2))
    print(f"summary: {summary}")


def _load_model(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["meta"]["cfg"]
    model_cfg = BlockFlowModelConfig(**ckpt["meta"]["model_cfg"])
    model = BlockFlowModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def _load_tasks(manifest_path: str, root: str, suites: list[str], limit: int | None):
    manifest = load_benchmark_manifest(manifest_path)
    selected = set(suites)
    out = []
    for suite, specs in manifest.suites.items():
        if suite not in selected:
            continue
        for spec in specs:
            out.append(load_materialized_task(spec, root=root))
            if limit is not None and len(out) >= int(limit):
                return out
    return out


def _load_legacy_87_tasks(
    *,
    manifest_path: str,
    manifest_root: str,
    feynman_root: str,
    seed: int = 0,
):
    tasks = _load_tasks(
        manifest_path,
        manifest_root,
        ["nguyen", "constant", "livermore", "jin"],
        None,
    )
    feynman = FeynmanCSVLoader(feynman_root)
    for name in feynman.names():
        tasks.append(feynman.load(name, seed=seed))
    return tasks


def _has_register_capacity(*, num_vars: int, K: int, block_size: int, max_blocks: int) -> bool:
    required = int(num_vars) + 1 + int(block_size) * max(int(max_blocks), 0)
    return int(K) >= required


def _evaluate_task(
    model,
    task,
    *,
    K,
    ops_ids,
    block_size,
    block_pool_budget,
    max_blocks,
    integration_steps,
    max_energy_growth,
    max_abs_semantic,
    energy_cfg,
):
    x_train = torch.tensor(task.X_train, dtype=torch.float32)
    y_train = torch.tensor(task.y_train, dtype=torch.float32)
    res = rollout_block_risk_flow(
        model,
        x_train,
        y_train,
        num_vars=int(task.X_train.shape[1]),
        K=K,
        ops_ids=ops_ids,
        block_size=block_size,
        max_blocks=max_blocks,
        block_pool_budget=block_pool_budget,
        integration_steps=integration_steps,
        max_energy_growth=max_energy_growth,
        max_abs_semantic=max_abs_semantic,
        energy_cfg=energy_cfg,
    )
    cols = [idx for idx, flag in enumerate(res.state.active.bool().tolist()) if flag] or list(range(len(res.state.exprs)))
    Btr_raw = evaluate_register_state(res.state, x_train).detach().cpu().numpy()
    Xte = torch.tensor(task.X_test, dtype=torch.float32)
    Bte_raw = evaluate_register_state(res.state, Xte).detach().cpu().numpy()
    cols, coef = _fit_coef(Btr_raw, task.y_train, cols, B_test=Bte_raw)
    pred = _predict(Bte_raw, cols, coef)
    r2 = r2_score(task.y_test, pred)
    expr = _expr_string(res.state, coef, cols, int(task.X_train.shape[1]))
    complexity = sum(res.state.exprs[c].complexity for i, c in enumerate(cols) if abs(coef[i]) > 1e-4)
    return EvalReport(
        task.name,
        r2,
        nmse(task.y_test, pred),
        int(complexity),
        expr,
        res.energy_trace,
        r2_zero=r2_zero(task.y_test, pred),
        acc_tau=accuracy_tau(task.y_test, pred),
        simplicity=simplicity(int(complexity)),
        steps=res.steps,
        energy_decrease=energy_decrease_ratio(res.energy_trace),
        solved=accuracy_rate(r2),
        diagnostics=res.diagnostics,
        task_metadata=dict(task.metadata),
        active_columns=[int(c) for c in cols],
        readout_coefficients=[float(c) for c in coef.tolist()],
    )


def _fit_coef(B, y, cols, *, B_test=None, rho: float = 1e-6, max_norm: float = 1e8):
    cols = _select_healthy_cols(B, cols, B_test=B_test, max_norm=max_norm)
    A = np.concatenate([B[:, cols], np.ones((B.shape[0], 1))], axis=1)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    yy = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    G = A.T @ A + float(rho) * np.eye(A.shape[1])
    try:
        coef = np.linalg.solve(G, A.T @ yy)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(A, yy, rcond=None)[0]
    return cols, np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)


def _select_healthy_cols(B, cols, *, B_test=None, max_norm: float = 1e8):
    arrays = [np.asarray(B)]
    if B_test is not None:
        arrays.append(np.asarray(B_test))
    merged = np.concatenate(arrays, axis=0)
    keep = []
    for col in cols:
        values = merged[:, int(col)]
        norm = np.linalg.norm(values) if np.all(np.isfinite(values)) else np.inf
        if np.isfinite(norm) and 0.0 < norm < float(max_norm):
            keep.append(int(col))
    return keep or [int(c) for c in cols]


def _predict(B, cols, coef):
    A = np.concatenate([B[:, cols], np.ones((B.shape[0], 1))], axis=1)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    pred = A @ coef
    return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)


def _expr_string(state, coef, cols, num_vars):
    terms = [
        f"{coef[i]:.4g}*({to_string(state.exprs[c], num_vars, simplify=True)})"
        for i, c in enumerate(cols)
        if abs(coef[i]) > 1e-4
    ]
    if abs(coef[-1]) > 1e-4:
        terms.append(f"{coef[-1]:.4g}")
    return " + ".join(terms) if terms else "0"


if __name__ == "__main__":
    main()
