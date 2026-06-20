#!/usr/bin/env python
"""Evaluate Path-Posterior Semantic-Fisher action-flow checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from semflow_sr.actions.action_executor import ActionExecutor
from semflow_sr.actions.action_space import ActionSpace
from semflow_sr.data.benchmark_loader import FeynmanCSVLoader, load_materialized_task
from semflow_sr.data.benchmark_manifest import load_benchmark_manifest
from semflow_sr.eval.evaluator import EvalReport
from semflow_sr.eval.metrics import accuracy_rate, accuracy_tau, energy_decrease_ratio, nmse, r2_score, r2_zero, simplicity
from semflow_sr.eval.results import save_results
from semflow_sr.flow.semantic_fisher import semantic_fisher_sphere_step
from semflow_sr.models.semantic_transformer import SemanticTransformer, SemanticTransformerConfig
from semflow_sr.path_posterior.action_support import (
    STOP_ACTION_ID,
    append_stop_action,
    action_features_with_stop,
    action_semantic_effects_with_stop,
    healthy_action_ids,
    is_stop_action,
)
from semflow_sr.path_posterior.target_sampler import PriorConfig, build_p_init
from semflow_sr.registers.executor import evaluate_register_state
from semflow_sr.registers.state import init_register_state
from semflow_sr.semantics.energy import ActionEnergy, ActionEnergyConfig
from semflow_sr.semantics.projection import ProjectionBackend
from semflow_sr.sr.ops import NAME_TO_ID
from semflow_sr.sr.printer import to_string


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/benchmark_suites/benchmark_manifest.json")
    ap.add_argument("--manifest_root", default="data/benchmark_suites")
    ap.add_argument("--manifest_suite", nargs="+", default=["nguyen"])
    ap.add_argument("--legacy_87", action="store_true")
    ap.add_argument("--feynman_root", default="data/materialized/feynman")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_tasks", type=int, default=None)
    ap.add_argument("--out", default="results/semantic_fisher_flow")
    ap.add_argument("--tag", default="path_posterior")
    ap.add_argument("--max_steps", type=int, default=6)
    ap.add_argument("--step_dt", type=float, default=1.0)
    ap.add_argument("--disable_stop", action="store_true")
    ap.add_argument("--max_abs_semantic", type=float, default=1e6)
    ap.add_argument("--max_energy_growth", type=float, default=100.0)
    ap.add_argument("--max_support_size", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, cfg = _load_model(args.ckpt, args.device)
    _configure_torch_threads(cfg.get("runtime", {}))
    gen = cfg["gen"]
    pp_cfg = cfg.get("path_posterior", {})
    max_support_size = args.max_support_size
    if max_support_size is None:
        raw_cap = pp_cfg.get("max_support_size")
        max_support_size = None if raw_cap is None else int(raw_cap)
    ops_ids = [NAME_TO_ID[o] for o in gen["ops"]]
    prior_cfg = PriorConfig(
        stop_bias_base=float(pp_cfg.get("stop_bias_base", -2.0)),
        stop_bias_slope=float(pp_cfg.get("stop_bias_slope", 0.35)),
    )
    energy_cfg = ActionEnergyConfig(**cfg.get("energy", {"lambda_op": 0.0}))
    tasks = _load_legacy_87_tasks(args) if args.legacy_87 else _load_manifest_tasks(args)
    if args.limit_tasks is not None:
        tasks = tasks[: int(args.limit_tasks)]
    reports = []
    skipped = []
    max_vars = int(gen["num_vars"])
    for task in tasks:
        task_vars = int(task.X_train.shape[1])
        if task_vars > max_vars:
            skipped.append({"task": task.name, "reason": f"task has {task_vars} vars but checkpoint supports {max_vars}"})
            continue
        if task_vars + 1 + int(args.max_steps) > int(gen["K"]):
            skipped.append({"task": task.name, "reason": "checkpoint K lacks enough append registers"})
            continue
        report = _evaluate_task(
            model,
            task,
            model_num_vars=max_vars,
            K=int(gen["K"]),
            ops_ids=ops_ids,
            max_steps=args.max_steps,
            step_dt=args.step_dt,
            energy_cfg=energy_cfg,
            enable_stop=not args.disable_stop,
            max_abs_semantic=args.max_abs_semantic,
            max_energy_growth=args.max_energy_growth,
            max_support_size=max_support_size,
            prior_cfg=prior_cfg,
        )
        reports.append(report)
        print(f"{task.name} r2={report.r2:.4f} steps={report.steps}")
    summary = save_results(reports, args.out, args.tag)
    summary.update(_diagnostic_summary(reports))
    summary["skipped"] = len(skipped)
    out = Path(args.out)
    (out / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    (out / f"{args.tag}_skipped.json").write_text(json.dumps(skipped, indent=2))
    print(f"summary: {summary}")


def _load_model(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["meta"]["cfg"]
    model_cfg = SemanticTransformerConfig(**ckpt["meta"]["model_cfg"])
    model = SemanticTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def _configure_torch_threads(runtime_cfg: dict):
    num_threads = runtime_cfg.get("torch_num_threads")
    interop_threads = runtime_cfg.get("torch_num_interop_threads")
    if num_threads is not None:
        torch.set_num_threads(max(int(num_threads), 1))
    if interop_threads is not None:
        try:
            torch.set_num_interop_threads(max(int(interop_threads), 1))
        except RuntimeError:
            pass


def _load_manifest_tasks(args):
    manifest = load_benchmark_manifest(args.manifest)
    selected = set(args.manifest_suite)
    tasks = []
    for suite, specs in manifest.suites.items():
        if suite not in selected:
            continue
        for spec in specs:
            tasks.append(load_materialized_task(spec, root=args.manifest_root))
    return tasks


def _load_legacy_87_tasks(args):
    selected = ["nguyen", "constant", "livermore", "jin"]
    args2 = argparse.Namespace(**{**vars(args), "manifest_suite": selected})
    tasks = _load_manifest_tasks(args2)
    loader = FeynmanCSVLoader(args.feynman_root)
    tasks.extend(loader.load(name, seed=args.seed) for name in loader.names())
    return tasks


def _evaluate_task(
    model,
    task,
    *,
    model_num_vars,
    K,
    ops_ids,
    max_steps,
    step_dt,
    energy_cfg,
    enable_stop,
    max_abs_semantic,
    max_energy_growth,
    max_support_size,
    prior_cfg,
):
    device = next(model.parameters()).device
    x_train = torch.tensor(_pad_features(task.X_train, model_num_vars), dtype=torch.float32, device=device)
    y_train = torch.tensor(task.y_train, dtype=torch.float32, device=device)
    task_num_vars = int(task.X_train.shape[1])
    state, energy_trace, diagnostics = _rollout(
        model,
        x_train,
        y_train,
        num_vars=task_num_vars,
        K=K,
        ops_ids=ops_ids,
        max_steps=max_steps,
        step_dt=step_dt,
        energy_cfg=energy_cfg,
        enable_stop=enable_stop,
        max_abs_semantic=max_abs_semantic,
        max_energy_growth=max_energy_growth,
        max_support_size=max_support_size,
        prior_cfg=prior_cfg,
    )
    cols = [idx for idx, flag in enumerate(state.active.bool().tolist()) if flag] or list(range(len(state.exprs)))
    Btr_raw = evaluate_register_state(state, x_train).detach().cpu().numpy()
    x_test = torch.tensor(_pad_features(task.X_test, model_num_vars), dtype=torch.float32, device=device)
    Bte_raw = evaluate_register_state(state, x_test).detach().cpu().numpy()
    cols, coef = _fit_coef(Btr_raw, task.y_train, cols, B_test=Bte_raw)
    pred = _predict(Bte_raw, cols, coef)
    r2 = r2_score(task.y_test, pred)
    expr = _expr_string(state, coef, cols, task_num_vars)
    complexity = sum(state.exprs[c].complexity for i, c in enumerate(cols) if abs(coef[i]) > 1e-4)
    return EvalReport(
        task.name,
        r2,
        nmse(task.y_test, pred),
        int(complexity),
        expr,
        energy_trace,
        r2_zero=r2_zero(task.y_test, pred),
        acc_tau=accuracy_tau(task.y_test, pred),
        simplicity=simplicity(int(complexity)),
        steps=len(diagnostics),
        energy_decrease=energy_decrease_ratio(energy_trace),
        solved=accuracy_rate(r2),
        diagnostics=diagnostics,
        task_metadata=dict(task.metadata),
        active_columns=[int(c) for c in cols],
        readout_coefficients=[float(c) for c in coef.tolist()],
    )


def _pad_features(X, model_num_vars: int):
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D feature matrix, got shape {arr.shape}")
    if arr.shape[1] > int(model_num_vars):
        raise ValueError(f"task has {arr.shape[1]} vars but checkpoint supports {model_num_vars}")
    if arr.shape[1] == int(model_num_vars):
        return arr
    pad = np.zeros((arr.shape[0], int(model_num_vars) - arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def _rollout(
    model,
    x,
    y,
    *,
    num_vars,
    K,
    ops_ids,
    max_steps,
    step_dt,
    energy_cfg,
    enable_stop,
    max_abs_semantic,
    max_energy_growth,
    max_support_size,
    prior_cfg,
):
    state = init_register_state(num_vars, K, device=x.device)
    space = ActionSpace(K, ops_ids)
    executor = ActionExecutor(space)
    energy = ActionEnergy(space, energy_cfg)
    proj = ProjectionBackend(energy_cfg.projection, energy_cfg.rho)
    energy_trace = [_energy(state, x, y, proj)]
    diagnostics = []
    for step in range(max(int(max_steps), 0)):
        raw_action_ids = space.valid_actions(state).to(device=x.device)
        raw_support_size = int(raw_action_ids.numel())
        raw_action_ids = _cap_support(raw_action_ids, max_support_size)
        B = torch.nan_to_num(evaluate_register_state(state, x))
        action_ids = healthy_action_ids(
            energy,
            B,
            y,
            raw_action_ids,
            max_abs_semantic=max_abs_semantic,
            max_energy_growth=max_energy_growth,
        )
        filtered_count = raw_support_size - int(action_ids.numel())
        action_ids = append_stop_action(action_ids, enabled=enable_stop)
        if action_ids.numel() == 0:
            break
        effect = action_semantic_effects_with_stop(energy, B, y, action_ids)
        p0 = build_p_init(action_ids.detach().cpu(), step=step, cfg=prior_cfg).to(device=x.device, dtype=B.dtype)
        feats = action_features_with_stop(space, state, action_ids).to(device=x.device, dtype=B.dtype)
        zeros = torch.zeros_like(p0)
        ones = torch.ones_like(p0)
        with torch.no_grad():
            out = model(
                x=x.unsqueeze(0),
                y=y.unsqueeze(0),
                B=B.unsqueeze(0),
                p_lambda=p0.unsqueeze(0),
                lambda_value=torch.zeros(1, device=x.device, dtype=B.dtype),
                action_feats=feats.unsqueeze(0),
                energies=zeros.unsqueeze(0),
                weights=ones.unsqueeze(0),
                semantic_stats=torch.zeros(1, action_ids.numel(), 8, device=x.device, dtype=B.dtype),
                gram=effect.gram.unsqueeze(0),
                action_mask=torch.ones(1, action_ids.numel(), device=x.device, dtype=torch.bool),
            )
        p_final = semantic_fisher_sphere_step(p0, out.lograte_logits.squeeze(0), dt=float(step_dt))
        idx = int(p_final.argmax().item())
        action_id = int(action_ids[idx].item())
        selected_stop = is_stop_action(action_id)
        if not selected_stop:
            state = executor.execute_symbolic(state, action_id)
        energy_trace.append(_energy(state, x, y, proj))
        diagnostics.append({
            "step": step,
            "support_size": int(action_ids.numel()),
            "raw_support_size": raw_support_size,
            "filtered_count": filtered_count,
            "selected_action": action_id,
            "selected_stop": bool(selected_stop),
            "selected_prob": float(p_final[idx].detach().cpu().item()),
            "energy": energy_trace[-1],
        })
        if selected_stop:
            break
    return state, energy_trace, diagnostics


def _cap_support(action_ids: torch.Tensor, max_support_size: int | None) -> torch.Tensor:
    if max_support_size is None:
        return action_ids
    budget = max(int(max_support_size), 0)
    if action_ids.numel() <= budget:
        return action_ids
    if budget == 0:
        return action_ids[:0]
    sorted_ids = action_ids.sort().values
    idx = torch.linspace(
        0,
        sorted_ids.numel() - 1,
        steps=budget,
        device=sorted_ids.device,
    ).round().long()
    return sorted_ids[idx].unique(sorted=True)


def _energy(state, x, y, proj):
    B = torch.nan_to_num(evaluate_register_state(state, x))
    return float(proj.residual_energy(B, y).detach().cpu().item())


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
    return np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0) @ coef


def _diagnostic_summary(reports):
    if not reports:
        return {
            "stop_task_fraction": 0.0,
            "stop_decision_count": 0,
            "filtered_action_fraction_mean": 0.0,
        }
    stopped = 0
    stop_count = 0
    filtered_fracs = []
    for report in reports:
        task_stopped = False
        for diag in report.diagnostics:
            if diag.get("selected_stop"):
                task_stopped = True
                stop_count += 1
            raw = float(diag.get("raw_support_size", 0))
            filt = float(diag.get("filtered_count", 0))
            if raw > 0:
                filtered_fracs.append(filt / raw)
        stopped += int(task_stopped)
    return {
        "stop_task_fraction": float(stopped / max(len(reports), 1)),
        "stop_decision_count": int(stop_count),
        "filtered_action_fraction_mean": float(np.mean(filtered_fracs)) if filtered_fracs else 0.0,
    }


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
