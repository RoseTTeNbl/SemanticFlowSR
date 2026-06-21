#!/usr/bin/env python
"""Evaluate Path-Posterior Semantic-Fisher action-flow checkpoints."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch

from semflow_sr.actions.action_executor import ActionExecutor
from semflow_sr.actions.action_space import ActionSpace
from semflow_sr.data.benchmark_loader import FeynmanCSVLoader, load_materialized_task
from semflow_sr.data.benchmark_manifest import load_benchmark_manifest
from semflow_sr.eval.evaluator import EvalReport
from semflow_sr.eval.metrics import accuracy_rate, accuracy_tau, energy_decrease_ratio, nmse, r2_score, r2_zero, simplicity
from semflow_sr.eval.results import save_results
from semflow_sr.flow.semantic_fisher import integrate_semantic_fisher_endpoint_path, semantic_fisher_sphere_step
from semflow_sr.models.semantic_transformer import SemanticTransformer, SemanticTransformerConfig
from semflow_sr.path_posterior.action_support import (
    STOP_ACTION_ID,
    append_stop_action,
    action_features_with_stop,
    action_semantic_effects_with_stop,
    healthy_action_ids,
    is_stop_action,
)
from semflow_sr.path_posterior.target_sampler import FutureGroupTargetConfig, PriorConfig, build_p_init, make_target_sampler
from semflow_sr.registers.executor import evaluate_register_state
from semflow_sr.registers.state import init_register_state
from semflow_sr.semantics.energy import ActionEnergy, ActionEnergyConfig
from semflow_sr.semantics.projection import ProjectionBackend
from semflow_sr.sr.ops import NAME_TO_ID
from semflow_sr.sr.printer import to_string


@dataclass
class EvalBundle:
    model: SemanticTransformer
    cfg: dict
    model_num_vars: int
    K: int
    ops_ids: list[int]
    energy_cfg: ActionEnergyConfig
    prior_cfg: PriorConfig
    enable_stop: bool
    max_support_size: int | None
    support_mode: str
    support_topk: int | None
    support_full_threshold: int | None
    pp_cfg: dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument(
        "--ckpt_by_vars",
        nargs="+",
        default=None,
        help="per-variable checkpoints, e.g. 1:ckpt_d1.pt d2=ckpt_d2.pt 3:ckpt_d3.pt",
    )
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
    ap.add_argument("--support_mode", default=None)
    ap.add_argument("--support_topk", type=int, default=None)
    ap.add_argument("--support_full_threshold", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt_map = _parse_ckpt_by_vars(args.ckpt_by_vars)
    if args.ckpt is None and not ckpt_map:
        ap.error("one of --ckpt or --ckpt_by_vars is required")
    bundles = _load_eval_bundles(args, ckpt_map)
    tasks = _load_legacy_87_tasks(args) if args.legacy_87 else _load_manifest_tasks(args)
    if args.limit_tasks is not None:
        tasks = tasks[: int(args.limit_tasks)]
    reports = []
    skipped = []
    for task in tasks:
        task_vars = int(task.X_train.shape[1])
        ckpt_key = _select_ckpt_key_for_task(task_vars, bundles)
        if ckpt_key is None:
            skipped.append({"task": task.name, "reason": f"no checkpoint covers {task_vars} variables"})
            continue
        bundle = bundles[int(ckpt_key)]
        if task_vars > bundle.model_num_vars:
            skipped.append({
                "task": task.name,
                "reason": f"task has {task_vars} vars but checkpoint supports {bundle.model_num_vars}",
                "ckpt_key": int(ckpt_key),
            })
            continue
        if task_vars + 1 + int(args.max_steps) > bundle.K:
            skipped.append({
                "task": task.name,
                "reason": "checkpoint K lacks enough append registers",
                "ckpt_key": int(ckpt_key),
                "K": int(bundle.K),
            })
            continue
        report = _evaluate_task(
            bundle.model,
            task,
            model_num_vars=bundle.model_num_vars,
            K=bundle.K,
            ops_ids=bundle.ops_ids,
            max_steps=args.max_steps,
            step_dt=args.step_dt,
            energy_cfg=bundle.energy_cfg,
            enable_stop=bundle.enable_stop,
            max_abs_semantic=args.max_abs_semantic,
            max_energy_growth=args.max_energy_growth,
            max_support_size=bundle.max_support_size,
            support_mode=bundle.support_mode,
            support_topk=bundle.support_topk,
            support_full_threshold=bundle.support_full_threshold,
            prior_cfg=bundle.prior_cfg,
            pp_cfg=bundle.pp_cfg,
        )
        reports.append(report)
        print(f"{task.name} d={task_vars} ckpt=d{ckpt_key} r2={report.r2:.4f} steps={report.steps}")
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


def _load_eval_bundles(args, ckpt_map: dict[int, str]) -> dict[int, EvalBundle]:
    mapping = dict(ckpt_map)
    if args.ckpt is not None:
        model, cfg = _load_model(args.ckpt, args.device)
        key = int(cfg["gen"]["num_vars"])
        mapping.setdefault(key, args.ckpt)
    bundles: dict[int, EvalBundle] = {}
    for key, path in sorted(mapping.items()):
        model, cfg = _load_model(path, args.device)
        if not bundles:
            _configure_torch_threads(cfg.get("runtime", {}))
        bundles[int(key)] = _build_eval_bundle(model, cfg, args)
    return bundles


def _build_eval_bundle(model, cfg: dict, args) -> EvalBundle:
    gen = cfg["gen"]
    pp_cfg = cfg.get("path_posterior", {})
    max_support_size = args.max_support_size
    if max_support_size is None:
        raw_cap = pp_cfg.get("max_support_size")
        max_support_size = None if raw_cap is None else int(raw_cap)
    support_mode = args.support_mode or str(pp_cfg.get("support_mode", "deterministic_cap"))
    support_topk = args.support_topk
    if support_topk is None and pp_cfg.get("support_topk") is not None:
        support_topk = int(pp_cfg.get("support_topk"))
    support_full_threshold = args.support_full_threshold
    if support_full_threshold is None and pp_cfg.get("support_full_threshold") is not None:
        support_full_threshold = int(pp_cfg.get("support_full_threshold"))
    energy_cfg = ActionEnergyConfig(**cfg.get("energy", {"lambda_op": 0.0}))
    return EvalBundle(
        model=model,
        cfg=cfg,
        model_num_vars=int(gen["num_vars"]),
        K=int(gen["K"]),
        ops_ids=[NAME_TO_ID[o] for o in gen["ops"]],
        energy_cfg=energy_cfg,
        prior_cfg=PriorConfig(
            mode=str(pp_cfg.get("p_init_mode", "stop_bias")),
            stop_bias_base=float(pp_cfg.get("stop_bias_base", -2.0)),
            stop_bias_slope=float(pp_cfg.get("stop_bias_slope", 0.35)),
        ),
        enable_stop=bool(pp_cfg.get("enable_stop", True)) and not bool(args.disable_stop),
        max_support_size=max_support_size,
        support_mode=support_mode,
        support_topk=support_topk,
        support_full_threshold=support_full_threshold,
        pp_cfg=dict(pp_cfg),
    )


def _parse_ckpt_by_vars(values: list[str] | None) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for raw in values or []:
        if ":" in raw:
            key, path = raw.split(":", 1)
        elif "=" in raw:
            key, path = raw.split("=", 1)
        else:
            raise ValueError(f"expected VAR:PATH or VAR=PATH entry, got {raw!r}")
        key = key.strip().lower()
        if key.startswith("d"):
            key = key[1:]
        var_count = int(key)
        if var_count <= 0:
            raise ValueError(f"checkpoint variable count must be positive, got {raw!r}")
        mapping[var_count] = path.strip()
    return mapping


def _select_ckpt_key_for_task(task_vars: int, ckpt_map: dict[int, object]) -> int | None:
    task_vars = int(task_vars)
    if task_vars in ckpt_map:
        return task_vars
    candidates = [int(k) for k in ckpt_map if int(k) >= task_vars]
    return min(candidates) if candidates else None


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
    support_mode,
    support_topk,
    support_full_threshold,
    prior_cfg,
    pp_cfg,
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
        support_mode=support_mode,
        support_topk=support_topk,
        support_full_threshold=support_full_threshold,
        prior_cfg=prior_cfg,
    )
    cols = [idx for idx, flag in enumerate(state.active.bool().tolist()) if flag] or list(range(len(state.exprs)))
    Btr_raw = evaluate_register_state(state, x_train).detach().cpu().numpy()
    x_test = torch.tensor(_pad_features(task.X_test, model_num_vars), dtype=torch.float32, device=device)
    Bte_raw = evaluate_register_state(state, x_test).detach().cpu().numpy()
    readout = _build_readout_report(
        state,
        Btr_raw,
        task.y_train,
        Bte_raw,
        task.y_test,
        num_vars=task_num_vars,
    )
    pred = np.asarray(readout["prediction"], dtype=float)
    r2 = float(readout["r2"])
    expr = str(readout["expression"])
    complexity = int(readout["complexity"])
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
        active_columns=[int(c) for c in readout["active_columns"]],
        readout_coefficients=[float(c) for c in readout["readout_coefficients"]],
        extra_metrics={k: v for k, v in readout.items() if k not in {
            "prediction",
            "r2",
            "nmse",
            "expression",
            "complexity",
            "active_columns",
            "readout_coefficients",
        }},
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
    support_mode,
    support_topk,
    support_full_threshold,
    prior_cfg,
):
    state = init_register_state(num_vars, K, device=x.device)
    space = ActionSpace(K, ops_ids)
    executor = ActionExecutor(space)
    energy = ActionEnergy(space, energy_cfg)
    target_sampler = make_target_sampler(
        str(pp_cfg.get("target_mode", "multi_step_group_advantage")),
        space,
        energy_cfg=energy_cfg,
        future_cfg=_future_target_cfg(pp_cfg),
    )
    proj = ProjectionBackend(energy_cfg.projection, energy_cfg.rho)
    energy_trace = [_energy(state, x, y, proj)]
    diagnostics = []
    rng = random.Random(12345)
    for step in range(max(int(max_steps), 0)):
        raw_action_ids = space.valid_actions(state).to(device=x.device)
        raw_support_size = int(raw_action_ids.numel())
        B = torch.nan_to_num(evaluate_register_state(state, x))
        full_rewards = energy.rewards(B, y, raw_action_ids) if raw_action_ids.numel() else torch.zeros(0, device=x.device)
        full_best_action = int(raw_action_ids[int(full_rewards.argmax().item())].item()) if full_rewards.numel() else -1
        raw_action_ids = _select_real_support(
            raw_action_ids,
            full_rewards,
            max_support_size=max_support_size,
            support_mode=support_mode,
            support_topk=support_topk,
            support_full_threshold=support_full_threshold,
            step=step,
        )
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
        target = target_sampler.build_target(
            state=state,
            action_ids=action_ids,
            p_init=p0.detach().cpu(),
            x=x,
            y=y,
            rng=rng,
        )
        q_hat = target.q_hat.to(device=x.device, dtype=B.dtype)
        teacher_path = integrate_semantic_fisher_endpoint_path(
            p0,
            q_hat,
            effect.gram,
            beta=float(pp_cfg.get("beta", 1.0)),
            gamma=float(pp_cfg.get("gamma", 0.1)),
            steps=int(pp_cfg.get("teacher_steps", 2)),
            gram_rank=pp_cfg.get("gram_rank", 8),
            gram_factors=effect.xi,
            q_smoothing=float(pp_cfg.get("target_smoothing", 1e-3)),
            teacher_mode=str(pp_cfg.get("teacher_mode", "endpoint_matching")),
        )
        teacher_final = teacher_path.policies[-1].to(device=x.device, dtype=B.dtype)
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
        support_oracle_rewards = _scores_for_action_ids(energy, B, y, action_ids)
        full_best_reward = float(full_rewards.max().detach().cpu().item()) if full_rewards.numel() else 0.0
        support_best_reward = float(support_oracle_rewards.max().detach().cpu().item()) if support_oracle_rewards.numel() else 0.0
        support_best_idx = int(support_oracle_rewards.argmax().item()) if support_oracle_rewards.numel() else -1
        target_scores = target.target_scores.to(device=x.device, dtype=B.dtype)
        target_ranks = _score_ranks(target_scores)
        q_idx = int(q_hat.argmax().item()) if q_hat.numel() else -1
        teacher_idx = int(teacher_final.argmax().item()) if teacher_final.numel() else -1
        selected_stop = is_stop_action(action_id)
        if not selected_stop:
            state = executor.execute_symbolic(state, action_id)
        energy_trace.append(_energy(state, x, y, proj))
        diagnostics.append({
            "step": step,
            "support_size": int(action_ids.numel()),
            "raw_support_size": raw_support_size,
            "filtered_count": filtered_count,
            "support_mode": str(support_mode),
            "full_best_action": full_best_action,
            "full_best_reward": full_best_reward,
            "full_best_in_support": bool((action_ids == full_best_action).any().detach().cpu().item()) if full_best_action >= 0 else False,
            "support_best_action": int(action_ids[support_best_idx].detach().cpu().item()) if support_best_idx >= 0 else -1,
            "support_best_reward": support_best_reward,
            "support_best_reward_gap": float(full_best_reward - support_best_reward),
            "target_top1_action": int(action_ids[q_idx].detach().cpu().item()) if q_idx >= 0 else -1,
            "target_top1_prob": float(q_hat[q_idx].detach().cpu().item()) if q_idx >= 0 else 0.0,
            "target_top1_reward": float(target_scores[q_idx].detach().cpu().item()) if q_idx >= 0 else 0.0,
            "target_top1_reward_rank": float(target_ranks[q_idx].detach().cpu().item()) if q_idx >= 0 else 0.0,
            "teacher_top1_action": int(action_ids[teacher_idx].detach().cpu().item()) if teacher_idx >= 0 else -1,
            "teacher_top1_prob": float(teacher_final[teacher_idx].detach().cpu().item()) if teacher_idx >= 0 else 0.0,
            "teacher_top1_reward": float(target_scores[teacher_idx].detach().cpu().item()) if teacher_idx >= 0 else 0.0,
            "teacher_top1_reward_rank": float(target_ranks[teacher_idx].detach().cpu().item()) if teacher_idx >= 0 else 0.0,
            "model_top1_action": action_id,
            "model_top1_prob": float(p_final[idx].detach().cpu().item()),
            "model_top1_reward": float(target_scores[idx].detach().cpu().item()) if target_scores.numel() else 0.0,
            "model_top1_reward_rank": float(target_ranks[idx].detach().cpu().item()) if target_ranks.numel() else 0.0,
            "target_teacher_top1_agreement": bool(q_idx == teacher_idx and q_idx >= 0),
            "teacher_model_top1_agreement": bool(teacher_idx == idx and teacher_idx >= 0),
            "selected_action": action_id,
            "selected_stop": bool(selected_stop),
            "selected_prob": float(p_final[idx].detach().cpu().item()),
            "energy": energy_trace[-1],
        })
        if selected_stop:
            break
    return state, energy_trace, diagnostics


def _future_target_cfg(pp_cfg: dict) -> FutureGroupTargetConfig:
    return FutureGroupTargetConfig(
        rank_eta=float(pp_cfg.get("weight_eta", 2.0)),
        smoothing=float(pp_cfg.get("target_smoothing", 1e-3)),
        score_to_shape=str(pp_cfg.get("score_to_shape", "group_exp")),
        advantage_eps=float(pp_cfg.get("advantage_eps", 1e-6)),
        advantage_clip=None if pp_cfg.get("advantage_clip", 5.0) is None else float(pp_cfg.get("advantage_clip", 5.0)),
        rollout_depth=int(pp_cfg.get("rollout_depth", 3)),
        rollouts_per_action=int(pp_cfg.get("rollouts_per_action", 1)),
        topk=int(pp_cfg.get("rollout_topk", 1)),
        max_rollout_support=pp_cfg.get("max_rollout_support", 16),
        terminal_op_penalty=0.0 if pp_cfg.get("terminal_op_penalty") is None else float(pp_cfg.get("terminal_op_penalty")),
        cache_path=pp_cfg.get("cache_path"),
        gp_population_path=pp_cfg.get("gp_population_path"),
        shape_samples=int(pp_cfg.get("shape_samples", 32)),
        gp_likelihood_weight=float(pp_cfg.get("gp_likelihood_weight", 1.0)),
        gp_fitness_weight=float(pp_cfg.get("gp_fitness_weight", 1.0)),
        importance_samples=None if pp_cfg.get("importance_samples") is None else int(pp_cfg.get("importance_samples")),
        mcmc_burn_in=int(pp_cfg.get("mcmc_burn_in", 16)),
    )


def _select_real_support(
    action_ids: torch.Tensor,
    rewards: torch.Tensor,
    *,
    max_support_size: int | None,
    support_mode: str,
    support_topk: int | None,
    support_full_threshold: int | None,
    step: int,
) -> torch.Tensor:
    mode = str(support_mode).strip().lower()
    if mode in {"deterministic_cap", "id_cap"}:
        return _cap_support(action_ids, max_support_size)
    if mode == "adaptive_full":
        threshold = support_full_threshold if support_full_threshold is not None else max_support_size
        if threshold is None or action_ids.numel() <= int(threshold):
            return action_ids
        mode = "reward_topk_random"
    if mode in {"reward_topk_random", "mixed_topk_random", "topk_reward"}:
        return _reward_aware_support(
            action_ids,
            rewards,
            max_support_size=max_support_size,
            support_topk=support_topk,
            random_fill=(mode != "topk_reward"),
            step=step,
        )
    raise ValueError(f"unknown support_mode: {support_mode}")


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


def _reward_aware_support(
    action_ids: torch.Tensor,
    rewards: torch.Tensor,
    *,
    max_support_size: int | None,
    support_topk: int | None,
    random_fill: bool,
    step: int,
) -> torch.Tensor:
    if max_support_size is None or action_ids.numel() <= max_support_size:
        return action_ids
    budget = max(int(max_support_size), 0)
    if budget == 0:
        return action_ids[:0]
    topk = support_topk if support_topk is not None else max(1, budget // 2)
    topk = min(max(int(topk), 1), budget, int(action_ids.numel()))
    selected = [int(i) for i in torch.topk(rewards, topk).indices.detach().cpu().tolist()]
    if random_fill and len(selected) < budget:
        selected_set = set(selected)
        remaining = [i for i in range(int(action_ids.numel())) if i not in selected_set]
        slots = min(budget - len(selected), len(remaining))
        if slots > 0:
            gen = torch.Generator(device=action_ids.device)
            gen.manual_seed(17_171 + int(step) * 1_000_003)
            rem = torch.tensor(remaining, dtype=torch.long, device=action_ids.device)
            perm = torch.randperm(rem.numel(), generator=gen, device=action_ids.device)
            selected.extend(int(i) for i in rem[perm[:slots]].detach().cpu().tolist())
    selected = _unique(selected)[:budget]
    idx = torch.tensor(selected, dtype=torch.long, device=action_ids.device)
    return action_ids[idx]


def _scores_for_action_ids(energy: ActionEnergy, B: torch.Tensor, y: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
    ids = torch.as_tensor(action_ids, dtype=torch.long, device=B.device)
    scores = torch.zeros(ids.numel(), dtype=B.dtype, device=B.device)
    normal = ids != STOP_ACTION_ID
    if bool(normal.any().item()):
        scores[normal] = energy.rewards(B, y, ids[normal])
    if bool((~normal).any().item()):
        scores[~normal] = -energy.residual_energy(B, y)
    return scores


def _score_ranks(scores: torch.Tensor) -> torch.Tensor:
    s = torch.nan_to_num(torch.as_tensor(scores, dtype=torch.float32, device=scores.device))
    if s.numel() == 0:
        return s
    order = s.argsort(descending=True)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, s.numel() + 1, dtype=s.dtype, device=s.device)
    return ranks


def _unique(xs: list[int]) -> list[int]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _energy(state, x, y, proj):
    B = torch.nan_to_num(evaluate_register_state(state, x))
    return float(proj.residual_energy(B, y).detach().cpu().item())


def _build_readout_report(
    state,
    B_train,
    y_train,
    B_test,
    y_test,
    *,
    num_vars: int,
) -> dict:
    cols = [idx for idx, flag in enumerate(state.active.bool().tolist()) if flag] or list(range(len(state.exprs)))
    cols, dense_coef = _fit_coef(B_train, y_train, cols, B_test=B_test)
    dense_pred = _predict(B_test, cols, dense_coef)
    dense_r2 = r2_score(y_test, dense_pred)
    dense_nmse = nmse(y_test, dense_pred)
    dense_complexity = _readout_complexity(state, dense_coef, cols)
    dense_expr = _expr_string(state, dense_coef, cols, num_vars)
    report = {
        "dense_r2": float(dense_r2),
        "dense_nmse": float(dense_nmse),
        "dense_complexity": int(dense_complexity),
        "dense_expression": dense_expr,
        "dense_active_columns": [int(c) for c in cols],
        "dense_readout_coefficients": [float(c) for c in dense_coef.tolist()],
        "dense_num_terms": int(_num_terms(dense_coef)),
        **_readout_structure_stats(state, B_train, cols, dense_coef, num_vars),
    }
    report.update({
        "prediction": dense_pred,
        "r2": float(dense_r2),
        "nmse": float(dense_nmse),
        "expression": dense_expr,
        "complexity": int(dense_complexity),
        "active_columns": [int(c) for c in cols],
        "readout_coefficients": [float(c) for c in dense_coef.tolist()],
    })
    return report


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


def _readout_complexity(state, coef, cols) -> int:
    return int(sum(state.exprs[c].complexity for i, c in enumerate(cols) if abs(float(coef[i])) > 1e-4))


def _num_terms(coef) -> int:
    return int(sum(1 for value in coef[:-1] if abs(float(value)) > 1e-4))


def _readout_structure_stats(state, B, cols, coef, num_vars: int) -> dict:
    coeff = np.asarray(coef[:-1], dtype=float)
    abs_coeff = np.abs(coeff)
    active = [int(c) for i, c in enumerate(cols) if i < len(coeff) and abs(float(coeff[i])) > 1e-4]
    A = np.nan_to_num(np.asarray(B)[:, cols], nan=0.0, posinf=0.0, neginf=0.0) if cols else np.zeros((len(B), 0))
    return {
        "num_nonzero_dense_coeffs": int(len(active)),
        "coeff_l1_norm": float(abs_coeff.sum()) if abs_coeff.size else 0.0,
        "coeff_l2_norm": float(np.linalg.norm(coeff)) if coeff.size else 0.0,
        "max_abs_coeff": float(abs_coeff.max()) if abs_coeff.size else 0.0,
        "coefficient_cancellation_score": _cancellation_score(A, coeff),
        "readout_condition_number": _condition_number(A),
        "semantic_duplicate_count": _semantic_duplicate_count(A),
        "canonical_duplicate_count": _canonical_duplicate_count(state, active, num_vars),
        "near_duplicate_column_pairs": _near_duplicate_pairs(A),
        "num_protected_ops": _count_expression_fragments(state, active, num_vars, ("Abs(", "log(", "sqrt(")),
        "num_exp_log_sqrt_abs": _count_expression_fragments(state, active, num_vars, ("exp(", "log(", "sqrt(", "Abs(")),
        "num_linear_terms_after_readout": int(len(active)),
    }


def _cancellation_score(A: np.ndarray, coeff: np.ndarray) -> float:
    if A.size == 0 or coeff.size == 0:
        return 0.0
    contributions = A * coeff.reshape(1, -1)
    denom = float(np.mean(np.abs(contributions).sum(axis=1))) + 1e-12
    numer = float(np.mean(np.abs(contributions.sum(axis=1))))
    return float(max(0.0, 1.0 - numer / denom))


def _condition_number(A: np.ndarray) -> float:
    if A.size == 0 or A.shape[1] == 0:
        return 0.0
    try:
        return float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        return float("inf")


def _semantic_duplicate_count(A: np.ndarray) -> int:
    if A.size == 0:
        return 0
    signatures = {}
    duplicates = 0
    for j in range(A.shape[1]):
        col = A[:, j]
        norm = np.linalg.norm(col - col.mean())
        if norm <= 1e-12:
            key = ("const",)
        else:
            key = tuple(np.round((col - col.mean()) / norm, decimals=8).tolist())
        duplicates += int(key in signatures)
        signatures[key] = True
    return int(duplicates)


def _near_duplicate_pairs(A: np.ndarray, threshold: float = 0.999) -> int:
    if A.size == 0 or A.shape[1] < 2:
        return 0
    centered = A - A.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    count = 0
    for i in range(A.shape[1]):
        if norms[i] <= 1e-12:
            continue
        for j in range(i + 1, A.shape[1]):
            if norms[j] <= 1e-12:
                continue
            corr = abs(float(centered[:, i] @ centered[:, j]) / float(norms[i] * norms[j]))
            count += int(corr >= threshold)
    return int(count)


def _canonical_duplicate_count(state, cols, num_vars: int) -> int:
    seen = set()
    duplicates = 0
    for col in cols:
        text = to_string(state.exprs[int(col)], num_vars, simplify=True)
        duplicates += int(text in seen)
        seen.add(text)
    return int(duplicates)


def _count_expression_fragments(state, cols, num_vars: int, fragments: tuple[str, ...]) -> int:
    total = 0
    for col in cols:
        text = to_string(state.exprs[int(col)], num_vars, simplify=True)
        total += sum(text.count(fragment) for fragment in fragments)
    return int(total)


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
