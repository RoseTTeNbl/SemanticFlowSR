"""Train Edge-Parameterized Semantic Flow on synthetic tasks."""
from __future__ import annotations

import argparse
import csv
import itertools
import random
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from ..data.symbolicgpt_subset import load_symbolicgpt_subset_tasks
from ..data.synthetic_generator import GenConfig, generate_expression, sample_probe_xy
from ..sr.ast import Expr, eval_expr
from ..sr.ops import get_op
from ..sr.parser import parse_formula
from ..sr.printer import to_string
from .benchmark import load_edge_flow_benchmark_tasks, task_tensors
from .closure_targets import (
    LEAF_PRODUCTION_ID,
    OPERATOR_FAMILY_IDS,
    OperatorClassificationRecord,
    ProductionRecord,
    SourceClassificationRecord,
    STOP_PRODUCTION_ID,
    StructuralClosureTargetBundle,
    StructuralSetProdTargetBundle,
    build_structural_closure_targets,
    build_structural_dynamic_oracle_targets,
    build_structural_setprod_targets,
)
from .natural_prior import construction_source_prior
from .conditional import (
    ConditionalEdgeFlowConfig,
    ConditionalEdgeFlowModel,
    ConditionalEdgeFlowSampler,
    conditional_elite_policy_loss,
)
from .circuit_sampler import CircuitSample, CircuitSampler
from .dataset import EdgeFlowBuildConfig, _target_rewards, build_edge_flow_records
from .gt_neighborhood import build_gt_neighborhood_samples
from .global_semantic import (
    augment_grid,
    build_gt_rewrite_candidates,
    build_gt_equivalence_target,
    classify_gt_proxy_samples,
    global_projection_loss,
    gt_proxy_projection_loss,
    semantic_kernel_matrix,
    semantic_vectors,
    semantic_flow_matching_loss,
)
from .global_theta import GlobalThetaNetwork, compile_expr_to_edge_sample, theta_vector_to_distribution
from .local_targets import InputSelectionFlowRecord, build_local_input_targets, decompose_formula_terms
from .model import EdgeFlowModel, EdgeFlowModelConfig, edge_flow_loss
from .path_compiler import compile_expr_to_spff_sample, compile_formula_to_spff_sample
from .pullback_chart import chart_regularization, project_tangent
from .pullback_teacher import build_pullback_teacher, build_simplex_teacher
from .proposals import load_diffusion_formula_proposals, simple_gp_proposals
from .reward import RewardConfig, evaluate_expression_rewards
from .semantic_teacher import decision_trace_rank, semantic_teacher_loss_for_samples
from .structure_posterior import (
    normalize_log_weights,
    structure_conditioned_log_weight,
    structure_similarity_score,
)
from .template import RegisterOperatorTemplate
from .theta_flow import (
    ThetaVelocityNetwork,
    center_theta_vector_by_template,
    simplex_theta_path,
    theta_semantic_pullback_loss,
)
from .term_graph import replay_term_graph_sample


@dataclass
class ConditionalTrainTask:
    task_id: str
    x: torch.Tensor
    y: torch.Tensor
    num_vars: int
    ground_truth: str = ""


def run(cfg: dict) -> Path:
    algorithm = str(cfg.get("algorithm", "semantic_pullback_fisher_flow")).lower()
    if algorithm in {"fixed_theta_edge_flow", "edge_parameterized_semantic_flow_matching"}:
        return _run_fixed_theta(cfg)
    if algorithm in {"conditional_semantic_edge_flow", "csef"}:
        return _run_conditional(cfg)
    if algorithm in {"semantic_pullback_fisher_flow", "spff", "spf", "conditional_semantic_edge_flow_spff"}:
        return _run_spff_conditional(cfg)
    if algorithm in {
        "semantic_global_projection",
        "global_semantic_projection",
        "complete_expression_projection",
    }:
        return _run_semantic_global_projection(cfg)
    if algorithm in {
        "semantic_flow_matching",
        "semantic_fm",
        "complete_expression_semantic_fm",
    }:
        return _run_semantic_flow_matching(cfg)
    if algorithm in {
        "explicit_global_theta_projection",
        "global_theta_projection",
        "theta_global_projection",
    }:
        return _run_explicit_global_theta_projection(cfg)
    if algorithm in {
        "direct_theta_gt_proxy_overfit",
        "direct_theta_overfit",
        "fixed_pool_theta_overfit",
    }:
        return _run_direct_theta_gt_proxy_overfit(cfg)
    if algorithm in {
        "amortized_theta_fixed_pool_overfit",
        "fixed_pool_global_theta_overfit",
        "amortized_global_theta_overfit",
    }:
        return _run_amortized_theta_fixed_pool_overfit(cfg, oracle_mode=False)
    if algorithm in {
        "theta_semantic_pullback_flow",
        "theta_space_semantic_pullback_flow",
        "fixed_pool_theta_semantic_flow",
    }:
        return _run_theta_semantic_pullback_flow(cfg)
    if algorithm in {
        "task_id_theta_oracle_overfit",
        "task_id_global_theta_oracle",
        "theta_oracle_task_id",
    }:
        return _run_amortized_theta_fixed_pool_overfit(cfg, oracle_mode=True)
    if algorithm in {"structural_closure", "simple_structural_closure", "term_structural_closure"}:
        return _run_structural_closure(cfg)
    if algorithm in {"structural_closure_setprod", "setprod", "term_structural_closure_setprod"}:
        return _run_structural_closure_setprod(cfg, algorithm_name="structural_closure_setprod")
    if algorithm in {
        "structural_closure_dynamic_oracle",
        "dynamic_oracle",
        "term_structural_closure_dynamic_oracle",
    }:
        return _run_structural_closure_setprod(cfg, algorithm_name="structural_closure_dynamic_oracle")
    if algorithm in {
        "structural_closure_setprod_flow",
        "setprod_flow",
        "term_structural_closure_setprod_flow",
        "structural_closure_setprod_simplex_flow",
    }:
        return _run_structural_closure_setprod(cfg, algorithm_name="structural_closure_setprod_flow")
    raise ValueError(f"unknown edge-flow algorithm: {algorithm}")


def _run_conditional(cfg: dict) -> Path:
    runtime = cfg.get("runtime", {})
    _configure_threads(runtime)
    device = _resolve_device(runtime)
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    template = _template_from_cfg(cfg)
    model_cfg = ConditionalEdgeFlowConfig(
        num_vars=template.num_vars,
        hidden=int(cfg.get("model", {}).get("hidden", 96)),
        head_terms=int(cfg.get("model", {}).get("head_terms", cfg.get("head", {}).get("terms", 3))),
        branches_per_register=int(cfg.get("model", {}).get("branches_per_register", 1)),
        update_mode=str(cfg.get("model", {}).get("update_mode", "carry_write")),
        write_registers_per_layer=int(cfg.get("model", {}).get("write_registers_per_layer", 0)),
        exclude_base_head_candidates=bool(cfg.get("model", {}).get("exclude_base_head_candidates", cfg.get("train", {}).get("exclude_base_head_candidates", False))),
        enable_keep_option=bool(cfg.get("model", {}).get("enable_keep_option", False)),
        mask_duplicate_branches=bool(cfg.get("model", {}).get("mask_duplicate_branches", False)),
        include_base_source_pool=bool(cfg.get("model", {}).get("include_base_source_pool", True)),
        task_encoder=str(cfg.get("model", {}).get("task_encoder", "mean")),
    )
    model = ConditionalEdgeFlowModel(model_cfg).to(device)
    train_cfg = cfg.get("train", {})
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)))
    tasks = _training_tasks(cfg, template, rng)
    rows: list[dict] = []
    epochs = int(train_cfg.get("epochs", 2))
    samples_per_task = int(train_cfg.get("samples_per_task", 64))
    elite_k = int(train_cfg.get("elite_k", 8))
    method = str(train_cfg.get("sampler_method", train_cfg.get("method", "policy")))
    flow_steps = int(train_cfg.get("flow_steps", 4))
    rank_temperature = train_cfg.get("elite_rank_temperature", train_cfg.get("rank_temperature"))
    rank_temperature = None if rank_temperature is None else float(rank_temperature)
    head_fit_mode = str(train_cfg.get("head_fit_mode", "linear"))
    reward_cfg = RewardConfig(
        complexity_weight=float(train_cfg.get("complexity_weight", 0.001)),
        head_fit_mode=head_fit_mode,
    )
    task_batch_size = max(int(train_cfg.get("task_batch_size", 1)), 1)
    replay_capacity = max(int(train_cfg.get("replay_capacity", 0)), 0)
    replay_ratio = max(float(train_cfg.get("replay_ratio", 0.0)), 0.0)
    entropy_bonus = float(train_cfg.get("entropy_bonus", 0.0))
    unique_elites = bool(train_cfg.get("unique_elites", True))
    inject_gt_elite = bool(train_cfg.get("inject_gt_elite", False))
    objective = str(train_cfg.get("objective", "active_ancestry")).lower()
    probability_path_geometry = str(train_cfg.get(
        "probability_path_geometry",
        train_cfg.get("teacher_path_geometry", "fisher"),
    ))
    teacher_target_mode = _teacher_target_mode(train_cfg)
    default_semantic_gamma = 1.0 if str(train_cfg.get("teacher_path_geometry", "")).lower() in {
        "semantic",
        "semantic_pullback",
        "semantic_quotient",
        "semantic_geometry",
    } else 0.0
    semantic_calibration_gamma = float(train_cfg.get("semantic_calibration_gamma", default_semantic_gamma))
    proposal_sources = _as_list(train_cfg.get("proposal_sources", []))
    proposal_count = int(train_cfg.get("proposal_count", 0))
    proposal_compile_only_trainable = bool(train_cfg.get("proposal_compile_only_trainable", True))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", train_cfg.get("patience", 0)))
    validation_tasks = _early_stopping_tasks(cfg, template, rng) if early_stopping_patience > 0 else []
    best_validation_score: float | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    replay: list[ConditionalTrainTask] = []
    optimizer_step = 0
    for epoch in range(epochs):
        epoch_tasks = list(tasks)
        rng.shuffle(epoch_tasks)
        for batch_index, batch in enumerate(_task_batches(epoch_tasks, task_batch_size)):
            replay_count = min(len(replay), int(round(len(batch) * replay_ratio)))
            replay_batch = rng.sample(replay, replay_count) if replay_count > 0 else []
            train_batch = list(batch) + replay_batch
            batch_losses: list[torch.Tensor] = []
            pending_rows: list[dict] = []
            sampler = ConditionalEdgeFlowSampler(
                template,
                model,
                method=method,
                flow_steps=flow_steps,
                time_sampling=train_cfg.get("teacher_time_sampling"),
            )
            for task in train_batch:
                x_dev = task.x.float().to(device)
                y_dev = task.y.float().to(device)
                samples = sampler.sample(
                    x_dev,
                    y_dev,
                    batch_size=samples_per_task,
                    rng=rng,
                    active_variable_count=task.num_vars,
                )
                proposal_samples = _proposal_samples(
                    task,
                    template=template,
                    model=model,
                    x=x_dev,
                    y=y_dev,
                    method=method,
                    flow_steps=flow_steps,
                    sources=proposal_sources,
                    count=proposal_count,
                    compile_only_trainable=proposal_compile_only_trainable,
                    rng=rng,
                    train_cfg=train_cfg,
                )
                gt_samples = _ground_truth_samples(
                    task,
                    template=template,
                    model=model,
                    x=x_dev,
                    y=y_dev,
                    method=method,
                    flow_steps=flow_steps,
                    device=device,
                    train_cfg=train_cfg,
                    rng=rng,
                ) if inject_gt_elite else []
                reward_samples = samples + proposal_samples + gt_samples
                rewards, reward_diag = _target_rewards(reward_samples, x_dev, y_dev, EdgeFlowBuildConfig(
                    samples_per_task=samples_per_task,
                    elite_k=elite_k,
                    complexity_weight=reward_cfg.complexity_weight,
                    validation_fraction=float(train_cfg.get("validation_fraction", 0.0)),
                    head_fit_mode=head_fit_mode,
                ))
                structure_scores = _structure_scores(
                    reward_samples,
                    ground_truth=str(task.ground_truth or ""),
                    num_vars=int(task.num_vars),
                )
                posterior_weights, posterior_diag = _posterior_weights(
                    rewards,
                    structure_scores,
                    train_cfg=train_cfg,
                )
                gt_diag = _gt_training_diagnostics(
                    reward_samples,
                    rewards.rewards,
                    posterior_weights,
                )
                if objective in {"semantic_teacher", "teacher_velocity", "semantic_velocity"}:
                    teacher_samples = (
                        gt_samples
                        if teacher_target_mode == "structural_denoising" and gt_samples
                        else reward_samples
                    )
                    teacher_weights = (
                        torch.ones(
                            len(teacher_samples),
                            dtype=posterior_weights.dtype if posterior_weights.numel() else torch.float32,
                            device=posterior_weights.device if posterior_weights.numel() else device,
                        )
                        if teacher_target_mode == "structural_denoising"
                        else posterior_weights
                    )
                    teacher_loss, teacher_metrics = semantic_teacher_loss_for_samples(
                        teacher_samples,
                        teacher_weights,
                        teacher_beta=float(train_cfg.get("teacher_beta", 1.0)),
                        teacher_smoothing=float(train_cfg.get("teacher_smoothing", 0.05)),
                        teacher_pinv_rtol=float(train_cfg.get("teacher_pinv_rtol", 1e-2)),
                        teacher_velocity_clip=_optional_float(train_cfg.get("teacher_velocity_clip", 5.0)),
                        teacher_path_geometry=str(train_cfg.get("teacher_path_geometry", "semantic")),
                        probability_path_geometry=probability_path_geometry,
                        semantic_calibration_gamma=semantic_calibration_gamma,
                        target_mode=teacher_target_mode,
                    )
                    active_weight = float(train_cfg.get("active_nll_weight", 0.0))
                    teacher_weight = float(train_cfg.get("teacher_loss_weight", 1.0))
                    if active_weight != 0.0:
                        active_loss, active_metrics = conditional_elite_policy_loss(
                            reward_samples,
                            rewards.rewards,
                            rewards.valid_mask,
                            elite_k=elite_k,
                            rank_temperature=rank_temperature,
                            entropy_bonus=entropy_bonus,
                            unique_elites=unique_elites,
                            gt_samples=gt_samples,
                        )
                        loss = teacher_weight * teacher_loss + active_weight * active_loss
                    else:
                        active_metrics = {}
                        loss = teacher_weight * teacher_loss
                    metrics = {
                        "loss": float(loss.detach().cpu().item()),
                        "loss_per_decision": float(loss.detach().cpu().item()),
                        "objective": objective,
                        "teacher_loss_weight": float(teacher_weight),
                        "teacher_pinv_rtol": float(train_cfg.get("teacher_pinv_rtol", 1e-2)),
                        "teacher_velocity_clip": float(_optional_float(train_cfg.get("teacher_velocity_clip", 5.0)) or 0.0),
                        "teacher_path_geometry": str(train_cfg.get("teacher_path_geometry", probability_path_geometry)),
                        "probability_path_geometry": probability_path_geometry,
                        "semantic_teacher_target_mode": teacher_target_mode,
                        "semantic_calibration_gamma": float(semantic_calibration_gamma),
                        **teacher_metrics,
                        **{f"active_aux_{key}": value for key, value in active_metrics.items()},
                    }
                else:
                    loss, metrics = conditional_elite_policy_loss(
                        reward_samples,
                        rewards.rewards,
                        rewards.valid_mask,
                        elite_k=elite_k,
                        rank_temperature=rank_temperature,
                        entropy_bonus=entropy_bonus,
                        unique_elites=unique_elites,
                        gt_samples=gt_samples,
                    )
                    metrics = {"objective": objective, **metrics}
                batch_losses.append(_loss_on_training_device(loss, device))
                valid_fraction = float(rewards.valid_mask.float().mean().item()) if rewards.valid_mask.numel() else 0.0
                sampled_rewards = rewards.rewards[:len(samples)] if samples else rewards.rewards
                unique_fraction = float(len({_sample_expression_key(sample) for sample in samples}) / max(len(samples), 1))
                decision_entropies = [
                    float(sample.entropy_tensor.detach().cpu().item())
                    for sample in samples
                    if sample.entropy_tensor is not None
                ]
                decision_counts = [
                    int((sample.diagnostics or {}).get("decision_count", len(sample.edge_choices)))
                    for sample in samples
                ]
                base_head_rates = [
                    float((sample.diagnostics or {}).get("base_head_selected_rate", 0.0))
                    for sample in samples
                ]
                row = {
                    "epoch": int(epoch),
                    "optimizer_step": int(optimizer_step),
                    "batch_index": int(batch_index),
                    "task_id": str(task.task_id),
                    "algorithm": "conditional_semantic_edge_flow",
                    "sampler_method": method,
                    "num_sampled_expressions": int(len(samples)),
                    "valid_expression_fraction": valid_fraction,
                    "unique_expression_fraction": unique_fraction,
                    "duplicate_expression_fraction": float(1.0 - unique_fraction),
                    "best_reward": float(sampled_rewards.max().item()) if sampled_rewards.numel() else 0.0,
                    "median_reward": float(sampled_rewards.median().item()) if sampled_rewards.numel() else 0.0,
                    "decision_entropy_mean": float(sum(decision_entropies) / max(len(decision_entropies), 1)),
                    "decision_count_mean": float(sum(decision_counts) / max(len(decision_counts), 1)),
                    "head_terms": int(model_cfg.head_terms),
                    "branches_per_register": int(model_cfg.branches_per_register),
                    "update_mode": str(model_cfg.update_mode),
                    "device": str(device),
                    "task_batch_size": int(len(train_batch)),
                    "replay_size": int(len(replay)),
                    "replay_count": int(len(replay_batch)),
                    "active_variable_count": int(task.num_vars),
                    "head_fit_mode": head_fit_mode,
                    "objective": objective,
                    "teacher_time_sampling": str(train_cfg.get("teacher_time_sampling", "")),
                    "teacher_path_geometry": str(train_cfg.get("teacher_path_geometry", "")),
                    "probability_path_geometry": probability_path_geometry,
                    "semantic_teacher_target_mode": teacher_target_mode,
                    "semantic_calibration_gamma": float(semantic_calibration_gamma),
                    "unique_elites": bool(unique_elites),
                    "entropy_weight": float(entropy_bonus),
                    "gt_injected": int(len(gt_samples)),
                    "proposal_sample_count": int(len(proposal_samples)),
                    "proposal_compile_success_rate": float(
                        sum(1 for sample in proposal_samples if (sample.diagnostics or {}).get("proposal_compile_success"))
                        / max(len(proposal_samples), 1)
                    ) if proposal_sources else 0.0,
                    "proposal_sources": ",".join(proposal_sources),
                    "gt_compile_success_rate": float(
                        sum(1 for sample in gt_samples if (sample.diagnostics or {}).get("gt_compile_success"))
                        / max(1, 1 if task.ground_truth else 0)
                    ) if inject_gt_elite else 0.0,
                    "gt_neighborhood_size": int(train_cfg.get("gt_neighborhood_size", train_cfg.get("gt_sampler_size", 0))),
                    "gt_neighborhood_compiled": int(sum(
                        1 for sample in gt_samples if (sample.diagnostics or {}).get("is_gt_neighborhood")
                    )),
                    "gt_neighborhood_compile_success_rate": float(max(
                        [
                            float((sample.diagnostics or {}).get("gt_neighborhood_compile_success_rate", 0.0))
                            for sample in gt_samples
                        ] or [0.0]
                    )),
                    "exclude_base_head_candidates": bool(model_cfg.exclude_base_head_candidates),
                    "base_head_selected_rate": float(sum(base_head_rates) / max(len(base_head_rates), 1)),
                    "structure_score_mean": float(structure_scores.mean().item()) if structure_scores.numel() else 0.0,
                    "sample_best_structure_score": float(structure_scores[:len(samples)].max().item()) if len(samples) and structure_scores.numel() else 0.0,
                    **posterior_diag,
                    **gt_diag,
                    **metrics,
                    **reward_diag,
                }
                pending_rows.append(row)
            if not batch_losses:
                continue
            loss = torch.stack(batch_losses).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            opt.step()
            for row in pending_rows:
                row["batch_loss"] = float(loss.detach().cpu().item())
                rows.append(row)
                print(
                    f"epoch {epoch} step {optimizer_step} task {row['task_id']} "
                    f"conditional_loss {float(row.get('loss', row['batch_loss'])):.6f}",
                    flush=True,
                )
            optimizer_step += 1
            if replay_capacity > 0:
                replay.extend(batch)
                if len(replay) > replay_capacity:
                    replay = replay[-replay_capacity:]
        if early_stopping_patience > 0 and validation_tasks:
            validation_score = _conditional_validation_score(
                model,
                template,
                validation_tasks,
                rng=rng,
                samples=max(1, int(train_cfg.get("early_stopping_samples", min(samples_per_task, 8)))),
                complexity_weight=reward_cfg.complexity_weight,
                method=method,
                flow_steps=flow_steps,
                head_fit_mode=head_fit_mode,
            )
            improved = best_validation_score is None or validation_score > best_validation_score + 1e-8
            if improved:
                best_validation_score = float(validation_score)
                best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if rows:
                rows[-1]["early_stopping_validation_score"] = float(validation_score)
                rows[-1]["early_stopping_best_score"] = float(best_validation_score or validation_score)
                rows[-1]["early_stopping_no_improve_epochs"] = int(epochs_without_improvement)
            if epochs_without_improvement >= early_stopping_patience:
                break
    if best_model_state is not None:
        model.load_state_dict(best_model_state, strict=False)
    out_dir = Path(cfg.get("out", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = str(cfg.get("checkpoint_name", "legacy_conditional_edge_flow.pt"))
    path = out_dir / ckpt_name
    model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({
        "model": model_state,
        "cfg": cfg,
        "model_cfg": asdict(model_cfg),
        "template": _template_payload(template),
        "algorithm": "conditional_semantic_edge_flow",
    }, path)
    _write_curve(out_dir / f"train_curve_{Path(ckpt_name).stem}.csv", rows)
    print(f"saved {path}")
    return path


def _run_spff_conditional(cfg: dict) -> Path:
    runtime = cfg.get("runtime", {})
    _configure_threads(runtime)
    device = _resolve_device(runtime)
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    template = _template_from_cfg(cfg)
    model_cfg = _spff_model_cfg(cfg, template)
    model = ConditionalEdgeFlowModel(model_cfg).to(device)
    train_cfg = cfg.get("train", {})
    tasks = _training_tasks(cfg, template, rng)
    rows: list[dict] = []
    chart_epochs = int(train_cfg.get("chart_pretrain_epochs", train_cfg.get("spff_chart_pretrain_epochs", 1)))
    velocity_epochs = int(train_cfg.get("velocity_epochs", train_cfg.get("spff_velocity_epochs", 1)))
    task_batch_size = max(int(train_cfg.get("task_batch_size", 1)), 1)
    lr = float(train_cfg.get("lr", 1e-3))
    chart_params = list(_spff_chart_parameters(model))
    velocity_params = list(_spff_velocity_parameters(model))
    if not velocity_params:
        raise RuntimeError("SPFF model did not create velocity parameters")
    if int(chart_epochs) > 0 and not chart_params:
        raise RuntimeError("SPFF chart pretrain requested but chart has no trainable parameters")
    chart_opt = torch.optim.AdamW(chart_params, lr=lr * float(train_cfg.get("chart_lr_scale", 1.0))) if chart_params else None
    velocity_opt = torch.optim.AdamW(velocity_params, lr=lr)
    optimizer_step = 0

    for stage, epochs, opt, train_chart, train_velocity in (
        ("chart_pretrain", chart_epochs, chart_opt, True, False),
        ("velocity", velocity_epochs, velocity_opt, False, True),
    ):
        if int(epochs) <= 0:
            continue
        if opt is None:
            continue
        _set_requires_grad(chart_params, bool(train_chart))
        _set_requires_grad(velocity_params, bool(train_velocity))
        for epoch in range(int(epochs)):
            epoch_tasks = list(tasks)
            rng.shuffle(epoch_tasks)
            for batch_index, batch in enumerate(_task_batches(epoch_tasks, task_batch_size)):
                batch_losses: list[torch.Tensor] = []
                pending_rows: list[dict] = []
                for task in batch:
                    x_dev = task.x.float().to(device)
                    y_dev = task.y.float().to(device)
                    records, target_diag = build_local_input_targets(
                        str(task.ground_truth or ""),
                        variable_count=int(task.num_vars),
                        template=template,
                        model=model,
                        x=x_dev,
                        y=y_dev,
                        smoothing=float(train_cfg.get("spff_p1_smoothing", 0.02)),
                        max_paths_per_term=int(train_cfg.get("spff_max_paths_per_term", 16)),
                        rng=rng,
                        task_id=str(task.task_id),
                        method=str(train_cfg.get("sampler_method", "policy")),
                        flow_steps=int(train_cfg.get("flow_steps", 1)),
                        flow_time=1.0,
                    )
                    loss, metrics = _spff_loss_for_records(
                        model,
                        records,
                        device=device,
                        train_cfg=train_cfg,
                        train_chart=bool(train_chart),
                        train_velocity=bool(train_velocity),
                        rng=rng,
                    )
                    batch_losses.append(_loss_on_training_device(loss, device))
                    pending_rows.append({
                        "epoch": int(epoch),
                        "optimizer_step": int(optimizer_step),
                        "batch_index": int(batch_index),
                        "stage": str(stage),
                        "task_id": str(task.task_id),
                        "algorithm": "semantic_pullback_fisher_flow",
                        "device": str(device),
                        "active_variable_count": int(task.num_vars),
                        "task_batch_size": int(len(batch)),
                        "head_terms": int(model_cfg.head_terms),
                        "branches_per_register": int(model_cfg.branches_per_register),
                        "update_mode": str(model_cfg.update_mode),
                        "spff_chart_type": str(model_cfg.spff_chart_type),
                        "spff_geometry": str(model_cfg.spff_geometry),
                        "spff_num_candidates": int(model_cfg.spff_num_candidates),
                        **target_diag,
                        **metrics,
                    })
                if not batch_losses:
                    continue
                batch_loss = torch.stack(batch_losses).mean()
                opt.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in opt.param_groups[0]["params"] if p.requires_grad],
                    float(train_cfg.get("grad_clip", 1.0)),
                )
                opt.step()
                for row in pending_rows:
                    row["batch_loss"] = float(batch_loss.detach().cpu().item())
                    rows.append(row)
                    print(
                        f"stage {stage} epoch {epoch} step {optimizer_step} task {row['task_id']} "
                        f"spff_loss {row['batch_loss']:.6f}",
                        flush=True,
                    )
                optimizer_step += 1
    _set_requires_grad(chart_params + velocity_params, True)
    out_dir = Path(cfg.get("out", "checkpoints/spff"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = str(cfg.get("checkpoint_name", "spff.pt"))
    path = out_dir / ckpt_name
    torch.save({
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "cfg": cfg,
        "model_cfg": asdict(model_cfg),
        "template": _template_payload(template),
        "algorithm": "semantic_pullback_fisher_flow",
    }, path)
    _write_curve(out_dir / f"train_curve_{Path(ckpt_name).stem}.csv", rows)
    print(f"saved {path}")
    return path


def _run_semantic_global_projection(cfg: dict) -> Path:
    runtime = cfg.get("runtime", {})
    _configure_threads(runtime)
    device = _resolve_device(runtime)
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    template = _template_from_cfg(cfg)
    model_cfg = _spff_model_cfg(cfg, template)
    model_cfg = replace(
        model_cfg,
        spff_enabled=False,
        term_factorized=True,
        term_num_heads=max(int(model_cfg.term_num_heads), max(int(model_cfg.head_terms), 1)),
    )
    model = ConditionalEdgeFlowModel(model_cfg).to(device)
    train_cfg = cfg.get("train", {})
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)))
    tasks = _training_tasks(cfg, template, rng)
    rows: list[dict] = []
    epochs = int(train_cfg.get("epochs", 2))
    task_batch_size = max(int(train_cfg.get("task_batch_size", 1)), 1)
    samples_per_task = int(train_cfg.get("samples_per_task", train_cfg.get("sample_pool_size", 32)))
    method = str(train_cfg.get("sampler_method", train_cfg.get("method", "policy")))
    flow_steps = int(train_cfg.get("flow_steps", 1))
    semantic_weight = float(train_cfg.get("semantic_weight", train_cfg.get("mmd_weight", 0.0)))
    semantic_kernel = str(train_cfg.get("semantic_kernel", "inner"))
    semantic_sigma = train_cfg.get("semantic_sigma")
    semantic_sigma = None if semantic_sigma is None else float(semantic_sigma)
    sign_invariant_kernel = bool(train_cfg.get("sign_invariant_kernel", True))
    semantic_aug_points = int(train_cfg.get("semantic_aug_points", 0))
    gt_injection_count = int(train_cfg.get("gt_injection_count", 1))
    optimizer_step = 0
    for epoch in range(int(epochs)):
        epoch_tasks = list(tasks)
        rng.shuffle(epoch_tasks)
        for batch_index, batch in enumerate(_task_batches(epoch_tasks, task_batch_size)):
            batch_losses: list[torch.Tensor] = []
            pending_rows: list[dict] = []
            sampler = ConditionalEdgeFlowSampler(
                template,
                model,
                method=method,
                flow_steps=flow_steps,
                time_sampling=train_cfg.get("teacher_time_sampling"),
            )
            for task in batch:
                x_dev = task.x.float().to(device)
                y_dev = task.y.float().to(device)
                samples = sampler.sample(
                    x_dev,
                    y_dev,
                    batch_size=samples_per_task,
                    rng=rng,
                    active_variable_count=int(task.num_vars),
                )
                gt_samples = _global_projection_gt_samples(
                    task,
                    template=template,
                    model=model,
                    x=x_dev,
                    y=y_dev,
                    method=method,
                    flow_steps=flow_steps,
                    train_cfg=train_cfg,
                    rng=rng,
                ) if gt_injection_count > 0 else []
                pool = list(samples) + list(gt_samples)
                try:
                    gt_expr = parse_formula(str(task.ground_truth or ""), [f"x{i}" for i in range(int(task.num_vars))])
                    target_weights, target_diag = build_gt_equivalence_target(
                        pool,
                        gt_expr,
                        num_vars=int(task.num_vars),
                        device=device,
                    )
                except Exception:
                    target_weights = torch.zeros(len(pool), dtype=torch.float32, device=device)
                    target_diag = {"gt_equiv_count": 0, "gt_equiv_fraction": 0.0}
                x_grid = augment_grid(x_dev, count=semantic_aug_points, rng=rng)
                loss, metrics = global_projection_loss(
                    pool,
                    target_weights,
                    x_grid,
                    semantic_weight=semantic_weight,
                    kernel_kind=semantic_kernel,
                    kernel_sigma=semantic_sigma,
                    sign_invariant_kernel=sign_invariant_kernel,
                    use_active_logprob=bool(train_cfg.get("use_active_logprob", True)),
                )
                if float(metrics.get("global_projection_target_mass_trainable", 0.0)) > 0.0:
                    batch_losses.append(_loss_on_training_device(loss, device))
                pending_rows.append({
                    "epoch": int(epoch),
                    "optimizer_step": int(optimizer_step),
                    "batch_index": int(batch_index),
                    "task_id": str(task.task_id),
                    "algorithm": "semantic_global_projection",
                    "device": str(device),
                    "active_variable_count": int(task.num_vars),
                    "task_batch_size": int(len(batch)),
                    "head_terms": int(model_cfg.head_terms),
                    "term_num_heads": int(model_cfg.term_num_heads),
                    "term_prior_type": str(model_cfg.term_prior_type),
                    "sampler_method": str(method),
                    "flow_steps": int(flow_steps),
                    "semantic_weight": float(semantic_weight),
                    "semantic_kernel": str(semantic_kernel),
                    "semantic_aug_points": int(semantic_aug_points),
                    "gt_injected_count": int(len(gt_samples)),
                    "num_sampled_expressions": int(len(samples)),
                    "loss": float(loss.detach().cpu().item()),
                    **{f"global_projection_{key}": value for key, value in target_diag.items()},
                    **metrics,
                })
            if not batch_losses:
                rows.extend(pending_rows)
                continue
            batch_loss = torch.stack(batch_losses).mean()
            opt.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            opt.step()
            for row in pending_rows:
                row["batch_loss"] = float(batch_loss.detach().cpu().item())
                rows.append(row)
                print(
                    f"epoch {epoch} step {optimizer_step} task {row['task_id']} "
                    f"global_projection_loss {row.get('global_projection_loss', 0.0):.6f}",
                    flush=True,
                )
            optimizer_step += 1
    out_dir = Path(cfg.get("out", "checkpoints/global_projection"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = str(cfg.get("checkpoint_name", "semantic_global_projection.pt"))
    path = out_dir / ckpt_name
    torch.save({
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "cfg": cfg,
        "model_cfg": asdict(model_cfg),
        "template": _template_payload(template),
        "algorithm": "semantic_global_projection",
    }, path)
    _write_curve(out_dir / f"train_curve_{Path(ckpt_name).stem}.csv", rows)
    print(f"saved {path}")
    return path


def _run_semantic_flow_matching(cfg: dict) -> Path:
    runtime = cfg.get("runtime", {})
    _configure_threads(runtime)
    device = _resolve_device(runtime)
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    template = _template_from_cfg(cfg)
    model_cfg = _spff_model_cfg(cfg, template)
    model_cfg = replace(
        model_cfg,
        spff_enabled=False,
        term_factorized=True,
        term_num_heads=max(int(model_cfg.term_num_heads), max(int(model_cfg.head_terms), 1)),
    )
    model = ConditionalEdgeFlowModel(model_cfg).to(device)
    train_cfg = cfg.get("train", {})
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)))
    tasks = _training_tasks(cfg, template, rng)
    rows: list[dict] = []
    epochs = int(train_cfg.get("epochs", 2))
    task_batch_size = max(int(train_cfg.get("task_batch_size", 1)), 1)
    samples_per_task = int(train_cfg.get("samples_per_task", train_cfg.get("sample_pool_size", 32)))
    method = str(train_cfg.get("sampler_method", train_cfg.get("method", "policy")))
    flow_steps = int(train_cfg.get("flow_steps", 1))
    semantic_kernel = str(train_cfg.get("semantic_kernel", "inner"))
    semantic_sigma = train_cfg.get("semantic_sigma")
    semantic_sigma = None if semantic_sigma is None else float(semantic_sigma)
    sign_invariant_kernel = bool(train_cfg.get("sign_invariant_kernel", True))
    semantic_aug_points = int(train_cfg.get("semantic_aug_points", 0))
    gt_injection_count = int(train_cfg.get("gt_injection_count", 1))
    flow_time_eps = float(train_cfg.get("flow_time_eps", 0.03))
    optimizer_step = 0
    for epoch in range(int(epochs)):
        epoch_tasks = list(tasks)
        rng.shuffle(epoch_tasks)
        for batch_index, batch in enumerate(_task_batches(epoch_tasks, task_batch_size)):
            batch_losses: list[torch.Tensor] = []
            pending_rows: list[dict] = []
            for task in batch:
                t_value = _sample_semantic_fm_time(train_cfg, rng)
                task_train_cfg = dict(train_cfg)
                task_train_cfg["gt_flow_time"] = float(t_value)
                x_dev = task.x.float().to(device)
                y_dev = task.y.float().to(device)
                sampler = ConditionalEdgeFlowSampler(
                    template,
                    model,
                    method=method,
                    flow_steps=flow_steps,
                    time_sampling=float(t_value),
                )
                samples = sampler.sample(
                    x_dev,
                    y_dev,
                    batch_size=samples_per_task,
                    rng=rng,
                    active_variable_count=int(task.num_vars),
                )
                gt_samples = _global_projection_gt_samples(
                    task,
                    template=template,
                    model=model,
                    x=x_dev,
                    y=y_dev,
                    method=method,
                    flow_steps=flow_steps,
                    train_cfg=task_train_cfg,
                    rng=rng,
                ) if gt_injection_count > 0 else []
                pool = list(samples) + list(gt_samples)
                try:
                    gt_expr = parse_formula(str(task.ground_truth or ""), [f"x{i}" for i in range(int(task.num_vars))])
                    target_weights, target_diag = build_gt_equivalence_target(
                        pool,
                        gt_expr,
                        num_vars=int(task.num_vars),
                        device=device,
                    )
                except Exception:
                    target_weights = torch.zeros(len(pool), dtype=torch.float32, device=device)
                    target_diag = {"gt_equiv_count": 0, "gt_equiv_fraction": 0.0}
                x_grid = augment_grid(x_dev, count=semantic_aug_points, rng=rng)
                log_rates: list[torch.Tensor | None] = []
                current_log_probs: list[torch.Tensor] = []
                for sample in pool:
                    log_rate = _semantic_fm_path_log_rate(
                        sample,
                        task,
                        template=template,
                        model=model,
                        x=x_dev,
                        y=y_dev,
                        method=method,
                        flow_steps=flow_steps,
                        t_value=float(t_value),
                        eps=float(flow_time_eps),
                    )
                    log_rates.append(log_rate)
                    current_log_probs.append(_sample_current_logprob(sample, device=device, dtype=torch.float32))
                current_log_prob_tensor = torch.stack(current_log_probs) if current_log_probs else torch.zeros(0, device=device)
                loss, metrics = semantic_flow_matching_loss(
                    pool,
                    target_weights,
                    x_grid,
                    log_rates=log_rates,
                    current_log_probs=current_log_prob_tensor,
                    interpolation_t=float(t_value),
                    kernel_kind=semantic_kernel,
                    kernel_sigma=semantic_sigma,
                    sign_invariant_kernel=sign_invariant_kernel,
                )
                if (
                    float(metrics.get("semantic_fm_target_mass_trainable", 0.0)) > 0.0
                    and int(metrics.get("semantic_fm_trainable_pool_size", 0)) > 0
                ):
                    batch_losses.append(_loss_on_training_device(loss, device))
                pending_rows.append({
                    "epoch": int(epoch),
                    "optimizer_step": int(optimizer_step),
                    "batch_index": int(batch_index),
                    "task_id": str(task.task_id),
                    "algorithm": "semantic_flow_matching",
                    "device": str(device),
                    "active_variable_count": int(task.num_vars),
                    "task_batch_size": int(len(batch)),
                    "head_terms": int(model_cfg.head_terms),
                    "term_num_heads": int(model_cfg.term_num_heads),
                    "term_prior_type": str(model_cfg.term_prior_type),
                    "sampler_method": str(method),
                    "flow_steps": int(flow_steps),
                    "flow_time": float(t_value),
                    "flow_time_eps": float(flow_time_eps),
                    "semantic_kernel": str(semantic_kernel),
                    "semantic_aug_points": int(semantic_aug_points),
                    "gt_injected_count": int(len(gt_samples)),
                    "num_sampled_expressions": int(len(samples)),
                    "loss": float(loss.detach().cpu().item()),
                    **{f"semantic_fm_{key}": value for key, value in target_diag.items()},
                    **metrics,
                })
            if not batch_losses:
                rows.extend(pending_rows)
                continue
            batch_loss = torch.stack(batch_losses).mean()
            opt.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            opt.step()
            for row in pending_rows:
                row["batch_loss"] = float(batch_loss.detach().cpu().item())
                rows.append(row)
                print(
                    f"epoch {epoch} step {optimizer_step} task {row['task_id']} "
                    f"semantic_fm_loss {row.get('semantic_fm_loss', 0.0):.6f}",
                    flush=True,
                )
            optimizer_step += 1
    out_dir = Path(cfg.get("out", "checkpoints/semantic_fm"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = str(cfg.get("checkpoint_name", "semantic_flow_matching.pt"))
    path = out_dir / ckpt_name
    torch.save({
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "cfg": cfg,
        "model_cfg": asdict(model_cfg),
        "template": _template_payload(template),
        "algorithm": "semantic_flow_matching",
    }, path)
    _write_curve(out_dir / f"train_curve_{Path(ckpt_name).stem}.csv", rows)
    print(f"saved {path}")
    return path


def _run_explicit_global_theta_projection(cfg: dict) -> Path:
    runtime = cfg.get("runtime", {})
    _configure_threads(runtime)
    device = _resolve_device(runtime)
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    template = _template_from_cfg(cfg)
    model_cfg = cfg.get("model", {})
    model = GlobalThetaNetwork(
        template,
        num_vars=int(template.num_vars),
        hidden=int(model_cfg.get("hidden", 64)),
    ).to(device)
    train_cfg = cfg.get("train", {})
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)))
    tasks = _training_tasks(cfg, template, rng)
    rows: list[dict] = []
    epochs = int(train_cfg.get("epochs", 2))
    task_batch_size = max(int(train_cfg.get("task_batch_size", 1)), 1)
    samples_per_task = int(train_cfg.get("samples_per_task", train_cfg.get("sample_pool_size", 32)))
    semantic_weight = float(train_cfg.get("semantic_weight", train_cfg.get("mmd_weight", 0.0)))
    semantic_kernel = str(train_cfg.get("semantic_kernel", "inner"))
    semantic_sigma = train_cfg.get("semantic_sigma")
    semantic_sigma = None if semantic_sigma is None else float(semantic_sigma)
    sign_invariant_kernel = bool(train_cfg.get("sign_invariant_kernel", True))
    semantic_aug_points = int(train_cfg.get("semantic_aug_points", 0))
    gt_injection_count = int(train_cfg.get("gt_injection_count", 1))
    theta_time = float(train_cfg.get("theta_time", 1.0))
    objective = str(train_cfg.get("objective", "global_projection")).strip().lower()
    sampler = CircuitSampler(template)
    optimizer_step = 0
    for epoch in range(int(epochs)):
        epoch_tasks = list(tasks)
        rng.shuffle(epoch_tasks)
        for batch_index, batch in enumerate(_task_batches(epoch_tasks, task_batch_size)):
            batch_losses: list[torch.Tensor] = []
            pending_rows: list[dict] = []
            for task in batch:
                x_dev = task.x.float().to(device)
                y_dev = task.y.float().to(device)
                theta_out = model(x_dev, y_dev, t=float(theta_time))
                samples = sampler.sample(theta_out.edge_distribution, batch_size=samples_per_task, rng=rng)
                gt_samples: list[CircuitSample] = []
                try:
                    gt_expr = parse_formula(str(task.ground_truth or ""), [f"x{i}" for i in range(int(task.num_vars))])
                except Exception:
                    gt_expr = None
                if gt_expr is not None and gt_injection_count > 0:
                    gt_samples = _explicit_theta_gt_samples(
                        task,
                        gt_expr,
                        template=template,
                        edge_distribution=theta_out.edge_distribution,
                        train_cfg=train_cfg,
                        start_id=len(samples),
                    )
                proxy_samples = _explicit_theta_proxy_samples(
                    task,
                    template=template,
                    edge_distribution=theta_out.edge_distribution,
                    x=x_dev,
                    y=y_dev,
                    train_cfg=train_cfg,
                    rng=rng,
                    start_id=len(samples) + len(gt_samples),
                ) if objective in {"gt_proxy_projection", "gt_mass_proxy", "contrastive_gt_proxy"} else []
                pool = list(samples) + list(gt_samples) + list(proxy_samples)
                if gt_expr is None:
                    target_weights = torch.zeros(len(pool), dtype=torch.float32, device=device)
                    target_diag = {"gt_equiv_count": 0, "gt_equiv_fraction": 0.0}
                    proxy_mask = torch.zeros(len(pool), dtype=torch.bool, device=device)
                    proxy_diag = {
                        "hard_proxy_negative_count": 0,
                        "semantic_near_but_symbolic_wrong_count": 0,
                    }
                else:
                    if objective in {"gt_proxy_projection", "gt_mass_proxy", "contrastive_gt_proxy"}:
                        target_weights, proxy_mask, proxy_diag = classify_gt_proxy_samples(
                            pool,
                            gt_expr,
                            x_dev,
                            y_dev,
                            num_vars=int(task.num_vars),
                            proxy_r2_threshold=float(train_cfg.get("proxy_r2_threshold", 0.85)),
                            device=device,
                        )
                        target_diag = {
                            "gt_equiv_count": proxy_diag.get("gt_equiv_count", 0),
                            "gt_equiv_fraction": proxy_diag.get("gt_equiv_fraction", 0.0),
                        }
                    else:
                        target_weights, target_diag = build_gt_equivalence_target(
                            pool,
                            gt_expr,
                            num_vars=int(task.num_vars),
                            device=device,
                        )
                        proxy_mask = torch.zeros(len(pool), dtype=torch.bool, device=device)
                        proxy_diag = {
                            "hard_proxy_negative_count": 0,
                            "semantic_near_but_symbolic_wrong_count": 0,
                        }
                x_grid = augment_grid(x_dev, count=semantic_aug_points, rng=rng)
                if objective in {"gt_proxy_projection", "gt_mass_proxy", "contrastive_gt_proxy"}:
                    loss, metrics = gt_proxy_projection_loss(
                        pool,
                        target_weights,
                        proxy_mask,
                        rank_margin=float(train_cfg.get("rank_margin", 0.5)),
                        gt_weight=float(train_cfg.get("gt_mass_weight", 1.0)),
                        rank_weight=float(train_cfg.get("rank_weight", 1.0)),
                        group_weight=float(train_cfg.get("group_weight", 0.0)),
                        pair_weight=float(train_cfg.get("pair_weight", 0.0)),
                        score_temperature=float(train_cfg.get("score_temperature", 1.0)),
                        score_length_alpha=float(train_cfg.get("score_length_alpha", 0.0)),
                        entropy_weight=float(train_cfg.get("entropy_weight", 0.0)),
                        use_active_logprob=True,
                    )
                else:
                    loss, metrics = global_projection_loss(
                        pool,
                        target_weights,
                        x_grid,
                        semantic_weight=semantic_weight,
                        kernel_kind=semantic_kernel,
                        kernel_sigma=semantic_sigma,
                        sign_invariant_kernel=sign_invariant_kernel,
                        use_active_logprob=True,
                    )
                trainable_gt_indices = _trainable_gt_indices(pool, target_weights)
                gt_grad_norm = _sample_logprob_grad_norm(
                    pool,
                    trainable_gt_indices,
                    tuple(model.parameters()),
                )
                gt_rewrite_injected_count = sum(
                    1 for sample in gt_samples
                    if bool((sample.diagnostics or {}).get("explicit_theta_gt_rewrite"))
                )
                trainable_mass = float(metrics.get(
                    "global_projection_target_mass_trainable",
                    metrics.get("gt_proxy_target_mass_trainable", 0.0),
                ))
                if trainable_mass > 0.0:
                    batch_losses.append(_loss_on_training_device(loss, device))
                pending_rows.append({
                    "epoch": int(epoch),
                    "optimizer_step": int(optimizer_step),
                    "batch_index": int(batch_index),
                    "task_id": str(task.task_id),
                    "algorithm": "explicit_global_theta_projection",
                    "device": str(device),
                    "active_variable_count": int(task.num_vars),
                    "task_batch_size": int(len(batch)),
                    "theta_dim": int(model.theta_dim),
                    "theta_time": float(theta_time),
                    "semantic_weight": float(semantic_weight),
                    "semantic_kernel": str(semantic_kernel),
                    "semantic_aug_points": int(semantic_aug_points),
                    "gt_injected_count": int(len(gt_samples)),
                    "gt_rewrite_injected_count": int(gt_rewrite_injected_count),
                    "gt_path_grad_norm": float(gt_grad_norm),
                    "proxy_sample_count": int(len(proxy_samples)),
                    "num_sampled_expressions": int(len(samples)),
                    "objective": str(objective),
                    "loss": float(loss.detach().cpu().item()),
                    **{f"global_projection_{key}": value for key, value in target_diag.items()},
                    **proxy_diag,
                    **metrics,
                })
            if not batch_losses:
                rows.extend(pending_rows)
                continue
            batch_loss = torch.stack(batch_losses).mean()
            opt.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            opt.step()
            for row in pending_rows:
                row["batch_loss"] = float(batch_loss.detach().cpu().item())
                rows.append(row)
                printed_loss = float(row.get("gt_proxy_loss", row.get("global_projection_loss", 0.0)))
                print(
                    f"epoch {epoch} step {optimizer_step} task {row['task_id']} "
                    f"explicit_theta_loss {printed_loss:.6f}",
                    flush=True,
                )
            optimizer_step += 1
    out_dir = Path(cfg.get("out", "checkpoints/global_theta"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = str(cfg.get("checkpoint_name", "explicit_global_theta_projection.pt"))
    path = out_dir / ckpt_name
    torch.save({
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "cfg": cfg,
        "template": _template_payload(template),
        "algorithm": "explicit_global_theta_projection",
        "theta_dim": int(model.theta_dim),
    }, path)
    _write_curve(out_dir / f"train_curve_{Path(ckpt_name).stem}.csv", rows)
    print(f"saved {path}")
    return path


def _run_direct_theta_gt_proxy_overfit(cfg: dict) -> Path:
    runtime = cfg.get("runtime", {})
    _configure_threads(runtime)
    device = _resolve_device(runtime)
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    template = _template_from_cfg(cfg)
    train_cfg = cfg.get("train", {})
    tasks = _training_tasks(cfg, template, rng)
    rows: list[dict] = []
    epochs = int(train_cfg.get("epochs", 20))
    samples_per_task = int(train_cfg.get("samples_per_task", train_cfg.get("sample_pool_size", 32)))
    task_limit = max(int(train_cfg.get("task_limit", cfg.get("num_tasks", len(tasks)))), 1)
    theta_params: list[torch.nn.Parameter] = []
    final_thetas: dict[str, torch.Tensor] = {}
    sampler = CircuitSampler(template)
    for task_index, task in enumerate(tasks[:task_limit]):
        x_dev = task.x.float().to(device)
        y_dev = task.y.float().to(device)
        try:
            gt_expr = parse_formula(str(task.ground_truth or ""), [f"x{i}" for i in range(int(task.num_vars))])
        except Exception:
            continue
        theta = torch.nn.Parameter(torch.zeros(_template_theta_dim(template), device=device))
        opt = torch.optim.AdamW([theta], lr=float(train_cfg.get("lr", 0.05)))
        with torch.no_grad():
            initial_dist = theta_vector_to_distribution(theta.detach(), template)
            model_samples = sampler.sample(initial_dist, batch_size=samples_per_task, rng=rng)
        gt_samples = _explicit_theta_gt_samples(
            task,
            gt_expr,
            template=template,
            edge_distribution=theta_vector_to_distribution(theta, template),
            train_cfg=train_cfg,
            start_id=len(model_samples),
        )
        proxy_samples = _explicit_theta_proxy_samples(
            task,
            template=template,
            edge_distribution=theta_vector_to_distribution(theta, template),
            x=x_dev,
            y=y_dev,
            train_cfg=train_cfg,
            rng=rng,
            start_id=len(model_samples) + len(gt_samples),
        )
        base_pool = list(model_samples) + list(gt_samples) + list(proxy_samples)
        target_weights, proxy_mask, proxy_diag = classify_gt_proxy_samples(
            base_pool,
            gt_expr,
            x_dev,
            y_dev,
            num_vars=int(task.num_vars),
            proxy_r2_threshold=float(train_cfg.get("proxy_r2_threshold", 0.85)),
            device=device,
        )
        for epoch in range(epochs):
            dist = theta_vector_to_distribution(theta, template)
            pool = _relogprob_edge_pool(base_pool, template=template, edge_distribution=dist)
            loss, metrics = gt_proxy_projection_loss(
                pool,
                target_weights,
                proxy_mask,
                rank_margin=float(train_cfg.get("rank_margin", 0.5)),
                gt_weight=float(train_cfg.get("gt_mass_weight", 0.1)),
                rank_weight=float(train_cfg.get("rank_weight", 0.0)),
                group_weight=float(train_cfg.get("group_weight", 1.0)),
                pair_weight=float(train_cfg.get("pair_weight", 1.0)),
                score_temperature=float(train_cfg.get("score_temperature", 1.0)),
                score_length_alpha=float(train_cfg.get("score_length_alpha", 0.5)),
                entropy_weight=float(train_cfg.get("entropy_weight", 0.0)),
                use_active_logprob=True,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], float(train_cfg.get("grad_clip", 5.0)))
            opt.step()
            rows.append({
                "epoch": int(epoch),
                "task_index": int(task_index),
                "task_id": str(task.task_id),
                "algorithm": "direct_theta_gt_proxy_overfit",
                "device": str(device),
                "theta_dim": int(theta.numel()),
                "gt_injected_count": int(len(gt_samples)),
                "gt_rewrite_injected_count": int(sum(
                    1 for sample in gt_samples
                    if bool((sample.diagnostics or {}).get("explicit_theta_gt_rewrite"))
                )),
                "proxy_sample_count": int(len(proxy_samples)),
                "num_sampled_expressions": int(len(model_samples)),
                "direct_theta_overfit_gt_mass": float(metrics.get("gt_proxy_gt_mass", 0.0)),
                "direct_theta_overfit_gap": float(metrics.get("gt_minus_proxy_logmean_gap", 0.0)),
                "loss": float(loss.detach().cpu().item()),
                **proxy_diag,
                **metrics,
            })
        theta_params.append(theta)
        final_thetas[str(task.task_id)] = theta.detach().cpu()
    out_dir = Path(cfg.get("out", "checkpoints/direct_theta_overfit"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = str(cfg.get("checkpoint_name", "direct_theta_gt_proxy_overfit.pt"))
    path = out_dir / ckpt_name
    torch.save({
        "theta": final_thetas,
        "cfg": cfg,
        "template": _template_payload(template),
        "algorithm": "direct_theta_gt_proxy_overfit",
    }, path)
    _write_curve(out_dir / f"train_curve_{Path(ckpt_name).stem}.csv", rows)
    print(f"saved {path}")
    return path


def _run_amortized_theta_fixed_pool_overfit(cfg: dict, *, oracle_mode: bool = False) -> Path:
    runtime = cfg.get("runtime", {})
    _configure_threads(runtime)
    device = _resolve_device(runtime)
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    template = _template_from_cfg(cfg)
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    tasks = _training_tasks(cfg, template, rng)
    task_limit = max(int(train_cfg.get("task_limit", cfg.get("num_tasks", len(tasks)))), 1)
    tasks = list(tasks[:task_limit])
    fixed_pool_device = torch.device(str(train_cfg.get("fixed_pool_device", str(device))))
    theta_dim = _template_theta_dim(template)
    if bool(oracle_mode):
        model: torch.nn.Module = torch.nn.Embedding(len(tasks), int(theta_dim)).to(device)
        torch.nn.init.zeros_(model.weight)
        algorithm_name = "task_id_theta_oracle_overfit"
        mode_name = "task_id_oracle"
        default_out = "checkpoints/task_id_theta_oracle"
        default_ckpt = "task_id_theta_oracle_overfit.pt"
    else:
        model = GlobalThetaNetwork(
            template,
            num_vars=int(template.num_vars),
            hidden=int(model_cfg.get("hidden", 64)),
        ).to(device)
        algorithm_name = "amortized_theta_fixed_pool_overfit"
        mode_name = "task_encoder"
        default_out = "checkpoints/amortized_theta_fixed_pool"
        default_ckpt = "amortized_theta_fixed_pool_overfit.pt"
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)))
    out_dir = Path(cfg.get("out", default_out))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = str(cfg.get("checkpoint_name", default_ckpt))
    path = out_dir / ckpt_name
    curve_path = out_dir / f"train_curve_{Path(ckpt_name).stem}.csv"
    state_key = "task_id_theta" if bool(oracle_mode) else "model"
    rows: list[dict] = []
    optimizer_step = 0
    start_epoch = 0
    resume_text = str(train_cfg.get("resume_checkpoint", cfg.get("resume_checkpoint", "")) or "").strip()
    if resume_text:
        resume_path = Path(resume_text)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_checkpoint does not exist: {resume_path}")
        resume_payload = torch.load(resume_path, map_location=device, weights_only=False)
        if state_key not in resume_payload:
            raise ValueError(f"resume_checkpoint is missing {state_key!r}: {resume_path}")
        model.load_state_dict(resume_payload[state_key])
        if isinstance(resume_payload.get("optimizer"), dict):
            opt.load_state_dict(resume_payload["optimizer"])
            _move_optimizer_state(opt, device)
        start_epoch = int(resume_payload.get("completed_epochs", 0))
        optimizer_step = int(resume_payload.get("optimizer_step", 0))
        if curve_path.exists():
            rows = _read_curve_rows(curve_path)

    def build_pool_item(task_index: int, task: ConditionalTrainTask, *, build_rng: random.Random) -> dict | None:
        if bool(train_cfg.get("log_fixed_pool_build", False)):
            print(
                f"building fixed pool {task_index + 1}/{len(tasks)} "
                f"task {task.task_id}",
                flush=True,
            )
        built = _build_fixed_gt_proxy_pool(
            task,
            template=template,
            train_cfg=train_cfg,
            rng=build_rng,
            device=fixed_pool_device,
        )
        if built is None:
            return None
        built["task_index"] = int(task_index)
        built["fixed_pool_device"] = str(fixed_pool_device)
        return built

    stream_fixed_pools = bool(train_cfg.get("stream_fixed_pools", False))
    fixed_pools: list[dict] = []
    if not stream_fixed_pools:
        for task_index, task in enumerate(tasks):
            built = build_pool_item(task_index, task, build_rng=rng)
            if built is None:
                continue
            fixed_pools.append(built)
    epochs = int(train_cfg.get("epochs", 10))
    task_batch_size = max(int(train_cfg.get("task_batch_size", 1)), 1)

    def save_checkpoint(completed_epochs: int) -> None:
        torch.save({
            state_key: {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": _tensor_tree_to_cpu(opt.state_dict()),
            "cfg": cfg,
            "template": _template_payload(template),
            "algorithm": str(algorithm_name),
            "amortization_mode": str(mode_name),
            "theta_dim": int(theta_dim),
            "fixed_pool_count": int(len(tasks) if stream_fixed_pools else len(fixed_pools)),
            "fixed_pool_device": str(fixed_pool_device),
            "stream_fixed_pools": bool(stream_fixed_pools),
            "completed_epochs": int(completed_epochs),
            "optimizer_step": int(optimizer_step),
            "curve_path": str(curve_path),
        }, path)

    for epoch in range(start_epoch, epochs):
        if stream_fixed_pools:
            epoch_indices = list(range(len(tasks)))
            rng.shuffle(epoch_indices)
            batch_iter = (
                [
                    item
                    for item in (
                        build_pool_item(
                            int(task_index),
                            tasks[int(task_index)],
                            build_rng=random.Random(_fixed_pool_task_seed(seed, int(task_index))),
                        )
                        for task_index in batch_indices
                    )
                    if item is not None
                ]
                for batch_indices in _task_batches(epoch_indices, task_batch_size)
            )
        else:
            epoch_pools = list(fixed_pools)
            rng.shuffle(epoch_pools)
            batch_iter = _task_batches(epoch_pools, task_batch_size)
        for batch_index, batch in enumerate(batch_iter):
            if not batch:
                continue
            batch_losses: list[torch.Tensor] = []
            pending_rows: list[dict] = []
            for item in batch:
                task = item["task"]
                x_dev = item["x"].to(device)
                y_dev = item["y"].to(device)
                if bool(oracle_mode):
                    task_index_tensor = torch.tensor(int(item["task_index"]), dtype=torch.long, device=device)
                    theta = model(task_index_tensor).flatten()
                    dist = theta_vector_to_distribution(theta, template)
                    theta_dim_value = int(theta.numel())
                else:
                    theta_out = model(x_dev, y_dev, t=float(train_cfg.get("theta_time", 1.0)))
                    theta = theta_out.theta_vector
                    dist = theta_out.edge_distribution
                    theta_dim_value = int(theta.numel())
                pool = _relogprob_edge_pool(item["base_pool"], template=template, edge_distribution=dist)
                loss, metrics = gt_proxy_projection_loss(
                    pool,
                    item["target_weights"],
                    item["proxy_mask"],
                    rank_margin=float(train_cfg.get("rank_margin", 0.5)),
                    gt_weight=float(train_cfg.get("gt_mass_weight", 0.1)),
                    rank_weight=float(train_cfg.get("rank_weight", 0.0)),
                    group_weight=float(train_cfg.get("group_weight", 1.0)),
                    pair_weight=float(train_cfg.get("pair_weight", 1.0)),
                    score_temperature=float(train_cfg.get("score_temperature", 1.0)),
                    score_length_alpha=float(train_cfg.get("score_length_alpha", 0.5)),
                    entropy_weight=float(train_cfg.get("entropy_weight", 0.0)),
                    use_active_logprob=True,
                )
                trainable_mass = float(metrics.get("gt_proxy_target_mass_trainable", 0.0))
                if trainable_mass > 0.0:
                    batch_losses.append(_loss_on_training_device(loss, device))
                gt_grad_norm = _sample_logprob_grad_norm(
                    pool,
                    _trainable_gt_indices(pool, item["target_weights"]),
                    tuple(model.parameters()),
                )
                row = {
                    "epoch": int(epoch),
                    "optimizer_step": int(optimizer_step),
                    "batch_index": int(batch_index),
                    "task_index": int(item["task_index"]),
                    "task_id": str(task.task_id),
                    "algorithm": str(algorithm_name),
                    "amortization_mode": str(mode_name),
                    "device": str(device),
                    "fixed_pool_device": str(item.get("fixed_pool_device", fixed_pool_device)),
                    "active_variable_count": int(task.num_vars),
                    "task_batch_size": int(len(batch)),
                    "theta_dim": int(theta_dim_value),
                    "gt_injected_count": int(item["gt_injected_count"]),
                    "gt_rewrite_injected_count": int(item["gt_rewrite_injected_count"]),
                    "proxy_sample_count": int(item["proxy_sample_count"]),
                    "num_sampled_expressions": int(item["num_sampled_expressions"]),
                    "fixed_pool_size": int(len(item["base_pool"])),
                    "fixed_pool_gt_path_grad_norm": float(gt_grad_norm),
                    "amortized_fixed_pool_gt_mass": float(metrics.get("gt_proxy_gt_mass", 0.0)),
                    "amortized_fixed_pool_gap": float(metrics.get("gt_minus_proxy_logmean_gap", 0.0)),
                    "amortized_fixed_pool_pairwise_win_rate": float(metrics.get("gt_pairwise_win_rate", 0.0)),
                    "task_id_oracle_gt_mass": float(metrics.get("gt_proxy_gt_mass", 0.0)) if bool(oracle_mode) else 0.0,
                    "task_id_oracle_gap": float(metrics.get("gt_minus_proxy_logmean_gap", 0.0)) if bool(oracle_mode) else 0.0,
                    "task_id_oracle_pairwise_win_rate": float(metrics.get("gt_pairwise_win_rate", 0.0)) if bool(oracle_mode) else 0.0,
                    "loss": float(loss.detach().cpu().item()),
                    **item["proxy_diag"],
                    **metrics,
                }
                pending_rows.append(row)
            if not batch_losses:
                rows.extend(pending_rows)
                continue
            batch_loss = torch.stack(batch_losses).mean()
            opt.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            opt.step()
            for row in pending_rows:
                row["batch_loss"] = float(batch_loss.detach().cpu().item())
                rows.append(row)
                if bool(train_cfg.get("log_fixed_pool_training", True)):
                    print(
                        f"epoch {epoch} step {optimizer_step} task {row['task_id']} "
                        f"{mode_name}_fixed_pool_loss {float(row.get('gt_proxy_loss', row['loss'])):.6f}",
                        flush=True,
                    )
            optimizer_step += 1
        if bool(train_cfg.get("save_every_epoch", False)):
            _write_curve(curve_path, rows)
            save_checkpoint(epoch + 1)
    save_checkpoint(epochs)
    _write_curve(curve_path, rows)
    print(f"saved {path}")
    return path


def _run_theta_semantic_pullback_flow(cfg: dict) -> Path:
    direct_path = Path(str(cfg.get("direct_checkpoint", cfg.get("teacher_checkpoint", ""))))
    if not str(direct_path):
        raise ValueError("theta_semantic_pullback_flow requires direct_checkpoint")
    direct = torch.load(direct_path, map_location="cpu", weights_only=False)
    if str(direct.get("algorithm", "")) != "direct_theta_gt_proxy_overfit":
        raise ValueError("direct_checkpoint must come from direct_theta_gt_proxy_overfit")

    direct_cfg = dict(direct.get("cfg") or {})
    runtime = cfg.get("runtime", direct_cfg.get("runtime", {}))
    _configure_threads(runtime)
    device = _resolve_device(runtime)
    seed = int(cfg.get("seed", direct_cfg.get("seed", 0)))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)

    template = _template_from_cfg(direct_cfg)
    direct_train_cfg = dict(direct_cfg.get("train") or {})
    train_cfg = {**direct_train_cfg, **dict(cfg.get("train") or {})}
    model_cfg = {**dict(direct_cfg.get("model") or {}), **dict(cfg.get("model") or {})}
    tasks = list(_training_tasks(direct_cfg, template, rng))
    task_limit = max(int(train_cfg.get("task_limit", cfg.get("num_tasks", direct_cfg.get("num_tasks", len(tasks))))), 1)
    tasks = tasks[:task_limit]
    theta_by_task = {str(key): torch.as_tensor(value).float() for key, value in dict(direct["theta"]).items()}

    fixed_pools: list[dict] = []
    for task_index, task in enumerate(tasks):
        task_id = str(task.task_id)
        if bool(train_cfg.get("log_fixed_pool_build", False)):
            print(
                f"theta-flow fixed-pool build {task_index + 1}/{len(tasks)} {task_id}",
                flush=True,
            )
        if task_id not in theta_by_task:
            if bool(train_cfg.get("log_fixed_pool_build", False)):
                print(f"theta-flow fixed-pool skip missing theta {task_id}", flush=True)
            continue
        built = _build_fixed_gt_proxy_pool(
            task,
            template=template,
            train_cfg=train_cfg,
            rng=rng,
            device=device,
        )
        if built is None:
            if bool(train_cfg.get("log_fixed_pool_build", False)):
                print(f"theta-flow fixed-pool skip build failed {task_id}", flush=True)
            continue
        built.update({
            "task_index": int(task_index),
            "task_id": task_id,
            "theta_star": theta_by_task[task_id].to(device),
        })
        fixed_pools.append(built)
    if not fixed_pools:
        raise ValueError("no fixed pools were built for theta semantic pullback flow")

    model = ThetaVelocityNetwork(
        template,
        num_vars=int(template.num_vars),
        hidden=int(model_cfg.get("hidden", 128)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)))
    epochs = int(train_cfg.get("epochs", 10))
    task_batch_size = max(int(train_cfg.get("task_batch_size", 1)), 1)
    sem_weight = float(train_cfg.get("semantic_pullback_weight", 1.0))
    theta_mse_weight = float(train_cfg.get("theta_mse_weight", 0.0))
    terminal_weight = float(train_cfg.get("terminal_probe_weight", 0.25))
    semantic_kernel = str(train_cfg.get("semantic_kernel", "inner"))
    semantic_sigma = train_cfg.get("semantic_sigma")
    semantic_sigma = None if semantic_sigma is None else float(semantic_sigma)
    sign_invariant_kernel = bool(train_cfg.get("sign_invariant_kernel", True))
    semantic_aug_points = int(train_cfg.get("semantic_aug_points", 0))

    rows: list[dict] = []
    optimizer_step = 0
    for epoch in range(epochs):
        epoch_pools = list(fixed_pools)
        rng.shuffle(epoch_pools)
        for batch_index, batch in enumerate(_task_batches(epoch_pools, task_batch_size)):
            losses: list[torch.Tensor] = []
            pending_rows: list[dict] = []
            for item in batch:
                theta0 = torch.zeros_like(item["theta_star"], device=device)
                t_value = _sample_semantic_fm_time(train_cfg, rng)
                theta_t_base, teacher_velocity = simplex_theta_path(
                    theta0,
                    item["theta_star"].to(device),
                    template,
                    t=t_value,
                    eps=float(train_cfg.get("simplex_path_eps", 1e-8)),
                )
                theta_t = theta_t_base.detach().clone().requires_grad_(True)
                predicted_velocity = model(item["x"].to(device), item["y"].to(device), theta_t, t=t_value)
                dist = theta_vector_to_distribution(theta_t, template)
                pool = _relogprob_edge_pool(item["base_pool"], template=template, edge_distribution=dist)
                scores = _pool_logprob_scores(pool, device=device)
                weights = torch.softmax(scores.detach(), dim=0)
                x_grid = augment_grid(item["x"].to(device), count=semantic_aug_points, rng=rng)
                semantics = semantic_vectors(pool, x_grid)
                kernel = semantic_kernel_matrix(
                    semantics,
                    kind=semantic_kernel,
                    sigma=semantic_sigma,
                    sign_invariant=sign_invariant_kernel,
                ).to(device)
                pullback_loss, pullback_metrics = theta_semantic_pullback_loss(
                    scores,
                    theta_t,
                    predicted_velocity=predicted_velocity,
                    teacher_velocity=teacher_velocity.to(device),
                    weights=weights,
                    semantic_kernel=kernel,
                )
                centered_pred = center_theta_vector_by_template(predicted_velocity, template)
                centered_teacher = center_theta_vector_by_template(teacher_velocity.to(device), template)
                theta_mse = (centered_pred - centered_teacher).pow(2).mean()

                terminal_theta = theta_t + (1.0 - float(t_value)) * predicted_velocity
                terminal_pool = _relogprob_edge_pool(
                    item["base_pool"],
                    template=template,
                    edge_distribution=theta_vector_to_distribution(terminal_theta, template),
                )
                terminal_loss, terminal_metrics = gt_proxy_projection_loss(
                    terminal_pool,
                    item["target_weights"],
                    item["proxy_mask"],
                    rank_margin=float(train_cfg.get("rank_margin", 0.5)),
                    gt_weight=float(train_cfg.get("gt_mass_weight", 0.1)),
                    rank_weight=float(train_cfg.get("rank_weight", 0.0)),
                    group_weight=float(train_cfg.get("group_weight", 1.0)),
                    pair_weight=float(train_cfg.get("pair_weight", 1.0)),
                    score_temperature=float(train_cfg.get("score_temperature", 1.0)),
                    score_length_alpha=float(train_cfg.get("score_length_alpha", 0.5)),
                    entropy_weight=float(train_cfg.get("entropy_weight", 0.0)),
                    use_active_logprob=True,
                )
                total_loss = (
                    sem_weight * pullback_loss
                    + theta_mse_weight * theta_mse
                    + terminal_weight * terminal_loss
                )
                trainable_mass = float(terminal_metrics.get("gt_proxy_target_mass_trainable", 0.0))
                if trainable_mass > 0.0 and int(scores.numel()) > 0:
                    losses.append(_loss_on_training_device(total_loss, device))

                kernel_detached = kernel.detach()
                kernel_rank = (
                    int(torch.linalg.matrix_rank(kernel_detached).detach().cpu().item())
                    if int(kernel_detached.numel()) else 0
                )
                gt_grad_norm = _sample_logprob_grad_norm(
                    terminal_pool,
                    _trainable_gt_indices(terminal_pool, item["target_weights"]),
                    tuple(model.parameters()),
                )
                row = {
                    "epoch": int(epoch),
                    "optimizer_step": int(optimizer_step),
                    "batch_index": int(batch_index),
                    "task_index": int(item["task_index"]),
                    "task_id": str(item["task_id"]),
                    "algorithm": "theta_semantic_pullback_flow",
                    "device": str(device),
                    "theta_dim": int(theta_t.numel()),
                    "flow_t": float(t_value),
                    "task_batch_size": int(len(batch)),
                    "fixed_pool_size": int(len(item["base_pool"])),
                    "gt_injected_count": int(item["gt_injected_count"]),
                    "gt_rewrite_injected_count": int(item["gt_rewrite_injected_count"]),
                    "proxy_sample_count": int(item["proxy_sample_count"]),
                    "num_sampled_expressions": int(item["num_sampled_expressions"]),
                    "fixed_pool_gt_path_grad_norm": float(gt_grad_norm),
                    "theta_velocity_mse_centered": float(theta_mse.detach().cpu().item()),
                    "theta_velocity_norm": float(predicted_velocity.detach().norm().cpu().item()),
                    "theta_teacher_velocity_norm": float(teacher_velocity.detach().norm().cpu().item()),
                    "theta_semantic_kernel_trace": float(kernel_detached.trace().cpu().item()) if int(kernel_detached.numel()) else 0.0,
                    "theta_semantic_kernel_rank": int(kernel_rank),
                    "terminal_pairwise_win_rate": float(terminal_metrics.get("gt_pairwise_win_rate", 0.0)),
                    "terminal_pc_gt_mass": float(terminal_metrics.get("proposal_corrected_gt_mass", 0.0)),
                    "terminal_gt_minus_proxy_gap": float(terminal_metrics.get("gt_minus_proxy_logmean_gap", 0.0)),
                    "loss": float(total_loss.detach().cpu().item()),
                    **item["proxy_diag"],
                    **pullback_metrics,
                    **terminal_metrics,
                }
                pending_rows.append(row)
            if not losses:
                rows.extend(pending_rows)
                continue
            batch_loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()
            for row in pending_rows:
                row["batch_loss"] = float(batch_loss.detach().cpu().item())
                rows.append(row)
                print(
                    f"epoch {epoch} step {optimizer_step} task {row['task_id']} "
                    f"theta_pullback {row['theta_semantic_pullback_loss']:.6f} "
                    f"terminal_pairwise {row['terminal_pairwise_win_rate']:.3f}",
                    flush=True,
                )
            optimizer_step += 1

    out_dir = Path(cfg.get("out", "checkpoints/theta_semantic_pullback_flow"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = str(cfg.get("checkpoint_name", "theta_semantic_pullback_flow.pt"))
    path = out_dir / ckpt_name
    torch.save({
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "cfg": cfg,
        "direct_cfg": direct_cfg,
        "template": _template_payload(template),
        "algorithm": "theta_semantic_pullback_flow",
        "theta_dim": int(_template_theta_dim(template)),
        "direct_checkpoint": str(direct_path),
        "fixed_pool_count": int(len(fixed_pools)),
    }, path)
    _write_curve(out_dir / f"train_curve_{Path(ckpt_name).stem}.csv", rows)
    print(f"saved {path}")
    return path


def _run_structural_closure(cfg: dict) -> Path:
    runtime = cfg.get("runtime", {})
    _configure_threads(runtime)
    device = _resolve_device(runtime)
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    template = _template_from_cfg(cfg)
    model_cfg = _spff_model_cfg(cfg, template)
    model_cfg = replace(
        model_cfg,
        spff_enabled=False,
        term_factorized=True,
        term_num_heads=max(int(model_cfg.term_num_heads), max(int(model_cfg.head_terms), 1)),
    )
    model = ConditionalEdgeFlowModel(model_cfg).to(device)
    train_cfg = cfg.get("train", {})
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)))
    tasks = _training_tasks(cfg, template, rng)
    rows: list[dict] = []
    epochs = int(train_cfg.get("epochs", train_cfg.get("velocity_epochs", 6)))
    task_batch_size = max(int(train_cfg.get("task_batch_size", 1)), 1)
    optimizer_step = 0
    for epoch in range(int(epochs)):
        epoch_tasks = list(tasks)
        rng.shuffle(epoch_tasks)
        for batch_index, batch in enumerate(_task_batches(epoch_tasks, task_batch_size)):
            batch_losses: list[torch.Tensor] = []
            pending_rows: list[dict] = []
            for task in batch:
                x_dev = task.x.float().to(device)
                y_dev = task.y.float().to(device)
                bundle = build_structural_closure_targets(
                    str(task.ground_truth or ""),
                    variable_count=int(task.num_vars),
                    template=template,
                    x=x_dev,
                    y=y_dev,
                    max_heads=int(model_cfg.term_num_heads),
                    task_id=str(task.task_id),
                    prior_type=str(model_cfg.term_prior_type),
                    min_prob_inside_support=float(train_cfg.get("min_prob_inside_support", 0.01)),
                    constant_values=tuple(float(v) for v in model_cfg.constant_values),
                )
                loss, metrics = _structural_closure_loss(
                    model,
                    bundle,
                    device=device,
                    train_cfg=train_cfg,
                )
                if bool(bundle.supported):
                    batch_losses.append(_loss_on_training_device(loss, device))
                pending_rows.append({
                    "epoch": int(epoch),
                    "optimizer_step": int(optimizer_step),
                    "batch_index": int(batch_index),
                    "task_id": str(task.task_id),
                    "algorithm": "structural_closure",
                    "device": str(device),
                    "active_variable_count": int(task.num_vars),
                    "task_batch_size": int(len(batch)),
                    "head_terms": int(model_cfg.head_terms),
                    "term_num_heads": int(model_cfg.term_num_heads),
                    "term_prior_type": str(model_cfg.term_prior_type),
                    **bundle.diagnostics,
                    **metrics,
                })
            if not batch_losses:
                rows.extend(pending_rows)
                continue
            batch_loss = torch.stack(batch_losses).mean()
            opt.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            opt.step()
            for row in pending_rows:
                row["batch_loss"] = float(batch_loss.detach().cpu().item())
                rows.append(row)
                print(
                    f"epoch {epoch} step {optimizer_step} task {row['task_id']} "
                    f"closure_loss {row.get('loss', 0.0):.6f}",
                    flush=True,
                )
            optimizer_step += 1
    out_dir = Path(cfg.get("out", "checkpoints/closure"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = str(cfg.get("checkpoint_name", "structural_closure.pt"))
    path = out_dir / ckpt_name
    torch.save({
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "cfg": cfg,
        "model_cfg": asdict(model_cfg),
        "template": _template_payload(template),
        "algorithm": "structural_closure",
    }, path)
    _write_curve(out_dir / f"train_curve_{Path(ckpt_name).stem}.csv", rows)
    print(f"saved {path}")
    return path


def _run_structural_closure_setprod(cfg: dict, *, algorithm_name: str = "structural_closure_setprod") -> Path:
    runtime = cfg.get("runtime", {})
    _configure_threads(runtime)
    device = _resolve_device(runtime)
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)
    template = _template_from_cfg(cfg)
    model_cfg = _spff_model_cfg(cfg, template)
    model_cfg = replace(
        model_cfg,
        spff_enabled=False,
        term_factorized=True,
        term_num_heads=max(int(model_cfg.term_num_heads), max(int(model_cfg.head_terms), 1)),
    )
    model = ConditionalEdgeFlowModel(model_cfg).to(device)
    train_cfg = cfg.get("train", {})
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)))
    tasks = _training_tasks(cfg, template, rng)
    rows: list[dict] = []
    epochs = int(train_cfg.get("epochs", 6))
    task_batch_size = max(int(train_cfg.get("task_batch_size", 1)), 1)
    optimizer_step = 0
    for epoch in range(int(epochs)):
        epoch_tasks = list(tasks)
        rng.shuffle(epoch_tasks)
        for batch_index, batch in enumerate(_task_batches(epoch_tasks, task_batch_size)):
            batch_losses: list[torch.Tensor] = []
            pending_rows: list[dict] = []
            for task in batch:
                x_dev = task.x.float().to(device)
                y_dev = task.y.float().to(device)
                target_builder = (
                    build_structural_dynamic_oracle_targets
                    if str(algorithm_name) == "structural_closure_dynamic_oracle"
                    else build_structural_setprod_targets
                )
                bundle = target_builder(
                    str(task.ground_truth or ""),
                    variable_count=int(task.num_vars),
                    template=template,
                    x=x_dev,
                    y=y_dev,
                    max_heads=int(model_cfg.term_num_heads),
                    task_id=str(task.task_id),
                    prior_type=str(model_cfg.term_prior_type),
                    constant_values=tuple(float(v) for v in model_cfg.constant_values),
                )
                loss, metrics = _structural_setprod_loss(
                    model,
                    bundle,
                    device=device,
                    train_cfg=train_cfg,
                )
                if bool(bundle.supported):
                    batch_losses.append(_loss_on_training_device(loss, device))
                pending_rows.append({
                    "epoch": int(epoch),
                    "optimizer_step": int(optimizer_step),
                    "batch_index": int(batch_index),
                    "task_id": str(task.task_id),
                    "algorithm": str(algorithm_name),
                    "device": str(device),
                    "active_variable_count": int(task.num_vars),
                    "task_batch_size": int(len(batch)),
                    "head_terms": int(model_cfg.head_terms),
                    "term_num_heads": int(model_cfg.term_num_heads),
                    "term_prior_type": str(model_cfg.term_prior_type),
                    **bundle.diagnostics,
                    **metrics,
                })
            if not batch_losses:
                rows.extend(pending_rows)
                continue
            batch_loss = torch.stack(batch_losses).mean()
            opt.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            opt.step()
            for row in pending_rows:
                row["batch_loss"] = float(batch_loss.detach().cpu().item())
                rows.append(row)
                print(
                    f"epoch {epoch} step {optimizer_step} task {row['task_id']} "
                    f"setprod_loss {row.get('setprod_loss', 0.0):.6f}",
                    flush=True,
                )
            optimizer_step += 1
    out_dir = Path(cfg.get("out", "checkpoints/closure"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = str(cfg.get("checkpoint_name", f"{algorithm_name}.pt"))
    path = out_dir / ckpt_name
    torch.save({
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "cfg": cfg,
        "model_cfg": asdict(model_cfg),
        "template": _template_payload(template),
        "algorithm": str(algorithm_name),
    }, path)
    _write_curve(out_dir / f"train_curve_{Path(ckpt_name).stem}.csv", rows)
    print(f"saved {path}")
    return path


def _run_fixed_theta(cfg: dict) -> Path:
    _configure_threads(cfg.get("runtime", {}))
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    template = _template_from_cfg(cfg)
    model = EdgeFlowModel(EdgeFlowModelConfig(
        num_vars=template.num_vars,
        hidden=int(cfg.get("model", {}).get("hidden", 64)),
    ))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("train", {}).get("lr", 1e-3)))
    rows: list[dict] = []
    train_cfg = cfg.get("train", {})
    build_cfg = EdgeFlowBuildConfig(
        samples_per_task=int(train_cfg.get("samples_per_task", 64)),
        elite_k=int(train_cfg.get("elite_k", 8)),
        target_smoothing=float(train_cfg.get("target_smoothing", 1e-2)),
        complexity_weight=float(train_cfg.get("complexity_weight", 0.001)),
        projection_mode=str(train_cfg.get("projection_mode", "global_topk")),
        validation_fraction=float(train_cfg.get("validation_fraction", 0.0)),
    )
    tasks = [(task.task_id, task.x, task.y) for task in _training_tasks(cfg, template, rng)]
    epochs = int(train_cfg.get("epochs", 2))
    for epoch in range(epochs):
        records = build_edge_flow_records(template, tasks=tasks, cfg=build_cfg, rng=rng)
        for rec in records:
            pred = model(rec)
            loss, metrics = edge_flow_loss(pred, rec)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            opt.step()
            row = {
                "epoch": epoch,
                "task_id": rec.task_id,
                **metrics,
                **{k: v for k, v in rec.diagnostics.items() if isinstance(v, (int, float))},
            }
            rows.append(row)
            print(f"epoch {epoch} task {rec.task_id} loss {metrics['loss']:.6f}", flush=True)
    out_dir = Path(cfg.get("out", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = str(cfg.get("checkpoint_name", "fixed_theta_edge_flow.pt"))
    path = out_dir / ckpt_name
    torch.save({
        "model": model.state_dict(),
        "cfg": cfg,
        "model_cfg": asdict(model.cfg),
        "template": _template_payload(template),
        "algorithm": "fixed_theta_edge_flow",
    }, path)
    _write_curve(out_dir / f"train_curve_{Path(ckpt_name).stem}.csv", rows)
    print(f"saved {path}")
    return path


def _template_payload(template: RegisterOperatorTemplate) -> dict:
    return {
        "num_vars": template.num_vars,
        "num_registers": template.num_registers,
        "num_layers": template.num_layers,
        "primitives": list(template.primitives),
        "mixture_modes": template.mixture_modes,
    }


def _template_from_cfg(cfg: dict) -> RegisterOperatorTemplate:
    t = cfg.get("template", {})
    return RegisterOperatorTemplate(
        num_vars=int(t.get("num_vars", cfg.get("gen", {}).get("num_vars", 1))),
        num_registers=int(t.get("num_registers", 4)),
        num_layers=int(t.get("num_layers", 2)),
        primitives=tuple(t.get("primitives", ["add", "mul", "square"])),
        mixture_modes=int(t.get("mixture_modes", 1)),
    )


def _spff_model_cfg(cfg: dict, template: RegisterOperatorTemplate) -> ConditionalEdgeFlowConfig:
    model_cfg = cfg.get("model", {})
    return ConditionalEdgeFlowConfig(
        num_vars=template.num_vars,
        hidden=int(model_cfg.get("hidden", 96)),
        head_terms=int(model_cfg.get("head_terms", cfg.get("head", {}).get("terms", 3))),
        branches_per_register=int(model_cfg.get("branches_per_register", 1)),
        update_mode=str(model_cfg.get("update_mode", "carry_write")),
        write_registers_per_layer=int(model_cfg.get("write_registers_per_layer", 0)),
        exclude_base_head_candidates=bool(model_cfg.get("exclude_base_head_candidates", True)),
        enable_keep_option=bool(model_cfg.get("enable_keep_option", False)),
        mask_duplicate_branches=bool(model_cfg.get("mask_duplicate_branches", False)),
        include_base_source_pool=bool(model_cfg.get("include_base_source_pool", False)),
        task_encoder=str(model_cfg.get("task_encoder", "mean")),
        spff_enabled=bool(model_cfg.get("spff_enabled", True)),
        term_factorized=bool(model_cfg.get("term_factorized", cfg.get("term", {}).get("enabled", False))),
        term_num_heads=int(model_cfg.get("term_num_heads", cfg.get("term", {}).get("num_heads", model_cfg.get("head_terms", 3)))),
        term_prior_strength=float(model_cfg.get("term_prior_strength", cfg.get("term", {}).get("prior_strength", 0.0))),
        term_prior_type=str(model_cfg.get("term_prior_type", cfg.get("term", {}).get("prior_type", "natural"))),
        spff_geometry=str(model_cfg.get("spff_geometry", cfg.get("spff", {}).get("geometry", "pullback"))),
        spff_chart_type=str(model_cfg.get("spff_chart_type", cfg.get("spff", {}).get("chart_type", "ode"))),
        spff_context_mode=str(model_cfg.get("spff_context_mode", cfg.get("spff", {}).get("context_mode", "semantic"))),
        spff_num_candidates=int(model_cfg.get("spff_num_candidates", template.num_registers)),
        spff_sem_dim=int(model_cfg.get("spff_sem_dim", cfg.get("spff", {}).get("sem_dim", model_cfg.get("hidden", 96)))),
        spff_chart_hidden=int(model_cfg.get("spff_chart_hidden", cfg.get("spff", {}).get("chart_hidden", model_cfg.get("hidden", 96)))),
        spff_velocity_hidden=int(model_cfg.get("spff_velocity_hidden", cfg.get("spff", {}).get("velocity_hidden", model_cfg.get("hidden", 96)))),
        spff_chart_rank=int(model_cfg.get("spff_chart_rank", cfg.get("spff", {}).get("chart_rank", 2))),
        spff_ode_steps=int(model_cfg.get("spff_ode_steps", cfg.get("spff", {}).get("ode_steps", 4))),
        spff_inference_steps=int(model_cfg.get("spff_inference_steps", cfg.get("spff", {}).get("inference_steps", 4))),
        spff_max_chart_velocity=float(model_cfg.get("spff_max_chart_velocity", cfg.get("spff", {}).get("max_chart_velocity", 0.05))),
        spff_max_velocity=float(model_cfg.get("spff_max_velocity", cfg.get("spff", {}).get("max_velocity", 1.0))),
        constant_values=tuple(float(v) for v in model_cfg.get("constant_values", cfg.get("constants", {}).get("values", [1.0]))),
    )


def _spff_chart_parameters(model: ConditionalEdgeFlowModel):
    if getattr(model, "spff_chart", None) is not None:
        yield from model.spff_chart.parameters()


def _spff_velocity_parameters(model: ConditionalEdgeFlowModel):
    if getattr(model, "spff_context_encoder", None) is not None:
        yield from model.spff_context_encoder.parameters()
    if getattr(model, "spff_velocity_head", None) is not None:
        yield from model.spff_velocity_head.parameters()


def _set_requires_grad(parameters, value: bool) -> None:
    for param in parameters:
        param.requires_grad_(bool(value))


def _spff_loss_for_records(
    model: ConditionalEdgeFlowModel,
    records: list[InputSelectionFlowRecord],
    *,
    device: torch.device,
    train_cfg: dict,
    train_chart: bool,
    train_velocity: bool,
    rng: random.Random,
) -> tuple[torch.Tensor, dict]:
    loss_terms: list[torch.Tensor] = []
    velocity_losses: list[float] = []
    iso_losses: list[float] = []
    identity_losses: list[float] = []
    roundtrip_losses: list[float] = []
    deformations: list[float] = []
    conditions: list[float] = []
    endpoint_distances: list[float] = []
    entropies: list[float] = []
    ranks: list[float] = []
    top1s: list[float] = []
    top3s: list[float] = []
    skipped = 0
    for record in records:
        if int(record.p0.numel()) != int(model.spff_num_candidates):
            skipped += 1
            continue
        mask = record.active_source_mask.to(device=device, dtype=torch.bool).flatten()
        p0 = record.p0.to(device=device, dtype=torch.float32).flatten().unsqueeze(0)
        p1 = record.p1.to(device=device, dtype=torch.float32).flatten().unsqueeze(0)
        candidate_semantics = record.candidate_semantics.to(device=device, dtype=torch.float32)
        target_semantics = record.target_semantics.to(device=device, dtype=torch.float32)
        try:
            sem_context = model.spff_context_from_semantics(
                candidate_semantics,
                target_semantics,
                layer_id=int(record.layer),
                target_register=int(record.target_register),
                branch_id=int(record.branch_id),
                arity_slot=int(record.slot),
                primitive_id=int(record.operator_id),
                source_mask=mask,
            )
        except Exception:
            skipped += 1
            continue
        if train_chart:
            if str(model.cfg.spff_geometry).lower() in {"simplex", "fisher", "plain_simplex"}:
                skipped += 1
            else:
                reg = chart_regularization(
                    model.spff_chart,
                    sem_context,
                    candidate_semantics.unsqueeze(0),
                    mask=mask.unsqueeze(0),
                    vertex_smoothing=float(train_cfg.get("spff_vertex_smoothing", 0.02)),
                    iso_weight=float(train_cfg.get("spff_chart_iso_weight", 1.0)),
                    identity_weight=float(train_cfg.get("spff_chart_identity_weight", 0.0)),
                    roundtrip_weight=float(train_cfg.get("spff_chart_roundtrip_weight", 0.0)),
                )
                loss_terms.append(reg.loss)
                iso_losses.append(float(reg.diagnostics.get("spff_chart_iso_loss", 0.0)))
                identity_losses.append(float(reg.diagnostics.get("spff_chart_identity_loss", 0.0)))
                roundtrip_losses.append(float(reg.diagnostics.get("spff_chart_roundtrip_loss", 0.0)))
                deformations.append(float(reg.diagnostics.get("spff_chart_deformation_norm", 0.0)))
                conditions.append(float(reg.diagnostics.get("spff_chart_condition_number", 1.0)))
        if train_velocity:
            t_value = _spff_sample_time(train_cfg, rng)
            if str(model.cfg.spff_geometry).lower() in {"simplex", "fisher", "plain_simplex"}:
                teacher = build_simplex_teacher(
                    p0,
                    p1,
                    torch.tensor([t_value], device=device),
                    mask=mask.unsqueeze(0),
                )
            else:
                teacher = build_pullback_teacher(
                    p0,
                    p1,
                    sem_context,
                    model.spff_chart,
                    torch.tensor([t_value], device=device),
                    mask=mask.unsqueeze(0),
                )
            pred = model.spff_velocity(sem_context, teacher.r_t, torch.tensor([t_value], device=device), mask=mask.unsqueeze(0))
            target = project_tangent(teacher.rdot_t.detach(), teacher.r_t.detach(), mask=mask.unsqueeze(0))
            err = (pred - target).pow(2)
            active = mask.unsqueeze(0).to(err.dtype)
            vel_loss = (err * active).sum() / active.sum().clamp_min(1.0)
            loss_terms.append(vel_loss)
            velocity_losses.append(float(vel_loss.detach().cpu().item()))
            endpoint_distances.append(float(teacher.diagnostics.get("spff_r_endpoint_distance", 0.0)))
            entropies.append(float(teacher.diagnostics.get("spff_p_entropy_after_flow", 0.0)))
        with torch.no_grad():
            probs = model.spff_source_probs_from_context(sem_context, source_mask=mask)
            gt = int(record.gt_source_index if record.gt_source_index is not None else int(torch.argmax(p1).item()))
            chosen = probs[max(0, min(gt, int(probs.numel()) - 1))]
            rank = int((probs > chosen).sum().item()) + 1
            ranks.append(float(rank))
            top1s.append(1.0 if rank == 1 else 0.0)
            top3s.append(1.0 if rank <= 3 else 0.0)
    if not loss_terms:
        zero = torch.zeros((), device=device, requires_grad=True)
        return zero, {
            "loss": 0.0,
            "spff_velocity_loss": 0.0,
            "spff_chart_iso_loss": 0.0,
            "spff_chart_identity_loss": 0.0,
            "spff_chart_roundtrip_loss": 0.0,
            "spff_chart_deformation_norm": 0.0,
            "spff_chart_condition_number": 0.0,
            "spff_r_endpoint_distance": 0.0,
            "spff_p_entropy_after_flow": 0.0,
            "spff_gt_source_rank": 0.0,
            "spff_gt_source_top1": 0.0,
            "spff_gt_source_top3": 0.0,
            "spff_train_record_count": int(len(records)),
            "spff_skipped_record_count": int(skipped),
            "spff_geometry": str(model.cfg.spff_geometry),
        }
    loss = torch.stack(loss_terms).mean()
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "spff_velocity_loss": _mean(velocity_losses),
        "spff_chart_iso_loss": _mean(iso_losses),
        "spff_chart_identity_loss": _mean(identity_losses),
        "spff_chart_roundtrip_loss": _mean(roundtrip_losses),
        "spff_chart_deformation_norm": _mean(deformations),
        "spff_chart_condition_number": _mean(conditions),
        "spff_r_endpoint_distance": _mean(endpoint_distances),
        "spff_p_entropy_after_flow": _mean(entropies),
        "spff_gt_source_rank": _mean(ranks),
        "spff_gt_source_top1": _mean(top1s),
        "spff_gt_source_top3": _mean(top3s),
        "spff_train_record_count": int(len(records)),
        "spff_skipped_record_count": int(skipped),
        "spff_geometry": str(model.cfg.spff_geometry),
    }


def _structural_closure_loss(
    model: ConditionalEdgeFlowModel,
    bundle: StructuralClosureTargetBundle,
    *,
    device: torch.device,
    train_cfg: dict,
) -> tuple[torch.Tensor, dict]:
    loss_terms: list[torch.Tensor] = []
    op_exact_losses: list[float] = []
    op_family_losses: list[float] = []
    src_losses: list[float] = []
    op_top1: list[float] = []
    op_top3: list[float] = []
    op_family_top1: list[float] = []
    src_top1: list[float] = []
    src_top3: list[float] = []
    src_ranks: list[float] = []
    skipped_op = 0
    skipped_src = 0
    op_exact_weight = float(train_cfg.get("operator_exact_weight", 1.0))
    op_family_weight = float(train_cfg.get("operator_family_weight", 1.0))
    source_weight = float(train_cfg.get("source_weight", 1.0))
    method = str(train_cfg.get("sampler_method", "policy"))
    flow_steps = int(train_cfg.get("flow_steps", 1))
    for record in bundle.operator_records:
        try:
            dist = _operator_distribution_for_closure_record(
                model,
                record,
                device=device,
                method=method,
                flow_steps=flow_steps,
            )
        except Exception:
            skipped_op += 1
            continue
        probs = dist["probs"].clamp_min(1e-12)
        gt_idx = int(record.gt_operator_index)
        exact_loss = -probs[gt_idx].log()
        family_probs = _operator_family_probs(probs, record.candidate_operator_ids).clamp_min(1e-12)
        family_loss = -family_probs[int(record.gt_family_id)].log()
        if op_exact_weight:
            loss_terms.append(float(op_exact_weight) * exact_loss)
        if op_family_weight:
            loss_terms.append(float(op_family_weight) * family_loss)
        op_exact_losses.append(float(exact_loss.detach().cpu().item()))
        op_family_losses.append(float(family_loss.detach().cpu().item()))
        gt_prob = probs[gt_idx]
        rank = int((probs > gt_prob).sum().detach().cpu().item()) + 1
        op_top1.append(1.0 if rank == 1 else 0.0)
        op_top3.append(1.0 if rank <= 3 else 0.0)
        op_family_top1.append(1.0 if int(torch.argmax(family_probs).item()) == int(record.gt_family_id) else 0.0)
    for record in bundle.source_records:
        try:
            dist = _source_distribution_for_closure_record(
                model,
                record,
                device=device,
                method=method,
                flow_steps=flow_steps,
            )
        except Exception:
            skipped_src += 1
            continue
        probs = dist["probs"].clamp_min(1e-12)
        valid_indices = [idx for idx in record.equivalent_source_indices if 0 <= int(idx) < int(probs.numel())]
        if not valid_indices:
            skipped_src += 1
            continue
        idx_tensor = torch.tensor(valid_indices, dtype=torch.long, device=probs.device)
        set_prob = probs.index_select(0, idx_tensor).sum().clamp_min(1e-12)
        src_loss = -set_prob.log()
        if source_weight:
            loss_terms.append(float(source_weight) * src_loss)
        src_losses.append(float(src_loss.detach().cpu().item()))
        best_gt = probs.index_select(0, idx_tensor).max()
        rank = int((probs > best_gt).sum().detach().cpu().item()) + 1
        top_idx = int(torch.argmax(probs).detach().cpu().item())
        src_ranks.append(float(rank))
        src_top1.append(1.0 if top_idx in set(valid_indices) else 0.0)
        src_top3.append(1.0 if rank <= 3 else 0.0)
    if not loss_terms:
        zero = torch.zeros((), device=device, requires_grad=True)
        return zero, _structural_zero_metrics(bundle, skipped_op=skipped_op, skipped_src=skipped_src)
    loss = torch.stack(loss_terms).mean()
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "closure_operator_exact_loss": _mean(op_exact_losses),
        "closure_operator_family_loss": _mean(op_family_losses),
        "closure_source_loss": _mean(src_losses),
        "closure_operator_top1": _mean(op_top1),
        "closure_operator_top3": _mean(op_top3),
        "closure_operator_family_top1": _mean(op_family_top1),
        "closure_source_top1": _mean(src_top1),
        "closure_source_top3": _mean(src_top3),
        "closure_source_rank": _mean(src_ranks),
        "closure_operator_skipped": int(skipped_op),
        "closure_source_skipped": int(skipped_src),
        "closure_supported_supervised": int(bundle.supported),
    }


def _structural_setprod_loss(
    model: ConditionalEdgeFlowModel,
    bundle: StructuralSetProdTargetBundle,
    *,
    device: torch.device,
    train_cfg: dict,
) -> tuple[torch.Tensor, dict]:
    method = str(train_cfg.get("sampler_method", train_cfg.get("method", "policy")))
    flow_steps = int(train_cfg.get("flow_steps", 1))
    flow_time = float(train_cfg.get("flow_time", 1.0))
    pair_losses: dict[tuple[int, int], list[torch.Tensor]] = {}
    top1s: list[float] = []
    top3s: list[float] = []
    op_top1s: list[float] = []
    op_family_top1s: list[float] = []
    stop_top1s: list[float] = []
    nlls: list[float] = []
    skipped = 0
    for record in bundle.production_records:
        try:
            record_loss, record_metrics = _fast_production_loss_for_record(
                model,
                record,
                device=device,
                method=method,
                flow_steps=flow_steps,
                flow_time=flow_time,
            )
        except Exception:
            skipped += 1
            continue
        pair_key = (int(record.head), int(record.term_index))
        pair_losses.setdefault(pair_key, []).append(record_loss)
        nlls.append(float(record_loss.detach().cpu().item()))
        top1s.append(float(record_metrics["production_top1"]))
        top3s.append(float(record_metrics["production_top3"]))
        op_top1s.append(float(record_metrics["operator_top1_derived"]))
        op_family_top1s.append(float(record_metrics["operator_family_top1_derived"]))
        if float(record_metrics.get("stop_record", 0.0)) > 0.0:
            stop_top1s.append(float(record_metrics.get("stop_top1", 0.0)))
    assignment_loss, assignment_info = _setprod_assignment_loss(pair_losses, device=device)
    if assignment_loss is None:
        zero = torch.zeros((), device=device, requires_grad=True)
        return zero, {
            "loss": 0.0,
            "setprod_loss": 0.0,
            "production_nll": 0.0,
            "production_top1": 0.0,
            "production_top3": 0.0,
            "production_valid_top1": 0.0,
            "production_valid_top3": 0.0,
            "production_exact_top1": 0.0,
            "stop_top1": 0.0,
            "operator_top1_derived": 0.0,
            "operator_family_top1_derived": 0.0,
            "head_assignment_cost": 0.0,
            "head_assignment_count": 0,
            "head_assignment": "",
            "setprod_record_count_train": int(len(bundle.production_records)),
            "setprod_skipped": int(skipped),
            "setprod_supported_supervised": int(bundle.supported),
            "setprod_sampler_method": str(method),
            "setprod_flow_steps": int(flow_steps),
            "setprod_flow_loss": 0.0,
        }
    total = assignment_loss
    return total, {
        "loss": float(total.detach().cpu().item()),
        "setprod_loss": float(total.detach().cpu().item()),
        "setprod_flow_loss": float(total.detach().cpu().item()) if method.lower() == "ode" else 0.0,
        "production_nll": _mean(nlls),
        "production_top1": _mean(top1s),
        "production_top3": _mean(top3s),
        "production_valid_top1": _mean(top1s),
        "production_valid_top3": _mean(top3s),
        "production_exact_top1": _mean(top1s) if str(bundle.diagnostics.get("target_type", "")) != "dynamic_oracle_positive_set" else 0.0,
        "stop_top1": _mean(stop_top1s),
        "operator_top1_derived": _mean(op_top1s),
        "operator_family_top1_derived": _mean(op_family_top1s),
        "head_assignment_cost": float(total.detach().cpu().item()),
        "head_assignment_count": int(assignment_info["assignment_count"]),
        "head_assignment": str(assignment_info["assignment"]),
        "setprod_record_count_train": int(len(bundle.production_records)),
        "setprod_skipped": int(skipped),
        "setprod_supported_supervised": int(bundle.supported),
        "setprod_sampler_method": str(method),
        "setprod_flow_steps": int(flow_steps),
    }


def _fast_production_loss_for_record(
    model: ConditionalEdgeFlowModel,
    record: ProductionRecord,
    *,
    device: torch.device,
    method: str = "policy",
    flow_steps: int = 1,
    flow_time: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Exact set-production NLL without enumerating unrelated productions.

    Since the chain distribution is normalized,
    sum_{op,k} q(op) prod_q q(k_q | op,k_<q) = 1.  The NLL therefore only needs
    the probability mass assigned to the equivalent production set.
    """

    regs = list(record.register_exprs)
    reg_sem = record.register_semantics.to(device=device, dtype=torch.float32)
    target = record.target_semantics.to(device=device, dtype=torch.float32)
    head_ctx_sem = record.head_context_semantics.to(device=device, dtype=torch.float32)
    head_ctx_active = record.head_context_active_mask.to(device=device, dtype=torch.bool)
    x_stub = _closure_x_stub(reg_sem, int(model.cfg.num_vars))
    tokens = model.register_tokens(x_stub, target, regs, reg_sem, layer_id=int(record.layer))
    target_token = model.head_context_target_token(
        x_stub,
        target,
        list(record.head_context_exprs),
        head_ctx_sem,
        head_index=int(record.head),
        layer_id=int(record.layer),
        active_mask=head_ctx_active,
    )
    active = record.active_register_mask.to(device=device, dtype=torch.bool)
    equivalent = tuple(sorted(set(
        (int(op_id), tuple(int(v) for v in sources))
        for op_id, sources in record.equivalent_productions
    )))
    if not equivalent:
        raise ValueError("production record has no equivalent productions")
    if (
        not record.candidate_operator_ids
        and all(int(op_id) == LEAF_PRODUCTION_ID for op_id, _ in equivalent)
    ):
        return _fast_leaf_production_loss(
            model,
            record,
            regs,
            tokens,
            active,
            device=device,
            method=method,
            flow_steps=flow_steps,
            flow_time=flow_time,
        )

    operator_ids = tuple(int(v) for v in record.candidate_operator_ids)
    if not operator_ids:
        raise ValueError("operator production record has no candidate operators")
    op_dist = model.operator_probs(
        target_token=target_token,
        primitive_ids=list(operator_ids),
        layer_id=int(record.layer),
        target_register=int(record.head),
        branch_id=0,
        method=str(method),
        flow_steps=int(flow_steps),
        return_details=True,
        flow_time=float(flow_time),
    )
    op_probs = op_dist["probs"].clamp_min(1e-12)
    op_pos = {int(op_id): idx for idx, op_id in enumerate(operator_ids)}
    source_cache: dict[tuple[int, int, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]] = {}

    def source_distribution(op_id: int, slot: int, chosen: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(op_id), int(slot), tuple(int(v) for v in chosen))
        cached = source_cache.get(key)
        if cached is not None:
            return cached
        prior_operator_id = None if int(op_id) == STOP_PRODUCTION_ID else int(op_id)
        source_p0 = construction_source_prior(
            regs,
            active,
            operator_id=prior_operator_id,
            prior_type=str(record.prior_type),
            chosen_sources=tuple(int(v) for v in chosen),
            min_prob=0.0,
        ).to(device=device, dtype=torch.float32)
        source_mask = active & (source_p0 > 0.0)
        dist = model.source_probs(
            target_token=target_token,
            source_tokens=tokens,
            layer_id=int(record.layer),
            target_register=int(record.head),
            branch_id=0,
            arity_slot=int(slot),
            primitive_id=int(op_id),
            method=str(method),
            flow_steps=int(flow_steps),
            source_mask=source_mask,
            source_p0=source_p0,
            return_details=True,
            flow_time=float(flow_time),
        )
        out = (dist["probs"].clamp_min(1e-12), source_mask)
        source_cache[key] = out
        return out

    eq_probs: list[torch.Tensor] = []
    eq_ops: set[int] = set()
    eq_source_top1 = False
    eq_source_top3 = False
    best_component_rank = 10**9
    stop_record = False
    stop_source_top1 = False
    for op_id, sources in equivalent:
        if int(op_id) not in op_pos:
            continue
        if int(op_id) == STOP_PRODUCTION_ID:
            arity = 1
            stop_record = True
        elif int(op_id) < 0:
            continue
        else:
            arity = int(get_op(int(op_id)).arity)
        if arity != len(sources):
            continue
        prob = op_probs[int(op_pos[int(op_id)])]
        source_ranks: list[int] = []
        valid = True
        chosen_prefix: list[int] = []
        for slot, src_idx in enumerate(sources):
            src_probs, source_mask = source_distribution(int(op_id), int(slot), tuple(chosen_prefix))
            idx = int(src_idx)
            if idx < 0 or idx >= int(src_probs.numel()) or not bool(source_mask[idx].item()):
                valid = False
                break
            prob = prob * src_probs[idx]
            rank = int((src_probs > src_probs[idx]).sum().detach().cpu().item()) + 1
            source_ranks.append(rank)
            chosen_prefix.append(idx)
        if not valid:
            continue
        eq_probs.append(prob)
        eq_ops.add(int(op_id))
        component_rank = max(source_ranks or [1])
        best_component_rank = min(best_component_rank, component_rank)
        eq_source_top1 = eq_source_top1 or component_rank == 1
        eq_source_top3 = eq_source_top3 or component_rank <= 3
        if int(op_id) == STOP_PRODUCTION_ID:
            stop_source_top1 = stop_source_top1 or component_rank == 1
    if not eq_probs:
        raise ValueError("equivalent production set has zero valid support")
    eq_prob = torch.stack(eq_probs).sum().clamp_min(1e-12)
    loss = -eq_prob.log()
    op_argmax = int(torch.argmax(op_probs).detach().cpu().item())
    pred_op_id = int(operator_ids[op_argmax])
    eq_op_positions = [op_pos[op_id] for op_id in eq_ops if op_id in op_pos]
    best_eq_op_prob = op_probs[eq_op_positions].max() if eq_op_positions else torch.zeros((), device=device)
    op_rank = int((op_probs > best_eq_op_prob).sum().detach().cpu().item()) + 1
    pred_family = _operator_family_id_for_op(pred_op_id)
    eq_families = {_operator_family_id_for_op(op_id) for op_id in eq_ops}
    return loss, {
        "production_top1": 1.0 if (pred_op_id in eq_ops and eq_source_top1) else 0.0,
        "production_top3": 1.0 if (op_rank <= 3 and eq_source_top3) else 0.0,
        "operator_top1_derived": 1.0 if pred_op_id in eq_ops else 0.0,
        "operator_family_top1_derived": 1.0 if pred_family in eq_families else 0.0,
        "stop_top1": 1.0 if (stop_record and pred_op_id == STOP_PRODUCTION_ID and stop_source_top1) else 0.0,
        "stop_record": 1.0 if stop_record else 0.0,
    }


def _fast_leaf_production_loss(
    model: ConditionalEdgeFlowModel,
    record: ProductionRecord,
    regs: list[Expr],
    tokens: torch.Tensor,
    active: torch.Tensor,
    *,
    device: torch.device,
    method: str = "policy",
    flow_steps: int = 1,
    flow_time: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    source_p0 = construction_source_prior(
        regs,
        active,
        operator_id=None,
        prior_type=str(record.prior_type),
        chosen_sources=tuple(),
        min_prob=0.0,
    ).to(device=device, dtype=torch.float32)
    leaf_mask = active & (source_p0 > 0.0)
    target_token = model.head_context_target_token(
        _closure_x_stub(record.register_semantics.to(device=device, dtype=torch.float32), int(model.cfg.num_vars)),
        record.target_semantics.to(device=device, dtype=torch.float32),
        list(record.head_context_exprs),
        record.head_context_semantics.to(device=device, dtype=torch.float32),
        head_index=int(record.head),
        layer_id=int(record.layer),
        active_mask=record.head_context_active_mask.to(device=device, dtype=torch.bool),
    )
    dist = model.source_probs(
        target_token=target_token,
        source_tokens=tokens,
        layer_id=int(record.layer),
        target_register=int(record.head),
        branch_id=0,
        arity_slot=0,
        primitive_id=-1,
        method=str(method),
        flow_steps=int(flow_steps),
        source_mask=leaf_mask,
        source_p0=source_p0,
        return_details=True,
        flow_time=float(flow_time),
    )
    probs = dist["probs"].clamp_min(1e-12)
    valid_indices = [
        int(sources[0])
        for op_id, sources in record.equivalent_productions
        if int(op_id) < 0 and len(sources) == 1 and 0 <= int(sources[0]) < int(probs.numel())
        and bool(leaf_mask[int(sources[0])].item())
    ]
    if not valid_indices:
        raise ValueError("leaf equivalent source set has zero valid support")
    idx = torch.tensor(sorted(set(valid_indices)), dtype=torch.long, device=probs.device)
    eq_prob = probs.index_select(0, idx).sum().clamp_min(1e-12)
    loss = -eq_prob.log()
    best_eq = probs.index_select(0, idx).max()
    rank = int((probs > best_eq).sum().detach().cpu().item()) + 1
    top_idx = int(torch.argmax(probs).detach().cpu().item())
    return loss, {
        "production_top1": 1.0 if top_idx in set(valid_indices) else 0.0,
        "production_top3": 1.0 if rank <= 3 else 0.0,
        "operator_top1_derived": 1.0,
        "operator_family_top1_derived": 1.0,
    }


def _setprod_assignment_loss(
    pair_losses: dict[tuple[int, int], list[torch.Tensor]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor | None, dict]:
    if not pair_losses:
        return None, {"assignment": "", "assignment_count": 0}
    heads = sorted({int(head) for head, _ in pair_losses})
    terms = sorted({int(term) for _, term in pair_losses})
    if not heads or not terms or len(heads) < len(terms):
        return None, {"assignment": "", "assignment_count": 0}
    pair_means: dict[tuple[int, int], torch.Tensor] = {
        key: torch.stack(values).mean()
        for key, values in pair_losses.items()
        if values
    }
    if not pair_means:
        return None, {"assignment": "", "assignment_count": 0}
    large = 1.0e6
    best_perm: tuple[int, ...] | None = None
    best_score: float | None = None
    for perm in itertools.permutations(heads, len(terms)):
        score = 0.0
        valid = True
        for head, term in zip(perm, terms):
            value = pair_means.get((int(head), int(term)))
            if value is None:
                score += large
                valid = False
            else:
                score += float(value.detach().cpu().item())
        if best_score is None or score < best_score:
            best_score = score
            best_perm = tuple(int(v) for v in perm)
        if valid and score == 0.0:
            break
    if best_perm is None:
        return None, {"assignment": "", "assignment_count": 0}
    selected: list[torch.Tensor] = []
    assignment_parts: list[str] = []
    for head, term in zip(best_perm, terms):
        value = pair_means.get((int(head), int(term)))
        if value is None:
            continue
        selected.append(value)
        assignment_parts.append(f"T{int(term)}->H{int(head)}")
    if not selected:
        return None, {"assignment": "", "assignment_count": 0}
    return torch.stack(selected).mean().to(device), {
        "assignment": ";".join(assignment_parts),
        "assignment_count": int(len(selected)),
    }


def _production_distribution_for_record(
    model: ConditionalEdgeFlowModel,
    record: ProductionRecord,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[int], list[tuple[int, tuple[int, ...]]]]:
    regs = list(record.register_exprs)
    reg_sem = record.register_semantics.to(device=device, dtype=torch.float32)
    target = record.target_semantics.to(device=device, dtype=torch.float32)
    head_ctx_sem = record.head_context_semantics.to(device=device, dtype=torch.float32)
    head_ctx_active = record.head_context_active_mask.to(device=device, dtype=torch.bool)
    x_stub = _closure_x_stub(reg_sem, int(model.cfg.num_vars))
    tokens = model.register_tokens(x_stub, target, regs, reg_sem, layer_id=int(record.layer))
    target_token = model.head_context_target_token(
        x_stub,
        target,
        list(record.head_context_exprs),
        head_ctx_sem,
        head_index=int(record.head),
        layer_id=int(record.layer),
        active_mask=head_ctx_active,
    )
    operator_ids = tuple(int(v) for v in record.candidate_operator_ids)
    active = record.active_register_mask.to(device=device, dtype=torch.bool)
    productions: list[tuple[int, tuple[int, ...]]] = []
    probs: list[torch.Tensor] = []
    if (
        not operator_ids
        and all(int(op_id) == LEAF_PRODUCTION_ID for op_id, _ in record.equivalent_productions)
    ):
        source_p0 = construction_source_prior(
            regs,
            active,
            operator_id=None,
            prior_type=str(record.prior_type),
            chosen_sources=tuple(),
            min_prob=0.0,
        ).to(device=device, dtype=torch.float32)
        leaf_mask = active & (source_p0 > 0.0)
        dist = model.source_probs(
            target_token=target_token,
            source_tokens=tokens,
            layer_id=int(record.layer),
            target_register=int(record.head),
            branch_id=0,
            arity_slot=0,
            primitive_id=-1,
            method="policy",
            flow_steps=1,
            source_mask=leaf_mask,
            source_p0=source_p0,
            return_details=True,
            flow_time=1.0,
        )
        for idx in range(int(active.numel())):
            if bool(leaf_mask[idx].item()):
                productions.append((LEAF_PRODUCTION_ID, (int(idx),)))
                probs.append(dist["probs"][int(idx)])
        if not probs:
            raise ValueError("leaf production record has no valid sources")
        prod_probs = torch.stack(probs).flatten()
        prod_probs = prod_probs / prod_probs.sum().clamp_min(1e-12)
        eq_set = set((int(op_id), tuple(int(v) for v in sources)) for op_id, sources in record.equivalent_productions)
        equivalent_indices = [idx for idx, prod in enumerate(productions) if prod in eq_set]
        return prod_probs, equivalent_indices, productions
    op_dist = model.operator_probs(
        target_token=target_token,
        primitive_ids=list(operator_ids),
        layer_id=int(record.layer),
        target_register=int(record.head),
        branch_id=0,
        method="policy",
        flow_steps=1,
        return_details=True,
        flow_time=1.0,
    )
    for op_pos, op_id in enumerate(operator_ids):
        if int(op_id) == STOP_PRODUCTION_ID:
            source_p0 = construction_source_prior(
                regs,
                active,
                operator_id=None,
                prior_type=str(record.prior_type),
                chosen_sources=tuple(),
                min_prob=0.0,
            ).to(device=device, dtype=torch.float32)
            source_mask = active & (source_p0 > 0.0)
            dist = model.source_probs(
                target_token=target_token,
                source_tokens=tokens,
                layer_id=int(record.layer),
                target_register=int(record.head),
                branch_id=0,
                arity_slot=0,
                primitive_id=STOP_PRODUCTION_ID,
                method="policy",
                flow_steps=1,
                source_mask=source_mask,
                source_p0=source_p0,
                return_details=True,
                flow_time=1.0,
            )
            for idx in range(int(active.numel())):
                if bool(source_mask[idx].item()):
                    productions.append((STOP_PRODUCTION_ID, (int(idx),)))
                    probs.append(op_dist["probs"][int(op_pos)] * dist["probs"][int(idx)])
            continue
        if int(op_id) < 0:
            continue
        op = get_op(int(op_id))
        if int(op.arity) <= 0:
            continue
        if op.arity == 1:
            source_p0 = construction_source_prior(
                regs,
                active,
                operator_id=int(op_id),
                prior_type=str(record.prior_type),
                chosen_sources=tuple(),
                min_prob=0.0,
            ).to(device=device, dtype=torch.float32)
            source_mask = active & (source_p0 > 0.0)
            dist = model.source_probs(
                target_token=target_token,
                source_tokens=tokens,
                layer_id=int(record.layer),
                target_register=int(record.head),
                branch_id=0,
                arity_slot=0,
                primitive_id=int(op_id),
                method="policy",
                flow_steps=1,
                source_mask=source_mask,
                source_p0=source_p0,
                return_details=True,
                flow_time=1.0,
            )
            for idx in range(int(active.numel())):
                if bool(source_mask[idx].item()):
                    productions.append((int(op_id), (int(idx),)))
                    probs.append(op_dist["probs"][int(op_pos)] * dist["probs"][int(idx)])
        elif op.arity == 2:
            first_p0 = construction_source_prior(
                regs,
                active,
                operator_id=int(op_id),
                prior_type=str(record.prior_type),
                chosen_sources=tuple(),
                min_prob=0.0,
            ).to(device=device, dtype=torch.float32)
            first_mask = active & (first_p0 > 0.0)
            first = model.source_probs(
                target_token=target_token,
                source_tokens=tokens,
                layer_id=int(record.layer),
                target_register=int(record.head),
                branch_id=0,
                arity_slot=0,
                primitive_id=int(op_id),
                method="policy",
                flow_steps=1,
                source_mask=first_mask,
                source_p0=first_p0,
                return_details=True,
                flow_time=1.0,
            )
            first_indices = [idx for idx in range(int(active.numel())) if bool(first_mask[idx].item())]
            second_cache: dict[tuple[bool, ...], tuple[torch.Tensor, torch.Tensor]] = {}
            for i in first_indices:
                second_p0 = construction_source_prior(
                    regs,
                    active,
                    operator_id=int(op_id),
                    prior_type=str(record.prior_type),
                    chosen_sources=(int(i),),
                    min_prob=0.0,
                ).to(device=device, dtype=torch.float32)
                second_mask = active & (second_p0 > 0.0)
                cache_key = tuple(bool(value) for value in second_mask.detach().cpu().tolist())
                cached = second_cache.get(cache_key)
                if cached is None:
                    second = model.source_probs(
                        target_token=target_token,
                        source_tokens=tokens,
                        layer_id=int(record.layer),
                        target_register=int(record.head),
                        branch_id=0,
                        arity_slot=1,
                        primitive_id=int(op_id),
                        method="policy",
                        flow_steps=1,
                        source_mask=second_mask,
                        source_p0=second_p0,
                        return_details=True,
                        flow_time=1.0,
                    )
                    second_probs = second["probs"]
                    second_cache[cache_key] = (second_probs, second_mask)
                else:
                    second_probs, second_mask = cached
                for j in range(int(active.numel())):
                    if not bool(second_mask[j].item()):
                        continue
                    productions.append((int(op_id), (int(i), int(j))))
                    probs.append(op_dist["probs"][int(op_pos)] * first["probs"][int(i)] * second_probs[int(j)])
    if not probs:
        raise ValueError("production record has no valid productions")
    prod_probs = torch.stack(probs).flatten()
    prod_probs = prod_probs / prod_probs.sum().clamp_min(1e-12)
    eq_set = set((int(op_id), tuple(int(v) for v in sources)) for op_id, sources in record.equivalent_productions)
    equivalent_indices = [idx for idx, prod in enumerate(productions) if prod in eq_set]
    return prod_probs, equivalent_indices, productions


def _operator_family_id_for_op(op_id: int) -> int:
    if int(op_id) < 0:
        return int(OPERATOR_FAMILY_IDS["other"])
    name = get_op(int(op_id)).name
    if name in {"add", "sub", "mul", "neg", "square", "cube"}:
        return int(OPERATOR_FAMILY_IDS["poly"])
    if name in {"sin", "cos"}:
        return int(OPERATOR_FAMILY_IDS["trig"])
    if name in {"protected_log", "protected_sqrt"}:
        return int(OPERATOR_FAMILY_IDS["logroot"])
    if name == "protected_div":
        return int(OPERATOR_FAMILY_IDS["rational"])
    return int(OPERATOR_FAMILY_IDS["other"])


def _operator_distribution_for_closure_record(
    model: ConditionalEdgeFlowModel,
    record: OperatorClassificationRecord,
    *,
    device: torch.device,
    method: str,
    flow_steps: int,
) -> dict:
    regs = list(record.register_exprs)
    reg_sem = record.register_semantics.to(device=device, dtype=torch.float32)
    residual = record.residual_semantics.to(device=device, dtype=torch.float32)
    x_stub = _closure_x_stub(reg_sem, int(model.cfg.num_vars))
    tokens = model.register_tokens(x_stub, residual, regs, reg_sem, layer_id=int(record.layer))
    target_token = tokens[-1]
    return model.operator_probs(
        target_token=target_token,
        primitive_ids=list(record.candidate_operator_ids),
        layer_id=int(record.layer),
        target_register=int(record.head),
        branch_id=0,
        method=str(method),
        flow_steps=int(flow_steps),
        return_details=True,
        flow_time=1.0,
    )


def _source_distribution_for_closure_record(
    model: ConditionalEdgeFlowModel,
    record: SourceClassificationRecord,
    *,
    device: torch.device,
    method: str,
    flow_steps: int,
) -> dict:
    regs = list(record.register_exprs)
    reg_sem = record.register_semantics.to(device=device, dtype=torch.float32)
    residual = record.residual_semantics.to(device=device, dtype=torch.float32)
    x_stub = _closure_x_stub(reg_sem, int(model.cfg.num_vars))
    tokens = model.register_tokens(x_stub, residual, regs, reg_sem, layer_id=int(record.layer))
    return model.source_probs(
        target_token=tokens[-1],
        source_tokens=tokens,
        layer_id=int(record.layer),
        target_register=int(record.head),
        branch_id=0,
        arity_slot=int(record.slot),
        primitive_id=int(record.operator_id),
        method=str(method),
        flow_steps=int(flow_steps),
        source_mask=record.source_mask.to(device=device, dtype=torch.bool),
        source_p0=record.source_p0.to(device=device, dtype=torch.float32),
        candidate_semantics=reg_sem,
        target_semantics=residual,
        return_details=True,
        flow_time=1.0,
    )


def _closure_x_stub(register_semantics: torch.Tensor, num_vars: int) -> torch.Tensor:
    if int(register_semantics.shape[1]) >= int(num_vars):
        return register_semantics[:, : int(num_vars)].float()
    pad = torch.zeros(
        (int(register_semantics.shape[0]), int(num_vars) - int(register_semantics.shape[1])),
        dtype=register_semantics.dtype,
        device=register_semantics.device,
    )
    return torch.cat([register_semantics.float(), pad], dim=1)


def _operator_family_probs(probs: torch.Tensor, operator_ids: tuple[int, ...]) -> torch.Tensor:
    out = torch.zeros(len(OPERATOR_FAMILY_IDS), dtype=probs.dtype, device=probs.device)
    for idx, op_id in enumerate(operator_ids):
        name = get_op(int(op_id)).name
        if name in {"add", "sub", "mul", "neg", "square", "cube"}:
            family = "poly"
        elif name in {"sin", "cos"}:
            family = "trig"
        elif name in {"protected_log", "protected_sqrt"}:
            family = "logroot"
        elif name == "protected_div":
            family = "rational"
        else:
            family = "other"
        out[int(OPERATOR_FAMILY_IDS[family])] = out[int(OPERATOR_FAMILY_IDS[family])] + probs[int(idx)]
    return out / out.sum().clamp_min(1e-12)


def _structural_zero_metrics(
    bundle: StructuralClosureTargetBundle,
    *,
    skipped_op: int,
    skipped_src: int,
) -> dict:
    return {
        "loss": 0.0,
        "closure_operator_exact_loss": 0.0,
        "closure_operator_family_loss": 0.0,
        "closure_source_loss": 0.0,
        "closure_operator_top1": 0.0,
        "closure_operator_top3": 0.0,
        "closure_operator_family_top1": 0.0,
        "closure_source_top1": 0.0,
        "closure_source_top3": 0.0,
        "closure_source_rank": 0.0,
        "closure_operator_skipped": int(skipped_op),
        "closure_source_skipped": int(skipped_src),
        "closure_supported_supervised": int(bundle.supported),
    }


def _spff_sample_time(train_cfg: dict, rng: random.Random) -> float:
    mode = str(train_cfg.get("spff_time_sampling", train_cfg.get("teacher_time_sampling", "uniform"))).lower()
    if mode in {"uniform", "random", "u01"}:
        return float(rng.random())
    try:
        return float(max(0.0, min(1.0, float(mode))))
    except ValueError:
        return 1.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _synthetic_tasks(cfg: dict, rng: random.Random) -> list[ConditionalTrainTask]:
    gen_cfg = cfg.get("gen", {})
    template_cfg = cfg.get("template", {})
    tasks = []
    min_vars = int(gen_cfg.get("num_vars_min", gen_cfg.get("num_vars", 1)))
    max_vars = int(gen_cfg.get("num_vars_max", template_cfg.get("num_vars", gen_cfg.get("num_vars", 1))))
    max_vars = min(max_vars, int(template_cfg.get("num_vars", max_vars)))
    for idx in range(int(cfg.get("num_tasks", 2))):
        num_vars = rng.randint(max(min_vars, 1), max(max_vars, max(min_vars, 1)))
        gen = GenConfig(
            num_vars=num_vars,
            max_depth=int(gen_cfg.get("max_depth", 3)),
            K=int(template_cfg.get("num_registers", gen_cfg.get("K", 4))),
            probe_size=int(gen_cfg.get("probe_size", 32)),
            ops=tuple(template_cfg.get("primitives", gen_cfg.get("ops", ["add", "mul", "square"]))),
        )
        expr = generate_expression(gen, rng)
        x, y = sample_probe_xy(expr, gen, rng)
        gt = to_string(expr, gen.num_vars, simplify=True)
        tasks.append(ConditionalTrainTask(f"synthetic_{idx}:{gt}", x, y, num_vars, gt))
    return tasks


def _training_tasks(
    cfg: dict,
    template: RegisterOperatorTemplate,
    rng: random.Random,
) -> list[ConditionalTrainTask]:
    data_cfg = cfg.get("data", {})
    source = str(data_cfg.get("source", "synthetic")).strip().lower()
    if source in {"synthetic", "generated"}:
        return _synthetic_tasks(cfg, rng)
    if source in {"benchmark", "benchmark_87", "87task", "87_task", "benchmark_manifest", "materialized"}:
        tasks = load_edge_flow_benchmark_tasks(
            manifest=data_cfg.get("manifest", "data/benchmark_suites/benchmark_manifest.json"),
            suites=list(data_cfg.get("suites", ["nguyen", "constant", "livermore", "jin"])),
            root=data_cfg.get("manifest_root", "data/benchmark_suites"),
            seed=int(data_cfg.get("seed", cfg.get("seed", 0))),
            legacy_87=bool(data_cfg.get("legacy_87", True)),
            feynman_root=data_cfg.get("feynman_root", "data/materialized/feynman"),
            limit=data_cfg.get("limit_tasks"),
        )
        out: list[ConditionalTrainTask] = []
        for task in tasks:
            if int(task.X_train.shape[1]) > template.num_vars:
                continue
            x_train, y_train, _, _ = task_tensors(task, template_num_vars=template.num_vars)
            x_train, y_train = _limit_task_points(
                x_train,
                y_train,
                max_points=data_cfg.get("max_train_points"),
                rng=rng,
            )
            out.append(ConditionalTrainTask(
                task.name,
                x_train,
                y_train,
                int(task.X_train.shape[1]),
                task.expression or str(task.metadata.get("ground_truth", "") or ""),
            ))
        if not out:
            raise ValueError("benchmark data source produced no compatible tasks")
        return out
    if source in {"symbolicgpt_subset", "symbolicgpt", "symbolicgpt_local"}:
        tasks = load_symbolicgpt_subset_tasks(
            data_cfg.get("root", "data/generated/symbolicgpt_subset"),
            splits=tuple(data_cfg.get("splits", ["train"])),
            limit=data_cfg.get("limit_tasks"),
            rng=rng,
            train_fraction=float(data_cfg.get("train_fraction", 0.8)),
        )
        out: list[ConditionalTrainTask] = []
        for task in tasks:
            if int(task.X_train.shape[1]) > template.num_vars:
                continue
            x_train, y_train, _, _ = task_tensors(task, template_num_vars=template.num_vars)
            x_train, y_train = _limit_task_points(
                x_train,
                y_train,
                max_points=data_cfg.get("max_train_points"),
                rng=rng,
            )
            out.append(ConditionalTrainTask(
                task.name,
                x_train,
                y_train,
                int(task.X_train.shape[1]),
                task.expression or "",
            ))
        if not out:
            raise ValueError("SymbolicGPT subset source produced no compatible tasks")
        return out
    if source in {"mixed", "benchmark_plus_synthetic", "materialized_plus_synthetic"}:
        bench_cfg = dict(data_cfg)
        bench_cfg["source"] = "benchmark_manifest"
        synthetic_count = int(data_cfg.get("synthetic_tasks", cfg.get("num_tasks", 0)))
        bench_tasks = _training_tasks({**cfg, "data": bench_cfg}, template, rng)
        syn_tasks = _synthetic_tasks({**cfg, "num_tasks": synthetic_count}, rng) if synthetic_count > 0 else []
        return bench_tasks + syn_tasks
    raise ValueError(f"unknown edge-flow training data source: {source}")


def _task_batches(tasks: list[ConditionalTrainTask], batch_size: int) -> list[list[ConditionalTrainTask]]:
    return [tasks[idx:idx + int(batch_size)] for idx in range(0, len(tasks), int(batch_size))]


def _fixed_pool_task_seed(seed: int, task_index: int) -> int:
    return int(seed) + 1_000_003 * int(task_index)


def _limit_task_points(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    max_points,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_points is None:
        return x, y
    n = int(x.shape[0])
    m = max(int(max_points), 1)
    if n <= m:
        return x, y
    indices = torch.tensor(rng.sample(range(n), m), dtype=torch.long)
    return x[indices], y[indices]


def _sample_expression_key(sample: CircuitSample) -> str:
    return str(sample.canonical or sample.expression)


def _loss_on_training_device(loss: torch.Tensor, device: torch.device) -> torch.Tensor:
    if loss.device == device:
        return loss
    return loss.to(device=device)


def _structure_scores(
    samples: list[CircuitSample],
    *,
    ground_truth: str,
    num_vars: int,
) -> torch.Tensor:
    text = str(ground_truth or "").strip()
    if not samples:
        return torch.zeros(0, dtype=torch.float32)
    if not text:
        return torch.zeros(len(samples), dtype=torch.float32)
    values = []
    for sample in samples:
        try:
            expr_text = to_string(sample.expression, int(num_vars), simplify=True)
        except Exception:
            expr_text = str(sample.expression)
        score = structure_similarity_score(expr_text, text)
        if sample.diagnostics is None:
            sample.diagnostics = {}
        sample.diagnostics["structure_score"] = float(score)
        values.append(float(score))
    return torch.tensor(values, dtype=torch.float32)


def _posterior_weights(rewards, structure_scores: torch.Tensor, *, train_cfg: dict) -> tuple[torch.Tensor, dict]:
    if rewards.rewards.numel() == 0:
        return torch.zeros(0, dtype=torch.float32), {
            "posterior_weight_entropy": 0.0,
            "posterior_weight_max": 0.0,
        }
    device = rewards.rewards.device
    structure = structure_scores.to(device=device, dtype=rewards.rewards.dtype)
    if structure.numel() != rewards.rewards.numel():
        structure = torch.zeros_like(rewards.rewards)
    log_weights = structure_conditioned_log_weight(
        r2=rewards.r2,
        complexity=rewards.complexity,
        structure_score=structure,
        beta_y=float(train_cfg.get("semantic_beta", train_cfg.get("beta_y", 1.0))),
        beta_g=float(train_cfg.get("structure_beta", train_cfg.get("beta_g", 0.0))),
        beta_c=float(train_cfg.get("posterior_complexity_beta", train_cfg.get("beta_c", 0.0))),
    )
    log_weights = torch.where(
        rewards.valid_mask,
        log_weights,
        torch.full_like(log_weights, -1.0e9),
    )
    weights = normalize_log_weights(log_weights).to(device)
    entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum()
    return weights, {
        "posterior_weight_entropy": float(entropy.detach().cpu().item()),
        "posterior_weight_max": float(weights.max().detach().cpu().item()) if weights.numel() else 0.0,
    }


def _gt_training_diagnostics(
    samples: list[CircuitSample],
    rewards: torch.Tensor,
    posterior_weights: torch.Tensor,
) -> dict:
    gt_indices = [
        idx for idx, sample in enumerate(samples)
        if bool((sample.diagnostics or {}).get("is_gt_elite", False))
    ]
    if not gt_indices:
        return {
            "gt_decision_top1_rate": 0.0,
            "gt_decision_top3_rate": 0.0,
            "gt_reward_rank": 0.0,
            "gt_posterior_weight": 0.0,
        }
    gt_idx = int(gt_indices[0])
    gt_sample = samples[gt_idx]
    traces = [trace for trace in getattr(gt_sample, "decision_traces", ()) if bool(trace.active)]
    ranks = [decision_trace_rank(trace) for trace in traces]
    top1 = sum(1 for rank in ranks if rank == 1) / max(len(ranks), 1)
    top3 = sum(1 for rank in ranks if 1 <= rank <= 3) / max(len(ranks), 1)
    if rewards.numel() and gt_idx < int(rewards.numel()):
        gt_reward = rewards[gt_idx]
        rank = int((rewards > gt_reward).sum().item()) + 1
    else:
        rank = 0
    posterior = (
        float(posterior_weights[gt_idx].detach().cpu().item())
        if posterior_weights.numel() and gt_idx < int(posterior_weights.numel())
        else 0.0
    )
    return {
        "gt_decision_top1_rate": float(top1),
        "gt_decision_top3_rate": float(top3),
        "gt_reward_rank": float(rank),
        "gt_posterior_weight": posterior,
    }


def _global_projection_gt_samples(
    task: ConditionalTrainTask,
    *,
    template: RegisterOperatorTemplate,
    model: ConditionalEdgeFlowModel,
    x: torch.Tensor,
    y: torch.Tensor,
    method: str,
    flow_steps: int,
    train_cfg: dict,
    rng: random.Random,
) -> list[CircuitSample]:
    del rng
    text = str(task.ground_truth or "").strip()
    if not text:
        return []
    compiled = compile_formula_to_spff_sample(
        text,
        variable_count=int(task.num_vars),
        template=template,
        model=model,
        x=x,
        y=y,
        method=str(method),
        flow_steps=int(flow_steps),
        flow_time=float(train_cfg.get("gt_flow_time", 1.0)),
    )
    normalized_endpoint = False
    if compiled is None:
        try:
            terms = decompose_formula_terms(text, [f"x{i}" for i in range(int(task.num_vars))])
        except Exception:
            terms = []
        nonconstant_terms = [term for term in terms if abs(float(getattr(term, "constant", 0.0))) == 0.0]
        if len(nonconstant_terms) == 1:
            term_expr = nonconstant_terms[0].expr
            try:
                term_y = torch.nan_to_num(
                    eval_expr(term_expr, x.float()),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).detach().float()
            except Exception:
                term_y = y
            compiled = compile_expr_to_spff_sample(
                term_expr,
                variable_count=int(task.num_vars),
                template=template,
                model=model,
                x=x,
                y=term_y,
                method=str(method),
                flow_steps=int(flow_steps),
                flow_time=float(train_cfg.get("gt_flow_time", 1.0)),
            )
            normalized_endpoint = compiled is not None
    if compiled is None or compiled.log_prob_tensor is None:
        return []
    diag = dict(compiled.diagnostics or {})
    diag.update({
        "is_gt_equivalence_endpoint": True,
        "gt_equivalence_canonical": True,
        "gt_equivalence_scale_normalized": bool(normalized_endpoint),
        "global_projection_gt_injected": True,
    })
    compiled.diagnostics = diag
    return [compiled]


def _sample_semantic_fm_time(train_cfg: dict, rng: random.Random) -> float:
    low = float(train_cfg.get("flow_time_min", train_cfg.get("t_min", 0.05)))
    high = float(train_cfg.get("flow_time_max", train_cfg.get("t_max", 0.95)))
    low = max(0.0, min(float(low), 1.0))
    high = max(0.0, min(float(high), 1.0))
    if high < low:
        low, high = high, low
    if abs(high - low) <= 1e-12:
        return float(low)
    return float(low + (high - low) * rng.random())


def _explicit_theta_proxy_samples(
    task: ConditionalTrainTask,
    *,
    template: RegisterOperatorTemplate,
    edge_distribution,
    x: torch.Tensor,
    y: torch.Tensor,
    train_cfg: dict,
    rng: random.Random,
    start_id: int,
) -> list[CircuitSample]:
    count = int(train_cfg.get("proxy_proposal_count", 0))
    if count <= 0:
        return []
    proposals = simple_gp_proposals(
        x,
        y,
        num_vars=int(task.num_vars),
        primitives=tuple(template.primitives),
        rng=rng,
        proposal_count=int(count),
        population_size=int(train_cfg.get("proxy_population_size", max(count * 4, 16))),
        generations=int(train_cfg.get("proxy_generations", 2)),
        max_depth=int(train_cfg.get("proxy_max_depth", 3)),
    )
    out: list[CircuitSample] = []
    seen: set[str] = set()
    for proposal in proposals:
        expr = proposal.expression
        if expr is None:
            try:
                expr = parse_formula(str(proposal.formula), [f"x{i}" for i in range(int(task.num_vars))])
            except Exception:
                continue
        key = str(expr)
        if key in seen:
            continue
        seen.add(key)
        compiled = compile_expr_to_edge_sample(
            expr,
            template=template,
            edge_distribution=edge_distribution,
            sample_id=int(start_id) + len(out),
        )
        if compiled is None:
            continue
        diag = dict(compiled.diagnostics or {})
        diag.update({
            "candidate_pool_role": "proxy_negative_candidate",
            "proxy_source": str(proposal.source),
            "proxy_formula": str(proposal.formula),
        })
        compiled.diagnostics = diag
        out.append(compiled)
    return out


def _build_fixed_gt_proxy_pool(
    task: ConditionalTrainTask,
    *,
    template: RegisterOperatorTemplate,
    train_cfg: dict,
    rng: random.Random,
    device: torch.device,
) -> dict | None:
    x_dev = task.x.float().to(device)
    y_dev = task.y.float().to(device)
    try:
        gt_expr = parse_formula(str(task.ground_truth or ""), [f"x{i}" for i in range(int(task.num_vars))])
    except Exception:
        return None
    theta0 = torch.zeros(_template_theta_dim(template), device=device)
    edge_distribution = theta_vector_to_distribution(theta0, template)
    sampler = CircuitSampler(template)
    samples_per_task = int(train_cfg.get("samples_per_task", train_cfg.get("sample_pool_size", 32)))
    model_samples = sampler.sample(edge_distribution, batch_size=samples_per_task, rng=rng)
    gt_samples = _explicit_theta_gt_samples(
        task,
        gt_expr,
        template=template,
        edge_distribution=edge_distribution,
        train_cfg=train_cfg,
        start_id=len(model_samples),
    )
    proxy_samples = _explicit_theta_proxy_samples(
        task,
        template=template,
        edge_distribution=edge_distribution,
        x=x_dev,
        y=y_dev,
        train_cfg=train_cfg,
        rng=rng,
        start_id=len(model_samples) + len(gt_samples),
    )
    base_pool = list(model_samples) + list(gt_samples) + list(proxy_samples)
    target_weights, proxy_mask, proxy_diag = classify_gt_proxy_samples(
        base_pool,
        gt_expr,
        x_dev,
        y_dev,
        num_vars=int(task.num_vars),
        proxy_r2_threshold=float(train_cfg.get("proxy_r2_threshold", 0.85)),
        device=device,
    )
    return {
        "task": task,
        "x": x_dev,
        "y": y_dev,
        "gt_expr": gt_expr,
        "base_pool": base_pool,
        "target_weights": target_weights.detach(),
        "proxy_mask": proxy_mask.detach(),
        "proxy_diag": dict(proxy_diag),
        "gt_injected_count": int(len(gt_samples)),
        "gt_rewrite_injected_count": int(sum(
            1 for sample in gt_samples
            if bool((sample.diagnostics or {}).get("explicit_theta_gt_rewrite"))
        )),
        "proxy_sample_count": int(len(proxy_samples)),
        "num_sampled_expressions": int(len(model_samples)),
    }


def _template_theta_dim(template: RegisterOperatorTemplate) -> int:
    total = int(template.mixture_modes)
    for group in template.groups:
        total += int(template.mixture_modes) * int(group.num_candidates)
    return int(total)


def _relogprob_edge_pool(
    samples: list[CircuitSample],
    *,
    template: RegisterOperatorTemplate,
    edge_distribution,
) -> list[CircuitSample]:
    out: list[CircuitSample] = []
    for sample in samples:
        log_prob_tensor = _edge_choices_logprob_tensor(
            sample.edge_choices,
            mode=int(sample.mode),
            template=template,
            edge_distribution=edge_distribution,
        )
        log_value = float(log_prob_tensor.detach().cpu().item()) if log_prob_tensor is not None else float(sample.log_prob)
        diag = dict(sample.diagnostics or {})
        diag.setdefault("decision_count", int(len(sample.edge_choices) + 1))
        diag.setdefault("active_decision_count", int(len(sample.edge_choices) + 1))
        out.append(CircuitSample(
            sample_id=int(sample.sample_id),
            mode=int(sample.mode),
            edge_choices=dict(sample.edge_choices),
            expression=sample.expression,
            log_prob=log_value,
            complexity=int(sample.complexity),
            canonical=sample.canonical,
            head_terms=tuple(sample.head_terms),
            log_prob_tensor=log_prob_tensor,
            active_log_prob_tensor=log_prob_tensor,
            entropy_tensor=sample.entropy_tensor,
            decision_traces=tuple(sample.decision_traces),
            semantic_teacher_loss_tensor=sample.semantic_teacher_loss_tensor,
            diagnostics=diag,
        ))
    return out


def _pool_logprob_scores(samples: list[CircuitSample], *, device: torch.device) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    for sample in samples:
        tensor = sample.active_log_prob_tensor if sample.active_log_prob_tensor is not None else sample.log_prob_tensor
        if tensor is None:
            tensor = torch.tensor(float(sample.log_prob), dtype=torch.float32, device=device)
        scores.append(tensor.to(device=device, dtype=torch.float32).reshape(()))
    if not scores:
        return torch.zeros(0, dtype=torch.float32, device=device)
    return torch.stack(scores)


def _edge_choices_logprob_tensor(
    choices: dict[str, int],
    *,
    mode: int,
    template: RegisterOperatorTemplate,
    edge_distribution,
) -> torch.Tensor | None:
    if not choices:
        return None
    terms: list[torch.Tensor] = [
        edge_distribution.mixture_probs[int(mode)].clamp_min(1e-12).log()
    ]
    for group in template.groups:
        if group.group_id not in choices:
            return None
        choice = int(choices[group.group_id])
        if choice < 0 or choice >= int(group.num_candidates):
            return None
        terms.append(edge_distribution.group_probs[group.group_id][int(mode), choice].clamp_min(1e-12).log())
    return torch.stack(terms).sum()


def _explicit_theta_gt_samples(
    task: ConditionalTrainTask,
    gt_expr: Expr,
    *,
    template: RegisterOperatorTemplate,
    edge_distribution,
    train_cfg: dict,
    start_id: int,
) -> list[CircuitSample]:
    max_canonical = max(int(train_cfg.get("gt_injection_count", 1)), 0)
    rewrite_count = max(int(train_cfg.get("gt_rewrite_count", 0)), 0)
    out: list[CircuitSample] = []
    seen: set[str] = set()

    def add_candidate(expr: Expr, *, role: str, scale_normalized: bool = False) -> None:
        if len(out) >= max_canonical + rewrite_count:
            return
        key = str(expr)
        if key in seen:
            return
        seen.add(key)
        compiled = compile_expr_to_edge_sample(
            expr,
            template=template,
            edge_distribution=edge_distribution,
            sample_id=int(start_id) + len(out),
        )
        if compiled is None or compiled.log_prob_tensor is None:
            return
        diag = dict(compiled.diagnostics or {})
        diag.update({
            "explicit_theta_gt_injected": True,
            "is_gt_equivalence_endpoint": True,
            "gt_equivalence_scale_normalized": bool(scale_normalized),
        })
        if role == "canonical":
            diag["explicit_theta_gt_canonical"] = True
        elif role == "normalized":
            diag["explicit_theta_gt_normalized"] = True
        else:
            diag["explicit_theta_gt_rewrite"] = True
        compiled.diagnostics = diag
        out.append(compiled)

    if max_canonical > 0:
        add_candidate(gt_expr, role="canonical")
        normalized = _single_term_gt_expr(task)
        if normalized is not None:
            add_candidate(normalized, role="normalized", scale_normalized=True)

    if rewrite_count > 0:
        rewrite_candidates = build_gt_rewrite_candidates(
            gt_expr,
            primitive_names=tuple(template.primitives),
            max_count=rewrite_count * 4,
            num_vars=int(task.num_vars),
        )
        for candidate in rewrite_candidates:
            before = len(out)
            add_candidate(candidate, role="rewrite")
            rewrite_injected = sum(
                1 for sample in out
                if bool((sample.diagnostics or {}).get("explicit_theta_gt_rewrite"))
            )
            if len(out) > before and rewrite_injected >= rewrite_count:
                break
    return out


def _trainable_gt_indices(samples: list[CircuitSample], target_weights: torch.Tensor) -> list[int]:
    target = torch.as_tensor(target_weights).detach().flatten()
    out: list[int] = []
    for idx, sample in enumerate(samples):
        if idx >= int(target.numel()) or float(target[int(idx)].item()) <= 0.0:
            continue
        tensor = sample.active_log_prob_tensor if sample.active_log_prob_tensor is not None else sample.log_prob_tensor
        if tensor is not None:
            out.append(int(idx))
    return out


def _sample_logprob_grad_norm(
    samples: list[CircuitSample],
    indices: list[int],
    parameters: tuple[torch.nn.Parameter, ...],
) -> float:
    tensors: list[torch.Tensor] = []
    for idx in indices:
        sample = samples[int(idx)]
        tensor = sample.active_log_prob_tensor if sample.active_log_prob_tensor is not None else sample.log_prob_tensor
        if tensor is not None and bool(tensor.requires_grad):
            tensors.append(tensor.reshape(()))
    if not tensors:
        return 0.0
    params = tuple(param for param in parameters if param.requires_grad)
    if not params:
        return 0.0
    grads = torch.autograd.grad(
        torch.stack(tensors).sum(),
        params,
        retain_graph=True,
        allow_unused=True,
    )
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().pow(2).sum().cpu().item())
    return float(total ** 0.5)


def _single_term_gt_expr(task: ConditionalTrainTask) -> Expr | None:
    text = str(task.ground_truth or "").strip()
    if not text:
        return None
    try:
        terms = decompose_formula_terms(text, [f"x{i}" for i in range(int(task.num_vars))])
    except Exception:
        return None
    nonconstant_terms = [term for term in terms if abs(float(getattr(term, "constant", 0.0))) == 0.0]
    if len(nonconstant_terms) != 1:
        return None
    return nonconstant_terms[0].expr


def _sample_current_logprob(sample: CircuitSample, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = sample.active_log_prob_tensor if sample.active_log_prob_tensor is not None else sample.log_prob_tensor
    if tensor is None:
        return torch.tensor(float(sample.log_prob), device=device, dtype=dtype)
    return tensor.detach().to(device=device, dtype=dtype).reshape(())


def _semantic_fm_path_log_rate(
    sample: CircuitSample,
    task: ConditionalTrainTask,
    *,
    template: RegisterOperatorTemplate,
    model: ConditionalEdgeFlowModel,
    x: torch.Tensor,
    y: torch.Tensor,
    method: str,
    flow_steps: int,
    t_value: float,
    eps: float,
) -> torch.Tensor | None:
    delta = max(float(eps), 1e-4)
    t_minus = max(0.0, float(t_value) - delta)
    t_plus = min(1.0, float(t_value) + delta)
    if t_plus <= t_minus:
        return None
    plus = _recompute_semantic_fm_sample(
        sample,
        task,
        template=template,
        model=model,
        x=x,
        y=y,
        method=method,
        flow_steps=flow_steps,
        flow_time=float(t_plus),
    )
    minus = _recompute_semantic_fm_sample(
        sample,
        task,
        template=template,
        model=model,
        x=x,
        y=y,
        method=method,
        flow_steps=flow_steps,
        flow_time=float(t_minus),
    )
    if plus is None or minus is None:
        return None
    plus_log = plus.active_log_prob_tensor if plus.active_log_prob_tensor is not None else plus.log_prob_tensor
    minus_log = minus.active_log_prob_tensor if minus.active_log_prob_tensor is not None else minus.log_prob_tensor
    if plus_log is None or minus_log is None:
        return None
    return (plus_log - minus_log) / float(t_plus - t_minus)


def _recompute_semantic_fm_sample(
    sample: CircuitSample,
    task: ConditionalTrainTask,
    *,
    template: RegisterOperatorTemplate,
    model: ConditionalEdgeFlowModel,
    x: torch.Tensor,
    y: torch.Tensor,
    method: str,
    flow_steps: int,
    flow_time: float,
) -> CircuitSample | None:
    diag = dict(sample.diagnostics or {})
    if bool(diag.get("global_projection_gt_injected")) or bool(diag.get("is_gt_equivalence_endpoint")):
        compiled = _global_projection_gt_samples(
            task,
            template=template,
            model=model,
            x=x,
            y=y,
            method=method,
            flow_steps=flow_steps,
            train_cfg={"gt_flow_time": float(flow_time)},
            rng=random.Random(0),
        )
        return compiled[0] if compiled else None
    if any(str(key).startswith("H") for key in (sample.edge_choices or {})):
        return replay_term_graph_sample(
            sample,
            x,
            y,
            template=template,
            model=model,
            method=method,
            flow_steps=flow_steps,
            flow_time=float(flow_time),
            active_variable_count=int(task.num_vars),
        )
    compiled = compile_expr_to_spff_sample(
        sample.expression,
        variable_count=int(task.num_vars),
        template=template,
        model=model,
        x=x,
        y=y,
        method=str(method),
        flow_steps=int(flow_steps),
        flow_time=float(flow_time),
    )
    return compiled


def _ground_truth_samples(
    task: ConditionalTrainTask,
    *,
    template: RegisterOperatorTemplate,
    model: ConditionalEdgeFlowModel,
    x: torch.Tensor,
    y: torch.Tensor,
    method: str,
    flow_steps: int,
    device: torch.device,
    train_cfg: dict,
    rng: random.Random,
) -> list[CircuitSample]:
    text = str(task.ground_truth or "").strip()
    if not text:
        return []
    target_shape_source = str(train_cfg.get("target_shape_source", "gt_neighborhood")).strip().lower()
    if target_shape_source in {
        "gt_neighborhood",
        "gt-neighborhood",
        "neighborhood",
        "gt_sampler",
        "structural_denoising",
        "structure_denoising",
        "clean_gt",
        "clean_gt_one_hot",
        "gt_denoising",
        "denoising",
    }:
        result = build_gt_neighborhood_samples(
            text,
            variable_count=int(task.num_vars),
            template=template,
            model=model,
            x=x,
            y=y,
            method=str(method),
            flow_steps=int(flow_steps),
            flow_time=_teacher_flow_time(train_cfg, rng),
            rng=rng,
            size=int(train_cfg.get("gt_neighborhood_size", train_cfg.get("gt_sampler_size", 8))),
            op_replace_prob=float(train_cfg.get("gt_neighborhood_op_replace_prob", 0.3)),
            source_replace_prob=float(train_cfg.get("gt_neighborhood_source_replace_prob", 0.3)),
        )
        for sample in result.samples:
            diag = dict(sample.diagnostics or {})
            diag.update(result.diagnostics)
            sample.diagnostics = diag
        return result.samples
    compiled = compile_formula_to_spff_sample(
        text,
        variable_count=int(task.num_vars),
        template=template,
        model=model,
        x=x,
        y=y,
        method=str(method),
        flow_steps=int(flow_steps),
        flow_time=_teacher_flow_time(train_cfg, rng),
    )
    if compiled is not None:
        return [compiled]
    try:
        expr = parse_formula(text, [f"x{i}" for i in range(int(task.num_vars))])
    except Exception:
        return []
    return [CircuitSample(
        sample_id=-1,
        mode=0,
        edge_choices={"GT": 0},
        expression=expr,
        log_prob=0.0,
        complexity=int(expr.complexity),
        head_terms=(expr,),
        log_prob_tensor=None,
        entropy_tensor=torch.zeros((), device=device),
        diagnostics={"decision_count": 0, "is_gt_elite": True, "active_variable_count": int(task.num_vars)},
    )]


def _teacher_target_mode(train_cfg: dict) -> str:
    target_shape_source = str(train_cfg.get("target_shape_source", "")).strip().lower().replace("-", "_")
    explicit = str(train_cfg.get("teacher_target_mode", "")).strip().lower().replace("-", "_")
    key = explicit or target_shape_source
    if key in {
        "structural_denoising",
        "structure_denoising",
        "clean_gt",
        "clean_gt_one_hot",
        "gt_denoising",
        "denoising",
    }:
        return "structural_denoising"
    return "posterior"


def _proposal_samples(
    task: ConditionalTrainTask,
    *,
    template: RegisterOperatorTemplate,
    model: ConditionalEdgeFlowModel,
    x: torch.Tensor,
    y: torch.Tensor,
    method: str,
    flow_steps: int,
    sources: list[str],
    count: int,
    compile_only_trainable: bool,
    rng: random.Random,
    train_cfg: dict,
) -> list[CircuitSample]:
    if not sources or int(count) <= 0:
        return []
    proposals = []
    source_set = {str(item).lower() for item in sources}
    if "gp" in source_set or "simple_gp" in source_set:
        proposals.extend(simple_gp_proposals(
            x,
            y,
            num_vars=int(task.num_vars),
            primitives=tuple(template.primitives),
            rng=rng,
            proposal_count=int(count),
            population_size=int(train_cfg.get("gp_population_size", max(16, int(count) * 4))),
            generations=int(train_cfg.get("gp_generations", 2)),
            max_depth=int(train_cfg.get("gp_max_depth", template.num_layers + 1)),
        ))
    if "diffusion" in source_set:
        proposals.extend(load_diffusion_formula_proposals(
            train_cfg.get("diffusion_proposal_path", ""),
            task_id=str(task.task_id),
            limit=int(count),
        ))
    out: list[CircuitSample] = []
    for proposal in proposals[:int(count)]:
        compiled = compile_formula_to_spff_sample(
            proposal.formula,
            variable_count=int(task.num_vars),
            template=template,
            model=model,
            x=x,
            y=y,
            method=str(method),
            flow_steps=int(flow_steps),
            flow_time=_teacher_flow_time(train_cfg, rng),
        )
        if compiled is not None:
            diag = dict(compiled.diagnostics or {})
            diag.update({
                "is_proposal_elite": True,
                "proposal_source": str(proposal.source),
                "proposal_compile_success": True,
            })
            compiled.diagnostics = diag
            out.append(compiled)
        elif not bool(compile_only_trainable) and proposal.expression is not None:
            out.append(CircuitSample(
                sample_id=-2,
                mode=0,
                edge_choices={"PROPOSAL": 0},
                expression=proposal.expression,
                log_prob=0.0,
                complexity=int(proposal.expression.complexity),
                head_terms=(proposal.expression,),
                log_prob_tensor=None,
                diagnostics={
                    "is_proposal_elite": True,
                    "proposal_source": str(proposal.source),
                    "proposal_compile_success": False,
                },
            ))
    return out


def _early_stopping_tasks(
    cfg: dict,
    template: RegisterOperatorTemplate,
    rng: random.Random,
) -> list[ConditionalTrainTask]:
    data_cfg = dict(cfg.get("data", {}))
    if str(data_cfg.get("source", "")).strip().lower() not in {"symbolicgpt_subset", "symbolicgpt", "symbolicgpt_local"}:
        return []
    data_cfg["splits"] = list(data_cfg.get("validation_splits", ["val"]))
    if data_cfg.get("early_stopping_limit_tasks") is not None:
        data_cfg["limit_tasks"] = data_cfg.get("early_stopping_limit_tasks")
    return _training_tasks({**cfg, "data": data_cfg}, template, rng)


def _conditional_validation_score(
    model: ConditionalEdgeFlowModel,
    template: RegisterOperatorTemplate,
    tasks: list[ConditionalTrainTask],
    *,
    rng: random.Random,
    samples: int,
    complexity_weight: float,
    method: str,
    flow_steps: int,
    head_fit_mode: str,
) -> float:
    if not tasks:
        return float("-inf")
    scores: list[float] = []
    sampler = ConditionalEdgeFlowSampler(template, model, method=method, flow_steps=flow_steps)
    with torch.no_grad():
        device = next(model.parameters()).device
        for task in tasks:
            x = task.x.float().to(device)
            y = task.y.float().to(device)
            sampled = sampler.sample(
                x,
                y,
                batch_size=int(samples),
                rng=rng,
                active_variable_count=int(task.num_vars),
            )
            rewards, _ = _target_rewards(sampled, x, y, EdgeFlowBuildConfig(
                samples_per_task=int(samples),
                elite_k=1,
                complexity_weight=float(complexity_weight),
                validation_fraction=0.0,
                head_fit_mode=str(head_fit_mode),
            ))
            if rewards.rewards.numel():
                scores.append(float(rewards.rewards.max().item()))
    return float(sum(scores) / max(len(scores), 1))


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _teacher_flow_time(train_cfg: dict, rng: random.Random) -> float | None:
    mode = train_cfg.get("teacher_time_sampling")
    if mode is None:
        return None
    if isinstance(mode, (float, int)):
        return float(max(0.0, min(1.0, float(mode))))
    text = str(mode).strip().lower()
    if text in {"", "none", "endpoint", "final", "one", "1", "deterministic"}:
        return None
    if text in {"uniform", "random", "u01"}:
        return float(max(0.0, min(1.0, rng.random())))
    try:
        return float(max(0.0, min(1.0, float(text))))
    except ValueError as exc:
        raise ValueError(f"unknown teacher_time_sampling mode: {mode}") from exc


def _optional_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "false", "off"}:
        return None
    return float(value)


def _write_curve(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_curve_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _tensor_tree_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _tensor_tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tensor_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tensor_tree_to_cpu(item) for item in value)
    return value


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _configure_threads(runtime: dict) -> None:
    if runtime.get("torch_num_threads") is not None:
        torch.set_num_threads(max(int(runtime["torch_num_threads"]), 1))
    if runtime.get("torch_num_interop_threads") is not None:
        try:
            torch.set_num_interop_threads(max(int(runtime["torch_num_interop_threads"]), 1))
        except RuntimeError:
            pass


def _resolve_device(runtime: dict | None = None) -> torch.device:
    runtime = runtime or {}
    requested = str(runtime.get("device", "auto")).strip().lower()
    if requested in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("runtime.device requested cuda, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"runtime.device requested {requested}, but CUDA is unavailable")
        index_text = requested.split(":", 1)[1]
        if not index_text.isdigit():
            raise ValueError(f"invalid CUDA device specifier: {requested}")
        index = int(index_text)
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"requested {requested}, but only {torch.cuda.device_count()} CUDA devices are visible")
        return torch.device(requested)
    raise ValueError(f"unsupported runtime.device: {requested}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
