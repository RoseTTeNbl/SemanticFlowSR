"""Train Edge-Parameterized Semantic Flow on synthetic tasks."""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from ..data.symbolicgpt_subset import load_symbolicgpt_subset_tasks
from ..data.synthetic_generator import GenConfig, generate_expression, sample_probe_xy
from ..sr.ast import Expr
from ..sr.parser import parse_formula
from ..sr.printer import to_string
from .benchmark import load_edge_flow_benchmark_tasks, task_tensors
from .conditional import (
    ConditionalEdgeFlowConfig,
    ConditionalEdgeFlowModel,
    ConditionalEdgeFlowSampler,
    conditional_elite_policy_loss,
)
from .circuit_sampler import CircuitSample
from .dataset import EdgeFlowBuildConfig, _target_rewards, build_edge_flow_records
from .gt_neighborhood import build_gt_neighborhood_samples
from .model import EdgeFlowModel, EdgeFlowModelConfig, edge_flow_loss
from .path_compiler import compile_formula_to_csef_sample
from .proposals import load_diffusion_formula_proposals, simple_gp_proposals
from .reward import RewardConfig, evaluate_expression_rewards
from .semantic_teacher import decision_trace_rank, semantic_teacher_loss_for_samples
from .structure_posterior import (
    normalize_log_weights,
    structure_conditioned_log_weight,
    structure_similarity_score,
)
from .template import RegisterOperatorTemplate


@dataclass
class ConditionalTrainTask:
    task_id: str
    x: torch.Tensor
    y: torch.Tensor
    num_vars: int
    ground_truth: str = ""


def run(cfg: dict) -> Path:
    algorithm = str(cfg.get("algorithm", "conditional_semantic_edge_flow")).lower()
    if algorithm in {"fixed_theta_edge_flow", "edge_parameterized_semantic_flow_matching"}:
        return _run_fixed_theta(cfg)
    if algorithm in {"conditional_semantic_edge_flow", "csef"}:
        return _run_conditional(cfg)
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
    ckpt_name = str(cfg.get("checkpoint_name", "conditional_edge_flow_smoke.pt"))
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
    ckpt_name = str(cfg.get("checkpoint_name", "edge_flow_smoke.pt"))
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
    compiled = compile_formula_to_csef_sample(
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
        compiled = compile_formula_to_csef_sample(
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
