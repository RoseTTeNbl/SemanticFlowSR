"""Train Semantic-Fisher Flow Matching action-support models."""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..data.collate import collate_velocity
from ..data.synthetic_generator import GenConfig
from ..models.semantic_transformer import SemanticTransformer, SemanticTransformerConfig
from ..semantics.energy import ActionEnergyConfig
from ..train.losses import SemanticFisherVelocityLoss
from ..utils.checkpoint import save_checkpoint
from ..utils.seed import get_device, set_seed
from ..path_posterior.dataset import (
    PathPosteriorBuildConfig,
    build_path_posterior_dataset,
    snapshot_behavior_model,
)


def run(cfg: dict):
    _configure_torch_threads(cfg.get("runtime", {}))
    set_seed(cfg.get("seed", 0))
    device = get_device()
    g = cfg["gen"]
    gen = GenConfig(
        num_vars=int(g["num_vars"]),
        max_depth=int(g["max_depth"]),
        K=int(g["K"]),
        probe_size=int(g["probe_size"]),
        ops=tuple(g["ops"]),
    )
    model_cfg_raw = cfg.get("model", {})
    model_cfg = SemanticTransformerConfig(
        d=gen.num_vars,
        K=gen.K,
        hidden=int(model_cfg_raw.get("hidden", 96)),
        row_layers=int(model_cfg_raw.get("row_layers", 1)),
        heads=int(model_cfg_raw.get("heads", 3)),
        output_mode="semantic_fisher_lograte",
    )
    model = SemanticTransformer(model_cfg).to(device)
    pp_cfg = cfg.get("path_posterior", {})
    energy_cfg = ActionEnergyConfig(**cfg.get("energy", {"lambda_op": 0.0}))
    build_cfg = PathPosteriorBuildConfig(
        target_mode=str(pp_cfg.get("target_mode", "future_group_l3")),
        num_trajectories=int(pp_cfg.get("num_trajectories", 16)),
        max_states_per_task=(
            None
            if pp_cfg.get("max_states_per_task") is None
            else int(pp_cfg.get("max_states_per_task"))
        ),
        max_steps=int(pp_cfg.get("max_steps", 6)),
        weight_eta=float(pp_cfg.get("weight_eta", 2.0)),
        target_smoothing=float(pp_cfg.get("target_smoothing", 1e-3)),
        score_to_shape=str(pp_cfg.get("score_to_shape", "rank_softmax")),
        advantage_eps=float(pp_cfg.get("advantage_eps", 1e-6)),
        advantage_clip=(
            None
            if pp_cfg.get("advantage_clip", 5.0) is None
            else float(pp_cfg.get("advantage_clip", 5.0))
        ),
        teacher_mode=str(pp_cfg.get("teacher_mode", "endpoint_matching")),
        p_init_mode=str(pp_cfg.get("p_init_mode", "stop_bias")),
        stop_bias_base=float(pp_cfg.get("stop_bias_base", -2.0)),
        stop_bias_slope=float(pp_cfg.get("stop_bias_slope", 0.35)),
        rollout_depth=int(pp_cfg.get("rollout_depth", 3)),
        rollouts_per_action=int(pp_cfg.get("rollouts_per_action", 1)),
        rollout_topk=int(pp_cfg.get("rollout_topk", 1)),
        max_rollout_support=pp_cfg.get("max_rollout_support", 16),
        beta=float(pp_cfg.get("beta", 1.0)),
        gamma=float(pp_cfg.get("gamma", 0.1)),
        gram_rank=pp_cfg.get("gram_rank", 8),
        teacher_steps=int(pp_cfg.get("teacher_steps", 2)),
        enable_stop=bool(pp_cfg.get("enable_stop", True)),
        max_abs_semantic=pp_cfg.get("max_abs_semantic", 1e6),
        max_energy_growth=pp_cfg.get("max_energy_growth", 100.0),
        max_support_size=pp_cfg.get("max_support_size", 64),
        support_mode=str(pp_cfg.get("support_mode", "deterministic_cap")),
        support_topk=(
            None
            if pp_cfg.get("support_topk") is None
            else int(pp_cfg.get("support_topk"))
        ),
        support_full_threshold=(
            None
            if pp_cfg.get("support_full_threshold") is None
            else int(pp_cfg.get("support_full_threshold"))
        ),
        terminal_op_penalty=pp_cfg.get("terminal_op_penalty"),
        cache_path=pp_cfg.get("cache_path"),
        gp_population_path=pp_cfg.get("gp_population_path"),
        shape_samples=int(pp_cfg.get("shape_samples", 32)),
        gp_likelihood_weight=float(pp_cfg.get("gp_likelihood_weight", 1.0)),
        gp_fitness_weight=float(pp_cfg.get("gp_fitness_weight", 1.0)),
        importance_samples=(
            None
            if pp_cfg.get("importance_samples") is None
            else int(pp_cfg.get("importance_samples"))
        ),
        mcmc_burn_in=int(pp_cfg.get("mcmc_burn_in", 16)),
    )
    train_cfg = cfg.get("train", {})
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 5e-4)))
    iterations = max(int(train_cfg.get("on_policy_iterations", 3)), 1)
    steps_per_iteration = max(int(train_cfg.get("steps_per_iteration", 50)), 1)
    batch_size = int(train_cfg.get("batch_size", 2))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 20))
    loss_fn = SemanticFisherVelocityLoss()
    rows = []
    step = 0
    ckpt_name = cfg.get("checkpoint_name", "path_posterior_flow.pt")
    for iteration in range(iterations):
        behavior = snapshot_behavior_model(model)
        iter_build_cfg = replace(build_cfg, behavior_policy_id=f"{Path(ckpt_name).stem}:iter_{iteration}")
        build_start = time.perf_counter()
        dataset = build_path_posterior_dataset(
            gen,
            num_tasks=int(cfg.get("num_tasks", 8)),
            behavior_model=behavior,
            seed=int(cfg.get("seed", 0)) + iteration,
            energy_cfg=energy_cfg,
            cfg=iter_build_cfg,
        )
        dataset_build_seconds = time.perf_counter() - build_start
        build_stats = _dataset_record_stats(dataset.records)
        records_per_sec = len(dataset) / max(dataset_build_seconds, 1e-12)
        build_stats["dataset_build_seconds"] = float(dataset_build_seconds)
        build_stats["dataset_records_per_sec"] = float(records_per_sec)
        print(
            "iter "
            f"{iteration} dataset records={len(dataset)} "
            f"build_s={dataset_build_seconds:.2f} "
            f"records_s={records_per_sec:.2f} "
            f"target_sampler_s_mean={build_stats.get('target_sampler_runtime_sec_mean', 0.0):.4f} "
            f"support_mean={build_stats.get('target_support_size_mean', 0.0):.1f}"
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_velocity)
        local_step = 0
        while local_step < steps_per_iteration:
            for batch in loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = model(
                    x=batch["x"],
                    y=batch["y"],
                    B=batch["B"],
                    p_lambda=batch["p_lambda"],
                    lambda_value=batch["lambda"],
                    action_feats=batch["action_feats"],
                    energies=batch["energies"],
                    weights=batch["weights"],
                    semantic_stats=batch.get("semantic_stats"),
                    gram=batch.get("gram"),
                    action_mask=batch["action_mask"],
                )
                loss, metrics = loss_fn(
                    p_start=batch["p_lambda"],
                    w_target=batch["w_target"],
                    w_pred=out.lograte_logits,
                    zdot_target=batch["zdot_target"],
                    z_dot_pred=out.z_dot_pred,
                    rewards=batch.get("rewards"),
                    mask=batch["action_mask"],
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                if step % log_every == 0:
                    row = {
                        "step": step,
                        "iteration": iteration,
                        "loss": float(loss.detach().cpu().item()),
                        "support_size_mean": float(batch["action_mask"].sum(dim=1).float().mean().detach().cpu()),
                        "num_records": len(dataset),
                    }
                    row.update(build_stats)
                    row.update(metrics)
                    rows.append(row)
                    print(f"iter {iteration} step {step} loss {row['loss']:.6f}")
                step += 1
                local_step += 1
                if local_step >= steps_per_iteration:
                    break
    out_dir = Path(cfg.get("out", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        out_dir / ckpt_name,
        model,
        optimizer=opt,
        meta={
            "cfg": cfg,
            "model_cfg": model_cfg.__dict__,
            "algorithm": _algorithm_name(build_cfg.target_mode),
            "gradient_steps": step,
            "final_loss": rows[-1]["loss"] if rows else None,
        },
    )
    _write_curve(out_dir / f"train_curve_{Path(ckpt_name).stem}.csv", rows)
    print(f"saved {out_dir / ckpt_name}")


def _write_curve(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _dataset_record_stats(records: list[dict]) -> dict[str, float]:
    scalar_keys = (
        "target_entropy",
        "p_init_entropy",
        "target_kl_q_pinit",
        "stop_target_mass",
        "target_score_gap",
        "target_sampler_runtime_sec",
        "target_support_size",
        "full_action_size",
    )
    out: dict[str, float] = {}
    for key in scalar_keys:
        values = [_record_scalar(rec[key]) for rec in records if key in rec]
        values = [v for v in values if v is not None]
        if values:
            out[f"{key}_mean"] = float(sum(values) / len(values))
    return out


def _record_scalar(value) -> float | None:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _algorithm_name(target_mode: str) -> str:
    mode = str(target_mode).strip().lower().replace("-", "_")
    if "one_step_group" in mode or "archive_one_step" in mode:
        return "semantic_fisher_flow_matching_one_step_group_advantage"
    if "one_step" in mode:
        return "semantic_fisher_flow_matching_one_step_target"
    if "future_group" in mode:
        return "semantic_fisher_flow_matching_future_group_l3"
    return "semantic_fisher_flow_matching"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
