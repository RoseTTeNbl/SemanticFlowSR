"""Edge-flow training record construction."""
from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from .circuit_sampler import CircuitSample, CircuitSampler
from .edge_distribution import EdgeDistribution
from .flow_teacher import build_fisher_slerp_record
from .projection import project_elites_to_edge_target
from .reward import RewardConfig, RewardBatch, evaluate_expression_rewards
from .template import RegisterOperatorTemplate


@dataclass(frozen=True)
class EdgeFlowBuildConfig:
    samples_per_task: int = 128
    elite_k: int = 16
    target_smoothing: float = 1e-2
    lambda_value: float | None = None
    complexity_weight: float = 0.001
    projection_mode: str = "global_topk"
    validation_fraction: float = 0.0


@dataclass
class EdgeFlowRecord:
    task_id: str
    x: torch.Tensor
    y: torch.Tensor
    template: RegisterOperatorTemplate
    theta0: EdgeDistribution
    theta_star: EdgeDistribution
    theta_lambda: EdgeDistribution
    z_lambda_mixture: torch.Tensor
    zdot_mixture: torch.Tensor
    z_lambda_groups: dict[str, torch.Tensor]
    zdot_groups: dict[str, torch.Tensor]
    sampled_expressions: list[str]
    rewards: torch.Tensor
    diagnostics: dict


def build_edge_flow_records(
    template: RegisterOperatorTemplate,
    *,
    tasks: list[tuple[str, torch.Tensor, torch.Tensor]],
    cfg: EdgeFlowBuildConfig | None = None,
    rng: random.Random | None = None,
) -> list[EdgeFlowRecord]:
    cfg = cfg or EdgeFlowBuildConfig()
    rng = rng or random.Random(0)
    theta0 = EdgeDistribution.uniform(template)
    sampler = CircuitSampler(template)
    records: list[EdgeFlowRecord] = []
    for task_id, x, y in tasks:
        samples = sampler.sample(theta0, batch_size=cfg.samples_per_task, rng=rng)
        rewards, reward_diag = _target_rewards(samples, x, y, cfg)
        theta_star, proj_diag = project_elites_to_edge_target(
            theta0,
            samples,
            rewards.rewards,
            rewards.valid_mask,
            elite_k=cfg.elite_k,
            smoothing=cfg.target_smoothing,
            projection_mode=cfg.projection_mode,
        )
        lam = rng.random() if cfg.lambda_value is None else float(cfg.lambda_value)
        teacher = build_fisher_slerp_record(theta0, theta_star, lam=lam)
        diagnostics = _diagnostics(samples, rewards, proj_diag)
        diagnostics.update(reward_diag)
        diagnostics.update(teacher.diagnostics)
        records.append(EdgeFlowRecord(
            task_id=str(task_id),
            x=x.float(),
            y=y.float(),
            template=template,
            theta0=theta0,
            theta_star=theta_star,
            theta_lambda=teacher.theta_lambda,
            z_lambda_mixture=teacher.z_lambda_mixture,
            zdot_mixture=teacher.zdot_mixture,
            z_lambda_groups=teacher.z_lambda_groups,
            zdot_groups=teacher.zdot_groups,
            sampled_expressions=[_expr_summary(sample) for sample in samples[: min(len(samples), 32)]],
            rewards=rewards.rewards,
            diagnostics=diagnostics,
        ))
    return records


def _diagnostics(samples: list[CircuitSample], rewards: RewardBatch, proj_diag: dict) -> dict:
    valid_count = int(rewards.valid_mask.sum().item())
    unique = len({sample.canonical or str(sample.expression) for sample in samples})
    out = {
        "num_sampled_expressions": int(len(samples)),
        "valid_expression_fraction": float(valid_count / max(len(samples), 1)),
        "unique_expression_fraction": float(unique / max(len(samples), 1)),
        "duplicate_expression_fraction": float(1.0 - unique / max(len(samples), 1)),
        "average_complexity": float(rewards.complexity.mean().item()) if rewards.complexity.numel() else 0.0,
        "best_reward": float(rewards.rewards.max().item()) if rewards.rewards.numel() else 0.0,
        "median_reward": float(rewards.rewards.median().item()) if rewards.rewards.numel() else 0.0,
    }
    out.update(proj_diag)
    return out


def _expr_summary(sample: CircuitSample) -> str:
    return str(sample.expression)


def _target_rewards(
    samples: list[CircuitSample],
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: EdgeFlowBuildConfig,
) -> tuple[RewardBatch, dict]:
    reward_cfg = RewardConfig(complexity_weight=cfg.complexity_weight)
    frac = max(0.0, min(float(cfg.validation_fraction), 0.9))
    if frac <= 0.0 or x.shape[0] < 8:
        return evaluate_expression_rewards(samples, x, y, reward_cfg), {"reward_validation_fraction": 0.0}
    cut = max(2, int(round(float(x.shape[0]) * (1.0 - frac))))
    cut = min(cut, int(x.shape[0]) - 2)
    train_rewards = evaluate_expression_rewards(samples, x[:cut], y[:cut], reward_cfg)
    val_rewards = evaluate_expression_rewards(samples, x[cut:], y[cut:], reward_cfg)
    robust = torch.minimum(train_rewards.r2, val_rewards.r2) - float(cfg.complexity_weight) * train_rewards.complexity
    valid = train_rewards.valid_mask & val_rewards.valid_mask
    robust = torch.where(valid, robust, torch.full_like(robust, float(reward_cfg.invalid_reward)))
    out = RewardBatch(
        rewards=robust,
        r2=train_rewards.r2,
        nmse=train_rewards.nmse,
        valid_mask=valid,
        affine_coef=train_rewards.affine_coef,
        complexity=train_rewards.complexity,
    )
    gap = (train_rewards.r2 - val_rewards.r2).abs()
    return out, {
        "reward_validation_fraction": float(frac),
        "target_reward_train_val_gap_mean": float(gap.mean().item()) if gap.numel() else 0.0,
        "target_reward_train_val_gap_max": float(gap.max().item()) if gap.numel() else 0.0,
    }
