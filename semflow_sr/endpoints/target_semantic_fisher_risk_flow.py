"""Risk-flow target endpoint for H1 action-support training records."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import TargetEndpoint
from ..actions.action_space import ActionSpace
from ..semantics.energy import ActionEnergyConfig
from ..trajectories.evaluator import GlobalTrajectoryEvaluator
from ..trajectories.risk_advantage import build_group_advantages
from ..trajectories.sampler import GrammarTrajectorySampler, TrajectorySampler
from ..utils.numerical import EPS, normalize_simplex


@dataclass
class SemanticFisherRiskFlowEndpoint(TargetEndpoint):
    """Complete-trajectory reward -> risk advantage -> visited first-action credit.

    This endpoint keeps the existing action-support training dataset usable for H1.
    H3/H5 block supports use ``build_risk_flow_block_target`` in search/trajectory
    code and share the same Semantic-Fisher target builder.
    """

    action_space: ActionSpace
    energy_cfg: ActionEnergyConfig | None = None
    sampler: TrajectorySampler | None = None
    num_samples: int = 64
    max_len: int = 5
    risk_mode: str = "top_alpha"
    risk_alpha: float = 0.1
    risk_normalize: str = "rank"
    aggregation: str = "mean"
    prior_smoothing: float = 0.05

    def __post_init__(self):
        self.energy_cfg = self.energy_cfg or ActionEnergyConfig()
        self.sampler = self.sampler or GrammarTrajectorySampler(self.action_space, seed=0)
        self.evaluator = GlobalTrajectoryEvaluator(self.action_space, self.energy_cfg)

    def build_p1(self, B, y, action_ids, energies, p0, context):
        state = context.get("state")
        x = context.get("x")
        if state is None or x is None:
            raise ValueError("SemanticFisherRiskFlowEndpoint requires context['state'] and context['x']")
        trajectories = self.sampler.sample(
            state,
            num_samples=int(self.num_samples),
            max_len=int(self.max_len),
            policy=context.get("policy"),
        )
        if not trajectories:
            adv = torch.zeros_like(p0)
            context["advantages"] = adv
            context["rewards"] = adv
            return p0
        eval_out = self.evaluator.evaluate(trajectories, x, y, initial_state=state)
        risk = build_group_advantages(
            eval_out.rewards.to(device=B.device, dtype=B.dtype),
            mode=self.risk_mode,
            alpha=float(self.risk_alpha),
            normalize=self.risk_normalize,
        )
        support_adv, counts = _first_action_support_advantages(
            trajectories,
            risk.trajectory_advantages.to(device=B.device, dtype=B.dtype),
            action_ids.to(device=B.device),
            aggregation=self.aggregation,
        )
        observed_prior = normalize_simplex(counts.to(device=B.device, dtype=B.dtype).clamp(min=0.0), dim=-1)
        p_start = normalize_simplex(
            (1.0 - float(self.prior_smoothing)) * observed_prior
            + float(self.prior_smoothing) * p0.to(device=B.device, dtype=B.dtype),
            dim=-1,
        )
        p_target = normalize_simplex(p_start * torch.exp(support_adv), dim=-1)
        context["p_start_override"] = p_start
        context["advantages"] = support_adv
        context["rewards"] = support_adv
        context["global_trajectory_rewards"] = support_adv
        context["rollout_rewards"] = support_adv
        context["trajectory_stats"] = {
            "target_mode": "semantic_fisher_risk_flow",
            "num_trajectories": len(trajectories),
            "risk_threshold": risk.risk_threshold,
            "top_alpha_fraction": float(risk.top_alpha_mask.float().mean().detach().cpu().item()),
            "trajectory_reward_mean": float(eval_out.rewards.mean().detach().cpu().item()),
            "trajectory_reward_std": float(eval_out.rewards.std(unbiased=False).detach().cpu().item()),
            "trajectory_reward_max": float(eval_out.rewards.max().detach().cpu().item()),
            "trajectory_advantage_mean": float(risk.trajectory_advantages.mean().detach().cpu().item()),
            "trajectory_advantage_std": float(risk.trajectory_advantages.std(unbiased=False).detach().cpu().item()),
            "unique_block_count_per_state": int((counts > 0).sum().detach().cpu().item()),
            "old_policy_entropy": _entropy(p_start),
            "block_advantage_std": float(support_adv.std(unbiased=False).detach().cpu().item()),
            "oracle_trajectory_r2": float(eval_out.final_r2.max().detach().cpu().item())
            if eval_out.final_r2.numel() else 0.0,
        }
        return p_target


def _first_action_support_advantages(
    trajectories,
    trajectory_advantages: torch.Tensor,
    action_ids: torch.Tensor,
    *,
    aggregation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    values: dict[int, list[torch.Tensor]] = {int(a): [] for a in action_ids.detach().cpu().tolist()}
    for idx, trajectory in enumerate(trajectories):
        if not trajectory.actions:
            continue
        action = int(trajectory.actions[0])
        if action in values:
            values[action].append(trajectory_advantages[idx])
    adv = []
    counts = []
    for action in action_ids.detach().cpu().tolist():
        vals = values[int(action)]
        if vals:
            stacked = torch.stack(vals)
            adv.append(_aggregate(stacked, aggregation))
            counts.append(float(len(vals)))
        else:
            adv.append(torch.tensor(0.0, device=trajectory_advantages.device, dtype=trajectory_advantages.dtype))
            counts.append(0.0)
    return torch.stack(adv), torch.tensor(counts, device=trajectory_advantages.device, dtype=trajectory_advantages.dtype)


def _aggregate(values: torch.Tensor, aggregation: str) -> torch.Tensor:
    if aggregation == "mean":
        return values.mean()
    if aggregation == "sum":
        return values.sum()
    if aggregation == "max":
        return values.max()
    if aggregation == "topk_mean":
        return torch.topk(values, min(3, values.numel())).values.mean()
    raise ValueError(f"unknown risk-flow aggregation: {aggregation}")


def _entropy(p: torch.Tensor) -> float:
    pp = p.detach().float().clamp(min=EPS)
    return float((-(pp * pp.log()).sum()).cpu().item())
