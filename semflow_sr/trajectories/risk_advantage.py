"""Group-relative risk-seeking trajectory advantages."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class RiskAdvantageOutput:
    trajectory_advantages: torch.Tensor
    risk_threshold: float
    top_alpha_mask: torch.Tensor
    reward_rank: torch.Tensor


def build_group_advantages(
    rewards,
    mode: str = "top_alpha",
    alpha: float = 0.1,
    normalize: str = "rank",
) -> RiskAdvantageOutput:
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.numel() == 0:
        empty_bool = torch.empty_like(rewards, dtype=torch.bool)
        empty_rank = torch.empty_like(rewards, dtype=torch.long)
        return RiskAdvantageOutput(rewards, 0.0, empty_bool, empty_rank)
    if mode not in {"top_alpha", "z_score", "rank"}:
        raise ValueError(f"unknown risk advantage mode: {mode}")
    ranks = _rank_desc(rewards)
    if mode == "z_score":
        centered = rewards - rewards.mean()
        adv = centered / centered.std(unbiased=False).clamp(min=1e-12)
        mask = adv > 0
        return RiskAdvantageOutput(torch.where(mask, adv, torch.zeros_like(adv)), float(rewards.mean()), mask, ranks)
    if mode == "rank":
        scores = _rank_scores(ranks, rewards.numel())
        return RiskAdvantageOutput(scores, float(rewards.median()), torch.ones_like(rewards, dtype=torch.bool), ranks)

    k = max(1, min(int(math.ceil(float(alpha) * rewards.numel())), rewards.numel()))
    top_vals, _ = torch.topk(rewards, k)
    threshold = top_vals.min()
    mask = rewards >= threshold
    if normalize == "rank":
        values = _rank_scores(ranks, rewards.numel())
        cutoff = values[mask].min()
    elif normalize == "zscore":
        values = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp(min=1e-12)
        cutoff = values[mask].min()
    elif normalize == "none":
        values = rewards
        cutoff = threshold
    else:
        raise ValueError(f"unknown risk advantage normalization: {normalize}")
    advantages = torch.where(mask, (values - cutoff).clamp(min=0.0), torch.zeros_like(values))
    return RiskAdvantageOutput(
        trajectory_advantages=torch.nan_to_num(advantages),
        risk_threshold=float(threshold.detach().cpu().item()),
        top_alpha_mask=mask,
        reward_rank=ranks,
    )


def _rank_desc(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort(descending=True)
    ranks = torch.empty_like(order)
    ar = torch.arange(order.numel(), device=values.device, dtype=order.dtype) + 1
    ranks.scatter_(0, order, ar)
    return ranks


def _rank_scores(ranks: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 1:
        return torch.ones_like(ranks, dtype=torch.float32)
    return (n - ranks.to(torch.float32)) / float(n - 1)
