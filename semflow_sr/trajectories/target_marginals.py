"""Convert terminal trajectory rewards to masked action-table marginals."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..utils.numerical import EPS, normalize_simplex
from .sampler import Trajectory


@dataclass
class MarginalTargetOutput:
    rho: torch.Tensor
    p0: torch.Tensor
    mask: torch.Tensor
    sample_weights: torch.Tensor
    reward_mean: torch.Tensor


def build_reward_weighted_marginals(
    trajectories: list[Trajectory],
    rewards: torch.Tensor,
    action_vocab_size: int,
    max_len: int,
    eta: float = 1.0,
    smoothing: float = 1e-2,
    base_p0: torch.Tensor | None = None,
    logprobs: torch.Tensor | None = None,
    likelihood_kappa: float = 0.0,
    likelihood_clip: float | None = None,
) -> MarginalTargetOutput:
    """Project reward-weighted complete trajectories onto per-position action marginals."""
    device = rewards.device
    dtype = rewards.dtype
    T = max(int(max_len), 1)
    A = int(action_vocab_size)
    mask = torch.zeros(T, A, device=device, dtype=torch.bool)
    counts = torch.zeros(T, A, device=device, dtype=dtype)
    reward_sums = torch.zeros(T, A, device=device, dtype=dtype)
    reward_counts = torch.zeros(T, A, device=device, dtype=dtype)
    weights = _trajectory_weights(
        rewards,
        eta=eta,
        logprobs=logprobs,
        likelihood_kappa=likelihood_kappa,
        likelihood_clip=likelihood_clip,
    )
    for i, trajectory in enumerate(trajectories):
        for t, action in enumerate(trajectory.actions[:T]):
            action_id = int(action)
            if t < len(trajectory.masks):
                mask[t] |= trajectory.masks[t].to(device=device, dtype=torch.bool)
            else:
                mask[t, action_id] = True
            counts[t, action_id] += weights[i]
            reward_sums[t, action_id] += rewards[i]
            reward_counts[t, action_id] += 1.0
    p0 = _base_prior(mask, base_p0, dtype=dtype)
    rho = torch.zeros_like(p0)
    eps = float(smoothing)
    for t in range(T):
        if not bool(mask[t].any()):
            continue
        count_sum = counts[t].sum()
        empirical = counts[t] / count_sum.clamp(min=EPS) if float(count_sum.detach().cpu()) > EPS else p0[t]
        rho[t] = normalize_simplex((1.0 - eps) * empirical + eps * p0[t], dim=-1)
    reward_mean = torch.where(reward_counts > 0, reward_sums / reward_counts.clamp(min=1.0), torch.zeros_like(reward_sums))
    return MarginalTargetOutput(
        rho=torch.nan_to_num(rho),
        p0=torch.nan_to_num(p0),
        mask=mask,
        sample_weights=weights,
        reward_mean=torch.nan_to_num(reward_mean),
    )


def build_effective_advantage(
    rho: torch.Tensor,
    p0: torch.Tensor,
    mask: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Build group-normalized effective advantage log(rho)-log(p0)."""
    raw = torch.zeros_like(rho)
    raw[mask] = rho[mask].clamp(min=eps).log() - p0[mask].clamp(min=eps).log()
    out = torch.zeros_like(raw)
    for t in range(raw.shape[0]):
        valid = mask[t]
        if not bool(valid.any()):
            continue
        vals = raw[t, valid]
        vals = vals - vals.mean()
        std = vals.std(unbiased=False)
        out[t, valid] = vals / std.clamp(min=eps)
    return torch.nan_to_num(out)


def _trajectory_weights(
    rewards: torch.Tensor,
    eta: float,
    logprobs: torch.Tensor | None = None,
    likelihood_kappa: float = 0.0,
    likelihood_clip: float | None = None,
) -> torch.Tensor:
    if rewards.numel() == 0:
        return rewards
    scores = float(eta) * _rank_normalize(rewards)
    if logprobs is not None and float(likelihood_kappa) != 0.0:
        lp = torch.nan_to_num(logprobs.to(device=rewards.device, dtype=rewards.dtype))
        correction = -float(likelihood_kappa) * lp
        if likelihood_clip is not None:
            correction = correction.clamp(min=-float(likelihood_clip), max=float(likelihood_clip))
        scores = scores + correction
    return torch.softmax(scores, dim=0)


def _rank_normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    order = values.argsort(descending=False)
    ranks = torch.empty_like(order, dtype=values.dtype)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=values.dtype)
    ranks = ranks / max(values.numel() - 1, 1)
    return ranks - ranks.mean()


def _base_prior(mask: torch.Tensor, base_p0: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
    if base_p0 is not None:
        p0 = base_p0.to(device=mask.device, dtype=dtype).clone()
        p0 = torch.where(mask, p0, torch.zeros_like(p0))
        return normalize_simplex(p0, dim=-1)
    p0 = mask.to(dtype=dtype)
    return normalize_simplex(p0, dim=-1)
