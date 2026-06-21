"""Projection from sampled expression elites to edge target distributions."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .circuit_sampler import CircuitSample
from .edge_distribution import EdgeDistribution


@dataclass
class EliteProjection:
    theta_star: EdgeDistribution
    diagnostics: dict


def project_elites_to_edge_target(
    theta0: EdgeDistribution,
    samples: list[CircuitSample],
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    elite_k: int,
    smoothing: float = 1e-2,
    projection_mode: str = "global_topk",
) -> tuple[EdgeDistribution, dict]:
    if not samples:
        return theta0.clone(), {"target_ess": 0.0, "per_mode_elite_count": [0] * theta0.template.mixture_modes}
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool)
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if valid_idx.numel() == 0:
        return theta0.clone(), {"target_ess": 0.0, "per_mode_elite_count": [0] * theta0.template.mixture_modes}
    H = theta0.template.mixture_modes
    mode = str(projection_mode).strip().lower()
    if mode not in {"global_topk", "per_mode_topk"}:
        raise ValueError(f"unknown projection_mode: {projection_mode}")
    elite_idx = _select_elite_indices(samples, rewards, valid_idx, elite_k=int(elite_k), H=H, mode=mode)
    if elite_idx.numel() == 0:
        return theta0.clone(), {"target_ess": 0.0, "per_mode_elite_count": [0] * H}
    weights = torch.full((elite_idx.numel(),), 1.0 / elite_idx.numel(), dtype=torch.float32)
    alpha_counts = torch.zeros(H, dtype=torch.float32)
    group_counts = {
        group.group_id: torch.zeros(H, group.num_candidates, dtype=torch.float32)
        for group in theta0.template.groups
    }
    per_mode_elite_count = [0 for _ in range(H)]
    per_mode_best_reward = [float("-inf") for _ in range(H)]
    for weight, sample_idx_t in zip(weights, elite_idx):
        sample_idx = int(sample_idx_t.item())
        sample = samples[sample_idx]
        h = int(sample.mode)
        alpha_counts[h] += float(weight.item())
        per_mode_elite_count[h] += 1
        per_mode_best_reward[h] = max(per_mode_best_reward[h], float(rewards[sample_idx].item()))
        for group in theta0.template.groups:
            choice = int(sample.edge_choices[group.group_id])
            group_counts[group.group_id][h, choice] += float(weight.item())
    alpha_star = _smooth_normalize(alpha_counts, theta0.mixture_probs, smoothing)
    group_probs: dict[str, torch.Tensor] = {}
    for group in theta0.template.groups:
        prior = theta0.group_probs[group.group_id]
        counts = group_counts[group.group_id]
        rows = []
        for h in range(H):
            if float(counts[h].sum().item()) <= 1e-12:
                rows.append(prior[h])
            else:
                rows.append(_smooth_normalize(counts[h], prior[h], smoothing))
        group_probs[group.group_id] = torch.stack(rows, dim=0)
    ess = float(1.0 / (weights * weights).sum().item())
    diagnostics = {
        "target_ess": ess,
        "elite_k": int(elite_idx.numel()),
        "projection_mode": mode,
        "best_reward": float(rewards[elite_idx].max().item()),
        "elite_reward_mean": float(rewards[elite_idx].mean().item()),
        "per_mode_elite_count": per_mode_elite_count,
        "per_mode_best_reward": [0.0 if value == float("-inf") else float(value) for value in per_mode_best_reward],
        "mode_entropy": _entropy(alpha_star),
        "target_edge_entropy_mean": float(torch.stack([_row_entropy(p).mean() for p in group_probs.values()]).mean().item()),
    }
    return EdgeDistribution(theta0.template, alpha_star, group_probs), diagnostics


def _select_elite_indices(
    samples: list[CircuitSample],
    rewards: torch.Tensor,
    valid_idx: torch.Tensor,
    *,
    elite_k: int,
    H: int,
    mode: str,
) -> torch.Tensor:
    k = max(int(elite_k), 1)
    if mode == "global_topk":
        kk = min(k, int(valid_idx.numel()))
        valid_rewards = rewards[valid_idx]
        elite_local = torch.topk(valid_rewards, k=kk).indices
        return valid_idx[elite_local]
    selected: list[int] = []
    valid_list = [int(i) for i in valid_idx.detach().cpu().tolist()]
    for h in range(int(H)):
        mode_indices = [idx for idx in valid_list if int(samples[idx].mode) == h]
        if not mode_indices:
            continue
        idx_t = torch.tensor(mode_indices, dtype=torch.long, device=rewards.device)
        kk = min(k, int(idx_t.numel()))
        local = torch.topk(rewards[idx_t], k=kk).indices
        selected.extend(int(i) for i in idx_t[local].detach().cpu().tolist())
    if not selected:
        return torch.zeros(0, dtype=torch.long, device=rewards.device)
    return torch.tensor(selected, dtype=torch.long, device=rewards.device)


def _smooth_normalize(counts: torch.Tensor, prior: torch.Tensor, smoothing: float) -> torch.Tensor:
    counts = torch.nan_to_num(counts.float()).clamp_min(0.0)
    prior = torch.nan_to_num(prior.float()).clamp_min(1e-12)
    if float(counts.sum().item()) <= 1e-12:
        base = prior / prior.sum().clamp_min(1e-12)
    else:
        base = counts / counts.sum().clamp_min(1e-12)
    out = (1.0 - float(smoothing)) * base + float(smoothing) * prior / prior.sum().clamp_min(1e-12)
    return out / out.sum().clamp_min(1e-12)


def _entropy(p: torch.Tensor) -> float:
    q = p.clamp_min(1e-12)
    return float((-(q * q.log()).sum()).item())


def _row_entropy(p: torch.Tensor) -> torch.Tensor:
    q = p.clamp_min(1e-12)
    return -(q * q.log()).sum(dim=-1)
