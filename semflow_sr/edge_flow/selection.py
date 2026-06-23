"""Selection scores for CSEF inference reranking."""
from __future__ import annotations

import torch


def structure_prior_scores(
    rewards: torch.Tensor,
    log_probs: torch.Tensor,
    complexities: torch.Tensor,
    *,
    prior_weight: float,
    complexity_weight: float,
) -> torch.Tensor:
    """Return ``R2 + lambda_prior log q / C - lambda_C complexity`` scores."""

    base = torch.as_tensor(rewards).float().flatten()
    prior = torch.as_tensor(log_probs, dtype=base.dtype, device=base.device).flatten()
    complexity = torch.as_tensor(complexities, dtype=base.dtype, device=base.device).flatten()
    if prior.numel() != base.numel():
        prior = torch.zeros_like(base)
    if complexity.numel() != base.numel():
        complexity = torch.ones_like(base)
    length = complexity.clamp_min(1.0)
    return base + float(prior_weight) * (prior / length) - float(complexity_weight) * complexity
