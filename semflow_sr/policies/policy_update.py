"""Local conservative policy improvement on a candidate support."""
from __future__ import annotations

from dataclasses import dataclass, field
import torch

from ..flow.natural_path import effective_advantage_from_target, smooth_simplex


@dataclass
class ProximalPolicyUpdate:
    p_start: torch.Tensor
    p_target: torch.Tensor
    raw_advantages: torch.Tensor
    effective_advantages: torch.Tensor
    beta: float
    damping_alpha: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def eta(self) -> float:
        """Compatibility alias for older callers."""
        return self.beta


def proximal_target_from_advantage(
    p_start: torch.Tensor,
    advantages: torch.Tensor,
    beta: float | None = None,
    *,
    eta: float | None = None,
    damping_alpha: float = 1.0,
    eps: float = 1e-12,
) -> ProximalPolicyUpdate:
    """Build rho(a) proportional to p_start(a) exp(beta A(a)).

    ``eta`` is accepted only as a legacy alias for ``beta``. If
    ``damping_alpha < 1``, return the damped endpoint and an effective advantage whose
    exponential path reconstructs that damped endpoint exactly.
    """
    if beta is None:
        if eta is None:
            raise TypeError("proximal_target_from_advantage requires beta")
        beta = eta
    elif eta is not None and float(eta) != float(beta):
        raise ValueError("beta and legacy eta alias disagree")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if not (0.0 < damping_alpha <= 1.0):
        raise ValueError("damping_alpha must be in (0, 1]")
    p_start = smooth_simplex(p_start, eps=eps)
    raw_adv = torch.nan_to_num(advantages.to(device=p_start.device, dtype=p_start.dtype))
    rho = torch.softmax(p_start.clamp_min(eps).log() + float(beta) * raw_adv, dim=-1)
    if damping_alpha < 1.0:
        p_target = (1.0 - float(damping_alpha)) * p_start + float(damping_alpha) * rho
        p_target = smooth_simplex(p_target, eps=eps)
        eff_adv = effective_advantage_from_target(p_start, p_target, beta, eps=eps, center=True)
    else:
        p_target = rho
        eff_adv = raw_adv - (p_start * raw_adv).sum(dim=-1, keepdim=True)
    return ProximalPolicyUpdate(
        p_start=p_start,
        p_target=p_target,
        raw_advantages=raw_adv,
        effective_advantages=eff_adv,
        beta=float(beta),
        damping_alpha=float(damping_alpha),
        metadata={
            "update": "exponential_proximal",
            "damped": bool(damping_alpha < 1.0),
            "beta": float(beta),
        },
    )
