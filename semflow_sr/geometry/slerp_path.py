"""Semantic Fisher slerp path with closed-form velocity (theory §1.9–1.10).

z0 = S(p0), z1 = S(p1), θ = arccos⟨z0,z1⟩.
  z_λ  = sin((1-λ)θ)/sinθ · z0 + sin(λθ)/sinθ · z1
  ż_λ  = -θ cos((1-λ)θ)/sinθ · z0 + θ cos(λθ)/sinθ · z1
  p_λ(a)  = (z_λ(a)²/w(a)²) / Σ_b (z_λ(b)²/w(b)²)
  ṗ_λ(a)  = p_λ(a) [ 2 ż_λ(a)/z_λ(a) - Σ_b p_λ(b)·2 ż_λ(b)/z_λ(b) ]

Edge cases: small θ -> normalized linear interpolation fallback (endpoint-difference
velocity); z_λ clamped away from 0 in the log-derivative term. Endpoints must be smoothed
to have positive support.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch

from ..utils.numerical import EPS, SQRT_EPS, normalize_simplex
from .semantic_chart import semantic_chart, inverse_semantic_chart

_SMALL_THETA = 1e-4


@dataclass
class PathSample:
    p_lambda: torch.Tensor
    dp_dlambda: torch.Tensor
    z_lambda: torch.Tensor
    dz_dlambda: torch.Tensor
    lambda_value: torch.Tensor
    meta: dict = field(default_factory=dict)


class SemanticFisherSlerpPath:
    def __init__(self, z_clamp: float = SQRT_EPS):
        self.z_clamp = z_clamp

    def sample(self, p0: torch.Tensor, p1: torch.Tensor, w: torch.Tensor,
               lambda_value: torch.Tensor | float) -> PathSample:
        z0 = semantic_chart(p0, w)
        z1 = semantic_chart(p1, w)
        lam = torch.as_tensor(lambda_value, dtype=z0.dtype, device=z0.device)
        if lam.dim() == 0:
            lam = lam.reshape([1] * z0.dim())        # broadcast over all dims
        else:
            lam = lam.unsqueeze(-1)                   # [bsz] -> [bsz,1]

        cos_t = (z0 * z1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        theta = torch.arccos(cos_t)
        sin_t = torch.sin(theta)
        small = theta.abs() < _SMALL_THETA

        # slerp (safe denom)
        sin_t_safe = torch.where(small, torch.ones_like(sin_t), sin_t)
        a0 = torch.sin((1 - lam) * theta) / sin_t_safe
        a1 = torch.sin(lam * theta) / sin_t_safe
        z_lam = a0 * z0 + a1 * z1
        da0 = -theta * torch.cos((1 - lam) * theta) / sin_t_safe
        da1 = theta * torch.cos(lam * theta) / sin_t_safe
        dz_lam = da0 * z0 + da1 * z1

        # small-theta fallback: normalized linear interp, velocity = (z1-z0)/norm proj
        z_lin = (1 - lam) * z0 + lam * z1
        z_lin_n = z_lin / z_lin.norm(dim=-1, keepdim=True).clamp(min=EPS)
        dz_lin = z1 - z0
        z_lam = torch.where(small, z_lin_n, z_lam)
        dz_lam = torch.where(small, dz_lin, dz_lam)

        # inverse chart -> p_lambda
        p_lam = inverse_semantic_chart(z_lam, w)

        # closed-form velocity via log-derivative of q=z^2
        z_safe = z_lam.sign() * z_lam.abs().clamp(min=self.z_clamp)
        ratio = 2.0 * dz_lam / z_safe                       # 2 ż/z
        mean_ratio = (p_lam * ratio).sum(dim=-1, keepdim=True)
        dp_lam = p_lam * (ratio - mean_ratio)
        dp_lam = dp_lam - dp_lam.mean(dim=-1, keepdim=True)  # enforce tangent (Σ=0)

        return PathSample(
            p_lambda=p_lam, dp_dlambda=dp_lam, z_lambda=z_lam, dz_dlambda=dz_lam,
            lambda_value=torch.as_tensor(lambda_value), meta={"theta": theta.squeeze(-1)},
        )
