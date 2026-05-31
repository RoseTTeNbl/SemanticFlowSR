"""Semantic-conditioned Fisher chart S_{B,y} and its inverse.

  S(p)(a)      = w(a)·√p(a) / ‖w·√p‖₂            (unit-norm point on the sphere)
  S⁻¹(z)(a) ∝ z(a)² / w(a)²                       (back to the simplex)

With w=1 this reduces to the ordinary square-root (Hellinger) chart.
"""
from __future__ import annotations
import torch
from ..utils.numerical import EPS, normalize_simplex


def semantic_chart(p: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    z_un = w * torch.sqrt(p.clamp(min=0.0))
    norm = z_un.norm(dim=-1, keepdim=True).clamp(min=eps)
    return z_un / norm


def inverse_semantic_chart(z: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    p_un = (z * z) / (w * w).clamp(min=eps)
    return normalize_simplex(p_un, dim=-1, eps=eps)
