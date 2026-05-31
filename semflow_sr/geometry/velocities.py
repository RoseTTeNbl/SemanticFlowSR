"""Velocity-space helpers: tangent projection and semantic Fisher metric norm."""
from __future__ import annotations
import torch
from ..utils.numerical import EPS


def tangent_project(v: torch.Tensor) -> torch.Tensor:
    """Project onto the tangent of the simplex: subtract mean so Σ v = 0."""
    return v - v.mean(dim=-1, keepdim=True)


def semantic_metric_norm_sq(v: torch.Tensor, p: torch.Tensor, w: torch.Tensor,
                            eps: float = EPS) -> torch.Tensor:
    """g_{B,y,p}(v,v) = ¼ Cov_{a~q}(v/p). q ∝ w² p (theory §1.8). Returns [...]."""
    q_un = (w * w) * p
    q = q_un / q_un.sum(dim=-1, keepdim=True).clamp(min=eps)
    ratio = v / p.clamp(min=eps)
    mean = (q * ratio).sum(dim=-1, keepdim=True)
    centered = ratio - mean
    return 0.25 * (q * centered * centered).sum(dim=-1)
