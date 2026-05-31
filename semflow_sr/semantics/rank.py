"""Effective-rank utilities (thin wrapper over the projection backend)."""
from __future__ import annotations
import torch
from .projection import ProjectionBackend


def effective_rank(B: torch.Tensor, rho: float = 1e-3) -> torch.Tensor:
    return ProjectionBackend("ridge", rho).effective_rank(B)
