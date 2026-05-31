"""Semantic-weighted Fisher / Hellinger distances on the action simplex (theory §1.7)."""
from __future__ import annotations
import torch
from ..utils.numerical import EPS
from .semantic_chart import semantic_chart


def semantic_fisher_distance(p0: torch.Tensor, p1: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    z0 = semantic_chart(p0, w); z1 = semantic_chart(p1, w)
    cos = (z0 * z1).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.arccos(cos)


def projection_frobenius_distance(Pi1: torch.Tensor, Pi2: torch.Tensor) -> torch.Tensor:
    """Explicit ‖Π1-Π2‖_F for already-materialized projectors (test helper)."""
    return (Pi1 - Pi2).pow(2).sum(dim=(-1, -2))
