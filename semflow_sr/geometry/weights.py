"""Semantic weights w_{B,y}(a) = exp(-η/2 · E_{B,y}(a))."""
from __future__ import annotations
import torch


def semantic_weights(energies: torch.Tensor, eta: float, w_min: float = 1e-8) -> torch.Tensor:
    log_w = -0.5 * eta * (energies - energies.min(dim=-1, keepdim=True).values)
    w = torch.exp(log_w)
    return torch.clamp(w, min=w_min)
