"""Semantic-oracle target p1 from action energies.

Softmax over -E with temperature τ (lower energy -> higher mass), optionally top-k
sparsified then renormalized. Always positive support via floor mixing with p0.
"""
from __future__ import annotations
import torch
from .base import TargetEndpoint
from ..utils.numerical import normalize_simplex


class SemanticOracleTarget(TargetEndpoint):
    def __init__(self, tau: float = 1.0, top_k: int | None = None, floor: float = 1e-3):
        self.tau = tau
        self.top_k = top_k
        self.floor = floor

    def build_p1(self, B, y, action_ids, energies, p0, context):
        logits = -(energies - energies.min()) / max(self.tau, 1e-6)
        p = torch.softmax(logits, dim=-1)
        if self.top_k is not None and self.top_k < p.shape[-1]:
            vals, idx = torch.topk(p, self.top_k, dim=-1)
            masked = torch.zeros_like(p)
            masked[idx] = vals
            p = normalize_simplex(masked, dim=-1)
        p = (1 - self.floor) * p + self.floor * p0
        return normalize_simplex(p, dim=-1)
