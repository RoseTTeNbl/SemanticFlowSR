"""Uniform prior p0 over the valid action support."""
from __future__ import annotations
import torch
from .base import PriorEndpoint


class UniformPrior(PriorEndpoint):
    def build_p0(self, B, y, action_ids, context):
        A = action_ids.shape[0]
        return torch.full((A,), 1.0 / max(A, 1), device=B.device, dtype=B.dtype)
