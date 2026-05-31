"""Ground-truth target p1 = (1-ε)·δ_{a*} + ε·p0, where a* is the GT action id.

Requires context['gt_action'] (the GT action *id*) and the action_ids support so we can
locate a* within the support. All targets keep positive support to stabilize velocity.
"""
from __future__ import annotations
import torch
from .base import TargetEndpoint
from ..utils.numerical import normalize_simplex


class GTTarget(TargetEndpoint):
    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def build_p1(self, B, y, action_ids, energies, p0, context):
        gt = context.get("gt_action")
        A = action_ids.shape[0]
        if gt is None:
            return p0
        pos = (action_ids == int(gt)).nonzero(as_tuple=False)
        if pos.numel() == 0:
            return p0
        delta = torch.zeros(A, device=B.device, dtype=B.dtype)
        delta[pos.item()] = 1.0
        p1 = (1 - self.epsilon) * delta + self.epsilon * p0
        return normalize_simplex(p1, dim=-1)
