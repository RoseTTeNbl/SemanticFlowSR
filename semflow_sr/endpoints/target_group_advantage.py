"""Group-advantage KL policy-improvement target.

Given rewards/advantages on a candidate support S, construct

    p1(a) ∝ p0(a) exp(eta_adv * A(a))

with optional proposal-probability correction for sampled supports.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from .base import TargetEndpoint
from ..utils.numerical import normalize_simplex


@dataclass
class GroupAdvantageTarget(TargetEndpoint):
    eta_adv: float = 1.0
    center: bool = True
    normalize: bool = True
    eps: float = 1e-6
    floor: float = 1e-3
    importance_correction: bool = True

    def advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        adv = rewards
        if self.center:
            adv = adv - adv.mean(dim=-1, keepdim=True)
        if self.normalize:
            std = adv.std(dim=-1, keepdim=True, unbiased=False)
            adv = adv / std.clamp(min=self.eps)
        return adv

    def build_p1(self, B, y, action_ids, energies, p0, context):
        rewards = context.get("rewards")
        if rewards is None:
            rewards = energies
        adv = self.advantages(rewards)
        logits = self.eta_adv * adv
        logits = logits - logits.max(dim=-1, keepdim=True).values
        un = p0 * torch.exp(logits)
        proposal = context.get("proposal_probs")
        if self.importance_correction and proposal is not None:
            un = un / proposal.to(device=un.device, dtype=un.dtype).clamp(min=self.eps)
        p = normalize_simplex(un, dim=-1)
        if self.floor > 0.0:
            p = normalize_simplex((1.0 - self.floor) * p + self.floor * p0, dim=-1)
        context["advantages"] = adv
        return p
