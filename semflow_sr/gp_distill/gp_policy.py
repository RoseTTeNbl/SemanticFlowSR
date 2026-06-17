"""GP-induced local policy provider.

This module is an independent extension. It is not used by base natural-flow
training unless explicitly configured by proximal/GP experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import torch

from ..policies.base_prior import PolicyProvider
from ..targets.base import LocalCondition, PolicyDistribution


@dataclass
class GPImplicitPolicyProvider(PolicyProvider):
    events: list[dict] = field(default_factory=list)
    alpha: float = 1.0
    proposal_correction: bool = True
    eps: float = 1e-12

    def get_policy(self, condition: LocalCondition) -> PolicyDistribution:
        weights = torch.zeros(condition.action_ids.numel(), device=condition.B.device, dtype=condition.B.dtype)
        index = {int(a): i for i, a in enumerate(condition.action_ids.detach().cpu().tolist())}
        for event in self.events:
            action = event.get("action_id", event.get("action_or_macro"))
            try:
                action_id = int(action)
            except (TypeError, ValueError):
                continue
            if action_id not in index:
                continue
            score = float(event.get("lineage_return", event.get("fitness", 0.0)))
            weight = math.exp(max(min(self.alpha * score, 50.0), -50.0))
            if self.proposal_correction:
                q = math.exp(float(event.get("proposal_logprob", 0.0)))
                weight = weight / max(q, self.eps)
            weights[index[action_id]] += float(weight)
        if float(weights.sum().detach().cpu()) <= self.eps:
            weights = torch.ones_like(weights)
            source = "gp_uniform_fallback"
        else:
            source = "gp"
        return PolicyDistribution(probs=weights, source=source, metadata={"num_events": len(self.events)})
