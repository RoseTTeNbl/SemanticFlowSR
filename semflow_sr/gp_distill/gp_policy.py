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


@dataclass
class GPPolicyDistillationPrior:
    """Distill GP event success likelihoods into action/operator prior scores.

    Events may contain ``action_id`` or ``op_id``/``op`` plus one of ``solved``,
    ``r2``, ``fitness`` or ``lineage_return``. The distilled score is a smoothed
    log-odds of success, suitable as an additive policy prior.
    """

    events: list[dict] = field(default_factory=list)
    success_r2_threshold: float = 0.999
    success_fitness_threshold: float = 0.0
    smoothing: float = 1.0

    def action_scores(self) -> dict[int, float]:
        return self._scores_for_key("action_id")

    def operator_scores(self) -> dict[int | str, float]:
        scores: dict[int | str, float] = {}
        scores.update(self._scores_for_key("op_id"))
        scores.update(self._scores_for_key("op"))
        return scores

    def merged_scores(self) -> tuple[dict[int, float], dict[int | str, float]]:
        return self.action_scores(), self.operator_scores()

    def _scores_for_key(self, key: str) -> dict:
        stats: dict = {}
        for event in self.events:
            if key not in event:
                continue
            raw_key = event[key]
            if raw_key is None:
                continue
            success = self._event_success(event)
            total, good = stats.get(raw_key, (0.0, 0.0))
            weight = max(float(event.get("weight", 1.0)), 0.0)
            stats[raw_key] = (total + weight, good + (weight if success else 0.0))
        out = {}
        for raw_key, (total, good) in stats.items():
            p_success = (good + self.smoothing) / max(total + 2.0 * self.smoothing, 1e-12)
            p_success = min(max(p_success, 1e-6), 1.0 - 1e-6)
            out[raw_key] = float(math.log(p_success / (1.0 - p_success)))
        return out

    def _event_success(self, event: dict) -> bool:
        if "solved" in event:
            return bool(event["solved"])
        if "r2" in event:
            return float(event["r2"]) >= self.success_r2_threshold
        if "fitness" in event:
            return float(event["fitness"]) >= self.success_fitness_threshold
        if "lineage_return" in event:
            return float(event["lineage_return"]) >= self.success_fitness_threshold
        return False
