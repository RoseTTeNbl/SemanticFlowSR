"""Search-improved advantage targets.

Search providers are treated like rollout providers: they estimate action scores on
the current support and return group-relative advantages. They do not modify the
centered semantic energy or the exponential natural-flow path.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .base import AdvantageOutput, LocalCondition, PolicyDistribution


@dataclass
class SearchImprovedAdvantageTarget:
    advantage_eps: float = 1e-6
    advantage_clip: float | None = 5.0

    def build_advantage(
        self,
        condition: LocalCondition,
        p_start: PolicyDistribution,
    ) -> AdvantageOutput:
        del p_start
        scores = condition.support_metadata.get("search_scores")
        if scores is None:
            raise ValueError("SearchImprovedAdvantageTarget requires support_metadata['search_scores']")
        scores = scores.to(device=condition.B.device, dtype=condition.B.dtype)
        mean = scores.mean(dim=-1, keepdim=True)
        centered = scores - mean
        std = centered.std(dim=-1, keepdim=True, unbiased=False)
        advantages = centered / (std + self.advantage_eps)
        advantages = torch.where(std <= self.advantage_eps, torch.zeros_like(advantages), advantages)
        advantages = advantages - advantages.mean(dim=-1, keepdim=True)
        if self.advantage_clip is not None:
            advantages = advantages.clamp(min=-float(self.advantage_clip), max=float(self.advantage_clip))
        return AdvantageOutput(
            scores=scores,
            advantages=advantages,
            score_mean=mean.squeeze(-1),
            score_std=std.squeeze(-1),
            metadata={"target_source": "search_improved_advantage"},
        )
