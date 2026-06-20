"""End-to-end complete-trajectory candidate target construction."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from ..actions.action_space import ActionSpace
from ..registers.state import RegisterState
from ..semantics.energy import ActionEnergyConfig
from .base import CandidateFlowTarget
from .cache import CandidateTargetCache
from .config import CandidateTrajectoryConfig, build_candidate_sampler
from .evaluator import CandidateEvaluator
from .target import CandidateTargetBuilder


@dataclass
class CandidateTrajectoryTargetFactory:
    """Build a Semantic-Fisher target over complete executable candidates."""

    space: ActionSpace
    energy_cfg: ActionEnergyConfig | None = None
    candidate_cfg: CandidateTrajectoryConfig | None = None
    target_builder: CandidateTargetBuilder | None = None

    def __post_init__(self):
        self.energy_cfg = self.energy_cfg or ActionEnergyConfig()
        self.candidate_cfg = self.candidate_cfg or CandidateTrajectoryConfig()
        self.target_builder = self.target_builder or CandidateTargetBuilder()
        self.sampler = build_candidate_sampler(
            self.space,
            self.candidate_cfg,
            energy_cfg=self.energy_cfg,
        )
        self.evaluator = CandidateEvaluator(self.space, self.energy_cfg)

    def build(
        self,
        state: RegisterState,
        B: torch.Tensor,
        y: torch.Tensor,
        *,
        budget: int | None = None,
        cache_metadata: dict | None = None,
    ) -> CandidateFlowTarget:
        candidates = self.sampler.sample(state, B=B, y=y, budget=budget)
        eval_out = self.evaluator.evaluate(state, B, y, candidates)
        target = self.target_builder.build(candidates, eval_out)
        self._maybe_cache(target, cache_metadata or {})
        return target

    def _maybe_cache(self, target: CandidateFlowTarget, metadata: dict) -> None:
        cfg = self.candidate_cfg.cache
        if not cfg.enabled or not cfg.write:
            return
        CandidateTargetCache(cfg.path).append({
            **metadata,
            "block_sizes": [str(x) for x in self.candidate_cfg.block_sizes],
            "candidate_count": len(target.candidates),
            "candidate_groups": [c.metadata.get("candidate_group") for c in target.candidates],
            "candidate_kinds": [c.kind for c in target.candidates],
            "gram_rank": self.target_builder.gram_rank,
            "reward_mean": float(target.rewards.mean().detach().cpu()) if target.rewards.numel() else 0.0,
            "reward_max": float(target.rewards.max().detach().cpu()) if target.rewards.numel() else 0.0,
        })
