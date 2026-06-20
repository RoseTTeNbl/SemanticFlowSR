"""Candidate-level Semantic-Fisher target construction."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from ..actions.action_space import ActionSpace
from ..flow.semantic_fisher import (
    semantic_fisher_lograte,
    semantic_fisher_simplex_velocity,
    semantic_fisher_sphere_step,
    semantic_fisher_sphere_velocity,
)
from ..sr.ops import NAME_TO_ID
from ..utils.numerical import normalize_simplex
from .base import CandidateEvalOutput, CandidateFlowTarget, SemanticCandidate


@dataclass
class CandidateTargetBuilder:
    beta: float = 1.0
    gamma: float = 0.1
    advantage_eps: float = 1e-6
    advantage_clip: float | None = 5.0
    gram_rank: int | None = None

    def build(
        self,
        candidates: list[SemanticCandidate],
        eval_out: CandidateEvalOutput,
    ) -> CandidateFlowTarget:
        p_start = normalize_simplex(torch.exp(eval_out.log_priors - eval_out.log_priors.max()), dim=-1)
        advantages = self._advantages(eval_out.rewards)
        w_target = semantic_fisher_lograte(
            p_start,
            advantages,
            eval_out.gram,
            beta=self.beta,
            gamma=self.gamma,
            gram_rank=self.gram_rank,
            gram_factors=eval_out.xi,
        )
        z = p_start.clamp(min=1e-12).sqrt()
        zdot = semantic_fisher_sphere_velocity(z, w_target)
        pdot = semantic_fisher_simplex_velocity(p_start, w_target)
        p_target = semantic_fisher_sphere_step(p_start, w_target, dt=1.0)
        return CandidateFlowTarget(
            candidates=candidates,
            p_start=p_start,
            scores=eval_out.rewards,
            rewards=eval_out.rewards,
            advantages=advantages,
            w_target=w_target,
            zdot_target=zdot,
            pdot_target=pdot,
            p_target=p_target,
            eval=eval_out,
        )

    def _advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        adv = rewards - rewards.mean(dim=-1, keepdim=True)
        adv = adv / adv.std(dim=-1, keepdim=True, unbiased=False).clamp(min=self.advantage_eps)
        if self.advantage_clip is not None:
            adv = adv.clamp(min=-float(self.advantage_clip), max=float(self.advantage_clip))
        return torch.nan_to_num(adv)


def candidate_gp_log_prior(
    candidate: SemanticCandidate,
    space: ActionSpace,
    gp_action_scores: dict[int, float] | None = None,
    gp_operator_scores: dict[int | str, float] | None = None,
    gp_weight: float = 1.0,
    complexity_weight: float = 0.0,
) -> float:
    """Score GP priors as candidate log-prior terms, not additive log-rate bias."""
    action_scores = {int(k): float(v) for k, v in (gp_action_scores or {}).items()}
    op_scores = _normalize_operator_scores(gp_operator_scores or {})
    score = 0.0
    matched = False
    for action_id in candidate.actions or []:
        action_id = int(action_id)
        spec = space.decode(action_id)
        if action_id in action_scores:
            score += action_scores[action_id]
            matched = True
        if int(spec.op_id) in op_scores:
            score += op_scores[int(spec.op_id)]
            matched = True
    if not matched:
        score = 0.0
    score = float(gp_weight) * score - float(complexity_weight) * float(candidate.complexity)
    return float(score)


def _normalize_operator_scores(raw: dict[int | str, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for key, value in raw.items():
        if isinstance(key, str) and not key.isdigit():
            if key not in NAME_TO_ID:
                continue
            op_id = NAME_TO_ID[key]
        else:
            op_id = int(key)
        out[int(op_id)] = float(value)
    return out
