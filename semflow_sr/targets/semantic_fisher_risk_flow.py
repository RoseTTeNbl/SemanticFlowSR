"""Semantic-Fisher risk-flow teacher targets over local block supports."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..candidates.base import CandidateEvalOutput, CandidateFlowTarget, SemanticCandidate
from ..flow.semantic_fisher import (
    semantic_fisher_lograte,
    semantic_fisher_simplex_velocity,
    semantic_fisher_sphere_step,
    semantic_fisher_sphere_velocity,
)
from ..utils.numerical import EPS, normalize_simplex


@dataclass
class SemanticFisherRiskFlowTargetBuilder:
    beta: float = 1.0
    gamma: float = 0.1
    gram_rank: int | None = None
    normalize_advantage: bool = False

    def build(
        self,
        *,
        candidates: list[SemanticCandidate],
        old_policy_probs: torch.Tensor,
        block_advantages: torch.Tensor,
        gram: torch.Tensor,
        xi: torch.Tensor | None = None,
        eval_output: CandidateEvalOutput | None = None,
    ) -> CandidateFlowTarget:
        if not candidates:
            raise ValueError("risk-flow target requires at least one block candidate")
        p_start = normalize_simplex(old_policy_probs.to(dtype=block_advantages.dtype).clamp(min=EPS), dim=-1)
        advantages = torch.nan_to_num(block_advantages.to(device=p_start.device, dtype=p_start.dtype))
        if self.normalize_advantage:
            advantages = _group_normalize(advantages)
        gram = torch.nan_to_num(gram.to(device=p_start.device, dtype=p_start.dtype))
        xi = None if xi is None else torch.nan_to_num(xi.to(device=p_start.device, dtype=p_start.dtype))
        w_target = semantic_fisher_lograte(
            p_start,
            advantages,
            gram,
            beta=float(self.beta),
            gamma=float(self.gamma),
            gram_rank=self.gram_rank,
            gram_factors=xi,
        )
        z = p_start.clamp(min=EPS).sqrt()
        p_target = semantic_fisher_sphere_step(p_start, w_target, dt=1.0)
        for cand in candidates:
            cand.metadata.setdefault("target_kind", "semantic_fisher_risk_flow")
        eval_output = eval_output or _synthetic_eval(advantages, gram, xi)
        return CandidateFlowTarget(
            candidates=candidates,
            p_start=p_start,
            scores=advantages,
            rewards=advantages,
            advantages=advantages,
            w_target=w_target,
            zdot_target=semantic_fisher_sphere_velocity(z, w_target),
            pdot_target=semantic_fisher_simplex_velocity(p_start, w_target),
            p_target=p_target,
            eval=eval_output,
        )


def _group_normalize(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=-1, keepdim=True)
    std = centered.std(dim=-1, keepdim=True, unbiased=False)
    return torch.nan_to_num(centered / std.clamp(min=EPS))


def _synthetic_eval(advantages: torch.Tensor, gram: torch.Tensor, xi: torch.Tensor | None) -> CandidateEvalOutput:
    device = advantages.device
    dtype = advantages.dtype
    n = advantages.numel()
    if xi is None:
        xi = torch.zeros(n, 0, device=device, dtype=dtype)
    residual_current = torch.zeros(xi.shape[-1], device=device, dtype=dtype)
    residual_next = torch.zeros(n, xi.shape[-1], device=device, dtype=dtype)
    return CandidateEvalOutput(
        residual_current=residual_current,
        residual_next=residual_next,
        xi=xi,
        gram=gram,
        rewards=advantages,
        energies=-advantages,
        complexities=torch.zeros(n, device=device, dtype=dtype),
        log_priors=torch.zeros(n, device=device, dtype=dtype),
        B_after=torch.empty(0, device=device, dtype=dtype),
    )
