"""GP implicit-policy target extension.

This extension fuses a current model/start policy with a GP-induced local policy.
It returns the effective advantage that makes the exponential natural-flow path reach
the KL-barycenter endpoint. It is never used by base training unless explicitly wired
by a GP experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from ..flow.natural_path import effective_advantage_from_target
from ..gp_distill.kl_barycenter import kl_barycenter
from .base import AdvantageOutput, LocalCondition, PolicyDistribution


@dataclass
class GPImplicitBarycenterTarget:
    alpha_gp: float = 0.3
    beta: float = 1.0
    eta: float | None = None
    eps: float = 1e-12

    def build_advantage(
        self,
        condition: LocalCondition,
        p_start: PolicyDistribution,
    ) -> AdvantageOutput:
        gp_policy = condition.support_metadata.get("gp_policy")
        if gp_policy is None:
            raise ValueError("GPImplicitBarycenterTarget requires support_metadata['gp_policy']")
        gp_probs = gp_policy.probs if isinstance(gp_policy, PolicyDistribution) else gp_policy
        gp_probs = gp_probs.to(device=p_start.probs.device, dtype=p_start.probs.dtype)
        rho = kl_barycenter(p_start.probs, gp_probs, alpha=self.alpha_gp, eps=self.eps)
        beta = float(self.beta if self.eta is None else self.eta)
        advantages = effective_advantage_from_target(p_start.probs, rho, eta=beta, eps=self.eps)
        return AdvantageOutput(
            scores=rho,
            advantages=advantages,
            score_mean=rho.mean(dim=-1),
            score_std=rho.std(dim=-1, unbiased=False),
            metadata={
                "target_source": "gp_implicit_barycenter",
                "alpha_gp": float(self.alpha_gp),
                "beta": beta,
            },
        )
