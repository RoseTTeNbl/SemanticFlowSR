"""Target records for semantic proximal flow iteration."""
from __future__ import annotations

from dataclasses import dataclass, field
import torch

from ..flow.natural_path import ExponentialNaturalFlowPath
from ..policies.policy_update import proximal_target_from_advantage
from ..targets.base import AdvantageOutput, LocalCondition, PolicyDistribution


@dataclass
class NaturalFlowTargetRecord:
    condition: LocalCondition
    p_start: torch.Tensor
    scores: torch.Tensor
    advantages: torch.Tensor
    p_target: torch.Tensor
    lambda_value: torch.Tensor
    p_lambda: torch.Tensor
    dp_dlambda: torch.Tensor
    policy_source: str
    target_source: str
    metadata: dict = field(default_factory=dict)


def build_natural_flow_target_record(
    condition: LocalCondition,
    policy: PolicyDistribution,
    advantage: AdvantageOutput,
    beta: float | None = None,
    lambda_value: torch.Tensor | float | None = None,
    damping_alpha: float = 1.0,
    eta: float | None = None,
) -> NaturalFlowTargetRecord:
    if lambda_value is None:
        raise TypeError("build_natural_flow_target_record requires lambda_value")
    if beta is None:
        if eta is None:
            raise TypeError("build_natural_flow_target_record requires beta")
        beta = eta
    elif eta is not None and float(eta) != float(beta):
        raise ValueError("beta and legacy eta alias disagree")
    update = proximal_target_from_advantage(
        policy.probs,
        advantage.advantages,
        beta=beta,
        damping_alpha=damping_alpha,
    )
    sample = ExponentialNaturalFlowPath().sample(
        p_start=policy.probs,
        advantages=update.effective_advantages,
        lambda_value=lambda_value,
        eta=beta,
        scores=advantage.scores,
        condition_metadata=condition.support_metadata,
    )
    return NaturalFlowTargetRecord(
        condition=condition,
        p_start=policy.probs,
        scores=advantage.scores,
        advantages=update.effective_advantages,
        p_target=update.p_target,
        lambda_value=sample.lambda_value,
        p_lambda=sample.p_lambda,
        dp_dlambda=sample.dp_dlambda,
        policy_source=policy.source,
        target_source=str(advantage.metadata.get("target_source", "unknown")),
        metadata={
            "raw_advantages": advantage.advantages,
            "score_mean": advantage.score_mean,
            "score_std": advantage.score_std,
            "policy_metadata": policy.metadata,
            "target_metadata": advantage.metadata,
            "update_metadata": update.metadata,
            "beta": float(beta),
        },
    )
