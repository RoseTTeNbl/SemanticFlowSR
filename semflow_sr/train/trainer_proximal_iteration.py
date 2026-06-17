"""Coordinator skeleton for semantic proximal flow iteration.

Outer iterations are intentionally separate from base training. A caller supplies a
snapshot/start-policy provider and an advantage provider; this coordinator builds
natural-flow target records that can be cached or fed into velocity training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random
import torch

from ..policies.base_prior import PolicyProvider
from ..targets.base import AdvantageOutput, LocalCondition, PolicyDistribution
from .target_dataset import NaturalFlowTargetRecord, build_natural_flow_target_record


class AdvantageProvider:
    def build_advantage(
        self,
        condition: LocalCondition,
        p_start: PolicyDistribution,
    ) -> AdvantageOutput: ...


@dataclass
class ProximalIterationConfig:
    beta: float = 1.0
    damping_alpha: float = 1.0
    seed: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def eta(self) -> float:
        """Compatibility alias for older configs/tests."""
        return self.beta


class SemanticProximalFlowIterationTrainer:
    """Build target records for one outer policy-improvement iteration."""

    def __init__(
        self,
        policy_provider: PolicyProvider,
        advantage_provider: AdvantageProvider,
        cfg: ProximalIterationConfig | None = None,
    ):
        self.policy_provider = policy_provider
        self.advantage_provider = advantage_provider
        self.cfg = cfg or ProximalIterationConfig()
        self._rng = random.Random(self.cfg.seed)

    def build_records(self, conditions: list[LocalCondition]) -> list[NaturalFlowTargetRecord]:
        records: list[NaturalFlowTargetRecord] = []
        for condition in conditions:
            policy = self.policy_provider.get_policy(condition)
            advantage = self.advantage_provider.build_advantage(condition, policy)
            lam = torch.tensor(self._rng.random(), device=condition.B.device, dtype=condition.B.dtype)
            records.append(
                build_natural_flow_target_record(
                    condition=condition,
                    policy=policy,
                    advantage=advantage,
                    beta=self.cfg.beta,
                    lambda_value=lam,
                    damping_alpha=self.cfg.damping_alpha,
                )
            )
        return records
