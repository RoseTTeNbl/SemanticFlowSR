"""Start-policy providers for local action supports."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from ..sr.ops import op_cost
from ..targets.base import LocalCondition, PolicyDistribution


class PolicyProvider(ABC):
    @abstractmethod
    def get_policy(self, condition: LocalCondition) -> PolicyDistribution: ...


@dataclass
class UniformPolicyProvider(PolicyProvider):
    eps: float = 1e-12

    def get_policy(self, condition: LocalCondition) -> PolicyDistribution:
        n = int(condition.action_ids.numel())
        probs = torch.full((n,), 1.0 / max(n, 1), device=condition.B.device, dtype=condition.B.dtype)
        return PolicyDistribution(probs=probs, source="uniform", metadata={"support_size": n})


@dataclass
class GrammarPolicyProvider(PolicyProvider):
    op_cost_weight: float = 0.1

    def get_policy(self, condition: LocalCondition) -> PolicyDistribution:
        K = condition.state.K
        action_ids = condition.action_ids.detach().cpu().tolist()
        costs = []
        for action_id in action_ids:
            op_id = int(action_id) // (K * K * K)
            costs.append(float(op_cost(op_id)))
        cost_t = torch.tensor(costs, device=condition.B.device, dtype=condition.B.dtype)
        logits = -float(self.op_cost_weight) * cost_t
        return PolicyDistribution(
            probs=torch.softmax(logits, dim=-1),
            source="grammar",
            logits=logits,
            metadata={"op_cost_weight": float(self.op_cost_weight)},
        )
