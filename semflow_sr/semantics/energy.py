"""Action energy and local reward.

Mainline energy is centered projection residual plus operator cost. Rank and movement
penalties remain available as ablations through nonzero config values.

Fully vectorized over the candidate action support.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from ..sr.ops import op_cost
from ..actions.action_space import ActionSpace
from ..actions.action_executor import ActionExecutor
from .projection import ProjectionBackend


@dataclass
class ActionEnergyConfig:
    lambda_rank: float = 0.0
    lambda_move: float = 0.0
    lambda_op: float = 0.01
    projection: str = "ridge"
    rho: float = 1e-3


@dataclass
class ActionEvaluation:
    energies: torch.Tensor
    rewards: torch.Tensor
    residual_after: torch.Tensor
    op_costs: torch.Tensor
    B_after: torch.Tensor


@dataclass
class SemanticEffectOutput:
    residual_current: torch.Tensor
    residual_next: torch.Tensor
    xi: torch.Tensor
    gram: torch.Tensor
    rewards: torch.Tensor
    op_costs: torch.Tensor
    B_after: torch.Tensor


class ActionEnergy:
    def __init__(self, action_space: ActionSpace, config: ActionEnergyConfig | None = None):
        self.space = action_space
        self.executor = ActionExecutor(action_space)
        self.cfg = config or ActionEnergyConfig()
        self.proj = ProjectionBackend(self.cfg.projection, self.cfg.rho)

    def compute(self, B: torch.Tensor, y: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        """B:[m,K], y:[m], action_ids:[A] -> energies:[A]."""
        return self.evaluate_actions(B, y, action_ids).energies

    def rewards(self, B: torch.Tensor, y: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        """Immediate group reward: E(B)-E(Ba)-lambda_op*C_op(a)."""
        return self.evaluate_actions(B, y, action_ids).rewards

    def action_semantic_effects(
        self,
        B: torch.Tensor,
        y: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> SemanticEffectOutput:
        """Return centered residual effects for each action on the shared backend."""
        residual_current = self.proj.residual_vector(B, y)
        eval_out = self.evaluate_actions(B, y, action_ids)
        residual_next = self.proj.residual_vector(eval_out.B_after, y)
        xi = residual_current.unsqueeze(0) - residual_next
        gram = xi @ xi.transpose(-1, -2)
        return SemanticEffectOutput(
            residual_current=residual_current,
            residual_next=residual_next,
            xi=xi,
            gram=gram,
            rewards=eval_out.rewards,
            op_costs=eval_out.op_costs,
            B_after=eval_out.B_after,
        )

    def evaluate_actions(self, B: torch.Tensor, y: torch.Tensor,
                         action_ids: torch.Tensor) -> ActionEvaluation:
        """Evaluate action residuals, energies and rewards with one semantic execution."""
        base = self.proj.residual_energy(B, y)
        Ba = self.executor.execute_semantic(B, action_ids)
        after = self.proj.residual_energy(Ba, y)
        cop = torch.tensor([op_cost(self.space.decode(int(a)).op_id) for a in action_ids.tolist()],
                           device=B.device, dtype=B.dtype)
        energies = after + self.cfg.lambda_op * cop
        if self.cfg.lambda_rank:
            energies = energies + self.cfg.lambda_rank * self.proj.effective_rank(Ba)
        if self.cfg.lambda_move:
            energies = energies + self.cfg.lambda_move * self.proj.projection_distance(Ba, B)
        rewards = base - after - self.cfg.lambda_op * cop
        return ActionEvaluation(energies=energies, rewards=rewards, residual_after=after,
                                op_costs=cop, B_after=Ba)
