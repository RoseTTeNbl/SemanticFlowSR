"""One-step centered semantic reward advantage target."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from ..actions.action_space import ActionSpace
from ..semantics.energy import ActionEnergy, ActionEnergyConfig
from .base import AdvantageOutput, LocalCondition, PolicyDistribution


@dataclass
class OneStepAdvantageTarget:
    action_space: ActionSpace
    energy_cfg: ActionEnergyConfig | None = None
    advantage_eps: float = 1e-6
    advantage_clip: float | None = 5.0

    def __post_init__(self):
        self.energy_cfg = self.energy_cfg or ActionEnergyConfig()
        self.energy = ActionEnergy(self.action_space, self.energy_cfg)

    def build_advantage(
        self,
        condition: LocalCondition,
        p_start: PolicyDistribution,
    ) -> AdvantageOutput:
        del p_start
        scores = self.energy.rewards(condition.B, condition.y, condition.action_ids)
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
            metadata={
                "target_source": "one_step_advantage",
                "projection": self.energy_cfg.projection,
                "rho": float(self.energy_cfg.rho),
                "lambda_op": float(self.energy_cfg.lambda_op),
                "lambda_rank": float(self.energy_cfg.lambda_rank),
                "lambda_move": float(self.energy_cfg.lambda_move),
            },
        )
