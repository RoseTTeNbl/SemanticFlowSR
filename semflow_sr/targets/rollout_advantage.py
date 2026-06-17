"""Rollout/search future-value advantage target.

This target estimates Q(c,a) by short completions after the first action, then
returns group-normalized advantages. It does not define the flow path; the path is
still the exponential natural flow in ``semflow_sr.flow``.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from ..actions.action_space import ActionSpace
from ..endpoints.target_rollout_fitness import RolloutFitnessTarget
from ..semantics.energy import ActionEnergy, ActionEnergyConfig
from .base import AdvantageOutput, LocalCondition, PolicyDistribution


@dataclass
class RolloutValueAdvantageTarget:
    action_space: ActionSpace
    energy_cfg: ActionEnergyConfig | None = None
    eta_adv: float = 1.0
    advantage_eps: float = 1e-6
    advantage_clip: float = 5.0
    max_completion_steps: int = 4
    n_rollouts_per_action: int = 4
    rollout_policy: str = "mixed"
    reward_aggregation: str = "topk_mean"
    topk: int = 2
    fitness: str = "normalized_energy_improvement"
    complexity_penalty: float = 0.01
    eval_topk: int | None = 32
    fallback_scale: float = 0.25
    seed: int = 0

    def __post_init__(self):
        self.energy_cfg = self.energy_cfg or ActionEnergyConfig()
        self.energy = ActionEnergy(self.action_space, self.energy_cfg)
        self.endpoint_impl = RolloutFitnessTarget(
            self.action_space,
            self.energy_cfg,
            eta_adv=self.eta_adv,
            advantage_eps=self.advantage_eps,
            advantage_clip=self.advantage_clip,
            smoothing=0.0,
            max_completion_steps=self.max_completion_steps,
            n_rollouts_per_action=self.n_rollouts_per_action,
            rollout_policy=self.rollout_policy,
            reward_aggregation=self.reward_aggregation,
            topk=self.topk,
            fitness=self.fitness,
            complexity_penalty=self.complexity_penalty,
            eval_topk=self.eval_topk,
            fallback_scale=self.fallback_scale,
            seed=self.seed,
        )

    def build_advantage(
        self,
        condition: LocalCondition,
        p_start: PolicyDistribution,
    ) -> AdvantageOutput:
        x = condition.support_metadata.get("x")
        if x is None:
            raise ValueError("RolloutValueAdvantageTarget requires support_metadata['x']")
        ev = self.energy.evaluate_actions(condition.B, condition.y, condition.action_ids)
        ctx = {
            "state": condition.state,
            "x": x,
            "y": condition.y,
            "rewards": ev.rewards,
            "sample_index": int(condition.support_metadata.get("sample_index", 0)),
        }
        self.endpoint_impl.build_p1(
            condition.B,
            condition.y,
            condition.action_ids,
            ev.energies,
            p_start.probs,
            ctx,
        )
        scores = ctx["rollout_rewards"]
        advantages = ctx["advantages"]
        return AdvantageOutput(
            scores=scores,
            advantages=advantages,
            score_mean=scores.mean(dim=-1),
            score_std=scores.std(dim=-1, unbiased=False),
            metadata={
                "target_source": "rollout_value_advantage",
                "rollout_policy": self.rollout_policy,
                "fitness": self.fitness,
                "rollout_stats": ctx.get("rollout_stats", {}),
            },
        )
