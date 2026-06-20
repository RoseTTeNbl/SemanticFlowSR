"""Complete trajectory samplers.

The sampler layer produces observed complete trajectories. RiskFlow turns terminal
trajectory reward into trajectory advantage, then assigns that advantage to the
visited block decisions used by the local semantic-Fisher target.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any

import torch

from ..actions.action_executor import ActionExecutor
from ..actions.action_space import ActionSpace
from ..registers.state import RegisterState


@dataclass
class Trajectory:
    actions: list[int]
    masks: list[torch.Tensor]
    logprob_base: float = 0.0
    expr: Any | None = None
    complexity: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def logprob_policy(self) -> float:
        return float(self.metadata.get("logprob_policy", self.logprob_base))

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    @property
    def expression(self):
        return self.expr


class TrajectorySampler:
    """Base sampler interface for complete executable action sequences."""

    def sample(
        self,
        state: RegisterState,
        num_samples: int,
        max_len: int,
        policy: Any | None = None,
    ) -> list[Trajectory]:
        raise NotImplementedError


class GrammarTrajectorySampler(TrajectorySampler):
    """Uniform masked grammar sampler over legal register actions."""

    def __init__(self, action_space: ActionSpace, seed: int = 0):
        self.space = action_space
        self.rng = random.Random(seed)
        self.executor = ActionExecutor(action_space)

    def sample(
        self,
        state: RegisterState,
        num_samples: int,
        max_len: int,
        policy: Any | None = None,
    ) -> list[Trajectory]:
        trajectories: list[Trajectory] = []
        for sample_idx in range(max(int(num_samples), 0)):
            current = state.clone()
            actions: list[int] = []
            masks: list[torch.Tensor] = []
            prefix_states = [current.clone()]
            logprob = 0.0
            for _ in range(max(int(max_len), 0)):
                mask = self.space.valid_mask(current)
                valid = mask.nonzero(as_tuple=False).squeeze(-1)
                if valid.numel() == 0:
                    break
                pos = self.rng.randrange(int(valid.numel()))
                action = int(valid[pos].detach().cpu().item())
                actions.append(action)
                masks.append(mask.clone())
                logprob -= math.log(float(valid.numel()))
                current = self.executor.execute_symbolic(current, action)
                prefix_states.append(current.clone())
            if not actions:
                continue
            trajectories.append(Trajectory(
                actions=actions,
                masks=masks,
                logprob_base=logprob,
                expr=current.exprs,
                complexity=_state_complexity(current),
                metadata={
                    "source": "grammar",
                    "sample_index": sample_idx,
                    "initial_state": state,
                    "prefix_states": prefix_states,
                    "final_state": current,
                },
            ))
        return trajectories


class ModelTrajectorySampler(GrammarTrajectorySampler):
    """Model sampler placeholder using grammar sampling until model proposal is wired."""

    def sample(self, state, num_samples, max_len, policy=None):
        out = super().sample(state, num_samples, max_len, policy=policy)
        for traj in out:
            traj.metadata["source"] = "model" if policy is not None else "grammar"
        return out


class GPTrajectorySampler(TrajectorySampler):
    """Sampler backed by precomputed GP final-population trajectories."""

    def __init__(self, trajectories: list[Trajectory] | None = None):
        self.trajectories = list(trajectories or [])

    def sample(self, state, num_samples, max_len, policy=None):
        out: list[Trajectory] = []
        for traj in self.trajectories[:max(int(num_samples), 0)]:
            cloned = Trajectory(
                actions=list(traj.actions[:max_len]),
                masks=[m.clone() for m in traj.masks[:max_len]],
                logprob_base=float(traj.logprob_base),
                expr=traj.expr,
                complexity=float(traj.complexity),
                metadata=dict(traj.metadata),
            )
            cloned.metadata.setdefault("initial_state", state)
            cloned.metadata["source"] = "gp"
            out.append(cloned)
        return out


class MixedTrajectorySampler(TrajectorySampler):
    """Budgeted mixture of trajectory samplers."""

    def __init__(self, samplers: list[TrajectorySampler], weights: list[float] | None = None):
        if not samplers:
            raise ValueError("MixedTrajectorySampler requires at least one sampler")
        self.samplers = samplers
        raw = weights or [1.0] * len(samplers)
        total = sum(max(float(w), 0.0) for w in raw)
        self.weights = [max(float(w), 0.0) / total for w in raw] if total > 0 else [1.0 / len(samplers)] * len(samplers)

    def sample(self, state, num_samples, max_len, policy=None):
        n = max(int(num_samples), 0)
        counts = [int(round(n * w)) for w in self.weights]
        while sum(counts) < n:
            counts[counts.index(min(counts))] += 1
        while sum(counts) > n:
            counts[counts.index(max(counts))] -= 1
        out: list[Trajectory] = []
        for sampler, count in zip(self.samplers, counts):
            out.extend(sampler.sample(state, count, max_len, policy=policy))
        return out[:n]


def _state_complexity(state: RegisterState) -> float:
    active = state.active.bool()
    if active.numel() == 0:
        return 0.0
    return float(state.complexity[active].sum().detach().cpu().item())
