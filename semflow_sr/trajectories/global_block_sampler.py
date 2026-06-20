"""Policy-aware complete trajectory sampling for RiskFlow SFSR.

The current implementation keeps the interface policy-aware even when the concrete
sampler is still grammar/random based. This makes cached records explicit about which
checkpoint or proposal policy produced them.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..actions.action_space import ActionSpace
from ..registers.state import RegisterState
from .sampler import GrammarTrajectorySampler, ModelTrajectorySampler, Trajectory, TrajectorySampler


@dataclass
class GlobalTrajectorySampler:
    action_space: ActionSpace
    seed: int = 0

    def sample_from_policy(
        self,
        state: RegisterState,
        model_or_policy,
        num_samples: int,
        max_len: int,
        temperature: float = 1.0,
        exploration: float = 0.0,
    ) -> list[Trajectory]:
        """Sample complete trajectories and record sampler provenance.

        If ``model_or_policy`` is itself a ``TrajectorySampler`` it is used directly.
        Otherwise we use the model-sampler placeholder plus grammar exploration. The
        placeholder currently samples from the legal grammar but labels records with
        the policy/checkpoint metadata so offline caches stay auditable.
        """
        n = max(int(num_samples), 0)
        if n == 0:
            return []
        if isinstance(model_or_policy, TrajectorySampler):
            trajectories = model_or_policy.sample(state, n, int(max_len), policy=model_or_policy)
        else:
            n_explore = min(n, max(0, int(round(n * float(exploration)))))
            n_policy = n - n_explore
            trajectories = []
            if n_policy:
                trajectories.extend(
                    ModelTrajectorySampler(self.action_space, seed=self.seed).sample(
                        state, n_policy, int(max_len), policy=model_or_policy
                    )
                )
            if n_explore:
                trajectories.extend(
                    GrammarTrajectorySampler(self.action_space, seed=self.seed + 17).sample(
                        state, n_explore, int(max_len)
                    )
                )
        policy_name = _policy_name(model_or_policy)
        for idx, trajectory in enumerate(trajectories):
            trajectory.metadata.setdefault("sampler_policy", policy_name)
            trajectory.metadata.setdefault("temperature", float(temperature))
            trajectory.metadata.setdefault("exploration", float(exploration))
            trajectory.metadata.setdefault("num_samples", int(num_samples))
            trajectory.metadata.setdefault("sample_index", idx)
            trajectory.metadata.setdefault("logprob_policy", trajectory.logprob_base)
        return trajectories[:n]


def _policy_name(model_or_policy) -> str:
    if model_or_policy is None:
        return "grammar"
    if isinstance(model_or_policy, str):
        return model_or_policy
    return getattr(model_or_policy, "checkpoint_path", None) or model_or_policy.__class__.__name__
