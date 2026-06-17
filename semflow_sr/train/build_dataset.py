"""Build local natural-flow velocity datasets from synthetic register traces."""
from __future__ import annotations
import random

from ..data.synthetic_generator import GenConfig, generate_trace_task
from ..data.trace_dataset import VelocityTraceDataset, build_step_records
from ..actions.action_space import ActionSpace
from ..semantics.energy import ActionEnergyConfig
from ..actions.support_sampler import SupportSampler
from ..endpoints.prior_uniform import UniformPrior
from ..endpoints.target_gt import GTTarget
from ..endpoints.target_semantic_oracle import SemanticOracleTarget
from ..endpoints.target_group_advantage import GroupAdvantageTarget
from ..endpoints.target_rollout_fitness import RolloutFitnessTarget
from ..sr.ops import NAME_TO_ID


def build_dataset(gen: GenConfig, num_tasks: int, target: str = "gt",
                  beta: float | None = None, seed: int = 0, max_support: int = 256,
                  energy_cfg: ActionEnergyConfig | None = None,
                  support_mode: str = "mixed_topk_random",
                  support_topk: int | None = None,
                  target_kwargs: dict | None = None,
                  cache_static: bool = True,
                  data_device: str = "cpu",
                  path_name: str = "semantic_fisher_pullback",
                  eta: float | None = None,
                  gamma: float = 0.1,
                  flow_training: dict | None = None) -> VelocityTraceDataset:
    if beta is None:
        beta = 1.0 if eta is None else eta
    elif eta is not None and float(eta) != float(beta):
        raise ValueError("beta and legacy eta alias disagree")
    rng = random.Random(seed)
    allowed = [NAME_TO_ID[o] for o in gen.ops]
    space = ActionSpace(gen.K, allowed)
    records = []
    for _ in range(num_tasks):
        task = generate_trace_task(gen, rng)
        if task is None:
            continue
        _, trace, x, y = task
        records.extend(build_step_records(trace, x, y))
    prior = UniformPrior()
    if target == "gt":
        tgt = GTTarget()
    elif target == "semantic_oracle":
        tgt = SemanticOracleTarget()
    elif target in {"one_step_advantage", "group_advantage", "semantic_advantage_flow"}:
        tgt = GroupAdvantageTarget(**(target_kwargs or {}))
    elif target in {"rollout_fitness_advantage", "rollout_fitness"}:
        tgt = RolloutFitnessTarget(space, energy_cfg or ActionEnergyConfig(), **(target_kwargs or {}))
    else:
        raise ValueError(f"unknown target endpoint: {target}")
    sampler = SupportSampler(mode=support_mode, max_support=max_support, topk=support_topk, seed=seed)
    return VelocityTraceDataset(records, space, prior, tgt, energy_cfg or ActionEnergyConfig(),
                                beta=beta, seed=seed, max_support=max_support,
                                support_sampler=sampler, cache_static=cache_static,
                                data_device=data_device, path_name=path_name,
                                gamma=gamma, flow_training=flow_training)
