"""Shared helpers for training entry scripts: build a velocity trace dataset from
synthetic tasks under a chosen target endpoint (GT or semantic oracle)."""
from __future__ import annotations
import random

from ..data.synthetic_generator import GenConfig, generate_trace_task
from ..data.trace_dataset import VelocityTraceDataset, build_step_records
from ..actions.action_space import ActionSpace
from ..semantics.energy import ActionEnergyConfig
from ..endpoints.prior_uniform import UniformPrior
from ..endpoints.target_gt import GTTarget
from ..endpoints.target_semantic_oracle import SemanticOracleTarget
from ..sr.ops import NAME_TO_ID


def build_dataset(gen: GenConfig, num_tasks: int, target: str = "gt",
                  eta: float = 1.0, seed: int = 0, max_support: int = 256,
                  energy_cfg: ActionEnergyConfig | None = None) -> VelocityTraceDataset:
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
    tgt = GTTarget() if target == "gt" else SemanticOracleTarget()
    return VelocityTraceDataset(records, space, prior, tgt, energy_cfg or ActionEnergyConfig(),
                                eta=eta, seed=seed, max_support=max_support)
