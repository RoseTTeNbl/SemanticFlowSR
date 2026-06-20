"""Block-policy trajectory records."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BlockDecision:
    state_id: str
    state: Any
    block_actions: tuple[int, ...]
    logprob_old: float
    table_logprobs: list[float]
    behavior_policy_id: str
    q_table: Any | None = None
    candidate_blocks: list[tuple[int, ...]] | None = None


@dataclass
class BlockTrajectory:
    task_id: str
    states: list[Any]
    decisions: list[BlockDecision]
    actions: list[int]
    trajectory_logprob: float
    reward: float | None = None
    final_r2: float | None = None
    complexity: float = 0.0
    source: str = "model"
    metadata: dict = field(default_factory=dict)
