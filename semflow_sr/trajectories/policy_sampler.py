"""Policy-aware trajectory records for risk-flow training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SampledTrajectory:
    task_id: str
    states: list[Any]
    blocks: list[tuple[int, ...]]
    actions: list[int]
    block_logprobs: list[float]
    trajectory_logprob: float
    reward: float | None = None
    final_r2: float | None = None
    complexity: float = 0.0
    behavior_policy_id: str = "unknown"
    source: str = "model"
    metadata: dict = field(default_factory=dict)

    @property
    def trajectory_id(self) -> str:
        return str(self.metadata.get("trajectory_id", f"{self.task_id}:{id(self)}"))
