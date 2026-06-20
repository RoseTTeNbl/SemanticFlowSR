"""Trajectory records used to collect states for target-sampled flow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PathDecision:
    state_id: str
    action_id: int
    action_ids: torch.Tensor
    p0: torch.Tensor
    state: Any


@dataclass
class PathTrajectory:
    task_id: str
    decisions: list[PathDecision]
    actions: list[int]
    reward: float | None = None
    final_r2: float | None = None
    metadata: dict | None = None
