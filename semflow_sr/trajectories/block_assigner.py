"""Assign terminal trajectory rewards to fixed-size block records."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .sampler import Trajectory


@dataclass
class BlockTrainingRecord:
    trajectory_id: int
    start: int
    prefix_actions: list[int]
    block_actions: list[int]
    global_reward: float
    block_mask: list[torch.Tensor]
    terminal_expr: Any | None = None
    metadata: dict = field(default_factory=dict)


class BlockTargetAssigner:
    def __init__(self, block_size: int):
        if int(block_size) <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = int(block_size)

    def assign(self, trajectories: list[Trajectory], rewards: torch.Tensor) -> list[BlockTrainingRecord]:
        records: list[BlockTrainingRecord] = []
        for trajectory_id, trajectory in enumerate(trajectories):
            actions = list(trajectory.actions)
            if len(actions) < self.block_size:
                continue
            reward = float(rewards[trajectory_id].detach().cpu().item())
            for start in range(0, len(actions) - self.block_size + 1):
                block = actions[start:start + self.block_size]
                records.append(BlockTrainingRecord(
                    trajectory_id=trajectory_id,
                    start=start,
                    prefix_actions=actions[:start],
                    block_actions=block,
                    global_reward=reward,
                    block_mask=[m.clone() for m in trajectory.masks[start:start + self.block_size]],
                    terminal_expr=trajectory.expr,
                    metadata={
                        "source": trajectory.metadata.get("source"),
                        "logprob_base": float(trajectory.logprob_base),
                    },
                ))
        return records
