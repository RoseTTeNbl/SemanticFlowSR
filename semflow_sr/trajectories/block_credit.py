"""Credit assignment from trajectory advantages to visited block decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

import torch


@dataclass(frozen=True)
class BlockCreditRecord:
    state_id: str
    state: Any
    block: tuple[int, ...]
    advantage: float
    old_logprob: float
    trajectory_id: str
    trajectory_reward: float | None = None
    source: str = "unknown"


@dataclass
class LocalBlockAdvantageGroup:
    state_id: str
    state: Any
    blocks: list[tuple[int, ...]]
    block_advantages: torch.Tensor
    old_logprob_blocks: torch.Tensor
    old_policy_probs: torch.Tensor
    trajectory_ids: list[list[str]] = field(default_factory=list)
    trajectory_rewards: list[list[float]] = field(default_factory=list)
    source_counts: list[int] = field(default_factory=list)
    aggregation: str = "mean"


def assign_trajectory_advantages_to_blocks(trajectories, advantages) -> list[BlockCreditRecord]:
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    records: list[BlockCreditRecord] = []
    for traj_idx, trajectory in enumerate(trajectories):
        adv = float(advantages[traj_idx].detach().cpu().item()) if traj_idx < advantages.numel() else 0.0
        traj_id = str(getattr(trajectory, "metadata", {}).get("trajectory_id", f"{trajectory.task_id}:{traj_idx}"))
        states = list(getattr(trajectory, "states", []))
        blocks = list(getattr(trajectory, "blocks", []))
        logprobs = list(getattr(trajectory, "block_logprobs", []))
        for step_idx, block in enumerate(blocks):
            state = states[step_idx] if step_idx < len(states) else None
            records.append(BlockCreditRecord(
                state_id=_state_id(state),
                state=state,
                block=tuple(int(a) for a in block),
                advantage=adv,
                old_logprob=float(logprobs[step_idx]) if step_idx < len(logprobs) else 0.0,
                trajectory_id=traj_id,
                trajectory_reward=getattr(trajectory, "reward", None),
                source=str(getattr(trajectory, "source", "unknown")),
            ))
    return records


def aggregate_local_block_advantages(
    records: list[BlockCreditRecord],
    aggregation: str = "mean",
    topk: int = 3,
) -> dict[str, LocalBlockAdvantageGroup]:
    if aggregation not in {"mean", "topk_mean", "sum", "max"}:
        raise ValueError(f"unknown block advantage aggregation: {aggregation}")
    by_state: dict[str, list[BlockCreditRecord]] = {}
    for record in records:
        by_state.setdefault(record.state_id, []).append(record)
    groups: dict[str, LocalBlockAdvantageGroup] = {}
    for state_id, state_records in by_state.items():
        by_block: dict[tuple[int, ...], list[BlockCreditRecord]] = {}
        order: list[tuple[int, ...]] = []
        for record in state_records:
            if record.block not in by_block:
                by_block[record.block] = []
                order.append(record.block)
            by_block[record.block].append(record)
        adv_values = []
        old_logprobs = []
        trajectory_ids = []
        trajectory_rewards = []
        source_counts = []
        for block in order:
            block_records = by_block[block]
            vals = torch.tensor([r.advantage for r in block_records], dtype=torch.float32)
            adv_values.append(_aggregate(vals, aggregation, topk=topk))
            old_probs = torch.tensor([math.exp(float(r.old_logprob)) for r in block_records], dtype=torch.float32)
            old_logprobs.append(old_probs.mean().clamp(min=1e-12).log())
            trajectory_ids.append([r.trajectory_id for r in block_records])
            trajectory_rewards.append([
                float(r.trajectory_reward) for r in block_records if r.trajectory_reward is not None
            ])
            source_counts.append(len(block_records))
        old_logprob_t = torch.stack(old_logprobs) if old_logprobs else torch.empty(0)
        old_probs_t = torch.softmax(old_logprob_t, dim=0) if old_logprob_t.numel() else old_logprob_t
        groups[state_id] = LocalBlockAdvantageGroup(
            state_id=state_id,
            state=state_records[0].state,
            blocks=order,
            block_advantages=torch.stack(adv_values) if adv_values else torch.empty(0),
            old_logprob_blocks=old_logprob_t,
            old_policy_probs=old_probs_t,
            trajectory_ids=trajectory_ids,
            trajectory_rewards=trajectory_rewards,
            source_counts=source_counts,
            aggregation=aggregation,
        )
    return groups


def _aggregate(values: torch.Tensor, aggregation: str, *, topk: int) -> torch.Tensor:
    if values.numel() == 0:
        return torch.tensor(0.0)
    if aggregation == "mean":
        return values.mean()
    if aggregation == "sum":
        return values.sum()
    if aggregation == "max":
        return values.max()
    k = min(max(int(topk), 1), values.numel())
    return torch.topk(values, k).values.mean()


def _state_id(state: Any) -> str:
    if state is None:
        return "unknown"
    if isinstance(state, str):
        return state
    return getattr(state, "state_id", None) or f"state:{id(state)}"
