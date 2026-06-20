"""Assign trajectory advantages to visited H-step block table coordinates."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .trajectory import BlockTrajectory


@dataclass
class TableAdvantageGroup:
    state_id: str
    state: object
    advantages: torch.Tensor
    counts: torch.Tensor
    mask: torch.Tensor
    trajectory_ids: list[str]
    block_count: int
    blocks: list[tuple[int, ...]]
    candidate_blocks: list[tuple[int, ...]]
    q_start: torch.Tensor


def build_table_advantages_from_trajectories(
    trajectories: list[BlockTrajectory],
    trajectory_advantages: torch.Tensor,
    *,
    block_size: int,
    action_vocab_size: int,
    masks: dict[str, torch.Tensor] | None = None,
) -> dict[str, TableAdvantageGroup]:
    """Mean-aggregate trajectory advantage onto every visited block coordinate."""
    h = int(block_size)
    a = int(action_vocab_size)
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, torch.Tensor] = {}
    states: dict[str, object] = {}
    traj_ids: dict[str, list[str]] = {}
    block_counts: dict[str, int] = {}
    blocks_by_state: dict[str, list[tuple[int, ...]]] = {}
    candidate_blocks_by_state: dict[str, list[tuple[int, ...]]] = {}
    q_tables: dict[str, list[torch.Tensor]] = {}
    for traj_idx, trajectory in enumerate(trajectories):
        adv = float(trajectory_advantages[traj_idx].detach().cpu().item()) if traj_idx < trajectory_advantages.numel() else 0.0
        tid = str(trajectory.metadata.get("trajectory_id", f"{trajectory.task_id}:{traj_idx}"))
        for decision in trajectory.decisions:
            sid = str(decision.state_id)
            sums.setdefault(sid, torch.zeros(h, a, dtype=torch.float32))
            counts.setdefault(sid, torch.zeros(h, a, dtype=torch.float32))
            states.setdefault(sid, decision.state)
            traj_ids.setdefault(sid, [])
            block_counts[sid] = block_counts.get(sid, 0) + 1
            traj_ids[sid].append(tid)
            blocks_by_state.setdefault(sid, []).append(tuple(int(a) for a in decision.block_actions[:h]))
            if decision.candidate_blocks is not None:
                candidate_blocks_by_state.setdefault(sid, []).extend(
                    tuple(int(a) for a in block[:h]) for block in decision.candidate_blocks
                )
            if decision.q_table is not None:
                q_tables.setdefault(sid, []).append(torch.as_tensor(decision.q_table, dtype=torch.float32))
            for pos, action_id in enumerate(decision.block_actions[:h]):
                if 0 <= int(action_id) < a:
                    sums[sid][pos, int(action_id)] += adv
                    counts[sid][pos, int(action_id)] += 1.0
    out: dict[str, TableAdvantageGroup] = {}
    for sid, total in sums.items():
        count = counts[sid]
        if masks is not None and sid in masks:
            mask = masks[sid].to(dtype=torch.bool)
        elif sid in q_tables and q_tables[sid]:
            mask = torch.stack(q_tables[sid]).sum(dim=0) > 0
        else:
            mask = count > 0
        adv = torch.where(count > 0, total / count.clamp(min=1.0), torch.zeros_like(total))
        if sid in q_tables and q_tables[sid]:
            q_start = torch.stack(q_tables[sid]).mean(dim=0)
            q_start = _row_normalize(q_start, mask)
        else:
            q_start = _row_normalize(count, mask)
        candidate_blocks = list(dict.fromkeys(candidate_blocks_by_state.get(sid, [])))
        visited_blocks = list(dict.fromkeys(blocks_by_state.get(sid, [])))
        if not candidate_blocks:
            candidate_blocks = visited_blocks
        out[sid] = TableAdvantageGroup(
            state_id=sid,
            state=states[sid],
            advantages=adv.masked_fill(~mask, 0.0),
            counts=count,
            mask=mask,
            trajectory_ids=traj_ids[sid],
            block_count=block_counts[sid],
            blocks=visited_blocks,
            candidate_blocks=candidate_blocks,
            q_start=q_start,
        )
    return out


def _row_normalize(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = torch.where(mask, values.clamp(min=0.0), torch.zeros_like(values))
    row_sum = masked.sum(dim=1, keepdim=True)
    uniform = mask.float() / mask.sum(dim=1, keepdim=True).clamp(min=1).float()
    return torch.where(row_sum > 1e-12, masked / row_sum.clamp(min=1e-12), uniform)
