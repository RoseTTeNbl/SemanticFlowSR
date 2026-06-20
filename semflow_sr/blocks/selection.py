"""Executable-block selection from an H x A table."""
from __future__ import annotations

import torch


def block_logprob_scores(q_table: torch.Tensor, blocks: list[tuple[int, ...]]) -> torch.Tensor:
    q = torch.nan_to_num(q_table.float()).clamp(min=1e-12)
    scores = []
    for block in blocks:
        if len(block) > q.shape[0]:
            raise ValueError("block length exceeds table height")
        vals = [q[h, int(action)].log() for h, action in enumerate(block)]
        scores.append(torch.stack(vals).sum())
    return torch.stack(scores) if scores else torch.empty(0, dtype=q.dtype, device=q.device)


def select_executable_block_from_table(q_table: torch.Tensor, blocks: list[tuple[int, ...]]):
    """Return the highest-probability executable block, never row-wise independent argmax."""
    scores = block_logprob_scores(q_table, blocks)
    if scores.numel() == 0:
        raise ValueError("at least one executable block is required")
    return tuple(int(a) for a in blocks[int(scores.argmax().item())]), scores
