"""Block candidate feature helpers for risk-flow block supports."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .action_features import action_features
from .action_space import ActionSpace
from ..registers.state import RegisterState


@dataclass
class BlockFeatureEncoder:
    space: ActionSpace
    max_block_size: int = 5

    def encode(self, state: RegisterState, blocks: list[list[int] | tuple[int, ...]]) -> torch.Tensor:
        rows = []
        device = state.active.device
        for block in blocks:
            actions = [int(a) for a in list(block)[: max(int(self.max_block_size), 1)]]
            if actions:
                ids = torch.tensor(actions, device=device, dtype=torch.long)
                feats = action_features(self.space, state, ids)
                pooled = feats.mean(dim=0)
                first = feats[0]
                last = feats[-1]
            else:
                dim = action_features(self.space, state, torch.tensor([0], device=device)).shape[-1]
                pooled = torch.zeros(dim, device=device)
                first = torch.zeros(dim, device=device)
                last = torch.zeros(dim, device=device)
            length = torch.tensor(
                [len(actions) / max(float(self.max_block_size), 1.0)],
                device=device,
                dtype=pooled.dtype,
            )
            rows.append(torch.cat([pooled, first, last, length], dim=0))
        if not rows:
            return torch.empty(0, 0, device=device)
        return torch.stack(rows, dim=0)
