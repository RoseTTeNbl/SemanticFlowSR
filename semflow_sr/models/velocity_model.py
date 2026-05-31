"""VelocityOutput container and tangent-projecting velocity head."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class VelocityOutput:
    v_pred: torch.Tensor       # [bsz,A], masked-tangent (sum≈0 over support)


class VelocityHead(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, action_tokens, mask=None) -> VelocityOutput:
        raw = self.out(action_tokens).squeeze(-1)              # [bsz,A]
        if mask is not None:
            raw = raw.masked_fill(~mask, 0.0)
            n = mask.sum(-1, keepdim=True).clamp(min=1)
            mean = raw.sum(-1, keepdim=True) / n
            v = (raw - mean) * mask                            # tangent over support
        else:
            v = raw - raw.mean(-1, keepdim=True)
        return VelocityOutput(v_pred=v)
