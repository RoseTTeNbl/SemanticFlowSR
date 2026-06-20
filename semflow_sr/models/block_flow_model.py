"""Block-policy H x A log-rate model."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BlockFlowModelConfig:
    d: int = 1
    K: int = 5
    block_size: int = 3
    action_vocab_size: int = 1
    hidden: int = 96


@dataclass
class BlockFlowOutput:
    lograte: torch.Tensor
    z_dot_pred: torch.Tensor


class BlockFlowModel(nn.Module):
    def __init__(self, cfg: BlockFlowModelConfig):
        super().__init__()
        self.cfg = cfg
        self.action_embed = nn.Embedding(cfg.action_vocab_size, cfg.hidden)
        self.pos_embed = nn.Embedding(cfg.block_size, cfg.hidden)
        self.scalar = nn.Linear(5, cfg.hidden)
        self.state = nn.Sequential(
            nn.Linear(6, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, 1),
        )

    def forward(
        self,
        B: torch.Tensor,
        y: torch.Tensor,
        q_lambda: torch.Tensor,
        lambda_value: torch.Tensor,
        mask: torch.Tensor,
        zeta: torch.Tensor | None = None,
    ) -> BlockFlowOutput:
        if B.dim() == 2:
            B = B.unsqueeze(0)
            y = y.unsqueeze(0)
            q_lambda = q_lambda.unsqueeze(0)
            mask = mask.unsqueeze(0)
            if zeta is not None:
                zeta = zeta.unsqueeze(0)
        device = q_lambda.device
        batch, h, a = q_lambda.shape
        action_ids = torch.arange(a, device=device).view(1, 1, a).expand(batch, h, a)
        pos_ids = torch.arange(h, device=device).view(1, h, 1).expand(batch, h, a)
        z_norm = torch.zeros_like(q_lambda)
        z_mean = torch.zeros_like(q_lambda)
        if zeta is not None and zeta.numel():
            z_norm = torch.nan_to_num(zeta.norm(dim=-1).to(device=device, dtype=q_lambda.dtype))
            z_mean = torch.nan_to_num(zeta.mean(dim=-1).to(device=device, dtype=q_lambda.dtype))
        lam = lambda_value.reshape(batch, 1, 1).to(device=device, dtype=q_lambda.dtype).expand(batch, h, a)
        scalars = torch.stack(
            [
                q_lambda.clamp(min=1e-12).log(),
                q_lambda,
                z_norm,
                z_mean,
                lam,
            ],
            dim=-1,
        )
        state_feat = _state_features(B, y).to(device=device, dtype=q_lambda.dtype)
        state_tok = self.state(state_feat).view(batch, 1, 1, -1)
        tokens = (
            self.action_embed(action_ids)
            + self.pos_embed(pos_ids)
            + self.scalar(scalars)
            + state_tok
        )
        raw = self.out(tokens).squeeze(-1)
        raw = raw.masked_fill(~mask, 0.0)
        mean = (q_lambda * raw).sum(dim=-1, keepdim=True)
        lograte = (raw - mean).masked_fill(~mask, 0.0)
        z_dot = 0.5 * q_lambda.clamp(min=1e-12).sqrt() * lograte
        return BlockFlowOutput(lograte=lograte, z_dot_pred=z_dot)


def _state_features(B: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    yy = torch.nan_to_num(y.float())
    bb = torch.nan_to_num(B.float())
    return torch.stack(
        [
            yy.mean(dim=1),
            yy.std(dim=1, unbiased=False),
            bb.mean(dim=(1, 2)),
            bb.std(dim=(1, 2), unbiased=False),
            bb.abs().amax(dim=(1, 2)),
            torch.ones(B.shape[0], device=B.device, dtype=torch.float32),
        ],
        dim=-1,
    )
