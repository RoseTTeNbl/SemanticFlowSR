"""Semantic Transformer: assembles row/register/action encoders + velocity head.

Predicts a tangent velocity v_θ(p_λ, B, y, λ) on the action simplex with Σ v = 0.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

from .row_encoder import RowEncoder
from .register_encoder import RegisterEncoder
from .action_encoder import ActionEncoder
from .velocity_model import VelocityHead, VelocityOutput


@dataclass
class SemanticTransformerConfig:
    d: int = 1
    K: int = 8
    hidden: int = 128
    row_layers: int = 2
    heads: int = 4


class SemanticTransformer(nn.Module):
    def __init__(self, cfg: SemanticTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.row = RowEncoder(cfg.d, cfg.K, cfg.hidden, cfg.row_layers, cfg.heads)
        self.reg = RegisterEncoder(cfg.hidden, cfg.heads)
        self.act = ActionEncoder(cfg.hidden, cfg.heads)
        self.lambda_proj = nn.Linear(1, cfg.hidden)
        self.head = VelocityHead(cfg.hidden)

    def forward(self, x, y, B, p_lambda, lambda_value, action_feats, energies, weights,
                action_mask=None, active=None) -> VelocityOutput:
        residual = y - self._project_y(B, y)
        row_tokens, ctx = self.row(x, y, B, residual)
        reg_tokens = self.reg(B, y, residual, active, row_tokens)
        lam = lambda_value.reshape(-1, 1).to(B.dtype)
        ctx = ctx + self.lambda_proj(lam)
        kpm = (~action_mask) if action_mask is not None else None
        act_tokens = self.act(action_feats, energies, weights, p_lambda, reg_tokens, ctx, kpm)
        return self.head(act_tokens, action_mask)

    @staticmethod
    def _project_y(B, y, rho: float = 1e-3):
        # ridge residual for the row feature (kept lightweight, batched K-space solve)
        G = B.transpose(-1, -2) @ B
        K = G.shape[-1]
        I = torch.eye(K, device=B.device, dtype=B.dtype).expand_as(G)
        coeff = torch.linalg.solve(G + rho * I, (B.transpose(-1, -2) @ y.unsqueeze(-1)))
        return (B @ coeff).squeeze(-1)
