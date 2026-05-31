"""Action encoder: per-action features + dynamic signals (energy, weight, p_λ),
cross-attending to register and global context.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ..actions.action_features import ACTION_FEATURE_DIM


class ActionEncoder(nn.Module):
    def __init__(self, hidden: int = 128, heads: int = 4):
        super().__init__()
        # action_feats + [energy, weight, p_lambda]
        self.proj = nn.Linear(ACTION_FEATURE_DIM + 3, hidden)
        self.reg_attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden + hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, feats, energies, weights, p_lambda, reg_tokens, ctx, key_padding_mask=None):
        # feats:[bsz,A,F] energies/weights/p_lambda:[bsz,A] reg_tokens:[bsz,K,H] ctx:[bsz,H]
        dyn = torch.stack([energies, weights, p_lambda], dim=-1)
        h = self.proj(torch.cat([feats, dyn], dim=-1))         # [bsz,A,H]
        a, _ = self.reg_attn(h, reg_tokens, reg_tokens)
        h = self.norm(h + a)
        g = ctx.unsqueeze(1).expand(-1, h.shape[1], -1)
        h = self.norm2(h + self.ffn(torch.cat([h, g], dim=-1)))
        return h                                               # [bsz,A,H]
