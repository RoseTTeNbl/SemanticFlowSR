"""Action encoder: per-action features + dynamic signals (energy, weight, p_λ),
cross-attending to register and global context.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ..actions.action_features import ACTION_FEATURE_DIM, SEMANTIC_ACTION_FEATURE_DIM


class ActionEncoder(nn.Module):
    def __init__(self, hidden: int = 128, heads: int = 4):
        super().__init__()
        # action_feats + semantic_stats + [energy, weight, p_lambda]
        self.proj = nn.Linear(ACTION_FEATURE_DIM + SEMANTIC_ACTION_FEATURE_DIM + 3, hidden)
        self.reg_attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(hidden)
        self.rel_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden + hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, feats, energies, weights, p_lambda, reg_tokens, ctx,
                semantic_stats=None, gram=None, key_padding_mask=None):
        # feats:[bsz,A,F] semantic_stats:[bsz,A,S] energies/weights/p_lambda:[bsz,A]
        dyn = torch.stack([energies, weights, p_lambda], dim=-1)
        if semantic_stats is None:
            semantic_stats = torch.zeros(
                feats.shape[0], feats.shape[1], SEMANTIC_ACTION_FEATURE_DIM,
                device=feats.device, dtype=feats.dtype,
            )
        h = self.proj(torch.cat([feats, semantic_stats, dyn], dim=-1))  # [bsz,A,H]
        a, _ = self.reg_attn(h, reg_tokens, reg_tokens)
        h = self.norm(h + a)
        if gram is not None:
            rel = _relation_mix(h, gram, key_padding_mask)
            h = self.rel_norm(h + rel)
        g = ctx.unsqueeze(1).expand(-1, h.shape[1], -1)
        h = self.norm2(h + self.ffn(torch.cat([h, g], dim=-1)))
        return h                                               # [bsz,A,H]


def _relation_mix(h: torch.Tensor, gram: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
    g = gram.to(device=h.device, dtype=h.dtype)
    if key_padding_mask is not None:
        valid = (~key_padding_mask).to(dtype=h.dtype)
        g = g * valid.unsqueeze(-1) * valid.unsqueeze(-2)
    denom = g.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return (g / denom) @ h
