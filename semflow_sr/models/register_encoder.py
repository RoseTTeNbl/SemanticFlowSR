"""Register encoder: per-column statistics, cross-attending to row tokens."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..utils.numerical import EPS

REGISTER_STAT_DIM = 8   # mean,std,norm,min,max,corr_y,corr_res,active


def register_stats(B, y, residual, active):
    """B:[bsz,m,K] -> [bsz,K,REGISTER_STAT_DIM]."""
    mean = B.mean(1); std = B.std(1); norm = B.norm(dim=1); mn = B.min(1).values; mx = B.max(1).values

    def corr(t):  # t:[bsz,m] correlated against each column of B
        tc = t - t.mean(1, keepdim=True)                  # [bsz,m]
        Bc = B - B.mean(1, keepdim=True)                  # [bsz,m,K]
        num = (Bc * tc.unsqueeze(-1)).sum(1)              # [bsz,K]
        den = (Bc.norm(dim=1) * tc.norm(dim=1, keepdim=True)).clamp(min=EPS)  # [bsz,K]
        return num / den
    cy = corr(y); cr = corr(residual)
    act = active if active is not None else torch.ones_like(mean)
    return torch.stack([mean, std, norm, mn, mx, cy, cr, act], dim=-1)


class RegisterEncoder(nn.Module):
    def __init__(self, hidden: int = 128, heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(REGISTER_STAT_DIM, hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, B, y, residual, active, row_tokens):
        stats = register_stats(B, y, residual, active)        # [bsz,K,S]
        h = self.proj(stats)
        a, _ = self.attn(h, row_tokens, row_tokens)
        h = self.norm(h + a)
        return h                                               # [bsz,K,hidden]
