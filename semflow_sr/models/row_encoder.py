"""Row encoder: permutation-invariant Transformer over probe rows.

Each row: concat(x_i, y_i, B[i,:], residual_i). No positional encoding. Returns a global
context vector (mean pool) and per-row tokens.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class RowEncoder(nn.Module):
    def __init__(self, d: int, K: int, hidden: int = 128, layers: int = 2, heads: int = 4):
        super().__init__()
        in_dim = d + 1 + K + 1            # x, y, B row, residual
        self.proj = nn.Linear(in_dim, hidden)
        enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True, dropout=0.0)
        self.tf = nn.TransformerEncoder(enc, layers)

    def forward(self, x, y, B, residual):
        # x:[bsz,m,d] y:[bsz,m] B:[bsz,m,K] residual:[bsz,m]
        tok = torch.cat([x, y.unsqueeze(-1), B, residual.unsqueeze(-1)], dim=-1)
        h = self.proj(tok)
        h = self.tf(h)
        ctx = h.mean(dim=1)              # [bsz,hidden]
        return h, ctx
