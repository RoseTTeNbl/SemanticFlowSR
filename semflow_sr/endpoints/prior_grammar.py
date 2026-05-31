"""Grammar / action-mask prior p0.

Uniform over the support but reweighted by an optional per-action grammar prior passed in
context['grammar_logits'] (e.g. favouring lower-cost operators). Defaults to uniform.
"""
from __future__ import annotations
import torch
from .base import PriorEndpoint
from ..utils.numerical import normalize_simplex


class GrammarPrior(PriorEndpoint):
    def build_p0(self, B, y, action_ids, context):
        A = action_ids.shape[0]
        logits = context.get("grammar_logits")
        if logits is None:
            return torch.full((A,), 1.0 / max(A, 1), device=B.device, dtype=B.dtype)
        return torch.softmax(logits.to(B.device).to(B.dtype), dim=-1)
