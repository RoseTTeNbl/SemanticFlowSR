"""Batched numeric evaluator for register states (delegates to ast.eval_expr)."""
from __future__ import annotations
import torch
from .ast import Expr, eval_expr


def evaluate_exprs(exprs: list[Expr], X: torch.Tensor) -> torch.Tensor:
    """Evaluate K expressions on probe X:[m,d] -> semantic matrix B:[m,K]."""
    cols = [eval_expr(e, X) for e in exprs]
    return torch.stack(cols, dim=1)
