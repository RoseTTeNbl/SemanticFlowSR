"""Lightweight expression AST for register contents.

A register holds an `Expr`: either a variable leaf, a constant leaf, or an operator
node referencing child Exprs. Numeric evaluation uses the protected ops; sympy
conversion is used for printing / simplification / ground-truth checking.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import sympy as sp
import torch

from .ops import OPERATORS, NAME_TO_ID, get_op


@dataclass(frozen=True)
class Expr:
    kind: str                       # "var" | "const" | "op"
    var_index: Optional[int] = None
    value: Optional[float] = None
    op_id: Optional[int] = None
    children: tuple["Expr", ...] = field(default_factory=tuple)

    # --- constructors ---
    @staticmethod
    def var(i: int) -> "Expr": return Expr("var", var_index=i)

    @staticmethod
    def const(v: float) -> "Expr": return Expr("const", value=float(v))

    @staticmethod
    def op(op_id: int, children: tuple["Expr", ...]) -> "Expr":
        return Expr("op", op_id=op_id, children=tuple(children))

    # --- properties ---
    @property
    def depth(self) -> int:
        if self.kind != "op":
            return 0
        return 1 + max(c.depth for c in self.children)

    @property
    def complexity(self) -> int:
        if self.kind != "op":
            return 1
        return 1 + sum(c.complexity for c in self.children)


def eval_expr(expr: Expr, X: torch.Tensor) -> torch.Tensor:
    """Numeric (protected) evaluation. X: [m, d] -> [m]."""
    if expr.kind == "var":
        return X[:, expr.var_index]
    if expr.kind == "const":
        return torch.full((X.shape[0],), float(expr.value), dtype=X.dtype, device=X.device)
    op = get_op(expr.op_id)
    args = [eval_expr(c, X) for c in expr.children]
    return op.fn(*args)


def to_sympy(expr: Expr, var_symbols: list[sp.Symbol]) -> sp.Expr:
    if expr.kind == "var":
        return var_symbols[expr.var_index]
    if expr.kind == "const":
        return sp.Float(expr.value)
    op = get_op(expr.op_id)
    args = [to_sympy(c, var_symbols) for c in expr.children]
    return op.sympy_fn(*args)
