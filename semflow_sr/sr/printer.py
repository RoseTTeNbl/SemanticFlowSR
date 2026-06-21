"""Symbolic printing and simplification via sympy."""
from __future__ import annotations
import sympy as sp
from .ast import Expr, to_sympy
from .ops import get_op


def var_symbols(num_vars: int) -> list[sp.Symbol]:
    return [sp.Symbol(f"x{i}") for i in range(num_vars)]


def to_string(expr: Expr, num_vars: int, simplify: bool = False) -> str:
    syms = var_symbols(num_vars)
    try:
        s = to_sympy(expr, syms)
    except Exception:
        return _structural_string(expr)
    if simplify:
        try:
            s = sp.simplify(s)
        except Exception:
            pass
    return str(s)


def simplify_sympy(expr: Expr, num_vars: int):
    syms = var_symbols(num_vars)
    try:
        s = to_sympy(expr, syms)
    except Exception:
        return sp.Symbol(_structural_string(expr))
    try:
        return sp.simplify(s)
    except Exception:
        return s


def _structural_string(expr: Expr) -> str:
    if expr.kind == "var":
        return f"x{expr.var_index}"
    if expr.kind == "const":
        return f"{float(expr.value):.6g}"
    op = get_op(int(expr.op_id)).name
    args = ", ".join(_structural_string(child) for child in expr.children)
    return f"{op}({args})"
