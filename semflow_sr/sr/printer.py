"""Symbolic printing and simplification via sympy."""
from __future__ import annotations
import sympy as sp
from .ast import Expr, to_sympy


def var_symbols(num_vars: int) -> list[sp.Symbol]:
    return [sp.Symbol(f"x{i}") for i in range(num_vars)]


def to_string(expr: Expr, num_vars: int, simplify: bool = False) -> str:
    syms = var_symbols(num_vars)
    s = to_sympy(expr, syms)
    if simplify:
        try:
            s = sp.simplify(s)
        except Exception:
            pass
    return str(s)


def simplify_sympy(expr: Expr, num_vars: int):
    syms = var_symbols(num_vars)
    s = to_sympy(expr, syms)
    try:
        return sp.simplify(s)
    except Exception:
        return s
