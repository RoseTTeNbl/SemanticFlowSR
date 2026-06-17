"""Operator registry shared by numeric (semantic) and symbolic executors.

Each operator has: name, arity, a protected numeric fn (tensor->tensor), an op cost
C_op(a) used in the action energy, and a sympy builder for symbolic printing/eval.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import sympy as sp

from . import protected_ops as P


@dataclass(frozen=True)
class Operator:
    name: str
    arity: int                       # 1 or 2
    fn: Callable                     # protected numeric tensor op
    cost: float                      # C_op(a) complexity weight
    sympy_fn: Callable               # builds a sympy expression


def _sym_div(a, b): return a / b
def _sym_sqrt(a): return sp.sqrt(sp.Abs(a))
def _sym_log(a): return sp.log(sp.Abs(a))


# Order defines op_id. Binary ops first, then unary.
_OPERATORS: list[Operator] = [
    Operator("add", 2, P.p_add, 1.0, lambda a, b: a + b),
    Operator("sub", 2, P.p_sub, 1.0, lambda a, b: a - b),
    Operator("mul", 2, P.p_mul, 1.0, lambda a, b: a * b),
    Operator("protected_div", 2, P.p_div, 1.5, _sym_div),
    Operator("neg", 1, P.p_neg, 1.0, lambda a: -a),
    Operator("sin", 1, P.p_sin, 2.0, sp.sin),
    Operator("cos", 1, P.p_cos, 2.0, sp.cos),
    Operator("square", 1, P.p_square, 1.5, lambda a: a ** 2),
    Operator("cube", 1, P.p_cube, 2.0, lambda a: a ** 3),
    Operator("protected_log", 1, P.p_log, 2.0, _sym_log),
    Operator("protected_sqrt", 1, P.p_sqrt, 2.0, _sym_sqrt),
    Operator("exp", 1, P.p_exp, 2.0, sp.exp),
]

OPERATORS: tuple[Operator, ...] = tuple(_OPERATORS)
NAME_TO_ID: dict[str, int] = {op.name: i for i, op in enumerate(OPERATORS)}
N_OPS: int = len(OPERATORS)


def get_op(op_id: int) -> Operator:
    return OPERATORS[op_id]


def op_cost(op_id: int) -> float:
    return OPERATORS[op_id].cost


def default_op_subset() -> list[str]:
    """Default operator set.

    add/sub are kept because linear readout only combines terminal columns; nonlinear
    compositions such as sin(x+x^2) require an intermediate add/sub register.
    """
    return ["add", "sub", "mul", "protected_div", "sin", "cos", "square", "cube",
            "exp", "protected_log", "protected_sqrt"]
