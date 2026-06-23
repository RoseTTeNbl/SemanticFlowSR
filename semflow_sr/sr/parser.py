"""Parse a string formula (sympy syntax) into an Expr over x0..x{d-1}.

Used to materialize formula benchmarks (Nguyen etc.). Supports the operators in
the registry; constants become const leaves.
"""
from __future__ import annotations
import sympy as sp
from .ast import Expr
from .ops import NAME_TO_ID

_FUNC_MAP = {
    "sin": "sin", "cos": "cos", "exp": "exp",
    "log": "protected_log", "sqrt": "protected_sqrt",
}


def parse_formula(formula: str, variables: list[str]) -> Expr:
    local = {v: sp.Symbol(v) for v in variables}
    s = sp.sympify(formula, locals=local)
    var_idx = {sp.Symbol(v): i for i, v in enumerate(variables)}
    return _from_sympy(sp.expand(s) if False else s, var_idx)


def _from_sympy(s: sp.Expr, var_idx: dict) -> Expr:
    if s.is_Symbol:
        return Expr.var(var_idx[s])
    if s.is_Number:
        return Expr.const(float(s))
    if s.is_Add:
        return _from_add(s, var_idx)
    if s.is_Mul:
        return _from_mul(s, var_idx)
    if s.is_Pow:
        base = _from_sympy(s.base, var_idx)
        exp = s.exp
        if exp == 2:
            return Expr.op(NAME_TO_ID["square"], (base,))
        if exp == 3:
            return Expr.op(NAME_TO_ID["cube"], (base,))
        if exp == sp.Rational(1, 2):
            return Expr.op(NAME_TO_ID["protected_sqrt"], (base,))
        # integer powers -> repeated mul
        if exp.is_Integer and int(exp) > 0:
            node = base
            for _ in range(int(exp) - 1):
                node = Expr.op(NAME_TO_ID["mul"], (node, base))
            return node
        if exp.is_Integer and int(exp) < 0:
            denom = _positive_integer_power(base, -int(exp))
            return Expr.op(NAME_TO_ID["protected_div"], (Expr.const(1.0), denom))
        raise ValueError(f"Unsupported power exponent: {exp}")
    if isinstance(s, sp.Function):
        fname = type(s).__name__
        if fname == "Abs":
            return _from_sympy(s.args[0], var_idx)
        if fname in _FUNC_MAP:
            return Expr.op(NAME_TO_ID[_FUNC_MAP[fname]], (_from_sympy(s.args[0], var_idx),))
    raise ValueError(f"Cannot parse sympy node: {s!r}")


def _fold(op_name: str, children: list[Expr]) -> Expr:
    op_id = NAME_TO_ID[op_name]
    node = children[0]
    for c in children[1:]:
        node = Expr.op(op_id, (node, c))
    return node


def _from_add(s: sp.Expr, var_idx: dict) -> Expr:
    positives: list[Expr] = []
    negatives: list[Expr] = []
    for arg in s.args:
        stripped = _strip_unary_negative(arg)
        if stripped is None:
            positives.append(_from_sympy(arg, var_idx))
        else:
            negatives.append(_from_sympy(stripped, var_idx))
    positive = _fold("add", positives) if positives else Expr.const(0.0)
    if not negatives:
        return positive
    negative = _fold("add", negatives)
    return Expr.op(NAME_TO_ID["sub"], (positive, negative))


def _from_mul(s: sp.Expr, var_idx: dict) -> Expr:
    numerators: list[Expr] = []
    denominators: list[Expr] = []
    for arg in s.args:
        if arg.is_Pow and arg.exp.is_Integer and int(arg.exp) < 0:
            base = _from_sympy(arg.base, var_idx)
            denominators.append(_positive_integer_power(base, -int(arg.exp)))
        else:
            numerators.append(_from_sympy(arg, var_idx))
    numerator = _fold("mul", numerators) if numerators else Expr.const(1.0)
    if not denominators:
        return numerator
    denominator = _fold("mul", denominators)
    return Expr.op(NAME_TO_ID["protected_div"], (numerator, denominator))


def _strip_unary_negative(s: sp.Expr) -> sp.Expr | None:
    coeff, rest = s.as_coeff_Mul()
    if coeff == -1:
        return rest
    return None


def _positive_integer_power(base: Expr, exponent: int) -> Expr:
    if int(exponent) <= 0:
        return Expr.const(1.0)
    if int(exponent) == 1:
        return base
    if int(exponent) == 2:
        return Expr.op(NAME_TO_ID["square"], (base,))
    if int(exponent) == 3:
        return Expr.op(NAME_TO_ID["cube"], (base,))
    node = base
    for _ in range(int(exponent) - 1):
        node = Expr.op(NAME_TO_ID["mul"], (node, base))
    return node
