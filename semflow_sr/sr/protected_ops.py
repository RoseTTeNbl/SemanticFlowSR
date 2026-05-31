"""Protected numeric operators.

All operators map real tensors -> real tensors and must NEVER emit NaN/Inf.
Protection follows standard SR conventions (gplearn / DSR style): floor denominators,
clamp domains for log/sqrt, saturate exp.
"""
from __future__ import annotations
import torch
from ..utils.numerical import clamp_finite, CLAMP_LARGE

_DIV_EPS = 1e-6
_LOG_EPS = 1e-6
_EXP_CLAMP = 30.0  # exp(30) ~ 1e13, kept below CLAMP_LARGE^2


def p_add(a, b): return clamp_finite(a + b)
def p_sub(a, b): return clamp_finite(a - b)
def p_mul(a, b): return clamp_finite(a * b)


def p_div(a, b):
    """Protected division: |b|<eps -> 1.0 (gplearn convention)."""
    safe_b = torch.where(b.abs() < _DIV_EPS, torch.ones_like(b), b)
    out = torch.where(b.abs() < _DIV_EPS, torch.ones_like(a), a / safe_b)
    return clamp_finite(out)


def p_neg(a): return clamp_finite(-a)
def p_square(a): return clamp_finite(a * a)
def p_cube(a): return clamp_finite(a * a * a)
def p_sin(a): return clamp_finite(torch.sin(a))
def p_cos(a): return clamp_finite(torch.cos(a))
def p_exp(a): return clamp_finite(torch.exp(a.clamp(-_EXP_CLAMP, _EXP_CLAMP)))
def p_log(a): return clamp_finite(torch.log(a.abs().clamp(min=_LOG_EPS)))
def p_sqrt(a): return clamp_finite(torch.sqrt(a.abs().clamp(min=0.0) + 0.0))
