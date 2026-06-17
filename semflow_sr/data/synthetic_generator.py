"""Synthetic expression generator and register-trace sampler.

Generates random expressions over the reduced operator set, samples a probe (x,y), then
compiles to a register trace. Each trace step is a local target sample for
semantic-Fisher local update supervision.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
import torch

from ..sr.ast import Expr, eval_expr
from ..sr.ops import NAME_TO_ID, get_op, default_op_subset
from ..registers.compiler import compile_expr


@dataclass
class GenConfig:
    num_vars: int = 1
    max_depth: int = 4
    K: int = 8
    probe_size: int = 128
    x_range: tuple[float, float] = (-2.0, 2.0)
    ops: tuple[str, ...] = tuple(default_op_subset())
    p_leaf: float = 0.4


def _rand_expr(cfg: GenConfig, rng: random.Random, depth: int) -> Expr:
    if depth >= cfg.max_depth or (depth > 0 and rng.random() < cfg.p_leaf):
        if rng.random() < 0.8 or cfg.num_vars == 0:
            return Expr.var(rng.randrange(cfg.num_vars))
        return Expr.const(1.0)
    op_name = rng.choice(cfg.ops)
    op_id = NAME_TO_ID[op_name]
    arity = get_op(op_id).arity
    children = tuple(_rand_expr(cfg, rng, depth + 1) for _ in range(arity))
    # 一元算子不作用于纯常数子节点(与 valid_mask 约束一致, 避免退化子树如 exp(1))
    if arity == 1 and children[0].kind == "const":
        return children[0] if cfg.num_vars == 0 else Expr.var(rng.randrange(cfg.num_vars))
    return Expr.op(op_id, children)


def generate_expression(cfg: GenConfig, rng: random.Random) -> Expr:
    return _rand_expr(cfg, rng, 0)


def sample_probe_xy(expr: Expr, cfg: GenConfig, rng: random.Random,
                    device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(rng.randrange(2**31))
    lo, hi = cfg.x_range
    x = torch.rand(cfg.probe_size, cfg.num_vars, generator=g) * (hi - lo) + lo
    y = eval_expr(expr, x)
    return x.to(device), y.to(device)


def generate_trace_task(cfg: GenConfig, rng: random.Random, max_tries: int = 20):
    """Returns (expr, trace, x, y) or None if no compilable, finite task found."""
    allowed = [NAME_TO_ID[o] for o in cfg.ops]
    for _ in range(max_tries):
        expr = generate_expression(cfg, rng)
        trace = compile_expr(expr, cfg.num_vars, cfg.K, allowed)
        if trace is None or len(trace) == 0:
            continue
        x, y = sample_probe_xy(expr, cfg, rng)
        if not torch.isfinite(y).all() or y.std() < 1e-6:
            continue
        return expr, trace, x, y
    return None
