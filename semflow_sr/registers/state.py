"""Register state and semantic state dataclasses.

A fixed-register symbolic program holds K register expressions. Actions overwrite a
register with op(read_1, read_2). Metadata (active/depth/complexity/age) feeds the
register encoder and action masks.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch

from ..sr.ast import Expr


@dataclass
class RegisterState:
    exprs: list[Expr]
    active: torch.Tensor       # [K] bool/float: register has been written / usable
    depth: torch.Tensor       # [K] long
    complexity: torch.Tensor  # [K] long
    age: torch.Tensor         # [K] long: steps since last write
    num_vars: int

    @property
    def K(self) -> int:
        return len(self.exprs)

    def clone(self) -> "RegisterState":
        return RegisterState(
            exprs=list(self.exprs),
            active=self.active.clone(),
            depth=self.depth.clone(),
            complexity=self.complexity.clone(),
            age=self.age.clone(),
            num_vars=self.num_vars,
        )


def init_register_state(num_vars: int, K: int, device="cpu") -> RegisterState:
    """Seed first `num_vars` registers with input variables, register[num_vars] with
    constant 1.0, remaining registers as inactive zeros (constant 0)."""
    assert K >= num_vars + 1, "K must hold all variables plus at least one constant"
    exprs: list[Expr] = []
    active = torch.zeros(K)
    depth = torch.zeros(K, dtype=torch.long)
    complexity = torch.ones(K, dtype=torch.long)
    age = torch.zeros(K, dtype=torch.long)
    for k in range(K):
        if k < num_vars:
            exprs.append(Expr.var(k)); active[k] = 1.0
        elif k == num_vars:
            exprs.append(Expr.const(1.0)); active[k] = 1.0
        else:
            exprs.append(Expr.const(0.0)); active[k] = 0.0
    return RegisterState(exprs, active.to(device), depth.to(device), complexity.to(device), age.to(device), num_vars)


@dataclass
class SemanticState:
    B: torch.Tensor             # [m, K]
    y: torch.Tensor             # [m]
    projection_cache: dict = field(default_factory=dict)
