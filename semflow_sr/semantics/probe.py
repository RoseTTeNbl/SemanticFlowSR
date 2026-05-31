"""Probe batch: all semantics are computed on a probe of (x, y)."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class ProbeBatch:
    x: torch.Tensor   # [m, d]
    y: torch.Tensor   # [m]

    def to(self, device):
        return ProbeBatch(self.x.to(device), self.y.to(device))


def sample_probe(X: torch.Tensor, y: torch.Tensor, m: int | None = None,
                 mode: str = "random", generator: torch.Generator | None = None) -> ProbeBatch:
    """mode: 'full' | 'random' | 'debug' (first m rows, deterministic)."""
    n = X.shape[0]
    if mode == "full" or m is None or m >= n:
        return ProbeBatch(X, y)
    if mode == "debug":
        idx = torch.arange(m)
    else:
        idx = torch.randperm(n, generator=generator)[:m]
    return ProbeBatch(X[idx], y[idx])
