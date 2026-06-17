"""Global numerical constants and protected numeric helpers used across modules."""
from __future__ import annotations
import torch

EPS = 1e-12          # generic positive floor
SQRT_EPS = 1e-8      # floor used inside sqrt/sphere geometry
CLAMP_LARGE = 1e6    # saturate protected ops to avoid Inf


def safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Protected division: a / b with denominator floored away from zero."""
    denom = torch.where(b.abs() < eps, torch.full_like(b, eps), b)
    out = a / denom
    return torch.nan_to_num(out, nan=0.0, posinf=CLAMP_LARGE, neginf=-CLAMP_LARGE)


def clamp_finite(x: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf and saturate magnitude. Never let NaN/Inf propagate silently."""
    return torch.nan_to_num(x, nan=0.0, posinf=CLAMP_LARGE, neginf=-CLAMP_LARGE).clamp(-CLAMP_LARGE, CLAMP_LARGE)


def normalize_simplex(p: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    """Clamp non-negative and renormalize to sum 1 along `dim`."""
    p = p.clamp(min=0.0)
    s = p.sum(dim=dim, keepdim=True).clamp(min=eps)
    return p / s
