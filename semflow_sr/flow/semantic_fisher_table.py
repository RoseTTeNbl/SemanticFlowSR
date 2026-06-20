"""Semantic-Fisher teacher and integration on masked H x A block-policy tables."""
from __future__ import annotations

import torch

from ..utils.numerical import EPS


def semantic_fisher_table_lograte(
    q: torch.Tensor,
    advantages: torch.Tensor,
    zeta: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 1.0,
    gamma: float = 0.1,
    gram_rank: int | None = None,
) -> torch.Tensor:
    """Solve the row-constrained semantic-Fisher log-rate teacher.

    ``q`` and ``advantages`` are shaped ``[H, A]``. ``zeta[h, a]`` is the
    table-coordinate semantic effect factor for coordinate ``(h, a)``.
    Conservation is enforced separately for each block position row.
    """
    q = torch.nan_to_num(q.float()).clamp(min=0.0)
    advantages = torch.nan_to_num(advantages.to(device=q.device, dtype=q.dtype))
    zeta = torch.nan_to_num(zeta.to(device=q.device, dtype=q.dtype))
    mask = mask.to(device=q.device, dtype=torch.bool)
    if q.shape != advantages.shape or q.shape != mask.shape:
        raise ValueError("q, advantages and mask must all have shape [H, A]")
    if zeta.shape[:2] != q.shape:
        raise ValueError("zeta must have shape [H, A, R]")
    row_sums = (q * mask).sum(dim=1, keepdim=True).clamp(min=EPS)
    q = torch.where(mask, q / row_sums, torch.zeros_like(q))

    valid = mask.reshape(-1)
    if not bool(valid.any()):
        return torch.zeros_like(q)
    h, a = q.shape
    p = q.reshape(-1)[valid]
    adv = advantages.reshape(-1)[valid]
    factors = zeta.reshape(h * a, -1)[valid]
    if gram_rank is not None and int(gram_rank) > 0 and factors.numel() and factors.shape[1] > int(gram_rank):
        factors = _compress_factors(factors, int(gram_rank))
    row_ids_all = torch.arange(h, device=q.device).unsqueeze(1).expand(h, a).reshape(-1)
    row_ids = row_ids_all[valid]
    row_count = h
    L = torch.zeros(p.numel(), row_count, device=q.device, dtype=q.dtype)
    L[torch.arange(p.numel(), device=q.device), row_ids] = 1.0

    def solve_inv(rhs: torch.Tensor) -> torch.Tensor:
        return _solve_inv_low_rank(rhs, p, factors, float(gamma))

    SA = solve_inv(adv)
    SL = solve_inv(L)
    weighted = p.unsqueeze(0) * L.transpose(0, 1)
    G = weighted @ SL
    r = weighted @ SA
    try:
        nu = torch.linalg.solve(G + 1e-8 * torch.eye(row_count, device=q.device, dtype=q.dtype), -r)
    except RuntimeError:
        nu = torch.linalg.pinv(G) @ (-r)
    w_valid = float(beta) * solve_inv(adv + L @ nu)
    out = torch.zeros(h * a, device=q.device, dtype=q.dtype)
    out[valid] = torch.nan_to_num(w_valid)
    return out.reshape(h, a).masked_fill(~mask, 0.0)


def semantic_fisher_table_sphere_step(
    q: torch.Tensor,
    w: torch.Tensor,
    mask: torch.Tensor,
    *,
    dt: float = 1.0,
) -> torch.Tensor:
    """Apply one row-wise square-root retraction step on the masked table."""
    q = torch.nan_to_num(q.float()).clamp(min=0.0)
    w = torch.nan_to_num(w.to(device=q.device, dtype=q.dtype))
    mask = mask.to(device=q.device, dtype=torch.bool)
    z = q.clamp(min=EPS).sqrt()
    z = torch.where(mask, z, torch.zeros_like(z))
    z_next = z + 0.5 * float(dt) * z * w
    z_next = torch.where(mask, z_next.clamp(min=0.0), torch.zeros_like(z_next))
    q_next = z_next.square()
    row_sums = q_next.sum(dim=1, keepdim=True)
    fallback = _uniform_mask(mask, dtype=q.dtype)
    return torch.where(row_sums > EPS, q_next / row_sums.clamp(min=EPS), fallback)


def semantic_fisher_table_integrate(
    q: torch.Tensor,
    w: torch.Tensor,
    mask: torch.Tensor,
    *,
    steps: int = 1,
    dt: float = 1.0,
) -> torch.Tensor:
    """Integrate a predicted table log-rate with repeated masked sphere steps."""
    out = q
    n = max(int(steps), 1)
    step_dt = float(dt) / n
    for _ in range(n):
        out = semantic_fisher_table_sphere_step(out, w, mask, dt=step_dt)
    return out


def _solve_inv_low_rank(rhs: torch.Tensor, p: torch.Tensor, factors: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma == 0.0 or factors.numel() == 0 or factors.shape[1] == 0:
        return rhs
    # Woodbury for (I + gamma Z Z^T P)^-1 rhs.
    z = factors
    pz = p.unsqueeze(1) * z
    small = torch.eye(z.shape[1], device=z.device, dtype=z.dtype) + gamma * (z.transpose(0, 1) @ pz)
    right = z.transpose(0, 1) @ rhs
    try:
        inner = torch.linalg.solve(small, right)
    except RuntimeError:
        inner = torch.linalg.pinv(small) @ right
    return rhs - gamma * (z @ inner)


def _compress_factors(factors: torch.Tensor, rank: int) -> torch.Tensor:
    if rank >= min(factors.shape):
        return factors
    try:
        u, s, _ = torch.linalg.svd(factors, full_matrices=False)
        return u[:, :rank] * s[:rank].unsqueeze(0)
    except RuntimeError:
        return factors[:, :rank]


def _uniform_mask(mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return mask.to(dtype=dtype) / counts.to(dtype=dtype)
