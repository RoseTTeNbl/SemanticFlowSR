"""Semantic-Fisher pullback flow primitives.

The flow is parameterized by a log-rate field ``w`` over the support, with simplex
velocity ``p_dot = p * w`` and square-root sphere velocity ``z_dot = 0.5 * z * w``.
For the exact semantic-Fisher target, ``w`` solves

    (I + gamma K P) w = beta (A + nu 1),

where ``P = diag(p)`` and ``nu`` enforces ``p^T w = 0``.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from ..flow.natural_path import smooth_simplex
from ..utils.numerical import EPS


@dataclass
class SemanticFisherTeacherPath:
    policies: list[torch.Tensor]
    logrates: list[torch.Tensor]
    sphere_velocities: list[torch.Tensor]
    dt: float


def semantic_fisher_lograte(
    p: torch.Tensor,
    advantage: torch.Tensor,
    gram: torch.Tensor,
    beta: torch.Tensor | float,
    gamma: torch.Tensor | float,
    gram_rank: int | None = None,
    gram_factors: torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Solve the semantic-Fisher linear system for the log-rate ``w``."""
    p = smooth_simplex(p, eps=eps)
    advantage = torch.nan_to_num(advantage.to(device=p.device, dtype=p.dtype))
    gram = torch.nan_to_num(gram.to(device=p.device, dtype=p.dtype))
    if gram_factors is not None:
        gram_factors = torch.nan_to_num(gram_factors.to(device=p.device, dtype=p.dtype))
    beta_t = torch.as_tensor(beta, device=p.device, dtype=p.dtype)
    gamma_t = torch.as_tensor(gamma, device=p.device, dtype=p.dtype)

    A = p.shape[-1]
    while gamma_t.dim() < p.dim():
        gamma_t = gamma_t.unsqueeze(-1)
    ones = torch.ones_like(advantage)

    rhs_adv = _solve_pullback_system(
        gram, p, gamma_t, advantage.unsqueeze(-1), gram_rank, gram_factors, eps
    ).squeeze(-1)
    rhs_ones = _solve_pullback_system(
        gram, p, gamma_t, ones.unsqueeze(-1), gram_rank, gram_factors, eps
    ).squeeze(-1)
    numerator = (p * rhs_adv).sum(dim=-1, keepdim=True)
    denominator = (p * rhs_ones).sum(dim=-1, keepdim=True).clamp_min(eps)
    nu = -numerator / denominator

    while beta_t.dim() < p.dim():
        beta_t = beta_t.unsqueeze(-1)
    w = beta_t * _solve_pullback_system(
        gram, p, gamma_t, (advantage + nu * ones).unsqueeze(-1), gram_rank, gram_factors, eps
    ).squeeze(-1)
    correction = (p * w).sum(dim=-1, keepdim=True)
    w = w - correction
    return torch.nan_to_num(w)


def semantic_fisher_simplex_velocity(p: torch.Tensor, lograte: torch.Tensor) -> torch.Tensor:
    """Map a log-rate ``w`` to a simplex tangent velocity ``p_dot``."""
    p = p.to(device=lograte.device, dtype=lograte.dtype)
    return p * lograte


def semantic_fisher_sphere_velocity(z: torch.Tensor, lograte: torch.Tensor) -> torch.Tensor:
    """Map a log-rate ``w`` to a square-root sphere tangent velocity ``z_dot``."""
    z = z.to(device=lograte.device, dtype=lograte.dtype)
    zdot = 0.5 * z * lograte
    return zdot - (zdot * z).sum(dim=-1, keepdim=True) * z


def semantic_fisher_sphere_step(
    p: torch.Tensor,
    lograte: torch.Tensor,
    dt: float = 1.0,
    eps: float = EPS,
) -> torch.Tensor:
    """Take one positive sphere-retraction step induced by a semantic-Fisher log-rate."""
    p = smooth_simplex(p, eps=eps)
    w = torch.nan_to_num(lograte.to(device=p.device, dtype=p.dtype))
    w = w - (p * w).sum(dim=-1, keepdim=True)
    z = p.sqrt()
    zdot = semantic_fisher_sphere_velocity(z, w)
    z_next = (z + float(dt) * zdot).clamp_min(eps)
    z_next = z_next / z_next.norm(dim=-1, keepdim=True).clamp_min(eps)
    p_next = z_next.square()
    return p_next / p_next.sum(dim=-1, keepdim=True).clamp_min(eps)


def integrate_semantic_fisher_teacher_path(
    p_start: torch.Tensor,
    advantage: torch.Tensor,
    gram: torch.Tensor,
    beta: torch.Tensor | float,
    gamma: torch.Tensor | float,
    steps: int,
    dt: float | None = None,
    gram_rank: int | None = None,
    gram_factors: torch.Tensor | None = None,
    eps: float = EPS,
) -> SemanticFisherTeacherPath:
    """Integrate the exact semantic-Fisher teacher field on a fixed local support.

    The state ``c=(B,y,S)`` and scores/Gram stay fixed, but the log-rate is recomputed
    at every intermediate policy because the pullback metric depends on ``P=diag(p)``.
    """
    n_steps = max(int(steps), 1)
    step_dt = float(1.0 / n_steps if dt is None else dt)
    p = smooth_simplex(p_start, eps=eps)
    policies = [p]
    logrates = []
    sphere_velocities = []
    for _ in range(n_steps):
        w = semantic_fisher_lograte(
            p, advantage, gram, beta=beta, gamma=gamma, gram_rank=gram_rank,
            gram_factors=gram_factors, eps=eps
        )
        z = p.clamp_min(eps).sqrt()
        zdot = semantic_fisher_sphere_velocity(z, w)
        logrates.append(w)
        sphere_velocities.append(zdot)
        p = semantic_fisher_sphere_step(p, w, dt=step_dt, eps=eps)
        policies.append(p)
    return SemanticFisherTeacherPath(
        policies=policies,
        logrates=logrates,
        sphere_velocities=sphere_velocities,
        dt=step_dt,
    )


def integrate_semantic_fisher_endpoint_path(
    p_start: torch.Tensor,
    q_target: torch.Tensor,
    gram: torch.Tensor,
    beta: torch.Tensor | float,
    gamma: torch.Tensor | float,
    steps: int,
    dt: float | None = None,
    gram_rank: int | None = None,
    gram_factors: torch.Tensor | None = None,
    q_smoothing: float = 1e-3,
    eps: float = EPS,
) -> SemanticFisherTeacherPath:
    """Integrate a target-sampled endpoint flow toward ``q_target``.

    Unlike ``integrate_semantic_fisher_teacher_path``, the driving log-ratio is
    recomputed at each lambda-time policy:

        r_lambda = log(q_eps) - log(p_lambda).
    """
    n_steps = max(int(steps), 1)
    step_dt = float(1.0 / n_steps if dt is None else dt)
    p0 = smooth_simplex(p_start, eps=eps)
    q = smooth_simplex(q_target.to(device=p0.device, dtype=p0.dtype), eps=eps)
    q_eps = smooth_simplex((1.0 - float(q_smoothing)) * q + float(q_smoothing) * p0, eps=eps)
    gram = gram.to(device=p0.device, dtype=p0.dtype)
    if gram_factors is not None:
        gram_factors = gram_factors.to(device=p0.device, dtype=p0.dtype)

    p = p0
    policies = [p]
    logrates = []
    sphere_velocities = []
    for _ in range(n_steps):
        advantage = torch.nan_to_num(q_eps.clamp_min(eps).log() - p.clamp_min(eps).log())
        w = semantic_fisher_lograte(
            p,
            advantage,
            gram,
            beta=beta,
            gamma=gamma,
            gram_rank=gram_rank,
            gram_factors=gram_factors,
            eps=eps,
        )
        z = p.clamp_min(eps).sqrt()
        zdot = semantic_fisher_sphere_velocity(z, w)
        logrates.append(w)
        sphere_velocities.append(zdot)
        p = semantic_fisher_sphere_step(p, w, dt=step_dt, eps=eps)
        policies.append(p)
    return SemanticFisherTeacherPath(
        policies=policies,
        logrates=logrates,
        sphere_velocities=sphere_velocities,
        dt=step_dt,
    )


def _solve_linear(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.solve(matrix, rhs)
    except RuntimeError:
        return torch.linalg.pinv(matrix) @ rhs


def _solve_pullback_system(
    gram: torch.Tensor,
    p: torch.Tensor,
    gamma_t: torch.Tensor,
    rhs: torch.Tensor,
    gram_rank: int | None,
    gram_factors: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    A = p.shape[-1]
    rank = None if gram_rank is None else int(gram_rank)
    if rank is None or rank <= 0 or rank >= A:
        eye = torch.eye(A, device=p.device, dtype=p.dtype)
        while eye.dim() < gram.dim():
            eye = eye.unsqueeze(0)
        P = torch.diag_embed(p)
        M = eye + gamma_t.unsqueeze(-1) * (gram @ P)
        return _solve_linear(M, rhs)
    try:
        U = (
            _low_rank_factor_from_factors(gram_factors, rank)
            if gram_factors is not None
            else _low_rank_factor_from_gram(gram, rank)
        )
    except RuntimeError:
        return _solve_pullback_system(gram, p, gamma_t, rhs, None, None, eps)
    return _woodbury_solve_pullback(U, p, gamma_t, rhs, eps)


def _low_rank_factor_from_factors(factors: torch.Tensor, rank: int) -> torch.Tensor:
    max_rank = min(int(rank), factors.shape[-1])
    if max_rank >= factors.shape[-1]:
        return factors
    try:
        u, s, _ = torch.linalg.svd(factors, full_matrices=False)
        max_rank = min(max_rank, s.shape[-1])
        return u[..., :max_rank] * s[..., :max_rank].unsqueeze(-2)
    except RuntimeError:
        return factors[..., :max_rank]


def _low_rank_factor_from_gram(gram: torch.Tensor, rank: int) -> torch.Tensor:
    gram = 0.5 * (gram + gram.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(gram)
    rank = min(int(rank), gram.shape[-1])
    idx = evals.argsort(dim=-1, descending=True)[..., :rank]
    gather_idx = idx.unsqueeze(-2).expand(*idx.shape[:-1], gram.shape[-1], rank)
    top_vecs = torch.gather(evecs, dim=-1, index=gather_idx)
    top_vals = torch.gather(evals, dim=-1, index=idx).clamp_min(0.0).sqrt().unsqueeze(-2)
    return top_vecs * top_vals


def _woodbury_solve_pullback(
    factors: torch.Tensor,
    p: torch.Tensor,
    gamma_t: torch.Tensor,
    rhs: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    rank = factors.shape[-1]
    eye = torch.eye(rank, device=factors.device, dtype=factors.dtype)
    while eye.dim() < factors.dim():
        eye = eye.unsqueeze(0)
    p_col = p.unsqueeze(-1)
    pu = p_col * factors
    inner = eye + gamma_t.unsqueeze(-1) * (factors.transpose(-1, -2) @ pu)
    projected_rhs = factors.transpose(-1, -2) @ (p_col * rhs)
    coef = _solve_linear(inner, projected_rhs)
    return rhs - gamma_t.unsqueeze(-1) * (factors @ coef)
