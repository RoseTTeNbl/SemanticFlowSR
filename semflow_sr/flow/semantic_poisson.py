"""Weighted-Poisson semantic transport on categorical Fisher manifolds.

The helpers in this module implement the reaction-to-transport lift.  They
operate on complete endpoint particles and never resample, assign, or match
particles.  A semantic density tilt is represented by the minimum-kinetic
Fisher natural-gradient field of a scalar potential.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _as_particle_blocks(value: torch.Tensor) -> tuple[torch.Tensor, bool]:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0), True
    if tensor.ndim != 3:
        raise ValueError("expected probabilities with shape [block, action] or [particle, block, action]")
    return tensor, False


def _support_normalize(
    probabilities: torch.Tensor,
    action_mask: torch.Tensor | None,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    particles, squeezed = _as_particle_blocks(probabilities)
    if action_mask is None:
        mask = torch.ones_like(particles, dtype=torch.bool)
    else:
        raw_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=particles.device)
        if raw_mask.ndim == 2:
            raw_mask = raw_mask.unsqueeze(0)
        if raw_mask.shape[0] == 1 and particles.shape[0] != 1:
            raw_mask = raw_mask.expand(particles.shape[0], -1, -1)
        if raw_mask.shape != particles.shape:
            raise ValueError("action_mask must broadcast to the probability shape")
        mask = raw_mask
    dtype = particles.dtype if particles.is_floating_point() else torch.float32
    particles = particles.to(dtype=dtype)
    p = torch.where(mask, particles.clamp_min(0.0), torch.zeros_like(particles))
    support_mass = p.sum(dim=-1, keepdim=True)
    if bool((support_mass <= 0).any().detach().cpu()):
        raise ValueError("each categorical block must contain positive supported mass")
    p = p / support_mass.clamp_min(float(eps))
    return (p.squeeze(0) if squeezed else p), (mask.squeeze(0) if squeezed else mask)


def fisher_natural_gradient(
    probabilities: torch.Tensor,
    covector: torch.Tensor,
    *,
    action_mask: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Convert an endpoint covector to a categorical Fisher tangent.

    For each block, ``v_a = p_a (g_a - E_p[g])``.  Unsupported actions are
    exactly zero and the tangent sums to zero within every block.
    """
    p, mask = _support_normalize(probabilities, action_mask, eps=float(eps))
    g = torch.as_tensor(covector, dtype=p.dtype, device=p.device)
    if g.shape != p.shape:
        raise ValueError("covector must have the same shape as probabilities")
    g = torch.where(mask, g, torch.zeros_like(g))
    centered = g - (p * g).sum(dim=-1, keepdim=True)
    tangent = torch.where(mask, p * centered, torch.zeros_like(p))
    return tangent - p * tangent.sum(dim=-1, keepdim=True)


def fisher_squared_norm(
    probabilities: torch.Tensor,
    tangent: torch.Tensor,
    *,
    action_mask: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Return the per-particle, per-block Fisher squared norm."""
    p, mask = _support_normalize(probabilities, action_mask, eps=float(eps))
    velocity = torch.as_tensor(tangent, dtype=p.dtype, device=p.device)
    if velocity.shape != p.shape:
        raise ValueError("tangent must have the same shape as probabilities")
    velocity = torch.where(mask, velocity, torch.zeros_like(velocity))
    return (velocity.square() / p.clamp_min(float(eps))).sum(dim=-1)


@dataclass(frozen=True)
class WeightedPoissonResult:
    loss: torch.Tensor
    tangent: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def weighted_poisson_loss(
    potential_values: torch.Tensor,
    probabilities: torch.Tensor,
    energies: torch.Tensor,
    *,
    particle_weights: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
    create_graph: bool = True,
    eps: float = 1.0e-8,
) -> WeightedPoissonResult:
    """Monte Carlo Dirichlet functional for the weighted Poisson equation.

    ``probabilities`` must require gradients and each scalar potential should
    depend only on its corresponding particle.  The semantic energy is treated
    as an observed endpoint value; no derivative through hard decoding is used.
    """
    p, _squeezed = _as_particle_blocks(probabilities)
    phi = torch.as_tensor(potential_values, device=p.device).flatten()
    energy = torch.as_tensor(energies, dtype=p.dtype, device=p.device).flatten()
    if int(phi.numel()) != int(p.shape[0]) or int(energy.numel()) != int(p.shape[0]):
        raise ValueError("one potential and energy are required per particle")
    if not probabilities.requires_grad:
        raise ValueError("probabilities must require gradients")
    if particle_weights is None:
        weights = torch.ones_like(energy) / max(int(energy.numel()), 1)
    else:
        weights = torch.as_tensor(particle_weights, dtype=p.dtype, device=p.device).flatten()
        if int(weights.numel()) != int(energy.numel()):
            raise ValueError("particle_weights must contain one value per particle")
        if bool((weights < 0).any().detach().cpu()) or float(weights.sum().detach().cpu()) <= 0.0:
            raise ValueError("particle_weights must be non-negative with positive mass")
        weights = weights / weights.sum()
    covector = torch.autograd.grad(
        phi.sum(),
        probabilities,
        create_graph=bool(create_graph),
        retain_graph=True,
        allow_unused=False,
    )[0]
    tangent = fisher_natural_gradient(probabilities, covector, action_mask=action_mask, eps=float(eps))
    block_norm = fisher_squared_norm(probabilities, tangent, action_mask=action_mask, eps=float(eps))
    if block_norm.ndim == 1:
        particle_norm = block_norm.sum().unsqueeze(0)
    else:
        particle_norm = block_norm.sum(dim=-1)
    energy_bar = (weights * energy).sum()
    phi_bar = (weights * phi).sum()
    centered_energy = energy - energy_bar
    centered_phi = phi - phi_bar
    dirichlet = 0.5 * (weights * particle_norm).sum()
    source = (weights * centered_energy.detach() * centered_phi).sum()
    loss = dirichlet + source
    diagnostics = {
        "dirichlet": dirichlet.detach(),
        "source": source.detach(),
        "energy_mean": energy_bar.detach(),
        "energy_variance": (weights * centered_energy.square()).sum().detach(),
        "potential_mean": phi_bar.detach(),
        "tangent_mass_error": tangent.sum(dim=-1).abs().max().detach(),
    }
    return WeightedPoissonResult(loss=loss, tangent=tangent, diagnostics=diagnostics)


def exponential_fisher_correction(
    probabilities: torch.Tensor,
    tangent: torch.Tensor,
    step_size: float | torch.Tensor,
    *,
    active_block_mask: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Apply a same-particle categorical exponential-chart correction.

    The multiplicative log-rate chart is the first-order Fisher exponential
    update and has derivative equal to ``tangent`` at zero step.  It keeps all
    particles and inactive blocks in their original order.
    """
    p, mask = _support_normalize(probabilities, action_mask, eps=float(eps))
    velocity = torch.as_tensor(tangent, dtype=p.dtype, device=p.device)
    if velocity.shape != p.shape:
        raise ValueError("tangent must have the same shape as probabilities")
    rate = torch.where(mask, velocity / p.clamp_min(float(eps)), torch.zeros_like(velocity))
    logits = torch.where(mask, p.clamp_min(float(eps)).log() + torch.as_tensor(step_size, dtype=p.dtype, device=p.device) * rate, torch.full_like(p, -torch.inf))
    corrected = torch.softmax(logits, dim=-1)
    corrected = torch.where(mask, corrected, torch.zeros_like(corrected))
    corrected = corrected / corrected.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    if active_block_mask is not None:
        active = torch.as_tensor(active_block_mask, dtype=torch.bool, device=p.device)
        while active.ndim < p.ndim:
            active = active.unsqueeze(-1)
        corrected = torch.where(active, corrected, p)
    return corrected


@dataclass(frozen=True)
class FiniteResidualTarget:
    probability_velocity: torch.Tensor
    logit_velocity: torch.Tensor


def finite_endpoint_residual(
    source_probabilities: torch.Tensor,
    target_probabilities: torch.Tensor,
    step_size: float,
    *,
    action_mask: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> FiniteResidualTarget:
    """Return the finite same-particle endpoint residual for diagnostics."""
    step = float(step_size)
    if not torch.isfinite(torch.tensor(step)) or step <= 0.0:
        raise ValueError("step_size must be finite and positive")
    p, mask = _support_normalize(source_probabilities, action_mask, eps=float(eps))
    q, _ = _support_normalize(target_probabilities, action_mask, eps=float(eps))
    if p.shape != q.shape:
        raise ValueError("source and target probabilities must have matching shapes")
    probability_velocity = (q - p) / step
    raw_logit = (q.clamp_min(float(eps)).log() - p.clamp_min(float(eps)).log()) / step
    support_count = mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
    support_mean = (raw_logit * mask.float()).sum(dim=-1, keepdim=True) / support_count
    logit_velocity = torch.where(mask, raw_logit - support_mean, torch.zeros_like(raw_logit))
    return FiniteResidualTarget(
        probability_velocity=probability_velocity,
        logit_velocity=logit_velocity,
    )


def poisson_summary(result: WeightedPoissonResult) -> dict[str, Any]:
    """Convert diagnostics to JSON-friendly scalars."""
    return {key: float(value.detach().cpu()) for key, value in result.diagnostics.items()}
