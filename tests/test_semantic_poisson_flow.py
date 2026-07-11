from __future__ import annotations

import torch

from semflow_sr.flow.semantic_poisson import (
    exponential_fisher_correction,
    finite_endpoint_residual,
    fisher_natural_gradient,
    fisher_squared_norm,
    weighted_poisson_loss,
)


def test_fisher_natural_gradient_is_supported_zero_mass_tangent():
    p = torch.tensor(
        [[[0.2, 0.8, 0.0], [0.4, 0.1, 0.5]]],
        dtype=torch.float64,
    )
    mask = torch.tensor([[True, True, False], [True, True, True]])
    covector = torch.tensor(
        [[[2.0, -1.0, 99.0], [1.0, 0.0, -2.0]]],
        dtype=torch.float64,
    )

    tangent = fisher_natural_gradient(p, covector, action_mask=mask)
    expected_norm = (p * (covector.masked_fill(~mask, 0.0) - (p * covector.masked_fill(~mask, 0.0)).sum(-1, keepdim=True)).square()).sum(-1)

    assert tangent.dtype == torch.float64
    assert torch.allclose(tangent.sum(dim=-1), torch.zeros((1, 2), dtype=torch.float64), atol=1e-12)
    assert tangent[0, 0, 2] == 0
    assert torch.allclose(fisher_squared_norm(p, tangent, action_mask=mask), expected_norm, atol=1e-10)


def test_constant_potential_has_zero_tangent_and_identity_correction():
    p = torch.tensor(
        [[[0.3, 0.7], [0.6, 0.4]], [[0.8, 0.2], [0.1, 0.9]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    phi = p.sum(dim=(-1, -2)) * 0.0 + 3.0
    energies = torch.tensor([0.0, 1.0], dtype=torch.float64)

    result = weighted_poisson_loss(phi, p, energies)
    corrected = exponential_fisher_correction(p.detach(), result.tangent.detach(), 0.1)

    assert torch.allclose(result.tangent, torch.zeros_like(result.tangent), atol=1e-12)
    assert torch.allclose(corrected, p.detach(), atol=1e-12)


def test_inactive_blocks_and_particle_order_are_preserved():
    p = torch.tensor(
        [
            [[0.25, 0.75], [0.7, 0.3]],
            [[0.9, 0.1], [0.2, 0.8]],
        ],
        dtype=torch.float64,
    )
    covector = torch.tensor(
        [
            [[-1.0, 1.0], [1.0, -1.0]],
            [[2.0, -2.0], [-2.0, 2.0]],
        ],
        dtype=torch.float64,
    )
    tangent = fisher_natural_gradient(p, covector)
    active = torch.tensor([[True, False], [True, False]])

    corrected = exponential_fisher_correction(p, tangent, 0.1, active_block_mask=active)

    assert corrected.shape == p.shape
    assert torch.allclose(corrected[:, 1], p[:, 1], atol=0.0, rtol=0.0)
    assert corrected[0, 0, 1] > p[0, 0, 1]
    assert corrected[1, 0, 0] > p[1, 0, 0]


def test_exponential_correction_first_order_derivative_matches_tangent():
    p = torch.tensor([[[0.35, 0.65]]], dtype=torch.float64)
    tangent = fisher_natural_gradient(p, torch.tensor([[[-0.7, 0.4]]], dtype=torch.float64))
    step = 1.0e-6

    corrected = exponential_fisher_correction(p, tangent, step)
    finite = (corrected - p) / step

    assert torch.allclose(finite, tangent, atol=1e-6, rtol=1e-5)


def test_weighted_poisson_toy_moves_mass_toward_lower_energy():
    p = torch.tensor(
        [[[0.8, 0.2]], [[0.2, 0.8]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    energy = torch.tensor([0.0, 1.0], dtype=torch.float64)
    coefficient = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([coefficient], lr=0.4)

    for _ in range(80):
        optimizer.zero_grad()
        phi = coefficient * p[:, 0, 1]
        result = weighted_poisson_loss(phi, p, energy, create_graph=True)
        result.loss.backward()
        optimizer.step()

    phi = coefficient * p[:, 0, 1]
    result = weighted_poisson_loss(phi, p, energy, create_graph=False)
    corrected = exponential_fisher_correction(p.detach(), result.tangent.detach(), 0.1)

    assert coefficient.item() < 0.0
    assert corrected[:, 0, 1].mean() < p.detach()[:, 0, 1].mean()
    assert result.diagnostics["energy_variance"] > 0


def test_finite_residual_is_centered_and_zero_for_identity():
    p = torch.tensor([[[0.3, 0.7], [0.6, 0.4]]], dtype=torch.float64)
    identity = finite_endpoint_residual(p, p, 0.1)
    assert torch.allclose(identity.probability_velocity, torch.zeros_like(p), atol=0.0, rtol=0.0)
    assert torch.allclose(identity.logit_velocity, torch.zeros_like(p), atol=0.0, rtol=0.0)

    tangent = fisher_natural_gradient(p, torch.tensor([[[1.0, -1.0], [-1.0, 2.0]]], dtype=torch.float64))
    q = exponential_fisher_correction(p, tangent, 0.1)
    residual = finite_endpoint_residual(p, q, 0.1)
    assert torch.allclose(residual.probability_velocity.sum(dim=-1), torch.zeros((1, 2), dtype=torch.float64), atol=1e-12)
    assert torch.allclose(residual.logit_velocity.mean(dim=-1), torch.zeros((1, 2), dtype=torch.float64), atol=1e-12)
