"""Source-preserving endpoint coupling for one-step semantic Fisher cycles."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


ONE_STEP_FISHER_OBJECTIVE_VERSION = "one_step_semantic_fisher_cycle_v4_lineage_proximal"
FISHER_TIME_BINS = (
    (0.0, 0.001),
    (0.001, 0.005),
    (0.005, 0.02),
    (0.02, 0.1),
    (0.1, 1.0),
)


@dataclass(frozen=True)
class SourcePreservingCoupling:
    resampled_endpoint_indices: torch.Tensor
    assigned_slot_for_source: torch.Tensor
    assigned_endpoint_indices: torch.Tensor
    target_probabilities: torch.Tensor
    active_masks: torch.Tensor
    pair_costs: torch.Tensor
    cost_matrix: torch.Tensor
    resample_counts: torch.Tensor


@dataclass(frozen=True)
class SourceConditionedTraceCoupling:
    capacity_atom_indices: torch.Tensor
    assigned_slot_for_source: torch.Tensor
    assigned_atom_indices: torch.Tensor
    target_probabilities: torch.Tensor
    active_masks: torch.Tensor
    pair_costs: torch.Tensor
    cost_matrix: torch.Tensor
    atom_cost_matrix: torch.Tensor
    capacity_counts: torch.Tensor


@dataclass(frozen=True)
class SemanticProximalUpdate:
    prior_weights: torch.Tensor
    posterior_weights: torch.Tensor
    temperature: float
    kl_divergence: float
    prior_expected_energy: float
    posterior_expected_energy: float
    effective_sample_size: float


@dataclass(frozen=True)
class LogSinkhornResult:
    plan: torch.Tensor
    row_marginal_error: float
    column_marginal_error: float
    iterations: int
    entropy: float


@dataclass(frozen=True)
class EntropicTraceCoupling:
    plan: torch.Tensor
    target_probabilities: torch.Tensor
    active_masks: torch.Tensor
    row_marginal: torch.Tensor
    target_marginal: torch.Tensor
    requested_target_marginal: torch.Tensor
    source_cost_matrix: torch.Tensor
    correction_cost_matrix: torch.Tensor
    total_cost_matrix: torch.Tensor
    entropy: float
    lambda_correction: float
    posterior_strength: float
    reference_path_cost: float
    expected_source_cost: float
    expected_correction_cost: float
    correction_ratio: float
    row_marginal_error: float
    column_marginal_error: float
    sinkhorn_iterations: int


class CorrectionBudgetError(ValueError):
    def __init__(self, message: str, *, best_ratio: float, ratio_limit: float):
        super().__init__(message)
        self.best_ratio = float(best_ratio)
        self.ratio_limit = float(ratio_limit)


def _normalize_probabilities(
    probabilities: torch.Tensor,
    eps: float = 1.0e-12,
    support_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    p = torch.as_tensor(probabilities).float().clamp_min(0.0)
    if support_mask is not None:
        mask = torch.as_tensor(support_mask, dtype=torch.bool, device=p.device)
        p = torch.where(mask, p, torch.zeros_like(p))
    return p / p.sum(dim=-1, keepdim=True).clamp_min(float(eps))


def _block_fisher_squared(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source = _normalize_probabilities(source)
    target = _normalize_probabilities(target).to(source.device)
    root_source = source.clamp_min(1.0e-12).sqrt()
    root_target = target.clamp_min(1.0e-12).sqrt()
    half_angle = torch.atan2(
        (root_source - root_target).norm(dim=-1),
        (root_source + root_target).norm(dim=-1).clamp_min(1.0e-12),
    )
    return (4.0 * half_angle).square()


def fisher_rao_probability_path_and_logit_velocity(
    source_probabilities: torch.Tensor,
    target_probabilities: torch.Tensor,
    t: float,
    *,
    eps: float = 1.0e-8,
    support_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Stable Fisher-Rao geodesic path on independent categorical blocks.

    The square-root chart turns a categorical block into a unit-sphere point.
    This helper uses an atan2 angle and an exact small-angle branch, so identical
    endpoints have exactly zero probability and logit velocity.
    """
    mask = None
    if support_mask is not None:
        mask = torch.as_tensor(support_mask, dtype=torch.bool, device=torch.as_tensor(source_probabilities).device)
    p0 = _normalize_probabilities(source_probabilities, eps=float(eps), support_mask=mask)
    p1 = _normalize_probabilities(target_probabilities, eps=float(eps), support_mask=mask).to(p0.device)
    if p0.shape != p1.shape:
        raise ValueError("source and target probabilities must have matching shape")
    root0 = p0.clamp_min(float(eps)).sqrt()
    root1 = p1.clamp_min(float(eps)).sqrt()
    if mask is not None:
        root0 = torch.where(mask, root0, torch.zeros_like(root0))
        root1 = torch.where(mask, root1, torch.zeros_like(root1))
    half_angle = torch.atan2(
        (root0 - root1).norm(dim=-1, keepdim=True),
        (root0 + root1).norm(dim=-1, keepdim=True).clamp_min(float(eps)),
    )
    omega = 2.0 * half_angle
    sin_omega = torch.sin(omega)
    tt = torch.as_tensor(float(t), dtype=p0.dtype, device=p0.device)
    small = omega.abs() < 1.0e-6
    denom = sin_omega.abs().clamp_min(float(eps))
    a = torch.where(small, 1.0 - tt, torch.sin((1.0 - tt) * omega) / denom)
    b = torch.where(small, tt, torch.sin(tt * omega) / denom)
    da = torch.where(small, torch.full_like(omega, -1.0), -omega * torch.cos((1.0 - tt) * omega) / denom)
    db = torch.where(small, torch.ones_like(omega), omega * torch.cos(tt * omega) / denom)
    root = a * root0 + b * root1
    droot = da * root0 + db * root1
    probability = root.square().clamp_min(float(eps))
    if mask is not None:
        probability = torch.where(mask, probability, torch.zeros_like(probability))
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    dprobability = 2.0 * root * droot
    if mask is not None:
        dprobability = torch.where(mask, dprobability, torch.zeros_like(dprobability))
    dprobability = dprobability - probability * dprobability.sum(dim=-1, keepdim=True)
    logit_velocity = dprobability / probability.clamp_min(float(eps))
    if mask is None:
        logit_velocity = logit_velocity - logit_velocity.mean(dim=-1, keepdim=True)
    else:
        support_count = mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        support_mean = (logit_velocity * mask.float()).sum(dim=-1, keepdim=True) / support_count
        logit_velocity = torch.where(mask, logit_velocity - support_mean, torch.zeros_like(logit_velocity))
    exact_same = (p0 - p1).abs().amax(dim=-1, keepdim=True) <= 10.0 * float(eps)
    probability = torch.where(exact_same, p0, probability)
    logit_velocity = torch.where(exact_same, torch.zeros_like(logit_velocity), logit_velocity)
    diagnostics = {
        "omega": omega.squeeze(-1),
        "max_angular_step": omega.squeeze(-1).abs(),
        "min_probability": probability.amin(dim=-1),
        "probability_tangent_zero_sum": dprobability.sum(dim=-1).abs(),
    }
    return probability, logit_velocity, diagnostics


def semantic_log_quality_weights(
    fitted_distances: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate endpoint weights from complete-expression semantic samples.

    Each endpoint receives ``log(mean(exp(-distance / temperature)))``. Invalid
    expression samples contribute zero mass rather than a fabricated penalty.
    """
    distances = torch.as_tensor(fitted_distances).float()
    valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=distances.device)
    if distances.ndim != 2 or valid.shape != distances.shape:
        raise ValueError("fitted_distances and valid_mask must have shape [endpoint, sample]")
    if distances.shape[1] == 0:
        raise ValueError("at least one complete-expression sample is required")
    tau = float(temperature)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("semantic temperature must be finite and positive")
    valid = valid & torch.isfinite(distances)
    scores = torch.where(valid, -distances / tau, torch.full_like(distances, -torch.inf))
    log_quality = torch.logsumexp(scores, dim=1) - float(np.log(int(distances.shape[1])))
    usable = torch.isfinite(log_quality)
    if not bool(usable.any().detach().cpu()):
        raise ValueError("all endpoint proposals have zero valid semantic quality")
    normalized_logits = torch.where(usable, log_quality, torch.full_like(log_quality, -torch.inf))
    weights = torch.softmax(normalized_logits, dim=0)
    return weights, log_quality


def kl_constrained_semantic_weights(
    energies: torch.Tensor,
    prior_weights: torch.Tensor | None = None,
    *,
    kl_budget: float,
    max_iterations: int = 80,
    tolerance: float = 1.0e-8,
) -> SemanticProximalUpdate:
    """Solve the finite-particle semantic KL-proximal update.

    The returned posterior is proportional to ``prior * exp(-energy / tau)``.
    The smallest temperature satisfying the KL budget is found by bisection,
    which gives the lowest expected energy within this exponential family.
    Zero-prior atoms retain exactly zero posterior mass.
    """
    raw_energies = torch.as_tensor(energies)
    if raw_energies.ndim != 1 or int(raw_energies.numel()) == 0:
        raise ValueError("energies must be a non-empty one-dimensional tensor")
    device = raw_energies.device
    output_dtype = raw_energies.dtype if raw_energies.is_floating_point() else torch.float32
    energy = raw_energies.to(dtype=torch.float64)
    if prior_weights is None:
        prior = torch.ones_like(energy)
    else:
        prior = torch.as_tensor(prior_weights, dtype=torch.float64, device=device).flatten()
        if int(prior.numel()) != int(energy.numel()):
            raise ValueError("prior_weights must contain one value per energy")
    if not bool(torch.isfinite(prior).all().detach().cpu()) or bool((prior < 0).any().detach().cpu()):
        raise ValueError("prior_weights must be finite and non-negative")
    if float(prior.sum().detach().cpu()) <= 0.0:
        raise ValueError("prior_weights must have positive total mass")
    prior = prior / prior.sum()
    live = prior > 0
    if not bool(torch.isfinite(energy[live]).all().detach().cpu()):
        raise ValueError("energies with positive prior mass must be finite")
    budget = float(kl_budget)
    if not np.isfinite(budget) or budget < 0.0:
        raise ValueError("kl_budget must be finite and non-negative")
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")

    live_prior = prior[live]
    live_energy = energy[live]
    log_prior = live_prior.log()
    energy_min = live_energy.min()

    def evaluate(temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
        logits = log_prior - (live_energy - energy_min) / float(temperature)
        weights = torch.softmax(logits, dim=0)
        divergence = (weights * (weights.clamp_min(1.0e-300).log() - log_prior)).sum()
        return weights, divergence

    spread = float((live_energy.max() - live_energy.min()).detach().cpu())
    if budget <= float(tolerance) or spread <= float(tolerance):
        live_posterior = live_prior
        divergence = torch.zeros((), dtype=torch.float64, device=device)
        temperature = float("inf")
    else:
        lower = max(1.0e-12, spread * 1.0e-10)
        lower_weights, lower_kl = evaluate(lower)
        if float(lower_kl.detach().cpu()) <= budget + float(tolerance):
            live_posterior = lower_weights
            divergence = lower_kl
            temperature = lower
        else:
            upper = max(spread, 1.0e-6)
            upper_weights, upper_kl = evaluate(upper)
            while float(upper_kl.detach().cpu()) > budget and upper < 1.0e16:
                upper *= 2.0
                upper_weights, upper_kl = evaluate(upper)
            if float(upper_kl.detach().cpu()) > budget + float(tolerance):
                raise RuntimeError("failed to bracket the semantic KL temperature")
            live_posterior = upper_weights
            divergence = upper_kl
            for _ in range(int(max_iterations)):
                middle = float(np.sqrt(lower * upper))
                middle_weights, middle_kl = evaluate(middle)
                if float(middle_kl.detach().cpu()) > budget:
                    lower = middle
                else:
                    upper = middle
                    live_posterior = middle_weights
                    divergence = middle_kl
                if upper / lower - 1.0 <= float(tolerance):
                    break
            temperature = upper

    posterior = torch.zeros_like(prior)
    posterior[live] = live_posterior
    posterior = posterior / posterior.sum().clamp_min(1.0e-300)
    prior_energy = float((prior * energy).sum().detach().cpu())
    posterior_energy = float((posterior * energy).sum().detach().cpu())
    ess = float((1.0 / posterior.square().sum().clamp_min(1.0e-300)).detach().cpu())
    return SemanticProximalUpdate(
        prior_weights=prior.to(dtype=output_dtype),
        posterior_weights=posterior.to(dtype=output_dtype),
        temperature=float(temperature),
        kl_divergence=float(divergence.detach().cpu()),
        prior_expected_energy=prior_energy,
        posterior_expected_energy=posterior_energy,
        effective_sample_size=ess,
    )


def log_domain_sinkhorn(
    row_marginal: torch.Tensor,
    column_marginal: torch.Tensor,
    cost_matrix: torch.Tensor,
    *,
    entropy: float,
    max_iterations: int = 500,
    tolerance: float = 1.0e-7,
) -> LogSinkhornResult:
    """Compute entropic optimal transport with prescribed finite marginals."""
    raw_cost = torch.as_tensor(cost_matrix)
    if raw_cost.ndim != 2 or min(raw_cost.shape) <= 0:
        raise ValueError("cost_matrix must be a non-empty matrix")
    device = raw_cost.device
    output_dtype = raw_cost.dtype if raw_cost.is_floating_point() else torch.float32
    cost = raw_cost.to(dtype=torch.float64)
    rows = torch.as_tensor(row_marginal, dtype=torch.float64, device=device).flatten()
    columns = torch.as_tensor(column_marginal, dtype=torch.float64, device=device).flatten()
    if int(rows.numel()) != int(cost.shape[0]) or int(columns.numel()) != int(cost.shape[1]):
        raise ValueError("marginal sizes must match cost_matrix")
    if not bool(torch.isfinite(rows).all().detach().cpu()) or not bool(torch.isfinite(columns).all().detach().cpu()):
        raise ValueError("Sinkhorn marginals must be finite")
    if bool((rows < 0).any().detach().cpu()) or bool((columns < 0).any().detach().cpu()):
        raise ValueError("Sinkhorn marginals must be non-negative")
    if float(rows.sum().detach().cpu()) <= 0.0 or float(columns.sum().detach().cpu()) <= 0.0:
        raise ValueError("Sinkhorn marginals must have positive total mass")
    rows = rows / rows.sum()
    columns = columns / columns.sum()
    regularization = float(entropy)
    if not np.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("Sinkhorn entropy must be finite and positive")
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")

    live_rows = rows > 0
    live_columns = columns > 0
    live_cost = cost[live_rows][:, live_columns]
    if bool((~torch.isfinite(live_cost)).all(dim=1).any().detach().cpu()):
        raise ValueError("every positive-mass row needs a finite transport edge")
    if bool((~torch.isfinite(live_cost)).all(dim=0).any().detach().cpu()):
        raise ValueError("every positive-mass column needs a finite transport edge")
    live_row_mass = rows[live_rows]
    live_column_mass = columns[live_columns]
    log_kernel = torch.where(
        torch.isfinite(live_cost),
        -live_cost / regularization,
        torch.full_like(live_cost, -torch.inf),
    )
    log_u = torch.zeros_like(live_row_mass)
    log_v = torch.zeros_like(live_column_mass)
    log_rows = live_row_mass.log()
    log_columns = live_column_mass.log()
    iterations = int(max_iterations)
    for iteration in range(1, int(max_iterations) + 1):
        log_u = log_rows - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_columns - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
        if iteration == 1 or iteration % 10 == 0 or iteration == int(max_iterations):
            live_plan = torch.exp(log_kernel + log_u[:, None] + log_v[None, :])
            row_error = (live_plan.sum(dim=1) - live_row_mass).abs().max()
            column_error = (live_plan.sum(dim=0) - live_column_mass).abs().max()
            if max(float(row_error.detach().cpu()), float(column_error.detach().cpu())) <= float(tolerance):
                iterations = iteration
                break
    live_plan = torch.exp(log_kernel + log_u[:, None] + log_v[None, :])
    plan = torch.zeros_like(cost)
    row_indices = live_rows.nonzero(as_tuple=False).flatten()
    column_indices = live_columns.nonzero(as_tuple=False).flatten()
    plan[row_indices[:, None], column_indices[None, :]] = live_plan
    row_error_value = float((plan.sum(dim=1) - rows).abs().max().detach().cpu())
    column_error_value = float((plan.sum(dim=0) - columns).abs().max().detach().cpu())
    return LogSinkhornResult(
        plan=plan.to(dtype=output_dtype),
        row_marginal_error=row_error_value,
        column_marginal_error=column_error_value,
        iterations=iterations,
        entropy=regularization,
    )


def systematic_resample_indices(
    weights: torch.Tensor,
    count: int,
    *,
    offset: float = 0.5,
) -> torch.Tensor:
    """Deterministically realize a weighted empirical marginal with ``count`` slots."""
    probabilities = torch.as_tensor(weights).float().flatten()
    if int(count) <= 0:
        raise ValueError("resample count must be positive")
    if int(probabilities.numel()) == 0:
        raise ValueError("cannot resample an empty endpoint set")
    if not bool(torch.isfinite(probabilities).all().detach().cpu()) or bool((probabilities < 0).any().detach().cpu()):
        raise ValueError("endpoint weights must be finite and non-negative")
    total = probabilities.sum()
    if float(total.detach().cpu()) <= 0.0:
        raise ValueError("endpoint weights must have positive total mass")
    normalized = probabilities / total
    phase = float(offset)
    if not 0.0 <= phase < 1.0:
        raise ValueError("systematic resampling offset must lie in [0, 1)")
    positions = (torch.arange(int(count), device=normalized.device, dtype=normalized.dtype) + phase) / float(count)
    cdf = normalized.cumsum(dim=0)
    cdf[-1] = 1.0
    return torch.searchsorted(cdf, positions, right=False).clamp_max(int(normalized.numel()) - 1)


def capacity_resample_indices(
    weights: torch.Tensor,
    count: int,
    *,
    min_one: bool = True,
    offset: float = 0.5,
) -> torch.Tensor:
    """Deterministically realize target capacities, preserving every live atom when possible."""
    probabilities = torch.as_tensor(weights).float().flatten()
    if int(count) <= 0:
        raise ValueError("resample count must be positive")
    if int(probabilities.numel()) == 0:
        raise ValueError("cannot resample an empty atom set")
    if not bool(torch.isfinite(probabilities).all().detach().cpu()) or bool((probabilities < 0).any().detach().cpu()):
        raise ValueError("atom weights must be finite and non-negative")
    total = probabilities.sum()
    if float(total.detach().cpu()) <= 0.0:
        raise ValueError("atom weights must have positive total mass")
    normalized = probabilities / total
    live = (normalized > 0).nonzero(as_tuple=False).flatten()
    if not bool(min_one):
        return systematic_resample_indices(normalized, int(count), offset=float(offset))
    if int(live.numel()) > int(count):
        top = torch.topk(normalized, k=int(count)).indices
        return top.sort().values
    base = live
    remaining = int(count) - int(base.numel())
    if remaining <= 0:
        return base
    extra = systematic_resample_indices(normalized, remaining, offset=float(offset))
    return torch.cat([base, extra], dim=0)


def block_fisher_squared_distance(
    source_probabilities: torch.Tensor,
    target_probabilities: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared Fisher sphere distance over active categorical blocks."""
    source = torch.as_tensor(source_probabilities).float()
    target = torch.as_tensor(target_probabilities).float().to(source.device)
    active = torch.as_tensor(active_mask, dtype=torch.bool, device=source.device).flatten()
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target probabilities must have shape [block, action]")
    if int(active.numel()) != int(source.shape[0]):
        raise ValueError("active mask must contain one value per block")
    if not bool(active.any().detach().cpu()):
        raise ValueError("an endpoint particle must have at least one active block")
    distance = _block_fisher_squared(source, target)
    return distance[active].mean()


def source_conditioned_trace_target_probabilities(
    source_probabilities: torch.Tensor,
    choices: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    projection_eps: float,
    projection_sharpness: float = 1.0,
) -> torch.Tensor:
    """Project a decoded trace to an entropy-controlled target in the source fiber.

    ``projection_eps`` smooths the selected one-hot action by the source block;
    ``projection_sharpness`` then mixes that sharp trace atom with the source
    block.  A sharpness of ``1`` is the historical epsilon-sharp projection,
    while smaller values keep endpoint entropy so that the trained flow can
    still sample nearby expressions instead of collapsing immediately.
    """
    source = _normalize_probabilities(source_probabilities)
    choice_tensor = torch.as_tensor(choices, dtype=torch.long, device=source.device).flatten()
    active = torch.as_tensor(active_mask, dtype=torch.bool, device=source.device).flatten()
    if source.ndim != 2:
        raise ValueError("source_probabilities must have shape [block, action]")
    if int(choice_tensor.numel()) != int(source.shape[0]) or int(active.numel()) != int(source.shape[0]):
        raise ValueError("choices and active_mask must contain one value per block")
    if bool(active.any().detach().cpu()):
        active_choices = choice_tensor[active]
        if bool(((active_choices < 0) | (active_choices >= int(source.shape[1]))).any().detach().cpu()):
            raise ValueError("active trace choice is outside the chart action range")
    eps = float(projection_eps)
    if not np.isfinite(eps) or eps < 0.0 or eps >= 1.0:
        raise ValueError("projection_eps must be finite and in [0, 1)")
    sharpness = float(projection_sharpness)
    if not np.isfinite(sharpness) or sharpness < 0.0 or sharpness > 1.0:
        raise ValueError("projection_sharpness must be finite and in [0, 1]")
    one_hot = torch.zeros_like(source)
    safe_choices = choice_tensor.clamp(0, int(source.shape[1]) - 1)
    one_hot.scatter_(1, safe_choices[:, None], 1.0)
    sharp_target = (1.0 - eps) * one_hot + eps * source
    active_target = sharpness * sharp_target + (1.0 - sharpness) * source
    target = torch.where(active[:, None], active_target, source)
    return _normalize_probabilities(target)


def source_conditioned_entropic_trace_coupling(
    source_probabilities: torch.Tensor,
    reference_endpoint_probabilities: torch.Tensor,
    atom_choices: torch.Tensor,
    atom_active_masks: torch.Tensor,
    posterior_atom_weights: torch.Tensor,
    *,
    prior_atom_weights: torch.Tensor | None = None,
    projection_eps: float,
    projection_sharpness: float = 1.0,
    correction_ratio_limit: float = 0.25,
    entropy_scale: float = 0.05,
    max_lambda_correction: float = 1.0e4,
    sinkhorn_max_iterations: int = 500,
    sinkhorn_tolerance: float = 1.0e-7,
) -> EntropicTraceCoupling:
    """Build the v3 soft source-to-complete-trace coupling.

    Rows have the uniform empirical source marginal and columns retain the
    semantic posterior over complete trace atoms.  Each source/trace pair has
    its own source-conditioned sharp endpoint, so inactive blocks remain in
    the source fiber.  If the requested posterior cannot meet the reference
    correction budget, its strength is reduced toward the supplied prior.
    """
    sources = _normalize_probabilities(torch.as_tensor(source_probabilities).float())
    references = _normalize_probabilities(
        torch.as_tensor(reference_endpoint_probabilities).float().to(sources.device)
    )
    choices = torch.as_tensor(atom_choices, dtype=torch.long, device=sources.device)
    active_masks = torch.as_tensor(atom_active_masks, dtype=torch.bool, device=sources.device)
    requested = torch.as_tensor(posterior_atom_weights).float().to(sources.device).flatten()
    if sources.ndim != 3 or references.shape != sources.shape:
        raise ValueError("source and reference endpoint probabilities must have shape [source, block, action]")
    if choices.ndim != 2 or active_masks.shape != choices.shape:
        raise ValueError("atom_choices and atom_active_masks must have shape [atom, block]")
    if int(choices.shape[1]) != int(sources.shape[1]):
        raise ValueError("trace atoms and sources must use the same block count")
    if int(requested.numel()) != int(choices.shape[0]):
        raise ValueError("posterior_atom_weights must contain one value per trace atom")
    if bool((active_masks.sum(dim=1) == 0).any().detach().cpu()):
        raise ValueError("every trace atom must have at least one active block")
    if not bool(torch.isfinite(requested).all().detach().cpu()) or bool((requested < 0).any().detach().cpu()):
        raise ValueError("posterior_atom_weights must be finite and non-negative")
    if float(requested.sum().detach().cpu()) <= 0.0:
        raise ValueError("posterior_atom_weights must have positive total mass")
    requested = requested / requested.sum()
    if prior_atom_weights is None:
        prior = requested.clone()
    else:
        prior = torch.as_tensor(prior_atom_weights).float().to(sources.device).flatten()
        if int(prior.numel()) != int(requested.numel()):
            raise ValueError("prior_atom_weights must contain one value per trace atom")
        if not bool(torch.isfinite(prior).all().detach().cpu()) or bool((prior < 0).any().detach().cpu()):
            raise ValueError("prior_atom_weights must be finite and non-negative")
        if float(prior.sum().detach().cpu()) <= 0.0:
            raise ValueError("prior_atom_weights must have positive total mass")
        prior = prior / prior.sum()
    ratio_limit = float(correction_ratio_limit)
    if not np.isfinite(ratio_limit) or ratio_limit < 0.0:
        raise ValueError("correction_ratio_limit must be finite and non-negative")
    scale = float(entropy_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("entropy_scale must be finite and positive")
    maximum_lambda = float(max_lambda_correction)
    if not np.isfinite(maximum_lambda) or maximum_lambda < 0.0:
        raise ValueError("max_lambda_correction must be finite and non-negative")

    source_count = int(sources.shape[0])
    atom_count = int(choices.shape[0])
    targets = torch.empty(
        (source_count, atom_count, int(sources.shape[1]), int(sources.shape[2])),
        dtype=sources.dtype,
        device=sources.device,
    )
    source_costs = torch.empty((source_count, atom_count), dtype=sources.dtype, device=sources.device)
    correction_costs = torch.empty_like(source_costs)
    for source_index in range(source_count):
        for atom_index in range(atom_count):
            target = source_conditioned_trace_target_probabilities(
                sources[source_index],
                choices[atom_index],
                active_masks[atom_index],
                projection_eps=float(projection_eps),
                projection_sharpness=float(projection_sharpness),
            )
            targets[source_index, atom_index] = target
            source_costs[source_index, atom_index] = _block_fisher_squared(
                sources[source_index], target
            ).mean()
            correction_costs[source_index, atom_index] = _block_fisher_squared(
                references[source_index], target
            ).mean()

    reference_path_cost = float(
        _block_fisher_squared(sources, references).mean(dim=1).mean().detach().cpu()
    )
    ratio_denominator = reference_path_cost + 1.0e-8
    finite_base_costs = (source_costs + correction_costs)[torch.isfinite(source_costs + correction_costs)]
    if int(finite_base_costs.numel()) == 0:
        raise ValueError("entropic coupling has no finite source/trace edge")
    median_cost = float(finite_base_costs.median().detach().cpu())
    entropy = max(1.0e-8, scale * max(median_cost, 1.0e-6))
    row_marginal = torch.full(
        (source_count,),
        1.0 / float(source_count),
        dtype=sources.dtype,
        device=sources.device,
    )

    def solve_at_lambda(
        target_marginal: torch.Tensor,
        lambda_correction: float,
    ) -> tuple[LogSinkhornResult, torch.Tensor, float]:
        total_cost = source_costs + float(lambda_correction) * correction_costs
        sinkhorn = log_domain_sinkhorn(
            row_marginal,
            target_marginal,
            total_cost,
            entropy=entropy,
            max_iterations=int(sinkhorn_max_iterations),
            tolerance=float(sinkhorn_tolerance),
        )
        plan = sinkhorn.plan.to(device=sources.device, dtype=sources.dtype)
        expected_correction = float((plan * correction_costs).sum().detach().cpu())
        return sinkhorn, total_cost, expected_correction / ratio_denominator

    def solve_with_budget(
        target_marginal: torch.Tensor,
    ) -> tuple[LogSinkhornResult, torch.Tensor, float, float] | None:
        sinkhorn, total_cost, ratio = solve_at_lambda(target_marginal, 0.0)
        if ratio <= ratio_limit + 1.0e-6:
            return sinkhorn, total_cost, 0.0, ratio
        if maximum_lambda <= 0.0:
            failed_ratios.append(float(ratio))
            return None
        lower = 0.0
        upper = min(1.0, maximum_lambda)
        upper_solution = solve_at_lambda(target_marginal, upper)
        while upper_solution[2] > ratio_limit + 1.0e-6 and upper < maximum_lambda:
            lower = upper
            upper = min(maximum_lambda, 2.0 * upper)
            upper_solution = solve_at_lambda(target_marginal, upper)
        if upper_solution[2] > ratio_limit + 1.0e-6:
            failed_ratios.append(float(upper_solution[2]))
            return None
        best_sinkhorn, best_total_cost, best_ratio = upper_solution
        for _ in range(24):
            middle = 0.5 * (lower + upper)
            middle_sinkhorn, middle_total_cost, middle_ratio = solve_at_lambda(target_marginal, middle)
            if middle_ratio > ratio_limit:
                lower = middle
            else:
                upper = middle
                best_sinkhorn = middle_sinkhorn
                best_total_cost = middle_total_cost
                best_ratio = middle_ratio
        return best_sinkhorn, best_total_cost, upper, best_ratio

    failed_ratios: list[float] = []
    selected_strength = 1.0
    selected_marginal = requested
    selected_solution = solve_with_budget(selected_marginal)
    if selected_solution is None and prior_atom_weights is not None:
        failed_strength = 1.0
        successful_strength: float | None = None
        successful_solution: tuple[LogSinkhornResult, torch.Tensor, float, float] | None = None
        for candidate_strength in (0.75, 0.5, 0.25, 0.1, 0.0):
            candidate_marginal = candidate_strength * requested + (1.0 - candidate_strength) * prior
            candidate_solution = solve_with_budget(candidate_marginal)
            if candidate_solution is None:
                failed_strength = candidate_strength
                continue
            successful_strength = candidate_strength
            successful_solution = candidate_solution
            break
        if successful_solution is not None and successful_strength is not None:
            lower_strength = successful_strength
            upper_strength = failed_strength
            for _ in range(12):
                middle_strength = 0.5 * (lower_strength + upper_strength)
                middle_marginal = middle_strength * requested + (1.0 - middle_strength) * prior
                middle_solution = solve_with_budget(middle_marginal)
                if middle_solution is None:
                    upper_strength = middle_strength
                else:
                    lower_strength = middle_strength
                    successful_solution = middle_solution
            selected_strength = lower_strength
            selected_marginal = selected_strength * requested + (1.0 - selected_strength) * prior
            selected_solution = successful_solution
    if selected_solution is None:
        best_ratio = min(failed_ratios) if failed_ratios else float("inf")
        raise CorrectionBudgetError(
            "semantic trace posterior violates the correction-cost budget even after prior shrinkage "
            f"(best_ratio={best_ratio:.6g}, limit={ratio_limit:.6g})",
            best_ratio=best_ratio,
            ratio_limit=ratio_limit,
        )

    sinkhorn, total_costs, lambda_correction, correction_ratio = selected_solution
    plan = sinkhorn.plan.to(device=sources.device, dtype=sources.dtype)
    expected_source_cost = float((plan * source_costs).sum().detach().cpu())
    expected_correction_cost = float((plan * correction_costs).sum().detach().cpu())
    return EntropicTraceCoupling(
        plan=plan,
        target_probabilities=targets,
        active_masks=active_masks,
        row_marginal=row_marginal,
        target_marginal=selected_marginal,
        requested_target_marginal=requested,
        source_cost_matrix=source_costs,
        correction_cost_matrix=correction_costs,
        total_cost_matrix=total_costs,
        entropy=entropy,
        lambda_correction=float(lambda_correction),
        posterior_strength=float(selected_strength),
        reference_path_cost=reference_path_cost,
        expected_source_cost=expected_source_cost,
        expected_correction_cost=expected_correction_cost,
        correction_ratio=float(correction_ratio),
        row_marginal_error=float(sinkhorn.row_marginal_error),
        column_marginal_error=float(sinkhorn.column_marginal_error),
        sinkhorn_iterations=int(sinkhorn.iterations),
    )


def source_conditioned_trace_fisher_coupling(
    source_probabilities: torch.Tensor,
    atom_choices: torch.Tensor,
    atom_active_masks: torch.Tensor,
    atom_weights: torch.Tensor,
    *,
    projection_eps: float,
    projection_sharpness: float = 1.0,
    resample_offset: float = 0.5,
    min_one_capacity: bool = True,
) -> SourceConditionedTraceCoupling:
    """Couple source particles to sharp trace atoms with source-conditioned targets."""
    sources = torch.as_tensor(source_probabilities).float()
    choices = torch.as_tensor(atom_choices, dtype=torch.long, device=sources.device)
    active_masks = torch.as_tensor(atom_active_masks, dtype=torch.bool, device=sources.device)
    weights = torch.as_tensor(atom_weights).float().to(sources.device).flatten()
    if sources.ndim != 3:
        raise ValueError("source_probabilities must have shape [source, block, action]")
    if choices.ndim != 2 or active_masks.shape != choices.shape:
        raise ValueError("atom_choices and atom_active_masks must have shape [atom, block]")
    if int(choices.shape[1]) != int(sources.shape[1]):
        raise ValueError("trace atoms and source probabilities must use the same block count")
    if int(weights.numel()) != int(choices.shape[0]):
        raise ValueError("atom_weights must contain one value per atom")
    if bool((active_masks.sum(dim=1) == 0).any().detach().cpu()):
        raise ValueError("every trace atom must have at least one active block")

    source_count = int(sources.shape[0])
    atom_count = int(choices.shape[0])
    capacity_atoms = capacity_resample_indices(
        weights,
        source_count,
        min_one=bool(min_one_capacity),
        offset=float(resample_offset),
    ).to(sources.device)
    atom_costs = torch.empty((source_count, atom_count), dtype=sources.dtype, device=sources.device)
    for source_index in range(source_count):
        for atom_index in range(atom_count):
            target = source_conditioned_trace_target_probabilities(
                sources[source_index],
                choices[atom_index],
                active_masks[atom_index],
                projection_eps=float(projection_eps),
                projection_sharpness=float(projection_sharpness),
            )
            atom_costs[source_index, atom_index] = block_fisher_squared_distance(
                sources[source_index],
                target,
                active_masks[atom_index],
            )
    costs = atom_costs.index_select(1, capacity_atoms)
    row_indices, column_indices = linear_sum_assignment(costs.detach().cpu().numpy())
    assigned_slot = torch.empty(source_count, dtype=torch.long, device=sources.device)
    assigned_slot[torch.as_tensor(row_indices, dtype=torch.long, device=sources.device)] = torch.as_tensor(
        column_indices,
        dtype=torch.long,
        device=sources.device,
    )
    assigned_atom = capacity_atoms.index_select(0, assigned_slot)
    materialized_targets = []
    assigned_masks = []
    for source_index in range(source_count):
        atom_index = int(assigned_atom[source_index].detach().cpu())
        assigned_masks.append(active_masks[atom_index])
        materialized_targets.append(source_conditioned_trace_target_probabilities(
            sources[source_index],
            choices[atom_index],
            active_masks[atom_index],
            projection_eps=float(projection_eps),
            projection_sharpness=float(projection_sharpness),
        ))
    target_probabilities = torch.stack(materialized_targets, dim=0)
    assigned_active_masks = torch.stack(assigned_masks, dim=0)
    source_indices = torch.arange(source_count, device=sources.device)
    pair_costs = costs[source_indices, assigned_slot]
    capacity_counts = torch.bincount(capacity_atoms, minlength=atom_count)
    return SourceConditionedTraceCoupling(
        capacity_atom_indices=capacity_atoms,
        assigned_slot_for_source=assigned_slot,
        assigned_atom_indices=assigned_atom,
        target_probabilities=target_probabilities,
        active_masks=assigned_active_masks,
        pair_costs=pair_costs,
        cost_matrix=costs,
        atom_cost_matrix=atom_costs,
        capacity_counts=capacity_counts,
    )


def source_preserving_fisher_coupling(
    source_probabilities: torch.Tensor,
    endpoint_probabilities: torch.Tensor,
    endpoint_active_masks: torch.Tensor,
    endpoint_weights: torch.Tensor,
    *,
    resample_offset: float = 0.5,
) -> SourcePreservingCoupling:
    """Build a finite coupling with exact empirical source occupancy.

    Tilted endpoints are first systematically resampled to the source particle
    count. A Hungarian assignment then pairs every source exactly once to one
    resampled endpoint slot using active-block Fisher distance.
    """
    sources = torch.as_tensor(source_probabilities).float()
    endpoints = torch.as_tensor(endpoint_probabilities).float().to(sources.device)
    active_masks = torch.as_tensor(endpoint_active_masks, dtype=torch.bool, device=sources.device)
    weights = torch.as_tensor(endpoint_weights).float().to(sources.device).flatten()
    if sources.ndim != 3 or endpoints.ndim != 3:
        raise ValueError("source and endpoint probabilities must have shape [particle, block, action]")
    if sources.shape[1:] != endpoints.shape[1:]:
        raise ValueError("source and endpoint charts must match")
    if active_masks.shape != endpoints.shape[:2]:
        raise ValueError("endpoint_active_masks must have shape [endpoint, block]")
    if int(weights.numel()) != int(endpoints.shape[0]):
        raise ValueError("endpoint_weights must contain one value per endpoint")
    if bool((active_masks.sum(dim=1) == 0).any().detach().cpu()):
        raise ValueError("every endpoint particle must have at least one active block")

    source_count = int(sources.shape[0])
    resampled = systematic_resample_indices(weights, source_count, offset=float(resample_offset))
    slot_endpoints = endpoints.index_select(0, resampled)
    slot_masks = active_masks.index_select(0, resampled)
    costs = torch.empty((source_count, source_count), dtype=sources.dtype, device=sources.device)
    for source_index in range(source_count):
        for slot_index in range(source_count):
            costs[source_index, slot_index] = block_fisher_squared_distance(
                sources[source_index],
                slot_endpoints[slot_index],
                slot_masks[slot_index],
            )

    row_indices, column_indices = linear_sum_assignment(costs.detach().cpu().numpy())
    assigned_slot = torch.empty(source_count, dtype=torch.long, device=sources.device)
    assigned_slot[torch.as_tensor(row_indices, dtype=torch.long, device=sources.device)] = torch.as_tensor(
        column_indices,
        dtype=torch.long,
        device=sources.device,
    )
    assigned_endpoint = resampled.index_select(0, assigned_slot)
    assigned_masks = active_masks.index_select(0, assigned_endpoint)
    assigned_endpoint_probabilities = endpoints.index_select(0, assigned_endpoint)
    materialized_targets = torch.where(
        assigned_masks[:, :, None],
        assigned_endpoint_probabilities,
        sources,
    )
    source_indices = torch.arange(source_count, device=sources.device)
    pair_costs = costs[source_indices, assigned_slot]
    resample_counts = torch.bincount(resampled, minlength=int(endpoints.shape[0]))
    return SourcePreservingCoupling(
        resampled_endpoint_indices=resampled,
        assigned_slot_for_source=assigned_slot,
        assigned_endpoint_indices=assigned_endpoint,
        target_probabilities=materialized_targets,
        active_masks=assigned_masks,
        pair_costs=pair_costs,
        cost_matrix=costs,
        resample_counts=resample_counts,
    )


def fisher_endpoint_map_loss(
    predicted_probabilities: torch.Tensor,
    target_probabilities: torch.Tensor,
    active_masks: torch.Tensor,
    *,
    inactive_identity_weight: float = 0.0,
) -> torch.Tensor:
    """Squared Fisher map loss for consistently paired endpoint targets."""
    predicted = torch.as_tensor(predicted_probabilities).float()
    target = torch.as_tensor(target_probabilities).float().to(predicted.device)
    active = torch.as_tensor(active_masks, dtype=torch.bool, device=predicted.device)
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("predicted and target probabilities must have shape [batch, block, action]")
    if active.shape != predicted.shape[:2]:
        raise ValueError("active_masks must have shape [batch, block]")
    identity_weight = float(inactive_identity_weight)
    if identity_weight < 0.0:
        raise ValueError("inactive identity weight must be non-negative")
    block_loss = _block_fisher_squared(predicted, target)
    weights = active.float() + identity_weight * (~active).float()
    if not bool((weights.sum(dim=1) > 0).all().detach().cpu()):
        raise ValueError("every endpoint map example must supervise at least one block")
    return (block_loss * weights).sum() / weights.sum().clamp_min(1.0)
