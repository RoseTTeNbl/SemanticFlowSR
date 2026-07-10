"""Shared semantic-mass posterior projection utilities.

The core object is a complete-trace distribution ``q_theta(z)`` and a target
kernel ``K_D(z)``.  The tilted posterior is ``q_plus(z) proportional q K``.
Construction families differ only in how complete traces project back to their
local parameterization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch

SEMANTIC_SIGNATURE_VERSION = "semantic_signature_curve_sortdiff_stats"


class ConstructionFamily(Protocol):
    """Common interface for semantic-mass construction parameterizations.

    The branch-specific training scripts own the expensive domain logic
    (grammar masks, graph execution, batching).  This protocol records the
    shared object used by the theory: a parameter ``theta`` induces a complete
    trace distribution, complete traces are scored by a target semantic kernel,
    and the posterior ``q_plus - q`` is projected back to the local chart.
    """

    def sample_traces(self, state: Any, task: Any, count: int, generator: torch.Generator) -> Sequence[Any]:
        """Sample complete expressions/traces from the current construction state."""
        ...

    def log_prob(self, state: Any, trace: Any) -> torch.Tensor:
        """Return the current construction log probability of a complete trace."""
        ...

    def score_components(self, trace: Any) -> Sequence[tuple[int, int]]:
        """Return local chart factors used by the sampled complete trace."""
        ...

    def decode_execute(self, trace: Any, x: torch.Tensor) -> tuple[Any, torch.Tensor]:
        """Decode a complete trace and evaluate its semantic pushforward."""
        ...

    def posterior_project(self, *args: Any, **kwargs: Any) -> Any:
        """Project posterior complete-trace mass back to this chart."""
        ...


def semantic_kernel_from_energy(energies: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return the target-neighborhood kernel ``K_D(z)=exp(-d_y(s_z)/tau)``.

    This is the semantic pushforward mass kernel from the theory.  The
    posterior weights below use a shifted log-sum-exp form for numerical
    stability, but diagnostics should still report the absolute semantic mass
    induced by this kernel.
    """
    energies = torch.as_tensor(energies, dtype=torch.float32)
    finite = torch.isfinite(energies)
    if not bool(finite.all().item()):
        fallback = energies[finite].max() if bool(finite.any().item()) else torch.tensor(0.0, device=energies.device)
        energies = torch.where(finite, energies, fallback)
    tau = max(float(temperature), 1.0e-8)
    return torch.exp((-energies / tau).clamp(-80.0, 30.0))


def posterior_weights_from_energy(energies: torch.Tensor, temperature: float) -> torch.Tensor:
    energies = torch.as_tensor(energies, dtype=torch.float32)
    finite = torch.isfinite(energies)
    if not bool(finite.all().item()):
        fallback = energies[finite].max() if bool(finite.any().item()) else torch.tensor(0.0, device=energies.device)
        energies = torch.where(finite, energies, fallback)
    tau = max(float(temperature), 1.0e-8)
    return torch.softmax(-(energies - energies.min()) / tau, dim=0)


def target_rank_utilities(target_distances: torch.Tensor) -> torch.Tensor:
    """Return signed rank utilities induced by target-semantic distance.

    The closest semantic samples receive values near ``+1`` and the farthest
    receive values near ``-1``.  The uniform prior mean is zero by construction
    when there is more than one sample, so a positive posterior mean directly
    means that semantic tilt has moved mass toward target-near samples and away
    from target-far samples.
    """
    target = torch.as_tensor(target_distances, dtype=torch.float32)
    n = int(target.numel())
    if n <= 1:
        return torch.zeros_like(target)
    finite = torch.isfinite(target)
    if not bool(finite.all().item()):
        fallback = target[finite].max() if bool(finite.any().item()) else torch.tensor(0.0, device=target.device)
        target = torch.where(finite, target, fallback)
    if float((target.max() - target.min()).detach().cpu().item()) <= 1.0e-8:
        return torch.zeros_like(target)
    order = torch.argsort(target)
    ranks = torch.empty(n, dtype=torch.float32, device=target.device)
    ranks[order] = torch.arange(n, dtype=torch.float32, device=target.device)
    return 1.0 - 2.0 * ranks / float(max(n - 1, 1))


def target_soft_utilities(
    target_distances: torch.Tensor,
    *,
    temperature: float,
    reference_distances: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a smooth target-neighborhood utility in ``[0, 1]``.

    Hard top/bottom quantiles are useful diagnostics but can be noisy at small
    sample count.  This utility uses the semantic-space kernel induced by the
    target distance and normalizes it on the prior sampled semantic support.
    A higher mean therefore directly means more probability mass on semantics
    close to the target.  Background/unrelated semantic mass is measured by the
    independent distance-based utility below, not by ``1 - utility``.
    """
    target = torch.as_tensor(target_distances, dtype=torch.float32)
    ref = target if reference_distances is None else torch.as_tensor(reference_distances, dtype=torch.float32, device=target.device)
    finite_ref = torch.isfinite(ref)
    if not bool(finite_ref.all().item()):
        fallback = ref[finite_ref].max() if bool(finite_ref.any().item()) else torch.tensor(0.0, device=target.device)
        ref = torch.where(finite_ref, ref, fallback)
    finite_target = torch.isfinite(target)
    if not bool(finite_target.all().item()):
        fallback = ref.max() if int(ref.numel()) else torch.tensor(0.0, device=target.device)
        target = torch.where(finite_target, target, fallback)
    if int(target.numel()) == 0 or int(ref.numel()) == 0:
        return torch.zeros_like(target)
    ref_kernel = semantic_kernel_from_energy(ref, temperature).to(target.device)
    kernel = semantic_kernel_from_energy(target, temperature).to(target.device)
    denom = (ref_kernel.max() - ref_kernel.min()).clamp_min(1.0e-8)
    if float((ref_kernel.max() - ref_kernel.min()).detach().cpu().item()) <= 1.0e-8:
        return torch.zeros_like(target)
    return ((kernel - ref_kernel.min()) / denom).clamp(0.0, 1.0)


def target_background_utilities(
    target_distances: torch.Tensor,
    *,
    reference_distances: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a continuous far/background semantic utility in ``[0, 1]``.

    ``target_soft_utilities`` measures smooth target-neighborhood mass through
    the semantic kernel.  This companion statistic measures the opposite
    direction on the semantic-distance support itself: low values are target
    related, high values are far/background semantics.  It is intentionally not
    defined as ``1 - target_soft_utility`` so that the diagnostics separately
    check "top target semantics up" and "far/unrelated semantics down".
    """
    target = torch.as_tensor(target_distances, dtype=torch.float32)
    ref = target if reference_distances is None else torch.as_tensor(reference_distances, dtype=torch.float32, device=target.device)
    finite_ref = torch.isfinite(ref)
    if not bool(finite_ref.all().item()):
        fallback = ref[finite_ref].max() if bool(finite_ref.any().item()) else torch.tensor(0.0, device=target.device)
        ref = torch.where(finite_ref, ref, fallback)
    finite_target = torch.isfinite(target)
    if not bool(finite_target.all().item()):
        fallback = ref.max() if int(ref.numel()) else torch.tensor(0.0, device=target.device)
        target = torch.where(finite_target, target, fallback)
    if int(target.numel()) == 0 or int(ref.numel()) == 0:
        return torch.zeros_like(target)
    denom = (ref.max() - ref.min()).clamp_min(1.0e-8)
    if float((ref.max() - ref.min()).detach().cpu().item()) <= 1.0e-8:
        return torch.zeros_like(target)
    return ((target - ref.min()) / denom).clamp(0.0, 1.0)


def target_semantic_contrast_utilities(
    target_distances: torch.Tensor,
    *,
    temperature: float,
    reference_distances: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the signed target-vs-far semantic utility.

    This is the compact mean statistic for the semantic pushforward tilt:
    target-neighborhood kernel utility should increase, while independent
    far/background distance utility should decrease.  Positive posterior mean
    improvement therefore directly encodes "raise target-top semantics and
    weaken unrelated/far semantics".
    """
    soft = target_soft_utilities(
        target_distances,
        temperature=float(temperature),
        reference_distances=reference_distances,
    )
    far = target_background_utilities(
        target_distances,
        reference_distances=reference_distances,
    ).to(soft.device)
    return soft - far


def target_top_far_tail_utilities(
    target_distances: torch.Tensor,
    *,
    temperature: float,
    top_fraction: float = 0.25,
    reference_distances: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a smooth signed utility focused on target-top and far-tail mass.

    ``target_semantic_contrast_utilities`` is a global smooth statistic.  This
    companion statistic is closer to the intended semantic-space effect: raise
    probability on the best target-neighborhood samples and weaken probability
    on the far/unrelated tail of the current sampled semantic support.

    The top/far neighborhoods are defined on the reference support, then
    smoothed by sigmoids so the mean statistic is less brittle than hard
    quantile mass.  Positive posterior mean improvement means mass has moved
    toward the target top region and/or away from the far tail.
    """
    target = torch.as_tensor(target_distances, dtype=torch.float32)
    ref = target if reference_distances is None else torch.as_tensor(reference_distances, dtype=torch.float32, device=target.device)
    finite_ref = torch.isfinite(ref)
    if not bool(finite_ref.all().item()):
        fallback = ref[finite_ref].max() if bool(finite_ref.any().item()) else torch.tensor(0.0, device=target.device)
        ref = torch.where(finite_ref, ref, fallback)
    finite_target = torch.isfinite(target)
    if not bool(finite_target.all().item()):
        fallback = ref.max() if int(ref.numel()) else torch.tensor(0.0, device=target.device)
        target = torch.where(finite_target, target, fallback)
    n = int(ref.numel())
    if int(target.numel()) == 0 or n <= 1:
        return torch.zeros_like(target)
    ref_range = ref.max() - ref.min()
    if float(ref_range.detach().cpu().item()) <= 1.0e-8:
        return torch.zeros_like(target)
    frac = min(max(float(top_fraction), 1.0e-6), 0.5)
    k = max(1, min(n, int(math.ceil(frac * float(n)))))
    order = torch.sort(ref).values
    top_threshold = order[k - 1]
    far_threshold = order[n - k]
    # The scale is semantic-distance based, not ODE/path based.  It keeps the
    # utility smooth while still making the prior top/far support meaningful.
    tau = max(float(temperature), 1.0e-8)
    scale = torch.clamp(ref_range * max(0.05, min(0.50, tau)), min=1.0e-6)
    top_gate = torch.sigmoid((top_threshold - target) / scale)
    far_gate = torch.sigmoid((target - far_threshold) / scale)
    return (top_gate - far_gate).clamp(-1.0, 1.0)


def projected_target_rank_utilities(
    prior_target_distances: torch.Tensor,
    projected_target_distances: torch.Tensor,
) -> torch.Tensor:
    """Map projected semantic distances onto the prior rank-utility scale."""
    prior = torch.as_tensor(prior_target_distances, dtype=torch.float32)
    projected = torch.as_tensor(projected_target_distances, dtype=torch.float32, device=prior.device)
    n = int(prior.numel())
    if n <= 1 or int(projected.numel()) == 0:
        return torch.zeros_like(projected)
    finite_prior = torch.isfinite(prior)
    if not bool(finite_prior.all().item()):
        fallback = prior[finite_prior].max() if bool(finite_prior.any().item()) else torch.tensor(0.0, device=prior.device)
        prior = torch.where(finite_prior, prior, fallback)
    finite_projected = torch.isfinite(projected)
    if not bool(finite_projected.all().item()):
        fallback = projected[finite_projected].max() if bool(finite_projected.any().item()) else prior.max()
        projected = torch.where(finite_projected, projected, fallback)
    if float((prior.max() - prior.min()).detach().cpu().item()) <= 1.0e-8:
        return torch.zeros_like(projected)
    sorted_prior = torch.sort(prior).values
    ranks = torch.searchsorted(sorted_prior, projected, right=False).to(dtype=torch.float32)
    ranks = ranks.clamp(0.0, float(n - 1))
    return 1.0 - 2.0 * ranks / float(max(n - 1, 1))


def semantic_tilt_energy(
    penalty_energies: torch.Tensor,
    target_distances: torch.Tensor | None = None,
    *,
    mode: str = "target_distance",
) -> torch.Tensor:
    """Return the scalar energy used for the semantic KL tilt.

    The main theory defines the tilt on the semantic pushforward distribution:
    the Radon-Nikodym factor is ``exp(-d_y(s)/tau)``.  Penalty-aware energies
    remain useful for diagnostics and invalid-sample fallbacks, but they should
    not silently replace the target semantic metric in the mainline.
    """
    penalty = torch.as_tensor(penalty_energies, dtype=torch.float32)
    if str(mode) == "penalized_energy" or target_distances is None:
        out = penalty
    elif str(mode) == "target_distance":
        out = torch.as_tensor(target_distances, dtype=torch.float32, device=penalty.device)
        if int(out.numel()) != int(penalty.numel()):
            raise ValueError("target_distances must have the same length as penalty_energies")
    else:
        raise ValueError(f"unknown semantic tilt energy mode: {mode}")
    finite = torch.isfinite(out)
    if not bool(finite.all().item()):
        fallback = out[finite].max() if bool(finite.any().item()) else (
            penalty[torch.isfinite(penalty)].max() if bool(torch.isfinite(penalty).any().item()) else torch.tensor(0.0, device=penalty.device)
        )
        out = torch.where(finite, out, fallback.to(out.device))
    return out


def _safe_semantic_values(values: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.float32).flatten()
    return torch.nan_to_num(values, nan=0.0, posinf=1.0e6, neginf=-1.0e6).clamp(-1.0e6, 1.0e6)


def _standardize_semantic_values(values: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    values = _safe_semantic_values(values)
    if int(values.numel()) <= 1:
        return torch.zeros_like(values)
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(float(eps))


def _signed_log_scalar(value: torch.Tensor) -> torch.Tensor:
    value = torch.nan_to_num(torch.as_tensor(value, dtype=torch.float32), nan=0.0, posinf=1.0e6, neginf=-1.0e6).clamp(-1.0e6, 1.0e6)
    return value.sign() * value.abs().log1p() / 8.0


def _semantic_corr_feature(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    v = _standardize_semantic_values(values)
    b = _standardize_semantic_values(basis).to(v.device)
    if int(v.numel()) == 0 or int(b.numel()) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=v.device)
    n = min(int(v.numel()), int(b.numel()))
    return (v[:n] * b[:n]).mean().clamp(-10.0, 10.0) / 4.0


def semantic_signature_vector(values: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
    """Return the inference-available semantic signature ``xi(expr(X), X)``.

    The semantic KL tilt should be defined in semantic space, not by an
    unrelated expression-space score.  A raw normalized output vector is often
    too weak for SR because structurally different expressions can approximate
    the sampled values.  This signature keeps the normalized curve but also
    adds variable-order shape information and small global statistics.  It uses
    only ``X`` and the evaluated expression values, so it is available during
    endpoint collection and inference diagnostics.
    """
    raw = _safe_semantic_values(values)
    device = raw.device
    curve = _standardize_semantic_values(raw)
    parts: list[torch.Tensor] = [curve]

    if x is not None:
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        if x_t.ndim == 1:
            x_t = x_t.view(-1, 1)
        if x_t.ndim == 2 and int(x_t.shape[0]) == int(curve.numel()):
            for dim in range(int(x_t.shape[1])):
                order = torch.argsort(_safe_semantic_values(x_t[:, dim]))
                ordered = curve[order]
                if int(ordered.numel()) >= 2:
                    first = torch.diff(ordered)
                    parts.append(0.50 * _standardize_semantic_values(first))
                if int(ordered.numel()) >= 3:
                    second = torch.diff(torch.diff(ordered))
                    parts.append(0.25 * _standardize_semantic_values(second))

            raw_std = raw.std(unbiased=False).clamp_min(1.0e-6) if int(raw.numel()) > 1 else raw.new_tensor(0.0)
            z = _standardize_semantic_values(raw)
            if int(z.numel()) > 0:
                skew = (z ** 3).mean().clamp(-20.0, 20.0) / 10.0
                kurt = (z ** 4).mean().clamp(0.0, 200.0).log1p() / 8.0
            else:
                skew = raw.new_tensor(0.0)
                kurt = raw.new_tensor(0.0)
            stats: list[torch.Tensor] = [
                _signed_log_scalar(raw.mean()).to(device),
                raw_std.clamp(0.0, 1.0e6).log1p() / 8.0,
                _signed_log_scalar(raw.min()).to(device) if int(raw.numel()) else raw.new_tensor(0.0),
                _signed_log_scalar(raw.max()).to(device) if int(raw.numel()) else raw.new_tensor(0.0),
                skew.to(device),
                kurt.to(device),
            ]
            for dim in range(int(x_t.shape[1])):
                x_col = torch.nan_to_num(x_t[:, dim].float(), nan=0.0, posinf=1.0e6, neginf=-1.0e6).clamp(-1.0e6, 1.0e6)
                bases = (
                    x_col,
                    x_col * x_col,
                    torch.sin(x_col),
                    torch.cos(x_col),
                )
                stats.extend(_semantic_corr_feature(raw, basis).to(device) for basis in bases)
            parts.append(0.50 * torch.stack(stats).float())

    return torch.cat([part.flatten().to(device=device, dtype=torch.float32) for part in parts], dim=0)


def semantic_signature_distance(values: torch.Tensor, target: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
    """Squared distance ``d_y`` used by the semantic-space KL tilt.

    This is a grouped semantic metric, not an unweighted mean over the
    concatenated signature vector.  The normalized output curve remains the
    dominant term, while variable-sorted first/second differences and compact
    global statistics add shape/dependency pressure.  Keeping the groups
    explicit avoids the long signature vector diluting the target-output term.
    """
    pred = _safe_semantic_values(values)
    tgt = _safe_semantic_values(target).to(pred.device)
    if int(pred.numel()) != int(tgt.numel()):
        raise ValueError("values and target must have matching lengths")
    pred_curve = _standardize_semantic_values(pred)
    tgt_curve = _standardize_semantic_values(tgt).to(pred.device)
    total = ((pred_curve - tgt_curve) ** 2).mean()

    if x is not None:
        x_t = torch.as_tensor(x, dtype=torch.float32, device=pred.device)
        if x_t.ndim == 1:
            x_t = x_t.view(-1, 1)
        if x_t.ndim == 2 and int(x_t.shape[0]) == int(pred_curve.numel()):
            first_terms: list[torch.Tensor] = []
            second_terms: list[torch.Tensor] = []
            for dim in range(int(x_t.shape[1])):
                order = torch.argsort(_safe_semantic_values(x_t[:, dim]))
                pred_ordered = pred_curve[order]
                tgt_ordered = tgt_curve[order]
                if int(pred_ordered.numel()) >= 2:
                    pred_first = _standardize_semantic_values(torch.diff(pred_ordered))
                    tgt_first = _standardize_semantic_values(torch.diff(tgt_ordered))
                    first_terms.append(((pred_first - tgt_first) ** 2).mean())
                if int(pred_ordered.numel()) >= 3:
                    pred_second = _standardize_semantic_values(torch.diff(torch.diff(pred_ordered)))
                    tgt_second = _standardize_semantic_values(torch.diff(torch.diff(tgt_ordered)))
                    second_terms.append(((pred_second - tgt_second) ** 2).mean())
            if first_terms:
                total = total + 0.25 * torch.stack(first_terms).mean()
            if second_terms:
                total = total + 0.10 * torch.stack(second_terms).mean()

            pred_sig = semantic_signature_vector(pred, x_t)
            tgt_sig = semantic_signature_vector(tgt, x_t).to(pred_sig.device)
            curve_len = int(pred_curve.numel())
            diff_len = 0
            for _dim in range(int(x_t.shape[1])):
                diff_len += max(curve_len - 1, 0)
                diff_len += max(curve_len - 2, 0)
            stats_start = min(curve_len + diff_len, int(pred_sig.numel()), int(tgt_sig.numel()))
            if stats_start < int(pred_sig.numel()) and stats_start < int(tgt_sig.numel()):
                total = total + 0.15 * ((pred_sig[stats_start:] - tgt_sig[stats_start:]) ** 2).mean()
    return total


def semantic_centroid_diagnostics(
    semantics: torch.Tensor,
    weights: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Check whether the weighted semantic pushforward mean moves to target.

    This is intentionally a diagnostic, not the definition of the posterior.
    The posterior is still sample-wise KL tilt by target distance.  The centroid
    fields make it visible whether that posterior also moves the first moment
    of the induced semantic distribution in the desired direction.
    """
    sem = torch.as_tensor(semantics, dtype=torch.float32)
    if sem.ndim == 1:
        sem = sem.view(1, -1)
    weights = torch.as_tensor(weights, dtype=torch.float32, device=sem.device).flatten()
    target = torch.as_tensor(target, dtype=torch.float32, device=sem.device).flatten()
    if int(sem.numel()) == 0 or int(weights.numel()) == 0 or int(target.numel()) == 0:
        return {
            "semantic_centroid_prior_distance": 0.0,
            "semantic_centroid_posterior_distance": 0.0,
            "semantic_centroid_distance_improvement": 0.0,
            "semantic_centroid_prior_corr": 0.0,
            "semantic_centroid_posterior_corr": 0.0,
            "semantic_centroid_corr_improvement": 0.0,
        }
    if int(sem.shape[0]) != int(weights.numel()):
        raise ValueError("semantic sample count must match weights")
    if int(sem.shape[1]) != int(target.numel()):
        raise ValueError("semantic dimension must match target")
    sem = torch.nan_to_num(sem, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    weights = weights / weights.sum().clamp_min(1.0e-8)
    prior_centroid = sem.mean(dim=0)
    posterior_centroid = (weights[:, None] * sem).sum(dim=0)
    prior_dist = ((prior_centroid - target) ** 2).mean()
    posterior_dist = ((posterior_centroid - target) ** 2).mean()
    prior_corr = (prior_centroid * target).mean()
    posterior_corr = (posterior_centroid * target).mean()
    return {
        "semantic_centroid_prior_distance": float(prior_dist.detach().cpu().item()),
        "semantic_centroid_posterior_distance": float(posterior_dist.detach().cpu().item()),
        "semantic_centroid_distance_improvement": float((prior_dist - posterior_dist).detach().cpu().item()),
        "semantic_centroid_prior_corr": float(prior_corr.detach().cpu().item()),
        "semantic_centroid_posterior_corr": float(posterior_corr.detach().cpu().item()),
        "semantic_centroid_corr_improvement": float((posterior_corr - prior_corr).detach().cpu().item()),
    }


def projected_semantic_centroid_diagnostics(
    prior_semantics: torch.Tensor,
    projected_semantics: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Compare semantic first moments before and after endpoint projection."""
    prior = torch.as_tensor(prior_semantics, dtype=torch.float32)
    projected = torch.as_tensor(projected_semantics, dtype=torch.float32, device=prior.device)
    if prior.ndim == 1:
        prior = prior.view(1, -1)
    if projected.ndim == 1:
        projected = projected.view(1, -1)
    target = torch.as_tensor(target, dtype=torch.float32, device=prior.device).flatten()
    if int(prior.numel()) == 0 or int(projected.numel()) == 0 or int(target.numel()) == 0:
        return {
            "projected_semantic_centroid_prior_distance": 0.0,
            "projected_semantic_centroid_distance": 0.0,
            "projected_semantic_centroid_distance_improvement": 0.0,
            "projected_semantic_centroid_prior_corr": 0.0,
            "projected_semantic_centroid_corr": 0.0,
            "projected_semantic_centroid_corr_improvement": 0.0,
        }
    if int(prior.shape[1]) != int(target.numel()) or int(projected.shape[1]) != int(target.numel()):
        raise ValueError("semantic dimension must match target")
    prior = torch.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
    projected = torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    prior_centroid = prior.mean(dim=0)
    projected_centroid = projected.mean(dim=0)
    prior_dist = ((prior_centroid - target) ** 2).mean()
    projected_dist = ((projected_centroid - target) ** 2).mean()
    prior_corr = (prior_centroid * target).mean()
    projected_corr = (projected_centroid * target).mean()
    return {
        "projected_semantic_centroid_prior_distance": float(prior_dist.detach().cpu().item()),
        "projected_semantic_centroid_distance": float(projected_dist.detach().cpu().item()),
        "projected_semantic_centroid_distance_improvement": float((prior_dist - projected_dist).detach().cpu().item()),
        "projected_semantic_centroid_prior_corr": float(prior_corr.detach().cpu().item()),
        "projected_semantic_centroid_corr": float(projected_corr.detach().cpu().item()),
        "projected_semantic_centroid_corr_improvement": float((projected_corr - prior_corr).detach().cpu().item()),
    }


def posterior_tilt_diagnostics(
    energies: torch.Tensor,
    weights: torch.Tensor,
    *,
    temperature: float | None = None,
    top_fraction: float = 0.25,
    target_distances: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summarize whether semantic posterior mass moves toward low energy.

    ``energies`` are the tilt energies used by the semantic kernel.  They may
    include invalid, collapse, or complexity penalties.  ``target_distances``
    are the pure target-semantic distances used to decide whether probability
    mass really moves toward the target neighborhood.  If omitted, energies are
    used for both roles for backward compatibility.
    """
    energies = torch.as_tensor(energies, dtype=torch.float32)
    weights = torch.as_tensor(weights, dtype=torch.float32, device=energies.device)
    if target_distances is None:
        target = energies
    else:
        target = torch.as_tensor(target_distances, dtype=torch.float32, device=energies.device)
        if int(target.numel()) != int(energies.numel()):
            raise ValueError("target_distances must have the same length as energies")
    if int(energies.numel()) == 0:
        return {
            "target_distance_prior_mean": 0.0,
            "target_distance_posterior_mean": 0.0,
            "target_distance_mean_improvement": 0.0,
            "target_distance_best": 0.0,
            "weighted_energy_mean": 0.0,
            "energy_mean_improvement": 0.0,
            "kernel_mass_prior_mean": 0.0,
            "kernel_mass_posterior_mean": 0.0,
            "kernel_mass_lift": 0.0,
            "top_mass_prior": 0.0,
            "top_mass_posterior": 0.0,
            "top_mass_lift": 0.0,
            "bottom_mass_prior": 0.0,
            "bottom_mass_posterior": 0.0,
            "bottom_mass_suppression": 0.0,
            "target_near_mass_prior": 0.0,
            "target_near_mass_posterior": 0.0,
            "target_near_mass_lift": 0.0,
            "target_far_mass_prior": 0.0,
            "target_far_mass_posterior": 0.0,
            "target_far_mass_suppression": 0.0,
            "target_top_mass_delta": 0.0,
            "target_far_mass_reduction": 0.0,
            "target_concentration_gain": 0.0,
            "target_near_far_contrast_prior_mean": 0.0,
            "target_near_far_contrast_posterior_mean": 0.0,
            "target_near_far_contrast_mean_improvement": 0.0,
            "target_top_tail_mass_contrast_prior_mean": 0.0,
            "target_top_tail_mass_contrast_posterior_mean": 0.0,
            "target_top_tail_mass_contrast_mean_improvement": 0.0,
            "target_rank_utility_prior_mean": 0.0,
            "target_rank_utility_posterior_mean": 0.0,
            "target_rank_utility_mean_improvement": 0.0,
            "target_soft_utility_prior_mean": 0.0,
            "target_soft_utility_posterior_mean": 0.0,
            "target_soft_utility_mean_improvement": 0.0,
            "target_background_utility_prior_mean": 0.0,
            "target_background_utility_posterior_mean": 0.0,
            "target_background_utility_reduction": 0.0,
            "target_far_utility_prior_mean": 0.0,
            "target_far_utility_posterior_mean": 0.0,
            "target_far_utility_reduction": 0.0,
            "target_semantic_contrast_utility_prior_mean": 0.0,
            "target_semantic_contrast_utility_posterior_mean": 0.0,
            "target_semantic_contrast_utility_mean_improvement": 0.0,
            "target_top_far_tail_utility_prior_mean": 0.0,
            "target_top_far_tail_utility_posterior_mean": 0.0,
            "target_top_far_tail_utility_mean_improvement": 0.0,
            "top_bottom_odds_lift": 0.0,
            "top_bottom_mass_ratio_lift": 0.0,
        }
    finite_energy = torch.isfinite(energies)
    if not bool(finite_energy.all().item()):
        fallback = energies[finite_energy].max() if bool(finite_energy.any().item()) else torch.tensor(0.0, device=energies.device)
        energies = torch.where(finite_energy, energies, fallback)
    finite_target = torch.isfinite(target)
    if not bool(finite_target.all().item()):
        fallback = target[finite_target].max() if bool(finite_target.any().item()) else energies.max()
        target = torch.where(finite_target, target, fallback)
    weights = weights / weights.sum().clamp_min(1.0e-8)
    order = torch.argsort(target)
    n = int(energies.numel())
    k = max(1, min(n, int(math.ceil(float(top_fraction) * float(n)))))
    top = order[:k]
    bottom = order[-k:]
    prior_top_mass = float(k) / float(max(n, 1))
    prior_bottom_mass = float(k) / float(max(n, 1))
    top_post = weights[top].sum()
    bottom_post = weights[bottom].sum()
    top_post_f = float(top_post.detach().cpu().item())
    bottom_post_f = float(bottom_post.detach().cpu().item())
    top_delta = top_post_f - prior_top_mass
    far_reduction = prior_bottom_mass - bottom_post_f
    concentration_gain = top_delta + far_reduction
    prior_contrast = prior_top_mass - prior_bottom_mass
    posterior_contrast = top_post_f - bottom_post_f
    contrast_improvement = posterior_contrast - prior_contrast
    # Main statistic: expectation of a semantic-space signed indicator
    # under q_plus minus its prior expectation under q.  This directly checks
    # "raise top target-neighborhood mass and suppress far-tail mass" without
    # relying on a semantic centroid.
    top_tail_mass_contrast_prior = prior_contrast
    top_tail_mass_contrast_posterior = posterior_contrast
    top_tail_mass_contrast_improvement = contrast_improvement
    rank_utility = target_rank_utilities(target).to(weights.device)
    rank_prior_mean = rank_utility.mean()
    rank_post_mean = (weights * rank_utility).sum()
    rank_improvement = rank_post_mean - rank_prior_mean
    if temperature is not None:
        soft_utility = target_soft_utilities(target, temperature=float(temperature)).to(weights.device)
        soft_prior_mean = soft_utility.mean()
        soft_post_mean = (weights * soft_utility).sum()
    else:
        soft_prior_mean = target.new_tensor(0.0)
        soft_post_mean = target.new_tensor(0.0)
    soft_improvement = soft_post_mean - soft_prior_mean
    background_utility = target_background_utilities(target).to(weights.device)
    background_prior = background_utility.mean()
    background_post = (weights * background_utility).sum()
    background_reduction = background_prior - background_post
    if temperature is not None:
        contrast_utility = target_semantic_contrast_utilities(target, temperature=float(temperature)).to(weights.device)
        contrast_prior_mean = contrast_utility.mean()
        contrast_post_mean = (weights * contrast_utility).sum()
    else:
        contrast_prior_mean = target.new_tensor(0.0)
        contrast_post_mean = target.new_tensor(0.0)
    contrast_utility_improvement = contrast_post_mean - contrast_prior_mean
    if temperature is not None:
        tail_utility = target_top_far_tail_utilities(
            target,
            temperature=float(temperature),
            top_fraction=float(top_fraction),
        ).to(weights.device)
        tail_prior_mean = tail_utility.mean()
        tail_post_mean = (weights * tail_utility).sum()
    else:
        tail_prior_mean = target.new_tensor(0.0)
        tail_post_mean = target.new_tensor(0.0)
    tail_utility_improvement = tail_post_mean - tail_prior_mean
    prior_mean = energies.mean()
    post_mean = (weights * energies).sum()
    target_prior_mean = target.mean()
    target_post_mean = (weights * target).sum()
    if temperature is not None:
        kernel = semantic_kernel_from_energy(energies, float(temperature)).to(weights.device)
        kernel_prior = kernel.mean()
        kernel_post = (weights * kernel).sum()
    else:
        kernel_prior = energies.new_tensor(0.0)
        kernel_post = energies.new_tensor(0.0)
    eps = 1.0e-8
    prior_odds = prior_top_mass / max(1.0 - prior_top_mass, eps)
    post_odds = top_post_f / max(1.0 - top_post_f, eps)
    prior_top_bottom_ratio = prior_top_mass / max(prior_bottom_mass, eps)
    post_top_bottom_ratio = top_post_f / max(bottom_post_f, eps)
    return {
        "energy_prior_mean": float(prior_mean.detach().cpu().item()),
        "weighted_energy_mean": float(post_mean.detach().cpu().item()),
        "energy_mean_improvement": float((prior_mean - post_mean).detach().cpu().item()),
        "target_distance_prior_mean": float(target_prior_mean.detach().cpu().item()),
        "target_distance_posterior_mean": float(target_post_mean.detach().cpu().item()),
        "target_distance_mean_improvement": float((target_prior_mean - target_post_mean).detach().cpu().item()),
        "target_distance_best": float(target.min().detach().cpu().item()),
        "kernel_mass_prior_mean": float(kernel_prior.detach().cpu().item()),
        "kernel_mass_posterior_mean": float(kernel_post.detach().cpu().item()),
        "kernel_mass_lift": float((kernel_post / kernel_prior.clamp_min(eps)).detach().cpu().item()) if temperature is not None else 0.0,
        "top_mass_prior": float(prior_top_mass),
        "top_mass_posterior": top_post_f,
        "top_mass_lift": top_post_f / max(prior_top_mass, eps),
        "bottom_mass_prior": float(prior_bottom_mass),
        "bottom_mass_posterior": bottom_post_f,
        "bottom_mass_suppression": bottom_post_f / max(prior_bottom_mass, eps),
        "target_near_mass_prior": float(prior_top_mass),
        "target_near_mass_posterior": top_post_f,
        "target_near_mass_lift": top_post_f / max(prior_top_mass, eps),
        "target_far_mass_prior": float(prior_bottom_mass),
        "target_far_mass_posterior": bottom_post_f,
        "target_far_mass_suppression": bottom_post_f / max(prior_bottom_mass, eps),
        "target_top_mass_delta": float(top_delta),
        "target_far_mass_reduction": float(far_reduction),
        "target_concentration_gain": float(concentration_gain),
        "target_near_far_contrast_prior_mean": float(prior_contrast),
        "target_near_far_contrast_posterior_mean": float(posterior_contrast),
        "target_near_far_contrast_mean_improvement": float(contrast_improvement),
        "target_top_tail_mass_contrast_prior_mean": float(top_tail_mass_contrast_prior),
        "target_top_tail_mass_contrast_posterior_mean": float(top_tail_mass_contrast_posterior),
        "target_top_tail_mass_contrast_mean_improvement": float(top_tail_mass_contrast_improvement),
        "target_rank_utility_prior_mean": float(rank_prior_mean.detach().cpu().item()),
        "target_rank_utility_posterior_mean": float(rank_post_mean.detach().cpu().item()),
        "target_rank_utility_mean_improvement": float(rank_improvement.detach().cpu().item()),
        "target_soft_utility_prior_mean": float(soft_prior_mean.detach().cpu().item()),
        "target_soft_utility_posterior_mean": float(soft_post_mean.detach().cpu().item()),
        "target_soft_utility_mean_improvement": float(soft_improvement.detach().cpu().item()),
        "target_background_utility_prior_mean": float(background_prior.detach().cpu().item()),
        "target_background_utility_posterior_mean": float(background_post.detach().cpu().item()),
        "target_background_utility_reduction": float(background_reduction.detach().cpu().item()),
        "target_far_utility_prior_mean": float(background_prior.detach().cpu().item()),
        "target_far_utility_posterior_mean": float(background_post.detach().cpu().item()),
        "target_far_utility_reduction": float(background_reduction.detach().cpu().item()),
        "target_semantic_contrast_utility_prior_mean": float(contrast_prior_mean.detach().cpu().item()),
        "target_semantic_contrast_utility_posterior_mean": float(contrast_post_mean.detach().cpu().item()),
        "target_semantic_contrast_utility_mean_improvement": float(contrast_utility_improvement.detach().cpu().item()),
        "target_top_far_tail_utility_prior_mean": float(tail_prior_mean.detach().cpu().item()),
        "target_top_far_tail_utility_posterior_mean": float(tail_post_mean.detach().cpu().item()),
        "target_top_far_tail_utility_mean_improvement": float(tail_utility_improvement.detach().cpu().item()),
        "top_bottom_odds_lift": float(post_odds / max(prior_odds, eps)),
        "top_bottom_mass_ratio_lift": float(post_top_bottom_ratio / max(prior_top_bottom_ratio, eps)),
    }


def projected_distribution_diagnostics(
    prior_energies: torch.Tensor,
    projected_energies: torch.Tensor,
    *,
    temperature: float,
    top_fraction: float = 0.25,
    prior_target_distances: torch.Tensor | None = None,
    projected_target_distances: torch.Tensor | None = None,
) -> dict[str, float]:
    """Check whether the projected parameter endpoint improves semantic mass.

    ``posterior_tilt_diagnostics`` verifies the ideal sample posterior
    ``q_theta^+``.  This helper verifies the practical approximation after that
    posterior is projected back to a parameterized distribution and re-sampled.
    It directly tests the desired behavior: target-near semantic samples should
    gain probability mass, while target-far samples should lose it.
    """
    prior = torch.as_tensor(prior_energies, dtype=torch.float32)
    projected = torch.as_tensor(projected_energies, dtype=torch.float32, device=prior.device)
    prior_target = prior if prior_target_distances is None else torch.as_tensor(prior_target_distances, dtype=torch.float32, device=prior.device)
    projected_target = projected if projected_target_distances is None else torch.as_tensor(projected_target_distances, dtype=torch.float32, device=prior.device)
    if int(prior_target.numel()) != int(prior.numel()):
        raise ValueError("prior_target_distances must have the same length as prior_energies")
    if int(projected_target.numel()) != int(projected.numel()):
        raise ValueError("projected_target_distances must have the same length as projected_energies")
    if int(prior.numel()) == 0 or int(projected.numel()) == 0:
        return {
            "projected_energy_mean": 0.0,
            "projected_energy_best": 0.0,
            "projected_energy_mean_improvement": 0.0,
            "projected_target_distance_mean": 0.0,
            "projected_target_distance_best": 0.0,
            "projected_target_distance_mean_improvement": 0.0,
            "projected_kernel_mass_mean": 0.0,
            "projected_kernel_mass_lift_vs_prior": 0.0,
            "projected_neighborhood_gap": 0.0,
            "projected_neighborhood_gap_valid": 0.0,
            "projected_top_neighborhood_mass": 0.0,
            "projected_top_neighborhood_lift": 0.0,
            "projected_irrelevant_mass": 0.0,
            "projected_irrelevant_suppression": 0.0,
            "projected_target_near_mass": 0.0,
            "projected_target_near_mass_lift": 0.0,
            "projected_target_far_mass": 0.0,
            "projected_target_far_mass_suppression": 0.0,
            "projected_target_top_mass_delta": 0.0,
            "projected_target_far_mass_reduction": 0.0,
            "projected_target_concentration_gain": 0.0,
            "projected_target_near_far_contrast_prior_mean": 0.0,
            "projected_target_near_far_contrast_mean": 0.0,
            "projected_target_near_far_contrast_mean_improvement": 0.0,
            "projected_target_top_tail_mass_contrast_prior_mean": 0.0,
            "projected_target_top_tail_mass_contrast_mean": 0.0,
            "projected_target_top_tail_mass_contrast_mean_improvement": 0.0,
            "projected_target_rank_utility_prior_mean": 0.0,
            "projected_target_rank_utility_mean": 0.0,
            "projected_target_rank_utility_mean_improvement": 0.0,
            "projected_target_soft_utility_prior_mean": 0.0,
            "projected_target_soft_utility_mean": 0.0,
            "projected_target_soft_utility_mean_improvement": 0.0,
            "projected_target_background_utility_prior_mean": 0.0,
            "projected_target_background_utility_mean": 0.0,
            "projected_target_background_utility_reduction": 0.0,
            "projected_target_far_utility_prior_mean": 0.0,
            "projected_target_far_utility_mean": 0.0,
            "projected_target_far_utility_reduction": 0.0,
            "projected_target_semantic_contrast_utility_prior_mean": 0.0,
            "projected_target_semantic_contrast_utility_mean": 0.0,
            "projected_target_semantic_contrast_utility_mean_improvement": 0.0,
            "projected_target_top_far_tail_utility_prior_mean": 0.0,
            "projected_target_top_far_tail_utility_mean": 0.0,
            "projected_target_top_far_tail_utility_mean_improvement": 0.0,
            "projected_top_bottom_odds_lift": 0.0,
            "projected_top_bottom_mass_ratio_lift": 0.0,
        }
    finite_prior = torch.isfinite(prior)
    finite_projected = torch.isfinite(projected)
    finite_prior_target = torch.isfinite(prior_target)
    finite_projected_target = torch.isfinite(projected_target)
    if not bool(finite_prior.all().item()):
        fallback = prior[finite_prior].max() if bool(finite_prior.any().item()) else torch.tensor(0.0, device=prior.device)
        prior = torch.where(finite_prior, prior, fallback)
    if not bool(finite_projected.all().item()):
        fallback = projected[finite_projected].max() if bool(finite_projected.any().item()) else torch.tensor(0.0, device=projected.device)
        projected = torch.where(finite_projected, projected, fallback)
    if not bool(finite_prior_target.all().item()):
        fallback = prior_target[finite_prior_target].max() if bool(finite_prior_target.any().item()) else prior.max()
        prior_target = torch.where(finite_prior_target, prior_target, fallback)
    if not bool(finite_projected_target.all().item()):
        fallback = projected_target[finite_projected_target].max() if bool(finite_projected_target.any().item()) else projected.max()
        projected_target = torch.where(finite_projected_target, projected_target, fallback)

    n = int(prior.numel())
    k = max(1, min(n, int(math.ceil(float(top_fraction) * float(n)))))
    order = torch.argsort(prior_target)
    top_threshold = prior_target[order[k - 1]]
    bottom_threshold = prior_target[order[n - k]]
    neighborhood_gap = float((bottom_threshold - top_threshold).detach().cpu().item())
    prior_top_mass = float(k) / float(max(n, 1))
    prior_bottom_mass = float(k) / float(max(n, 1))

    prior_kernel = semantic_kernel_from_energy(prior, float(temperature)).to(prior.device)
    projected_kernel = semantic_kernel_from_energy(projected, float(temperature)).to(prior.device)
    eps = 1.0e-8
    prior_soft = target_soft_utilities(prior_target, temperature=float(temperature)).to(prior.device)
    projected_soft = target_soft_utilities(
        projected_target,
        temperature=float(temperature),
        reference_distances=prior_target,
    ).to(prior.device)
    soft_prior_mean = float(prior_soft.mean().detach().cpu().item()) if int(prior_soft.numel()) else 0.0
    soft_projected_mean = float(projected_soft.mean().detach().cpu().item()) if int(projected_soft.numel()) else 0.0
    soft_improvement = soft_projected_mean - soft_prior_mean
    prior_background = target_background_utilities(prior_target).to(prior.device)
    projected_background = target_background_utilities(
        projected_target,
        reference_distances=prior_target,
    ).to(prior.device)
    background_prior_mean = float(prior_background.mean().detach().cpu().item()) if int(prior_background.numel()) else 0.0
    background_projected_mean = float(projected_background.mean().detach().cpu().item()) if int(projected_background.numel()) else 0.0
    background_reduction = background_prior_mean - background_projected_mean
    prior_contrast_utility = target_semantic_contrast_utilities(
        prior_target,
        temperature=float(temperature),
    ).to(prior.device)
    projected_contrast_utility = target_semantic_contrast_utilities(
        projected_target,
        temperature=float(temperature),
        reference_distances=prior_target,
    ).to(prior.device)
    contrast_utility_prior_mean = float(prior_contrast_utility.mean().detach().cpu().item()) if int(prior_contrast_utility.numel()) else 0.0
    contrast_utility_projected_mean = float(projected_contrast_utility.mean().detach().cpu().item()) if int(projected_contrast_utility.numel()) else 0.0
    contrast_utility_improvement = contrast_utility_projected_mean - contrast_utility_prior_mean
    prior_tail_utility = target_top_far_tail_utilities(
        prior_target,
        temperature=float(temperature),
        top_fraction=float(top_fraction),
    ).to(prior.device)
    projected_tail_utility = target_top_far_tail_utilities(
        projected_target,
        temperature=float(temperature),
        top_fraction=float(top_fraction),
        reference_distances=prior_target,
    ).to(prior.device)
    tail_utility_prior_mean = float(prior_tail_utility.mean().detach().cpu().item()) if int(prior_tail_utility.numel()) else 0.0
    tail_utility_projected_mean = float(projected_tail_utility.mean().detach().cpu().item()) if int(projected_tail_utility.numel()) else 0.0
    tail_utility_improvement = tail_utility_projected_mean - tail_utility_prior_mean
    if neighborhood_gap <= eps:
        # With tied/overlapping prior quantiles, a projected sample can satisfy
        # both "target-near" and "target-far" threshold tests.  In that case the
        # top/bottom neighborhood diagnostics are undefined; kernel, energy,
        # and smooth utility diagnostics remain valid.
        top_mass = projected.new_tensor(0.0)
        bottom_mass = projected.new_tensor(0.0)
        post_top = 0.0
        top_lift = 0.0
        bottom_suppression = 0.0
        top_delta = 0.0
        far_reduction = 0.0
        concentration_gain = 0.0
        prior_contrast = 0.0
        projected_contrast = 0.0
        contrast_improvement = 0.0
        top_tail_mass_contrast_prior = 0.0
        top_tail_mass_contrast_projected = 0.0
        top_tail_mass_contrast_improvement = 0.0
        projected_rank_mean = 0.0
        rank_improvement = 0.0
        odds_lift = 0.0
        ratio_lift = 0.0
        gap_valid = 0.0
    else:
        top_mass = (projected_target <= top_threshold).float().mean()
        bottom_mass = (projected_target >= bottom_threshold).float().mean()
        prior_odds = prior_top_mass / max(1.0 - prior_top_mass, eps)
        post_top = float(top_mass.detach().cpu().item())
        post_odds = post_top / max(1.0 - post_top, eps)
        prior_top_bottom_ratio = prior_top_mass / max(prior_bottom_mass, eps)
        post_top_bottom_ratio = post_top / max(float(bottom_mass.detach().cpu().item()), eps)
        top_lift = post_top / max(prior_top_mass, eps)
        bottom_mass_f = float(bottom_mass.detach().cpu().item())
        bottom_suppression = bottom_mass_f / max(prior_bottom_mass, eps)
        top_delta = post_top - prior_top_mass
        far_reduction = prior_bottom_mass - bottom_mass_f
        concentration_gain = top_delta + far_reduction
        prior_contrast = prior_top_mass - prior_bottom_mass
        projected_contrast = post_top - bottom_mass_f
        contrast_improvement = projected_contrast - prior_contrast
        top_tail_mass_contrast_prior = prior_contrast
        top_tail_mass_contrast_projected = projected_contrast
        top_tail_mass_contrast_improvement = contrast_improvement
        rank_values = projected_target_rank_utilities(prior_target, projected_target)
        projected_rank_mean = float(rank_values.mean().detach().cpu().item()) if int(rank_values.numel()) else 0.0
        # Prior target-rank utility is centered at zero by construction.
        rank_improvement = projected_rank_mean
        odds_lift = float(post_odds / max(prior_odds, eps))
        ratio_lift = float(post_top_bottom_ratio / max(prior_top_bottom_ratio, eps))
        gap_valid = 1.0
    return {
        "projected_energy_mean": float(projected.mean().detach().cpu().item()),
        "projected_energy_best": float(projected.min().detach().cpu().item()),
        "projected_energy_mean_improvement": float((prior.mean() - projected.mean()).detach().cpu().item()),
        "projected_target_distance_mean": float(projected_target.mean().detach().cpu().item()),
        "projected_target_distance_best": float(projected_target.min().detach().cpu().item()),
        "projected_target_distance_mean_improvement": float((prior_target.mean() - projected_target.mean()).detach().cpu().item()),
        "projected_kernel_mass_mean": float(projected_kernel.mean().detach().cpu().item()),
        "projected_kernel_mass_lift_vs_prior": float((projected_kernel.mean() / prior_kernel.mean().clamp_min(eps)).detach().cpu().item()),
        "projected_neighborhood_gap": float(neighborhood_gap),
        "projected_neighborhood_gap_valid": float(gap_valid),
        "projected_top_neighborhood_mass": post_top,
        "projected_top_neighborhood_lift": float(top_lift),
        "projected_irrelevant_mass": float(bottom_mass.detach().cpu().item()),
        "projected_irrelevant_suppression": float(bottom_suppression),
        "projected_target_near_mass": post_top,
        "projected_target_near_mass_lift": float(top_lift),
        "projected_target_far_mass": float(bottom_mass.detach().cpu().item()),
        "projected_target_far_mass_suppression": float(bottom_suppression),
        "projected_target_top_mass_delta": float(top_delta),
        "projected_target_far_mass_reduction": float(far_reduction),
        "projected_target_concentration_gain": float(concentration_gain),
        "projected_target_near_far_contrast_prior_mean": float(prior_contrast),
        "projected_target_near_far_contrast_mean": float(projected_contrast),
        "projected_target_near_far_contrast_mean_improvement": float(contrast_improvement),
        "projected_target_top_tail_mass_contrast_prior_mean": float(top_tail_mass_contrast_prior),
        "projected_target_top_tail_mass_contrast_mean": float(top_tail_mass_contrast_projected),
        "projected_target_top_tail_mass_contrast_mean_improvement": float(top_tail_mass_contrast_improvement),
        "projected_target_rank_utility_prior_mean": 0.0,
        "projected_target_rank_utility_mean": float(projected_rank_mean),
        "projected_target_rank_utility_mean_improvement": float(rank_improvement),
        "projected_target_soft_utility_prior_mean": float(soft_prior_mean),
        "projected_target_soft_utility_mean": float(soft_projected_mean),
        "projected_target_soft_utility_mean_improvement": float(soft_improvement),
        "projected_target_background_utility_prior_mean": float(background_prior_mean),
        "projected_target_background_utility_mean": float(background_projected_mean),
        "projected_target_background_utility_reduction": float(background_reduction),
        "projected_target_far_utility_prior_mean": float(background_prior_mean),
        "projected_target_far_utility_mean": float(background_projected_mean),
        "projected_target_far_utility_reduction": float(background_reduction),
        "projected_target_semantic_contrast_utility_prior_mean": float(contrast_utility_prior_mean),
        "projected_target_semantic_contrast_utility_mean": float(contrast_utility_projected_mean),
        "projected_target_semantic_contrast_utility_mean_improvement": float(contrast_utility_improvement),
        "projected_target_top_far_tail_utility_prior_mean": float(tail_utility_prior_mean),
        "projected_target_top_far_tail_utility_mean": float(tail_utility_projected_mean),
        "projected_target_top_far_tail_utility_mean_improvement": float(tail_utility_improvement),
        "projected_top_bottom_odds_lift": float(odds_lift),
        "projected_top_bottom_mass_ratio_lift": float(ratio_lift),
    }


def semantic_pushforward_acceptance_diagnostics(
    tilt_diag: dict[str, float],
    projected_diag: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return binary diagnostics for the intended semantic pushforward effect.

    The endpoint correction is only useful if the semantic KL tilt both raises
    target-near sample mass and suppresses target-far sample mass.  Mean energy
    and semantic-centroid shifts are diagnostics only: a few moderate samples
    can improve the mean, and a multimodal SR posterior can have an ambiguous
    first moment, while the target-neighborhood probability mass is still the
    actual object being tilted.
    """
    ideal_kernel = float(tilt_diag.get("kernel_mass_lift", 0.0))
    ideal_energy = float(tilt_diag.get("energy_mean_improvement", 0.0))
    ideal_top = float(tilt_diag.get("top_mass_lift", 0.0))
    ideal_bottom = float(tilt_diag.get("bottom_mass_suppression", 0.0))
    ideal_ratio = float(tilt_diag.get("top_bottom_mass_ratio_lift", 0.0))
    ideal_concentration = float(tilt_diag.get("target_concentration_gain", 0.0))
    ideal_contrast = float(tilt_diag.get("target_near_far_contrast_mean_improvement", ideal_concentration))
    ideal_top_tail_mass_contrast = float(
        tilt_diag.get("target_top_tail_mass_contrast_mean_improvement", ideal_contrast)
    )
    ideal_rank = float(tilt_diag.get("target_rank_utility_mean_improvement", ideal_contrast))
    ideal_soft = float(tilt_diag.get("target_soft_utility_mean_improvement", ideal_rank))
    ideal_background = float(tilt_diag.get("target_far_utility_reduction", tilt_diag.get("target_background_utility_reduction", ideal_soft)))
    ideal_semantic_contrast = float(
        tilt_diag.get(
            "target_semantic_contrast_utility_mean_improvement",
            min(ideal_soft, ideal_background),
        )
    )
    ideal_tail = float(
        tilt_diag.get(
            "target_top_far_tail_utility_mean_improvement",
            ideal_semantic_contrast,
        )
    )
    ideal_target_distance = float(tilt_diag.get("target_distance_mean_improvement", ideal_energy))
    ideal_ok = (
        ideal_kernel > 1.0
        and ideal_energy > 0.0
        and ideal_target_distance > 0.0
        and ideal_top > 1.0
        and ideal_bottom < 1.0
        and ideal_ratio > 1.0
        and ideal_concentration > 0.0
        and ideal_contrast > 0.0
        and ideal_top_tail_mass_contrast > 0.0
        and ideal_rank > 0.0
        and ideal_soft > 0.0
        and ideal_background > 0.0
        and ideal_semantic_contrast > 0.0
        and ideal_tail > 0.0
    )

    if projected_diag is None:
        return {
            "semantic_pushforward_ideal_accept": float(ideal_ok),
            "semantic_pushforward_projected_accept": 0.0,
            "semantic_pushforward_accept": float(ideal_ok),
        }

    projected_kernel = float(projected_diag.get("projected_kernel_mass_lift_vs_prior", 0.0))
    projected_energy = float(projected_diag.get("projected_energy_mean_improvement", 0.0))
    projected_target_distance = float(projected_diag.get("projected_target_distance_mean_improvement", projected_energy))
    gap_valid = float(projected_diag.get("projected_neighborhood_gap_valid", 0.0))
    projected_ok = (
        projected_kernel > 1.0
        and projected_energy > 0.0
        and projected_target_distance > 0.0
        and gap_valid > 0.5
        and float(projected_diag.get("projected_top_neighborhood_lift", 0.0)) > 1.0
        and float(projected_diag.get("projected_irrelevant_suppression", 0.0)) < 1.0
        and float(projected_diag.get("projected_top_bottom_mass_ratio_lift", 0.0)) > 1.0
        and float(projected_diag.get("projected_target_concentration_gain", 0.0)) > 0.0
        and float(
            projected_diag.get(
                "projected_target_near_far_contrast_mean_improvement",
                projected_diag.get("projected_target_concentration_gain", 0.0),
            )
        ) > 0.0
        and float(
            projected_diag.get(
                "projected_target_top_tail_mass_contrast_mean_improvement",
                projected_diag.get(
                    "projected_target_near_far_contrast_mean_improvement",
                    projected_diag.get("projected_target_concentration_gain", 0.0),
                ),
            )
        ) > 0.0
        and float(
            projected_diag.get(
                "projected_target_rank_utility_mean_improvement",
                projected_diag.get(
                    "projected_target_near_far_contrast_mean_improvement",
                    projected_diag.get("projected_target_concentration_gain", 0.0),
                ),
            )
        ) > 0.0
        and float(projected_diag.get("projected_target_soft_utility_mean_improvement", 0.0)) > 0.0
        and float(
            projected_diag.get(
                "projected_target_far_utility_reduction",
                projected_diag.get("projected_target_background_utility_reduction", 0.0),
            )
        ) > 0.0
        and float(
            projected_diag.get(
                "projected_target_semantic_contrast_utility_mean_improvement",
                min(
                    float(projected_diag.get("projected_target_soft_utility_mean_improvement", 0.0)),
                    float(
                        projected_diag.get(
                            "projected_target_far_utility_reduction",
                            projected_diag.get("projected_target_background_utility_reduction", 0.0),
                        )
                    ),
                ),
            )
        ) > 0.0
        and float(
            projected_diag.get(
                "projected_target_top_far_tail_utility_mean_improvement",
                projected_diag.get("projected_target_semantic_contrast_utility_mean_improvement", 0.0),
            )
        ) > 0.0
    )
    return {
        "semantic_pushforward_ideal_accept": float(ideal_ok),
        "semantic_pushforward_projected_accept": float(projected_ok),
        "semantic_pushforward_accept": float(ideal_ok and projected_ok),
    }


def semantic_mass_value(probs: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    probs = torch.as_tensor(probs, dtype=torch.float32)
    kernel = torch.as_tensor(kernel, dtype=torch.float32, device=probs.device)
    return (probs * kernel).sum()


def natural_gradient_probability_step(probs: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Exact categorical natural-gradient direction for log E_q[K]."""
    probs = torch.as_tensor(probs, dtype=torch.float32)
    probs = probs / probs.sum().clamp_min(1.0e-8)
    kernel = torch.as_tensor(kernel, dtype=torch.float32, device=probs.device)
    mass = semantic_mass_value(probs, kernel).clamp_min(1.0e-8)
    posterior = probs * kernel / mass
    return posterior - probs


def probability_correction_to_centered_logits(
    probs: torch.Tensor,
    dp: torch.Tensor,
    *,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    probs = torch.as_tensor(probs, dtype=torch.float32)
    dp = torch.as_tensor(dp, dtype=torch.float32, device=probs.device)
    dp = dp - probs * dp.sum()
    v = dp / probs.clamp_min(float(eps))
    return v - v.mean()


@dataclass(frozen=True)
class PosteriorTraceSample:
    choices: tuple[int, ...]
    active_blocks: tuple[int, ...]


def project_trace_posterior_to_blocks(
    block_probs: Sequence[torch.Tensor],
    samples: Sequence[PosteriorTraceSample],
    weights: torch.Tensor,
) -> list[torch.Tensor]:
    """Project ``q_plus - q`` samples to block/action probability corrections.

    Inactive coordinates receive exactly zero correction because their reward is
    independent of the sampled expression ancestor set.
    """
    weights = torch.as_tensor(weights, dtype=torch.float32)
    if int(weights.numel()) != len(samples):
        raise ValueError("weights/sample count mismatch")
    out: list[torch.Tensor] = []
    for bidx, p_in in enumerate(block_probs):
        p = torch.as_tensor(p_in, dtype=torch.float32)
        q = torch.zeros_like(p)
        mass = torch.zeros((), dtype=p.dtype, device=p.device)
        for sidx, sample in enumerate(samples):
            if int(bidx) not in set(int(v) for v in sample.active_blocks):
                continue
            action = int(sample.choices[int(bidx)])
            if 0 <= action < int(q.numel()):
                w = weights[int(sidx)].to(q.device)
                q[action] = q[action] + w
                mass = mass + w
        if float(mass.detach().cpu().item()) <= 1.0e-8:
            out.append(torch.zeros_like(p))
            continue
        posterior = q / mass.clamp_min(1.0e-8)
        out.append(mass * (posterior - p))
    return out


@dataclass
class GraphDagEdgeSimplexFamily:
    block_sizes: tuple[int, ...]

    def posterior_project(
        self,
        block_probs: Sequence[torch.Tensor],
        samples: Sequence[PosteriorTraceSample],
        weights: torch.Tensor,
    ) -> list[torch.Tensor]:
        if len(block_probs) != len(self.block_sizes):
            raise ValueError("block probability count does not match graph family")
        for idx, (p, size) in enumerate(zip(block_probs, self.block_sizes)):
            if int(torch.as_tensor(p).numel()) != int(size):
                raise ValueError(f"block {idx} size mismatch")
        return project_trace_posterior_to_blocks(block_probs, samples, weights)


@dataclass
class TokenPolicyFamily:
    vocab: tuple[str, ...]
    seq_len: int

    def posterior_project(
        self,
        token_probs: torch.Tensor,
        sequences: Sequence[Sequence[int]],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Posterior next-token marginal correction for fixed-length sequences."""
        probs = torch.as_tensor(token_probs, dtype=torch.float32)
        if probs.ndim != 2:
            raise ValueError("token_probs must have shape [seq_len, vocab]")
        weights = torch.as_tensor(weights, dtype=torch.float32, device=probs.device)
        if int(weights.numel()) != len(sequences):
            raise ValueError("weights/sequence count mismatch")
        q = torch.zeros_like(probs)
        for sidx, seq in enumerate(sequences):
            for pos in range(min(int(self.seq_len), len(seq))):
                tok = int(seq[pos])
                if 0 <= tok < int(probs.shape[1]):
                    q[pos, tok] = q[pos, tok] + weights[int(sidx)]
        row_mass = q.sum(dim=-1, keepdim=True)
        posterior = torch.where(row_mass > 1.0e-8, q / row_mass.clamp_min(1.0e-8), probs)
        return posterior - probs
