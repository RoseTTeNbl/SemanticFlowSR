"""Legacy local probability-shape teacher velocity.

Flow matching happens on explicit categorical probability shapes. The default
teacher path is Fisher-Rao via square-root sphere geodesics. Semantic
information does not change that path; it only calibrates the local velocity
error in the loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class DecisionTrace:
    group_id: str
    choice: int
    current_probs: torch.Tensor
    candidate_semantics: torch.Tensor | None
    predicted_sqrt_velocity: torch.Tensor | None
    initial_probs: torch.Tensor | None = None
    velocity_fn: Callable[[torch.Tensor, float], torch.Tensor] | None = None
    flow_time: float = 1.0
    candidate_keys: tuple[str, ...] = ()
    context_key: str = ""
    active: bool = False
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticTeacherResult:
    mass_velocity: torch.Tensor
    log_rate: torch.Tensor
    sqrt_velocity: torch.Tensor
    diagnostics: dict


@dataclass(frozen=True)
class TeacherPathState:
    current_probs: torch.Tensor
    mass_velocity: torch.Tensor
    log_rate: torch.Tensor
    sqrt_velocity: torch.Tensor
    diagnostics: dict


def teacher_path_state(
    initial_probs: torch.Tensor,
    candidate_semantics: torch.Tensor | None,
    target_probs: torch.Tensor,
    *,
    flow_time: float,
    geometry: str = "semantic",
    beta: float = 1.0,
    pinv_rtol: float = 1e-2,
    velocity_clip: float | None = None,
    min_prob: float = 1e-8,
) -> TeacherPathState:
    """Build the FM intermediate state and teacher velocity.

    ``fisher`` uses the square-root sphere geodesic induced by Fisher-Rao on
    the local categorical simplex. ``euclidean`` keeps the old linear chord as
    an ablation. ``semantic`` is accepted as a compatibility alias for
    ``fisher`` because semantics no longer define the probability path.
    """

    raw_p0 = torch.as_tensor(initial_probs).float().flatten()
    raw_q = torch.as_tensor(target_probs, dtype=raw_p0.dtype, device=raw_p0.device).flatten()
    if raw_p0.numel() != raw_q.numel():
        raise ValueError("initial_probs and target_probs must have the same length")
    support = raw_p0 > 0.0
    if not bool(support.any()):
        support = torch.ones_like(raw_p0, dtype=torch.bool)
    p0 = _simplex_vector(raw_p0[support], min_prob=min_prob)
    q = _simplex_vector(raw_q[support], min_prob=min_prob, device=p0.device, dtype=p0.dtype)
    t = float(max(0.0, min(1.0, float(flow_time))))
    geometry_key = _normalize_geometry(geometry)
    delta = q - p0
    diagnostics: dict = {
        "teacher_path_geometry": geometry_key,
        "probability_path_geometry": geometry_key,
        "teacher_path_time": float(t),
    }
    if geometry_key == "euclidean":
        current = (1.0 - t) * p0 + t * q
        mass_velocity = float(beta) * delta
        sqrt_velocity = 0.5 * mass_velocity / current.clamp_min(float(min_prob)).sqrt()
        log_rate = mass_velocity / current.clamp_min(float(min_prob))
        log_rate = log_rate - (current * log_rate).sum()
        mass_velocity = current * log_rate
        sqrt_velocity = 0.5 * mass_velocity / current.clamp_min(float(min_prob)).sqrt()
        diagnostics.update({
            "fisher_angle": 0.0,
            "semantic_null_residual_norm": 0.0,
            "semantic_path_null_velocity_norm": 0.0,
            "semantic_path_residual_norm": 0.0,
            "semantic_path_kernel_rank": 0,
            "semantic_kernel_rank": 0,
        })
    elif geometry_key == "fisher":
        current_sqrt, sqrt_velocity, angle = _fisher_sphere_state(
            p0,
            q,
            t=t,
            beta=float(beta),
            min_prob=float(min_prob),
        )
        current = current_sqrt.pow(2)
        current = current / current.sum().clamp_min(float(min_prob))
        mass_velocity = 2.0 * current_sqrt * sqrt_velocity
        mass_velocity = mass_velocity - mass_velocity.mean()
        log_rate = mass_velocity / current.clamp_min(float(min_prob))
        log_rate = log_rate - (current * log_rate).sum()
        mass_velocity = current * log_rate
        sqrt_velocity = 0.5 * mass_velocity / current.clamp_min(float(min_prob)).sqrt()
        diagnostics.update({
            "fisher_angle": float(angle),
            "semantic_null_residual_norm": 0.0,
            "semantic_path_null_velocity_norm": 0.0,
            "semantic_path_residual_norm": 0.0,
            "semantic_path_kernel_rank": 0,
            "semantic_kernel_rank": 0,
        })
    else:
        raise ValueError(f"unknown teacher_path_geometry: {geometry}")
    current = current.clamp_min(float(min_prob))
    current = current / current.sum().clamp_min(float(min_prob))
    unclipped_norm = sqrt_velocity.detach().norm()
    clip_value = None if velocity_clip is None else float(velocity_clip)
    velocity_scale = 1.0
    if clip_value is not None and clip_value > 0.0:
        norm_value = float(unclipped_norm.cpu().item())
        if norm_value > clip_value:
            velocity_scale = float(clip_value / max(norm_value, 1e-12))
            mass_velocity = mass_velocity * velocity_scale
            log_rate = log_rate * velocity_scale
            sqrt_velocity = sqrt_velocity * velocity_scale
    diagnostics.update({
        "semantic_teacher_velocity_norm": float(sqrt_velocity.detach().norm().cpu().item()),
        "semantic_teacher_unclipped_velocity_norm": float(unclipped_norm.cpu().item()),
        "semantic_teacher_velocity_scale": float(velocity_scale),
        "teacher_path_current_entropy": float(
            (-(current.detach().clamp_min(1e-12) * current.detach().clamp_min(1e-12).log()).sum()).cpu().item()
        ),
        "teacher_path_endpoint_l1": float((q.detach() - p0.detach()).abs().sum().cpu().item()),
        "teacher_path_state_l1_from_initial": float((current.detach() - p0.detach()).abs().sum().cpu().item()),
    })
    full_current = torch.zeros_like(raw_p0, device=p0.device, dtype=p0.dtype)
    full_mass_velocity = torch.zeros_like(full_current)
    full_log_rate = torch.zeros_like(full_current)
    full_sqrt_velocity = torch.zeros_like(full_current)
    active_support = support.to(device=p0.device)
    full_current[active_support] = current
    full_mass_velocity[active_support] = mass_velocity
    full_log_rate[active_support] = log_rate
    full_sqrt_velocity[active_support] = sqrt_velocity
    return TeacherPathState(
        current_probs=full_current,
        mass_velocity=full_mass_velocity,
        log_rate=full_log_rate,
        sqrt_velocity=full_sqrt_velocity,
        diagnostics=diagnostics,
    )


def semantic_only_teacher_velocity(
    current_probs: torch.Tensor,
    candidate_semantics: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    beta: float = 1.0,
    pinv_rtol: float = 1e-2,
    velocity_clip: float | None = None,
) -> SemanticTeacherResult:
    """Compute a quotient-space semantic teacher velocity.

    This intentionally omits the Fisher probability norm. The semantic Gram
    matrix can be singular; residual components outside the semantic range are
    projected away and reported as null residual.
    """

    p = torch.as_tensor(current_probs).float().flatten()
    q = torch.as_tensor(target_probs, dtype=p.dtype, device=p.device).flatten()
    if p.numel() != q.numel():
        raise ValueError("current_probs and target_probs must have the same length")
    p = p.clamp_min(1e-8)
    p = p / p.sum().clamp_min(1e-8)
    q = q.clamp_min(1e-8)
    q = q / q.sum().clamp_min(1e-8)
    sem = torch.as_tensor(candidate_semantics, dtype=p.dtype, device=p.device)
    if sem.ndim == 1:
        sem = sem.unsqueeze(1)
    if int(sem.shape[1]) != int(p.numel()):
        raise ValueError("candidate_semantics must have one column per candidate")
    sem = _normalize_columns(torch.nan_to_num(sem.float()))
    gram = sem.transpose(0, 1) @ sem / max(int(sem.shape[0]), 1)
    d = int(p.numel())
    eye = torch.eye(d, dtype=p.dtype, device=p.device)
    ones = torch.ones(d, dtype=p.dtype, device=p.device)
    center = eye - torch.outer(ones, ones) / float(max(d, 1))
    gram_tangent = center @ gram @ center
    residual = center @ (q.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())
    pinv = torch.linalg.pinv(gram_tangent, rtol=float(pinv_rtol))
    mass_velocity = float(beta) * (center @ (pinv @ residual))
    mass_velocity = mass_velocity - mass_velocity.mean()
    log_rate = mass_velocity / p.clamp_min(1e-8)
    log_rate = log_rate - (p * log_rate).sum()
    mass_velocity = p * log_rate
    sqrt_velocity = 0.5 * p.sqrt() * log_rate
    unclipped_norm = sqrt_velocity.detach().norm()
    clip_value = None if velocity_clip is None else float(velocity_clip)
    velocity_scale = 1.0
    if clip_value is not None and clip_value > 0.0:
        norm_value = float(unclipped_norm.cpu().item())
        if norm_value > clip_value:
            velocity_scale = float(clip_value / max(norm_value, 1e-12))
            mass_velocity = mass_velocity * velocity_scale
            log_rate = log_rate * velocity_scale
            sqrt_velocity = sqrt_velocity * velocity_scale
    projected = gram_tangent @ (pinv @ residual)
    null = residual - projected
    diagnostics = {
        "semantic_null_residual_norm": float(null.detach().norm().cpu().item()),
        "semantic_teacher_velocity_norm": float(sqrt_velocity.detach().norm().cpu().item()),
        "semantic_teacher_unclipped_velocity_norm": float(unclipped_norm.cpu().item()),
        "semantic_teacher_velocity_scale": float(velocity_scale),
        "semantic_kernel_rank": int(torch.linalg.matrix_rank(gram_tangent.detach()).cpu().item()),
    }
    return SemanticTeacherResult(
        mass_velocity=mass_velocity,
        log_rate=log_rate,
        sqrt_velocity=sqrt_velocity,
        diagnostics=diagnostics,
    )


def semantic_teacher_loss_for_trace(
    trace: DecisionTrace,
    target_probs: torch.Tensor,
    *,
    beta: float = 1.0,
    pinv_rtol: float = 1e-2,
    velocity_clip: float | None = None,
    teacher_path_geometry: str = "semantic",
    semantic_calibration_gamma: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    if trace.predicted_sqrt_velocity is None:
        zero = torch.zeros((), dtype=trace.current_probs.dtype, device=trace.current_probs.device)
        return zero, {
            "semantic_teacher_loss": 0.0,
            "semantic_null_residual_norm": 0.0,
            "semantic_teacher_skipped": 1.0,
            "teacher_path_geometry": _normalize_geometry(teacher_path_geometry),
            "probability_path_geometry": _normalize_geometry(teacher_path_geometry),
        }
    initial = trace.initial_probs if trace.initial_probs is not None else trace.current_probs.detach()
    teacher = teacher_path_state(
        initial.detach(),
        trace.candidate_semantics.detach() if trace.candidate_semantics is not None else None,
        target_probs.detach(),
        flow_time=float(trace.flow_time),
        geometry=str(teacher_path_geometry),
        beta=float(beta),
        pinv_rtol=float(pinv_rtol),
        velocity_clip=velocity_clip,
    )
    if trace.velocity_fn is not None:
        pred = trace.velocity_fn(
            teacher.current_probs.to(trace.predicted_sqrt_velocity.device, trace.predicted_sqrt_velocity.dtype),
            float(trace.flow_time),
        )
        recomputed = 1.0
    else:
        pred = trace.predicted_sqrt_velocity
        recomputed = 0.0
    loss, loss_diag = semantic_calibrated_velocity_loss(
        pred,
        teacher.sqrt_velocity.to(pred.device, pred.dtype),
        teacher.current_probs.to(pred.device, pred.dtype),
        trace.candidate_semantics.to(pred.device, pred.dtype) if trace.candidate_semantics is not None else None,
        gamma=float(semantic_calibration_gamma),
    )
    diag = dict(teacher.diagnostics)
    diag.update({
        "semantic_teacher_loss": float(loss.detach().cpu().item()),
        "semantic_teacher_skipped": 0.0,
        "semantic_teacher_recomputed_velocity": float(recomputed),
        **loss_diag,
    })
    return loss, diag


def semantic_calibrated_velocity_loss(
    pred_sqrt_velocity: torch.Tensor,
    target_sqrt_velocity: torch.Tensor,
    current_probs: torch.Tensor,
    candidate_semantics: torch.Tensor | None,
    *,
    gamma: float,
) -> tuple[torch.Tensor, dict]:
    """Return ``e^T(I + gamma Pi_s K Pi_s)e / d``.

    The target velocity is still the Fisher/Euclidean path velocity. Semantics
    only calibrate the error scale. ``gamma=0`` is exactly the uncalibrated
    square-root velocity MSE.
    """

    pred = torch.as_tensor(pred_sqrt_velocity).flatten()
    target = torch.as_tensor(target_sqrt_velocity, device=pred.device, dtype=pred.dtype).flatten()
    if pred.numel() != target.numel():
        raise ValueError("predicted and target sqrt velocities must have the same length")
    d = int(pred.numel())
    err = pred - target
    base = err.pow(2).mean()
    gamma_value = float(gamma)
    diagnostics = {
        "semantic_calibration_gamma": float(gamma_value),
        "semantic_calibration_energy": 0.0,
        "semantic_calibration_rank": 0,
    }
    if d == 0 or gamma_value <= 0.0 or candidate_semantics is None:
        diagnostics["semantic_calibration_loss"] = float(base.detach().cpu().item())
        return base, diagnostics
    probs = _simplex_vector(
        torch.as_tensor(current_probs, device=pred.device, dtype=pred.dtype).flatten(),
        min_prob=1e-8,
        device=pred.device,
        dtype=pred.dtype,
    )
    if probs.numel() != d:
        diagnostics["semantic_calibration_loss"] = float(base.detach().cpu().item())
        return base, diagnostics
    sem = _semantic_matrix(candidate_semantics, probs)
    if int(sem.shape[1]) != d:
        diagnostics["semantic_calibration_loss"] = float(base.detach().cpu().item())
        return base, diagnostics
    gram = sem.transpose(0, 1) @ sem / max(int(sem.shape[0]), 1)
    s = probs.clamp_min(1e-8).sqrt()
    s = s / s.norm().clamp_min(1e-8)
    eye = torch.eye(d, dtype=pred.dtype, device=pred.device)
    projector = eye - torch.outer(s, s)
    calibrated = projector @ gram.to(pred.device, pred.dtype) @ projector
    sem_energy = err @ (calibrated @ err)
    loss = base + gamma_value * sem_energy / float(max(d, 1))
    diagnostics.update({
        "semantic_calibration_energy": float(sem_energy.detach().cpu().item()),
        "semantic_calibration_rank": int(torch.linalg.matrix_rank(calibrated.detach()).cpu().item()),
        "semantic_calibration_loss": float(loss.detach().cpu().item()),
    })
    return loss, diagnostics


def one_hot_smoothed_target(
    current_probs: torch.Tensor,
    choice: int,
    *,
    smoothing: float,
) -> torch.Tensor:
    p = torch.as_tensor(current_probs).float().flatten()
    target = torch.zeros_like(p)
    if p.numel():
        target[max(0, min(int(choice), int(p.numel()) - 1))] = 1.0
    support = p > 0.0
    if bool(support.any()):
        uniform = support.to(dtype=p.dtype)
        uniform = uniform / uniform.sum().clamp_min(1e-8)
    else:
        uniform = torch.full_like(p, 1.0 / max(int(p.numel()), 1))
    return (1.0 - float(smoothing)) * target + float(smoothing) * uniform


def decision_trace_rank(trace: DecisionTrace) -> int:
    probs = torch.as_tensor(trace.current_probs).detach().flatten()
    if probs.numel() == 0:
        return 0
    choice = max(0, min(int(trace.choice), int(probs.numel()) - 1))
    chosen = probs[choice]
    return int((probs > chosen).sum().item()) + 1


def semantic_teacher_loss_for_samples(
    samples,
    sample_weights: torch.Tensor,
    *,
    teacher_beta: float,
    teacher_smoothing: float,
    teacher_pinv_rtol: float = 1e-2,
    teacher_velocity_clip: float | None = None,
    teacher_path_geometry: str = "semantic",
    probability_path_geometry: str | None = None,
    semantic_calibration_gamma: float = 0.0,
    target_mode: str = "posterior",
) -> tuple[torch.Tensor, dict]:
    target_mode_key = _normalize_target_mode(target_mode)
    if target_mode_key == "structural_denoising":
        target_by_trace, target_diag = structural_denoising_targets_for_samples(
            samples,
            smoothing=float(teacher_smoothing),
        )
        effective_sample_weights = torch.ones(
            len(samples),
            dtype=sample_weights.dtype if sample_weights.numel() else torch.float32,
            device=sample_weights.device if sample_weights.numel() else torch.device("cpu"),
        )
    else:
        target_by_trace, target_diag = local_posterior_targets_for_samples(
            samples,
            sample_weights,
            smoothing=float(teacher_smoothing),
        )
        target_diag["semantic_teacher_target_mode"] = "posterior"
        effective_sample_weights = sample_weights
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    nulls: list[float] = []
    times: list[float] = []
    velocity_norms: list[float] = []
    unclipped_velocity_norms: list[float] = []
    velocity_scales: list[float] = []
    path_state_l1s: list[float] = []
    path_endpoint_l1s: list[float] = []
    path_entropies: list[float] = []
    recomputed_flags: list[float] = []
    calibration_losses: list[float] = []
    calibration_energies: list[float] = []
    path_geometry = _normalize_geometry(probability_path_geometry or teacher_path_geometry)
    skipped = 0
    total = 0
    device = effective_sample_weights.device if effective_sample_weights.numel() else torch.device("cpu")
    dtype = effective_sample_weights.dtype if effective_sample_weights.numel() else torch.float32
    for sample in samples:
        for trace in getattr(sample, "decision_traces", ()):
            probs = getattr(trace, "current_probs", None)
            if isinstance(probs, torch.Tensor):
                device = probs.device
                dtype = probs.dtype
                break
        else:
            continue
        break
    for idx, sample in enumerate(samples):
        traces = list(getattr(sample, "decision_traces", ()))
        active_trace_indices = [trace_idx for trace_idx, trace in enumerate(traces) if bool(trace.active)]
        if not active_trace_indices:
            continue
        sample_loss_terms: list[torch.Tensor] = []
        sample_loss_weights: list[torch.Tensor] = []
        weight = _sample_weight(effective_sample_weights, idx, dtype=dtype, device=device)
        for trace_idx in active_trace_indices:
            trace = traces[int(trace_idx)]
            total += 1
            target = target_by_trace.get((int(idx), int(trace_idx)))
            if target is None:
                target = one_hot_smoothed_target(trace.current_probs, trace.choice, smoothing=float(teacher_smoothing))
            loss, diag = semantic_teacher_loss_for_trace(
                trace,
                target,
                beta=float(teacher_beta),
                pinv_rtol=float(teacher_pinv_rtol),
                velocity_clip=teacher_velocity_clip,
                teacher_path_geometry=path_geometry,
                semantic_calibration_gamma=float(semantic_calibration_gamma),
            )
            if float(diag.get("semantic_teacher_skipped", 0.0)) > 0.0:
                skipped += 1
                continue
            losses.append(loss)
            weights.append(weight.to(loss.device, loss.dtype))
            sample_loss_terms.append(loss.detach())
            sample_loss_weights.append(weight.to(loss.device, loss.dtype))
            nulls.append(float(diag.get("semantic_null_residual_norm", 0.0)))
            times.append(float(trace.flow_time))
            velocity_norms.append(float(diag.get("semantic_teacher_velocity_norm", 0.0)))
            unclipped_velocity_norms.append(float(diag.get("semantic_teacher_unclipped_velocity_norm", 0.0)))
            velocity_scales.append(float(diag.get("semantic_teacher_velocity_scale", 1.0)))
            path_state_l1s.append(float(diag.get("teacher_path_state_l1_from_initial", 0.0)))
            path_endpoint_l1s.append(float(diag.get("teacher_path_endpoint_l1", 0.0)))
            path_entropies.append(float(diag.get("teacher_path_current_entropy", 0.0)))
            recomputed_flags.append(float(diag.get("semantic_teacher_recomputed_velocity", 0.0)))
            calibration_losses.append(float(diag.get("semantic_calibration_loss", 0.0)))
            calibration_energies.append(float(diag.get("semantic_calibration_energy", 0.0)))
        if sample_loss_terms and hasattr(sample, "semantic_teacher_loss_tensor"):
            stacked = torch.stack(sample_loss_terms)
            w = torch.stack(sample_loss_weights).to(stacked.device, stacked.dtype)
            w = w / w.sum().clamp_min(1e-8)
            sample.semantic_teacher_loss_tensor = (stacked * w).sum()
    if not losses:
        zero = torch.zeros((), dtype=dtype, device=device, requires_grad=True)
        return zero, {
            "semantic_teacher_loss_mean": 0.0,
            "semantic_null_residual_norm_mean": 0.0,
            "semantic_teacher_trace_count": int(total),
            "semantic_teacher_skipped_count": int(skipped),
            "semantic_teacher_velocity_norm_mean": 0.0,
            "semantic_teacher_unclipped_velocity_norm_mean": 0.0,
            "semantic_teacher_velocity_scale_mean": 0.0,
            "teacher_path_geometry": path_geometry,
            "probability_path_geometry": path_geometry,
            "teacher_path_state_l1_from_initial_mean": 0.0,
            "teacher_path_endpoint_l1_mean": 0.0,
            "teacher_path_current_entropy_mean": 0.0,
            "semantic_teacher_recomputed_velocity_rate": 0.0,
            "semantic_calibration_gamma": float(semantic_calibration_gamma),
            "semantic_calibration_loss_mean": 0.0,
            "semantic_calibration_energy_mean": 0.0,
            **target_diag,
        }
    loss_values = torch.stack(losses)
    w = torch.stack(weights)
    w = w / w.sum().clamp_min(1e-8)
    loss = (loss_values * w).sum()
    return loss, {
        "semantic_teacher_loss_mean": float(loss_values.detach().mean().cpu().item()),
        "semantic_null_residual_norm_mean": float(sum(nulls) / max(len(nulls), 1)),
        "semantic_teacher_trace_count": int(total),
        "semantic_teacher_skipped_count": int(skipped),
        "semantic_teacher_time_mean": float(sum(times) / max(len(times), 1)),
        "semantic_teacher_velocity_norm_mean": float(sum(velocity_norms) / max(len(velocity_norms), 1)),
        "semantic_teacher_unclipped_velocity_norm_mean": float(
            sum(unclipped_velocity_norms) / max(len(unclipped_velocity_norms), 1)
        ),
        "semantic_teacher_velocity_scale_mean": float(sum(velocity_scales) / max(len(velocity_scales), 1)),
        "teacher_path_geometry": path_geometry,
        "probability_path_geometry": path_geometry,
        "teacher_path_state_l1_from_initial_mean": float(sum(path_state_l1s) / max(len(path_state_l1s), 1)),
        "teacher_path_endpoint_l1_mean": float(sum(path_endpoint_l1s) / max(len(path_endpoint_l1s), 1)),
        "teacher_path_current_entropy_mean": float(sum(path_entropies) / max(len(path_entropies), 1)),
        "semantic_teacher_recomputed_velocity_rate": float(sum(recomputed_flags) / max(len(recomputed_flags), 1)),
        "semantic_calibration_gamma": float(semantic_calibration_gamma),
        "semantic_calibration_loss_mean": float(sum(calibration_losses) / max(len(calibration_losses), 1)),
        "semantic_calibration_energy_mean": float(sum(calibration_energies) / max(len(calibration_energies), 1)),
        **target_diag,
    }


def _normalize_geometry(geometry: str) -> str:
    key = str(geometry or "fisher").strip().lower().replace("-", "_")
    aliases = {
        "fisher": "fisher",
        "fisher_rao": "fisher",
        "fr": "fisher",
        "sphere": "fisher",
        "spherical": "fisher",
        "semantic": "fisher",
        "semantic_pullback": "fisher",
        "semantic_quotient": "fisher",
        "semantic_geometry": "fisher",
        "euclidean": "euclidean",
        "euler": "euclidean",
        "linear": "euclidean",
        "parameter_euclidean": "euclidean",
    }
    if key not in aliases:
        raise ValueError(f"unknown teacher_path_geometry: {geometry}")
    return aliases[key]


def _normalize_target_mode(value: str) -> str:
    key = str(value or "posterior").strip().lower().replace("-", "_")
    if key in {"posterior", "endpoint_posterior", "reward_posterior", "local_posterior"}:
        return "posterior"
    if key in {
        "structural_denoising",
        "structure_denoising",
        "clean_gt",
        "clean_gt_one_hot",
        "gt_denoising",
        "denoising",
    }:
        return "structural_denoising"
    raise ValueError(f"unknown semantic teacher target mode: {value}")


def _fisher_sphere_state(
    p0: torch.Tensor,
    q: torch.Tensor,
    *,
    t: float,
    beta: float,
    min_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    s0 = p0.clamp_min(float(min_prob)).sqrt()
    s0 = s0 / s0.norm().clamp_min(float(min_prob))
    s1 = q.clamp_min(float(min_prob)).sqrt()
    s1 = s1 / s1.norm().clamp_min(float(min_prob))
    dot = torch.dot(s0, s1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    if float(sin_theta.detach().cpu().item()) < 1e-6:
        direction = s1 - s0
        current = s0 + float(t) * direction
        current = current / current.norm().clamp_min(float(min_prob))
        tangent = direction - current * torch.dot(current, direction)
        return current, float(beta) * tangent, float(theta.detach().cpu().item())
    a = torch.sin((1.0 - float(t)) * theta) / sin_theta
    b = torch.sin(float(t) * theta) / sin_theta
    current = a * s0 + b * s1
    current = current / current.norm().clamp_min(float(min_prob))
    velocity = (
        -theta * torch.cos((1.0 - float(t)) * theta) / sin_theta * s0
        + theta * torch.cos(float(t) * theta) / sin_theta * s1
    )
    velocity = velocity - current * torch.dot(current, velocity)
    return current, float(beta) * velocity, float(theta.detach().cpu().item())


def _simplex_vector(
    values: torch.Tensor,
    *,
    min_prob: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    out = torch.as_tensor(values, device=device, dtype=dtype).float().flatten()
    if dtype is not None:
        out = out.to(dtype=dtype)
    if device is not None:
        out = out.to(device=device)
    if out.numel() == 0:
        raise ValueError("simplex vector must be non-empty")
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(float(min_prob))
    return out / out.sum().clamp_min(float(min_prob))


def _semantic_matrix(candidate_semantics: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    sem = torch.as_tensor(candidate_semantics, dtype=probs.dtype, device=probs.device)
    if sem.ndim == 1:
        sem = sem.unsqueeze(1)
    if int(sem.shape[1]) != int(probs.numel()):
        raise ValueError("candidate_semantics must have one column per candidate")
    return _normalize_columns(torch.nan_to_num(sem.float()).to(device=probs.device, dtype=probs.dtype))


def _normalize_columns(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    scale = centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return centered / scale


def local_posterior_targets_for_samples(
    samples,
    sample_weights: torch.Tensor,
    *,
    smoothing: float,
) -> tuple[dict[tuple[int, int], torch.Tensor], dict]:
    """Project endpoint posterior mass back to local decision marginals.

    For each exact local decision context, this computes
    q*(a | context) from posterior-weighted complete expressions. The context
    is keyed by the decision id and candidate keys when available, so structure
    evidence affects the target only through endpoint posterior weights.
    """

    device = sample_weights.device if sample_weights.numel() else torch.device("cpu")
    dtype = sample_weights.dtype if sample_weights.numel() else torch.float32
    grouped: dict[tuple, list[tuple[int, int, DecisionTrace, torch.Tensor]]] = {}
    for sample_idx, sample in enumerate(samples):
        weight = _sample_weight(sample_weights, int(sample_idx), dtype=dtype, device=device)
        for trace_idx, trace in enumerate(getattr(sample, "decision_traces", ())):
            if not bool(trace.active):
                continue
            p = torch.as_tensor(trace.current_probs).flatten()
            if p.numel() == 0:
                continue
            key = _trace_context_key(trace)
            grouped.setdefault(key, []).append((int(sample_idx), int(trace_idx), trace, weight))

    out: dict[tuple[int, int], torch.Tensor] = {}
    entropies: list[float] = []
    masses: list[float] = []
    esses: list[float] = []
    for items in grouped.values():
        ref = items[0][2]
        p = torch.as_tensor(ref.current_probs).float().flatten()
        d = int(p.numel())
        target_mass = torch.zeros(d, dtype=dtype, device=device)
        total_mass = torch.zeros((), dtype=dtype, device=device)
        for _, _, trace, weight in items:
            choice = max(0, min(int(trace.choice), d - 1))
            w = weight.to(device=device, dtype=dtype)
            target_mass[choice] = target_mass[choice] + w
            total_mass = total_mass + w
        if float(total_mass.detach().cpu().item()) <= 0.0:
            ref_support = torch.as_tensor(ref.current_probs, dtype=dtype, device=device).flatten() > 0.0
            if bool(ref_support.any()):
                posterior = ref_support.to(dtype=dtype)
                posterior = posterior / posterior.sum().clamp_min(1e-8)
            else:
                posterior = torch.full((d,), 1.0 / max(d, 1), dtype=dtype, device=device)
        else:
            posterior = target_mass / total_mass.clamp_min(1e-8)
        ess = float(1.0 / posterior.detach().cpu().clamp_min(1e-12).pow(2).sum().item()) if d else 0.0
        for sample_idx, trace_idx, trace, _ in items:
            p_trace = torch.as_tensor(trace.current_probs, dtype=dtype, device=device).flatten()
            support = p_trace > 0.0
            p_trace = torch.where(support, p_trace.clamp_min(1e-8), torch.zeros_like(p_trace))
            if bool(p_trace.sum().detach().cpu().item() > 0.0):
                p_trace = p_trace / p_trace.sum().clamp_min(1e-8)
            else:
                p_trace = torch.full_like(p_trace, 1.0 / max(d, 1))
            target = (1.0 - float(smoothing)) * posterior.to(p_trace.device, p_trace.dtype) + float(smoothing) * p_trace
            target = torch.where(support, target, torch.zeros_like(target))
            target = target / target.sum().clamp_min(1e-8)
            out[(int(sample_idx), int(trace_idx))] = target
        entropy = -(posterior.clamp_min(1e-12) * posterior.clamp_min(1e-12).log()).sum()
        entropies.append(float(entropy.detach().cpu().item()))
        masses.append(float(total_mass.detach().cpu().item()))
        esses.append(ess)
    diag = {
        "semantic_teacher_local_group_count": int(len(grouped)),
        "semantic_teacher_target_entropy_mean": float(sum(entropies) / max(len(entropies), 1)),
        "semantic_teacher_local_posterior_mass_mean": float(sum(masses) / max(len(masses), 1)),
        "semantic_teacher_local_ess_mean": float(sum(esses) / max(len(esses), 1)),
    }
    return out, diag


def structural_denoising_targets_for_samples(
    samples,
    *,
    smoothing: float,
) -> tuple[dict[tuple[int, int], torch.Tensor], dict]:
    """Build clean-GT local targets for noisy GT-neighborhood traces.

    The canonical GT sample supplies the clean action for each local decision
    group. Every active trace with the same structural decision key receives a
    smoothed one-hot target at that clean action, independent of reward or
    endpoint posterior weights.
    """

    clean_traces: dict[tuple, DecisionTrace] = {}
    clean_count = 0
    for sample in samples:
        diag = getattr(sample, "diagnostics", None) or {}
        if not (bool(diag.get("gt_neighborhood_canonical", False)) or bool(diag.get("is_gt_elite", False))):
            continue
        for trace in getattr(sample, "decision_traces", ()):
            if not bool(trace.active):
                continue
            key = _trace_structural_denoising_key(trace)
            clean_traces.setdefault(key, trace)
            clean_count += 1
    if not clean_traces:
        for sample in samples[:1]:
            for trace in getattr(sample, "decision_traces", ()):
                if not bool(trace.active):
                    continue
                clean_traces.setdefault(_trace_structural_denoising_key(trace), trace)
                clean_count += 1

    out: dict[tuple[int, int], torch.Tensor] = {}
    entropies: list[float] = []
    matched = 0
    total = 0
    for sample_idx, sample in enumerate(samples):
        for trace_idx, trace in enumerate(getattr(sample, "decision_traces", ())):
            if not bool(trace.active):
                continue
            p = torch.as_tensor(trace.current_probs).flatten()
            if p.numel() == 0:
                continue
            total += 1
            clean = clean_traces.get(_trace_structural_denoising_key(trace))
            choice = int(clean.choice) if clean is not None else int(trace.choice)
            if clean is not None:
                matched += 1
            target = one_hot_smoothed_target(trace.current_probs, choice, smoothing=float(smoothing))
            target = target.to(device=trace.current_probs.device, dtype=trace.current_probs.dtype)
            target = target / target.sum().clamp_min(1e-8)
            out[(int(sample_idx), int(trace_idx))] = target
            entropy = -(target.detach().clamp_min(1e-12) * target.detach().clamp_min(1e-12).log()).sum()
            entropies.append(float(entropy.cpu().item()))

    return out, {
        "semantic_teacher_target_mode": "structural_denoising",
        "semantic_teacher_local_group_count": int(len(clean_traces)),
        "semantic_teacher_target_entropy_mean": float(sum(entropies) / max(len(entropies), 1)),
        "semantic_teacher_local_posterior_mass_mean": 0.0,
        "semantic_teacher_local_ess_mean": 0.0,
        "semantic_teacher_clean_trace_count": int(clean_count),
        "semantic_teacher_clean_trace_match_rate": float(matched / max(total, 1)),
    }


def _trace_context_key(trace: DecisionTrace) -> tuple:
    keys = tuple(str(item) for item in getattr(trace, "candidate_keys", ()) or ())
    context = str(getattr(trace, "context_key", "") or "")
    if keys:
        return (str(trace.group_id), context, keys)
    d = int(torch.as_tensor(trace.current_probs).numel())
    return (str(trace.group_id), context, d)


def _trace_structural_denoising_key(trace: DecisionTrace) -> tuple:
    keys = tuple(str(item) for item in getattr(trace, "candidate_keys", ()) or ())
    if keys:
        return (str(trace.group_id), keys)
    d = int(torch.as_tensor(trace.current_probs).numel())
    return (str(trace.group_id), d)


def _sample_weight(
    sample_weights: torch.Tensor,
    idx: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if idx < int(sample_weights.numel()):
        return sample_weights[int(idx)].to(device=device, dtype=dtype)
    return torch.tensor(1.0, dtype=dtype, device=device)
