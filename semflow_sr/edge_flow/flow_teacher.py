"""Teacher paths on edge probability product-of-simplexes."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .edge_distribution import EdgeDistribution


@dataclass
class EdgeFlowTeacherRecord:
    theta_lambda: EdgeDistribution
    z_lambda_mixture: torch.Tensor
    zdot_mixture: torch.Tensor
    z_lambda_groups: dict[str, torch.Tensor]
    zdot_groups: dict[str, torch.Tensor]
    diagnostics: dict


def build_fisher_slerp_record(
    theta0: EdgeDistribution,
    theta_star: EdgeDistribution,
    *,
    lam: float,
) -> EdgeFlowTeacherRecord:
    lam = min(max(float(lam), 0.0), 1.0)
    z_mix, dz_mix = _slerp_sqrt(theta0.mixture_probs, theta_star.mixture_probs, lam)
    group_z: dict[str, torch.Tensor] = {}
    group_dz: dict[str, torch.Tensor] = {}
    group_probs: dict[str, torch.Tensor] = {}
    for group in theta0.template.groups:
        z, dz = _slerp_sqrt(theta0.group_probs[group.group_id], theta_star.group_probs[group.group_id], lam)
        group_z[group.group_id] = z
        group_dz[group.group_id] = dz
        group_probs[group.group_id] = (z * z) / (z * z).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    mix_probs = (z_mix * z_mix) / (z_mix * z_mix).sum().clamp_min(1e-12)
    theta_lambda = EdgeDistribution(theta0.template, mix_probs, group_probs)
    diagnostics = {
        "lambda": lam,
        "velocity_norm": float(dz_mix.square().sum().sqrt().item()),
        "simplex_mass_error": float(abs(float(mix_probs.sum().item()) - 1.0)),
    }
    return EdgeFlowTeacherRecord(theta_lambda, z_mix, dz_mix, group_z, group_dz, diagnostics)


def _slerp_sqrt(p0: torch.Tensor, p1: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    z0 = p0.clamp_min(1e-12).sqrt()
    z1 = p1.clamp_min(1e-12).sqrt()
    z0 = z0 / z0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    z1 = z1 / z1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dot = (z0 * z1).sum(dim=-1, keepdim=True).clamp(-0.999999, 0.999999)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    near = sin_omega.abs() < 1e-6
    a = torch.sin((1.0 - lam) * omega) / sin_omega.clamp_min(1e-12)
    b = torch.sin(lam * omega) / sin_omega.clamp_min(1e-12)
    da = -omega * torch.cos((1.0 - lam) * omega) / sin_omega.clamp_min(1e-12)
    db = omega * torch.cos(lam * omega) / sin_omega.clamp_min(1e-12)
    z = a * z0 + b * z1
    dz = da * z0 + db * z1
    linear_z = (1.0 - lam) * z0 + lam * z1
    linear_z = linear_z / linear_z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    linear_dz = z1 - z0
    linear_dz = linear_dz - (linear_z * linear_dz).sum(dim=-1, keepdim=True) * linear_z
    z = torch.where(near, linear_z, z)
    dz = torch.where(near, linear_dz, dz)
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dz = dz - (z * dz).sum(dim=-1, keepdim=True) * z
    return z, dz
