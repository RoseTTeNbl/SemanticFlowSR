"""Action-score heads for local natural-flow updates.

The main model output is a support-local semantic-Fisher log-rate. A potential score
path is retained for the plain Fisher closed-form ablation.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class VelocityOutput:
    v_pred: torch.Tensor                    # [bsz,A], derived tangent velocity
    potential_logits: torch.Tensor | None = None  # [bsz,A], unnormalized potential
    lograte_logits: torch.Tensor | None = None    # [bsz,A], semantic-Fisher log-rate
    z_dot_pred: torch.Tensor | None = None        # [bsz,A], sphere tangent


class VelocityHead(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, action_tokens, mask=None) -> VelocityOutput:
        raw = self.out(action_tokens).squeeze(-1)              # [bsz,A]
        return potential_to_velocity_output(raw, p_lambda=None, beta=1.0, mask=mask)


def potential_to_velocity_output(raw: torch.Tensor, p_lambda: torch.Tensor | None,
                                 beta: torch.Tensor | float = 1.0,
                                 mask: torch.Tensor | None = None) -> VelocityOutput:
    """Convert raw potential scores into a derived natural-flow velocity."""
    if mask is not None:
        potential = raw.masked_fill(~mask, 0.0)
        n = mask.sum(-1, keepdim=True).clamp(min=1)
        mean_uniform = potential.sum(-1, keepdim=True) / n
        potential = (potential - mean_uniform) * mask
    else:
        potential = raw - raw.mean(-1, keepdim=True)

    if p_lambda is None:
        v = potential
        if mask is not None:
            v = v - (v.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp(min=1))
            v = v * mask
        else:
            v = v - v.mean(-1, keepdim=True)
    else:
        p = p_lambda.to(device=potential.device, dtype=potential.dtype)
        if mask is not None:
            p = p * mask
            p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        beta_t = torch.as_tensor(beta, device=potential.device, dtype=potential.dtype)
        while beta_t.dim() < potential.dim():
            beta_t = beta_t.unsqueeze(-1)
        mean_potential = (p * potential).sum(dim=-1, keepdim=True)
        v = beta_t * p * (potential - mean_potential)
        if mask is not None:
            v = v * mask
        v = v - v.sum(-1, keepdim=True) / potential.shape[-1]
    z_dot = None
    if p_lambda is not None:
        z = p_lambda.to(device=potential.device, dtype=potential.dtype).clamp(min=1e-12).sqrt()
        z_dot = 0.5 * z * (v / p_lambda.clamp(min=1e-12))
        z_dot = z_dot - (z_dot * z).sum(dim=-1, keepdim=True) * z
    return VelocityOutput(v_pred=v, potential_logits=potential, z_dot_pred=z_dot)


def lograte_to_velocity_output(raw: torch.Tensor, p_lambda: torch.Tensor,
                               mask: torch.Tensor | None = None) -> VelocityOutput:
    """Convert raw scores into a mass-preserving log-rate and its induced tangents."""
    p = p_lambda.to(device=raw.device, dtype=raw.dtype)
    if mask is not None:
        raw = raw.masked_fill(~mask, 0.0)
        p = p * mask
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    centered = raw - (p * raw).sum(dim=-1, keepdim=True)
    if mask is not None:
        centered = centered * mask
    v = p * centered
    if mask is not None:
        v = v * mask
    z = p.clamp(min=1e-12).sqrt()
    z_dot = 0.5 * z * centered
    z_dot = z_dot - (z_dot * z).sum(dim=-1, keepdim=True) * z
    if mask is not None:
        z_dot = z_dot * mask
    return VelocityOutput(v_pred=v, lograte_logits=centered, z_dot_pred=z_dot)
