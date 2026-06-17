"""KL-barycenter fusion of model and GP implicit policies."""
from __future__ import annotations

import torch


def kl_barycenter(pi_model: torch.Tensor, pi_gp: torch.Tensor, alpha: float,
                  eps: float = 1e-12) -> torch.Tensor:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    log_rho = (1.0 - float(alpha)) * pi_model.clamp_min(eps).log()
    log_rho = log_rho + float(alpha) * pi_gp.to(device=pi_model.device, dtype=pi_model.dtype).clamp_min(eps).log()
    return torch.softmax(log_rho, dim=-1)
