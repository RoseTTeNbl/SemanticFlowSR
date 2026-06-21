"""Reward evaluation for complete sampled expressions."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..sr.ast import eval_expr
from .circuit_sampler import CircuitSample


@dataclass(frozen=True)
class RewardConfig:
    complexity_weight: float = 0.001
    invalid_reward: float = -1.0e6


@dataclass
class RewardBatch:
    rewards: torch.Tensor
    r2: torch.Tensor
    nmse: torch.Tensor
    valid_mask: torch.Tensor
    affine_coef: torch.Tensor
    complexity: torch.Tensor


def evaluate_expression_rewards(
    samples: list[CircuitSample],
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: RewardConfig | None = None,
) -> RewardBatch:
    cfg = cfg or RewardConfig()
    rewards = []
    r2_values = []
    nmse_values = []
    valid = []
    coefs = []
    complexity = []
    for sample in samples:
        try:
            semantics = torch.nan_to_num(eval_expr(sample.expression, x), nan=0.0, posinf=0.0, neginf=0.0)
            finite = torch.isfinite(semantics).all() and semantics.abs().max() < 1e6
        except Exception:
            semantics = torch.zeros_like(y)
            finite = torch.tensor(False)
        a, b, pred = _affine_fit(semantics, y)
        r2 = _r2(y, pred) if bool(finite) else torch.tensor(0.0, dtype=y.dtype, device=y.device)
        mse = torch.mean((y - pred) ** 2)
        denom = torch.var(y).clamp_min(1e-12)
        nmse = mse / denom
        reward = r2 - float(cfg.complexity_weight) * float(sample.complexity)
        if not bool(finite):
            reward = torch.tensor(float(cfg.invalid_reward), dtype=y.dtype, device=y.device)
        rewards.append(torch.as_tensor(reward, dtype=torch.float32))
        r2_values.append(torch.as_tensor(r2, dtype=torch.float32))
        nmse_values.append(torch.as_tensor(nmse, dtype=torch.float32))
        valid.append(bool(finite))
        coefs.append(torch.tensor([float(a), float(b)], dtype=torch.float32))
        complexity.append(float(sample.complexity))
    return RewardBatch(
        rewards=torch.stack(rewards) if rewards else torch.zeros(0),
        r2=torch.stack(r2_values) if r2_values else torch.zeros(0),
        nmse=torch.stack(nmse_values) if nmse_values else torch.zeros(0),
        valid_mask=torch.tensor(valid, dtype=torch.bool),
        affine_coef=torch.stack(coefs) if coefs else torch.zeros(0, 2),
        complexity=torch.tensor(complexity, dtype=torch.float32),
    )


def _affine_fit(s: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s = torch.nan_to_num(s.float())
    y = torch.nan_to_num(y.float())
    A = torch.stack([s, torch.ones_like(s)], dim=1)
    G = A.transpose(0, 1) @ A + 1e-6 * torch.eye(2, dtype=A.dtype, device=A.device)
    rhs = A.transpose(0, 1) @ y
    try:
        coef = torch.linalg.solve(G, rhs)
    except RuntimeError:
        coef = torch.linalg.pinv(G) @ rhs
    pred = A @ coef
    return coef[0], coef[1], pred


def _r2(y: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum().clamp_min(1e-12)
    return 1.0 - ss_res / ss_tot

