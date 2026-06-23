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
    head_fit_mode: str = "linear"


@dataclass
class RewardBatch:
    rewards: torch.Tensor
    r2: torch.Tensor
    nmse: torch.Tensor
    valid_mask: torch.Tensor
    affine_coef: torch.Tensor
    complexity: torch.Tensor
    selected_term_index: torch.Tensor
    best_raw_term_r2: torch.Tensor
    fitted_head_gain: torch.Tensor
    head_coef_nonzero_count: torch.Tensor
    head_coef_norm: torch.Tensor


def evaluate_expression_rewards(
    samples: list[CircuitSample],
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: RewardConfig | None = None,
) -> RewardBatch:
    cfg = cfg or RewardConfig()
    device = y.device
    rewards = []
    r2_values = []
    nmse_values = []
    valid = []
    coefs = []
    complexity = []
    selected_indices = []
    best_raw_term_r2_values = []
    fitted_head_gain_values = []
    head_coef_nonzero_values = []
    head_coef_norm_values = []
    max_coef_len = 0
    for sample in samples:
        try:
            terms = tuple(sample.head_terms) if sample.head_terms else (sample.expression,)
            columns = [
                torch.nan_to_num(eval_expr(term, x), nan=0.0, posinf=0.0, neginf=0.0)
                for term in terms
            ]
            semantics = torch.stack(columns, dim=1) if columns else torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
            finite = torch.isfinite(semantics).all() and semantics.abs().max() < 1e6
        except Exception:
            semantics = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
            finite = torch.tensor(False)
        coef, pred, selected_idx = _fit_sample_prediction(
            semantics,
            y,
            mode=str(cfg.head_fit_mode),
        )
        best_raw_r2 = _best_single_term_r2(semantics, y)
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
        coef = coef.detach().float().cpu()
        max_coef_len = max(max_coef_len, int(coef.numel()))
        coefs.append(coef)
        complexity.append(float(sample.complexity))
        selected_indices.append(int(selected_idx))
        coef_for_diag = coef.detach().float().to(device)
        structural_coef = coef_for_diag[:-1] if int(coef_for_diag.numel()) > 1 else coef_for_diag
        best_raw_term_r2_values.append(torch.as_tensor(best_raw_r2, dtype=torch.float32))
        fitted_head_gain_values.append(torch.as_tensor(r2 - best_raw_r2, dtype=torch.float32))
        head_coef_nonzero_values.append(torch.as_tensor((structural_coef.abs() > 1e-8).sum(), dtype=torch.long))
        head_coef_norm_values.append(torch.as_tensor(structural_coef.norm(), dtype=torch.float32))
    if coefs:
        padded = []
        for coef in coefs:
            if int(coef.numel()) < max_coef_len:
                coef = torch.cat([coef, torch.zeros(max_coef_len - int(coef.numel()))])
            padded.append(coef)
        coef_tensor = torch.stack(padded)
    else:
        coef_tensor = torch.zeros(0, 2)
    return RewardBatch(
        rewards=torch.stack(rewards) if rewards else torch.zeros(0, dtype=torch.float32, device=device),
        r2=torch.stack(r2_values) if r2_values else torch.zeros(0, dtype=torch.float32, device=device),
        nmse=torch.stack(nmse_values) if nmse_values else torch.zeros(0, dtype=torch.float32, device=device),
        valid_mask=torch.tensor(valid, dtype=torch.bool, device=device),
        affine_coef=coef_tensor,
        complexity=torch.tensor(complexity, dtype=torch.float32, device=device),
        selected_term_index=torch.tensor(selected_indices, dtype=torch.long, device=device),
        best_raw_term_r2=torch.stack(best_raw_term_r2_values).to(device) if best_raw_term_r2_values else torch.zeros(0, dtype=torch.float32, device=device),
        fitted_head_gain=torch.stack(fitted_head_gain_values).to(device) if fitted_head_gain_values else torch.zeros(0, dtype=torch.float32, device=device),
        head_coef_nonzero_count=torch.stack(head_coef_nonzero_values).to(device) if head_coef_nonzero_values else torch.zeros(0, dtype=torch.long, device=device),
        head_coef_norm=torch.stack(head_coef_norm_values).to(device) if head_coef_norm_values else torch.zeros(0, dtype=torch.float32, device=device),
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


def _linear_fit(semantics: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    S = torch.nan_to_num(semantics.float())
    if S.ndim == 1:
        S = S.unsqueeze(1)
    y = torch.nan_to_num(y.float())
    ones = torch.ones((S.shape[0], 1), dtype=S.dtype, device=S.device)
    A = torch.cat([S, ones], dim=1)
    eye = torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
    G = A.transpose(0, 1) @ A + 1e-6 * eye
    rhs = A.transpose(0, 1) @ y
    try:
        coef = torch.linalg.solve(G, rhs)
    except RuntimeError:
        coef = torch.linalg.pinv(G) @ rhs
    pred = A @ coef
    return coef, pred


def _fit_sample_prediction(
    semantics: torch.Tensor,
    y: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    mode = str(mode).lower()
    if mode in {"linear", "sparse_linear", "dense", "fit"}:
        coef, pred = _linear_fit(semantics, y)
        return coef, pred, -1
    if mode not in {"selector", "select", "single", "single_term"}:
        raise ValueError(f"unknown head_fit_mode: {mode}")
    S = torch.nan_to_num(semantics.float())
    if S.ndim == 1:
        S = S.unsqueeze(1)
    best_idx = 0
    best_r2: torch.Tensor | None = None
    best_coef: torch.Tensor | None = None
    best_pred: torch.Tensor | None = None
    for idx in range(int(S.shape[1])):
        term_coef, term_pred = _linear_fit(S[:, idx:idx + 1], y)
        term_r2 = _r2(y, term_pred)
        if best_r2 is None or bool(term_r2 > best_r2):
            best_idx = idx
            best_r2 = term_r2
            best_coef = term_coef
            best_pred = term_pred
    if best_coef is None or best_pred is None:
        coef, pred = _linear_fit(S, y)
        return coef, pred, -1
    full = torch.zeros(int(S.shape[1]) + 1, dtype=best_coef.dtype, device=best_coef.device)
    full[best_idx] = best_coef[0]
    full[-1] = best_coef[1]
    return full, best_pred, best_idx


def _best_single_term_r2(semantics: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    S = torch.nan_to_num(semantics.float())
    if S.ndim == 1:
        S = S.unsqueeze(1)
    best: torch.Tensor | None = None
    for idx in range(int(S.shape[1])):
        _, term_pred = _linear_fit(S[:, idx:idx + 1], y)
        term_r2 = _r2(y, term_pred)
        if best is None or bool(term_r2 > best):
            best = term_r2
    if best is None:
        return torch.tensor(0.0, dtype=y.dtype, device=y.device)
    return best


def _r2(y: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum().clamp_min(1e-12)
    return 1.0 - ss_res / ss_tot
