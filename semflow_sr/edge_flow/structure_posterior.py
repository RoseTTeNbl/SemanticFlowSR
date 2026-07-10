"""Structure-conditioned endpoint posterior utilities."""
from __future__ import annotations

import math
import re

import torch

from .benchmark import skeleton_match, token_sequence_metrics


def structure_similarity_score(generated: str, ground_truth: str) -> float:
    """Return a bounded AST/string structure similarity score.

    The score is an endpoint-evidence feature. It is intentionally not used as a
    local geometry term: semantic teacher geometry is handled separately.
    """

    gt = str(ground_truth or "").strip()
    pred = str(generated or "").strip()
    if not gt or not pred:
        return 0.0
    metrics = token_sequence_metrics(gt, pred)
    gt_len = max(_token_count(gt), 1)
    pred_len = max(_token_count(pred), 1)
    scale = max(gt_len, pred_len, 1)
    edit_kernel = math.exp(-float(metrics["formula_edit_distance"]) / float(scale))
    skeleton = 1.0 if skeleton_match(gt, pred) else 0.0
    value = (
        0.25 * float(metrics["formula_bleu"])
        + 0.20 * float(metrics["formula_token_accuracy"])
        + 0.25 * float(edit_kernel)
        + 0.30 * skeleton
    )
    return float(max(0.0, min(1.0, value)))


def structure_conditioned_log_weight(
    *,
    r2: torch.Tensor,
    complexity: torch.Tensor,
    structure_score: torch.Tensor,
    beta_y: float,
    beta_g: float,
    beta_c: float,
) -> torch.Tensor:
    """Unnormalized log evidence for Q*(e | D, g)."""

    r2_t = torch.as_tensor(r2).float()
    complexity_t = torch.as_tensor(complexity, dtype=r2_t.dtype, device=r2_t.device)
    structure_t = torch.as_tensor(structure_score, dtype=r2_t.dtype, device=r2_t.device)
    return (
        float(beta_y) * r2_t
        + float(beta_g) * structure_t
        - float(beta_c) * complexity_t
    )


def normalize_log_weights(log_weights: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(log_weights).float()
    if values.numel() == 0:
        return values
    values = torch.nan_to_num(values, nan=-1.0e9, posinf=1.0e9, neginf=-1.0e9)
    shifted = values - values.max()
    weights = torch.softmax(shifted, dim=0)
    return weights / weights.sum().clamp_min(1e-12)


def _token_count(text: str) -> int:
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+\.\d+|\d+|\*\*|[()+\-*/]", str(text or ""))
    return int(len(tokens) or len(str(text).split()) or 1)
