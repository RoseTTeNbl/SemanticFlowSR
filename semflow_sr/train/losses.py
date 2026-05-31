"""Losses: strict velocity matching (+ optional semantic-metric-weighted term)."""
from __future__ import annotations
import torch

from ..geometry.velocities import semantic_metric_norm_sq


def velocity_mse(v_pred, dp_dlambda, mask=None):
    diff = (v_pred - dp_dlambda)
    if mask is not None:
        diff = diff * mask
        denom = mask.sum().clamp(min=1)
        return (diff * diff).sum() / denom
    return (diff * diff).mean()


def metric_weighted_velocity_loss(v_pred, dp_dlambda, p_lambda, weights, mask=None,
                                  metric_weight: float = 0.0):
    loss = velocity_mse(v_pred, dp_dlambda, mask)
    if metric_weight > 0.0:
        gnorm = semantic_metric_norm_sq(v_pred - dp_dlambda, p_lambda.clamp(min=1e-8), weights)
        loss = loss + metric_weight * gnorm.mean()
    return loss
