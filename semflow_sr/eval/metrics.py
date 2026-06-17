"""SR metrics: R², NMSE, accuracy-rate-style threshold, complexity.

Aligned with SRBench (R², solution rate, simplicity) and TPSR/NeSymReS
(r2_zero, pointwise accuracy_l1 with relative tolerance).
"""
from __future__ import annotations
import numpy as np


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def r2_zero(y_true, y_pred) -> float:
    """max(0, R²) — TPSR/NeSymReS convention, robust to wildly wrong predictions."""
    return max(0.0, r2_score(y_true, y_pred))


def nmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2) / (np.var(y_true) + 1e-12))


def accuracy_tau(y_true, y_pred, tau: float = 0.05, point_frac: float = 0.95) -> float:
    """Pointwise relative-tolerance hit rate ≥ point_frac (TPSR accuracy_l1)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rel = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-9)
    return float(np.mean(rel < tau) >= point_frac)


def simplicity(num_components: int) -> float:
    """SRBench simplicity: -log_5(#components). Higher (less negative) is simpler."""
    return float(-np.round(np.log(max(num_components, 1)) / np.log(5), 3))


def accuracy_rate(r2: float, threshold: float = 0.999) -> bool:
    return r2 >= threshold


def energy_decrease_ratio(trace: list[float]) -> float:
    """Fraction of initial residual energy removed: (E0-Ef)/E0."""
    if not trace or trace[0] < 1e-12:
        return 0.0
    return float((trace[0] - trace[-1]) / trace[0])
