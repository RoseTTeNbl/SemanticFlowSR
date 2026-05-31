"""SR metrics: R², NMSE, accuracy-rate-style threshold, complexity."""
from __future__ import annotations
import numpy as np


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def nmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2) / (np.var(y_true) + 1e-12))


def accuracy_rate(r2: float, threshold: float = 0.999) -> bool:
    return r2 >= threshold


def energy_decrease_ratio(trace: list[float]) -> float:
    """Fraction of initial residual energy removed: (E0-Ef)/E0."""
    if not trace or trace[0] < 1e-12:
        return 0.0
    return float((trace[0] - trace[-1]) / trace[0])
