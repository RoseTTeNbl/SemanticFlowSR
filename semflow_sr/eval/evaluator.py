"""Evaluator: fit the velocity-rollout searcher to an SRTask and report metrics.

The rollout produces a register state; we read out the target register expression, refit
a least-squares scale/offset on train, and report R²/NMSE/complexity on test.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from ..data.benchmark_loader import SRTask
from ..search.rollout_velocity import rollout_velocity
from ..sr.evaluator import evaluate_exprs
from ..sr.printer import to_string
from .metrics import r2_score, nmse, energy_decrease_ratio


@dataclass
class EvalReport:
    name: str
    r2: float
    nmse: float
    complexity: int
    expression: str
    energy_trace: list[float]


def _readout(state, X_np, y_np):
    """Pick the register column best correlated with y, linear-refit, return (col, expr_idx)."""
    X = torch.tensor(X_np, dtype=torch.float32)
    B = torch.nan_to_num(evaluate_exprs(state.exprs, X)).numpy()
    y = y_np
    best_k, best_r2, best_pred = 0, -1e9, None
    for k in range(B.shape[1]):
        b = B[:, k]
        A = np.stack([b, np.ones_like(b)], 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        r2 = r2_score(y, pred)
        if r2 > best_r2:
            best_k, best_r2, best_pred = k, r2, (coef,)
    return best_k, best_pred[0]


def evaluate_task(model, task: SRTask, K: int, ops_ids, device, **rollout_kw) -> EvalReport:
    num_vars = task.X_train.shape[1]
    x = torch.tensor(task.X_train, dtype=torch.float32)
    y = torch.tensor(task.y_train, dtype=torch.float32)
    res = rollout_velocity(model, x, y, num_vars, K, ops_ids, device, **rollout_kw)
    k, coef = _readout(res.state, task.X_train, task.y_train)
    Xte = torch.tensor(task.X_test, dtype=torch.float32)
    Bte = torch.nan_to_num(evaluate_exprs(res.state.exprs, Xte)).numpy()
    pred = coef[0] * Bte[:, k] + coef[1]
    expr_str = to_string(res.state.exprs[k], num_vars, simplify=True)
    return EvalReport(task.name, r2_score(task.y_test, pred), nmse(task.y_test, pred),
                      res.state.exprs[k].complexity, expr_str, res.energy_trace)
