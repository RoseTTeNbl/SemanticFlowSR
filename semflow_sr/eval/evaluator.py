"""Evaluator: fit the velocity-rollout searcher to an SRTask and report metrics.

The rollout produces a register state; we read out a least-squares linear combination over
ALL active register columns (the projection that the action energy already uses, so the
operator coefficients are auto-optimized), then report R²/NMSE/complexity on test.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from ..data.benchmark_loader import SRTask
from ..search.rollout_velocity import rollout_velocity
from ..sr.evaluator import evaluate_exprs
from ..sr.printer import to_string
from .metrics import (r2_score, r2_zero, nmse, accuracy_tau, simplicity,
                      accuracy_rate, energy_decrease_ratio)


@dataclass
class EvalReport:
    name: str
    r2: float
    nmse: float
    complexity: int
    expression: str
    energy_trace: list[float]
    r2_zero: float = 0.0
    acc_tau: float = 0.0
    simplicity: float = 0.0
    steps: int = 0
    energy_decrease: float = 0.0
    solved: bool = False
    diagnostics: list[dict] | None = None
    task_metadata: dict | None = None
    active_columns: list[int] | None = None
    readout_coefficients: list[float] | None = None

    def to_record(self) -> dict:
        return {"name": self.name, "r2": self.r2, "r2_zero": self.r2_zero,
                "acc_tau": self.acc_tau, "nmse": self.nmse, "complexity": self.complexity,
                "simplicity": self.simplicity, "steps": self.steps,
                "energy_decrease": self.energy_decrease, "solved": self.solved,
                "expression": self.expression, "energy_trace": self.energy_trace,
                "diagnostics": self.diagnostics or [],
                "task_metadata": self.task_metadata or {},
                "active_columns": self.active_columns or [],
                "readout_coefficients": self.readout_coefficients or []}


def _active_cols(state) -> list[int]:
    act = state.active.bool().tolist()
    return [k for k in range(len(state.exprs)) if act[k]] or list(range(len(state.exprs)))


def _select_healthy_cols(B: np.ndarray, cols: list[int], max_norm: float = 1e6) -> list[int]:
    """溢出列挑选(而非清零): 保留在探针上处处有限、列范数适中的列, 剔除 exp/div 溢出的病态列。
    若全部被剔除则保留原列(交由下游 nan_to_num 兜底)。"""
    keep = [k for k in cols
            if np.all(np.isfinite(B[:, k])) and 0.0 < np.linalg.norm(B[:, k]) < max_norm]
    return keep or cols


def _fit_coef(B: np.ndarray, y: np.ndarray, cols: list[int], rho: float = 1e-6):
    """Least-squares y ≈ Σ_k c_k B[:,k] + c0 over active columns (ridge for stability)."""
    A = np.concatenate([B[:, cols], np.ones((B.shape[0], 1))], axis=1)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)   # exp/div 可能溢出 -> 清理
    G = A.T @ A + rho * np.eye(A.shape[1])
    try:
        return np.linalg.solve(G, A.T @ y)            # [len(cols)+1]
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, y, rcond=None)[0]   # 奇异 -> 最小二乘回退


def _predict(B: np.ndarray, cols: list[int], coef: np.ndarray) -> np.ndarray:
    A = np.concatenate([B[:, cols], np.ones((B.shape[0], 1))], axis=1)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return A @ coef


def evaluate_task(model, task: SRTask, K: int, ops_ids, device, **rollout_kw) -> EvalReport:
    num_vars = task.X_train.shape[1]
    x = torch.tensor(task.X_train, dtype=torch.float32)
    y = torch.tensor(task.y_train, dtype=torch.float32)
    res = rollout_velocity(model, x, y, num_vars, K, ops_ids, device, **rollout_kw)

    cols = _active_cols(res.state)
    Btr_raw = evaluate_exprs(res.state.exprs, x).numpy()       # 未清理, 判断列健康度
    Xte = torch.tensor(task.X_test, dtype=torch.float32)
    Bte_raw = evaluate_exprs(res.state.exprs, Xte).numpy()
    # 溢出列挑选: 列在训练+测试点上都需健康(剔除外推时 exp/div 溢出的列, 防止预测爆炸)
    cols = _select_healthy_cols(np.concatenate([Btr_raw, Bte_raw], 0), cols)
    Btr = np.nan_to_num(Btr_raw)
    coef = _fit_coef(Btr, task.y_train, cols)

    Bte = np.nan_to_num(Bte_raw)
    pred = _predict(Bte, cols, coef)

    # symbolic readout: linear combination of the columns with non-negligible coefficients
    terms = [f"{coef[i]:.4g}*({to_string(res.state.exprs[c], num_vars, simplify=True)})"
             for i, c in enumerate(cols) if abs(coef[i]) > 1e-4]
    if abs(coef[-1]) > 1e-4:
        terms.append(f"{coef[-1]:.4g}")
    expr_str = " + ".join(terms) if terms else "0"
    complexity = sum(res.state.exprs[c].complexity for i, c in enumerate(cols) if abs(coef[i]) > 1e-4)
    r2 = r2_score(task.y_test, pred)
    meta = dict(task.metadata)
    meta.update({"n_vars": int(num_vars), "variables": list(task.variable_names),
                 "ground_truth": task.expression})
    return EvalReport(task.name, r2, nmse(task.y_test, pred), int(complexity), expr_str,
                      res.energy_trace, r2_zero=r2_zero(task.y_test, pred),
                      acc_tau=accuracy_tau(task.y_test, pred), simplicity=simplicity(int(complexity)),
                      steps=res.steps, energy_decrease=energy_decrease_ratio(res.energy_trace),
                      solved=accuracy_rate(r2), diagnostics=res.diagnostics,
                      task_metadata=meta, active_columns=[int(c) for c in cols],
                      readout_coefficients=[float(c) for c in coef.tolist()])
