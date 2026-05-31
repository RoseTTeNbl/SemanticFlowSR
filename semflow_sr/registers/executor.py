"""Evaluate a RegisterState on a probe to obtain the semantic matrix B."""
from __future__ import annotations
import torch
from ..sr.evaluator import evaluate_exprs
from .state import RegisterState


def evaluate_register_state(state: RegisterState, X: torch.Tensor) -> torch.Tensor:
    """X:[m,d] -> B:[m,K]."""
    return evaluate_exprs(state.exprs, X)
