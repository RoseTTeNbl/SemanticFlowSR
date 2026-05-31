"""Build the semantic matrix B from a register state and a probe."""
from __future__ import annotations
import torch
from ..registers.state import RegisterState
from ..registers.executor import evaluate_register_state
from .probe import ProbeBatch


def semantic_matrix(state: RegisterState, probe: ProbeBatch) -> torch.Tensor:
    """-> B:[m,K], with non-finite columns zeroed for stability."""
    B = evaluate_register_state(state, probe.x)
    return torch.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
