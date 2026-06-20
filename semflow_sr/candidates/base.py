"""Shared dataclasses for candidate-level flow."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SemanticCandidate:
    """Executable candidate for candidate-level Semantic-Fisher flow.

    ``actions`` is the canonical executable representation for action and block
    candidates. Expression candidates may carry precomputed semantic state in
    ``metadata`` until a full expression executor is wired in.
    """

    candidate_id: int
    kind: str
    actions: list[int] | None = None
    expr: Any | None = None
    log_prior: float = 0.0
    complexity: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class CandidateEvalOutput:
    residual_current: torch.Tensor
    residual_next: torch.Tensor
    xi: torch.Tensor
    gram: torch.Tensor
    rewards: torch.Tensor
    energies: torch.Tensor
    complexities: torch.Tensor
    log_priors: torch.Tensor
    B_after: torch.Tensor


@dataclass
class CandidateFlowTarget:
    candidates: list[SemanticCandidate]
    p_start: torch.Tensor
    scores: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    w_target: torch.Tensor
    zdot_target: torch.Tensor
    pdot_target: torch.Tensor
    p_target: torch.Tensor
    eval: CandidateEvalOutput
