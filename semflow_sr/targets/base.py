"""Shared data structures for semantic proximal flow targets."""
from __future__ import annotations

from dataclasses import dataclass, field
import torch

from ..registers.state import RegisterState
from ..utils.numerical import EPS, normalize_simplex


@dataclass
class LocalCondition:
    """Local action-simplex condition c=(B,y,S)."""

    state: RegisterState
    B: torch.Tensor
    y: torch.Tensor
    action_ids: torch.Tensor
    support_metadata: dict = field(default_factory=dict)


@dataclass
class PolicyDistribution:
    """Positive support-local policy distribution."""

    probs: torch.Tensor
    source: str
    logits: torch.Tensor | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.probs = normalize_simplex(self.probs.clamp(min=EPS), dim=-1)


@dataclass
class AdvantageOutput:
    scores: torch.Tensor
    advantages: torch.Tensor
    score_mean: torch.Tensor
    score_std: torch.Tensor
    metadata: dict = field(default_factory=dict)
