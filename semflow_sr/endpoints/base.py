"""Abstract endpoint providers. p0 (prior) and p1 (target) define the velocity path."""
from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class PriorEndpoint(ABC):
    @abstractmethod
    def build_p0(self, B: torch.Tensor, y: torch.Tensor, action_ids: torch.Tensor,
                 context: dict) -> torch.Tensor: ...


class TargetEndpoint(ABC):
    @abstractmethod
    def build_p1(self, B: torch.Tensor, y: torch.Tensor, action_ids: torch.Tensor,
                 energies: torch.Tensor, p0: torch.Tensor, context: dict) -> torch.Tensor: ...
