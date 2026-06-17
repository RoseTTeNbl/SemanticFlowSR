"""Legacy endpoint-provider interfaces.

The current training code names the start and target distributions p_start and
p_target. These p0/p1 methods remain as compatibility shims for older endpoint
providers and ablation entry points.
"""
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
