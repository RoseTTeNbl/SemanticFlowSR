"""Action validity masks (re-exposed for the model to mask invalid actions)."""
from __future__ import annotations
import torch
from .action_space import ActionSpace
from ..registers.state import RegisterState


def support_mask(action_space: ActionSpace, state: RegisterState) -> torch.Tensor:
    return action_space.valid_mask(state)
