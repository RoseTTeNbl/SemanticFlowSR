"""Base natural-flow trainer entry point.

The implementation uses the existing velocity trainer, but this module gives the
current theory a stable name: base training learns the p-space velocity of the
exponential natural-flow path induced by one-step centered semantic rewards.
"""
from __future__ import annotations

from .trainer_velocity import TrainConfig, train_velocity

__all__ = ["TrainConfig", "train_velocity"]
