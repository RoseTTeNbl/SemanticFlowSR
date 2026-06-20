"""Completion policy descriptors for legacy rollout/search target construction.

The current online evaluator implements random, semantic_greedy and mixed policies.
This module keeps the policy names explicit so proximal target metadata can state
which evaluator generated future-value scores.
"""
from __future__ import annotations

from dataclasses import dataclass


VALID_COMPLETION_POLICIES = {"random", "semantic_greedy", "mixed", "gp_guided"}


@dataclass(frozen=True)
class CompletionPolicySpec:
    name: str = "mixed"
    on_policy: bool = False

    def __post_init__(self):
        if self.name not in VALID_COMPLETION_POLICIES:
            raise ValueError(f"unknown completion policy: {self.name}")
