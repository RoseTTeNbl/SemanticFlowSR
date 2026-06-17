"""Event records emitted by external GP/search runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
from pathlib import Path
import torch


@dataclass
class GPSemanticEvent:
    generation: int
    parent_ids: list[int]
    child_expr: Any
    parent_exprs: list[Any]
    action_or_macro: Any
    state_signature: torch.Tensor
    child_semantics: torch.Tensor
    fitness: float
    lineage_return: float = 0.0
    proposal_logprob: float = 0.0
    selection_logprob: float = 0.0
    survived: bool = False
    metadata: dict = field(default_factory=dict)


class GPEventLogger:
    """Append-only JSONL logger for GP semantic events."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: GPSemanticEvent) -> None:
        rec = {
            "generation": int(event.generation),
            "parent_ids": [int(x) for x in event.parent_ids],
            "child_expr": str(event.child_expr),
            "parent_exprs": [str(x) for x in event.parent_exprs],
            "action_or_macro": event.action_or_macro,
            "state_signature": event.state_signature.detach().cpu().tolist(),
            "child_semantics": event.child_semantics.detach().cpu().tolist(),
            "fitness": float(event.fitness),
            "lineage_return": float(event.lineage_return),
            "proposal_logprob": float(event.proposal_logprob),
            "selection_logprob": float(event.selection_logprob),
            "survived": bool(event.survived),
            "metadata": event.metadata,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
