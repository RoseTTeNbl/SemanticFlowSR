"""Register-program trace: an ordered list of (state_before, action_id) steps that
build a target expression. Used to generate the local velocity-flow trace dataset."""
from __future__ import annotations
from dataclasses import dataclass, field

from .state import RegisterState


@dataclass
class TraceStep:
    state: RegisterState     # state BEFORE the action
    action_id: int           # ground-truth action taken
    write: int               # write register (target slot after this step)


@dataclass
class RegisterTrace:
    steps: list[TraceStep] = field(default_factory=list)
    final_state: RegisterState | None = None
    target_register: int = 0          # register holding the final expression
    num_vars: int = 1

    def __len__(self) -> int:
        return len(self.steps)
