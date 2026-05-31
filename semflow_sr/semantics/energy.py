"""Action energy E_{B,y}(a) — the semantic conditioning mechanism for the whole flow.

  E = ½‖y-Π_a y‖²  +  λ_r r_eff(Bᵃ)  +  λ_m ‖Π_a-Π_B‖²_F  +  λ_op C_op(a)

Fully vectorized over the candidate action support.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from ..sr.ops import op_cost
from ..actions.action_space import ActionSpace
from ..actions.action_executor import ActionExecutor
from .projection import ProjectionBackend


@dataclass
class ActionEnergyConfig:
    lambda_rank: float = 0.01
    lambda_move: float = 0.01
    lambda_op: float = 0.01
    projection: str = "ridge"
    rho: float = 1e-3


class ActionEnergy:
    def __init__(self, action_space: ActionSpace, config: ActionEnergyConfig | None = None):
        self.space = action_space
        self.executor = ActionExecutor(action_space)
        self.cfg = config or ActionEnergyConfig()
        self.proj = ProjectionBackend(self.cfg.projection, self.cfg.rho)

    def compute(self, B: torch.Tensor, y: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        """B:[m,K], y:[m], action_ids:[A] -> energies:[A]."""
        Ba = self.executor.execute_semantic(B, action_ids)     # [A,m,K]
        fit = self.proj.residual_energy(Ba, y)                  # [A]
        rank = self.proj.effective_rank(Ba)                     # [A]
        move = self.proj.projection_distance(Ba, B)             # [A]
        cop = torch.tensor([op_cost(self.space.decode(int(a)).op_id) for a in action_ids.tolist()],
                           device=B.device, dtype=B.dtype)
        E = fit + self.cfg.lambda_rank * rank + self.cfg.lambda_move * move + self.cfg.lambda_op * cop
        return E
