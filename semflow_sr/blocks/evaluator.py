"""Terminal reward evaluation for block trajectories."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..actions.action_executor import ActionExecutor
from ..actions.action_space import ActionSpace
from ..registers.executor import evaluate_register_state
from ..registers.state import RegisterState
from ..semantics.energy import ActionEnergyConfig
from ..semantics.projection import ProjectionBackend
from .trajectory import BlockTrajectory


@dataclass
class BlockTrajectoryEvalOutput:
    rewards: torch.Tensor
    final_energies: torch.Tensor
    final_r2: torch.Tensor
    complexities: torch.Tensor


class BlockTrajectoryEvaluator:
    def __init__(self, action_space: ActionSpace, energy_cfg: ActionEnergyConfig | None = None):
        self.space = action_space
        self.executor = ActionExecutor(action_space)
        self.cfg = energy_cfg or ActionEnergyConfig()
        self.proj = ProjectionBackend(self.cfg.projection, self.cfg.rho)

    def evaluate(
        self,
        trajectories: list[BlockTrajectory],
        x: torch.Tensor,
        y: torch.Tensor,
        initial_state: RegisterState,
    ) -> BlockTrajectoryEvalOutput:
        b0 = torch.nan_to_num(evaluate_register_state(initial_state, x))
        e0 = self.proj.residual_energy(b0, y)
        energies = []
        r2s = []
        complexities = []
        for trajectory in trajectories:
            final_state = trajectory.metadata.get("final_state")
            if final_state is None:
                final_state = self._execute(initial_state, trajectory.actions)
                trajectory.metadata["final_state"] = final_state
            b_final = torch.nan_to_num(evaluate_register_state(final_state, x))
            energy = self.proj.residual_energy(b_final, y)
            energies.append(energy)
            r2s.append(_fit_r2(b_final, y, final_state))
            complexities.append(_state_complexity(final_state))
        if energies:
            energy_t = torch.stack(energies).to(dtype=b0.dtype, device=b0.device)
            r2_t = torch.tensor(r2s, dtype=b0.dtype, device=b0.device)
            comp_t = torch.tensor(complexities, dtype=b0.dtype, device=b0.device)
        else:
            energy_t = torch.empty(0, dtype=b0.dtype, device=b0.device)
            r2_t = torch.empty(0, dtype=b0.dtype, device=b0.device)
            comp_t = torch.empty(0, dtype=b0.dtype, device=b0.device)
        rewards = e0 - energy_t - float(self.cfg.lambda_op) * comp_t
        for trajectory, reward, r2, comp in zip(trajectories, rewards, r2_t, comp_t):
            trajectory.reward = float(reward.detach().cpu().item())
            trajectory.final_r2 = float(r2.detach().cpu().item())
            trajectory.complexity = float(comp.detach().cpu().item())
        return BlockTrajectoryEvalOutput(
            rewards=torch.nan_to_num(rewards),
            final_energies=torch.nan_to_num(energy_t),
            final_r2=torch.nan_to_num(r2_t),
            complexities=comp_t,
        )

    def _execute(self, state: RegisterState, actions: list[int]) -> RegisterState:
        current = state.clone()
        for action_id in actions:
            current = self.executor.execute_symbolic(current, int(action_id))
        return current


def _state_complexity(state: RegisterState) -> float:
    active = state.active.bool()
    return float(state.complexity[active].sum().detach().cpu().item()) if active.numel() else 0.0


def _fit_r2(B: torch.Tensor, y: torch.Tensor, state: RegisterState) -> float:
    active = state.active.bool()
    cols = active.nonzero(as_tuple=False).squeeze(-1)
    if cols.numel() == 0:
        cols = torch.arange(B.shape[1], device=B.device)
    A = torch.cat([B[:, cols], torch.ones(B.shape[0], 1, device=B.device, dtype=B.dtype)], dim=1)
    yy = torch.nan_to_num(y.to(device=B.device, dtype=B.dtype))
    A = torch.nan_to_num(A)
    try:
        coef = torch.linalg.lstsq(A, yy.unsqueeze(-1)).solution.squeeze(-1)
    except RuntimeError:
        coef = torch.linalg.pinv(A) @ yy
    pred = A @ coef
    ss = (yy - yy.mean()).square().sum()
    if float(ss.detach().cpu()) <= 1e-12:
        return 0.0
    return float((1.0 - (yy - pred).square().sum() / ss.clamp(min=1e-12)).detach().cpu().item())
