"""Terminal global reward evaluation for sampled trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..actions.action_executor import ActionExecutor
from ..actions.action_space import ActionSpace
from ..registers.executor import evaluate_register_state
from ..registers.state import RegisterState
from ..semantics.energy import ActionEnergyConfig
from ..semantics.projection import ProjectionBackend
from .sampler import Trajectory


@dataclass
class TrajectoryEvalOutput:
    rewards: torch.Tensor
    final_energies: torch.Tensor
    final_r2: torch.Tensor
    complexities: torch.Tensor
    expressions: list[Any]
    initial_energy: torch.Tensor


class GlobalTrajectoryEvaluator:
    """Evaluate complete trajectories with terminal centered residual reward."""

    def __init__(
        self,
        action_space: ActionSpace,
        energy_cfg: ActionEnergyConfig | None = None,
        complexity_penalty: float | None = None,
    ):
        self.space = action_space
        self.energy_cfg = energy_cfg or ActionEnergyConfig()
        self.proj = ProjectionBackend(self.energy_cfg.projection, self.energy_cfg.rho)
        self.executor = ActionExecutor(action_space)
        self.complexity_penalty = (
            float(self.energy_cfg.lambda_op) if complexity_penalty is None else float(complexity_penalty)
        )

    def evaluate(
        self,
        trajectories: list[Trajectory],
        x: torch.Tensor,
        y: torch.Tensor,
        initial_state: RegisterState | None = None,
    ) -> TrajectoryEvalOutput:
        if initial_state is None:
            initial_state = _initial_state_from_trajectories(trajectories)
        if initial_state is None:
            raise ValueError("initial_state is required when trajectories do not carry metadata['initial_state']")
        x = x.detach()
        y = y.detach()
        B0 = torch.nan_to_num(evaluate_register_state(initial_state, x))
        e0 = self.proj.residual_energy(B0, y)
        final_energies = []
        final_r2 = []
        complexities = []
        expressions = []
        for trajectory in trajectories:
            final_state = trajectory.metadata.get("final_state")
            if final_state is None:
                final_state = self._execute(initial_state, trajectory.actions)
            B_final = torch.nan_to_num(evaluate_register_state(final_state, x))
            energy = self.proj.residual_energy(B_final, y)
            final_energies.append(energy)
            final_r2.append(_fit_r2(B_final, y, final_state))
            complexity = float(trajectory.complexity) if trajectory.complexity else _state_complexity(final_state)
            complexities.append(complexity)
            expressions.append(final_state.exprs)
        if final_energies:
            energies_t = torch.stack(final_energies).to(dtype=B0.dtype)
            r2_t = torch.tensor(final_r2, device=B0.device, dtype=B0.dtype)
            comp_t = torch.tensor(complexities, device=B0.device, dtype=B0.dtype)
        else:
            energies_t = torch.empty(0, device=B0.device, dtype=B0.dtype)
            r2_t = torch.empty(0, device=B0.device, dtype=B0.dtype)
            comp_t = torch.empty(0, device=B0.device, dtype=B0.dtype)
        rewards = e0 - energies_t - self.complexity_penalty * comp_t
        return TrajectoryEvalOutput(
            rewards=torch.nan_to_num(rewards),
            final_energies=torch.nan_to_num(energies_t),
            final_r2=torch.nan_to_num(r2_t),
            complexities=comp_t,
            expressions=expressions,
            initial_energy=e0,
        )

    def _execute(self, state: RegisterState, actions: list[int]) -> RegisterState:
        current = state.clone()
        for action_id in actions:
            current = self.executor.execute_symbolic(current, int(action_id))
        return current


def _initial_state_from_trajectories(trajectories: list[Trajectory]) -> RegisterState | None:
    for trajectory in trajectories:
        state = trajectory.metadata.get("initial_state")
        if state is not None:
            return state
    return None


def _state_complexity(state: RegisterState) -> float:
    active = state.active.bool()
    return float(state.complexity[active].sum().detach().cpu().item()) if active.numel() else 0.0


def _fit_r2(B: torch.Tensor, y: torch.Tensor, state: RegisterState) -> float:
    active = state.active.bool()
    cols = active.nonzero(as_tuple=False).squeeze(-1)
    if cols.numel() == 0:
        cols = torch.arange(B.shape[1], device=B.device)
    A = torch.cat([B[:, cols], torch.ones(B.shape[0], 1, device=B.device, dtype=B.dtype)], dim=1)
    A = torch.nan_to_num(A)
    yy = torch.nan_to_num(y.to(device=B.device, dtype=B.dtype))
    try:
        coef = torch.linalg.lstsq(A, yy.unsqueeze(-1)).solution.squeeze(-1)
    except RuntimeError:
        gram = A.transpose(0, 1) @ A + 1e-6 * torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
        coef = torch.linalg.pinv(gram) @ A.transpose(0, 1) @ yy
    pred = A @ coef
    ss_tot = (yy - yy.mean()).square().sum()
    if float(ss_tot.detach().cpu()) <= 1e-12:
        return 0.0
    return float((1.0 - (yy - pred).square().sum() / ss_tot.clamp(min=1e-12)).detach().cpu().item())
