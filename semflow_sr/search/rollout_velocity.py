"""Velocity-field rollout inference (theory §13).

Integrate v_θ from p0 over a λ-grid (simplex Euler step + renormalize), pick an action,
execute symbolically, repeat until residual energy ≤ ε or max steps. Also provides a
random-policy rollout for the diagnostic comparison.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
import torch

from ..registers.state import RegisterState, init_register_state
from ..registers.executor import evaluate_register_state
from ..actions.action_space import ActionSpace
from ..actions.action_executor import ActionExecutor
from ..actions.action_features import action_features
from ..semantics.energy import ActionEnergy, ActionEnergyConfig
from ..semantics.projection import ProjectionBackend
from ..geometry.weights import semantic_weights
from ..endpoints.prior_uniform import UniformPrior
from ..utils.numerical import normalize_simplex


@dataclass
class RolloutResult:
    state: RegisterState
    energy_trace: list[float]
    steps: int


def _residual_energy(state, x, y, proj):
    B = torch.nan_to_num(evaluate_register_state(state, x))
    return proj.residual_energy(B, y).item()


def rollout_velocity(model, x, y, num_vars, K, ops_ids, device,
                     max_steps: int = 16, grid: int = 5, step_size: float = 0.3,
                     eta: float = 1.0, eps: float = 1e-4, energy_cfg=None,
                     greedy: bool = True, max_support: int = 256) -> RolloutResult:
    model.eval()
    space = ActionSpace(K, ops_ids)
    execu = ActionExecutor(space)
    energy = ActionEnergy(space, energy_cfg or ActionEnergyConfig())
    proj = ProjectionBackend((energy_cfg or ActionEnergyConfig()).projection,
                             (energy_cfg or ActionEnergyConfig()).rho)
    prior = UniformPrior()
    state = init_register_state(num_vars, K, device)
    x = x.to(device); y = y.to(device)
    trace = [_residual_energy(state, x, y, proj)]

    for _ in range(max_steps):
        if trace[-1] <= eps:
            break
        action_ids = space.valid_actions(state)
        if action_ids.numel() > max_support:
            action_ids = action_ids[torch.randperm(action_ids.numel())[:max_support]]
        B = torch.nan_to_num(evaluate_register_state(state, x))
        energies = energy.compute(B, y, action_ids)
        w = semantic_weights(energies, eta)
        feats = action_features(space, state, action_ids)
        p = prior.build_p0(B, y, action_ids, {})
        with torch.no_grad():
            for j in range(grid):
                lam = torch.tensor([j / max(grid - 1, 1)], device=device)
                out = model(x=x.unsqueeze(0), y=y.unsqueeze(0), B=B.unsqueeze(0),
                            p_lambda=p.unsqueeze(0), lambda_value=lam,
                            action_feats=feats.unsqueeze(0), energies=energies.unsqueeze(0),
                            weights=w.unsqueeze(0),
                            action_mask=torch.ones(1, action_ids.numel(), dtype=torch.bool, device=device))
                p = normalize_simplex(p + step_size * out.v_pred.squeeze(0))
        choice = int(action_ids[p.argmax()]) if greedy else int(action_ids[torch.multinomial(p, 1)])
        state = execu.execute_symbolic(state, choice)
        trace.append(_residual_energy(state, x, y, proj))
    return RolloutResult(state=state, energy_trace=trace, steps=len(trace) - 1)


def rollout_random(x, y, num_vars, K, ops_ids, device, max_steps: int = 16,
                   energy_cfg=None, seed: int = 0) -> RolloutResult:
    rng = random.Random(seed)
    space = ActionSpace(K, ops_ids)
    execu = ActionExecutor(space)
    proj = ProjectionBackend((energy_cfg or ActionEnergyConfig()).projection,
                             (energy_cfg or ActionEnergyConfig()).rho)
    state = init_register_state(num_vars, K, device)
    x = x.to(device); y = y.to(device)
    trace = [_residual_energy(state, x, y, proj)]
    for _ in range(max_steps):
        ids = space.valid_actions(state)
        if ids.numel() == 0:
            break
        choice = int(ids[rng.randrange(ids.numel())])
        state = execu.execute_symbolic(state, choice)
        trace.append(_residual_energy(state, x, y, proj))
    return RolloutResult(state=state, energy_trace=trace, steps=len(trace) - 1)
