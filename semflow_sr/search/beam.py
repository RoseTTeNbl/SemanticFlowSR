"""Beam search over the velocity-induced action distribution (optional, kept simple).

At each step, for each beam state we get an action distribution from `policy_fn`, expand
the top-`beam_width` actions, score resulting states by residual energy, and keep the best
`beam_width` overall. Returns the best terminal state.
"""
from __future__ import annotations
import torch

from ..registers.state import init_register_state
from ..actions.action_space import ActionSpace
from ..actions.action_executor import ActionExecutor
from ..registers.executor import evaluate_register_state
from ..semantics.projection import ProjectionBackend
from ..semantics.energy import ActionEnergyConfig


def _residual(state, x, y, proj):
    B = torch.nan_to_num(evaluate_register_state(state, x))
    return proj.residual_energy(B, y).item()


def beam_search(policy_fn, x, y, num_vars, K, ops_ids, device, beam_width: int = 4,
                max_steps: int = 12, energy_cfg=None):
    """policy_fn(state, action_ids, x, y) -> probs over action_ids (1-D)."""
    space = ActionSpace(K, ops_ids)
    execu = ActionExecutor(space)
    proj = ProjectionBackend((energy_cfg or ActionEnergyConfig()).projection,
                             (energy_cfg or ActionEnergyConfig()).rho)
    x = x.to(device); y = y.to(device)
    init = init_register_state(num_vars, K, device)
    beams = [(init, _residual(init, x, y, proj))]
    best = beams[0]
    for _ in range(max_steps):
        cand = []
        for state, _ in beams:
            ids = space.valid_actions(state)
            if ids.numel() == 0:
                continue
            probs = policy_fn(state, ids, x, y)
            top = torch.topk(probs, min(beam_width, ids.numel())).indices
            for j in top.tolist():
                ns = execu.execute_symbolic(state, int(ids[j]))
                cand.append((ns, _residual(ns, x, y, proj)))
        if not cand:
            break
        cand.sort(key=lambda t: t[1])
        beams = cand[:beam_width]
        if beams[0][1] < best[1]:
            best = beams[0]
    return best[0], best[1]
