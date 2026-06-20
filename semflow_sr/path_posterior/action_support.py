"""Action-support helpers for the path-posterior mainline.

STOP is a virtual action local to the path-posterior algorithm. It is not part
of the global ActionSpace encoding, so legacy action-space code stays unchanged.
"""
from __future__ import annotations

import torch

from ..actions.action_features import ACTION_FEATURE_DIM, action_features
from ..actions.action_space import ActionSpace
from ..registers.state import RegisterState
from ..semantics.energy import ActionEnergy, SemanticEffectOutput

STOP_ACTION_ID = -1


def append_stop_action(action_ids: torch.Tensor, *, enabled: bool = True) -> torch.Tensor:
    ids = torch.as_tensor(action_ids, dtype=torch.long, device=action_ids.device)
    if not enabled:
        return ids
    if (ids == STOP_ACTION_ID).any():
        return ids
    return torch.cat([ids, torch.tensor([STOP_ACTION_ID], dtype=torch.long, device=ids.device)])


def is_stop_action(action_id: int) -> bool:
    return int(action_id) == STOP_ACTION_ID


def action_features_with_stop(
    space: ActionSpace,
    state: RegisterState,
    action_ids: torch.Tensor,
) -> torch.Tensor:
    ids = torch.as_tensor(action_ids, dtype=torch.long, device=action_ids.device)
    feats = torch.zeros(ids.numel(), ACTION_FEATURE_DIM, dtype=torch.float32, device=ids.device)
    normal_mask = ids != STOP_ACTION_ID
    if bool(normal_mask.any().item()):
        feats[normal_mask] = action_features(space, state, ids[normal_mask])
    if bool((~normal_mask).any().item()):
        # STOP is represented by a reserved, low-magnitude feature row. Keeping
        # it finite and simple avoids coupling STOP to any real operator id.
        feats[~normal_mask, 0] = 1.0
    return feats


def action_semantic_effects_with_stop(
    energy: ActionEnergy,
    B: torch.Tensor,
    y: torch.Tensor,
    action_ids: torch.Tensor,
) -> SemanticEffectOutput:
    ids = torch.as_tensor(action_ids, dtype=torch.long, device=action_ids.device)
    residual_current = energy.proj.residual_vector(B, y)
    A = ids.numel()
    residual_next = residual_current.unsqueeze(0).expand(A, -1).clone()
    xi = torch.zeros(A, residual_current.numel(), dtype=B.dtype, device=B.device)
    rewards = torch.zeros(A, dtype=B.dtype, device=B.device)
    op_costs = torch.zeros(A, dtype=B.dtype, device=B.device)
    B_after = B.unsqueeze(0).expand(A, -1, -1).clone()

    normal_mask = ids != STOP_ACTION_ID
    if bool(normal_mask.any().item()):
        normal_ids = ids[normal_mask]
        normal = energy.action_semantic_effects(B, y, normal_ids)
        residual_next[normal_mask] = normal.residual_next
        xi[normal_mask] = normal.xi
        rewards[normal_mask] = normal.rewards
        op_costs[normal_mask] = normal.op_costs
        B_after[normal_mask] = normal.B_after

    gram = xi @ xi.transpose(-1, -2)
    return SemanticEffectOutput(
        residual_current=residual_current,
        residual_next=residual_next,
        xi=xi,
        gram=gram,
        rewards=rewards,
        op_costs=op_costs,
        B_after=B_after,
    )


def healthy_action_ids(
    energy: ActionEnergy,
    B: torch.Tensor,
    y: torch.Tensor,
    action_ids: torch.Tensor,
    *,
    max_abs_semantic: float | None = 1e6,
    max_energy_growth: float | None = 100.0,
) -> torch.Tensor:
    """Filter numerically unsafe real actions.

    This is a validity guard, not a reward. STOP is added by the caller after
    filtering and is always safe.
    """
    ids = torch.as_tensor(action_ids, dtype=torch.long, device=action_ids.device)
    if ids.numel() == 0:
        return ids
    normal_ids = ids[ids != STOP_ACTION_ID]
    if normal_ids.numel() == 0:
        return normal_ids
    eval_out = energy.evaluate_actions(B, y, normal_ids)
    finite = torch.isfinite(eval_out.B_after).flatten(1).all(dim=1)
    finite = finite & torch.isfinite(eval_out.energies)
    if max_abs_semantic is not None:
        max_abs = eval_out.B_after.abs().flatten(1).max(dim=1).values
        finite = finite & (max_abs <= float(max_abs_semantic))
    if max_energy_growth is not None:
        base = energy.proj.residual_energy(B, y).abs().clamp(min=1.0)
        allowed = base * (1.0 + float(max_energy_growth))
        finite = finite & (eval_out.energies <= allowed)
    return normal_ids[finite]
