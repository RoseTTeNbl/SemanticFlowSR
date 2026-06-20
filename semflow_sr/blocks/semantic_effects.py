"""Whole-block semantic effects projected to H x A table coordinates."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..actions.action_executor import ActionExecutor
from ..actions.action_space import ActionSpace
from ..registers.state import RegisterState
from ..semantics.energy import ActionEnergyConfig
from ..semantics.projection import ProjectionBackend
from .enumeration import block_table_mask


@dataclass
class TableSemanticEffectOutput:
    zeta: torch.Tensor
    mask: torch.Tensor
    factors: torch.Tensor
    xi_blocks: torch.Tensor
    block_weights: torch.Tensor


def compute_table_semantic_effects(
    state: RegisterState,
    B: torch.Tensor,
    y: torch.Tensor,
    action_space: ActionSpace,
    blocks: list[tuple[int, ...]],
    *,
    block_size: int,
    block_weights: torch.Tensor | None = None,
    energy_cfg: ActionEnergyConfig | None = None,
) -> TableSemanticEffectOutput:
    energy_cfg = energy_cfg or ActionEnergyConfig()
    proj = ProjectionBackend(energy_cfg.projection, energy_cfg.rho)
    executor = ActionExecutor(action_space)
    residual_current = proj.residual_vector(B, y)
    xi_rows = []
    for block in blocks:
        B_after = _execute_block_semantic(B, action_space, block[: int(block_size)])
        residual_next = proj.residual_vector(B_after, y)
        xi_rows.append(residual_current - residual_next)
    xi = torch.stack(xi_rows) if xi_rows else torch.empty(0, B.shape[0], device=B.device, dtype=B.dtype)
    weights = (
        torch.ones(len(blocks), device=B.device, dtype=B.dtype)
        if block_weights is None
        else block_weights.to(device=B.device, dtype=B.dtype)
    )
    mask = block_table_mask(blocks, block_size=int(block_size), action_vocab_size=action_space.size).to(device=B.device)
    zeta = torch.zeros(int(block_size), action_space.size, xi.shape[-1] if xi.numel() else B.shape[0], device=B.device, dtype=B.dtype)
    denom = torch.zeros(int(block_size), action_space.size, device=B.device, dtype=B.dtype)
    for idx, block in enumerate(blocks):
        for h, action_id in enumerate(block[: int(block_size)]):
            a = int(action_id)
            zeta[h, a] += weights[idx] * xi[idx]
            denom[h, a] += weights[idx]
    zeta = torch.where(denom.unsqueeze(-1) > 0, zeta / denom.clamp(min=1e-12).unsqueeze(-1), zeta)
    factors = zeta[mask]
    return TableSemanticEffectOutput(zeta=zeta, mask=mask, factors=factors, xi_blocks=xi, block_weights=weights)


def _execute_block_semantic(B: torch.Tensor, action_space: ActionSpace, actions: tuple[int, ...]) -> torch.Tensor:
    executor = ActionExecutor(action_space)
    current = B
    for action_id in actions:
        current = executor.execute_semantic(current, torch.tensor([int(action_id)], device=B.device)).squeeze(0)
    return current
