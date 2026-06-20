"""Inference for block-only RiskFlow."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..actions.action_executor import ActionExecutor
from ..actions.action_space import ActionSpace
from ..flow.semantic_fisher_table import semantic_fisher_table_integrate
from ..registers.executor import evaluate_register_state
from ..registers.state import init_register_state
from ..semantics.energy import ActionEnergyConfig
from ..semantics.projection import ProjectionBackend
from .enumeration import block_table_mask, enumerate_executable_blocks
from .semantic_effects import compute_table_semantic_effects
from .selection import block_logprob_scores


@dataclass
class BlockRiskFlowResult:
    state: object
    energy_trace: list[float]
    diagnostics: list[dict]
    steps: int


def rollout_block_risk_flow(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    num_vars: int,
    K: int,
    ops_ids: list[int],
    block_size: int = 3,
    max_blocks: int = 4,
    block_pool_budget: int = 128,
    integration_steps: int = 2,
    max_energy_growth: float = 100.0,
    max_abs_semantic: float = 1e8,
    energy_cfg: ActionEnergyConfig | None = None,
) -> BlockRiskFlowResult:
    device = next(model.parameters()).device
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)
    state = init_register_state(num_vars, K, device=device)
    space = ActionSpace(K, ops_ids)
    executor = ActionExecutor(space)
    energy_cfg = energy_cfg or ActionEnergyConfig()
    proj = ProjectionBackend(energy_cfg.projection, energy_cfg.rho)
    diagnostics = []
    energy_trace = [_energy(state, x, y, proj)]
    model.eval()
    for block_step in range(max(int(max_blocks), 0)):
        blocks = enumerate_executable_blocks(space, state, block_size=block_size, budget=block_pool_budget)
        if not blocks:
            break
        B = torch.nan_to_num(evaluate_register_state(state, x))
        sem = compute_table_semantic_effects(
            state,
            B,
            y,
            space,
            blocks,
            block_size=block_size,
            energy_cfg=energy_cfg,
        )
        q0 = _uniform_table(sem.mask, dtype=B.dtype).to(device)
        with torch.no_grad():
            out = model(
                B=B.unsqueeze(0),
                y=y.unsqueeze(0),
                q_lambda=q0.unsqueeze(0),
                lambda_value=torch.zeros(1, device=device, dtype=B.dtype),
                mask=sem.mask.unsqueeze(0),
                zeta=sem.zeta.unsqueeze(0),
            )
        q_final = semantic_fisher_table_integrate(
            q0,
            out.lograte.squeeze(0),
            sem.mask,
            steps=integration_steps,
            dt=1.0,
        )
        block, scores = _select_healthy_executable_block(
            q_final,
            blocks,
            state,
            x,
            y,
            space,
            proj=proj,
            max_energy_growth=max_energy_growth,
            max_abs_semantic=max_abs_semantic,
        )
        for action_id in block:
            state = executor.execute_symbolic(state, int(action_id))
        energy_trace.append(_energy(state, x, y, proj))
        diagnostics.append({
            "block_step": block_step,
            "block_size": int(block_size),
            "num_executable_blocks": len(blocks),
            "selected_block": [int(a) for a in block],
            "selected_block_score": float(scores.max().detach().cpu().item()),
            "table_entropy": _table_entropy(q_final, sem.mask),
            "energy": energy_trace[-1],
        })
    return BlockRiskFlowResult(state=state, energy_trace=energy_trace, diagnostics=diagnostics, steps=len(diagnostics))


def _select_healthy_executable_block(
    q_table: torch.Tensor,
    blocks: list[tuple[int, ...]],
    state,
    x: torch.Tensor,
    y: torch.Tensor,
    space: ActionSpace,
    *,
    proj: ProjectionBackend | None = None,
    max_energy_growth: float = 100.0,
    max_abs_semantic: float = 1e8,
):
    """Select the best-probability block that does not create a numerically bad state.

    This is a validity guard. It does not score blocks by local reward; it only
    skips candidates whose semantics are non-finite, extremely large, or make the
    centered residual energy explode beyond a fixed tolerance.
    """
    scores = block_logprob_scores(q_table, blocks)
    if scores.numel() == 0:
        raise ValueError("at least one executable block is required")
    proj = proj or ProjectionBackend()
    executor = ActionExecutor(space)
    current_energy = _energy(state, x, y, proj)
    allowed_energy = max(float(current_energy) * float(max_energy_growth), float(current_energy) + 1e-9)
    order = torch.argsort(scores, descending=True).detach().cpu().tolist()
    fallback_idx = int(order[0])
    for idx in order:
        candidate_state = state.clone()
        for action_id in blocks[int(idx)]:
            candidate_state = executor.execute_symbolic(candidate_state, int(action_id))
        B = torch.nan_to_num(evaluate_register_state(candidate_state, x), nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(B).all():
            continue
        if float(B.abs().amax().detach().cpu().item()) >= float(max_abs_semantic):
            continue
        candidate_energy = float(proj.residual_energy(B, y).detach().cpu().item())
        if not torch.isfinite(torch.tensor(candidate_energy)):
            continue
        if candidate_energy <= allowed_energy:
            return tuple(int(a) for a in blocks[int(idx)]), scores
    return tuple(int(a) for a in blocks[fallback_idx]), scores


def _uniform_table(mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return mask.to(dtype=dtype) / counts.to(dtype=dtype)


def _energy(state, x: torch.Tensor, y: torch.Tensor, proj: ProjectionBackend) -> float:
    B = torch.nan_to_num(evaluate_register_state(state, x))
    return float(proj.residual_energy(B, y).detach().cpu().item())


def _table_entropy(q: torch.Tensor, mask: torch.Tensor) -> float:
    p = q.clamp(min=1e-12)
    return float((-(p * p.log()) * mask).sum().detach().cpu().item())
