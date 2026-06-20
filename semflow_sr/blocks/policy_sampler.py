"""Sampling executable H-step blocks from a model table policy."""
from __future__ import annotations

import torch

from ..actions.action_executor import ActionExecutor
from ..actions.action_space import ActionSpace
from ..registers.executor import evaluate_register_state
from ..registers.state import RegisterState
from .enumeration import block_table_mask, enumerate_executable_blocks
from .semantic_effects import compute_table_semantic_effects
from .selection import block_logprob_scores
from .trajectory import BlockDecision, BlockTrajectory


def executable_block_distribution(q_table: torch.Tensor, blocks: list[tuple[int, ...]]) -> torch.Tensor:
    """Normalize factorized table probabilities over the executable block pool."""
    scores = block_logprob_scores(q_table, blocks)
    if scores.numel() == 0:
        return scores
    return torch.softmax(scores, dim=0)


def sample_executable_block_from_table(
    q_table: torch.Tensor,
    blocks: list[tuple[int, ...]],
    *,
    generator: torch.Generator | None = None,
):
    """Sample a block from exactly the model-induced executable-block distribution.

    There is intentionally no uniform exploration mixture here. More exploration in
    the main algorithm means drawing more trajectories from the same behavior policy.
    """
    probs = executable_block_distribution(q_table, blocks)
    if probs.numel() == 0:
        raise ValueError("at least one executable block is required")
    idx = int(torch.multinomial(probs, 1, generator=generator).item())
    return tuple(int(a) for a in blocks[idx]), float(probs[idx].clamp(min=1e-12).log().item()), probs


class ModelBlockTrajectorySampler:
    """Sample complete trajectories from a model-induced H x A block policy."""

    def __init__(
        self,
        action_space: ActionSpace,
        *,
        block_size: int = 3,
        block_pool_budget: int = 128,
        behavior_policy_id: str = "model",
        seed: int = 0,
    ):
        self.space = action_space
        self.block_size = int(block_size)
        self.block_pool_budget = int(block_pool_budget)
        self.behavior_policy_id = str(behavior_policy_id)
        self.executor = ActionExecutor(action_space)
        self.generator = torch.Generator().manual_seed(int(seed))

    def sample(
        self,
        *,
        task_id: str,
        initial_state: RegisterState,
        x: torch.Tensor,
        y: torch.Tensor,
        model,
        num_trajectories: int,
        max_blocks: int,
    ) -> list[BlockTrajectory]:
        out: list[BlockTrajectory] = []
        for traj_idx in range(max(int(num_trajectories), 0)):
            current = initial_state.clone()
            states = []
            decisions = []
            actions = []
            total_logprob = 0.0
            for _ in range(max(int(max_blocks), 0)):
                blocks = enumerate_executable_blocks(
                    self.space,
                    current,
                    block_size=self.block_size,
                    budget=self.block_pool_budget,
                )
                if not blocks:
                    break
                B = torch.nan_to_num(evaluate_register_state(current, x))
                sem = compute_table_semantic_effects(
                    current,
                    B,
                    y,
                    self.space,
                    blocks,
                    block_size=self.block_size,
                )
                q_old = self._policy_table(model, B, y, sem.mask, sem.zeta)
                block, logprob, _ = sample_executable_block_from_table(q_old, blocks, generator=self.generator)
                table_logprobs = [float(q_old[h, int(a)].clamp(min=1e-12).log().detach().cpu().item()) for h, a in enumerate(block)]
                state_id = _state_id(current)
                states.append(current.clone())
                decisions.append(BlockDecision(
                    state_id=state_id,
                    state=current.clone(),
                    block_actions=block,
                    logprob_old=logprob,
                    table_logprobs=table_logprobs,
                    behavior_policy_id=self.behavior_policy_id,
                    q_table=q_old.detach().cpu(),
                    candidate_blocks=list(blocks),
                ))
                total_logprob += logprob
                for action_id in block:
                    actions.append(int(action_id))
                    current = self.executor.execute_symbolic(current, int(action_id))
            if actions:
                out.append(BlockTrajectory(
                    task_id=task_id,
                    states=states,
                    decisions=decisions,
                    actions=actions,
                    trajectory_logprob=total_logprob,
                    source="model",
                    metadata={
                        "trajectory_id": f"{task_id}:{traj_idx}",
                        "final_state": current,
                        "behavior_policy_id": self.behavior_policy_id,
                    },
                ))
        return out

    def _policy_table(
        self,
        model,
        B: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        zeta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q0 = _uniform_table(mask, dtype=B.dtype)
        if zeta is None:
            zeta = torch.zeros(mask.shape[0], mask.shape[1], B.shape[0], device=B.device, dtype=B.dtype)
        else:
            zeta = zeta.to(device=B.device, dtype=B.dtype)
        if model is None:
            return q0
        with torch.no_grad():
            out = model(
                B=B.unsqueeze(0),
                y=y.unsqueeze(0),
                q_lambda=q0.unsqueeze(0),
                lambda_value=torch.zeros(1, device=B.device, dtype=B.dtype),
                mask=mask.unsqueeze(0),
                zeta=zeta.unsqueeze(0),
            )
        logits = out.lograte.squeeze(0)
        return _masked_softmax_rows(logits, mask)


def _uniform_table(mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return mask.to(dtype=dtype) / counts.to(dtype=dtype)


def _masked_softmax_rows(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~mask, -torch.inf)
    q = torch.softmax(masked, dim=-1)
    return torch.where(mask, torch.nan_to_num(q), torch.zeros_like(q))


def _state_id(state: RegisterState) -> str:
    return "|".join(str(expr) for expr in state.exprs) + ":" + "".join("1" if bool(v) else "0" for v in state.active.detach().cpu())
