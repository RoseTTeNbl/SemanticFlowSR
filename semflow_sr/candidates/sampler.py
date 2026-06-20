"""Candidate samplers for complete-trajectory Semantic-Fisher flow."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import torch

from ..actions.action_executor import ActionExecutor
from ..actions.action_space import ActionSpace
from ..registers.executor import evaluate_register_state
from ..registers.state import RegisterState
from ..semantics.energy import ActionEnergy, ActionEnergyConfig
from ..sr.ops import op_cost
from .base import SemanticCandidate


class CandidateSampler:
    def sample(
        self,
        state: RegisterState,
        *,
        B: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        budget: int | None = None,
    ) -> list[SemanticCandidate]:
        raise NotImplementedError


@dataclass
class ActionCandidateSampler(CandidateSampler):
    space: ActionSpace

    def from_action_ids(self, action_ids: torch.Tensor) -> list[SemanticCandidate]:
        candidates = []
        for i, action in enumerate(action_ids.detach().cpu().tolist()):
            action_id = int(action)
            spec = self.space.decode(action_id)
            candidates.append(
                SemanticCandidate(
                    candidate_id=i,
                    kind="action",
                    actions=[action_id],
                    log_prior=0.0,
                    complexity=float(op_cost(spec.op_id)),
                    metadata={"op_id": int(spec.op_id), "horizon": 1},
                )
            )
        return candidates

    def sample(
        self,
        state: RegisterState,
        *,
        B: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        budget: int | None = None,
    ) -> list[SemanticCandidate]:
        action_ids = self.space.valid_actions(state)
        if budget is not None:
            action_ids = action_ids[: max(int(budget), 0)]
        return self.from_action_ids(action_ids)


@dataclass
class BlockCandidateSampler(CandidateSampler):
    """Semi-autoregressive terminal block sampler.

    The sampler keeps the branching bounded. When ``B`` and ``y`` are provided it
    selects first/next actions by centered one-step reward; otherwise it falls back to
    deterministic action order. The candidate reward is still terminal: the whole
    action sequence is executed into ``B^omega`` before scoring.
    """

    space: ActionSpace
    horizon: int = 3
    first_topk: int = 8
    branch_topk: int = 4
    energy_cfg: ActionEnergyConfig | None = None

    def sample(
        self,
        state: RegisterState,
        *,
        B: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        budget: int | None = None,
    ) -> list[SemanticCandidate]:
        if self.horizon < 1:
            return []
        if self.horizon == 1:
            return ActionCandidateSampler(self.space).sample(state, B=B, y=y, x=x, budget=budget)
        if B is None and x is not None:
            B = evaluate_register_state(state, x)
        energy = ActionEnergy(self.space, self.energy_cfg)
        executor = ActionExecutor(self.space)
        first_ids = self.space.valid_actions(state).to(B.device if B is not None else state.active.device)
        first_ids = _top_action_ids(energy, B, y, first_ids, self.first_topk)
        candidates: list[SemanticCandidate] = []
        max_budget = math.inf if budget is None else max(int(budget), 0)

        def extend(cur_state: RegisterState, cur_B: torch.Tensor | None, prefix: list[int]) -> None:
            if len(candidates) >= max_budget:
                return
            if len(prefix) == self.horizon:
                candidates.append(_candidate_from_actions(len(candidates), "block", self.space, prefix))
                return
            next_ids = self.space.valid_actions(cur_state).to(
                cur_B.device if cur_B is not None else cur_state.active.device
            )
            next_ids = _top_action_ids(energy, cur_B, y, next_ids, self.branch_topk)
            for action_id in next_ids.detach().cpu().tolist():
                if len(candidates) >= max_budget:
                    break
                next_state = executor.execute_symbolic(cur_state, int(action_id))
                next_B = None
                if cur_B is not None:
                    next_B = executor.execute_semantic(cur_B, torch.tensor([int(action_id)], device=cur_B.device))[0]
                extend(next_state, next_B, [*prefix, int(action_id)])

        for first in first_ids.detach().cpu().tolist():
            if len(candidates) >= max_budget:
                break
            next_state = executor.execute_symbolic(state, int(first))
            next_B = None
            if B is not None:
                next_B = executor.execute_semantic(B, torch.tensor([int(first)], device=B.device))[0]
            extend(next_state, next_B, [int(first)])
        return candidates


@dataclass
class FullCandidateSampler(CandidateSampler):
    """Sampler for full trajectory / expression candidates.

    Full candidates are expected to carry precomputed terminal semantics in
    ``metadata["B_after"]`` until expression execution is wired to the candidate
    evaluator. They live on the same candidate simplex as H1 and H3 candidates.
    """

    precomputed: list[SemanticCandidate] | None = None

    def sample(
        self,
        state: RegisterState,
        *,
        B: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        budget: int | None = None,
    ) -> list[SemanticCandidate]:
        items = list(self.precomputed or [])
        if budget is not None:
            items = items[: max(int(budget), 0)]
        out: list[SemanticCandidate] = []
        for i, cand in enumerate(items):
            metadata = dict(cand.metadata)
            metadata.setdefault("horizon", "full")
            out.append(
                replace(
                    cand,
                    candidate_id=i,
                    kind="full",
                    metadata=metadata,
                )
            )
        return out


@dataclass
class ExpressionCandidateSampler(FullCandidateSampler):
    """Backward-compatible alias for precomputed full-expression candidates."""

    expressions: list | None = None

    def __post_init__(self):
        if self.precomputed is None and self.expressions is not None:
            self.precomputed = [
                SemanticCandidate(candidate_id=i, kind="full", expr=expr, metadata={"horizon": "full"})
                for i, expr in enumerate(self.expressions)
            ]


@dataclass(frozen=True)
class CandidateSamplerGroup:
    name: str
    sampler: CandidateSampler
    budget: int | None = None


@dataclass
class TrajectoryCandidateSampler(CandidateSampler):
    """Sample complete executable candidates from named H1/H3/full groups."""

    groups: list[CandidateSamplerGroup] = field(default_factory=list)

    def sample(
        self,
        state: RegisterState,
        *,
        B: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        budget: int | None = None,
    ) -> list[SemanticCandidate]:
        out: list[SemanticCandidate] = []
        remaining = None if budget is None else max(int(budget), 0)
        for group in self.groups:
            if remaining is not None and remaining <= 0:
                break
            group_budget = group.budget
            if remaining is not None:
                group_budget = remaining if group_budget is None else min(int(group_budget), remaining)
            part = group.sampler.sample(state, B=B, y=y, x=x, budget=group_budget)
            for candidate in part:
                metadata = dict(candidate.metadata)
                metadata["candidate_group"] = group.name
                candidate = replace(candidate, candidate_id=len(out), metadata=metadata)
                out.append(candidate)
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        break
        return out


def _candidate_from_actions(
    candidate_id: int,
    kind: str,
    space: ActionSpace,
    actions: list[int],
) -> SemanticCandidate:
    complexity = 0.0
    op_ids = []
    for action_id in actions:
        spec = space.decode(int(action_id))
        op_ids.append(int(spec.op_id))
        complexity += float(op_cost(spec.op_id))
    return SemanticCandidate(
        candidate_id=candidate_id,
        kind=kind,
        actions=[int(a) for a in actions],
        log_prior=0.0,
        complexity=complexity,
        metadata={"op_ids": op_ids, "horizon": len(actions)},
    )


def _top_action_ids(
    energy: ActionEnergy,
    B: torch.Tensor | None,
    y: torch.Tensor | None,
    action_ids: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    if action_ids.numel() == 0:
        return action_ids
    k = min(max(int(topk), 0), int(action_ids.numel()))
    if k == 0:
        return action_ids[:0]
    if B is None or y is None:
        return action_ids[:k]
    rewards = energy.rewards(B, y, action_ids.to(B.device))
    pos = torch.topk(rewards, k).indices
    return action_ids.to(B.device)[pos]
