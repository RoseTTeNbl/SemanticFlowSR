"""Candidate evaluation on centered semantic residuals."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from ..actions.action_executor import ActionExecutor
from ..actions.action_space import ActionSpace
from ..semantics.energy import ActionEnergyConfig
from ..semantics.projection import ProjectionBackend
from ..sr.ops import op_cost
from ..registers.state import RegisterState
from .base import CandidateEvalOutput, SemanticCandidate


@dataclass
class CandidateEvaluator:
    space: ActionSpace
    energy_cfg: ActionEnergyConfig | None = None

    def __post_init__(self):
        self.energy_cfg = self.energy_cfg or ActionEnergyConfig()
        self.executor = ActionExecutor(self.space)
        self.proj = ProjectionBackend(self.energy_cfg.projection, self.energy_cfg.rho)

    def evaluate(
        self,
        state: RegisterState,
        B: torch.Tensor,
        y: torch.Tensor,
        candidates: list[SemanticCandidate],
    ) -> CandidateEvalOutput:
        if not candidates:
            raise ValueError("CandidateEvaluator requires at least one candidate")
        residual_current = self.proj.residual_vector(B, y)
        base_energy = self.proj.residual_energy(B, y)
        B_after = torch.stack([self._execute_candidate_semantic(B, cand) for cand in candidates], dim=0)
        residual_next = self.proj.residual_vector(B_after, y)
        after_energy = self.proj.residual_energy(B_after, y)
        complexities = torch.tensor([self._candidate_complexity(c) for c in candidates], device=B.device, dtype=B.dtype)
        log_priors = torch.tensor([float(c.log_prior) for c in candidates], device=B.device, dtype=B.dtype)
        energies = after_energy + float(self.energy_cfg.lambda_op) * complexities
        rewards = base_energy - after_energy - float(self.energy_cfg.lambda_op) * complexities
        xi = residual_current.unsqueeze(0) - residual_next
        gram = xi @ xi.transpose(-1, -2)
        return CandidateEvalOutput(
            residual_current=residual_current,
            residual_next=residual_next,
            xi=xi,
            gram=gram,
            rewards=rewards,
            energies=energies,
            complexities=complexities,
            log_priors=log_priors,
            B_after=B_after,
        )

    def _execute_candidate_semantic(self, B: torch.Tensor, candidate: SemanticCandidate) -> torch.Tensor:
        if candidate.kind in {"action", "block"}:
            cur = B
            for action_id in candidate.actions or []:
                cur = self.executor.execute_semantic(
                    cur,
                    torch.tensor([int(action_id)], device=B.device, dtype=torch.long),
                )[0]
            return cur
        if candidate.kind in {"full", "expression"} and "B_after" in candidate.metadata:
            return candidate.metadata["B_after"].to(device=B.device, dtype=B.dtype)
        raise NotImplementedError(
            "full/expression candidates require metadata['B_after'] until expression execution is wired"
        )

    def _candidate_complexity(self, candidate: SemanticCandidate) -> float:
        if candidate.complexity:
            return float(candidate.complexity)
        total = 0.0
        for action_id in candidate.actions or []:
            total += float(op_cost(self.space.decode(int(action_id)).op_id))
        return total
