"""Trace dataset: per-step velocity-matching samples.

Each sample materializes the full geometry pipeline for one (state, gt_action) step:
  B, y, valid_action_ids, energies, weights, p0, p1, λ, p_λ, ṗ_λ, gt_action_id.
p_λ / ṗ_λ come from the closed-form semantic Fisher slerp. λ is resampled per epoch
in `build_sample`, so a step yields fresh λ each time it is drawn.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
import torch
from torch.utils.data import Dataset

from ..semantics.probe import ProbeBatch
from ..semantics.energy import ActionEnergy, ActionEnergyConfig
from ..actions.action_space import ActionSpace
from ..actions.action_features import action_features
from ..registers.executor import evaluate_register_state
from ..geometry.weights import semantic_weights
from ..geometry.slerp_path import SemanticFisherSlerpPath
from ..endpoints.base import PriorEndpoint, TargetEndpoint
from ..registers.trace import TraceStep


@dataclass
class StepRecord:
    """A lightweight, picklable record of one trace step (no tensors materialized yet)."""
    state: object            # RegisterState
    gt_action_id: int
    x: torch.Tensor          # [m,d]
    y: torch.Tensor          # [m]


def build_step_records(trace, x, y) -> list[StepRecord]:
    return [StepRecord(state=s.state, gt_action_id=s.action_id, x=x, y=y) for s in trace.steps]


class VelocityTraceDataset(Dataset):
    def __init__(self, records: list[StepRecord], action_space: ActionSpace,
                 prior: PriorEndpoint, target: TargetEndpoint,
                 energy_cfg: ActionEnergyConfig | None = None,
                 eta: float = 1.0, seed: int = 0, max_support: int | None = 256):
        self.records = records
        self.space = action_space
        self.prior = prior
        self.target = target
        self.energy = ActionEnergy(action_space, energy_cfg)
        self.eta = eta
        self.path = SemanticFisherSlerpPath()
        self.max_support = max_support
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        state = rec.state
        B = evaluate_register_state(state, rec.x)
        B = torch.nan_to_num(B)
        y = rec.y
        action_ids = self.space.valid_actions(state)
        # keep GT action in support; optionally subsample for tractability
        if self.max_support is not None and action_ids.numel() > self.max_support:
            action_ids = self._subsample(action_ids, rec.gt_action_id)
        energies = self.energy.compute(B, y, action_ids)
        w = semantic_weights(energies, self.eta)
        ctx = {"gt_action": rec.gt_action_id}
        p0 = self.prior.build_p0(B, y, action_ids, ctx)
        p1 = self.target.build_p1(B, y, action_ids, energies, p0, ctx)
        lam = self._rng.random()
        ps = self.path.sample(p0, p1, w, lam)
        feats = action_features(self.space, state, action_ids)
        gt_pos = (action_ids == rec.gt_action_id).nonzero(as_tuple=False)
        return {
            "x": rec.x, "y": y, "B": B,
            "action_ids": action_ids, "action_feats": feats,
            "energies": energies, "weights": w,
            "p0": p0, "p1": p1, "lambda": torch.tensor(lam, dtype=torch.float32),
            "p_lambda": ps.p_lambda, "dp_dlambda": ps.dp_dlambda,
            "gt_action_pos": torch.tensor(gt_pos.item() if gt_pos.numel() else -1),
        }

    def _subsample(self, action_ids: torch.Tensor, gt: int) -> torch.Tensor:
        n = action_ids.numel()
        perm = torch.randperm(n)[: self.max_support]
        sub = action_ids[perm]
        if (sub == gt).any():
            return sub
        sub[0] = gt
        return sub
