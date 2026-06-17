"""Trace dataset: local action-flow supervision records.

The mainline path is the semantic-Fisher pullback flow at ``lambda=0``: the dataset
stores the current policy, action semantic effects, the pullback Gram matrix, and the
exact log-rate / sphere-tangent target. Closed-form exponential Fisher paths remain
available as ablations.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
import torch
from torch.utils.data import Dataset

from ..semantics.probe import ProbeBatch
from ..semantics.energy import ActionEnergy, ActionEnergyConfig
from ..actions.action_space import ActionSpace
from ..actions.action_features import SEMANTIC_ACTION_FEATURE_DIM
from ..actions.action_features import action_features
from ..actions.support_sampler import SupportSampler
from ..registers.executor import evaluate_register_state
from ..flow.natural_path import ExponentialNaturalFlowPath, effective_advantage_from_target
from ..flow.semantic_fisher import (
    semantic_fisher_lograte,
    semantic_fisher_simplex_velocity,
    semantic_fisher_sphere_step,
    semantic_fisher_sphere_velocity,
    integrate_semantic_fisher_teacher_path,
)
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
                 beta: float | None = None, seed: int = 0, max_support: int | None = 256,
                 support_sampler: SupportSampler | None = None,
                 cache_static: bool = True, data_device: str | torch.device = "cpu",
                 path_name: str = "semantic_fisher_pullback",
                 eta: float | None = None,
                 gamma: float = 0.1,
                 gram_rank: int | None = None,
                 flow_training: dict | None = None):
        self.records = records
        self.space = action_space
        self.prior = prior
        self.target = target
        self.energy = ActionEnergy(action_space, energy_cfg)
        if beta is None:
            beta = 1.0 if eta is None else eta
        elif eta is not None and float(eta) != float(beta):
            raise ValueError("beta and legacy eta alias disagree")
        self.beta = float(beta)
        self.eta = self.beta  # compatibility alias for older tests/scripts
        self.gamma = float(gamma)
        self.gram_rank = None if gram_rank is None else int(gram_rank)
        flow_training = flow_training or {}
        self.train_along_path = bool(flow_training.get("train_along_path", False))
        self.target_integration_steps = max(int(flow_training.get("target_integration_steps", 1)), 1)
        self.num_time_samples = max(int(flow_training.get("num_time_samples", self.target_integration_steps)), 1)
        if path_name not in {"semantic_fisher_pullback", "exponential_natural_flow"}:
            raise ValueError(f"unknown path_name: {path_name}")
        self.path_name = path_name
        self.natural_path = ExponentialNaturalFlowPath()
        self.max_support = max_support
        self._rng = random.Random(seed)
        self.support_sampler = support_sampler or SupportSampler(max_support=max_support, seed=seed)
        self.cache_static = cache_static
        self.data_device = torch.device(data_device)
        self._static_cache: dict[int, dict] = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        if self.cache_static and idx in self._static_cache:
            return self._with_fresh_path(self._static_cache[idx])
        static = self._build_static(idx)
        if self.cache_static:
            self._static_cache[idx] = static
        return self._with_fresh_path(static)

    def _build_static(self, idx: int) -> dict:
        rec = self.records[idx]
        state = rec.state
        x = rec.x.to(self.data_device)
        y = rec.y.to(self.data_device)
        B = evaluate_register_state(state, x)
        B = torch.nan_to_num(B)
        full_action_ids = self.space.valid_actions(state).to(self.data_device)
        full_eval = self.energy.evaluate_actions(B, y, full_action_ids)
        full_rewards = full_eval.rewards
        support = self.support_sampler.sample(full_action_ids, rewards=full_rewards,
                                              gt_action_id=rec.gt_action_id, sample_index=idx)
        action_ids = support.action_ids
        ev = self.energy.evaluate_actions(B, y, action_ids)
        effect = self.energy.action_semantic_effects(B, y, action_ids)
        energies = ev.energies
        rewards = ev.rewards
        ctx = {
            "gt_action": rec.gt_action_id,
            "rewards": rewards,
            "proposal_probs": support.proposal_probs.to(device=B.device, dtype=B.dtype),
            "support_mode": support.mode,
            "full_action_size": support.full_size,
            "state": state,
            "x": x,
            "y": y,
            "sample_index": idx,
        }
        p_start = self.prior.build_p0(B, y, action_ids, ctx)
        plain_p_target = self.target.build_p1(B, y, action_ids, energies, p_start, ctx)
        target_rewards = ctx.get("rewards", rewards)
        raw_advantages = ctx.get("advantages", torch.zeros_like(target_rewards))
        advantages = effective_advantage_from_target(p_start, plain_p_target, self.beta)
        w = torch.ones_like(p_start)
        if self.path_name == "semantic_fisher_pullback":
            advantages = torch.nan_to_num(raw_advantages.to(device=B.device, dtype=B.dtype))
            w_target = semantic_fisher_lograte(
                p_start,
                advantages,
                effect.gram,
                beta=self.beta,
                gamma=self.gamma,
                gram_rank=self.gram_rank,
                gram_factors=effect.xi,
            )
            z0 = p_start.clamp(min=1e-12).sqrt()
            zdot_target = semantic_fisher_sphere_velocity(z0, w_target)
            pdot_target = semantic_fisher_simplex_velocity(p_start, w_target)
            p_target = semantic_fisher_sphere_step(p_start, w_target, dt=1.0)
        else:
            p_target = plain_p_target
            w_target = None
            zdot_target = None
            pdot_target = None
        feats = action_features(self.space, state, action_ids)
        semantic_stats = _semantic_stats_from_effect(effect)
        gt_pos = (action_ids == rec.gt_action_id).nonzero(as_tuple=False)
        one_step_rewards = ctx.get("one_step_rewards", rewards)
        rollout_rewards = ctx.get("rollout_rewards", target_rewards)
        rollout_stats = ctx.get("rollout_stats", {})
        per_action = rollout_stats.get("per_action", []) if isinstance(rollout_stats, dict) else []
        if per_action:
            rollout_eval_mask = torch.tensor([bool(x.get("rollout_evaluated", False)) for x in per_action],
                                             device=B.device, dtype=torch.float32)
            rollout_rank_shift = _per_action_tensor(per_action, "rank_shift", B)
            rollout_n_rollouts = _per_action_tensor(per_action, "n_rollouts", B)
            rollout_best_score = _per_action_tensor(per_action, "best_score", B)
            rollout_score_std = _per_action_tensor(per_action, "score_std", B)
            rollout_best_final_energy = _per_action_tensor(per_action, "best_final_energy", B)
            rollout_best_final_r2 = _per_action_tensor(per_action, "best_final_r2", B)
        else:
            rollout_eval_mask = torch.zeros_like(target_rewards)
            rollout_rank_shift = torch.zeros_like(target_rewards)
            rollout_n_rollouts = torch.zeros_like(target_rewards)
            rollout_best_score = torch.zeros_like(target_rewards)
            rollout_score_std = torch.zeros_like(target_rewards)
            rollout_best_final_energy = torch.zeros_like(target_rewards)
            rollout_best_final_r2 = torch.zeros_like(target_rewards)
        return {
            "x": x, "y": y, "B": B,
            "action_ids": action_ids, "action_feats": feats,
            "semantic_stats": semantic_stats,
            "energies": energies, "rewards": target_rewards, "advantages": advantages,
            "scores": target_rewards, "target_advantages": raw_advantages,
            "one_step_rewards": one_step_rewards, "rollout_rewards": rollout_rewards,
            "rollout_eval_mask": rollout_eval_mask,
            "rollout_rank_shift": rollout_rank_shift,
            "rollout_n_rollouts": rollout_n_rollouts,
            "rollout_best_score": rollout_best_score,
            "rollout_score_std": rollout_score_std,
            "rollout_best_final_energy": rollout_best_final_energy,
            "rollout_best_final_r2": rollout_best_final_r2,
            "proposal_probs": ctx["proposal_probs"], "weights": w,
            "residual_current": effect.residual_current,
            "residual_next": effect.residual_next,
            "xi": effect.xi,
            "gram": effect.gram,
            "gamma": torch.tensor(self.gamma, dtype=B.dtype, device=B.device),
            "w_target": w_target if w_target is not None else torch.zeros_like(p_start),
            "zdot_target": zdot_target if zdot_target is not None else torch.zeros_like(p_start),
            "p_start": p_start, "p_target": p_target,
            "p0": p_start, "p1": p_target,
            "plain_p_target": plain_p_target,
            "pdot_target": pdot_target if pdot_target is not None else torch.zeros_like(p_start),
            "gt_action_pos": torch.tensor(gt_pos.item() if gt_pos.numel() else -1),
            "full_action_size": torch.tensor(support.full_size, dtype=torch.long),
        }

    def _with_fresh_path(self, static: dict) -> dict:
        out = dict(static)
        if self.path_name == "semantic_fisher_pullback":
            if self.train_along_path:
                max_idx = max(self.num_time_samples, 1)
                sample_idx = self._rng.randint(1, max_idx)
                step_idx = min(
                    int(round(sample_idx * self.target_integration_steps / max_idx)),
                    self.target_integration_steps,
                )
                step_idx = max(step_idx, 1)
                path = integrate_semantic_fisher_teacher_path(
                    out["p_start"],
                    out["advantages"],
                    out["gram"],
                    beta=self.beta,
                    gamma=self.gamma,
                    steps=self.target_integration_steps,
                    gram_rank=self.gram_rank,
                    gram_factors=out["xi"],
                )
                p_cur = path.policies[step_idx]
                w_cur = semantic_fisher_lograte(
                    p_cur, out["advantages"], out["gram"],
                    beta=self.beta, gamma=self.gamma, gram_rank=self.gram_rank,
                    gram_factors=out["xi"],
                )
                z_cur = p_cur.clamp(min=1e-12).sqrt()
                zdot_cur = semantic_fisher_sphere_velocity(z_cur, w_cur)
                out["lambda"] = torch.tensor(
                    step_idx / self.target_integration_steps,
                    dtype=torch.float32,
                    device=out["p_start"].device,
                )
                out["p_lambda"] = p_cur
                out["w_target"] = w_cur
                out["pdot_target"] = semantic_fisher_simplex_velocity(p_cur, w_cur)
                out["zdot_target"] = zdot_cur
                out["dp_dlambda"] = out["pdot_target"]
                out["z_lambda"] = z_cur
                out["dz_dlambda"] = zdot_cur
                out["p_target"] = path.policies[-1]
                out["p1"] = out["p_target"]
            else:
                z0 = out["p_start"].clamp(min=1e-12).sqrt()
                out["lambda"] = torch.tensor(0.0, dtype=torch.float32, device=out["p_start"].device)
                out["p_lambda"] = out["p_start"]
                out["dp_dlambda"] = out["pdot_target"]
                out["z_lambda"] = z0
                out["dz_dlambda"] = out["zdot_target"]
            return out
        lam = self._rng.random()
        ps = self.natural_path.sample(out["p_start"], out["advantages"], lam, eta=self.beta,
                                      scores=out.get("scores"))
        z_lambda = ps.z_lambda
        dz_dlambda = ps.dz_dlambda
        out["lambda"] = torch.tensor(lam, dtype=torch.float32, device=out["p_start"].device)
        out["p_lambda"] = ps.p_lambda
        out["dp_dlambda"] = ps.dp_dlambda
        out["z_lambda"] = z_lambda
        out["dz_dlambda"] = dz_dlambda
        return out


def _per_action_tensor(per_action: list[dict], key: str, B: torch.Tensor) -> torch.Tensor:
    return torch.tensor([float(item.get(key, 0.0)) for item in per_action],
                        device=B.device, dtype=B.dtype)


def _semantic_stats_from_effect(effect, eps: float = 1e-12) -> torch.Tensor:
    e0 = effect.residual_current
    ea = effect.residual_next
    xi = effect.xi
    gram = effect.gram
    e0_norm = e0.norm().clamp(min=eps)
    xi_norm = xi.norm(dim=-1) / e0_norm
    align = (xi * e0.unsqueeze(0)).sum(dim=-1) / (e0_norm * e0_norm)
    cos = (xi * e0.unsqueeze(0)).sum(dim=-1) / (xi.norm(dim=-1) * e0_norm).clamp(min=eps)
    gram_mean = gram.mean(dim=-1) / (e0_norm * e0_norm)
    offdiag = gram.clone()
    offdiag.fill_diagonal_(float("-inf"))
    gram_max = offdiag.max(dim=-1).values
    gram_max = torch.where(torch.isfinite(gram_max), gram_max, torch.zeros_like(gram_mean))
    gram_max = gram_max / (e0_norm * e0_norm)
    residual_next_norm = ea.norm(dim=-1) / e0_norm
    residual_drop = 0.5 * (e0.square().sum() - ea.square().sum(dim=-1))
    residual_drop = residual_drop / (0.5 * e0.square().sum()).clamp(min=eps)
    return torch.stack(
        [
            xi_norm,
            align,
            cos,
            gram_mean,
            gram_max,
            effect.op_costs,
            residual_next_norm,
            residual_drop,
        ],
        dim=-1,
    ).to(dtype=torch.float32)
