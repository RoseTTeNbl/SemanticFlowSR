"""TargetSampler endpoint builders for Semantic-Fisher Flow Matching.

Target samplers produce endpoint probability shapes ``q_hat`` only. Semantic
effects and Gram matrices are computed by the teacher builder.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import random

import torch

from ..actions.action_executor import ActionExecutor
from ..actions.action_space import ActionSpace
from ..registers.executor import evaluate_register_state
from ..registers.state import RegisterState
from ..semantics.energy import ActionEnergy, ActionEnergyConfig
from ..sr.ops import op_cost
from ..gp_distill.trace_likelihood import compute_gp_individual_logprob
from .action_support import STOP_ACTION_ID, append_stop_action, healthy_action_ids, is_stop_action


@dataclass(frozen=True)
class TargetShape:
    action_ids: torch.Tensor
    q_hat: torch.Tensor
    target_scores: torch.Tensor
    target_counts: torch.Tensor
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PriorConfig:
    stop_bias_base: float = -2.0
    stop_bias_slope: float = 0.35
    mode: str = "stop_bias"


@dataclass(frozen=True)
class TargetSamplerConfig:
    temperature: float = 1.0
    rank_eta: float = 2.0
    smoothing: float = 1e-3
    score_to_shape: str = "rank_softmax"
    advantage_eps: float = 1e-6
    advantage_clip: float | None = 5.0


@dataclass(frozen=True)
class FutureGroupTargetConfig(TargetSamplerConfig):
    rollout_depth: int = 3
    rollouts_per_action: int = 1
    topk: int = 1
    max_rollout_support: int | None = 16
    terminal_op_penalty: float = 0.0
    cache_path: str | None = None
    gp_population_path: str | None = None
    shape_samples: int = 32
    gp_likelihood_weight: float = 1.0
    gp_fitness_weight: float = 1.0
    importance_samples: int | None = None
    mcmc_burn_in: int = 16


def build_p_init(
    action_ids: torch.Tensor,
    *,
    step: int,
    cfg: PriorConfig | None = None,
) -> torch.Tensor:
    """Deterministic initial probability shape over the local support."""
    cfg = cfg or PriorConfig()
    mode = str(cfg.mode).strip().lower()
    ids = torch.as_tensor(action_ids, dtype=torch.long)
    logits = torch.zeros(ids.numel(), dtype=torch.float32, device=ids.device)
    if mode not in {"stop_bias", "uniform"}:
        raise ValueError(f"unknown p_init mode: {cfg.mode}")
    if mode == "stop_bias":
        stop_mask = ids == STOP_ACTION_ID
        if bool(stop_mask.any().item()):
            logits[stop_mask] = float(cfg.stop_bias_base) + float(cfg.stop_bias_slope) * float(max(int(step), 0))
    return torch.softmax(logits, dim=0)


def rank_softmax_target(
    scores: torch.Tensor,
    p_init: torch.Tensor,
    *,
    eta: float = 2.0,
    smoothing: float = 1e-3,
) -> torch.Tensor:
    scores = torch.nan_to_num(torch.as_tensor(scores, dtype=torch.float32))
    p = _normalize(torch.as_tensor(p_init, dtype=torch.float32))
    if scores.numel() == 0:
        return scores
    if scores.numel() == 1:
        return torch.ones_like(scores)
    order = scores.argsort(descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(0, order, torch.arange(scores.numel(), dtype=order.dtype, device=scores.device))
    rank_norm = 1.0 - ranks.to(torch.float32) / float(scores.numel() - 1)
    q = torch.softmax(float(eta) * rank_norm, dim=0)
    return _normalize(q + float(smoothing) * p.to(device=q.device))


def group_exp_target(
    scores: torch.Tensor,
    p_init: torch.Tensor,
    *,
    eta: float = 1.0,
    smoothing: float = 1e-3,
    advantage_eps: float = 1e-6,
    advantage_clip: float | None = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an archive-compatible Boltzmann tilt target from group scores."""
    s = torch.nan_to_num(torch.as_tensor(scores, dtype=torch.float32))
    p = _normalize(torch.as_tensor(p_init, dtype=torch.float32).to(device=s.device))
    if s.numel() == 0:
        return s, s
    centered = s - s.mean()
    std = centered.std(unbiased=False)
    if float(std.item()) <= float(advantage_eps):
        advantages = torch.zeros_like(centered)
    else:
        advantages = centered / std.clamp_min(float(advantage_eps))
    advantages = advantages - advantages.mean()
    if advantage_clip is not None:
        advantages = advantages.clamp(min=-float(advantage_clip), max=float(advantage_clip))
    logits = float(eta) * advantages
    logits = logits - logits.max()
    q = p * torch.exp(logits)
    q = _normalize(q)
    if float(smoothing) > 0.0:
        q = _normalize((1.0 - float(smoothing)) * q + float(smoothing) * p)
    return q, advantages


def _scores_to_shape(
    scores: torch.Tensor,
    p_init: torch.Tensor,
    cfg: TargetSamplerConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = str(cfg.score_to_shape).strip().lower()
    if mode in {"rank_softmax", "rank"}:
        q = rank_softmax_target(
            scores,
            p_init,
            eta=cfg.rank_eta,
            smoothing=cfg.smoothing,
        )
        return q, torch.zeros_like(torch.as_tensor(scores, dtype=torch.float32))
    if mode in {"group_exp", "group_advantage", "boltzmann_group"}:
        return group_exp_target(
            scores,
            p_init,
            eta=cfg.rank_eta,
            smoothing=cfg.smoothing,
            advantage_eps=cfg.advantage_eps,
            advantage_clip=cfg.advantage_clip,
        )
    raise ValueError(f"unknown score_to_shape mode: {cfg.score_to_shape}")


class OneStepTargetSampler:
    name = "one_step"
    sampler_id = 1

    def __init__(
        self,
        action_space: ActionSpace,
        *,
        energy_cfg: ActionEnergyConfig | None = None,
        cfg: TargetSamplerConfig | None = None,
    ):
        self.space = action_space
        self.energy = ActionEnergy(action_space, energy_cfg)
        self.cfg = cfg or TargetSamplerConfig()

    def build_target(
        self,
        *,
        state: RegisterState,
        action_ids: torch.Tensor,
        p_init: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        rng: random.Random,
    ) -> TargetShape:
        del rng
        ids = torch.as_tensor(action_ids, dtype=torch.long, device=x.device)
        B = torch.nan_to_num(evaluate_register_state(state, x))
        scores = torch.zeros(ids.numel(), dtype=B.dtype, device=B.device)
        stop_mask = ids == STOP_ACTION_ID
        if bool(stop_mask.any().item()):
            scores[stop_mask] = -self.energy.residual_energy(B, y)
        normal_mask = ids != STOP_ACTION_ID
        if bool(normal_mask.any().item()):
            scores[normal_mask] = self.energy.rewards(B, y, ids[normal_mask])
        q_hat, advantages = _scores_to_shape(
            scores.detach().cpu(),
            p_init.detach().cpu(),
            self.cfg,
        )
        return TargetShape(
            action_ids=ids.detach().cpu(),
            q_hat=q_hat,
            target_scores=scores.detach().cpu().float(),
            target_counts=torch.ones(ids.numel(), dtype=torch.float32),
            diagnostics={
                "target_sampler_name": self.name,
                "target_sampler_id": self.sampler_id,
                "score_to_shape": self.cfg.score_to_shape,
                "advantage_min": float(advantages.min().item()) if advantages.numel() else 0.0,
                "advantage_max": float(advantages.max().item()) if advantages.numel() else 0.0,
            },
        )


class OneStepGroupAdvantageTargetSampler(OneStepTargetSampler):
    name = "one_step_group_advantage"
    sampler_id = 6

    def __init__(
        self,
        action_space: ActionSpace,
        *,
        energy_cfg: ActionEnergyConfig | None = None,
        cfg: TargetSamplerConfig | None = None,
    ):
        cfg = cfg or TargetSamplerConfig()
        if str(cfg.score_to_shape).strip().lower() in {"rank_softmax", "rank"}:
            cfg = replace(cfg, score_to_shape="group_exp")
        super().__init__(action_space, energy_cfg=energy_cfg, cfg=cfg)


class FutureGroupTargetSampler:
    name = "future_group_l3"
    sampler_id = 2

    def __init__(
        self,
        action_space: ActionSpace,
        *,
        energy_cfg: ActionEnergyConfig | None = None,
        cfg: FutureGroupTargetConfig | None = None,
    ):
        self.space = action_space
        self.energy_cfg = energy_cfg or ActionEnergyConfig(lambda_op=0.0)
        self.energy = ActionEnergy(action_space, self.energy_cfg)
        self.executor = ActionExecutor(action_space)
        self.cfg = cfg or FutureGroupTargetConfig()

    def build_target(
        self,
        *,
        state: RegisterState,
        action_ids: torch.Tensor,
        p_init: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        rng: random.Random,
    ) -> TargetShape:
        ids = torch.as_tensor(action_ids, dtype=torch.long, device=x.device)
        B0 = torch.nan_to_num(evaluate_register_state(state, x))
        e0 = self.energy.residual_energy(B0, y)
        scores = torch.zeros(ids.numel(), dtype=B0.dtype, device=B0.device)
        counts = torch.zeros(ids.numel(), dtype=torch.float32)
        for pos, action_id_t in enumerate(ids.detach().cpu().tolist()):
            action_id = int(action_id_t)
            if is_stop_action(action_id):
                scores[pos] = -e0
                counts[pos] = 1.0
                continue
            first_state = self.executor.execute_symbolic(state, action_id)
            rewards = []
            for _ in range(max(int(self.cfg.rollouts_per_action), 1)):
                terminal_state, extra_actions = self._sample_short_continuation(first_state, x, y, rng)
                B_final = torch.nan_to_num(evaluate_register_state(terminal_state, x))
                ef = self.energy.residual_energy(B_final, y)
                complexity_cost = _action_op_cost(self.space, [action_id] + extra_actions)
                reward = e0 - ef - float(self.cfg.terminal_op_penalty) * complexity_cost
                rewards.append(reward)
            reward_t = torch.stack(rewards)
            k = min(max(int(self.cfg.topk), 1), reward_t.numel())
            scores[pos] = reward_t.topk(k).values.mean()
            counts[pos] = float(reward_t.numel())

        q_hat, advantages = _scores_to_shape(
            scores.detach().cpu(),
            p_init.detach().cpu(),
            self.cfg,
        )
        return TargetShape(
            action_ids=ids.detach().cpu(),
            q_hat=q_hat,
            target_scores=scores.detach().cpu().float(),
            target_counts=counts,
            diagnostics={
                "target_sampler_name": self.name,
                "target_sampler_id": self.sampler_id,
                "rollout_depth": int(self.cfg.rollout_depth),
                "rollouts_per_action": int(self.cfg.rollouts_per_action),
                "score_to_shape": self.cfg.score_to_shape,
                "advantage_min": float(advantages.min().item()) if advantages.numel() else 0.0,
                "advantage_max": float(advantages.max().item()) if advantages.numel() else 0.0,
            },
        )

    def _sample_short_continuation(
        self,
        state: RegisterState,
        x: torch.Tensor,
        y: torch.Tensor,
        rng: random.Random,
    ) -> tuple[RegisterState, list[int]]:
        current = state.clone()
        actions: list[int] = []
        for _ in range(max(int(self.cfg.rollout_depth), 0)):
            raw_ids = self.space.valid_actions(current).to(device=x.device)
            raw_ids = _cap_support(raw_ids, self.cfg.max_rollout_support)
            if raw_ids.numel() == 0:
                break
            B = torch.nan_to_num(evaluate_register_state(current, x))
            healthy = healthy_action_ids(
                self.energy,
                B,
                y,
                raw_ids,
                max_abs_semantic=1e6,
                max_energy_growth=100.0,
            )
            support = append_stop_action(healthy, enabled=True)
            if support.numel() == 0:
                break
            idx = rng.randrange(int(support.numel()))
            action_id = int(support[idx].item())
            if is_stop_action(action_id):
                break
            actions.append(action_id)
            current = self.executor.execute_symbolic(current, action_id)
        return current, actions


class MultiStepGroupAdvantageTargetSampler(FutureGroupTargetSampler):
    name = "multi_step_group_advantage"
    sampler_id = 2

    def __init__(
        self,
        action_space: ActionSpace,
        *,
        energy_cfg: ActionEnergyConfig | None = None,
        cfg: FutureGroupTargetConfig | None = None,
    ):
        cfg = cfg or FutureGroupTargetConfig()
        if str(cfg.score_to_shape).strip().lower() in {"rank_softmax", "rank"}:
            cfg = replace(cfg, score_to_shape="group_exp")
        super().__init__(action_space, energy_cfg=energy_cfg, cfg=cfg)


class CachedTrajectoryFitnessTargetSampler:
    name = "cached_trajectory_fitness"
    sampler_id = 3

    def __init__(
        self,
        action_space: ActionSpace,
        *,
        energy_cfg: ActionEnergyConfig | None = None,
        cfg: FutureGroupTargetConfig | None = None,
    ):
        del energy_cfg
        self.space = action_space
        self.cfg = cfg or FutureGroupTargetConfig()
        self.records = _read_records(self.cfg.cache_path)

    def build_target(
        self,
        *,
        state: RegisterState,
        action_ids: torch.Tensor,
        p_init: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        rng: random.Random,
    ) -> TargetShape:
        del state, x, y
        ids = torch.as_tensor(action_ids, dtype=torch.long)
        shape = _records_to_probability_shape(
            self.records,
            ids,
            p_init,
            rng=rng,
            cfg=self.cfg,
            include_gp_likelihood=False,
        )
        return TargetShape(
            action_ids=ids.detach().cpu(),
            q_hat=shape["q_hat"],
            target_scores=shape["scores"],
            target_counts=shape["counts"],
            diagnostics={
                "target_sampler_name": self.name,
                "target_sampler_id": self.sampler_id,
                "num_cached_records": len(self.records),
                "samples_probability_shape": True,
                "fallback_to_p_init": bool(shape["fallback"]),
            },
        )


class GPCandidateFitnessTargetSampler:
    name = "gp_candidate_fitness"
    sampler_id = 4

    def __init__(
        self,
        action_space: ActionSpace,
        *,
        energy_cfg: ActionEnergyConfig | None = None,
        cfg: FutureGroupTargetConfig | None = None,
    ):
        del energy_cfg
        self.space = action_space
        self.cfg = cfg or FutureGroupTargetConfig()
        self.records = _read_records(self.cfg.gp_population_path)

    def build_target(
        self,
        *,
        state: RegisterState,
        action_ids: torch.Tensor,
        p_init: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        rng: random.Random,
    ) -> TargetShape:
        del state, x, y
        ids = torch.as_tensor(action_ids, dtype=torch.long)
        shape = _records_to_probability_shape(
            self.records,
            ids,
            p_init,
            rng=rng,
            cfg=self.cfg,
            include_gp_likelihood=True,
        )
        return TargetShape(
            action_ids=ids.detach().cpu(),
            q_hat=shape["q_hat"],
            target_scores=shape["scores"],
            target_counts=shape["counts"],
            diagnostics={
                "target_sampler_name": self.name,
                "target_sampler_id": self.sampler_id,
                "num_gp_records": len(self.records),
                "samples_probability_shape": True,
                "fallback_to_p_init": bool(shape["fallback"]),
            },
        )


class ShapeSamplingTargetSampler:
    sampler_id = 5

    def __init__(
        self,
        action_space: ActionSpace,
        *,
        energy_cfg: ActionEnergyConfig | None = None,
        cfg: FutureGroupTargetConfig | None = None,
        mode: str = "importance_sampling",
    ):
        self.space = action_space
        self.energy = ActionEnergy(action_space, energy_cfg)
        self.cfg = cfg or FutureGroupTargetConfig()
        self.name = str(mode).strip().lower().replace("-", "_")
        if self.name not in {"importance_sampling", "mcmc_shape"}:
            raise ValueError(f"unknown shape-sampling mode: {mode}")

    def build_target(
        self,
        *,
        state: RegisterState,
        action_ids: torch.Tensor,
        p_init: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        rng: random.Random,
    ) -> TargetShape:
        ids = torch.as_tensor(action_ids, dtype=torch.long, device=x.device)
        scores = self._one_step_scores(state, ids, x, y)
        log_target = _score_log_target(scores.detach().cpu(), eta=self.cfg.rank_eta)
        p = _normalize(torch.as_tensor(p_init, dtype=torch.float32).detach().cpu())
        if self.name == "importance_sampling":
            counts = _importance_shape_counts(
                log_target,
                p,
                rng,
                n_samples=self.cfg.importance_samples or self.cfg.shape_samples,
            )
        else:
            counts = _mcmc_shape_counts(
                log_target,
                p,
                rng,
                n_samples=self.cfg.shape_samples,
                burn_in=self.cfg.mcmc_burn_in,
            )
        q_hat = _normalize(counts + float(self.cfg.smoothing) * p)
        return TargetShape(
            action_ids=ids.detach().cpu(),
            q_hat=q_hat,
            target_scores=scores.detach().cpu().float(),
            target_counts=counts.float(),
            diagnostics={
                "target_sampler_name": self.name,
                "target_sampler_id": self.sampler_id,
                "shape_samples": int(self.cfg.shape_samples),
                "samples_probability_shape": True,
            },
        )

    def _one_step_scores(
        self,
        state: RegisterState,
        action_ids: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        B = torch.nan_to_num(evaluate_register_state(state, x))
        scores = torch.zeros(action_ids.numel(), dtype=B.dtype, device=B.device)
        normal_mask = action_ids != STOP_ACTION_ID
        if bool(normal_mask.any().item()):
            scores[normal_mask] = self.energy.rewards(B, y, action_ids[normal_mask])
        return scores


def make_target_sampler(
    mode: str,
    action_space: ActionSpace,
    *,
    energy_cfg: ActionEnergyConfig | None,
    future_cfg: FutureGroupTargetConfig,
) -> (
    OneStepTargetSampler
    | FutureGroupTargetSampler
    | MultiStepGroupAdvantageTargetSampler
    | CachedTrajectoryFitnessTargetSampler
    | GPCandidateFitnessTargetSampler
):
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"one_step_group_advantage", "one_step_group_exp", "archive_one_step"}:
        return OneStepGroupAdvantageTargetSampler(action_space, energy_cfg=energy_cfg, cfg=future_cfg)
    if normalized in {"multi_step_group_advantage", "future_group_advantage", "future_group_l3_group_advantage"}:
        return MultiStepGroupAdvantageTargetSampler(action_space, energy_cfg=energy_cfg, cfg=future_cfg)
    if normalized in {"cached_trajectory_fitness", "cached_fitness", "trajectory_cache_fitness"}:
        return CachedTrajectoryFitnessTargetSampler(action_space, energy_cfg=energy_cfg, cfg=future_cfg)
    if normalized in {"gp_candidate_fitness", "gp_generated_candidate_fitness", "gp_population_fitness"}:
        return GPCandidateFitnessTargetSampler(action_space, energy_cfg=energy_cfg, cfg=future_cfg)
    raise ValueError(f"unknown target sampler mode: {mode}")


def _cap_support(action_ids: torch.Tensor, max_support_size: int | None) -> torch.Tensor:
    if max_support_size is None:
        return action_ids
    budget = max(int(max_support_size), 0)
    if action_ids.numel() <= budget:
        return action_ids
    if budget == 0:
        return action_ids[:0]
    sorted_ids = action_ids.sort().values
    idx = torch.linspace(
        0,
        sorted_ids.numel() - 1,
        steps=budget,
        device=sorted_ids.device,
    ).round().long()
    return sorted_ids[idx].unique(sorted=True)


def _action_op_cost(action_space: ActionSpace, actions: list[int]) -> float:
    total = 0.0
    for action_id in actions:
        spec = action_space.decode(int(action_id))
        total += float(op_cost(spec.op_id))
    return total


def _normalize(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    values = torch.nan_to_num(values.float()).clamp(min=0.0)
    total = values.sum()
    if float(total.item()) <= eps:
        return torch.ones_like(values) / max(values.numel(), 1)
    return values / total.clamp(min=eps)


def _read_records(path: str | None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text().strip()
    if not text:
        return []
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    raw = json.loads(text)
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    if isinstance(raw, dict):
        for key in ("population", "trajectories", "records", "events"):
            value = raw.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
    return []


def _records_to_probability_shape(
    records: list[dict],
    action_ids: torch.Tensor,
    p_init: torch.Tensor,
    *,
    rng: random.Random,
    cfg: FutureGroupTargetConfig,
    include_gp_likelihood: bool,
) -> dict:
    ids = torch.as_tensor(action_ids, dtype=torch.long).detach().cpu()
    p = _normalize(torch.as_tensor(p_init, dtype=torch.float32).detach().cpu())
    index = {int(action_id): pos for pos, action_id in enumerate(ids.tolist())}
    items: list[tuple[int, float, float]] = []
    for record in records:
        first = _first_supported_action(record, index)
        if first is None:
            continue
        fitness = _record_fitness(record)
        log_weight = float(cfg.gp_fitness_weight) * fitness
        if include_gp_likelihood:
            log_weight += float(cfg.gp_likelihood_weight) * _record_gp_logprob(record)
        items.append((first, fitness, log_weight))

    scores = torch.zeros(ids.numel(), dtype=torch.float32)
    counts = torch.zeros(ids.numel(), dtype=torch.float32)
    if not items:
        return {
            "q_hat": p,
            "scores": scores,
            "counts": counts,
            "fallback": True,
        }

    logits = torch.tensor([item[2] for item in items], dtype=torch.float32)
    probs = torch.softmax(torch.nan_to_num(logits, nan=-50.0, posinf=50.0, neginf=-50.0), dim=0)
    n_samples = max(int(cfg.shape_samples), 1)
    for _ in range(n_samples):
        item_idx = _sample_index(probs, rng)
        action_pos, fitness, _ = items[item_idx]
        counts[action_pos] += 1.0
        scores[action_pos] = max(scores[action_pos], float(fitness))
    q_hat = _normalize(counts + float(cfg.smoothing) * p)
    return {
        "q_hat": q_hat,
        "scores": scores,
        "counts": counts,
        "fallback": False,
    }


def _first_supported_action(record: dict, index: dict[int, int]) -> int | None:
    actions = record.get("actions", record.get("trajectory", []))
    if isinstance(actions, int):
        actions = [actions]
    if not isinstance(actions, list):
        return None
    for action in actions:
        try:
            action_id = int(action)
        except (TypeError, ValueError):
            continue
        if action_id in index:
            return index[action_id]
    return None


def _record_fitness(record: dict) -> float:
    for key in ("fitness", "reward", "score", "lineage_return", "r2"):
        if key in record and record[key] is not None:
            return float(record[key])
    return 0.0


def _record_gp_logprob(record: dict) -> float:
    for key in ("gp_logprob", "logprob", "logprob_base"):
        if key in record and record[key] is not None:
            return float(record[key])
    if "event_log" in record:
        return float(compute_gp_individual_logprob(record["event_log"]).item())
    if "events" in record:
        return float(compute_gp_individual_logprob(record["events"]).item())
    return 0.0


def _score_log_target(scores: torch.Tensor, *, eta: float) -> torch.Tensor:
    scores = torch.nan_to_num(torch.as_tensor(scores, dtype=torch.float32))
    if scores.numel() <= 1:
        return torch.zeros_like(scores)
    order = scores.argsort(descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(0, order, torch.arange(scores.numel(), dtype=order.dtype, device=scores.device))
    rank_norm = 1.0 - ranks.to(torch.float32) / float(scores.numel() - 1)
    return float(eta) * rank_norm


def _importance_shape_counts(
    log_target: torch.Tensor,
    proposal: torch.Tensor,
    rng: random.Random,
    *,
    n_samples: int,
) -> torch.Tensor:
    proposal = _normalize(proposal)
    target = torch.softmax(log_target, dim=0)
    counts = torch.zeros_like(proposal)
    for _ in range(max(int(n_samples), 1)):
        idx = _sample_index(proposal, rng)
        weight = float(target[idx].item()) / max(float(proposal[idx].item()), 1e-12)
        counts[idx] += float(weight)
    return counts


def _mcmc_shape_counts(
    log_target: torch.Tensor,
    proposal_start: torch.Tensor,
    rng: random.Random,
    *,
    n_samples: int,
    burn_in: int,
) -> torch.Tensor:
    counts = torch.zeros_like(log_target, dtype=torch.float32)
    cur = _sample_index(_normalize(proposal_start), rng)
    total_steps = max(int(n_samples), 1) + max(int(burn_in), 0)
    for step in range(total_steps):
        cand = rng.randrange(int(log_target.numel()))
        accept_logprob = float(log_target[cand].item() - log_target[cur].item())
        if math.log(max(rng.random(), 1e-12)) <= min(accept_logprob, 0.0):
            cur = cand
        if step >= int(burn_in):
            counts[cur] += 1.0
    return counts


def _sample_index(probs: torch.Tensor, rng: random.Random) -> int:
    p = _normalize(probs).detach().cpu()
    r = rng.random()
    total = 0.0
    for idx, value in enumerate(p.tolist()):
        total += float(value)
        if r <= total:
            return int(idx)
    return max(int(p.numel()) - 1, 0)
