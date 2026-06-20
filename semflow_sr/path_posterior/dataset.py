"""Dataset builder for Semantic-Fisher Flow Matching."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import random
import time

import torch
from torch.utils.data import Dataset

from ..actions.action_space import ActionSpace
from ..data.synthetic_generator import GenConfig, generate_trace_task
from ..flow.semantic_fisher import (
    integrate_semantic_fisher_endpoint_path,
    semantic_fisher_simplex_velocity,
    semantic_fisher_sphere_step,
)
from ..registers.executor import evaluate_register_state
from ..registers.state import init_register_state
from ..semantics.energy import ActionEnergy, ActionEnergyConfig
from ..sr.ops import NAME_TO_ID
from .sampler import ActionPathSampler
from .action_support import (
    STOP_ACTION_ID,
    action_features_with_stop,
    action_semantic_effects_with_stop,
)
from .target import (
    PathDecision,
)
from .target_sampler import (
    FutureGroupTargetConfig,
    PriorConfig,
    TargetShape,
    build_p_init,
    make_target_sampler,
)


@dataclass
class PathPosteriorBuildConfig:
    target_mode: str = "future_group_l3"
    num_trajectories: int = 16
    max_states_per_task: int | None = 32
    max_steps: int = 6
    weight_eta: float = 2.0
    target_smoothing: float = 1e-3
    stop_bias_base: float = -2.0
    stop_bias_slope: float = 0.35
    rollout_depth: int = 3
    rollouts_per_action: int = 1
    rollout_topk: int = 1
    max_rollout_support: int | None = 16
    beta: float = 1.0
    gamma: float = 0.1
    gram_rank: int | None = 8
    teacher_steps: int = 2
    enable_stop: bool = True
    max_abs_semantic: float | None = 1e6
    max_energy_growth: float | None = 100.0
    max_support_size: int | None = 64
    terminal_op_penalty: float | None = None
    cache_path: str | None = None
    gp_population_path: str | None = None
    shape_samples: int = 32
    gp_likelihood_weight: float = 1.0
    gp_fitness_weight: float = 1.0
    importance_samples: int | None = None
    mcmc_burn_in: int = 16
    behavior_policy_id: str | None = None


class PathPosteriorDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def build_path_posterior_dataset(
    gen: GenConfig,
    *,
    num_tasks: int,
    behavior_model,
    seed: int,
    energy_cfg: ActionEnergyConfig | None = None,
    cfg: PathPosteriorBuildConfig | None = None,
) -> PathPosteriorDataset:
    cfg = cfg or PathPosteriorBuildConfig()
    energy_cfg = energy_cfg or ActionEnergyConfig(lambda_op=0.0)
    rng = random.Random(seed)
    allowed = [NAME_TO_ID[o] for o in gen.ops]
    space = ActionSpace(gen.K, allowed)
    energy = ActionEnergy(space, energy_cfg)
    target_sampler = make_target_sampler(
        cfg.target_mode,
        space,
        energy_cfg=energy_cfg,
        future_cfg=FutureGroupTargetConfig(
            rank_eta=cfg.weight_eta,
            smoothing=cfg.target_smoothing,
            rollout_depth=cfg.rollout_depth,
            rollouts_per_action=cfg.rollouts_per_action,
            topk=cfg.rollout_topk,
            max_rollout_support=cfg.max_rollout_support,
            terminal_op_penalty=0.0 if cfg.terminal_op_penalty is None else float(cfg.terminal_op_penalty),
            cache_path=cfg.cache_path,
            gp_population_path=cfg.gp_population_path,
            shape_samples=cfg.shape_samples,
            gp_likelihood_weight=cfg.gp_likelihood_weight,
            gp_fitness_weight=cfg.gp_fitness_weight,
            importance_samples=cfg.importance_samples,
            mcmc_burn_in=cfg.mcmc_burn_in,
        ),
    )
    sampler = ActionPathSampler(
        space,
        energy_cfg=energy_cfg,
        behavior_policy_id=cfg.behavior_policy_id or f"path_posterior_seed_{seed}",
        seed=seed,
        enable_stop=cfg.enable_stop,
        max_abs_semantic=cfg.max_abs_semantic,
        max_energy_growth=cfg.max_energy_growth,
        max_support_size=cfg.max_support_size,
    )
    records: list[dict] = []
    task_idx = 0
    tries = 0
    while task_idx < int(num_tasks) and tries < int(num_tasks) * 20:
        tries += 1
        task = generate_trace_task(gen, rng)
        if task is None:
            continue
        _, _, x, y = task
        initial_state = init_register_state(gen.num_vars, gen.K, device=x.device)
        trajectories = sampler.sample(
            task_id=f"synthetic:{task_idx}",
            initial_state=initial_state,
            x=x,
            y=y,
            model=behavior_model,
            num_trajectories=cfg.num_trajectories,
            max_steps=cfg.max_steps,
        )
        if not trajectories:
            continue
        prefix_decisions = _collect_prefix_decisions(trajectories, cfg.max_states_per_task)
        for decision in prefix_decisions:
            local = _build_target_shape_local(
                target_sampler,
                decision=decision,
                x=x,
                y=y,
                cfg=cfg,
                rng=rng,
            )
            if local is None:
                continue
            _append_teacher_records(
                records,
                space=space,
                energy=energy,
                x=x,
                y=y,
                cfg=cfg,
                state=local["state"],
                action_ids=local["action_ids"],
                p_init=local["p_init"],
                q_hat=local["q_hat"],
                extra=local["extra"],
            )
        task_idx += 1
    if not records:
        raise RuntimeError("failed to build any path-posterior records")
    return PathPosteriorDataset(records)


def snapshot_behavior_model(model):
    behavior = copy.deepcopy(model).to("cpu")
    behavior.eval()
    return behavior


def _append_teacher_records(
    records: list[dict],
    *,
    space: ActionSpace,
    energy: ActionEnergy,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: PathPosteriorBuildConfig,
    state,
    action_ids: torch.Tensor,
    p_init: torch.Tensor,
    q_hat: torch.Tensor,
    extra: dict[str, torch.Tensor] | None = None,
) -> None:
    B = torch.nan_to_num(evaluate_register_state(state, x))
    action_ids = torch.as_tensor(action_ids, dtype=torch.long, device=B.device)
    if action_ids.numel() == 0:
        return
    effect = action_semantic_effects_with_stop(energy, B, y, action_ids)
    p_init = torch.as_tensor(p_init, dtype=B.dtype, device=B.device)
    q_hat = torch.as_tensor(q_hat, dtype=B.dtype, device=B.device)
    teacher_path = integrate_semantic_fisher_endpoint_path(
        p_init,
        q_hat,
        effect.gram,
        beta=cfg.beta,
        gamma=cfg.gamma,
        steps=cfg.teacher_steps,
        gram_rank=cfg.gram_rank,
        gram_factors=effect.xi,
        q_smoothing=cfg.target_smoothing,
    )
    feats = action_features_with_stop(space, state, action_ids)
    extra = extra or {}
    for path_idx, (p_lambda, w_target, zdot) in enumerate(zip(
        teacher_path.policies[:-1],
        teacher_path.logrates,
        teacher_path.sphere_velocities,
    )):
        p_lambda = p_lambda.to(device=B.device, dtype=B.dtype)
        w_target = w_target.to(device=B.device, dtype=B.dtype)
        zdot = zdot.to(device=B.device, dtype=B.dtype)
        z_lambda = p_lambda.clamp(min=1e-12).sqrt()
        pdot = semantic_fisher_simplex_velocity(p_lambda, w_target)
        p_target = semantic_fisher_sphere_step(p_lambda, w_target, dt=1.0)
        advantages = torch.nan_to_num(
            q_hat.clamp(min=1e-12).log() - p_lambda.clamp(min=1e-12).log()
        )
        record = {
            "x": x.float(),
            "y": y.float(),
            "B": B.float(),
            "action_ids": action_ids.detach().cpu(),
            "action_feats": feats.float().cpu(),
            "semantic_stats": torch.zeros(action_ids.numel(), 8, dtype=torch.float32),
            "energies": torch.zeros(action_ids.numel(), dtype=torch.float32),
            "rewards": q_hat.float().detach().cpu(),
            "scores": q_hat.float().detach().cpu(),
            "advantages": advantages.float().detach().cpu(),
            "target_advantages": advantages.float().detach().cpu(),
            "proposal_probs": p_init.float().detach().cpu(),
            "one_step_rewards": torch.zeros(action_ids.numel(), dtype=torch.float32),
            "rollout_rewards": q_hat.float().detach().cpu(),
            "weights": torch.ones(action_ids.numel(), dtype=torch.float32),
            "residual_current": effect.residual_current.float().detach().cpu(),
            "residual_next": effect.residual_next.float().detach().cpu(),
            "xi": effect.xi.float().detach().cpu(),
            "gram": effect.gram.float().detach().cpu(),
            "gamma": torch.tensor(float(cfg.gamma), dtype=torch.float32),
            "w_target": w_target.float().detach().cpu(),
            "pdot_target": pdot.float().detach().cpu(),
            "zdot_target": zdot.float().detach().cpu(),
            "p_start": p_init.float().detach().cpu(),
            "p_target": p_target.float().detach().cpu(),
            "plain_p_target": q_hat.float().detach().cpu(),
            "p0": p_init.float().detach().cpu(),
            "p1": p_target.float().detach().cpu(),
            "p_lambda": p_lambda.float().detach().cpu(),
            "dp_dlambda": pdot.float().detach().cpu(),
            "z_lambda": z_lambda.float().detach().cpu(),
            "dz_dlambda": zdot.float().detach().cpu(),
            "lambda": torch.tensor(float(path_idx) * float(teacher_path.dt), dtype=torch.float32),
            "gt_action_pos": torch.tensor(-1, dtype=torch.long),
            "full_action_size": torch.tensor(action_ids.numel(), dtype=torch.long),
        }
        for key, value in extra.items():
            record[key] = torch.as_tensor(value).float().detach().cpu()
        records.append(record)


def _build_target_shape_local(
    target_sampler,
    *,
    decision: PathDecision,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: PathPosteriorBuildConfig,
    rng: random.Random,
) -> dict | None:
    state = decision.state
    action_ids = torch.as_tensor(decision.action_ids, dtype=torch.long, device=x.device)
    if action_ids.numel() == 0:
        return None
    p_init = build_p_init(
        action_ids.detach().cpu(),
        step=_construction_step(state),
        cfg=PriorConfig(
            stop_bias_base=cfg.stop_bias_base,
            stop_bias_slope=cfg.stop_bias_slope,
        ),
    )
    sampler_start = time.perf_counter()
    target: TargetShape = target_sampler.build_target(
        state=state,
        action_ids=action_ids,
        p_init=p_init.to(device=x.device),
        x=x,
        y=y,
        rng=rng,
    )
    sampler_seconds = time.perf_counter() - sampler_start
    diagnostics = target.diagnostics or {}
    target_sampler_id = int(diagnostics.get("target_sampler_id", 0))
    score_gap = _score_gap(target.target_scores)
    return {
        "state": state,
        "action_ids": target.action_ids,
        "p_init": p_init,
        "q_hat": target.q_hat,
        "extra": {
            "target_scores": target.target_scores,
            "target_counts": target.target_counts,
            "target_sampler_id": torch.tensor(float(target_sampler_id), dtype=torch.float32),
            "target_entropy": _entropy(target.q_hat),
            "p_init_entropy": _entropy(p_init),
            "target_kl_q_pinit": _kl(target.q_hat, p_init),
            "stop_target_mass": _stop_mass(target.action_ids, target.q_hat),
            "target_score_gap": score_gap,
            "target_sampler_runtime_sec": torch.tensor(float(sampler_seconds), dtype=torch.float32),
            "target_support_size": torch.tensor(float(target.action_ids.numel()), dtype=torch.float32),
        },
    }


def _collect_prefix_decisions(
    trajectories,
    max_states: int | None,
) -> list[PathDecision]:
    by_state: dict[str, PathDecision] = {}
    for trajectory in trajectories:
        for decision in trajectory.decisions:
            if decision.state_id not in by_state:
                by_state[decision.state_id] = decision
                if max_states is not None and len(by_state) >= int(max_states):
                    return list(by_state.values())
    return list(by_state.values())


def _construction_step(state) -> int:
    active = int(state.active.bool().sum().detach().cpu().item())
    return max(active - int(state.num_vars) - 1, 0)


def _entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.as_tensor(p, dtype=torch.float32).clamp_min(eps)
    p = p / p.sum().clamp_min(eps)
    return -(p * p.log()).sum()


def _stop_mass(action_ids: torch.Tensor, q_hat: torch.Tensor) -> torch.Tensor:
    ids = torch.as_tensor(action_ids, dtype=torch.long)
    q = torch.as_tensor(q_hat, dtype=torch.float32)
    pos = (ids == STOP_ACTION_ID).nonzero(as_tuple=False)
    if pos.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    return q[int(pos[0].item())].float()


def _kl(q: torch.Tensor, p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q = torch.as_tensor(q, dtype=torch.float32).clamp_min(eps)
    p = torch.as_tensor(p, dtype=torch.float32).clamp_min(eps)
    q = q / q.sum().clamp_min(eps)
    p = p / p.sum().clamp_min(eps)
    return (q * (q.log() - p.log())).sum()


def _score_gap(scores: torch.Tensor) -> torch.Tensor:
    s = torch.as_tensor(scores, dtype=torch.float32)
    if s.numel() < 2:
        return torch.tensor(0.0, dtype=torch.float32)
    top2 = torch.topk(torch.nan_to_num(s), k=2).values
    return (top2[0] - top2[1]).float()
