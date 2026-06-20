"""Global terminal-reward trajectory marginal target endpoint."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import TargetEndpoint
from ..actions.action_space import ActionSpace
from ..semantics.energy import ActionEnergyConfig
from ..trajectories.evaluator import GlobalTrajectoryEvaluator
from ..trajectories.cache import TrajectoryTargetCache
from ..trajectories.sampler import GrammarTrajectorySampler, TrajectorySampler
from ..trajectories.target_marginals import build_reward_weighted_marginals
from ..gp_distill.trajectory_pool import load_gp_trajectory_population
from ..utils.numerical import EPS, normalize_simplex


@dataclass
class GlobalTrajectoryTarget(TargetEndpoint):
    """Route-B target: terminal trajectory rewards projected to action-table marginals."""

    action_space: ActionSpace
    energy_cfg: ActionEnergyConfig | None = None
    sampler: TrajectorySampler | None = None
    num_samples: int = 64
    max_len: int = 5
    eta: float = 1.0
    eta_adv: float | None = None
    smoothing: float = 1e-2
    complexity_penalty: float | None = None
    cache_path: str | None = None
    cache_write: bool = True
    prior_mode: str = "uniform"
    prior_smoothing: float = 1e-2
    likelihood_kappa: float = 0.0
    likelihood_clip: float | None = None
    gp_population_path: str | None = None
    gp_sample_mode: str = "base_plus_gp"

    def __post_init__(self):
        self.energy_cfg = self.energy_cfg or ActionEnergyConfig()
        if self.eta_adv is not None:
            self.eta = float(self.eta_adv)
        self.sampler = self.sampler or GrammarTrajectorySampler(self.action_space, seed=0)
        self.evaluator = GlobalTrajectoryEvaluator(
            self.action_space,
            self.energy_cfg,
            complexity_penalty=self.complexity_penalty,
        )

    def build_p1(self, B, y, action_ids, energies, p0, context):
        state = context.get("state")
        x = context.get("x")
        if state is None or x is None:
            raise ValueError("GlobalTrajectoryTarget requires context['state'] and context['x']")
        trajectories = self._sample_trajectories(state, context)
        if not trajectories:
            adv = torch.zeros_like(p0)
            context["advantages"] = adv
            context["rewards"] = torch.zeros_like(p0)
            context["trajectory_stats"] = {"num_trajectories": 0, "target_mode": "global_trajectory_marginal"}
            return p0
        eval_out = self.evaluator.evaluate(trajectories, x, y, initial_state=state)
        self._maybe_cache(context, trajectories, eval_out)
        logprobs = _trajectory_logprobs(trajectories, device=B.device, dtype=B.dtype)
        base_p0 = None
        if self.prior_mode in {"gp_frequency", "trajectory_frequency"}:
            base_p0 = _trajectory_frequency_prior(
                trajectories,
                action_vocab_size=self.action_space.size,
                max_len=max(int(self.max_len), 1),
                device=B.device,
                dtype=B.dtype,
                smoothing=float(self.prior_smoothing),
            )
        elif self.prior_mode != "uniform":
            raise ValueError(f"unknown global trajectory prior_mode: {self.prior_mode}")
        marginals = build_reward_weighted_marginals(
            trajectories,
            eval_out.rewards.to(device=B.device, dtype=B.dtype),
            action_vocab_size=self.action_space.size,
            max_len=max(int(self.max_len), 1),
            eta=float(self.eta),
            smoothing=float(self.smoothing),
            base_p0=base_p0,
            logprobs=logprobs,
            likelihood_kappa=float(self.likelihood_kappa),
            likelihood_clip=self.likelihood_clip,
        )
        action_ids = action_ids.to(device=B.device, dtype=torch.long)
        rho0 = marginals.rho[0].to(device=B.device, dtype=B.dtype)
        p0_table = marginals.p0[0].to(device=B.device, dtype=B.dtype)
        reward_mean0 = marginals.reward_mean[0].to(device=B.device, dtype=B.dtype)
        rho_support = rho0[action_ids].clamp(min=EPS)
        p0_support = normalize_simplex(p0_table[action_ids].clamp(min=EPS), dim=-1)
        if self.prior_mode in {"gp_frequency", "trajectory_frequency"}:
            context["p_start_override"] = p0_support
        reward_support = reward_mean0[action_ids]
        valid_observed = marginals.mask[0].to(device=B.device)[action_ids]
        # Keep all current legal actions alive even if the finite trajectory sample did not
        # observe them. Their target mass is the smoothing floor inherited from p0.
        p_target = normalize_simplex(rho_support, dim=-1)
        p0_effective = context.get("p_start_override", p0)
        raw_adv = p_target.clamp(min=EPS).log() - p0_effective.clamp(min=EPS).log()
        raw_adv = _group_normalize(raw_adv)
        context.setdefault("one_step_rewards", context.get("rewards", torch.zeros_like(p0)))
        context["advantages"] = raw_adv
        context["rewards"] = reward_support
        context["global_trajectory_rewards"] = reward_support
        context["rollout_rewards"] = reward_support
        context["trajectory_stats"] = {
            "num_trajectories": len(trajectories),
            "target_mode": "global_trajectory_marginal",
            "max_len": int(self.max_len),
            "eta": float(self.eta),
            "smoothing": float(self.smoothing),
            "observed_action_fraction": float(valid_observed.float().mean().detach().cpu().item())
            if valid_observed.numel() else 0.0,
            "candidate_oracle_r2": float(eval_out.final_r2.max().detach().cpu().item())
            if eval_out.final_r2.numel() else 0.0,
            "candidate_oracle_reward": float(eval_out.rewards.max().detach().cpu().item())
            if eval_out.rewards.numel() else 0.0,
            "prior_mode": str(self.prior_mode),
            "gp_likelihood_weighted": bool(logprobs is not None and float(self.likelihood_kappa) != 0.0),
            "gp_likelihood_ess": _ess(marginals.sample_weights),
            "gp_population_loaded": int(context.get("_gp_population_loaded", 0)),
        }
        return p_target

    def _sample_trajectories(self, state, context: dict):
        gp_trajs = []
        if self.gp_population_path:
            gp_trajs = load_gp_trajectory_population(
                self.gp_population_path,
                self.action_space,
                state,
                max_len=int(self.max_len),
            )
            context["_gp_population_loaded"] = len(gp_trajs)
        mode = str(self.gp_sample_mode)
        if mode == "gp_only":
            return gp_trajs[: int(self.num_samples)]
        if mode != "base_plus_gp":
            raise ValueError(f"unknown gp_sample_mode: {self.gp_sample_mode}")
        base_budget = max(int(self.num_samples) - len(gp_trajs), 0)
        base = self.sampler.sample(
            state,
            num_samples=base_budget,
            max_len=int(self.max_len),
            policy=context.get("policy"),
        )
        return [*gp_trajs[: int(self.num_samples)], *base][: int(self.num_samples)]

    def _maybe_cache(self, context: dict, trajectories, eval_out) -> None:
        if not self.cache_path or not self.cache_write:
            return
        cache = TrajectoryTargetCache(self.cache_path)
        task_id = str(context.get("task_id", context.get("sample_index", "")))
        for idx, trajectory in enumerate(trajectories):
            cache.append({
                "task_id": task_id,
                "trajectory_id": f"{task_id}:{idx}" if task_id else str(idx),
                "actions": [int(a) for a in trajectory.actions],
                "masks": [
                    [int(i) for i in mask.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()]
                    for mask in trajectory.masks
                ],
                "reward": float(eval_out.rewards[idx].detach().cpu().item()),
                "r2": float(eval_out.final_r2[idx].detach().cpu().item()),
                "complexity": float(eval_out.complexities[idx].detach().cpu().item()),
                "source": str(trajectory.metadata.get("source", "base")),
                "gp_logprob": trajectory.metadata.get("gp_logprob"),
                "final_expression": str(eval_out.expressions[idx]),
            })


def _group_normalize(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=-1, keepdim=True)
    std = centered.std(dim=-1, keepdim=True, unbiased=False)
    return torch.nan_to_num(centered / std.clamp(min=EPS))


def _trajectory_logprobs(trajectories, device, dtype) -> torch.Tensor | None:
    vals = []
    for trajectory in trajectories:
        value = trajectory.metadata.get("gp_logprob")
        if value is None:
            return None
        vals.append(float(value))
    return torch.tensor(vals, device=device, dtype=dtype) if vals else None


def _trajectory_frequency_prior(
    trajectories,
    *,
    action_vocab_size: int,
    max_len: int,
    device,
    dtype,
    smoothing: float,
) -> torch.Tensor:
    T = max(int(max_len), 1)
    A = int(action_vocab_size)
    mask = torch.zeros(T, A, device=device, dtype=torch.bool)
    counts = torch.zeros(T, A, device=device, dtype=dtype)
    for trajectory in trajectories:
        for t, action in enumerate(trajectory.actions[:T]):
            action_id = int(action)
            if t < len(trajectory.masks):
                mask[t] |= trajectory.masks[t].to(device=device, dtype=torch.bool)
            else:
                mask[t, action_id] = True
            counts[t, action_id] += 1.0
    uniform = normalize_simplex(mask.to(dtype=dtype), dim=-1)
    empirical = normalize_simplex(counts, dim=-1)
    return normalize_simplex((1.0 - float(smoothing)) * empirical + float(smoothing) * uniform, dim=-1)


def _ess(weights: torch.Tensor) -> float:
    if weights.numel() == 0:
        return 0.0
    w = weights.detach().float()
    return float((w.sum().square() / w.square().sum().clamp(min=EPS)).detach().cpu().item())
