"""Rollout-fitness advantage provider for compatibility experiments.

The current mainline treats rollout/search as evaluator-based improvement providers:
they estimate future-value scores on the support, normalize them into advantages,
and induce a conservative target through the semantic-Fisher pullback solver.

This file keeps the old endpoint-style build_p1 API so legacy training/evaluation
entry points continue to run.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import torch

from .base import TargetEndpoint
from ..actions.action_space import ActionSpace
from ..actions.action_executor import ActionExecutor
from ..registers.executor import evaluate_register_state
from ..registers.state import RegisterState
from ..semantics.energy import ActionEnergy, ActionEnergyConfig
from ..semantics.projection import ProjectionBackend
from ..sr.ops import NAME_TO_ID, op_cost
from ..utils.numerical import normalize_simplex


@dataclass
class FitnessResult:
    score: float
    final_energy: float
    final_r2: float
    complexity: int
    sequence: list[int]


@dataclass
class FitnessScorer:
    energy_cfg: ActionEnergyConfig
    fitness: str = "normalized_energy_improvement"
    complexity_penalty: float = 0.01
    final_complexity_penalty: float = 0.001
    eps: float = 1e-8

    def __post_init__(self):
        self.proj = ProjectionBackend(self.energy_cfg.projection, self.energy_cfg.rho)

    def score(self, initial_energy: float, final_state: RegisterState,
              x: torch.Tensor, y: torch.Tensor, sequence: list[int]) -> FitnessResult:
        B_final = torch.nan_to_num(evaluate_register_state(final_state, x))
        final_energy = float(self.proj.residual_energy(B_final, y).detach().cpu())
        complexity = int(final_state.complexity[final_state.active.bool()].sum().detach().cpu().item())
        y_var = float(((y - y.mean()) ** 2).sum().detach().cpu())
        final_r2 = 1.0 - (2.0 * final_energy) / max(y_var, self.eps)
        cost = sum(op_cost_from_action_id(final_state.K, a) for a in sequence)
        if self.fitness == "residual_energy":
            value = -final_energy
        elif self.fitness == "r2":
            value = final_r2
        elif self.fitness == "normalized_energy_improvement":
            value = (initial_energy - final_energy) / max(abs(initial_energy), self.eps)
        else:
            raise ValueError(f"unknown rollout fitness: {self.fitness}")
        value -= self.complexity_penalty * cost
        value -= self.final_complexity_penalty * complexity
        return FitnessResult(float(value), final_energy, float(final_r2), complexity,
                             [int(a) for a in sequence])


def op_cost_from_action_id(K: int, action_id: int) -> float:
    return float(op_cost(op_id_from_action_id(K, action_id)))


def op_id_from_action_id(K: int, action_id: int) -> int:
    return int(action_id) // (K * K * K)


def _normalize_operator_scores(raw: dict[int | str, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for key, value in raw.items():
        if isinstance(key, str) and not key.isdigit():
            if key not in NAME_TO_ID:
                continue
            op_id = NAME_TO_ID[key]
        else:
            op_id = int(key)
        out[int(op_id)] = float(value)
    return out


@dataclass
class CompletionResult:
    scores: list[FitnessResult]


class RolloutEvaluator:
    def __init__(self, action_space: ActionSpace, energy_cfg: ActionEnergyConfig,
                 max_completion_steps: int = 4, rollout_policy: str = "mixed",
                 fitness: str = "normalized_energy_improvement",
                 complexity_penalty: float = 0.01,
                 final_complexity_penalty: float = 0.001,
                 gp_action_scores: dict[int, float] | None = None,
                 gp_operator_scores: dict[int | str, float] | None = None,
                 seed: int = 0):
        if rollout_policy not in {"random", "semantic_greedy", "mixed", "gp_guided"}:
            raise ValueError(f"unknown rollout policy: {rollout_policy}")
        self.space = action_space
        self.executor = ActionExecutor(action_space)
        self.energy = ActionEnergy(action_space, energy_cfg)
        self.proj = ProjectionBackend(energy_cfg.projection, energy_cfg.rho)
        self.scorer = FitnessScorer(
            energy_cfg,
            fitness=fitness,
            complexity_penalty=complexity_penalty,
            final_complexity_penalty=final_complexity_penalty,
        )
        self.max_completion_steps = int(max_completion_steps)
        self.rollout_policy = rollout_policy
        self.gp_action_scores = {int(k): float(v) for k, v in (gp_action_scores or {}).items()}
        self.gp_operator_scores = _normalize_operator_scores(gp_operator_scores or {})
        self.seed = int(seed)

    def evaluate_after_action(self, state: RegisterState, first_action: int,
                              x: torch.Tensor, y: torch.Tensor,
                              n_rollouts: int, sample_index: int = 0) -> CompletionResult:
        B0 = torch.nan_to_num(evaluate_register_state(state, x))
        initial_energy = float(self.proj.residual_energy(B0, y).detach().cpu())
        out = []
        for rollout_idx in range(int(n_rollouts)):
            rng = random.Random(self.seed + sample_index * 1_000_003 + rollout_idx * 97)
            cur = self.executor.execute_symbolic(state, int(first_action))
            seq = [int(first_action)]
            for step in range(self.max_completion_steps):
                ids = self.space.valid_actions(cur).to(x.device)
                if ids.numel() == 0:
                    break
                policy = self._policy_for_rollout(rollout_idx, step)
                if policy == "semantic_greedy":
                    B = torch.nan_to_num(evaluate_register_state(cur, x))
                    rewards = self.energy.rewards(B, y, ids)
                    pos = int(rewards.argmax().detach().cpu().item())
                elif policy == "gp_guided":
                    pos = self._gp_guided_position(ids, rng)
                else:
                    pos = rng.randrange(int(ids.numel()))
                action = int(ids[pos].detach().cpu().item())
                cur = self.executor.execute_symbolic(cur, action)
                seq.append(action)
            out.append(self.scorer.score(initial_energy, cur, x, y, seq))
        return CompletionResult(scores=out)

    def _policy_for_rollout(self, rollout_idx: int, step: int) -> str:
        if self.rollout_policy == "mixed":
            return "semantic_greedy" if (rollout_idx + step) % 2 == 0 else "random"
        return self.rollout_policy

    def _gp_guided_position(self, ids: torch.Tensor, rng: random.Random) -> int:
        if not self.gp_action_scores and not self.gp_operator_scores:
            return rng.randrange(int(ids.numel()))
        vals = []
        for a in ids:
            action_id = int(a.detach().cpu().item())
            action_score = self.gp_action_scores.get(action_id, float("-inf"))
            op_score = self.gp_operator_scores.get(op_id_from_action_id(self.space.K, action_id), float("-inf"))
            vals.append(max(action_score, op_score))
        scores = torch.tensor(vals, device=ids.device, dtype=torch.float32)
        if not torch.isfinite(scores).any():
            return rng.randrange(int(ids.numel()))
        return int(scores.argmax().detach().cpu().item())


@dataclass
class RolloutFitnessTarget(TargetEndpoint):
    action_space: ActionSpace
    energy_cfg: ActionEnergyConfig
    eta_adv: float = 1.0
    advantage_eps: float = 1e-6
    advantage_clip: float = 5.0
    smoothing: float = 0.02
    max_completion_steps: int = 4
    n_rollouts_per_action: int = 4
    rollout_policy: str = "mixed"
    reward_aggregation: str = "topk_mean"
    topk: int = 2
    fitness: str = "normalized_energy_improvement"
    complexity_penalty: float = 0.01
    final_complexity_penalty: float = 0.001
    eval_topk: int | None = 32
    fallback_scale: float = 0.25
    gp_action_scores: dict[int, float] | None = None
    gp_operator_scores: dict[int | str, float] | None = None
    seed: int = 0

    def __post_init__(self):
        if self.reward_aggregation not in {"mean", "max", "topk_mean"}:
            raise ValueError(f"unknown reward aggregation: {self.reward_aggregation}")
        self.evaluator = RolloutEvaluator(
            self.action_space,
            self.energy_cfg,
            max_completion_steps=self.max_completion_steps,
            rollout_policy=self.rollout_policy,
            fitness=self.fitness,
            complexity_penalty=self.complexity_penalty,
            final_complexity_penalty=self.final_complexity_penalty,
            gp_action_scores=self.gp_action_scores,
            gp_operator_scores=self.gp_operator_scores,
            seed=self.seed,
        )

    def advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        adv = rewards - rewards.mean(dim=-1, keepdim=True)
        std = adv.std(dim=-1, keepdim=True, unbiased=False)
        adv = adv / std.clamp(min=self.advantage_eps)
        return adv.clamp(min=-self.advantage_clip, max=self.advantage_clip)

    def build_p1(self, B, y, action_ids, energies, p0, context):
        state = context.get("state")
        x = context.get("x")
        yy = context.get("y", y)
        if state is None or x is None:
            raise ValueError("RolloutFitnessTarget requires context['state'] and context['x']")
        one_step = context.get("rewards")
        if one_step is None:
            one_step = -energies
        rollout_rewards, stats = self._rollout_rewards(state, x, yy, action_ids, one_step,
                                                       sample_index=int(context.get("sample_index", 0)))
        adv = self.advantages(rollout_rewards)
        logits = (self.eta_adv * adv)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        p = normalize_simplex(p0 * torch.exp(logits), dim=-1)
        if self.smoothing > 0.0:
            p = normalize_simplex((1.0 - self.smoothing) * p + self.smoothing * p0, dim=-1)
        context["rewards"] = rollout_rewards
        context["rollout_rewards"] = rollout_rewards
        context["one_step_rewards"] = one_step
        context["advantages"] = adv
        context["rollout_stats"] = stats
        return p

    def _rollout_rewards(self, state, x, y, action_ids, one_step, sample_index: int):
        n = int(action_ids.numel())
        rewards = (one_step * float(self.fallback_scale)).clone()
        eval_positions = self._eval_positions(one_step, n)
        per_action = []
        for i in range(n):
            action_id = int(action_ids[i].detach().cpu().item())
            item = {
                "action_id": action_id,
                "one_step_reward": float(one_step[i].detach().cpu()),
                "rollout_evaluated": i in eval_positions,
            }
            if i in eval_positions:
                result = self.evaluator.evaluate_after_action(
                    state, action_id, x, y,
                    n_rollouts=self.n_rollouts_per_action,
                    sample_index=sample_index * 10_000 + i,
                )
                scores = result.scores
                agg = self._aggregate([s.score for s in scores])
                rewards[i] = torch.tensor(agg, device=one_step.device, dtype=one_step.dtype)
                best = max(scores, key=lambda s: s.score)
                vals = [s.score for s in scores]
                item.update({
                    "n_rollouts": len(scores),
                    "mean_score": float(sum(vals) / max(len(vals), 1)),
                    "best_score": float(best.score),
                    "topk_mean_score": float(self._topk_mean(vals)),
                    "score_std": float(torch.tensor(vals).std(unbiased=False).item()) if vals else 0.0,
                    "best_final_energy": float(best.final_energy),
                    "best_final_r2": float(best.final_r2),
                    "best_complexity": int(best.complexity),
                    "best_completion_sequence": best.sequence,
                })
            per_action.append(item)
        stats = self._stats(action_ids, one_step, rewards, eval_positions, per_action)
        return rewards, stats

    def _eval_positions(self, one_step: torch.Tensor, n: int) -> set[int]:
        if self.eval_topk is None or self.eval_topk >= n:
            return set(range(n))
        k = max(0, int(self.eval_topk))
        if k == 0:
            return set()
        return set(int(i) for i in torch.topk(one_step, min(k, n)).indices.detach().cpu().tolist())

    def _aggregate(self, values: list[float]) -> float:
        if not values:
            return 0.0
        if self.reward_aggregation == "max":
            return float(max(values))
        if self.reward_aggregation == "topk_mean":
            return float(self._topk_mean(values))
        return float(sum(values) / len(values))

    def _topk_mean(self, values: list[float]) -> float:
        if not values:
            return 0.0
        k = min(max(int(self.topk), 1), len(values))
        vals = sorted(values, reverse=True)[:k]
        return float(sum(vals) / k)

    def _stats(self, action_ids, one_step, rollout_rewards, eval_positions, per_action):
        top_one = int(one_step.argmax().detach().cpu().item())
        top_roll = int(rollout_rewards.argmax().detach().cpu().item())
        corr = _safe_corr(one_step, rollout_rewards)
        rollout_rank = _rank_positions(rollout_rewards)
        one_rank = _rank_positions(one_step)
        for i, item in enumerate(per_action):
            item["one_step_rank"] = int(one_rank[i])
            item["rollout_rank"] = int(rollout_rank[i])
            item["rank_shift"] = int(one_rank[i] - rollout_rank[i])
            item["rollout_reward"] = float(rollout_rewards[i].detach().cpu())
        return {
            "n_rollout_evaluated": int(len(eval_positions)),
            "one_step_top1_action": int(action_ids[top_one]),
            "rollout_top1_action": int(action_ids[top_roll]),
            "top1_rollout_action": int(action_ids[top_roll]),
            "top1_agreement": bool(top_one == top_roll),
            "one_step_top1_reward": float(one_step[top_one].detach().cpu()),
            "rollout_top1_reward": float(rollout_rewards[top_roll].detach().cpu()),
            "one_step_rollout_corr": corr,
            "per_action": per_action,
        }


def _rank_positions(values: torch.Tensor) -> list[int]:
    ranks = []
    for i in range(int(values.numel())):
        ranks.append(int((values > values[i]).sum().detach().cpu().item()) + 1)
    return ranks


def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().double().cpu()
    bb = b.detach().double().cpu()
    if aa.numel() < 2 or not torch.isfinite(aa).all() or not torch.isfinite(bb).all():
        return 0.0
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    std_a = aa.std(unbiased=False)
    std_b = bb.std(unbiased=False)
    if not torch.isfinite(std_a) or not torch.isfinite(std_b) or float(std_a) < 1e-12 or float(std_b) < 1e-12:
        return 0.0
    corr = (aa * bb).mean() / (std_a * std_b).clamp(min=1e-12)
    if not torch.isfinite(corr):
        return 0.0
    return float(corr.clamp(min=-1.0, max=1.0))
