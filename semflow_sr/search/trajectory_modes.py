"""Block/full trajectory execution modes.

These helpers make the current transition explicit: trajectory rewards can supervise
local action flow, and separate execution modes can commit a whole sampled block or
full trajectory when an experiment asks for it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from ..actions.action_executor import ActionExecutor
from ..actions.action_features import action_features
from ..actions.action_space import ActionSpace
from ..candidates.base import CandidateEvalOutput, CandidateFlowTarget, SemanticCandidate
from ..candidates.evaluator import CandidateEvaluator
from ..candidates.sampler import BlockCandidateSampler
from ..flow.semantic_fisher import (
    semantic_fisher_lograte,
    semantic_fisher_simplex_velocity,
    semantic_fisher_sphere_step,
    semantic_fisher_sphere_velocity,
)
from ..registers.executor import evaluate_register_state
from ..registers.state import RegisterState
from ..semantics.energy import ActionEnergyConfig
from ..semantics.energy import ActionEnergy
from ..semantics.projection import ProjectionBackend
from ..utils.numerical import EPS, normalize_simplex
from ..trajectories.evaluator import GlobalTrajectoryEvaluator
from ..trajectories.global_block_sampler import GlobalTrajectorySampler
from ..trajectories.risk_advantage import build_group_advantages
from ..trajectories.sampler import GrammarTrajectorySampler, Trajectory, TrajectorySampler
from ..targets.semantic_fisher_risk_flow import SemanticFisherRiskFlowTargetBuilder


@dataclass
class TrajectoryExecutionResult:
    state: RegisterState
    energy_trace: list[float]
    selected_actions: list[int]
    steps_committed: int
    diagnostics: dict


def select_and_commit_block(
    state: RegisterState,
    x: torch.Tensor,
    y: torch.Tensor,
    space: ActionSpace,
    *,
    block_size: int = 3,
    candidate_actions: list[list[int]] | None = None,
    budget: int | None = 64,
    energy_cfg: ActionEnergyConfig | None = None,
) -> TrajectoryExecutionResult:
    """Select a terminal-reward block candidate and execute every action in it."""
    energy_cfg = energy_cfg or ActionEnergyConfig()
    x = x.detach()
    y = y.detach()
    B = torch.nan_to_num(evaluate_register_state(state, x))
    candidates = _block_candidates(space, state, B, y, block_size, candidate_actions, budget, energy_cfg)
    if not candidates:
        return TrajectoryExecutionResult(
            state=state,
            energy_trace=[_energy(state, x, y, energy_cfg)],
            selected_actions=[],
            steps_committed=0,
            diagnostics={"execution_mode": "block_commit", "candidate_count": 0},
        )
    eval_out = CandidateEvaluator(space, energy_cfg).evaluate(state, B, y, candidates)
    choice = int(eval_out.rewards.argmax().detach().cpu().item())
    selected = candidates[choice]
    next_state = _execute_actions_symbolic(space, state, selected.actions or [])
    initial_energy = _energy(state, x, y, energy_cfg)
    final_energy = _energy(next_state, x, y, energy_cfg)
    reward_ranks = _rank_desc(eval_out.rewards)
    return TrajectoryExecutionResult(
        state=next_state,
        energy_trace=[initial_energy, final_energy],
        selected_actions=[int(a) for a in (selected.actions or [])],
        steps_committed=len(selected.actions or []),
        diagnostics={
            "execution_mode": "block_commit",
            "candidate_count": len(candidates),
            "block_size": int(block_size),
            "selected_candidate_id": int(selected.candidate_id),
            "selected_candidate_rank": int(reward_ranks[choice].detach().cpu().item()),
            "selected_reward": float(eval_out.rewards[choice].detach().cpu().item()),
            "oracle_reward": float(eval_out.rewards.max().detach().cpu().item()),
            "energy_before": initial_energy,
            "energy_after": final_energy,
        },
    )


def select_full_trajectory(
    state: RegisterState,
    x: torch.Tensor,
    y: torch.Tensor,
    space: ActionSpace,
    *,
    trajectories: list[Trajectory] | None = None,
    sampler: TrajectorySampler | None = None,
    num_samples: int = 64,
    max_len: int = 8,
    energy_cfg: ActionEnergyConfig | None = None,
) -> TrajectoryExecutionResult:
    """Select and execute a complete trajectory by terminal reward."""
    energy_cfg = energy_cfg or ActionEnergyConfig()
    sampler = sampler or GrammarTrajectorySampler(space, seed=0)
    trajectories = list(trajectories or sampler.sample(state, num_samples=num_samples, max_len=max_len))
    if not trajectories:
        return TrajectoryExecutionResult(
            state=state,
            energy_trace=[_energy(state, x, y, energy_cfg)],
            selected_actions=[],
            steps_committed=0,
            diagnostics={"execution_mode": "full_selector", "candidate_count": 0},
        )
    eval_out = GlobalTrajectoryEvaluator(space, energy_cfg).evaluate(trajectories, x, y, initial_state=state)
    choice = int(eval_out.rewards.argmax().detach().cpu().item())
    selected = trajectories[choice]
    final_state = selected.metadata.get("final_state")
    if final_state is None:
        final_state = _execute_actions_symbolic(space, state, selected.actions)
    initial_energy = _energy(state, x, y, energy_cfg)
    final_energy = _energy(final_state, x, y, energy_cfg)
    reward_ranks = _rank_desc(eval_out.rewards)
    return TrajectoryExecutionResult(
        state=final_state,
        energy_trace=[initial_energy, final_energy],
        selected_actions=[int(a) for a in selected.actions],
        steps_committed=len(selected.actions),
        diagnostics={
            "execution_mode": "full_selector",
            "candidate_count": len(trajectories),
            "selected_candidate_id": choice,
            "selected_candidate_rank": int(reward_ranks[choice].detach().cpu().item()),
            "selected_reward": float(eval_out.rewards[choice].detach().cpu().item()),
            "oracle_reward": float(eval_out.rewards.max().detach().cpu().item()),
            "oracle_best_r2": float(eval_out.final_r2.max().detach().cpu().item()),
            "energy_before": initial_energy,
            "energy_after": final_energy,
        },
    )


def build_global_block_target(
    state: RegisterState,
    B: torch.Tensor,
    y: torch.Tensor,
    space: ActionSpace,
    trajectories: list[Trajectory],
    trajectory_eval,
    *,
    block_size: int,
    aggregation: str = "mean",
    p0_mode: str = "uniform",
    beta: float = 1.0,
    gamma: float = 0.1,
    eta: float = 1.0,
    risk_mode: str = "top_alpha",
    risk_alpha: float = 0.1,
    risk_normalize: str = "rank",
    gram_rank: int | None = None,
    energy_cfg: ActionEnergyConfig | None = None,
) -> CandidateFlowTarget:
    """Build a block-level semantic-Fisher target from complete trajectories."""
    energy_cfg = energy_cfg or ActionEnergyConfig()
    candidates, block_rewards, counts = _first_block_candidates(
        space,
        trajectories,
        trajectory_eval,
        block_size=max(int(block_size), 1),
        aggregation=aggregation,
    )
    if not candidates:
        raise ValueError("global block target requires at least one valid first block")
    eval_out = CandidateEvaluator(space, energy_cfg).evaluate(state, B, y, candidates)
    rewards = block_rewards.to(device=B.device, dtype=B.dtype)
    if p0_mode == "uniform":
        p_start = torch.full_like(rewards, 1.0 / max(rewards.numel(), 1))
    elif p0_mode == "frequency":
        p_start = normalize_simplex(counts.to(device=B.device, dtype=B.dtype), dim=-1)
    else:
        raise ValueError(f"unknown block p0_mode: {p0_mode}")
    rho = torch.softmax(float(eta) * _rank_normalize(rewards), dim=-1)
    advantages = _group_normalize(rho.clamp(min=EPS).log() - p_start.clamp(min=EPS).log())
    w_target = semantic_fisher_lograte(
        p_start,
        advantages,
        eval_out.gram,
        beta=beta,
        gamma=gamma,
        gram_rank=gram_rank,
        gram_factors=eval_out.xi,
    )
    z = p_start.clamp(min=EPS).sqrt()
    p_target = semantic_fisher_sphere_step(p_start, w_target, dt=1.0)
    return CandidateFlowTarget(
        candidates=candidates,
        p_start=p_start,
        scores=rewards,
        rewards=rewards,
        advantages=advantages,
        w_target=w_target,
        zdot_target=semantic_fisher_sphere_velocity(z, w_target),
        pdot_target=semantic_fisher_simplex_velocity(p_start, w_target),
        p_target=p_target,
        eval=replace(eval_out, rewards=rewards),
    )


def build_risk_flow_block_target(
    state: RegisterState,
    B: torch.Tensor,
    y: torch.Tensor,
    space: ActionSpace,
    trajectories: list[Trajectory],
    trajectory_eval,
    *,
    block_size: int,
    risk_mode: str = "top_alpha",
    risk_alpha: float = 0.1,
    risk_normalize: str = "rank",
    aggregation: str = "mean",
    beta: float = 1.0,
    gamma: float = 0.1,
    gram_rank: int | None = None,
    energy_cfg: ActionEnergyConfig | None = None,
) -> CandidateFlowTarget:
    """Build the main risk-flow target from trajectory advantages.

    Complete trajectory rewards first become group-relative trajectory advantages.
    Those advantages are then assigned to the actually sampled first block at the
    current prefix. This replaces the legacy max-over-suffix block reward target.
    """
    energy_cfg = energy_cfg or ActionEnergyConfig()
    block_size = max(int(block_size), 1)
    risk = build_group_advantages(
        trajectory_eval.rewards.to(device=B.device, dtype=B.dtype),
        mode=risk_mode,
        alpha=float(risk_alpha),
        normalize=risk_normalize,
    )
    candidates, block_advantages, old_policy_probs, counts, source_indices = _first_block_advantage_candidates(
        trajectories,
        risk.trajectory_advantages.to(device=B.device, dtype=B.dtype),
        block_size=block_size,
        aggregation=aggregation,
    )
    if not candidates:
        raise ValueError("risk-flow block target requires at least one valid first block")
    eval_out = CandidateEvaluator(space, energy_cfg).evaluate(state, B, y, candidates)
    for cand, count, indices in zip(candidates, counts, source_indices):
        cand.metadata.update({
            "target_kind": "semantic_fisher_risk_flow",
            "aggregation": aggregation,
            "risk_alpha": float(risk_alpha),
            "source_count": int(count),
            "source_indices": [int(i) for i in indices],
        })
    target = SemanticFisherRiskFlowTargetBuilder(
        beta=beta,
        gamma=gamma,
        gram_rank=gram_rank,
        normalize_advantage=False,
    ).build(
        candidates=candidates,
        old_policy_probs=old_policy_probs.to(device=B.device, dtype=B.dtype),
        block_advantages=block_advantages.to(device=B.device, dtype=B.dtype),
        gram=eval_out.gram,
        xi=eval_out.xi,
        eval_output=replace(eval_out, rewards=block_advantages.to(device=B.device, dtype=B.dtype)),
    )
    target.eval.log_priors[:] = old_policy_probs.to(device=B.device, dtype=B.dtype).clamp(min=EPS).log()
    return target


def select_global_block_commit(
    state: RegisterState,
    x: torch.Tensor,
    y: torch.Tensor,
    space: ActionSpace,
    *,
    sampler: GlobalTrajectorySampler | None = None,
    model_or_policy=None,
    trajectories: list[Trajectory] | None = None,
    block_size: int = 3,
    num_samples: int = 64,
    max_len: int = 8,
    temperature: float = 1.0,
    exploration: float = 0.0,
    aggregation: str = "mean",
    p0_mode: str = "uniform",
    selector_mode: str = "exact",
    learned_scores: torch.Tensor | None = None,
    beta: float = 1.0,
    gamma: float = 0.1,
    eta: float = 1.0,
    risk_mode: str = "top_alpha",
    risk_alpha: float = 0.1,
    risk_normalize: str = "rank",
    gram_rank: int | None = None,
    energy_cfg: ActionEnergyConfig | None = None,
) -> TrajectoryExecutionResult:
    """Sample full trajectories, aggregate first-H blocks, select one block, and commit it."""
    energy_cfg = energy_cfg or ActionEnergyConfig()
    x = x.detach()
    y = y.detach()
    sampler = sampler or GlobalTrajectorySampler(space, seed=0)
    trajectories = list(
        trajectories
        if trajectories is not None
        else sampler.sample_from_policy(
            state,
            model_or_policy,
            num_samples=int(num_samples),
            max_len=int(max_len),
            temperature=float(temperature),
            exploration=float(exploration),
        )
    )
    initial_energy = _energy(state, x, y, energy_cfg)
    if not trajectories:
        return TrajectoryExecutionResult(
            state=state,
            energy_trace=[initial_energy],
            selected_actions=[],
            steps_committed=0,
            diagnostics={"execution_mode": "global_block_commit", "trajectory_num_samples": 0, "num_unique_blocks": 0},
        )
    eval_out = GlobalTrajectoryEvaluator(space, energy_cfg).evaluate(trajectories, x, y, initial_state=state)
    B = torch.nan_to_num(evaluate_register_state(state, x))
    target = build_risk_flow_block_target(
        state,
        B,
        y,
        space,
        trajectories,
        eval_out,
        block_size=block_size,
        risk_mode=risk_mode,
        risk_alpha=risk_alpha,
        risk_normalize=risk_normalize,
        aggregation=aggregation,
        beta=beta,
        gamma=gamma,
        gram_rank=gram_rank,
        energy_cfg=energy_cfg,
    )
    oracle_pos = int(target.rewards.argmax().detach().cpu().item())
    exact_pos = int(target.p_target.argmax().detach().cpu().item())
    learned_pos = None
    if selector_mode == "oracle":
        choice = oracle_pos
    elif selector_mode == "exact":
        choice = exact_pos
    elif selector_mode == "learned":
        if learned_scores is None and callable(model_or_policy):
            learned_scores = _learned_first_action_proxy_scores(
                model_or_policy,
                state,
                B,
                x,
                y,
                space,
                target,
                energy_cfg,
                beta,
            )
        if learned_scores is None:
            raise ValueError("selector_mode='learned' requires learned_scores over block candidates")
        learned_scores = learned_scores.to(device=target.rewards.device, dtype=target.rewards.dtype)
        if learned_scores.numel() != target.rewards.numel():
            raise ValueError("learned_scores must have one value per block candidate")
        learned_pos = int(learned_scores.argmax().detach().cpu().item())
        choice = learned_pos
    else:
        raise ValueError(f"unknown global block selector_mode: {selector_mode}")
    selected = target.candidates[choice]
    next_state = _execute_actions_symbolic(space, state, selected.actions or [])
    final_energy = _energy(next_state, x, y, energy_cfg)
    reward_ranks = _rank_desc(target.rewards)
    p_ranks = _rank_desc(target.p_target)
    diag = {
        "execution_mode": "global_block_commit",
        "target_kind": "semantic_fisher_risk_flow",
        "selector_mode": str(selector_mode),
        "block_size": int(block_size),
        "trajectory_num_samples": len(trajectories),
        "num_unique_blocks": len(target.candidates),
        "aggregation": str(aggregation),
        "p0_mode": "old_policy_observed_frequency",
        "risk_mode": str(risk_mode),
        "risk_alpha": float(risk_alpha),
        "risk_normalize": str(risk_normalize),
        "selected_candidate_id": int(selected.candidate_id),
        "selected_actions": [int(a) for a in (selected.actions or [])],
        "selected_reward": float(target.rewards[choice].detach().cpu().item()),
        "selected_probability": float(target.p_target[choice].detach().cpu().item()),
        "oracle_reward": float(target.rewards[oracle_pos].detach().cpu().item()),
        "oracle_block_reward_rank": int(reward_ranks[oracle_pos].detach().cpu().item()),
        "exact_block_reward_rank": int(reward_ranks[exact_pos].detach().cpu().item()),
        "selected_block_reward_rank": int(reward_ranks[choice].detach().cpu().item()),
        "selected_probability_rank": int(p_ranks[choice].detach().cpu().item()),
        "trajectory_oracle_r2": float(eval_out.final_r2.max().detach().cpu().item())
        if eval_out.final_r2.numel() else 0.0,
        "block_oracle_r2": float(target.candidates[oracle_pos].metadata.get("best_r2", 0.0)),
        "selected_block_best_r2": float(selected.metadata.get("best_r2", 0.0)),
        "p_start_entropy": _entropy(target.p_start),
        "p_final_entropy": _entropy(target.p_target),
        "update_kl": float(
            (target.p_target * (target.p_target.clamp(min=EPS).log() - target.p_start.clamp(min=EPS).log()))
            .sum()
            .detach()
            .cpu()
            .item()
        ),
        "energy_before": initial_energy,
        "energy_after": final_energy,
    }
    if learned_pos is not None:
        diag["learned_block_reward_rank"] = int(reward_ranks[learned_pos].detach().cpu().item())
        diag["learned_selector_kind"] = "first_action_proxy"
    return TrajectoryExecutionResult(
        state=next_state,
        energy_trace=[initial_energy, final_energy],
        selected_actions=[int(a) for a in (selected.actions or [])],
        steps_committed=len(selected.actions or []),
        diagnostics=diag,
    )


def _block_candidates(
    space: ActionSpace,
    state: RegisterState,
    B: torch.Tensor,
    y: torch.Tensor,
    block_size: int,
    candidate_actions: list[list[int]] | None,
    budget: int | None,
    energy_cfg: ActionEnergyConfig,
) -> list[SemanticCandidate]:
    if candidate_actions is None:
        return BlockCandidateSampler(
            space,
            horizon=int(block_size),
            first_topk=max(1, int(budget or 64)),
            branch_topk=max(1, min(int(budget or 64), 8)),
            energy_cfg=energy_cfg,
        ).sample(state, B=B, y=y, budget=budget)
    out = []
    for idx, actions in enumerate(candidate_actions):
        out.append(SemanticCandidate(
            candidate_id=idx,
            kind="block",
            actions=[int(a) for a in actions[: int(block_size)]],
            complexity=float(len(actions[: int(block_size)])),
            metadata={"horizon": int(block_size), "candidate_group": f"H{int(block_size)}"},
        ))
    return out


def _first_block_candidates(
    space: ActionSpace,
    trajectories: list[Trajectory],
    eval_out,
    *,
    block_size: int,
    aggregation: str,
) -> tuple[list[SemanticCandidate], torch.Tensor, torch.Tensor]:
    groups: dict[tuple[int, ...], list[int]] = {}
    order: list[tuple[int, ...]] = []
    for idx, trajectory in enumerate(trajectories):
        block = tuple(int(a) for a in trajectory.actions[:block_size])
        if not block:
            continue
        if block not in groups:
            groups[block] = []
            order.append(block)
        groups[block].append(idx)
    candidates: list[SemanticCandidate] = []
    rewards = []
    counts = []
    for block in order:
        indices = groups[block]
        idx_t = torch.tensor(indices, device=eval_out.rewards.device, dtype=torch.long)
        vals = eval_out.rewards[idx_t]
        reward, best_local = _aggregate_rewards(vals, aggregation)
        best_idx = indices[int(best_local)]
        complexities = eval_out.complexities[idx_t]
        complexity = float(complexities.min().detach().cpu().item()) if complexities.numel() else float(len(block))
        candidate = SemanticCandidate(
            candidate_id=len(candidates),
            kind="block",
            actions=[int(a) for a in block],
            log_prior=0.0,
            complexity=complexity,
            metadata={
                "horizon": len(block),
                "source_count": len(indices),
                "source_indices": [int(i) for i in indices],
                "best_trajectory_id": str(best_idx),
                "best_r2": float(eval_out.final_r2[best_idx].detach().cpu().item())
                if eval_out.final_r2.numel() else 0.0,
                "terminal_reward": float(reward.detach().cpu().item()),
            },
        )
        candidates.append(candidate)
        rewards.append(reward)
        counts.append(float(len(indices)))
    if rewards:
        reward_t = torch.stack(rewards)
        count_t = torch.tensor(counts, device=reward_t.device, dtype=reward_t.dtype)
    else:
        reward_t = torch.empty(0, device=eval_out.rewards.device, dtype=eval_out.rewards.dtype)
        count_t = torch.empty(0, device=eval_out.rewards.device, dtype=eval_out.rewards.dtype)
    return candidates, reward_t, count_t


def _first_block_advantage_candidates(
    trajectories: list[Trajectory],
    trajectory_advantages: torch.Tensor,
    *,
    block_size: int,
    aggregation: str,
) -> tuple[list[SemanticCandidate], torch.Tensor, torch.Tensor, list[int], list[list[int]]]:
    groups: dict[tuple[int, ...], list[int]] = {}
    order: list[tuple[int, ...]] = []
    for idx, trajectory in enumerate(trajectories):
        block = tuple(int(a) for a in trajectory.actions[:block_size])
        if not block:
            continue
        if block not in groups:
            groups[block] = []
            order.append(block)
        groups[block].append(idx)
    candidates: list[SemanticCandidate] = []
    advantages = []
    old_probs = []
    counts = []
    source_indices = []
    for block in order:
        indices = groups[block]
        idx_t = torch.tensor(indices, device=trajectory_advantages.device, dtype=torch.long)
        vals = trajectory_advantages[idx_t]
        adv, _ = _aggregate_advantages(vals, aggregation)
        candidates.append(SemanticCandidate(
            candidate_id=len(candidates),
            kind="block",
            actions=[int(a) for a in block],
            log_prior=0.0,
            complexity=float(len(block)),
            metadata={"horizon": len(block), "candidate_group": f"H{len(block)}"},
        ))
        advantages.append(adv)
        counts.append(len(indices))
        source_indices.append(indices)
        old_probs.append(float(len(indices)))
    if advantages:
        adv_t = torch.stack(advantages)
        old_p = normalize_simplex(torch.tensor(old_probs, device=adv_t.device, dtype=adv_t.dtype), dim=-1)
    else:
        adv_t = torch.empty(0, device=trajectory_advantages.device, dtype=trajectory_advantages.dtype)
        old_p = torch.empty(0, device=trajectory_advantages.device, dtype=trajectory_advantages.dtype)
    return candidates, adv_t, old_p, counts, source_indices


def _aggregate_advantages(values: torch.Tensor, aggregation: str) -> tuple[torch.Tensor, int]:
    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device, dtype=values.dtype), 0
    if aggregation == "mean":
        pos = int(values.argmax().detach().cpu().item())
        return values.mean(), pos
    if aggregation == "topk_mean":
        k = min(3, int(values.numel()))
        vals, pos = torch.topk(values, k)
        return vals.mean(), int(pos[0].detach().cpu().item())
    if aggregation == "sum":
        pos = int(values.argmax().detach().cpu().item())
        return values.sum(), pos
    if aggregation == "max":
        pos = int(values.argmax().detach().cpu().item())
        return values[pos], pos
    raise ValueError(f"unknown block advantage aggregation: {aggregation}")


def _aggregate_rewards(values: torch.Tensor, aggregation: str) -> tuple[torch.Tensor, int]:
    if values.numel() == 0:
        return torch.tensor(0.0), 0
    if aggregation == "max":
        pos = int(values.argmax().detach().cpu().item())
        return values[pos], pos
    if aggregation == "topk_mean":
        k = min(3, int(values.numel()))
        vals, pos = torch.topk(values, k)
        return vals.mean(), int(pos[0].detach().cpu().item())
    if aggregation == "softmax_weighted_reward":
        weights = torch.softmax(_rank_normalize(values), dim=0)
        pos = int(values.argmax().detach().cpu().item())
        return (weights * values).sum(), pos
    raise ValueError(f"unknown block reward aggregation: {aggregation}")


def _learned_first_action_proxy_scores(
    model,
    state: RegisterState,
    B: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    space: ActionSpace,
    target: CandidateFlowTarget,
    energy_cfg: ActionEnergyConfig,
    beta: float,
) -> torch.Tensor | None:
    """Score blocks with the current action model applied to each block's first action.

    This is an explicit compatibility bridge. A true learned GlobalBlock model should
    replace it with block features over all actions in the block.
    """
    first_actions = [int((c.actions or [])[0]) for c in target.candidates if c.actions]
    if len(first_actions) != len(target.candidates):
        return None
    action_ids = torch.tensor(first_actions, device=B.device, dtype=torch.long)
    energy = ActionEnergy(space, energy_cfg)
    ev = energy.evaluate_actions(B, y, action_ids)
    effect = energy.action_semantic_effects(B, y, action_ids)
    feats = action_features(space, state, action_ids)
    semantic_stats = _semantic_stats_from_effect(effect)
    lam = torch.zeros(1, device=B.device, dtype=B.dtype)
    with torch.no_grad():
        out = model(
            x=x.unsqueeze(0),
            y=y.unsqueeze(0),
            B=B.unsqueeze(0),
            p_lambda=target.p_start.unsqueeze(0),
            lambda_value=lam,
            action_feats=feats.unsqueeze(0),
            energies=ev.energies.unsqueeze(0),
            weights=torch.ones_like(target.p_start).unsqueeze(0),
            semantic_stats=semantic_stats.unsqueeze(0),
            gram=effect.gram.unsqueeze(0),
            beta_value=float(beta),
            action_mask=torch.ones(1, action_ids.numel(), dtype=torch.bool, device=B.device),
        )
    lograte = getattr(out, "lograte_logits", None)
    if lograte is not None:
        return lograte.squeeze(0)
    potential = getattr(out, "potential_logits", None)
    if potential is not None:
        return potential.squeeze(0)
    v_pred = getattr(out, "v_pred", None)
    return v_pred.squeeze(0) if v_pred is not None else None


def _execute_actions_symbolic(space: ActionSpace, state: RegisterState, actions: list[int]) -> RegisterState:
    executor = ActionExecutor(space)
    current = state.clone()
    for action in actions:
        current = executor.execute_symbolic(current, int(action))
    return current


def _energy(state: RegisterState, x: torch.Tensor, y: torch.Tensor, cfg: ActionEnergyConfig) -> float:
    B = torch.nan_to_num(evaluate_register_state(state, x))
    return float(ProjectionBackend(cfg.projection, cfg.rho).residual_energy(B, y).detach().cpu().item())


def _rank_desc(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort(descending=True)
    ranks = torch.empty_like(order)
    ar = torch.arange(order.numel(), device=values.device, dtype=order.dtype) + 1
    ranks.scatter_(0, order, ar)
    return ranks


def _rank_normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    order = values.argsort(descending=False)
    ranks = torch.empty_like(order, dtype=values.dtype)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=values.dtype)
    ranks = ranks / max(values.numel() - 1, 1)
    return ranks - ranks.mean()


def _group_normalize(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=-1, keepdim=True)
    std = centered.std(dim=-1, keepdim=True, unbiased=False)
    return torch.nan_to_num(centered / std.clamp(min=EPS))


def _entropy(p: torch.Tensor) -> float:
    p = p.detach().float().clamp(min=EPS)
    return float((-(p * p.log()).sum()).cpu().item())


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
