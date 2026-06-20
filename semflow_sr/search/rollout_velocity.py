"""Semantic-Fisher rollout inference."""
from __future__ import annotations
from dataclasses import dataclass, field
import random
import torch

from ..registers.state import RegisterState, init_register_state
from ..registers.executor import evaluate_register_state
from ..actions.action_space import ActionSpace
from ..actions.action_executor import ActionExecutor
from ..actions.action_features import action_features
from ..actions.support_sampler import SupportSampler
from ..semantics.energy import ActionEnergy, ActionEnergyConfig
from ..semantics.projection import ProjectionBackend
from ..flow.natural_path import effective_advantage_from_target
from ..flow.semantic_fisher import (
    semantic_fisher_lograte,
    semantic_fisher_sphere_step,
)
from ..inference.iterative_policy_update import beta_for_update, closed_form_policy_update
from ..endpoints.prior_uniform import UniformPrior
from ..endpoints.target_group_advantage import GroupAdvantageTarget
from ..endpoints.target_rollout_fitness import RolloutFitnessTarget
from ..endpoints.target_global_trajectory import GlobalTrajectoryTarget
from ..sr.ops import NAME_TO_ID, get_op
from ..utils.numerical import EPS
from .trajectory_modes import select_and_commit_block, select_full_trajectory, select_global_block_commit
from ..gp_distill.trajectory_pool import load_gp_trajectory_population
from ..trajectories.sampler import GrammarTrajectorySampler


@dataclass
class RolloutResult:
    state: RegisterState
    energy_trace: list[float]
    steps: int
    diagnostics: list[dict] = field(default_factory=list)


def _residual_energy(state, x, y, proj):
    B = torch.nan_to_num(evaluate_register_state(state, x))
    return proj.residual_energy(B, y).item()


def _entropy(p: torch.Tensor) -> float:
    pp = p.clamp(min=EPS)
    return float((-(pp * pp.log()).sum()).detach().cpu())


def _rank_desc(values: torch.Tensor, index: int) -> int:
    target = values[index]
    return int((values > target).sum().detach().cpu().item()) + 1


def _float(x) -> float:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().item()
    return float(x)


def _action_record(space: ActionSpace, action_id: int) -> dict:
    spec = space.decode(int(action_id))
    return {
        "id": int(action_id),
        "op_id": int(spec.op_id),
        "op": get_op(spec.op_id).name,
        "read_1": int(spec.read_1),
        "read_2": int(spec.read_2),
        "write": int(spec.write),
    }


def _normalize_gp_operator_scores(raw: dict[int | str, float] | None) -> dict[int, float]:
    out: dict[int, float] = {}
    for key, value in (raw or {}).items():
        if isinstance(key, str) and not key.isdigit():
            if key not in NAME_TO_ID:
                continue
            op_id = NAME_TO_ID[key]
        else:
            op_id = int(key)
        out[int(op_id)] = float(value)
    return out


def _gp_policy_prior(
    space: ActionSpace,
    action_ids: torch.Tensor,
    gp_action_scores: dict[int, float] | None,
    gp_operator_scores: dict[int | str, float] | None,
    eps: float = EPS,
) -> torch.Tensor | None:
    action_scores = {int(k): float(v) for k, v in (gp_action_scores or {}).items()}
    operator_scores = _normalize_gp_operator_scores(gp_operator_scores)
    if not action_scores and not operator_scores:
        return None
    vals = []
    matched = False
    for action in action_ids:
        action_id = int(action.detach().cpu().item())
        spec = space.decode(action_id)
        candidates = []
        if action_id in action_scores:
            candidates.append(action_scores[action_id])
        if int(spec.op_id) in operator_scores:
            candidates.append(operator_scores[int(spec.op_id)])
        if candidates:
            matched = True
            vals.append(max(candidates))
        else:
            vals.append(0.0)
    if not matched:
        return None
    guide = torch.tensor(vals, device=action_ids.device, dtype=torch.float32)
    guide = guide - guide.mean()
    std = guide.std(unbiased=False)
    if float(std.detach().cpu()) > eps:
        guide = guide / std.clamp(min=eps)
    return guide


def _support_contains(action_ids: torch.Tensor, action_id: int) -> bool:
    return bool((action_ids == int(action_id)).any().detach().cpu().item())


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


def rollout_velocity(model, x, y, num_vars, K, ops_ids, device,
                     max_steps: int = 16, grid: int = 5, step_size: float = 1.0,
                     eta: float = 1.0, beta: float | None = None, eps: float = 1e-4, energy_cfg=None,
                     greedy: bool = True, max_support: int = 256,
                     support_mode: str = "mixed_topk_random", support_topk: int | None = None,
                     target: str = "one_step_advantage", target_kwargs: dict | None = None,
                     num_policy_updates: int = 1,
                     integration_method: str = "semantic_fisher_sphere",
                     ode_steps: int = 1,
                     update_mode: str = "fixed_beta",
                     target_kl: float = 0.05,
                     beta_max: float = 10.0,
                     bisection_steps: int = 20,
                     gamma: float = 0.1,
                     gram_rank: int | None = None,
                     support_full_threshold: int | None = None,
                     gp_policy_weight: float = 0.0,
                     gp_action_scores: dict[int, float] | None = None,
                     gp_operator_scores: dict[int | str, float] | None = None,
                     record_diagnostics: bool = False,
                     record_path: bool = False,
                     execution_mode: str = "action",
                     block_size: int = 3,
                     block_candidate_budget: int | None = 64,
                     trajectory_num_samples: int = 64,
                     trajectory_max_len: int | None = None,
                     trajectory_temperature: float = 1.0,
                     trajectory_exploration: float = 0.0,
                     block_aggregation: str = "mean",
                     block_p0_mode: str = "uniform",
                     global_block_selector: str = "exact",
                     risk_mode: str = "top_alpha",
                     risk_alpha: float = 0.1,
                     risk_normalize: str = "rank",
                     gp_population_path: str | None = None,
                     gp_sample_mode: str = "base_plus_gp") -> RolloutResult:
    model.eval()
    space = ActionSpace(K, ops_ids)
    execu = ActionExecutor(space)
    energy_cfg = energy_cfg or ActionEnergyConfig()
    energy = ActionEnergy(space, energy_cfg)
    proj = ProjectionBackend(energy_cfg.projection, energy_cfg.rho)
    prior = UniformPrior()
    support_sampler = SupportSampler(mode=support_mode, max_support=max_support,
                                     topk=support_topk, full_threshold=support_full_threshold, seed=0)
    beta_value = float(eta if beta is None else beta)
    target_kwargs = target_kwargs or {}
    group_target = None
    rollout_target = None
    global_trajectory_target = None
    if target in {"one_step_advantage", "group_advantage", "semantic_advantage_flow"}:
        group_target = GroupAdvantageTarget(**target_kwargs)
    elif target in {"rollout_fitness_advantage", "rollout_fitness"}:
        rollout_target = RolloutFitnessTarget(space, energy_cfg, **target_kwargs)
    elif target in {"global_trajectory", "global_trajectory_marginal", "trajectory_marginal",
                    "semantic_fisher_risk_flow", "risk_flow"}:
        global_trajectory_target = GlobalTrajectoryTarget(space, energy_cfg, **target_kwargs)
    elif target != "energy":
        raise ValueError(f"unknown rollout target: {target}")
    state = init_register_state(num_vars, K, device)
    x = x.to(device); y = y.to(device)
    trace = [_residual_energy(state, x, y, proj)]
    diagnostics: list[dict] = []
    if execution_mode == "block_commit":
        result = select_and_commit_block(
            state,
            x,
            y,
            space,
            block_size=min(int(block_size), max(int(max_steps), 1)),
            budget=block_candidate_budget,
            energy_cfg=energy_cfg,
        )
        diag = dict(result.diagnostics)
        diag["selected_actions"] = [_action_record(space, a) for a in result.selected_actions]
        return RolloutResult(
            state=result.state,
            energy_trace=result.energy_trace,
            steps=result.steps_committed,
            diagnostics=[diag] if record_diagnostics else [],
        )
    if execution_mode == "global_block_commit":
        committed = 0
        while committed < int(max_steps) and trace[-1] > eps:
            remaining = max(int(max_steps) - committed, 1)
            result = select_global_block_commit(
                state,
                x,
                y,
                space,
                model_or_policy=model,
                block_size=min(int(block_size), remaining),
                num_samples=int(trajectory_num_samples),
                max_len=int(trajectory_max_len or remaining),
                temperature=float(trajectory_temperature),
                exploration=float(trajectory_exploration),
                aggregation=str(block_aggregation),
                p0_mode=str(block_p0_mode),
                selector_mode=str(global_block_selector),
                beta=beta_value,
                gamma=float(gamma),
                eta=float(beta_value),
                risk_mode=str(risk_mode),
                risk_alpha=float(risk_alpha),
                risk_normalize=str(risk_normalize),
                gram_rank=gram_rank,
                energy_cfg=energy_cfg,
            )
            if result.steps_committed <= 0:
                break
            state = result.state
            committed += int(result.steps_committed)
            trace.append(float(result.energy_trace[-1]))
            if record_diagnostics:
                diag = dict(result.diagnostics)
                diag["step"] = len(diagnostics)
                diag["selected_actions"] = [_action_record(space, a) for a in result.selected_actions]
                diag["steps_committed_total"] = int(committed)
                diagnostics.append(diag)
        return RolloutResult(
            state=state,
            energy_trace=trace,
            steps=committed,
            diagnostics=diagnostics,
        )
    if execution_mode == "full_selector":
        full_trajectories = None
        if gp_population_path:
            gp_trajs = load_gp_trajectory_population(
                gp_population_path,
                space,
                state,
                max_len=int(trajectory_max_len or max_steps),
            )
            if gp_sample_mode == "gp_only":
                full_trajectories = gp_trajs[: int(trajectory_num_samples)]
            elif gp_sample_mode == "base_plus_gp":
                base_budget = max(int(trajectory_num_samples) - len(gp_trajs), 0)
                base_trajs = GrammarTrajectorySampler(space, seed=0).sample(
                    state,
                    num_samples=base_budget,
                    max_len=int(trajectory_max_len or max_steps),
                )
                full_trajectories = [*gp_trajs, *base_trajs][: int(trajectory_num_samples)]
            else:
                raise ValueError(f"unknown gp_sample_mode: {gp_sample_mode}")
        result = select_full_trajectory(
            state,
            x,
            y,
            space,
            trajectories=full_trajectories,
            num_samples=int(trajectory_num_samples),
            max_len=int(trajectory_max_len or max_steps),
            energy_cfg=energy_cfg,
        )
        diag = dict(result.diagnostics)
        diag["selected_actions"] = [_action_record(space, a) for a in result.selected_actions]
        diag["gp_population_path"] = str(gp_population_path or "")
        diag["gp_sample_mode"] = str(gp_sample_mode) if gp_population_path else ""
        return RolloutResult(
            state=result.state,
            energy_trace=result.energy_trace,
            steps=result.steps_committed,
            diagnostics=[diag] if record_diagnostics else [],
        )
    if execution_mode != "action":
        raise ValueError(f"unknown execution_mode: {execution_mode}")

    for step_idx in range(max_steps):
        if trace[-1] <= eps:
            break
        full_action_ids = space.valid_actions(state).to(device)
        if full_action_ids.numel() == 0:        # append 语义: 槽用尽 -> 停机
            break
        B = torch.nan_to_num(evaluate_register_state(state, x))
        full_rewards = energy.rewards(B, y, full_action_ids)
        support = support_sampler.sample(full_action_ids, rewards=full_rewards, sample_index=len(trace))
        action_ids = support.action_ids
        ev = energy.evaluate_actions(B, y, action_ids)
        effect = energy.action_semantic_effects(B, y, action_ids)
        energies = ev.energies
        one_step_rewards = ev.rewards
        feats = action_features(space, state, action_ids)
        semantic_stats = _semantic_stats_from_effect(effect)
        gram = effect.gram
        gp_prior = _gp_policy_prior(space, action_ids, gp_action_scores, gp_operator_scores)
        p = prior.build_p0(B, y, action_ids, {})
        p_start = p.clone()
        p_target = None
        plain_p_target = None
        advantages = None
        rollout_stats = None
        target_rewards = one_step_rewards
        proposal = support.proposal_probs.to(device=device, dtype=one_step_rewards.dtype)
        if group_target is not None:
            p1_context = {
                "rewards": one_step_rewards,
                "proposal_probs": proposal,
            }
            plain_p_target = group_target.build_p1(B, y, action_ids, energies, p_start, p1_context)
            advantages = p1_context.get("advantages")
            target_rewards = p1_context.get("rewards", one_step_rewards)
        elif rollout_target is not None:
            p1_context = {
                "rewards": one_step_rewards,
                "proposal_probs": proposal,
                "state": state,
                "x": x,
                "y": y,
                "sample_index": step_idx,
            }
            plain_p_target = rollout_target.build_p1(B, y, action_ids, energies, p_start, p1_context)
            advantages = p1_context.get("advantages")
            target_rewards = p1_context.get("rollout_rewards", p1_context.get("rewards", one_step_rewards))
            rollout_stats = p1_context.get("rollout_stats")
        elif global_trajectory_target is not None:
            p1_context = {
                "rewards": one_step_rewards,
                "proposal_probs": proposal,
                "state": state,
                "x": x,
                "y": y,
                "sample_index": step_idx,
            }
            plain_p_target = global_trajectory_target.build_p1(B, y, action_ids, energies, p_start, p1_context)
            advantages = p1_context.get("advantages")
            target_rewards = p1_context.get("global_trajectory_rewards", p1_context.get("rewards", one_step_rewards))
            rollout_stats = {"trajectory_stats": p1_context.get("trajectory_stats", {})}
        else:
            p_target = None
        w = torch.ones_like(p)
        exact_lograte = None
        if advantages is not None:
            exact_lograte = semantic_fisher_lograte(
                p_start,
                advantages.to(device=B.device, dtype=B.dtype),
                gram,
                beta=beta_value,
                gamma=gamma,
                gram_rank=gram_rank,
                gram_factors=effect.xi,
            )
            p_target = semantic_fisher_sphere_step(p_start, exact_lograte, dt=1.0)
            advantages_eff = exact_lograte
        elif plain_p_target is not None:
            advantages_eff = effective_advantage_from_target(p_start, plain_p_target, beta_value)
        else:
            advantages_eff = advantages
        path_diag: list[dict] = []
        final_score = None
        with torch.no_grad():
            for update_idx in range(max(int(num_policy_updates), 1)):
                p_update_start = p.clone()
                if integration_method in {"semantic_fisher_sphere", "semantic_fisher_ode"}:
                    n_ode = 1 if integration_method == "semantic_fisher_sphere" else max(int(ode_steps), 1)
                    used_beta = beta_value
                    for ode_idx in range(n_ode):
                        lam_value = torch.tensor(
                            [(ode_idx / max(n_ode, 1)) if integration_method == "semantic_fisher_ode" else 0.0],
                            device=device,
                            dtype=B.dtype,
                        )
                        out = model(
                            x=x.unsqueeze(0), y=y.unsqueeze(0), B=B.unsqueeze(0),
                            p_lambda=p.unsqueeze(0), lambda_value=lam_value,
                            action_feats=feats.unsqueeze(0), energies=energies.unsqueeze(0),
                            weights=w.unsqueeze(0), semantic_stats=semantic_stats.unsqueeze(0),
                            gram=gram.unsqueeze(0), beta_value=beta_value,
                            action_mask=torch.ones(1, action_ids.numel(), dtype=torch.bool, device=device),
                        )
                        lograte = getattr(out, "lograte_logits", None)
                        v = out.v_pred.squeeze(0)
                        if lograte is not None:
                            final_score = lograte.squeeze(0)
                            if gp_prior is not None and float(gp_policy_weight) != 0.0:
                                final_score = final_score + float(gp_policy_weight) * gp_prior.to(
                                    device=final_score.device, dtype=final_score.dtype
                                )
                            p = semantic_fisher_sphere_step(p, final_score, dt=step_size)
                        else:
                            score = getattr(out, "potential_logits", None)
                            if score is None:
                                final_score = v
                                if gp_prior is not None and float(gp_policy_weight) != 0.0:
                                    final_score = final_score + float(gp_policy_weight) * gp_prior.to(
                                        device=final_score.device, dtype=final_score.dtype
                                    )
                                p = semantic_fisher_sphere_step(p, final_score, dt=step_size)
                            else:
                                score_vec = score.squeeze(0)
                                if gp_prior is not None and float(gp_policy_weight) != 0.0:
                                    score_vec = score_vec + float(gp_policy_weight) * gp_prior.to(
                                        device=score_vec.device, dtype=score_vec.dtype
                                    )
                                used_beta = beta_for_update(
                                    p,
                                    score_vec,
                                    mode=update_mode,
                                    beta=beta_value,
                                    target_kl=target_kl,
                                    beta_max=beta_max,
                                    bisection_steps=bisection_steps,
                                )
                                p = closed_form_policy_update(p, score_vec, beta=used_beta)
                                final_score = score_vec
                        if record_diagnostics and record_path:
                            path_diag.append({
                                "update": int(update_idx),
                                "ode_step": int(ode_idx),
                                "lambda": _float((ode_idx + 1) / max(n_ode, 1)),
                                "p_entropy": _entropy(p),
                                "p_top1_mass": _float(p.max()),
                                "velocity_norm": _float(v.norm()),
                                "velocity_abs_max": _float(v.abs().max()),
                                "tangent_error": _float(v.sum().abs()),
                                "beta": _float(used_beta),
                                "integration_method": integration_method,
                            })
                elif integration_method == "closed_form":
                    lam = torch.zeros(1, device=device)
                    out = model(x=x.unsqueeze(0), y=y.unsqueeze(0), B=B.unsqueeze(0),
                                p_lambda=p.unsqueeze(0), lambda_value=lam,
                                action_feats=feats.unsqueeze(0), energies=energies.unsqueeze(0),
                                weights=w.unsqueeze(0), semantic_stats=semantic_stats.unsqueeze(0),
                                gram=gram.unsqueeze(0), beta_value=beta_value,
                                action_mask=torch.ones(1, action_ids.numel(), dtype=torch.bool, device=device))
                    score = getattr(out, "potential_logits", None)
                    v = out.v_pred.squeeze(0)
                    if score is None:
                        # Compatibility for log-rate-only mocks/checkpoints used in tests.
                        final_score_vec = v
                        if gp_prior is not None and float(gp_policy_weight) != 0.0:
                            final_score_vec = final_score_vec + float(gp_policy_weight) * gp_prior.to(
                                device=final_score_vec.device, dtype=final_score_vec.dtype
                            )
                        p = semantic_fisher_sphere_step(p, final_score_vec, dt=step_size)
                        score = final_score_vec.unsqueeze(0)
                        used_beta = beta_value
                    else:
                        score_vec = score.squeeze(0)
                        if gp_prior is not None and float(gp_policy_weight) != 0.0:
                            score_vec = score_vec + float(gp_policy_weight) * gp_prior.to(
                                device=score_vec.device, dtype=score_vec.dtype
                            )
                        used_beta = beta_for_update(
                            p,
                            score_vec,
                            mode=update_mode,
                            beta=beta_value,
                            target_kl=target_kl,
                            beta_max=beta_max,
                            bisection_steps=bisection_steps,
                        )
                        p = closed_form_policy_update(p, score_vec, beta=used_beta)
                    final_score = score.squeeze(0)
                    if record_diagnostics and record_path:
                        path_diag.append({
                            "update": int(update_idx),
                            "lambda": 1.0,
                            "p_entropy": _entropy(p),
                            "p_top1_mass": _float(p.max()),
                            "velocity_norm": _float(v.norm()),
                            "velocity_abs_max": _float(v.abs().max()),
                            "tangent_error": _float(v.sum().abs()),
                            "beta": _float(used_beta),
                            "integration_method": integration_method,
                        })
                else:
                    raise ValueError(f"unknown integration method: {integration_method}")
                if record_diagnostics and record_path:
                    path_diag.append({
                        "update": int(update_idx),
                        "lambda": 1.0,
                        "p_entropy": _entropy(p),
                        "p_top1_mass": _float(p.max()),
                        "update_kl": _float((p * (p.clamp(min=EPS).log() - p_update_start.clamp(min=EPS).log())).sum()),
                        "update_distance": _float((p - p_update_start).abs().sum()),
                        "beta": beta_value,
                        "integration_method": integration_method,
                    })
        if greedy:
            choice_pos = int(p.argmax().detach().cpu().item())
        else:
            choice_pos = int(torch.multinomial(p, 1).detach().cpu().item())
        choice = int(action_ids[choice_pos])
        diag = None
        if record_diagnostics:
            full_best_pos = int(full_rewards.argmax().detach().cpu().item())
            full_best_action = int(full_action_ids[full_best_pos])
            inv_prop = 1.0 / proposal.clamp(min=EPS)
            ess = (inv_prop.sum() ** 2 / inv_prop.square().sum().clamp(min=EPS)).clamp(max=proposal.numel())
            diag = {
                "step": int(step_idx),
                "energy_before": float(trace[-1]),
                "full_action_size": int(support.full_size),
                "support_size": int(action_ids.numel()),
                "support_mode": str(support.mode),
                "full_best_action": _action_record(space, full_best_action),
                "full_best_reward": _float(full_rewards[full_best_pos]),
                "full_best_in_support": _support_contains(action_ids, full_best_action),
                "support_best_reward": _float(one_step_rewards.max()),
                "support_best_reward_gap": _float(full_rewards[full_best_pos] - one_step_rewards.max()),
                "reward_mean": _float(target_rewards.mean()),
                "reward_std": _float(target_rewards.std(unbiased=False)) if target_rewards.numel() else 0.0,
                "reward_min": _float(target_rewards.min()),
                "reward_max": _float(target_rewards.max()),
                "one_step_reward_mean": _float(one_step_rewards.mean()),
                "one_step_reward_std": _float(one_step_rewards.std(unbiased=False)) if one_step_rewards.numel() else 0.0,
                "p_start_entropy": _entropy(p_start),
                "p0_entropy": _entropy(p_start),
                "p_final_entropy": _entropy(p),
                "p_final_top1_mass": _float(p.max()),
                "update_beta": beta_value,
                "update_mode": update_mode,
                "integration_method": integration_method,
                "selected_action": _action_record(space, choice),
                "selected_probability": _float(p[choice_pos]),
                "selected_reward": _float(target_rewards[choice_pos]),
                "selected_one_step_reward": _float(one_step_rewards[choice_pos]),
                "selected_energy": _float(energies[choice_pos]),
                "selected_reward_rank": _rank_desc(target_rewards, choice_pos),
                "selected_one_step_reward_rank": _rank_desc(one_step_rewards, choice_pos),
                "selected_probability_rank": _rank_desc(p, choice_pos),
                "proposal_prob_min": _float(proposal.min()),
                "proposal_prob_max": _float(proposal.max()),
                "correction_weight_max": _float(inv_prop.max()),
                "importance_ess": _float(ess),
                "gp_policy_weight": float(gp_policy_weight),
                "gp_policy_applied": bool(gp_prior is not None and float(gp_policy_weight) != 0.0),
                "path": path_diag,
            }
            if gp_prior is not None:
                gp_vec = gp_prior.to(device=p.device, dtype=p.dtype)
                diag.update({
                    "selected_gp_prior": _float(gp_vec[choice_pos]),
                    "selected_gp_prior_rank": _rank_desc(gp_vec, choice_pos),
                    "gp_prior_top1_action": _action_record(space, int(action_ids[int(gp_vec.argmax().detach().cpu().item())])),
                })
            if advantages is not None:
                diag.update({
                    "advantage_min": _float(advantages.min()),
                    "advantage_max": _float(advantages.max()),
                    "selected_advantage": _float(advantages[choice_pos]),
                    "selected_advantage_rank": _rank_desc(advantages, choice_pos),
                })
            if advantages_eff is not None:
                diag.update({
                    "effective_advantage_min": _float(advantages_eff.min()),
                    "effective_advantage_max": _float(advantages_eff.max()),
                    "selected_effective_advantage": _float(advantages_eff[choice_pos]),
                    "selected_effective_advantage_rank": _rank_desc(advantages_eff, choice_pos),
                })
            if final_score is not None:
                score_vec = final_score.to(device=p.device, dtype=p.dtype)
                diag.update({
                    "predicted_score_min": _float(score_vec.min()),
                    "predicted_score_max": _float(score_vec.max()),
                    "selected_predicted_score": _float(score_vec[choice_pos]),
                    "selected_predicted_score_rank": _rank_desc(score_vec, choice_pos),
                    "predicted_top1_reward_rank": _rank_desc(target_rewards, int(score_vec.argmax().detach().cpu().item())),
                })
            if exact_lograte is not None:
                exact_top = int(exact_lograte.argmax().detach().cpu().item())
                diag.update({
                    "exact_semantic_fisher_top1_reward_rank": _rank_desc(target_rewards, exact_top),
                    "exact_semantic_fisher_top1_action": _action_record(space, int(action_ids[exact_top])),
                })
            if plain_p_target is not None:
                plain_top = int(plain_p_target.argmax().detach().cpu().item())
                diag.update({
                    "plain_fisher_top1_reward_rank": _rank_desc(target_rewards, plain_top),
                    "plain_fisher_top1_action": _action_record(space, int(action_ids[plain_top])),
                })
            if rollout_stats is not None:
                per_action = rollout_stats.get("per_action", [])
                selected_rollout = per_action[choice_pos] if choice_pos < len(per_action) else {}
                trajectory_stats = rollout_stats.get("trajectory_stats", {})
                diag.update({
                    "n_rollout_evaluated": int(rollout_stats.get("n_rollout_evaluated", 0)),
                    "rollout_eval_fraction": float(rollout_stats.get("n_rollout_evaluated", 0)) / max(int(action_ids.numel()), 1),
                    "rollout_reward_mean": _float(target_rewards.mean()),
                    "rollout_reward_std": _float(target_rewards.std(unbiased=False)) if target_rewards.numel() else 0.0,
                    "one_step_rollout_corr": float(rollout_stats.get("one_step_rollout_corr", 0.0)),
                    "one_step_rollout_top1_agreement": bool(rollout_stats.get("top1_agreement", False)),
                    "one_step_top1_action_id": int(rollout_stats.get("one_step_top1_action", -1)),
                    "rollout_top1_action_id": int(rollout_stats.get("rollout_top1_action", -1)),
                    "selected_rollout_rank": int(selected_rollout.get("rollout_rank", -1)),
                    "selected_rank_shift": int(selected_rollout.get("rank_shift", 0)),
                    "selected_rollout_evaluated": bool(selected_rollout.get("rollout_evaluated", False)),
                    "selected_rollout_best_score": float(selected_rollout.get("best_score", 0.0)),
                    "selected_rollout_best_final_energy": float(selected_rollout.get("best_final_energy", 0.0)),
                    "selected_rollout_best_final_r2": float(selected_rollout.get("best_final_r2", 0.0)),
                    "trajectory_target_mode": str(trajectory_stats.get("target_mode", "")),
                    "trajectory_num_samples": int(trajectory_stats.get("num_trajectories", 0)),
                    "trajectory_oracle_r2": float(trajectory_stats.get("candidate_oracle_r2", 0.0)),
                    "trajectory_oracle_reward": float(trajectory_stats.get("candidate_oracle_reward", 0.0)),
                })
            if p_target is not None:
                kl = (p_target * (p_target.clamp(min=EPS).log() - p_start.clamp(min=EPS).log())).sum()
                p1_top_pos = int(p_target.argmax().detach().cpu().item())
                diag.update({
                    "p_target_entropy": _entropy(p_target),
                    "kl_p_target_p_start": _float(kl),
                    "p_target_top1_mass": _float(p_target.max()),
                    "p1_entropy": _entropy(p_target),
                    "kl_p1_p0": _float(kl),
                    "p1_top1_mass": _float(p_target.max()),
                    "p1_top1_action": _action_record(space, int(action_ids[p1_top_pos])),
                })
        state = execu.execute_symbolic(state, choice)
        trace.append(_residual_energy(state, x, y, proj))
        if diag is not None:
            diag["energy_after"] = float(trace[-1])
            diag["energy_delta"] = float(trace[-2] - trace[-1])
            diagnostics.append(diag)
    return RolloutResult(state=state, energy_trace=trace, steps=len(trace) - 1,
                         diagnostics=diagnostics)


def rollout_random(x, y, num_vars, K, ops_ids, device, max_steps: int = 16,
                   energy_cfg=None, seed: int = 0) -> RolloutResult:
    rng = random.Random(seed)
    space = ActionSpace(K, ops_ids)
    execu = ActionExecutor(space)
    proj = ProjectionBackend((energy_cfg or ActionEnergyConfig()).projection,
                             (energy_cfg or ActionEnergyConfig()).rho)
    state = init_register_state(num_vars, K, device)
    x = x.to(device); y = y.to(device)
    trace = [_residual_energy(state, x, y, proj)]
    for _ in range(max_steps):
        ids = space.valid_actions(state)
        if ids.numel() == 0:
            break
        choice = int(ids[rng.randrange(ids.numel())])
        state = execu.execute_symbolic(state, choice)
        trace.append(_residual_energy(state, x, y, proj))
    return RolloutResult(state=state, energy_trace=trace, steps=len(trace) - 1)
