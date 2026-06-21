import torch
import random
import json

from semflow_sr.data.synthetic_generator import GenConfig
from semflow_sr.models.semantic_transformer import SemanticTransformer, SemanticTransformerConfig
from semflow_sr.actions.action_space import ActionSpace
from semflow_sr.registers.executor import evaluate_register_state
from semflow_sr.registers.state import init_register_state
from semflow_sr.sr.ops import NAME_TO_ID
from semflow_sr.semantics.energy import ActionEnergy, ActionEnergyConfig
from semflow_sr.path_posterior.action_support import (
    STOP_ACTION_ID,
    action_features_with_stop,
    action_semantic_effects_with_stop,
    append_stop_action,
)
from semflow_sr.path_posterior.dataset import (
    PathPosteriorBuildConfig,
    build_path_posterior_dataset,
)
from semflow_sr.path_posterior.sampler import ActionPathSampler
from semflow_sr.path_posterior.target_sampler import (
    CachedTrajectoryFitnessTargetSampler,
    FutureGroupTargetConfig,
    FutureGroupTargetSampler,
    GPCandidateFitnessTargetSampler,
    PriorConfig,
    build_p_init,
    group_exp_target,
    make_target_sampler,
)
from semflow_sr.flow.semantic_fisher import (
    integrate_semantic_fisher_endpoint_path,
    semantic_fisher_lograte,
)


def test_sffm_prior_stop_bias_grows_with_construction_step():
    action_ids = torch.tensor([10, 20, STOP_ACTION_ID])

    early = build_p_init(action_ids, step=0, cfg=PriorConfig(stop_bias_base=-3.0, stop_bias_slope=0.5))
    late = build_p_init(action_ids, step=6, cfg=PriorConfig(stop_bias_base=-3.0, stop_bias_slope=0.5))

    assert torch.allclose(early.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(late.sum(), torch.tensor(1.0), atol=1e-6)
    assert late[-1] > early[-1]
    assert early[0] == early[1]


def test_sffm_endpoint_teacher_uses_lambda_dependent_logratio():
    p_start = torch.tensor([0.5, 0.5])
    q_hat = torch.tensor([0.9, 0.1])
    gram = torch.zeros(2, 2)

    path = integrate_semantic_fisher_endpoint_path(
        p_start,
        q_hat,
        gram,
        beta=1.0,
        gamma=0.0,
        steps=2,
    )

    assert len(path.logrates) == 2
    assert path.policies[-1][0] > p_start[0]
    assert path.logrates[1][0].abs() < path.logrates[0][0].abs()


def test_sffm_fixed_potential_teacher_reuses_q_over_pinit_potential():
    p_start = torch.tensor([0.5, 0.5])
    q_hat = torch.tensor([0.9, 0.1])
    gram = torch.zeros(2, 2)

    path = integrate_semantic_fisher_endpoint_path(
        p_start,
        q_hat,
        gram,
        beta=1.0,
        gamma=0.0,
        steps=2,
        q_smoothing=0.0,
        teacher_mode="fixed_potential_from_q",
    )

    phi = q_hat.clamp_min(1e-12).log() - p_start.clamp_min(1e-12).log()
    expected_first = semantic_fisher_lograte(p_start, phi, gram, beta=1.0, gamma=0.0)
    expected_second = semantic_fisher_lograte(path.policies[1], phi, gram, beta=1.0, gamma=0.0)

    assert torch.allclose(path.logrates[0], expected_first, atol=1e-6)
    assert torch.allclose(path.logrates[1], expected_second, atol=1e-6)


def test_group_exp_target_preserves_group_standardized_score_gaps():
    scores = torch.tensor([10.0, 0.0, -10.0])
    p_init = torch.tensor([0.2, 0.3, 0.5])

    q_hat, advantages = group_exp_target(
        scores,
        p_init,
        eta=1.5,
        smoothing=0.0,
        advantage_clip=5.0,
    )

    expected_adv = (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1e-6)
    expected_q = p_init * torch.exp(1.5 * expected_adv)
    expected_q = expected_q / expected_q.sum()

    assert torch.allclose(advantages, expected_adv, atol=1e-6)
    assert torch.allclose(q_hat, expected_q, atol=1e-6)
    assert q_hat[0] / q_hat[1] > q_hat[1] / q_hat[2]


def test_make_target_sampler_exposes_one_step_group_advantage():
    gen = GenConfig(num_vars=1, max_depth=2, K=5, probe_size=8, ops=("add", "mul", "square"))
    space = ActionSpace(gen.K, [NAME_TO_ID[o] for o in gen.ops])
    sampler = make_target_sampler(
        "one_step_group_advantage",
        space,
        energy_cfg=ActionEnergyConfig(lambda_op=0.01),
        future_cfg=FutureGroupTargetConfig(),
    )

    assert sampler.name == "one_step_group_advantage"


def test_future_group_l3_target_sampler_returns_dense_target_shape():
    gen = GenConfig(num_vars=1, max_depth=2, K=5, probe_size=8, ops=("add", "mul", "square"))
    _, _, x, y = __import__("semflow_sr.data.synthetic_generator", fromlist=["generate_trace_task"]).generate_trace_task(gen, random.Random(2))
    state = init_register_state(gen.num_vars, gen.K, device=x.device)
    space = ActionSpace(gen.K, [NAME_TO_ID[o] for o in gen.ops])
    action_ids = append_stop_action(space.valid_actions(state)[:3])
    p_init = build_p_init(action_ids, step=0, cfg=PriorConfig(stop_bias_base=-2.0, stop_bias_slope=0.5))
    sampler = FutureGroupTargetSampler(
        space,
        energy_cfg=ActionEnergyConfig(lambda_op=0.0),
        cfg=FutureGroupTargetConfig(rollout_depth=2, rollouts_per_action=1, topk=1, max_rollout_support=3),
    )

    target = sampler.build_target(
        state=state,
        action_ids=action_ids,
        p_init=p_init,
        x=x,
        y=y,
        rng=random.Random(0),
    )

    assert torch.equal(target.action_ids, action_ids.cpu())
    assert torch.allclose(target.q_hat.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(target.q_hat >= 0)
    assert target.target_scores.shape == action_ids.shape
    assert target.diagnostics["target_sampler_name"] == "future_group_l3"


def test_stop_action_has_zero_semantic_effect_and_valid_features():
    gen = GenConfig(num_vars=1, max_depth=2, K=5, probe_size=8, ops=("add", "mul", "square"))
    _, _, x, y = __import__("semflow_sr.data.synthetic_generator", fromlist=["generate_trace_task"]).generate_trace_task(gen, __import__("random").Random(0))
    state = init_register_state(gen.num_vars, gen.K, device=x.device)
    space = ActionSpace(gen.K, [NAME_TO_ID[o] for o in gen.ops])
    B = evaluate_register_state(state, x)
    action_ids = append_stop_action(space.valid_actions(state)[:3])

    feats = action_features_with_stop(space, state, action_ids)
    effect = action_semantic_effects_with_stop(
        ActionEnergy(space, ActionEnergyConfig(lambda_op=0.0)),
        B,
        y,
        action_ids,
    )

    stop_idx = (action_ids == STOP_ACTION_ID).nonzero(as_tuple=False).item()
    assert feats.shape[0] == action_ids.numel()
    assert torch.allclose(effect.xi[stop_idx], torch.zeros_like(effect.xi[stop_idx]))
    assert torch.allclose(effect.residual_next[stop_idx], effect.residual_current)
    assert torch.allclose(effect.gram[stop_idx], torch.zeros_like(effect.gram[stop_idx]))
    assert torch.allclose(effect.gram[:, stop_idx], torch.zeros_like(effect.gram[:, stop_idx]))


def test_action_sampler_caps_real_support_before_adding_stop():
    gen = GenConfig(num_vars=1, max_depth=2, K=5, probe_size=8, ops=("add", "mul", "square"))
    _, _, x, y = __import__("semflow_sr.data.synthetic_generator", fromlist=["generate_trace_task"]).generate_trace_task(gen, __import__("random").Random(1))
    state = init_register_state(gen.num_vars, gen.K, device=x.device)
    space = ActionSpace(gen.K, [NAME_TO_ID[o] for o in gen.ops])
    sampler = ActionPathSampler(space, seed=0, enable_stop=True, max_support_size=2)

    trajectories = sampler.sample(
        task_id="toy",
        initial_state=state,
        x=x,
        y=y,
        model=None,
        num_trajectories=1,
        max_steps=1,
    )

    support = trajectories[0].decisions[0].action_ids
    assert STOP_ACTION_ID in support.tolist()
    assert (support != STOP_ACTION_ID).sum().item() <= 2


def test_action_sampler_reward_topk_support_keeps_full_best_action():
    gen = GenConfig(num_vars=1, max_depth=2, K=5, probe_size=8, ops=("add", "mul", "square"))
    _, _, x, y = __import__("semflow_sr.data.synthetic_generator", fromlist=["generate_trace_task"]).generate_trace_task(gen, __import__("random").Random(11))
    state = init_register_state(gen.num_vars, gen.K, device=x.device)
    space = ActionSpace(gen.K, [NAME_TO_ID[o] for o in gen.ops])
    energy = ActionEnergy(space, ActionEnergyConfig(lambda_op=0.0))
    B = torch.nan_to_num(evaluate_register_state(state, x))
    full_action_ids = space.valid_actions(state).to(device=x.device)
    full_rewards = energy.rewards(B, y, full_action_ids)
    full_best_action = int(full_action_ids[int(full_rewards.argmax().item())].item())
    sampler = ActionPathSampler(
        space,
        seed=0,
        enable_stop=False,
        max_support_size=2,
        support_mode="reward_topk_random",
        support_topk=1,
    )

    trajectories = sampler.sample(
        task_id="toy",
        initial_state=state,
        x=x,
        y=y,
        model=None,
        num_trajectories=1,
        max_steps=1,
    )

    support = trajectories[0].decisions[0].action_ids
    assert full_best_action in support.tolist()
    assert support.numel() <= 2


def test_probability_shape_dataset_uses_q_hat_endpoint_not_local_reward():
    gen = GenConfig(
        num_vars=1,
        max_depth=2,
        K=5,
        probe_size=8,
        ops=("add", "mul", "square"),
    )
    model = SemanticTransformer(SemanticTransformerConfig(d=1, K=5, hidden=16, row_layers=1, heads=1))
    dataset = build_path_posterior_dataset(
        gen,
        num_tasks=1,
        behavior_model=model,
        seed=0,
        cfg=PathPosteriorBuildConfig(
            target_mode="multi_step_group_advantage",
            num_trajectories=2,
            max_states_per_task=1,
            max_steps=2,
            gamma=0.1,
            max_support_size=4,
            rollout_depth=2,
            rollouts_per_action=1,
            rollout_topk=1,
            max_rollout_support=3,
        ),
    )

    rec = dataset[0]
    assert rec["action_ids"].ndim == 1
    assert torch.allclose(rec["p_start"].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(rec["plain_p_target"].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(rec["one_step_rewards"] == 0)
    assert "target_scores" in rec
    assert "target_counts" in rec
    assert "target_sampler_id" in rec
    assert "target_sampler_runtime_sec" in rec
    assert "target_kl_q_pinit" in rec
    assert "target_score_gap" in rec
    q_eps = 0.999 * rec["plain_p_target"] + 0.001 * rec["p_start"]
    q_eps = q_eps / q_eps.sum()
    assert torch.allclose(
        rec["advantages"],
        torch.log(q_eps.clamp(min=1e-12)) - torch.log(rec["p_lambda"].clamp(min=1e-12)),
        atol=1e-5,
    )


def test_probability_shape_dataset_supports_one_step_target_group():
    gen = GenConfig(
        num_vars=1,
        max_depth=2,
        K=5,
        probe_size=8,
        ops=("add", "mul", "square"),
    )
    model = SemanticTransformer(SemanticTransformerConfig(d=1, K=5, hidden=16, row_layers=1, heads=1))
    dataset = build_path_posterior_dataset(
        gen,
        num_tasks=1,
        behavior_model=model,
        seed=1,
        cfg=PathPosteriorBuildConfig(
            target_mode="one_step_group_advantage",
            num_trajectories=2,
            max_states_per_task=1,
            max_steps=2,
            gamma=0.1,
            max_support_size=4,
        ),
    )

    rec = dataset[0]
    assert rec["action_ids"].ndim == 1
    assert int(rec["target_sampler_id"].item()) == 6
    assert torch.allclose(rec["plain_p_target"].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(rec["plain_p_target"] >= 0)
    assert torch.all(rec["one_step_rewards"] == 0)


def test_probability_shape_dataset_records_oracle_target_teacher_layers():
    gen = GenConfig(
        num_vars=1,
        max_depth=2,
        K=5,
        probe_size=8,
        ops=("add", "mul", "square"),
    )
    model = SemanticTransformer(SemanticTransformerConfig(d=1, K=5, hidden=16, row_layers=1, heads=1))
    dataset = build_path_posterior_dataset(
        gen,
        num_tasks=1,
        behavior_model=model,
        seed=3,
        energy_cfg=ActionEnergyConfig(lambda_op=0.0),
        cfg=PathPosteriorBuildConfig(
            target_mode="multi_step_group_advantage",
            num_trajectories=2,
            max_states_per_task=1,
            max_steps=2,
            gamma=0.1,
            max_support_size=4,
            rollout_depth=1,
            rollouts_per_action=1,
            rollout_topk=1,
            max_rollout_support=3,
        ),
    )

    rec = dataset[0]
    for key in (
        "full_best_reward",
        "support_best_reward",
        "support_best_reward_gap",
        "full_best_in_support",
        "target_top1_reward_rank",
        "teacher_top1_reward_rank",
        "target_top1_action_id",
        "teacher_top1_action_id",
    ):
        assert key in rec
    assert rec["support_best_reward"] >= rec["full_best_reward"] - rec["support_best_reward_gap"].abs() - 1e-6
    assert rec["target_top1_reward_rank"] >= 1
    assert rec["teacher_top1_reward_rank"] >= 1


def test_cached_trajectory_fitness_target_sampler_returns_probability_shape(tmp_path):
    gen = GenConfig(num_vars=1, max_depth=2, K=5, probe_size=8, ops=("add", "mul", "square"))
    _, _, x, y = __import__("semflow_sr.data.synthetic_generator", fromlist=["generate_trace_task"]).generate_trace_task(gen, random.Random(3))
    state = init_register_state(gen.num_vars, gen.K, device=x.device)
    space = ActionSpace(gen.K, [NAME_TO_ID[o] for o in gen.ops])
    action_ids = append_stop_action(space.valid_actions(state)[:3])
    good_action = int(action_ids[1].item())
    weak_action = int(action_ids[0].item())
    cache_path = tmp_path / "cached_trajectories.jsonl"
    cache_path.write_text(
        "\n".join([
            json.dumps({"actions": [weak_action], "fitness": 0.1}),
            json.dumps({"actions": [good_action], "fitness": 2.0}),
            json.dumps({"actions": [good_action], "fitness": 1.5}),
        ])
    )
    p_init = build_p_init(action_ids, step=0)
    sampler = CachedTrajectoryFitnessTargetSampler(
        space,
        cfg=FutureGroupTargetConfig(cache_path=str(cache_path), shape_samples=16),
    )

    target = sampler.build_target(
        state=state,
        action_ids=action_ids,
        p_init=p_init,
        x=x,
        y=y,
        rng=random.Random(0),
    )

    assert torch.allclose(target.q_hat.sum(), torch.tensor(1.0), atol=1e-6)
    assert target.q_hat.numel() == action_ids.numel()
    assert target.diagnostics["target_sampler_name"] == "cached_trajectory_fitness"
    assert target.q_hat[1] > target.q_hat[0]
    assert target.target_counts.sum() > 1.0


def test_gp_candidate_fitness_target_sampler_samples_simplex_point_from_likelihood(tmp_path):
    gen = GenConfig(num_vars=1, max_depth=2, K=5, probe_size=8, ops=("add", "mul", "square"))
    _, _, x, y = __import__("semflow_sr.data.synthetic_generator", fromlist=["generate_trace_task"]).generate_trace_task(gen, random.Random(4))
    state = init_register_state(gen.num_vars, gen.K, device=x.device)
    space = ActionSpace(gen.K, [NAME_TO_ID[o] for o in gen.ops])
    action_ids = append_stop_action(space.valid_actions(state)[:3])
    likely_action = int(action_ids[2].item())
    unlikely_action = int(action_ids[0].item())
    population_path = tmp_path / "gp_population.json"
    population_path.write_text(json.dumps({
        "population": [
            {"actions": [unlikely_action], "fitness": 10.0, "gp_logprob": -20.0},
            {"actions": [likely_action], "fitness": 1.0, "gp_logprob": 0.0},
        ]
    }))
    p_init = build_p_init(action_ids, step=0)
    sampler = GPCandidateFitnessTargetSampler(
        space,
        cfg=FutureGroupTargetConfig(
            gp_population_path=str(population_path),
            shape_samples=64,
            gp_likelihood_weight=1.0,
            gp_fitness_weight=0.0,
        ),
    )

    target = sampler.build_target(
        state=state,
        action_ids=action_ids,
        p_init=p_init,
        x=x,
        y=y,
        rng=random.Random(1),
    )

    assert torch.allclose(target.q_hat.sum(), torch.tensor(1.0), atol=1e-6)
    assert target.diagnostics["target_sampler_name"] == "gp_candidate_fitness"
    assert target.q_hat[2] > target.q_hat[0]
    assert target.target_counts.sum() >= 1.0


def test_make_target_sampler_exposes_only_four_experiment_settings(tmp_path):
    gen = GenConfig(num_vars=1, max_depth=2, K=5, probe_size=8, ops=("add", "mul", "square"))
    space = ActionSpace(gen.K, [NAME_TO_ID[o] for o in gen.ops])
    cache_path = tmp_path / "cache.jsonl"
    cache_path.write_text(json.dumps({"actions": [0], "fitness": 1.0}) + "\n")
    cfg = FutureGroupTargetConfig(cache_path=str(cache_path), gp_population_path=str(cache_path))

    assert make_target_sampler("one_step_group_advantage", space, energy_cfg=None, future_cfg=cfg).name == "one_step_group_advantage"
    assert make_target_sampler("multi_step_group_advantage", space, energy_cfg=None, future_cfg=cfg).name == "multi_step_group_advantage"
    assert make_target_sampler("cached_trajectory_fitness", space, energy_cfg=None, future_cfg=cfg).name == "cached_trajectory_fitness"
    assert make_target_sampler("gp_candidate_fitness", space, energy_cfg=None, future_cfg=cfg).name == "gp_candidate_fitness"
    for removed_mode in ("one_step", "future_group_l3", "importance_sampling", "mcmc_shape"):
        try:
            make_target_sampler(removed_mode, space, energy_cfg=None, future_cfg=cfg)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{removed_mode} should not be exposed as a formal TargetSampler setting")
