import random

import torch

import semflow_sr.edge_flow.circuit_sampler as circuit_sampler_module
from semflow_sr.edge_flow.circuit_sampler import CircuitSampler
from semflow_sr.edge_flow.edge_distribution import EdgeDistribution
from semflow_sr.edge_flow.flow_teacher import build_fisher_slerp_record
from semflow_sr.edge_flow.projection import project_elites_to_edge_target
from semflow_sr.edge_flow.reward import RewardConfig, evaluate_expression_rewards
from semflow_sr.edge_flow.template import RegisterOperatorTemplate
from semflow_sr.sr.ast import Expr
from semflow_sr.sr.ops import NAME_TO_ID
from semflow_sr.sr.printer import to_string


def test_edge_template_sampling_projection_and_teacher_flow():
    x = torch.linspace(-1.0, 1.0, 32).unsqueeze(1)
    y = x[:, 0] ** 2
    template = RegisterOperatorTemplate(
        num_vars=1,
        num_registers=4,
        num_layers=2,
        primitives=("add", "mul", "square"),
        mixture_modes=2,
    )
    theta0 = EdgeDistribution.uniform(template)
    sampler = CircuitSampler(template)

    samples = sampler.sample(theta0, batch_size=12, rng=random.Random(0))
    assert len(samples) == 12
    assert all(sample.expression is not None for sample in samples)
    assert all(set(sample.edge_choices) == set(template.group_ids) for sample in samples)
    assert {sample.mode for sample in samples} == {0, 1}

    reward = evaluate_expression_rewards(samples, x, y, RewardConfig(complexity_weight=0.001))
    assert reward.rewards.shape == (12,)
    assert reward.valid_mask.any()

    theta_star, diagnostics = project_elites_to_edge_target(
        theta0,
        samples,
        reward.rewards,
        reward.valid_mask,
        elite_k=4,
        smoothing=0.05,
    )
    assert set(theta_star.group_probs) == set(theta0.group_probs)
    assert torch.allclose(theta_star.mixture_probs.sum(), torch.tensor(1.0), atol=1e-6)
    for group in template.groups:
        probs = theta_star.group_probs[group.group_id]
        assert probs.shape == (template.mixture_modes, group.num_candidates)
        assert torch.allclose(probs.sum(dim=1), torch.ones(template.mixture_modes), atol=1e-5)
    assert diagnostics["target_ess"] > 0
    assert sum(diagnostics["per_mode_elite_count"]) == 4

    record = build_fisher_slerp_record(theta0, theta_star, lam=0.4)
    assert torch.allclose(record.theta_lambda.mixture_probs.sum(), torch.tensor(1.0), atol=1e-6)
    assert record.zdot_mixture.shape == theta0.mixture_probs.shape
    for group in template.groups:
        z = record.z_lambda_groups[group.group_id]
        zdot = record.zdot_groups[group.group_id]
        assert z.shape == zdot.shape
        assert torch.allclose((z * zdot).sum(dim=1), torch.zeros(template.mixture_modes), atol=1e-5)


def test_non_stratified_mode_sampling_uses_provided_rng(monkeypatch):
    template = RegisterOperatorTemplate(
        num_vars=1,
        num_registers=3,
        num_layers=1,
        primitives=("add",),
        mixture_modes=2,
    )
    theta = EdgeDistribution.uniform(template)
    theta.mixture_probs = torch.tensor([0.5, 0.5])
    sampler = CircuitSampler(template)

    monkeypatch.setattr(circuit_sampler_module.random, "random", lambda: 0.99)
    modes_a = [
        sample.mode
        for sample in sampler.sample(theta, batch_size=12, rng=random.Random(7), mode_policy="sample")
    ]
    monkeypatch.setattr(circuit_sampler_module.random, "random", lambda: 0.01)
    modes_b = [
        sample.mode
        for sample in sampler.sample(theta, batch_size=12, rng=random.Random(7), mode_policy="sample")
    ]

    assert modes_a == modes_b


def test_per_mode_projection_keeps_elites_from_each_mode():
    template = RegisterOperatorTemplate(
        num_vars=1,
        num_registers=3,
        num_layers=1,
        primitives=("add",),
        mixture_modes=2,
    )
    theta0 = EdgeDistribution.uniform(template)
    sampler = CircuitSampler(template)
    samples = sampler.sample(theta0, batch_size=8, rng=random.Random(0))
    rewards = torch.tensor([
        10.0 if sample.mode == 0 else 1.0
        for sample in samples
    ])
    valid = torch.ones(len(samples), dtype=torch.bool)

    _, global_diag = project_elites_to_edge_target(
        theta0,
        samples,
        rewards,
        valid,
        elite_k=2,
        smoothing=0.01,
        projection_mode="global_topk",
    )
    _, per_mode_diag = project_elites_to_edge_target(
        theta0,
        samples,
        rewards,
        valid,
        elite_k=2,
        smoothing=0.01,
        projection_mode="per_mode_topk",
    )

    assert global_diag["per_mode_elite_count"] == [2, 0]
    assert per_mode_diag["per_mode_elite_count"] == [2, 2]
    assert per_mode_diag["projection_mode"] == "per_mode_topk"


def test_symbolic_printer_falls_back_for_protected_division_by_zero():
    expr = Expr.op(NAME_TO_ID["protected_div"], (Expr.const(1.0), Expr.const(0.0)))

    text = to_string(expr, num_vars=1, simplify=True)

    assert "protected_div" in text
