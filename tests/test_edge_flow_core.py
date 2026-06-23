import random

import pytest
import torch

import semflow_sr.edge_flow.circuit_sampler as circuit_sampler_module
from semflow_sr.edge_flow.circuit_sampler import CircuitSample, CircuitSampler
from semflow_sr.edge_flow.conditional import (
    ConditionalEdgeFlowConfig,
    ConditionalEdgeFlowModel,
    ConditionalEdgeFlowSampler,
    conditional_elite_policy_loss,
)
from semflow_sr.edge_flow.benchmark import token_sequence_metrics
from semflow_sr.edge_flow.edge_distribution import EdgeDistribution
from semflow_sr.edge_flow.flow_teacher import build_fisher_slerp_record
from semflow_sr.edge_flow.path_compiler import compile_formula_to_csef_sample
from semflow_sr.edge_flow.projection import project_elites_to_edge_target
from semflow_sr.edge_flow.reward import RewardConfig, evaluate_expression_rewards
from semflow_sr.edge_flow.template import RegisterOperatorTemplate
from semflow_sr.sr.ast import Expr
from semflow_sr.sr.ops import NAME_TO_ID
from semflow_sr.sr.parser import parse_formula
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


def test_conditional_sampler_unrolls_shared_cell_with_sparse_head_policy_and_ode():
    x = torch.linspace(-1.0, 1.0, 24).unsqueeze(1)
    y = x[:, 0] + x[:, 0] ** 2
    template = RegisterOperatorTemplate(
        num_vars=1,
        num_registers=4,
        num_layers=2,
        primitives=("add", "mul", "square"),
        mixture_modes=1,
    )
    model = ConditionalEdgeFlowModel(ConditionalEdgeFlowConfig(num_vars=1, hidden=32, head_terms=3))
    sampler = ConditionalEdgeFlowSampler(template, model, method="policy", flow_steps=2)

    policy_samples = sampler.sample(x, y, batch_size=5, rng=random.Random(1))
    ode_samples = ConditionalEdgeFlowSampler(template, model, method="ode", flow_steps=2).sample(
        x,
        y,
        batch_size=5,
        rng=random.Random(1),
    )

    assert len(policy_samples) == 5
    assert len(ode_samples) == 5
    assert all(len(sample.head_terms) == 3 for sample in policy_samples)
    assert all(sample.log_prob_tensor is not None for sample in policy_samples)
    assert all(torch.isfinite(sample.log_prob_tensor) for sample in policy_samples)
    assert any(key.startswith("HEAD:TERM") for key in policy_samples[0].edge_choices)
    assert any(":ARG" in key for key in policy_samples[0].edge_choices)
    assert policy_samples[0].diagnostics["sampler_method"] == "policy"
    assert ode_samples[0].diagnostics["sampler_method"] == "ode"


def test_linear_head_reward_fits_multiple_terms_by_default():
    x = torch.linspace(-1.0, 1.0, 48).unsqueeze(1)
    y = x[:, 0] + x[:, 0] ** 2
    term_x = Expr.var(0)
    term_x2 = Expr.op(NAME_TO_ID["square"], (Expr.var(0),))
    sample = CircuitSample(
        sample_id=0,
        mode=0,
        edge_choices={},
        expression=Expr.op(NAME_TO_ID["add"], (term_x, term_x2)),
        log_prob=0.0,
        complexity=3,
        head_terms=(term_x, term_x2),
    )

    rewards = evaluate_expression_rewards([sample], x, y, RewardConfig(complexity_weight=0.0))

    assert rewards.r2[0].item() > 0.999
    assert torch.allclose(rewards.rewards[0], rewards.r2[0], atol=1e-6)
    assert rewards.affine_coef.shape == (1, 3)
    assert rewards.selected_term_index.tolist() == [-1]
    assert torch.allclose(rewards.affine_coef[0], torch.tensor([1.0, 1.0, 0.0]), atol=1e-3)
    assert rewards.head_coef_nonzero_count.tolist() == [2]
    assert rewards.best_raw_term_r2[0].item() < rewards.r2[0].item()
    assert rewards.fitted_head_gain[0].item() > 0.0


def test_selector_head_reward_can_still_use_only_one_term_for_ablation():
    x = torch.linspace(-1.0, 1.0, 48).unsqueeze(1)
    y = x[:, 0] ** 2
    term_x = Expr.var(0)
    term_x2 = Expr.op(NAME_TO_ID["square"], (Expr.var(0),))
    sample = CircuitSample(
        sample_id=0,
        mode=0,
        edge_choices={},
        expression=Expr.op(NAME_TO_ID["add"], (term_x, term_x2)),
        log_prob=0.0,
        complexity=3,
        head_terms=(term_x, term_x2),
    )

    rewards = evaluate_expression_rewards([sample], x, y, RewardConfig(complexity_weight=0.0, head_fit_mode="selector"))

    assert rewards.r2[0].item() > 0.999
    assert rewards.affine_coef.shape == (1, 3)
    assert rewards.selected_term_index.tolist() == [1]
    assert abs(float(rewards.affine_coef[0, 0])) < 1e-4
    assert torch.allclose(rewards.affine_coef[0, 1:], torch.tensor([1.0, 0.0]), atol=1e-3)


def test_conditional_elite_policy_loss_backpropagates_through_sample_log_probs():
    x = torch.linspace(-1.0, 1.0, 24).unsqueeze(1)
    y = x[:, 0] + x[:, 0] ** 2
    template = RegisterOperatorTemplate(
        num_vars=1,
        num_registers=4,
        num_layers=1,
        primitives=("add", "mul", "square"),
        mixture_modes=1,
    )
    model = ConditionalEdgeFlowModel(ConditionalEdgeFlowConfig(num_vars=1, hidden=32, head_terms=2))
    samples = ConditionalEdgeFlowSampler(template, model, method="policy", flow_steps=1).sample(
        x,
        y,
        batch_size=8,
        rng=random.Random(2),
    )
    rewards = evaluate_expression_rewards(samples, x, y, RewardConfig(complexity_weight=0.001))

    loss, metrics = conditional_elite_policy_loss(
        samples,
        rewards.rewards,
        rewards.valid_mask,
        elite_k=3,
        entropy_bonus=0.01,
        unique_elites=True,
    )
    loss.backward()

    grad_norm = sum(
        float(param.grad.abs().sum().item())
        for param in model.parameters()
        if param.grad is not None
    )
    assert torch.isfinite(loss)
    assert metrics["conditional_elite_count"] == 3
    assert "entropy_bonus" in metrics
    assert grad_norm > 0.0


def test_conditional_elite_policy_loss_is_per_decision_normalized():
    samples = [
        CircuitSample(
            sample_id=0,
            mode=0,
            edge_choices={f"g{i}": 0 for i in range(50)},
            expression=Expr.var(0),
            log_prob=-100.0,
            complexity=1,
            log_prob_tensor=torch.tensor(-100.0, requires_grad=True),
            diagnostics={"decision_count": 50},
        ),
        CircuitSample(
            sample_id=1,
            mode=0,
            edge_choices={f"g{i}": 0 for i in range(50)},
            expression=Expr.var(0),
            log_prob=-50.0,
            complexity=1,
            log_prob_tensor=torch.tensor(-50.0, requires_grad=True),
            diagnostics={"decision_count": 50},
        ),
    ]
    rewards = torch.tensor([0.1, 1.0])
    valid = torch.ones(2, dtype=torch.bool)

    loss, metrics = conditional_elite_policy_loss(samples, rewards, valid, elite_k=2)

    assert torch.allclose(loss, torch.tensor(1.5))
    assert metrics["loss"] == metrics["loss_per_decision"]


def test_conditional_elite_policy_loss_uses_active_ancestry_when_available():
    sample = CircuitSample(
        sample_id=0,
        mode=0,
        edge_choices={f"g{i}": 0 for i in range(50)},
        expression=Expr.var(0),
        log_prob=-100.0,
        complexity=1,
        log_prob_tensor=torch.tensor(-100.0, requires_grad=True),
        active_log_prob_tensor=torch.tensor(-3.0, requires_grad=True),
        diagnostics={"decision_count": 50, "active_decision_count": 1},
    )

    loss, metrics = conditional_elite_policy_loss(
        [sample],
        torch.tensor([1.0]),
        torch.ones(1, dtype=torch.bool),
        elite_k=1,
    )

    assert torch.allclose(loss, torch.tensor(3.0))
    assert metrics["active_ancestry_loss"] is True
    assert metrics["active_decision_count_mean"] == 1.0


def test_conditional_elite_policy_loss_adds_ground_truth_elite_and_diversifies():
    expr_a = Expr.var(0)
    expr_b = Expr.op(NAME_TO_ID["square"], (Expr.var(0),))
    samples = [
        CircuitSample(
            sample_id=0,
            mode=0,
            edge_choices={"a": 0},
            expression=expr_a,
            log_prob=-1.0,
            complexity=1,
            log_prob_tensor=torch.tensor(-1.0, requires_grad=True),
            diagnostics={"decision_count": 1},
        ),
        CircuitSample(
            sample_id=1,
            mode=0,
            edge_choices={"a": 0},
            expression=expr_a,
            log_prob=-2.0,
            complexity=1,
            log_prob_tensor=torch.tensor(-2.0, requires_grad=True),
            diagnostics={"decision_count": 1},
        ),
    ]
    gt = CircuitSample(
        sample_id=2,
        mode=0,
        edge_choices={"gt": 0},
        expression=expr_b,
        log_prob=-3.0,
        complexity=2,
        log_prob_tensor=torch.tensor(-3.0, requires_grad=True),
        diagnostics={"decision_count": 1, "is_gt_elite": True},
    )

    loss, metrics = conditional_elite_policy_loss(
        samples,
        torch.tensor([10.0, 9.0]),
        torch.ones(2, dtype=torch.bool),
        elite_k=2,
        unique_elites=True,
        gt_samples=[gt],
    )

    assert torch.isfinite(loss)
    assert metrics["conditional_elite_count"] == 2
    assert metrics["gt_elite_count"] == 1
    assert metrics["conditional_unique_elite_count"] == 2


def test_conditional_model_uses_register_root_pair_decisions_without_semantic_stat_features():
    model = ConditionalEdgeFlowModel(ConditionalEdgeFlowConfig(num_vars=1, hidden=32, head_terms=2, branches_per_register=1))
    assert not hasattr(model, "cell")
    assert hasattr(model, "source_target_scorer")
    assert hasattr(model, "semantic_encoder")


def test_conditional_sampler_masks_inactive_padded_variables_and_uses_carry_write_update():
    base = torch.linspace(-1.0, 1.0, 20).unsqueeze(1)
    x = torch.cat([base, torch.zeros_like(base), torch.zeros_like(base)], dim=1)
    y = x[:, 0] + x[:, 0] ** 2
    template = RegisterOperatorTemplate(
        num_vars=3,
        num_registers=5,
        num_layers=1,
        primitives=("add", "mul", "square"),
        mixture_modes=1,
    )
    model = ConditionalEdgeFlowModel(ConditionalEdgeFlowConfig(
        num_vars=3,
        hidden=32,
        head_terms=2,
        branches_per_register=1,
        update_mode="carry_write",
        write_registers_per_layer=2,
    ))
    sample = ConditionalEdgeFlowSampler(template, model, method="policy", flow_steps=1).sample(
        x,
        y,
        batch_size=1,
        rng=random.Random(3),
    )[0]

    assert any(":TARGET" in key and ":SRC" in key for key in sample.edge_choices)
    assert any(":BRANCH" in key and ":OP" in key for key in sample.edge_choices)
    assert sample.diagnostics["decision_count"] == len(sample.edge_choices)
    assert sample.diagnostics["branches_per_register"] == 1
    assert sample.diagnostics["update_mode"] == "carry_write"
    assert sample.diagnostics["additive_update_count"] == 0
    assert sample.diagnostics["carried_register_count"] > 0
    assert sample.diagnostics["written_register_count"] == 2
    assert sample.diagnostics["active_variable_count"] == 1
    assert sample.active_log_prob_tensor is not None
    assert sample.diagnostics["active_decision_count"] < sample.diagnostics["decision_count"]
    first_layer_sources = [
        value for key, value in sample.edge_choices.items()
        if key.startswith("L0:") and key.endswith(":SRC")
    ]
    assert first_layer_sources
    assert set(first_layer_sources).isdisjoint({1, 2})
    target_op_groups = [
        key for key in sample.edge_choices
        if key.startswith("L0:") and key.endswith(":OP")
    ]
    assert len(target_op_groups) == 2


def test_conditional_sampler_can_exclude_base_terms_from_training_head():
    x = torch.linspace(-1.0, 1.0, 24).unsqueeze(1)
    y = x[:, 0] + x[:, 0] ** 2
    template = RegisterOperatorTemplate(
        num_vars=1,
        num_registers=4,
        num_layers=1,
        primitives=("add", "mul", "square"),
        mixture_modes=1,
    )
    model = ConditionalEdgeFlowModel(ConditionalEdgeFlowConfig(
        num_vars=1,
        hidden=32,
        head_terms=2,
        exclude_base_head_candidates=True,
    ))

    sample = ConditionalEdgeFlowSampler(template, model, method="policy", flow_steps=1).sample(
        x,
        y,
        batch_size=1,
        rng=random.Random(8),
        active_variable_count=1,
    )[0]

    assert sample.diagnostics["base_head_candidate_count"] > 0
    assert sample.diagnostics["base_head_selected_count"] == 0
    assert all(not _is_base_leaf(term) for term in sample.head_terms)


def test_conditional_sampler_records_keep_choices_and_masks_duplicate_branch_keys():
    x = torch.linspace(-1.0, 1.0, 20).unsqueeze(1)
    y = x[:, 0] ** 2
    template = RegisterOperatorTemplate(
        num_vars=1,
        num_registers=5,
        num_layers=2,
        primitives=("add", "mul", "square"),
        mixture_modes=1,
    )
    model = ConditionalEdgeFlowModel(ConditionalEdgeFlowConfig(
        num_vars=1,
        hidden=32,
        head_terms=2,
        update_mode="carry_write",
        write_registers_per_layer=3,
        enable_keep_option=True,
        mask_duplicate_branches=True,
        include_base_source_pool=False,
    ))

    sample = ConditionalEdgeFlowSampler(template, model, method="policy", flow_steps=1).sample(
        x,
        y,
        batch_size=1,
        rng=random.Random(11),
        active_variable_count=1,
    )[0]

    update_keys = [key for key in sample.edge_choices if key.endswith(":UPDATE_ACTION")]
    assert update_keys
    assert sample.diagnostics["keep_option_enabled"] is True
    assert sample.diagnostics["update_choice_count"] == len(update_keys)
    assert sample.diagnostics["include_base_source_pool"] is False
    branch_keys = sample.diagnostics["sampled_branch_keys"]
    assert len(branch_keys) == len(set(branch_keys))


def test_gt_formula_compiler_builds_trainable_csef_sample_for_simple_expression():
    x = torch.linspace(-1.0, 1.0, 16).unsqueeze(1)
    y = x[:, 0] ** 2 + x[:, 0]
    template = RegisterOperatorTemplate(
        num_vars=1,
        num_registers=5,
        num_layers=2,
        primitives=("add", "mul", "square"),
        mixture_modes=1,
    )
    model = ConditionalEdgeFlowModel(ConditionalEdgeFlowConfig(
        num_vars=1,
        hidden=32,
        head_terms=1,
        exclude_base_head_candidates=True,
    ))

    sample = compile_formula_to_csef_sample(
        "x0**2 + x0",
        variable_count=1,
        template=template,
        model=model,
        x=x,
        y=y,
    )

    assert sample is not None
    assert sample.log_prob_tensor is not None
    assert sample.active_log_prob_tensor is not None
    assert sample.diagnostics["is_gt_elite"] is True
    assert sample.diagnostics["gt_compile_success"] is True
    assert sample.diagnostics["active_decision_count"] > 0
    assert sample.decision_traces
    assert any(trace.active for trace in sample.decision_traces)


def test_gt_neighborhood_builds_reward_weighted_local_probability_shape():
    from semflow_sr.edge_flow.gt_neighborhood import build_gt_neighborhood_samples
    from semflow_sr.edge_flow.reward import RewardConfig, evaluate_expression_rewards
    from semflow_sr.edge_flow.semantic_teacher import local_posterior_targets_for_samples
    from semflow_sr.edge_flow.structure_posterior import normalize_log_weights

    x = torch.linspace(-1.0, 1.0, 16).unsqueeze(1)
    y = x[:, 0] ** 2 + x[:, 0]
    template = RegisterOperatorTemplate(
        num_vars=1,
        num_registers=5,
        num_layers=2,
        primitives=("add", "mul", "square"),
        mixture_modes=1,
    )
    model = ConditionalEdgeFlowModel(ConditionalEdgeFlowConfig(
        num_vars=1,
        hidden=32,
        head_terms=1,
        exclude_base_head_candidates=True,
    ))

    result = build_gt_neighborhood_samples(
        "x0**2 + x0",
        variable_count=1,
        template=template,
        model=model,
        x=x,
        y=y,
        method="policy",
        flow_steps=1,
        flow_time=0.5,
        rng=random.Random(7),
        size=4,
        op_replace_prob=1.0,
        source_replace_prob=1.0,
    )

    assert result.samples
    assert any((sample.diagnostics or {}).get("gt_neighborhood_canonical") for sample in result.samples)
    assert all(sample.log_prob_tensor is not None for sample in result.samples)
    assert all(sample.decision_traces for sample in result.samples)

    rewards = evaluate_expression_rewards(result.samples, x, y, RewardConfig(complexity_weight=0.0))
    weights = normalize_log_weights(rewards.r2)
    targets, diag = local_posterior_targets_for_samples(result.samples, weights, smoothing=0.0)

    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert targets
    assert diag["semantic_teacher_local_group_count"] >= 1
    for target in targets.values():
        assert torch.allclose(target.sum(), torch.tensor(1.0), atol=1e-6)


def test_parser_and_gt_compiler_support_protected_division_formula():
    expr = parse_formula("x0/x1", ["x0", "x1"])
    assert expr.kind == "op"
    assert expr.op_id == NAME_TO_ID["protected_div"]

    x0 = torch.linspace(0.5, 1.5, 16)
    x1 = torch.linspace(1.0, 2.0, 16)
    x = torch.stack([x0, x1], dim=1)
    y = x[:, 0] / x[:, 1]
    template = RegisterOperatorTemplate(
        num_vars=2,
        num_registers=6,
        num_layers=1,
        primitives=("add", "mul", "protected_div"),
        mixture_modes=1,
    )
    model = ConditionalEdgeFlowModel(ConditionalEdgeFlowConfig(
        num_vars=2,
        hidden=24,
        head_terms=1,
        exclude_base_head_candidates=True,
    ))

    sample = compile_formula_to_csef_sample(
        "x0/x1",
        variable_count=2,
        template=template,
        model=model,
        x=x,
        y=y,
    )

    assert sample is not None
    assert sample.expression.op_id == NAME_TO_ID["protected_div"]
    assert sample.log_prob_tensor is not None


def test_parser_handles_abs_wrappers_and_unary_negative_as_protected_structure():
    log_expr = parse_formula("log(Abs(x0))", ["x0"])
    sqrt_expr = parse_formula("sqrt(Abs(x1))", ["x0", "x1"])
    sub_expr = parse_formula("-x1**3 + x1", ["x0", "x1"])

    assert log_expr.op_id == NAME_TO_ID["protected_log"]
    assert log_expr.children[0] == Expr.var(0)
    assert sqrt_expr.op_id == NAME_TO_ID["protected_sqrt"]
    assert sqrt_expr.children[0] == Expr.var(1)
    assert sub_expr.op_id == NAME_TO_ID["sub"]


def test_token_sequence_metrics_report_bleu_accuracy_and_edit_distance():
    metrics = token_sequence_metrics("sin(x0) + x0", "sin(x0) + x1")

    assert 0.0 <= metrics["formula_bleu"] <= 1.0
    assert 0.0 <= metrics["formula_token_accuracy"] < 1.0
    assert metrics["formula_edit_distance"] == 1.0


def test_structure_conditioned_posterior_prefers_matching_skeleton_over_proxy():
    from semflow_sr.edge_flow.structure_posterior import (
        normalize_log_weights,
        structure_conditioned_log_weight,
        structure_similarity_score,
    )

    gt = "sin(x0)*cos(x1)"
    matching = "2.0*sin(x0)*cos(x1) + 0.1"
    proxy = "1.4*x0 - 0.5*x1 + 0.3"

    matching_score = structure_similarity_score(matching, gt)
    proxy_score = structure_similarity_score(proxy, gt)
    assert matching_score > proxy_score

    log_weights = torch.tensor([
        structure_conditioned_log_weight(
            r2=torch.tensor(0.95),
            complexity=torch.tensor(6.0),
            structure_score=torch.tensor(proxy_score),
            beta_y=2.0,
            beta_g=5.0,
            beta_c=0.0,
        ),
        structure_conditioned_log_weight(
            r2=torch.tensor(0.90),
            complexity=torch.tensor(8.0),
            structure_score=torch.tensor(matching_score),
            beta_y=2.0,
            beta_g=5.0,
            beta_c=0.0,
        ),
    ])
    weights = normalize_log_weights(log_weights)

    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert weights[1] > weights[0]


def test_semantic_only_teacher_velocity_handles_singular_kernel():
    from semflow_sr.edge_flow.semantic_teacher import semantic_only_teacher_velocity

    current = torch.tensor([0.5, 0.5])
    target = torch.tensor([0.05, 0.95])
    candidate_semantics = torch.stack([
        torch.ones(8),
        torch.ones(8),
    ], dim=1)

    result = semantic_only_teacher_velocity(
        current,
        candidate_semantics,
        target,
        beta=1.0,
    )

    assert torch.isfinite(result.sqrt_velocity).all()
    assert torch.allclose(result.mass_velocity.sum(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose((current.sqrt() * result.sqrt_velocity).sum(), torch.tensor(0.0), atol=1e-5)
    assert result.diagnostics["semantic_null_residual_norm"] > 0.0


def test_teacher_path_state_euclidean_builds_true_intermediate_state():
    from semflow_sr.edge_flow.semantic_teacher import teacher_path_state

    p0 = torch.tensor([0.2, 0.3, 0.5])
    target = torch.tensor([0.8, 0.1, 0.1])
    result = teacher_path_state(
        p0,
        None,
        target,
        flow_time=0.25,
        geometry="euclidean",
    )

    expected_p = torch.tensor([0.35, 0.25, 0.4])
    expected_u = target - p0

    assert torch.allclose(result.current_probs, expected_p, atol=1e-6)
    assert torch.allclose(result.mass_velocity, expected_u, atol=1e-6)
    assert torch.allclose(result.mass_velocity.sum(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(result.sqrt_velocity, 0.5 * expected_u / expected_p.sqrt(), atol=1e-6)
    assert result.diagnostics["teacher_path_geometry"] == "euclidean"


def test_teacher_path_state_semantic_alias_uses_fisher_path_not_semantic_projection():
    from semflow_sr.edge_flow.semantic_teacher import teacher_path_state

    p0 = torch.tensor([0.5, 0.5])
    target = torch.tensor([0.05, 0.95])
    candidate_semantics = torch.stack([
        torch.ones(8),
        torch.ones(8),
    ], dim=1)

    result = teacher_path_state(
        p0,
        candidate_semantics,
        target,
        flow_time=0.5,
        geometry="semantic",
    )

    assert torch.isfinite(result.current_probs).all()
    assert torch.isfinite(result.sqrt_velocity).all()
    assert torch.allclose(result.current_probs.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(result.mass_velocity.sum(), torch.tensor(0.0), atol=1e-6)
    assert result.diagnostics["teacher_path_geometry"] == "fisher"
    assert result.diagnostics["probability_path_geometry"] == "fisher"
    assert result.diagnostics["semantic_path_null_velocity_norm"] == 0.0
    assert result.diagnostics["semantic_path_kernel_rank"] == 0
    assert result.diagnostics["fisher_angle"] > 0.0


def test_fisher_path_hits_endpoints_and_is_sphere_tangent():
    from semflow_sr.edge_flow.semantic_teacher import teacher_path_state

    p0 = torch.tensor([0.2, 0.3, 0.5])
    target = torch.tensor([0.8, 0.1, 0.1])
    start = teacher_path_state(p0, None, target, flow_time=0.0, geometry="fisher")
    end = teacher_path_state(p0, None, target, flow_time=1.0, geometry="fisher")
    mid = teacher_path_state(p0, None, target, flow_time=0.4, geometry="fisher")

    assert torch.allclose(start.current_probs, p0, atol=1e-6)
    assert torch.allclose(end.current_probs, target, atol=1e-5)
    assert torch.allclose((mid.current_probs.sqrt() * mid.sqrt_velocity).sum(), torch.tensor(0.0), atol=1e-5)
    assert mid.diagnostics["teacher_path_geometry"] == "fisher"


def test_semantic_calibrated_velocity_loss_preserves_fisher_minimizer():
    from semflow_sr.edge_flow.semantic_teacher import semantic_calibrated_velocity_loss

    pred = torch.tensor([0.1, -0.2, 0.1])
    target = torch.tensor([0.0, -0.1, 0.1])
    probs = torch.tensor([0.2, 0.3, 0.5])
    sem = torch.tensor([
        [1.0, 0.0, -1.0],
        [0.5, 1.0, -0.5],
        [0.0, -1.0, 1.0],
    ])

    loss0, diag0 = semantic_calibrated_velocity_loss(pred, target, probs, sem, gamma=0.0)
    expected = torch.mean((pred - target) ** 2)
    assert torch.allclose(loss0, expected, atol=1e-7)
    assert diag0["semantic_calibration_gamma"] == 0.0

    loss1, diag1 = semantic_calibrated_velocity_loss(pred, target, probs, sem, gamma=2.0)
    assert torch.isfinite(loss1)
    assert loss1 >= expected
    assert diag1["semantic_calibration_gamma"] == 2.0
    zero, _ = semantic_calibrated_velocity_loss(target, target, probs, sem, gamma=2.0)
    assert torch.allclose(zero, torch.tensor(0.0), atol=1e-7)


def test_semantic_teacher_loss_reevaluates_model_velocity_at_path_state():
    from semflow_sr.edge_flow.semantic_teacher import DecisionTrace, semantic_teacher_loss_for_trace

    p0 = torch.tensor([0.5, 0.5])
    target = torch.tensor([0.9, 0.1])
    calls: list[tuple[torch.Tensor, float]] = []

    def velocity_fn(probs: torch.Tensor, flow_time: float) -> torch.Tensor:
        calls.append((probs.detach().clone(), float(flow_time)))
        return torch.zeros_like(probs)

    trace = DecisionTrace(
        group_id="HEAD:TERM0:SRC",
        choice=0,
        current_probs=p0,
        initial_probs=p0,
        candidate_semantics=torch.eye(2),
        predicted_sqrt_velocity=torch.full((2,), 99.0),
        velocity_fn=velocity_fn,
        flow_time=0.25,
        candidate_keys=("x0", "square(x0)"),
        active=True,
    )

    loss, diag = semantic_teacher_loss_for_trace(
        trace,
        target,
        teacher_path_geometry="euclidean",
        velocity_clip=None,
    )

    expected_p = torch.tensor([0.6, 0.4])
    assert calls
    assert torch.allclose(calls[0][0], expected_p, atol=1e-6)
    assert calls[0][1] == 0.25
    assert diag["teacher_path_geometry"] == "euclidean"
    assert diag["semantic_teacher_recomputed_velocity"] == 1.0
    assert torch.isfinite(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is required for teacher loss device regression")
def test_semantic_teacher_loss_zero_fallback_uses_trace_device_on_cuda():
    from types import SimpleNamespace

    from semflow_sr.edge_flow.semantic_teacher import DecisionTrace, semantic_teacher_loss_for_samples

    trace = DecisionTrace(
        group_id="HEAD:TERM0:SRC",
        choice=0,
        current_probs=torch.tensor([0.5, 0.5], device="cuda"),
        initial_probs=torch.tensor([0.5, 0.5], device="cuda"),
        candidate_semantics=None,
        predicted_sqrt_velocity=None,
        active=True,
    )
    sample = SimpleNamespace(decision_traces=[trace])

    loss, diag = semantic_teacher_loss_for_samples(
        [sample],
        torch.tensor([1.0]),
        teacher_beta=1.0,
        teacher_smoothing=0.05,
    )

    assert loss.device.type == "cuda"
    assert diag["semantic_teacher_skipped_count"] == 1


def test_semantic_teacher_projects_endpoint_posterior_to_local_target():
    from semflow_sr.edge_flow.semantic_teacher import DecisionTrace, local_posterior_targets_for_samples

    candidate_semantics = torch.eye(2)
    traces = [
        DecisionTrace(
            group_id="HEAD:TERM0:SRC",
            choice=0,
            current_probs=torch.tensor([0.5, 0.5]),
            candidate_semantics=candidate_semantics,
            predicted_sqrt_velocity=torch.zeros(2),
            candidate_keys=("x0", "square(x0)"),
            active=True,
        ),
        DecisionTrace(
            group_id="HEAD:TERM0:SRC",
            choice=1,
            current_probs=torch.tensor([0.5, 0.5]),
            candidate_semantics=candidate_semantics,
            predicted_sqrt_velocity=torch.zeros(2),
            candidate_keys=("x0", "square(x0)"),
            active=True,
        ),
    ]
    samples = [
        CircuitSample(0, 0, {}, Expr.var(0), 0.0, 1, decision_traces=(traces[0],)),
        CircuitSample(1, 0, {}, Expr.op(NAME_TO_ID["square"], (Expr.var(0),)), 0.0, 2, decision_traces=(traces[1],)),
    ]

    targets, diag = local_posterior_targets_for_samples(
        samples,
        torch.tensor([0.1, 0.9]),
        smoothing=0.0,
    )

    assert diag["semantic_teacher_local_group_count"] == 1
    assert torch.allclose(targets[(0, 0)], torch.tensor([0.1, 0.9]), atol=1e-6)
    assert torch.allclose(targets[(1, 0)], torch.tensor([0.1, 0.9]), atol=1e-6)
    assert targets[(0, 0)][1] > targets[(0, 0)][0]


def test_structural_denoising_targets_flow_to_clean_gt_action_not_weighted_proxy():
    from semflow_sr.edge_flow.semantic_teacher import DecisionTrace, structural_denoising_targets_for_samples

    candidate_semantics = torch.eye(2)
    clean_trace = DecisionTrace(
        group_id="HEAD:TERM0:SRC",
        choice=0,
        current_probs=torch.tensor([0.5, 0.5]),
        candidate_semantics=candidate_semantics,
        predicted_sqrt_velocity=torch.zeros(2),
        candidate_keys=("x0", "square(x0)"),
        active=True,
    )
    noisy_proxy_trace = DecisionTrace(
        group_id="HEAD:TERM0:SRC",
        choice=1,
        current_probs=torch.tensor([0.5, 0.5]),
        candidate_semantics=candidate_semantics,
        predicted_sqrt_velocity=torch.zeros(2),
        candidate_keys=("x0", "square(x0)"),
        active=True,
    )
    clean = CircuitSample(
        0,
        0,
        {},
        Expr.var(0),
        0.0,
        1,
        decision_traces=(clean_trace,),
        diagnostics={"gt_neighborhood_canonical": True, "is_gt_elite": True},
    )
    proxy = CircuitSample(
        1,
        0,
        {},
        Expr.op(NAME_TO_ID["square"], (Expr.var(0),)),
        0.0,
        2,
        decision_traces=(noisy_proxy_trace,),
        diagnostics={"gt_neighborhood_canonical": False},
    )

    targets, diag = structural_denoising_targets_for_samples(
        [clean, proxy],
        smoothing=0.0,
    )

    assert torch.allclose(targets[(0, 0)], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(targets[(1, 0)], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert diag["semantic_teacher_target_mode"] == "structural_denoising"
    assert diag["semantic_teacher_clean_trace_match_rate"] == 1.0


def test_smoothed_clean_target_uses_uniform_support_not_current_probs():
    from semflow_sr.edge_flow.semantic_teacher import one_hot_smoothed_target

    target = one_hot_smoothed_target(torch.tensor([0.98, 0.01, 0.01]), 1, smoothing=0.3)

    assert torch.allclose(target, torch.tensor([0.1, 0.8, 0.1]), atol=1e-6)


def test_structure_prior_rerank_penalizes_low_probability_complex_proxy():
    from semflow_sr.edge_flow.selection import structure_prior_scores

    rewards = torch.tensor([0.96, 0.94])
    log_probs = torch.tensor([-120.0, -4.0])
    complexities = torch.tensor([30.0, 8.0])

    scores = structure_prior_scores(
        rewards,
        log_probs,
        complexities,
        prior_weight=0.25,
        complexity_weight=0.001,
    )

    assert scores[1] > scores[0]


def test_conditional_sampler_records_semantic_teacher_traces_for_active_choices():
    x = torch.linspace(-1.0, 1.0, 24).unsqueeze(1)
    y = x[:, 0] + x[:, 0] ** 2
    template = RegisterOperatorTemplate(
        num_vars=1,
        num_registers=4,
        num_layers=1,
        primitives=("add", "mul", "square"),
        mixture_modes=1,
    )
    model = ConditionalEdgeFlowModel(ConditionalEdgeFlowConfig(num_vars=1, hidden=32, head_terms=2))

    sample = ConditionalEdgeFlowSampler(template, model, method="policy", flow_steps=1, time_sampling="uniform").sample(
        x,
        y,
        batch_size=1,
        rng=random.Random(13),
        active_variable_count=1,
    )[0]

    assert sample.decision_traces
    active = [trace for trace in sample.decision_traces if trace.active]
    assert active
    assert any(trace.group_id.startswith("HEAD:TERM") for trace in active)
    assert all(trace.candidate_semantics is not None for trace in active)
    assert all(trace.predicted_sqrt_velocity is not None for trace in active)
    assert all(0.0 <= float(trace.flow_time) <= 1.0 for trace in active)
    assert any(float(trace.flow_time) != 1.0 for trace in active)
    assert all(trace.candidate_keys for trace in active)


def _is_base_leaf(expr: Expr) -> bool:
    return expr.kind in {"var", "const"}
