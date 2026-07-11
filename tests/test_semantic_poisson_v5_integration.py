from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_complete_expression_semantic_fm import (  # noqa: E402
    DEFAULT_OPS,
    ConditionalSemanticPotentialV5,
    PoissonResidualVelocityV5,
    RegisterOperatorSimplexTemplate,
    ResidualVelocityHeadV5,
    TaskConditionedVelocityNetV5,
    graph_action_mask,
    masked_block_softmax,
    random_theta,
)


def _template():
    return RegisterOperatorSimplexTemplate(
        num_vars=1,
        num_layers=2,
        num_registers=5,
        ops=tuple(DEFAULT_OPS[:4]),
        output_terms=1,
    )


def test_v5_velocity_interface_has_no_route_or_theta0_argument():
    parameters = inspect.signature(TaskConditionedVelocityNetV5.forward).parameters
    assert list(parameters) == ["self", "x", "y", "theta", "t"]


def test_v5_base_and_residual_field_are_finite_and_route_free():
    template = _template()
    base = TaskConditionedVelocityNetV5(
        template,
        32,
        global_state_mode="full",
        metadata_embedding_dim=0,
        task_encoder_mode="stats",
        task_conditioning="xy",
    )
    flow = PoissonResidualVelocityV5(base)
    flow.add_residual(ResidualVelocityHeadV5(template, 32), 0.1)
    theta = random_theta(template, scale=0.5, device=torch.device("cpu"))
    x = torch.linspace(-1.0, 1.0, 12).unsqueeze(1)
    y = x[:, 0].square()

    velocity = flow(x, y, theta, 0.4)

    assert velocity.shape == theta.shape
    assert torch.isfinite(velocity).all()
    assert flow.velocity_parameterization == "direct_velocity"


def test_v5_residual_stages_share_one_feature_trunk_call():
    template = _template()
    base = TaskConditionedVelocityNetV5(
        template,
        32,
        global_state_mode="full",
        metadata_embedding_dim=0,
        task_encoder_mode="stats",
        task_conditioning="xy",
    )
    flow = PoissonResidualVelocityV5(base)
    flow.add_residual(ResidualVelocityHeadV5(template, 32), 0.1)
    flow.add_residual(ResidualVelocityHeadV5(template, 32), 0.1)
    calls = []
    hook = flow.residual_trunk.register_forward_hook(lambda *_args: calls.append(1))
    theta = random_theta(template, scale=0.5, device=torch.device("cpu"))
    x = torch.linspace(-1.0, 1.0, 12).unsqueeze(1)
    y = x[:, 0].square()

    flow(x, y, theta, 0.4)
    hook.remove()

    assert len(calls) == 1


def test_v5_potential_depends_on_task_and_endpoint_probabilities():
    template = _template()
    theta = torch.stack([
        random_theta(template, scale=0.2, device=torch.device("cpu")),
        random_theta(template, scale=0.7, device=torch.device("cpu")),
    ])
    probabilities = torch.stack([
        masked_block_softmax(row.view(len(template.blocks), template.source_count), template)
        for row in theta
    ]).requires_grad_(True)
    potential = ConditionalSemanticPotentialV5(template, 32)
    x = torch.linspace(-1.0, 1.0, 10).unsqueeze(1)
    y = x[:, 0]

    values = potential(probabilities, x, y)
    gradient = torch.autograd.grad(values.sum(), probabilities)[0]

    assert values.shape == (2,)
    assert gradient.shape == probabilities.shape
    assert torch.isfinite(gradient).all()
    assert graph_action_mask(template).shape == probabilities.shape[1:]


def test_v51_batched_velocity_matches_scalar_calls():
    template = _template()
    base = TaskConditionedVelocityNetV5(
        template,
        32,
        global_state_mode="full",
        metadata_embedding_dim=0,
        task_encoder_mode="stats",
        task_conditioning="xy",
    )
    x = torch.linspace(-1.0, 1.0, 12).unsqueeze(1)
    y = x[:, 0].square() + x[:, 0]
    theta = torch.stack([
        random_theta(template, scale=0.3, device=torch.device("cpu")),
        random_theta(template, scale=0.7, device=torch.device("cpu")),
    ])
    batched = base.forward_batch(x, y, theta, 0.4)
    scalar = torch.stack([base(x, y, row, 0.4) for row in theta])
    assert torch.allclose(batched, scalar, atol=1.0e-6, rtol=1.0e-5)
