import random

import torch

from semflow_sr.edge_flow.benchmark import skeleton_match, write_benchmark_result_files
from semflow_sr.edge_flow.dataset import EdgeFlowBuildConfig, build_edge_flow_records
from semflow_sr.edge_flow.model import EdgeFlowModel, EdgeFlowModelConfig, edge_flow_loss
from semflow_sr.edge_flow.template import RegisterOperatorTemplate


def test_edge_flow_records_and_model_loss_are_well_formed():
    template = RegisterOperatorTemplate(
        num_vars=1,
        num_registers=4,
        num_layers=2,
        primitives=("add", "mul", "square"),
        mixture_modes=1,
    )
    x = torch.linspace(-1.0, 1.0, 32).unsqueeze(1)
    y = x[:, 0] ** 2
    records = build_edge_flow_records(
        template,
        tasks=[("square", x, y)],
        cfg=EdgeFlowBuildConfig(samples_per_task=16, elite_k=4, lambda_value=0.5),
        rng=random.Random(0),
    )

    assert len(records) == 1
    rec = records[0]
    assert rec.task_id == "square"
    assert rec.theta0.template == template
    assert rec.theta_star.template == template
    assert rec.theta_lambda.template == template
    assert rec.diagnostics["num_sampled_expressions"] == 16
    assert rec.diagnostics["elite_k"] == 4

    model = EdgeFlowModel(EdgeFlowModelConfig(num_vars=1, hidden=32))
    pred = model(rec)
    loss, metrics = edge_flow_loss(pred, rec)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert "loss_mixture" in metrics
    assert set(pred.group_zdot) == set(rec.zdot_groups)


def test_edge_flow_records_can_use_validation_robust_rewards():
    template = RegisterOperatorTemplate(
        num_vars=1,
        num_registers=4,
        num_layers=2,
        primitives=("add", "mul", "square"),
        mixture_modes=1,
    )
    x = torch.linspace(-1.0, 1.0, 40).unsqueeze(1)
    y = x[:, 0] ** 2
    records = build_edge_flow_records(
        template,
        tasks=[("square", x, y)],
        cfg=EdgeFlowBuildConfig(samples_per_task=16, elite_k=4, lambda_value=0.5, validation_fraction=0.25),
        rng=random.Random(0),
    )

    rec = records[0]
    assert rec.diagnostics["reward_validation_fraction"] == 0.25
    assert "target_reward_train_val_gap_mean" in rec.diagnostics


def test_benchmark_result_writer_outputs_expression_and_group_stats(tmp_path):
    records = [
        {
            "task_id": "jin/Jin-1",
            "suite": "jin",
            "num_vars": 1,
            "ground_truth": "x0**2",
            "expression": "1.0*(x0**2) + 0.0",
            "r2": 1.0,
            "nmse": 0.0,
            "reward": 0.99,
            "complexity": 2,
            "valid_expression_fraction": 1.0,
            "unique_expression_fraction": 0.5,
        },
        {
            "task_id": "feynman/foo",
            "suite": "feynman",
            "num_vars": 2,
            "ground_truth": "",
            "expression": "0.0*(x0) + 1.0",
            "r2": 0.25,
            "nmse": 0.75,
            "reward": 0.24,
            "complexity": 1,
            "valid_expression_fraction": 0.8,
            "unique_expression_fraction": 0.4,
        },
    ]

    summary = write_benchmark_result_files(records, tmp_path, "edge_flow_test")

    assert summary["n_tasks"] == 2
    assert (tmp_path / "edge_flow_test_samples.jsonl").exists()
    expression_csv = (tmp_path / "edge_flow_test_task_expressions.csv").read_text()
    assert "ground_truth" in expression_csv
    assert "1.0*(x0**2) + 0.0" in expression_csv
    expression_md = (tmp_path / "edge_flow_test_task_expressions.md").read_text()
    assert "| jin/Jin-1 |" in expression_md
    stats_csv = (tmp_path / "edge_flow_test_statistics_by_group.csv").read_text()
    assert "all,all,2" in stats_csv
    assert "suite,jin,1" in stats_csv
    assert "num_vars,2,1" in stats_csv
    stats_json = (tmp_path / "edge_flow_test_statistics_by_group.json").read_text()
    assert '"group_type": "suite"' in stats_json
    assert "skeleton_accuracy" in stats_csv
    assert summary["skeleton_accuracy"] == 0.5
    assert (tmp_path / "edge_flow_test_diagnostics.json").exists()


def test_expression_skeleton_match_ignores_constants_and_protected_abs():
    assert skeleton_match("sqrt(1.23*x0)", "1.10906*(sqrt(Abs(x0))) + -6e-06")
    assert skeleton_match("2.7*x0**2", "-4.2*(x0**2) + 1.0")
    assert not skeleton_match("2.7*x0**2", "-4.2*(x0**3) + 1.0")
    assert not skeleton_match("x0**0.426", "0.88*(sqrt(Abs(x0))) + 0.1")
