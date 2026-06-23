import importlib.util
import csv
import random
from pathlib import Path

import pytest
import torch

from semflow_sr.data.symbolicgpt_subset import SymbolicGPTSubsetConfig, generate_symbolicgpt_subset
from semflow_sr.edge_flow.benchmark import expression_skeleton, skeleton_match, with_skeleton_metrics, write_benchmark_result_files
from semflow_sr.edge_flow.circuit_sampler import CircuitSample
from semflow_sr.edge_flow.dataset import EdgeFlowBuildConfig, build_edge_flow_records
from semflow_sr.edge_flow.model import EdgeFlowModel, EdgeFlowModelConfig, edge_flow_loss
from semflow_sr.edge_flow.proposals import load_diffusion_formula_proposals, simple_gp_proposals
from semflow_sr.edge_flow.reward import RewardConfig, evaluate_expression_rewards
from semflow_sr.edge_flow.template import RegisterOperatorTemplate
from semflow_sr.edge_flow.train_edge_flow import _resolve_device, run as run_edge_flow_train
from semflow_sr.sr.ast import Expr


def test_training_device_resolver_supports_cpu_and_auto():
    assert _resolve_device({"device": "cpu"}).type == "cpu"
    assert isinstance(_resolve_device({"device": "auto"}), torch.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is required for reward device regression")
def test_reward_batch_keeps_metric_tensors_on_cuda():
    x = torch.linspace(-1.0, 1.0, 8, device="cuda").unsqueeze(1)
    y = x[:, 0] ** 2
    sample = CircuitSample(
        sample_id=0,
        mode=0,
        edge_choices={},
        expression=Expr.var(0),
        log_prob=0.0,
        complexity=1,
        head_terms=(Expr.var(0),),
    )

    rewards = evaluate_expression_rewards([sample], x, y, RewardConfig())

    assert rewards.rewards.device.type == "cuda"
    assert rewards.r2.device.type == "cuda"
    assert rewards.valid_mask.device.type == "cuda"
    assert rewards.complexity.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is required for loss device regression")
def test_training_loss_device_normalization_allows_cpu_zero_and_cuda_losses():
    from semflow_sr.edge_flow.train_edge_flow import _loss_on_training_device

    device = torch.device("cuda")
    losses = [
        _loss_on_training_device(torch.zeros((), requires_grad=True), device),
        _loss_on_training_device(torch.ones((), device=device, requires_grad=True), device),
    ]

    stacked = torch.stack(losses)

    assert stacked.device.type == "cuda"


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


def test_skeleton_metrics_prefer_final_expression_and_prune_tiny_terms():
    row = with_skeleton_metrics({
        "ground_truth": "x0**2 + x0",
        "expression": "1.0000001*x0**2 + 0.9999999*x0 + 1e-9*sin(x0)",
        "raw_expression": "x0",
    })

    assert row["skeleton_match"] is True
    assert expression_skeleton("x0 + 1e-9*sin(x0)") == expression_skeleton("x0")


def test_structure_metrics_include_symbolic_and_operator_dependency_audits():
    equivalent = with_skeleton_metrics({
        "ground_truth": "x0**2 + x0",
        "expression": "x0*(x0 + 1)",
    })
    dependency = with_skeleton_metrics({
        "ground_truth": "sin(x0)*cos(x1)",
        "expression": "2.0*sin(x0)*cos(x1) + 3.0",
    })

    assert equivalent["simplified_symbolic_equivalence"] is True
    assert "operator_dependency_match" in dependency
    assert dependency["operator_dependency_match"] is True


def test_train_entrypoint_defaults_to_conditional_semantic_edge_flow(tmp_path):
    ckpt = run_edge_flow_train({
        "seed": 0,
        "out": str(tmp_path),
        "checkpoint_name": "csef_test.pt",
        "num_tasks": 1,
        "runtime": {"torch_num_threads": 1, "torch_num_interop_threads": 1, "device": "cpu"},
        "template": {
            "num_vars": 1,
            "num_registers": 3,
            "num_layers": 1,
            "mixture_modes": 1,
            "primitives": ["add", "mul", "square"],
        },
        "gen": {"max_depth": 2, "probe_size": 16},
        "model": {"hidden": 24, "head_terms": 2},
        "train": {
            "epochs": 1,
            "samples_per_task": 8,
            "elite_k": 2,
            "sampler_method": "ode",
            "flow_steps": 2,
            "complexity_weight": 0.001,
            "lr": 0.001,
        },
    })

    saved = torch.load(ckpt, map_location="cpu", weights_only=False)

    assert saved["algorithm"] == "conditional_semantic_edge_flow"
    assert saved["model_cfg"]["head_terms"] == 2
    curve = (tmp_path / "train_curve_csef_test.csv")
    assert curve.exists()
    header = curve.read_text().splitlines()[0]
    assert "device" in header
    assert "fitted_head_gain_mean" in header
    assert "head_coef_norm_mean" in header


def test_conditional_training_writes_minibatch_and_replay_fields(tmp_path):
    ckpt = run_edge_flow_train({
        "seed": 1,
        "out": str(tmp_path),
        "checkpoint_name": "csef_batch_test.pt",
        "num_tasks": 3,
        "runtime": {"torch_num_threads": 1, "torch_num_interop_threads": 1, "device": "cpu"},
        "template": {
            "num_vars": 2,
            "num_registers": 4,
            "num_layers": 1,
            "mixture_modes": 1,
            "primitives": ["add", "mul", "square"],
        },
        "gen": {"num_vars_min": 1, "num_vars_max": 2, "max_depth": 2, "probe_size": 12},
        "model": {"hidden": 24, "head_terms": 1, "update_mode": "replace"},
        "train": {
            "epochs": 1,
            "task_batch_size": 2,
            "replay_capacity": 2,
            "replay_ratio": 0.5,
            "samples_per_task": 4,
            "elite_k": 1,
            "sampler_method": "policy",
            "flow_steps": 1,
            "complexity_weight": 0.001,
            "lr": 0.001,
        },
    })

    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    header = (tmp_path / "train_curve_csef_batch_test.csv").read_text().splitlines()[0]

    assert saved["model_cfg"]["update_mode"] == "replace"
    assert "optimizer_step" in header
    assert "task_batch_size" in header
    assert "replay_size" in header
    assert "head_fit_mode" in header


def test_conditional_training_loads_symbolicgpt_subset_source(tmp_path):
    data_root = tmp_path / "symbolicgpt_subset"
    generate_symbolicgpt_subset(
        SymbolicGPTSubsetConfig(
            root=data_root,
            train_count=2,
            val_count=0,
            test_count=0,
            num_vars=3,
            num_points=16,
            max_depth=2,
            seed=4,
        )
    )

    ckpt = run_edge_flow_train({
        "seed": 4,
        "out": str(tmp_path),
        "checkpoint_name": "csef_symbolicgpt_test.pt",
        "runtime": {"torch_num_threads": 1, "torch_num_interop_threads": 1, "device": "cpu"},
        "data": {
            "source": "symbolicgpt_subset",
            "root": str(data_root),
            "splits": ["train"],
            "limit_tasks": 2,
            "max_train_points": 12,
        },
        "template": {
            "num_vars": 3,
            "num_registers": 5,
            "num_layers": 1,
            "mixture_modes": 1,
            "primitives": ["add", "mul", "square"],
        },
        "model": {"hidden": 24, "head_terms": 2, "update_mode": "carry_write", "write_registers_per_layer": 2},
        "train": {
            "epochs": 1,
            "samples_per_task": 4,
            "elite_k": 1,
            "sampler_method": "policy",
            "flow_steps": 1,
            "complexity_weight": 0.001,
            "lr": 0.001,
            "head_fit_mode": "linear",
        },
    })

    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    header = (tmp_path / "train_curve_csef_symbolicgpt_test.csv").read_text().splitlines()[0]

    assert saved["algorithm"] == "conditional_semantic_edge_flow"
    assert saved["model_cfg"]["update_mode"] == "carry_write"
    assert saved["model_cfg"]["head_terms"] == 2
    assert "active_decision_count_mean" in header


def test_conditional_training_saves_anti_proxy_and_pointnet_config_fields(tmp_path):
    ckpt = run_edge_flow_train({
        "seed": 6,
        "out": str(tmp_path),
        "checkpoint_name": "csef_antiproxy_test.pt",
        "num_tasks": 1,
        "runtime": {"torch_num_threads": 1, "torch_num_interop_threads": 1, "device": "cpu"},
        "template": {
            "num_vars": 1,
            "num_registers": 4,
            "num_layers": 1,
            "mixture_modes": 1,
            "primitives": ["add", "mul", "square"],
        },
        "gen": {"num_vars": 1, "max_depth": 2, "probe_size": 12},
        "model": {
            "hidden": 24,
            "head_terms": 2,
            "task_encoder": "pointnet",
            "exclude_base_head_candidates": True,
            "enable_keep_option": True,
            "mask_duplicate_branches": True,
            "include_base_source_pool": False,
        },
        "train": {
            "epochs": 1,
            "samples_per_task": 4,
            "elite_k": 1,
            "sampler_method": "policy",
            "flow_steps": 1,
            "complexity_weight": 0.0,
            "lr": 0.001,
            "head_fit_mode": "linear",
        },
    })

    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    header = (tmp_path / "train_curve_csef_antiproxy_test.csv").read_text().splitlines()[0]

    assert saved["model_cfg"]["task_encoder"] == "pointnet"
    assert saved["model_cfg"]["exclude_base_head_candidates"] is True
    assert saved["model_cfg"]["enable_keep_option"] is True
    assert saved["model_cfg"]["mask_duplicate_branches"] is True
    assert saved["model_cfg"]["include_base_source_pool"] is False
    assert "base_head_selected_rate" in header
    assert "exclude_base_head_candidates" in header


def test_conditional_training_writes_semantic_teacher_and_structure_diagnostics(tmp_path):
    ckpt = run_edge_flow_train({
        "seed": 8,
        "out": str(tmp_path),
        "checkpoint_name": "csef_teacher_test.pt",
        "num_tasks": 1,
        "runtime": {"torch_num_threads": 1, "torch_num_interop_threads": 1, "device": "cpu"},
        "template": {
            "num_vars": 1,
            "num_registers": 4,
            "num_layers": 1,
            "mixture_modes": 1,
            "primitives": ["add", "mul", "square"],
        },
        "gen": {"num_vars": 1, "max_depth": 2, "probe_size": 16},
        "model": {
            "hidden": 24,
            "head_terms": 2,
            "exclude_base_head_candidates": True,
        },
        "train": {
            "epochs": 1,
            "samples_per_task": 4,
            "elite_k": 1,
            "objective": "semantic_teacher",
            "sampler_method": "policy",
            "flow_steps": 1,
            "complexity_weight": 0.0,
            "structure_beta": 4.0,
            "semantic_beta": 1.0,
            "teacher_beta": 1.0,
            "teacher_smoothing": 0.05,
            "teacher_pinv_rtol": 0.01,
            "teacher_velocity_clip": 5.0,
            "teacher_time_sampling": "uniform",
            "teacher_path_geometry": "semantic",
            "active_nll_weight": 0.1,
            "lr": 0.001,
            "head_fit_mode": "linear",
            "inject_gt_elite": True,
        },
    })

    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    header = (tmp_path / "train_curve_csef_teacher_test.csv").read_text().splitlines()[0]

    assert saved["algorithm"] == "conditional_semantic_edge_flow"
    assert "semantic_teacher_loss_mean" in header
    assert "semantic_null_residual_norm_mean" in header
    assert "semantic_teacher_local_group_count" in header
    assert "semantic_teacher_time_mean" in header
    assert "semantic_teacher_velocity_norm_mean" in header
    assert "semantic_teacher_velocity_scale_mean" in header
    assert "teacher_path_geometry" in header
    assert "probability_path_geometry" in header
    assert "semantic_calibration_gamma" in header
    assert "semantic_calibration_loss_mean" in header
    assert "semantic_calibration_energy_mean" in header
    assert "teacher_path_state_l1_from_initial_mean" in header
    assert "teacher_path_endpoint_l1_mean" in header
    assert "teacher_path_current_entropy_mean" in header
    assert "semantic_teacher_recomputed_velocity_rate" in header
    assert "structure_score_mean" in header
    assert "teacher_time_sampling" in header
    assert "gt_decision_top1_rate" in header
    assert "gt_decision_top3_rate" in header
    assert "gt_reward_rank" in header
    assert "gt_neighborhood_compiled" in header
    assert "gt_neighborhood_compile_success_rate" in header


def test_conditional_training_uses_structural_denoising_teacher_targets(tmp_path):
    ckpt = run_edge_flow_train({
        "seed": 9,
        "out": str(tmp_path),
        "checkpoint_name": "csef_structural_denoising_test.pt",
        "num_tasks": 1,
        "runtime": {"torch_num_threads": 1, "torch_num_interop_threads": 1, "device": "cpu"},
        "template": {
            "num_vars": 1,
            "num_registers": 4,
            "num_layers": 1,
            "mixture_modes": 1,
            "primitives": ["add", "mul", "square"],
        },
        "gen": {"num_vars": 1, "max_depth": 2, "probe_size": 16},
        "model": {
            "hidden": 24,
            "head_terms": 1,
            "exclude_base_head_candidates": True,
        },
        "train": {
            "epochs": 1,
            "samples_per_task": 0,
            "elite_k": 1,
            "objective": "semantic_teacher",
            "sampler_method": "policy",
            "flow_steps": 1,
            "complexity_weight": 0.0,
            "teacher_beta": 1.0,
            "teacher_smoothing": 0.05,
            "teacher_velocity_clip": 5.0,
            "teacher_time_sampling": "uniform",
            "teacher_path_geometry": "semantic",
            "probability_path_geometry": "fisher",
            "semantic_calibration_gamma": 1.0,
            "lr": 0.001,
            "head_fit_mode": "linear",
            "inject_gt_elite": True,
            "target_shape_source": "structural_denoising",
            "gt_neighborhood_size": 4,
            "gt_neighborhood_op_replace_prob": 1.0,
            "gt_neighborhood_source_replace_prob": 1.0,
        },
    })

    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    rows = list(csv.DictReader((tmp_path / "train_curve_csef_structural_denoising_test.csv").open()))

    assert saved["algorithm"] == "conditional_semantic_edge_flow"
    assert rows
    assert rows[0]["semantic_teacher_target_mode"] == "structural_denoising"
    assert float(rows[0]["semantic_teacher_clean_trace_match_rate"]) > 0.0


def test_simple_gp_proposals_return_formula_candidates():
    x = torch.linspace(-1.0, 1.0, 16).unsqueeze(1)
    y = x[:, 0] ** 2

    proposals = simple_gp_proposals(
        x,
        y,
        num_vars=1,
        primitives=("add", "mul", "square"),
        rng=random.Random(0),
        proposal_count=3,
        population_size=8,
        generations=1,
        max_depth=2,
    )

    assert 1 <= len(proposals) <= 3
    assert all(proposal.source == "gp" for proposal in proposals)
    assert all(proposal.formula for proposal in proposals)


def test_diffusion_formula_proposals_filter_by_task_id(tmp_path):
    path = tmp_path / "proposals.jsonl"
    path.write_text(
        "\n".join([
            '{"task_id": "task/a", "formula": "x0+x1", "source": "diffusion"}',
            '{"task_id": "task/b", "formula": "sin(x0)", "source": "diffusion"}',
            '{"task_id": "task/a", "formula": "x0*x1", "source": "diffusion"}',
        ])
    )

    proposals = load_diffusion_formula_proposals(path, task_id="task/a", limit=8)

    assert [proposal.formula for proposal in proposals] == ["x0+x1", "x0*x1"]
    assert all(proposal.source == "diffusion" for proposal in proposals)


def test_diffusion_proposer_cli_trains_and_writes_task_scoped_proposals(tmp_path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_diffusion_proposer.py"
    spec = importlib.util.spec_from_file_location("train_diffusion_proposer", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_from_args = module.run_from_args

    data_root = tmp_path / "symbolicgpt_subset"
    generate_symbolicgpt_subset(
        SymbolicGPTSubsetConfig(
            root=data_root,
            train_count=2,
            val_count=1,
            test_count=0,
            num_vars=3,
            num_points=12,
            max_depth=2,
            seed=7,
        )
    )
    ckpt = tmp_path / "diffusion.pt"
    proposals = tmp_path / "proposals.jsonl"
    curve = tmp_path / "curve.json"

    run_from_args([
        "--root", str(data_root),
        "--ckpt", str(ckpt),
        "--proposals_out", str(proposals),
        "--curve_out", str(curve),
        "--epochs", "1",
        "--batch_size", "2",
        "--hidden", "16",
        "--train_limit", "2",
        "--val_limit", "1",
        "--proposals_per_task", "2",
        "--device", "cpu",
        "--generation_mode", "teacher_noised",
        "--fallback_to_teacher",
        "--seed", "7",
    ])

    assert ckpt.exists()
    assert curve.exists()
    rows = [line for line in proposals.read_text().splitlines() if line.strip()]
    assert len(rows) == 4
    assert all('"task_id": "symbolicgpt_subset/train/' in row for row in rows)
