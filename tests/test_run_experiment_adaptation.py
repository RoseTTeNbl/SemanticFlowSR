import pytest
import importlib.util
import sys
from pathlib import Path
import json

import pandas as pd


def _load_run_experiment():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_experiment = _load_run_experiment()
parse_ckpt_by_vars = run_experiment.parse_ckpt_by_vars
parse_target_kwargs = run_experiment.parse_target_kwargs
select_runner_for_task = run_experiment.select_runner_for_task
missing_checkpoint_dims = run_experiment.missing_checkpoint_dims
task_passes_dim_filter = run_experiment.task_passes_dim_filter
load_gp_distilled_scores = run_experiment.load_gp_distilled_scores
resolve_gp_policy_weight = run_experiment.resolve_gp_policy_weight
gather_manifest_tasks = run_experiment.gather_manifest_tasks


def test_parse_ckpt_by_vars_maps_dimensions_to_paths():
    parsed = parse_ckpt_by_vars(["1:ckpt_1.pt", "2:ckpt_2.pt", "10:many.pt"])

    assert parsed == {1: "ckpt_1.pt", 2: "ckpt_2.pt", 10: "many.pt"}


def test_parse_ckpt_by_vars_rejects_bad_specs():
    with pytest.raises(ValueError):
        parse_ckpt_by_vars(["1=bad.pt"])


def test_select_runner_for_task_skips_missing_dimension():
    runners = {1: object(), 2: object()}

    assert select_runner_for_task(2, runners) is runners[2]
    assert select_runner_for_task(3, runners) is None


def test_parse_target_kwargs_reads_json_object():
    parsed = parse_target_kwargs('{"max_completion_steps": 1, "eval_topk": 4}')

    assert parsed == {"max_completion_steps": 1, "eval_topk": 4}


def test_run_experiment_accepts_global_block_commit_mode():
    choices = run_experiment.build_arg_parser()._option_string_actions["--execution_mode"].choices

    assert "global_block_commit" in choices


def test_parse_target_kwargs_rejects_non_object():
    with pytest.raises(ValueError):
        parse_target_kwargs("[1, 2, 3]")


def test_missing_checkpoint_dims_reports_all_uncovered_task_dimensions():
    tasks = [
        type("Task", (), {"X_train": __import__("numpy").zeros((4, 1))})(),
        type("Task", (), {"X_train": __import__("numpy").zeros((4, 2))})(),
        type("Task", (), {"X_train": __import__("numpy").zeros((4, 3))})(),
    ]

    assert missing_checkpoint_dims(tasks, {1: object(), 3: object()}) == [2]


def test_task_passes_dim_filter_supports_multivariate_sweeps():
    task_1d = type("Task", (), {"X_train": __import__("numpy").zeros((4, 1))})()
    task_2d = type("Task", (), {"X_train": __import__("numpy").zeros((4, 2))})()
    task_3d = type("Task", (), {"X_train": __import__("numpy").zeros((4, 3))})()

    assert not task_passes_dim_filter(task_1d, min_vars=2, max_vars=None)
    assert task_passes_dim_filter(task_2d, min_vars=2, max_vars=None)
    assert task_passes_dim_filter(task_3d, min_vars=2, max_vars=3)
    assert not task_passes_dim_filter(task_3d, min_vars=None, max_vars=2)


def test_load_gp_distilled_scores_reads_event_likelihoods(tmp_path):
    path = tmp_path / "events.json"
    path.write_text(__import__("json").dumps({
        "events": [
            {"action_id": 3, "op": "square", "solved": True},
            {"action_id": 4, "op": "sin", "solved": False},
        ]
    }))

    action_scores, operator_scores = load_gp_distilled_scores(str(path))

    assert action_scores[3] > action_scores[4]
    assert operator_scores["square"] > operator_scores["sin"]


def test_gp_policy_weight_defaults_to_disabled_for_candidate_prior_path():
    assert resolve_gp_policy_weight(None, {"rollout_policy": "gp_guided"}, {"1": 2.0}, {"square": 1.0}) == 0.0
    assert resolve_gp_policy_weight(0.1, {"rollout_policy": "gp_guided"}, {"1": 2.0}, {"square": 1.0}) == 0.1


def test_gather_manifest_tasks_loads_new_benchmark_split(tmp_path):
    task_dir = tmp_path / "materialized" / "toy_suite" / "toy"
    task_dir.mkdir(parents=True)
    pd.DataFrame({"x0": [0.0, 1.0], "target": [0.0, 1.0]}).to_csv(task_dir / "train.csv", index=False)
    pd.DataFrame({"x0": [2.0], "target": [4.0]}).to_csv(task_dir / "test.csv", index=False)
    manifest_path = tmp_path / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps({
        "version": "1.0",
        "suites": {
            "toy_suite": [{
                "task_id": "toy_suite/toy",
                "suite": "toy_suite",
                "num_vars": 1,
                "variable_names": ["x0"],
                "train_path": "materialized/toy_suite/toy/train.csv",
                "test_path": "materialized/toy_suite/toy/test.csv",
                "target_column": "target",
                "ground_truth": "x0**2",
            }]
        },
    }))

    tasks = gather_manifest_tasks(
        manifest_path,
        suites=["toy_suite"],
        root=tmp_path,
        min_vars=1,
        max_vars=1,
        limit=1,
    )

    assert len(tasks) == 1
    assert tasks[0].name == "toy_suite/toy"
    assert tasks[0].metadata["suite"] == "toy_suite"
