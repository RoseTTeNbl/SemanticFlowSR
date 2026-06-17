import pytest
import importlib.util
import sys
from pathlib import Path


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
