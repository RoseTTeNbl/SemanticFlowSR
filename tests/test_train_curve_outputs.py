import csv

from semflow_sr.train.train_velocity_gt import _save_curves, target_from_config


def test_save_curves_uses_run_name_to_avoid_overwriting(tmp_path):
    rows = [
        {"step": 0, "epoch": 0, "loss": 1.0, "reward": None},
        {"step": 1, "epoch": 0, "loss": 0.5, "reward": 0.8},
    ]

    _save_curves(rows, tmp_path, run_name="velocity_d2")

    csv_path = tmp_path / "train_curve_velocity_d2.csv"
    assert csv_path.exists()
    loaded = list(csv.DictReader(csv_path.open()))
    assert loaded[0]["step"] == "0"
    assert (tmp_path / "train_curve_velocity_d2.png").exists()


def test_train_entry_target_defaults_to_config_target_name():
    assert target_from_config({"target": {"name": "rollout_fitness_advantage"}}) == "rollout_fitness_advantage"
    assert target_from_config({}) == "gt"
