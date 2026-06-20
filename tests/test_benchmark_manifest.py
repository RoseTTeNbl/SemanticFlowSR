import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semflow_sr.data.benchmark_loader import load_materialized_task
from semflow_sr.data.benchmark_manifest import (
    BenchmarkTaskSpec,
    BenchmarkSuiteSpec,
    build_benchmark_index,
    load_benchmark_manifest,
    write_benchmark_manifest,
)
from semflow_sr.data.benchmark_prepare import (
    PMLBFilter,
    filter_pmlb_metadata,
    materialize_arrays,
    parse_srsd_text_table,
    srsd_problem_names_from_siblings,
)
from semflow_sr.data.benchmark_validate import validate_benchmark_manifest, write_validation_reports


def test_materialized_task_manifest_round_trips_and_loads_split(tmp_path):
    root = tmp_path / "materialized"
    task_dir = root / "formula_dev" / "Toy-1"
    X_train = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    y_train = np.array([1.0, 3.0, 5.0])
    X_val = np.array([[3.0, 4.0]])
    y_val = np.array([7.0])
    X_test = np.array([[4.0, 5.0], [5.0, 6.0]])
    y_test = np.array([9.0, 11.0])

    spec = materialize_arrays(
        task_id="formula_dev/Toy-1",
        suite="formula_dev",
        root=root,
        name="Toy-1",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        variable_names=["x0", "x1"],
        ground_truth="2*x0 + 1",
        domain="synthetic",
        tags=["dev"],
    )
    manifest = BenchmarkSuiteSpec(version="1.0", suites={"formula_dev": [spec]})
    manifest_path = tmp_path / "benchmark_manifest.json"

    write_benchmark_manifest(manifest, manifest_path)
    loaded = load_benchmark_manifest(manifest_path)
    task = load_materialized_task(loaded.suites["formula_dev"][0], root=tmp_path)

    assert loaded.version == "1.0"
    assert spec.train_path == "materialized/formula_dev/Toy-1/train.csv"
    assert (task_dir / "val.csv").exists()
    assert task.name == "formula_dev/Toy-1"
    assert task.variable_names == ["x0", "x1"]
    assert task.metadata["suite"] == "formula_dev"
    assert task.metadata["ground_truth"] == "2*x0 + 1"
    assert np.allclose(task.X_train, X_train)
    assert np.allclose(task.y_test, y_test)


def test_build_benchmark_index_classifies_main_and_appendix_suites():
    manifest = BenchmarkSuiteSpec(
        version="1.0",
        suites={
            "srsd_feynman_easy": [
                BenchmarkTaskSpec(
                    task_id="srsd_feynman_easy/E1",
                    suite="srsd_feynman_easy",
                    num_vars=2,
                    variable_names=["x0", "x1"],
                    train_path="a/train.csv",
                    val_path="a/val.csv",
                    test_path="a/test.csv",
                    has_dummy_vars=False,
                    metrics=["r2", "nmse", "symbolic_equivalence", "complexity"],
                    split="main",
                )
            ],
            "srsd_feynman_easy_dummy": [
                BenchmarkTaskSpec(
                    task_id="srsd_feynman_easy_dummy/E1_dummy",
                    suite="srsd_feynman_easy_dummy",
                    num_vars=5,
                    variable_names=["x0", "x1", "d0", "d1", "d2"],
                    train_path="b/train.csv",
                    val_path="b/val.csv",
                    test_path="b/test.csv",
                    has_dummy_vars=True,
                    metrics=["r2", "nmse", "variable_selection", "complexity"],
                    split="appendix",
                )
            ],
        },
    )

    rows = build_benchmark_index(manifest)

    assert rows[0]["suite"] == "srsd_feynman_easy"
    assert rows[0]["n_tasks"] == 1
    assert rows[0]["split"] == "main"
    assert rows[1]["has_dummy_vars"] is True
    assert rows[1]["metrics"] == ["r2", "nmse", "variable_selection", "complexity"]


def test_pmlb_filter_selects_regression_subset_with_fixed_limits():
    meta = pd.DataFrame(
        [
            {"dataset": "small_ok", "task": "regression", "n_samples": 1000, "n_features": 10, "n_missing_values": 0},
            {"dataset": "too_wide", "task": "regression", "n_samples": 1000, "n_features": 30, "n_missing_values": 0},
            {"dataset": "classification", "task": "classification", "n_samples": 1000, "n_features": 5, "n_missing_values": 0},
            {"dataset": "too_large", "task": "regression", "n_samples": 10000, "n_features": 5, "n_missing_values": 0},
        ]
    )

    names = filter_pmlb_metadata(meta, PMLBFilter(max_samples=5000, max_features=20, limit=10))

    assert names == ["small_ok"]


def test_manifest_loader_rejects_missing_split_paths(tmp_path):
    manifest_path = tmp_path / "bad.json"
    manifest_path.write_text(json.dumps({
        "version": "1.0",
        "suites": {
            "bad": [{
                "task_id": "bad/Toy",
                "suite": "bad",
                "num_vars": 1,
                "variable_names": ["x0"],
                "train_path": "missing/train.csv",
                "test_path": "missing/test.csv",
            }]
        },
    }))

    manifest = load_benchmark_manifest(manifest_path)
    with pytest.raises(FileNotFoundError):
        load_materialized_task(manifest.suites["bad"][0], root=tmp_path)


def test_srsd_text_parser_treats_last_column_as_target():
    X, y = parse_srsd_text_table("1.0 2.0 2.0\n3.0 4.0 12.0\n")

    assert X.shape == (2, 2)
    assert y.tolist() == [2.0, 12.0]


def test_srsd_problem_names_require_train_val_test_files():
    siblings = [
        {"rfilename": "train/feynman-i.12.1.txt"},
        {"rfilename": "val/feynman-i.12.1.txt"},
        {"rfilename": "test/feynman-i.12.1.txt"},
        {"rfilename": "train/incomplete.txt"},
        {"rfilename": "test/incomplete.txt"},
    ]

    assert srsd_problem_names_from_siblings(siblings) == ["feynman-i.12.1"]


def test_manifest_validation_reports_loadable_tasks_and_failures(tmp_path):
    root = tmp_path / "materialized"
    good = materialize_arrays(
        task_id="suite/good",
        suite="suite",
        root=root,
        name="good",
        X_train=np.array([[0.0], [1.0]]),
        y_train=np.array([0.0, 1.0]),
        X_test=np.array([[2.0]]),
        y_test=np.array([2.0]),
        variable_names=["x0"],
    )
    bad = BenchmarkTaskSpec(
        task_id="suite/bad",
        suite="suite",
        num_vars=2,
        variable_names=["x0", "x1"],
        train_path=good.train_path,
        test_path=good.test_path,
    )
    manifest_path = tmp_path / "manifest.json"
    write_benchmark_manifest(BenchmarkSuiteSpec(version="1.0", suites={"suite": [good, bad]}), manifest_path)

    result = validate_benchmark_manifest(manifest_path, root=tmp_path)

    assert result.summary["n_tasks"] == 2
    assert result.summary["n_valid"] == 1
    assert result.summary["n_failed"] == 1
    assert result.suite_rows[0]["suite"] == "suite"
    assert result.failures[0]["task_id"] == "suite/bad"
    assert "missing variable columns" in result.failures[0]["error"]


def test_manifest_validation_writes_summary_failures_and_task_rows(tmp_path):
    root = tmp_path / "materialized"
    spec = materialize_arrays(
        task_id="suite/good",
        suite="suite",
        root=root,
        name="good",
        X_train=np.array([[0.0], [1.0]]),
        y_train=np.array([0.0, 1.0]),
        X_test=np.array([[2.0]]),
        y_test=np.array([2.0]),
        variable_names=["x0"],
    )
    manifest_path = tmp_path / "manifest.json"
    write_benchmark_manifest(BenchmarkSuiteSpec(version="1.0", suites={"suite": [spec]}), manifest_path)
    result = validate_benchmark_manifest(manifest_path, root=tmp_path)

    write_validation_reports(result, tmp_path / "reports")

    summary = json.loads((tmp_path / "reports" / "manifest_validation_summary.json").read_text())
    failures = (tmp_path / "reports" / "manifest_validation_failures.jsonl").read_text()
    task_rows = pd.read_csv(tmp_path / "reports" / "manifest_validation_tasks.csv")
    assert summary["n_valid"] == 1
    assert failures == ""
    assert task_rows.loc[0, "task_id"] == "suite/good"
