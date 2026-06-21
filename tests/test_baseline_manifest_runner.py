import json

import numpy as np

from semflow_sr.data.benchmark_manifest import BenchmarkSuiteSpec, write_benchmark_manifest
from semflow_sr.data.benchmark_prepare import materialize_arrays
from semflow_sr.eval.baseline_runner import collect_tasks, run_baseline_records
from semflow_sr.eval.baseline_sanity import summarize_sanity_results


def _make_manifest(tmp_path):
    root = tmp_path / "materialized"
    a = materialize_arrays(
        task_id="suite_a/A",
        suite="suite_a",
        root=root,
        name="A",
        X_train=np.array([[0.0], [1.0]]),
        y_train=np.array([0.0, 1.0]),
        X_test=np.array([[2.0]]),
        y_test=np.array([2.0]),
        variable_names=["x0"],
    )
    b = materialize_arrays(
        task_id="suite_b/B",
        suite="suite_b",
        root=root,
        name="B",
        X_train=np.array([[0.0], [1.0]]),
        y_train=np.array([1.0, 2.0]),
        X_test=np.array([[2.0]]),
        y_test=np.array([3.0]),
        variable_names=["x0"],
    )
    manifest = BenchmarkSuiteSpec(version="1.0", suites={"suite_a": [a], "suite_b": [b]})
    path = tmp_path / "manifest.json"
    write_benchmark_manifest(manifest, path)
    return path


def test_collect_tasks_reads_manifest_and_filters_suites(tmp_path):
    manifest_path = _make_manifest(tmp_path)

    tasks = collect_tasks(manifest=manifest_path, suites=["suite_b"], root=tmp_path)

    assert [t.name for t in tasks] == ["suite_b/B"]
    assert tasks[0].metadata["suite"] == "suite_b"


def test_collect_tasks_limit_supports_baseline_smoke_runs(tmp_path):
    manifest_path = _make_manifest(tmp_path)

    tasks = collect_tasks(manifest=manifest_path, root=tmp_path, limit=1)

    assert len(tasks) == 1
    assert tasks[0].name == "suite_a/A"


def test_collect_tasks_legacy_87_combines_formula_manifest_and_feynman_loader(tmp_path):
    manifest_path = _make_manifest(tmp_path)
    feynman_root = tmp_path / "feynman"
    task_dir = feynman_root / "F1"
    task_dir.mkdir(parents=True)
    (task_dir / "metadata.json").write_text(json.dumps({"name": "F1", "n_vars": 1}))
    (task_dir / "seed_0_train.csv").write_text("x0,target\n0,0\n1,1\n")
    (task_dir / "seed_0_test.csv").write_text("x0,target\n2,2\n")

    tasks = collect_tasks(
        manifest=manifest_path,
        suites=["suite_a"],
        root=tmp_path,
        legacy_87=True,
        feynman_root=feynman_root,
        seed=0,
    )

    assert [task.name for task in tasks] == ["suite_a/A", "F1"]


def test_run_baseline_records_writes_common_result_schema(tmp_path):
    manifest_path = _make_manifest(tmp_path)
    tasks = collect_tasks(manifest=manifest_path, root=tmp_path)

    out = run_baseline_records(
        tasks,
        lambda Xtr, ytr, Xte, yte: {
            "r2": 1.0,
            "nmse": 0.0,
            "expression": "x0",
        },
        out_path=tmp_path / "results.json",
        method="fake",
        budget={"seconds": 1},
    )
    saved = json.loads((tmp_path / "results.json").read_text())

    assert set(out) == {"suite_a/A", "suite_b/B"}
    assert saved["suite_a/A"]["method"] == "fake"
    assert saved["suite_a/A"]["suite"] == "suite_a"
    assert saved["suite_a/A"]["budget"] == {"seconds": 1}
    assert saved["suite_a/A"]["n_train"] == 2
    assert saved["suite_a/A"]["status"] == "ok"
    assert "runtime_sec" in saved["suite_a/A"]


def test_run_baseline_records_keeps_going_after_task_failure(tmp_path):
    manifest_path = _make_manifest(tmp_path)
    tasks = collect_tasks(manifest=manifest_path, root=tmp_path)

    def sometimes_fails(Xtr, ytr, Xte, yte):
        if float(ytr[0]) == 0.0:
            raise RuntimeError("boom")
        return {"r2": 0.5, "nmse": 0.1, "expression": "x0 + 1"}

    out = run_baseline_records(
        tasks,
        sometimes_fails,
        out_path=tmp_path / "results.json",
        method="fragile",
        budget={"seconds": 1},
    )

    assert out["suite_a/A"]["status"] == "failed"
    assert out["suite_a/A"]["error_type"] == "RuntimeError"
    assert "boom" in out["suite_a/A"]["error"]
    assert out["suite_b/B"]["status"] == "ok"


def test_run_baseline_records_writes_incrementally(tmp_path):
    manifest_path = _make_manifest(tmp_path)
    tasks = collect_tasks(manifest=manifest_path, root=tmp_path)
    out_path = tmp_path / "results.json"
    seen_during_second = {}

    def records_file_state(Xtr, ytr, Xte, yte):
        if out_path.exists():
            seen_during_second.update(json.loads(out_path.read_text()))
        return {"r2": 1.0, "nmse": 0.0, "expression": "x0"}

    run_baseline_records(
        tasks,
        records_file_state,
        out_path=out_path,
        method="incremental",
    )

    assert "suite_a/A" in seen_during_second
    assert "suite_b/B" not in seen_during_second


def test_run_baseline_records_resumes_existing_results(tmp_path):
    manifest_path = _make_manifest(tmp_path)
    tasks = collect_tasks(manifest=manifest_path, root=tmp_path)
    out_path = tmp_path / "results.json"
    out_path.write_text(json.dumps({
        "suite_a/A": {
            "task_id": "suite_a/A",
            "method": "resume",
            "r2": 0.5,
            "status": "ok",
        }
    }))
    calls = []

    def record_call(Xtr, ytr, Xte, yte):
        calls.append(float(ytr[0]))
        return {"r2": 1.0, "nmse": 0.0, "expression": "x0"}

    out = run_baseline_records(
        tasks,
        record_call,
        out_path=out_path,
        method="resume",
        resume=True,
    )

    assert calls == [1.0]
    assert out["suite_a/A"]["r2"] == 0.5
    assert out["suite_b/B"]["r2"] == 1.0


def test_baseline_sanity_summary_flags_failed_simple_tasks():
    records = {
        "sanity/y=x": {"r2": 0.9999, "method": "fake"},
        "sanity/y=x2": {"r2": -0.2, "method": "fake"},
    }

    summary, failed = summarize_sanity_results(records, threshold=0.99)

    assert summary["n_tasks"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert failed[0]["task_id"] == "sanity/y=x2"
