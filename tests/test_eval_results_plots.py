import csv

from semflow_sr.eval.evaluator import EvalReport
from semflow_sr.eval.results import save_results


def test_save_results_writes_metric_csv_without_default_plots(tmp_path):
    reports = [
        EvalReport(
            name="a",
            r2=0.9,
            nmse=0.1,
            complexity=3,
            expression="x0",
            energy_trace=[2.0, 1.0, 0.5],
            solved=True,
            steps=2,
            energy_decrease=0.75,
        ),
        EvalReport(
            name="b",
            r2=0.5,
            nmse=0.5,
            complexity=5,
            expression="x1",
            energy_trace=[3.0, 2.0],
            solved=False,
            steps=1,
            energy_decrease=0.33,
        ),
    ]

    save_results(reports, tmp_path, "demo")

    metrics_path = tmp_path / "demo_metrics.csv"
    assert metrics_path.exists()
    rows = list(csv.DictReader(metrics_path.open()))
    assert [r["name"] for r in rows] == ["a", "b"]
    assert not (tmp_path / "demo_r2_curve.png").exists()
    assert not (tmp_path / "demo_energy_traces.png").exists()


def test_save_results_can_write_explicit_per_task_plots(tmp_path):
    reports = [
        EvalReport(
            name="a",
            r2=0.9,
            nmse=0.1,
            complexity=3,
            expression="x0",
            energy_trace=[2.0, 1.0, 0.5],
            solved=True,
            steps=2,
            energy_decrease=0.75,
        )
    ]

    save_results(reports, tmp_path, "demo", make_plots=True)

    assert (tmp_path / "demo_r2_curve.png").exists()
    assert (tmp_path / "demo_energy_traces.png").exists()
