import json
from pathlib import Path

import importlib.util
import sys


def _load_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_paper_results.py"
    spec = importlib.util.spec_from_file_location("build_paper_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def test_build_paper_results_tables_and_plots(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({
        "Nguyen-1": {"r2": 1.0},
        "Jin-1": {"r2": 0.5},
    }))
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(
        json.dumps({
            "name": "Nguyen-1",
            "r2": 0.9,
            "solved": False,
            "diagnostics": [{"selected_reward_rank": 2, "predicted_top1_reward_rank": 3}],
            "task_metadata": {"suite": "nguyen", "n_vars": 1},
        }) + "\n" +
        json.dumps({
            "name": "Jin-1",
            "r2": 1.0,
            "solved": True,
            "diagnostics": [
                {"selected_reward_rank": 1, "predicted_top1_reward_rank": 1, "one_step_rollout_corr": 0.5},
                {"one_step_rollout_corr": 9.0},
            ],
            "task_metadata": {"suite": "jin", "n_vars": 2},
        }) + "\n"
    )
    methods = [
        builder.MethodSpec("base", "Baselines", str(baseline_path.relative_to(tmp_path)), "baseline_json"),
        builder.MethodSpec("ours", "SFSR", str(samples_path.relative_to(tmp_path)), "samples_jsonl"),
    ]

    total, by_suite, dataset, diag = builder.build_tables(tmp_path, methods)
    out = tmp_path / "paper"
    out.mkdir()
    builder.write_csv(out / "paper_total.csv", total, ["group", "method", "coverage", "r2_mean", "r2_median", "solution_rate"])
    builder.plot_totals(total, out)
    builder.plot_suite_solution(by_suite, out)
    builder.plot_diagnostics(diag, out)

    assert total[0]["coverage"] == 2
    assert total[1]["solution_rate"] == 0.5
    assert {r["suite"] for r in by_suite} == {"nguyen", "jin"}
    assert {r["suite"] for r in dataset} == {"all", "nguyen", "jin"}
    assert next(r for r in dataset if r["suite"] == "all")["n_tasks"] == 2
    assert next(r for r in dataset if r["suite"] == "all")["dims"] == '{"1": 1, "2": 1}'
    assert diag[1]["selected_reward_rank_mean"] == 1.5
    assert diag[1]["one_step_rollout_corr_mean"] == 0.5
    assert (out / "paper_total.csv").exists()
    assert (out / "paper_total_r2_solution.png").exists()
    assert (out / "paper_suite_solution.png").exists()
    assert (out / "paper_action_ranking.png").exists()
