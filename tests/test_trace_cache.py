from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semflow_sr.flow.trace_cache import load_trace_cache, trace_record, write_trace_cache
from semflow_sr.sr.ast import eval_expr
from semflow_sr.sr.parser import parse_formula
from scripts.train_complete_expression_semantic_fm import (
    DEFAULT_OPS,
    RegisterOperatorSimplexTemplate,
    TaskBundle,
    _bootstrap_gate_decision,
    compile_expr_to_register_trace,
    execute_choices,
)


def _template() -> RegisterOperatorSimplexTemplate:
    return RegisterOperatorSimplexTemplate(
        num_vars=1, num_layers=6, num_registers=9,
        ops=tuple(DEFAULT_OPS[:9]), output_terms=1,
    )


def _task(template: RegisterOperatorSimplexTemplate) -> TaskBundle:
    x = torch.linspace(-1.0, 1.0, 32).unsqueeze(1)
    expression = "x0**3 + x0**2 + x0"
    y = eval_expr(parse_formula(expression, ["x0"]), x)
    return TaskBundle(
        task_id="unit/equivalent", suite="unit", split="train", num_vars=1,
        variable_names=["x0"], x_train=x, y_train=y, x_test=x.clone(), y_test=y.clone(),
        ground_truth=expression, traces=[], compile_failures=[],
    )


def test_equivalent_register_traces_preserve_semantics_and_vary_choices() -> None:
    template = _template()
    task = _task(template)
    expr = parse_formula(task.ground_truth, task.variable_names)
    traces = [compile_expr_to_register_trace(template, expr, random.Random(seed)) for seed in range(16)]
    unique = {tuple(trace["choices"]) for trace in traces}
    assert len(unique) >= 2
    for trace in traces:
        decoded, _terms, _layers = execute_choices(template, trace["choices"])
        assert torch.allclose(eval_expr(decoded, task.x_train), task.y_train, atol=1.0e-6)
        trace["semantic_oracle_raw_r2"] = 1.0
    task.traces = traces


def test_trace_cache_round_trip_and_checksum_rejection(tmp_path: Path) -> None:
    template = _template()
    task = _task(template)
    expr = parse_formula(task.ground_truth, task.variable_names)
    traces = []
    for seed in range(8):
        trace = compile_expr_to_register_trace(template, expr, random.Random(seed))
        trace["semantic_oracle_raw_r2"] = 1.0
        traces.append(trace)
    write_trace_cache(tmp_path, template, [trace_record(task, template, traces, [])])
    records, manifest = load_trace_cache(tmp_path, template)
    assert manifest["record_count"] == 1
    assert len(records[task.task_id]["traces"]) == 8
    records_path = tmp_path / "compiled_trace_families_v1.jsonl"
    records_path.write_text(records_path.read_text() + "\n")
    try:
        load_trace_cache(tmp_path, template)
    except ValueError as exc:
        assert "checksum" in str(exc)
    else:
        raise AssertionError("modified cache must be rejected")


def test_bootstrap_gate_rejects_missing_or_weak_metrics() -> None:
    bootstrap = {
        "flow_relative_fisher_loss": 0.04,
        "flow_inactive_relative_to_active_zero": 0.005,
        **{f"flow_{key}_relative_fisher_loss": 0.09 for key in ("low", "mid", "high", "readout", "op", "arg")},
    }
    population = {
        "fitted_r2_gt_0_95_rate": 1.0,
        "raw_r2_gt_0_999_rate": 1.0,
        "gt_trace_family_hit_rate": 1.0,
        "skeleton_match_rate": 1.0,
        "operator_dependency_match_rate": 1.0,
    }
    assert _bootstrap_gate_decision("A", bootstrap, population)["passed"]
    bootstrap["flow_arg_relative_fisher_loss"] = 0.11
    assert not _bootstrap_gate_decision("A", bootstrap, population)["passed"]
