#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_complete_expression_semantic_fm import (
    CONSTRUCTION_GRAPHS,
    DEFAULT_OPS,
    FixedSymbolConditionedVelocityNet,
    build_task_bundles,
    canonical_construction_graph,
    evaluate_expression,
    execute_choices,
    expand_ops,
    load_all_task_sources,
    make_construction_template,
    masked_single_block_softmax,
    random_theta,
    rollout,
    sample_choices,
    split_blocks,
    terminal_summary,
    with_structural_metrics,
    _inherit_graph_architecture_from_checkpoint,
    _jsonable,
    _load_matching_state,
    _resolve_device,
)
from semflow_sr.sr.printer import to_string


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _summ(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
    }


def _exp_logprob(value: float) -> float:
    if value < -745.0:
        return 0.0
    return float(math.exp(float(value)))


def _vars(text: str) -> list[str]:
    return sorted(set(re.findall(r"\bx\d+\b", str(text or ""))))


def _trace_probability_stats(
    template: Any,
    theta: torch.Tensor,
    trace: dict[str, Any],
) -> dict[str, float]:
    blocks = split_blocks(theta, template)
    choices = [int(v) for v in trace.get("choices", [])]
    active = [int(v) for v in trace.get("active_block_indices", [])]
    probs: list[float] = []
    matches: list[float] = []
    for bidx in active:
        if bidx < 0 or bidx >= len(blocks) or bidx >= len(choices):
            continue
        action = int(choices[bidx])
        p = masked_single_block_softmax(blocks[bidx].float(), template, bidx)
        if action < 0 or action >= int(p.numel()):
            continue
        prob = float(p[action].detach().cpu().item())
        probs.append(max(prob, 1.0e-12))
        matches.append(float(int(torch.argmax(p).detach().cpu().item()) == action))
    log_probs = [math.log(value) for value in probs]
    logprob_sum = float(np.sum(log_probs)) if log_probs else 0.0
    return {
        "active_block_count": float(len(probs)),
        "active_mean_prob": float(np.mean(probs)) if probs else 0.0,
        "active_min_prob": float(np.min(probs)) if probs else 0.0,
        "active_logprob_mean": float(np.mean(log_probs)) if log_probs else 0.0,
        "active_logprob_sum": float(logprob_sum),
        "exact_trace_probability_estimate": _exp_logprob(logprob_sum),
        "active_argmax_match": float(np.mean(matches)) if matches else 0.0,
        "active_argmax_full_match": float(all(value >= 1.0 for value in matches)) if matches else 0.0,
    }


def _endpoint_family_stats(
    template: Any,
    theta: torch.Tensor,
    task: Any,
) -> dict[str, Any]:
    trace_rows: list[dict[str, Any]] = []
    for trace_index, trace in enumerate(task.traces):
        stats = _trace_probability_stats(template, theta, trace)
        trace_rows.append({"trace_index": int(trace_index), **stats})
    best = max(trace_rows, key=lambda row: float(row["active_logprob_sum"])) if trace_rows else {}
    family_prob_upper = min(1.0, float(sum(_finite(row.get("exact_trace_probability_estimate", 0.0)) for row in trace_rows)))
    term = terminal_summary(theta, template)
    masked_max_probs: list[float] = []
    masked_entropies: list[float] = []
    for bidx, logits in enumerate(split_blocks(theta, template)):
        p = masked_single_block_softmax(logits.float(), template, int(bidx))
        support_count = max(int((p > 0.0).sum().detach().cpu().item()), 2)
        masked_max_probs.append(float(p.max().detach().cpu().item()))
        masked_entropies.append(float((-(p * p.clamp_min(1.0e-8).log()).sum() / math.log(support_count)).detach().cpu().item()))
    return {
        "best_trace_index": int(best.get("trace_index", -1)),
        "best_active_block_count": int(best.get("active_block_count", 0)),
        "best_active_mean_prob": _finite(best.get("active_mean_prob", 0.0)),
        "best_active_min_prob": _finite(best.get("active_min_prob", 0.0)),
        "best_active_logprob_mean": _finite(best.get("active_logprob_mean", 0.0)),
        "best_active_logprob_sum": _finite(best.get("active_logprob_sum", -1.0e9), -1.0e9),
        "best_exact_trace_probability_estimate": _finite(best.get("exact_trace_probability_estimate", 0.0)),
        "best_argmax_match": _finite(best.get("active_argmax_match", 0.0)),
        "best_argmax_full_match": _finite(best.get("active_argmax_full_match", 0.0)),
        "gt_trace_family_probability_upper_bound": float(family_prob_upper),
        "gt_trace_family_argmax_full_match_any": float(any(_finite(row.get("active_argmax_full_match", 0.0)) >= 1.0 for row in trace_rows)),
        "terminal_max_prob_mean": _finite(term.get("terminal_max_prob_mean", 0.0)),
        "terminal_entropy_mean": _finite(term.get("terminal_entropy_mean", 0.0)),
        "endpoint_masked_terminal_max_prob_mean": float(np.mean(masked_max_probs)) if masked_max_probs else 0.0,
        "endpoint_masked_terminal_entropy_mean": float(np.mean(masked_entropies)) if masked_entropies else 0.0,
    }


def _affine_expression(raw: str, coeffs: list[float], intercept: float) -> str:
    coef = float(coeffs[0]) if coeffs else 1.0
    return f"{coef:.6g}*({raw}) + {float(intercept):.6g}"


def _empty_empirical_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "theta1_sample_count": int(args.theta1_samples),
        "theta1_valid_sample_count": 0,
        "theta1_invalid_sample_count": int(args.theta1_samples),
        "theta1_unique_expression_count": 0,
        "empirical_gt_consistent_sample_rate": 0.0,
        "empirical_raw_gt_consistent_sample_rate": 0.0,
        "empirical_symbolic_equiv_rate": 0.0,
        "empirical_skeleton_match_rate": 0.0,
        "empirical_operator_dependency_match_rate": 0.0,
        "empirical_variable_set_match_rate": 0.0,
        "empirical_affine_r2_ge_threshold_rate": 0.0,
        "empirical_raw_r2_ge_threshold_rate": 0.0,
        "empirical_no_variable_rate": 0.0,
        "empirical_single_variable_rate": 0.0,
        "empirical_multi_variable_rate": 0.0,
        "empirical_best_r2": 0.0,
        "empirical_best_raw_r2": 0.0,
        "empirical_best_raw_expression": "",
        "empirical_best_expression": "",
    }


def _empty_structural_metrics() -> dict[str, Any]:
    return {
        "simplified_symbolic_equivalence": False,
        "skeleton_match": False,
        "operator_dependency_match": False,
        "gt_skeleton": "",
        "pred_skeleton": "",
        "operator_dependency_gt": "",
        "operator_dependency_pred": "",
        "formula_bleu": 0.0,
        "formula_token_accuracy": 0.0,
        "formula_edit_distance": 0.0,
    }


def _sample_theta1_expressions(
    model: FixedSymbolConditionedVelocityNet,
    task: Any,
    theta1: torch.Tensor,
    args: argparse.Namespace,
    gen: torch.Generator,
    theta0_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_rows: list[dict[str, Any]] = []
    unique_raw: set[str] = set()
    best: dict[str, Any] | None = None
    gt_vars = _vars(str(task.ground_truth))
    invalid = 0
    for sample_index in range(int(args.theta1_samples)):
        try:
            choices = sample_choices(theta1, model.template, gen)
            expr, _terms, _layers = execute_choices(model.template, choices)
            raw_expr = to_string(expr, int(model.template.num_vars), simplify=False)
            metrics = evaluate_expression(expr, task)
        except Exception as exc:
            invalid += 1
            sample_rows.append({
                "task_id": str(task.task_id),
                "theta0_index": int(theta0_index),
                "sample_index": int(sample_index),
                "eval_status": "invalid",
                "error": str(exc)[:200],
            })
            continue
        unique_raw.add(raw_expr)
        affine_expr = _affine_expression(raw_expr, metrics.get("term_fit_coefficients", [1.0]), metrics.get("term_fit_intercept", 0.0))
        if str(args.sample_eval_mode) == "full":
            structural = with_structural_metrics({
                "task_id": str(task.task_id),
                "suite": str(task.suite),
                "eval_status": "ok",
                "ground_truth": str(task.ground_truth),
                "expression": raw_expr,
                "raw_expression": raw_expr,
            })
        else:
            structural = _empty_structural_metrics()
        pred_vars = _vars(raw_expr)
        variable_set_match = bool(pred_vars == gt_vars)
        symbolic_equiv = bool(structural.get("simplified_symbolic_equivalence", False))
        skeleton_match = bool(structural.get("skeleton_match", False))
        opdep_match = bool(structural.get("operator_dependency_match", False))
        affine_r2 = _finite(metrics.get("r2", 0.0))
        raw_r2 = _finite(metrics.get("raw_test_r2_without_affine", 0.0))
        high_affine = bool(affine_r2 >= float(args.recover_r2_threshold))
        high_raw = bool(raw_r2 >= float(args.recover_raw_r2_threshold))
        structurally_close = bool(variable_set_match and (skeleton_match or opdep_match))
        light_close = bool(variable_set_match and raw_r2 >= float(args.recover_raw_r2_threshold))
        gt_consistent = bool(symbolic_equiv or (structurally_close and high_affine))
        if str(args.sample_eval_mode) == "light":
            gt_consistent = bool(light_close)
        raw_gt_consistent = bool(symbolic_equiv or (structurally_close and high_raw) or light_close)
        row = {
            "task_id": str(task.task_id),
            "suite": str(task.suite),
            "split": str(task.split),
            "theta0_index": int(theta0_index),
            "sample_index": int(sample_index),
            "eval_status": "ok",
            "ground_truth": str(task.ground_truth),
            "raw_expression": raw_expr,
            "expression": affine_expr,
            "gt_vars": gt_vars,
            "pred_vars": pred_vars,
            "variable_set_match": variable_set_match,
            "gt_consistent": gt_consistent,
            "raw_gt_consistent": raw_gt_consistent,
            "affine_r2_ge_threshold": high_affine,
            "raw_r2_ge_threshold": high_raw,
            "complexity": int(getattr(expr, "complexity", 0)),
            **metrics,
            **{k: v for k, v in structural.items() if k not in {"task_id", "suite", "split", "eval_status", "ground_truth", "expression", "raw_expression"}},
        }
        sample_rows.append(row)
        score = affine_r2 + 0.25 * raw_r2 + 0.10 * float(gt_consistent) - 1.0e-3 * float(row["complexity"])
        if best is None or float(score) > float(best.get("_selection_score", -1.0e9)):
            best = dict(row, _selection_score=float(score))
    ok_rows = [row for row in sample_rows if row.get("eval_status") == "ok"]
    if not ok_rows:
        return _empty_empirical_summary(args), sample_rows

    def rate(key: str) -> float:
        return float(mean([1.0 if bool(row.get(key, False)) else 0.0 for row in ok_rows]))

    no_var = 0
    single_var = 0
    multi_var = 0
    for row in ok_rows:
        var_count = len(row.get("pred_vars", []))
        if var_count == 0:
            no_var += 1
        elif var_count == 1:
            single_var += 1
        else:
            multi_var += 1
    best = best or ok_rows[0]
    total = max(len(ok_rows), 1)
    return {
        "theta1_sample_count": int(args.theta1_samples),
        "theta1_valid_sample_count": int(len(ok_rows)),
        "theta1_invalid_sample_count": int(invalid),
        "theta1_unique_expression_count": int(len(unique_raw)),
        "empirical_gt_consistent_sample_rate": rate("gt_consistent"),
        "empirical_raw_gt_consistent_sample_rate": rate("raw_gt_consistent"),
        "empirical_symbolic_equiv_rate": rate("simplified_symbolic_equivalence"),
        "empirical_skeleton_match_rate": rate("skeleton_match"),
        "empirical_operator_dependency_match_rate": rate("operator_dependency_match"),
        "empirical_variable_set_match_rate": rate("variable_set_match"),
        "empirical_affine_r2_ge_threshold_rate": rate("affine_r2_ge_threshold"),
        "empirical_raw_r2_ge_threshold_rate": rate("raw_r2_ge_threshold"),
        "empirical_no_variable_rate": float(no_var / total),
        "empirical_single_variable_rate": float(single_var / total),
        "empirical_multi_variable_rate": float(multi_var / total),
        "empirical_best_r2": _finite(best.get("r2", 0.0)),
        "empirical_best_raw_r2": _finite(best.get("raw_test_r2_without_affine", 0.0)),
        "empirical_best_raw_expression": str(best.get("raw_expression", "")),
        "empirical_best_expression": str(best.get("expression", "")),
    }, sample_rows


def _task_probe_rows(
    model: FixedSymbolConditionedVelocityNet,
    task: Any,
    args: argparse.Namespace,
    gen: torch.Generator,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for theta0_index in range(int(args.theta0_samples)):
        theta0 = random_theta(model.template, scale=float(args.theta0_noise_scale), device=device)
        theta1, rollout_diag = rollout(
            model,
            None,
            task,
            theta0,
            steps=int(args.ode_steps),
            mode="off",
            args=args,
            generator=gen,
        )
        family_diag = _endpoint_family_stats(model.template, theta1, task)
        empirical_diag, empirical_rows = _sample_theta1_expressions(model, task, theta1, args, gen, int(theta0_index))
        sample_rows.extend(empirical_rows)
        rows.append({
            "task_id": str(task.task_id),
            "suite": str(task.suite),
            "split": str(task.split),
            "theta0_index": int(theta0_index),
            "gt_trace_count": int(len(task.traces)),
            **family_diag,
            **empirical_diag,
            "rollout_guidance_step_count": _finite(rollout_diag.get("guidance_step_count", 0.0)),
        })
    return rows, sample_rows


def _task_summary(task_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    probs = [_finite(row["best_exact_trace_probability_estimate"]) for row in rows]
    family_probs = [_finite(row.get("gt_trace_family_probability_upper_bound", 0.0)) for row in rows]
    mean_probs = [_finite(row["best_active_mean_prob"]) for row in rows]
    argmax = [_finite(row["best_argmax_match"]) for row in rows]
    logsum = [_finite(row["best_active_logprob_sum"]) for row in rows]
    empirical = [_finite(row.get("empirical_gt_consistent_sample_rate", 0.0)) for row in rows]
    empirical_raw = [_finite(row.get("empirical_raw_gt_consistent_sample_rate", 0.0)) for row in rows]
    best_r2 = [_finite(row.get("empirical_best_r2", 0.0)) for row in rows]
    best_raw_r2 = [_finite(row.get("empirical_best_raw_r2", 0.0)) for row in rows]
    var_match = [_finite(row.get("empirical_variable_set_match_rate", 0.0)) for row in rows]
    no_var = [_finite(row.get("empirical_no_variable_rate", 0.0)) for row in rows]
    single_var = [_finite(row.get("empirical_single_variable_rate", 0.0)) for row in rows]
    multi_var = [_finite(row.get("empirical_multi_variable_rate", 0.0)) for row in rows]
    return {
        "task_id": task_id,
        "theta0_samples": int(len(rows)),
        "recover_prob": _summ(probs),
        "trace_family_probability_upper_bound": _summ(family_probs),
        "empirical_gt_consistent_sample_rate": _summ(empirical),
        "empirical_raw_gt_consistent_sample_rate": _summ(empirical_raw),
        "empirical_best_r2": _summ(best_r2),
        "empirical_best_raw_r2": _summ(best_raw_r2),
        "empirical_variable_set_match_rate": _summ(var_match),
        "empirical_no_variable_rate": _summ(no_var),
        "empirical_single_variable_rate": _summ(single_var),
        "empirical_multi_variable_rate": _summ(multi_var),
        "active_mean_prob": _summ(mean_probs),
        "argmax_match": _summ(argmax),
        "active_logprob_sum": _summ(logsum),
        "stable_recover_prob_rate_ge_1e_3": float(mean([1.0 if v >= 1.0e-3 else 0.0 for v in probs])) if probs else 0.0,
        "stable_recover_prob_rate_ge_1e_6": float(mean([1.0 if v >= 1.0e-6 else 0.0 for v in probs])) if probs else 0.0,
        "theta0_rate_any_empirical_gt_consistent": float(mean([1.0 if v > 0.0 else 0.0 for v in empirical])) if empirical else 0.0,
        "theta0_rate_empirical_gt_consistent_ge_0_01": float(mean([1.0 if v >= 0.01 else 0.0 for v in empirical])) if empirical else 0.0,
        "theta0_rate_empirical_gt_consistent_ge_0_10": float(mean([1.0 if v >= 0.10 else 0.0 for v in empirical])) if empirical else 0.0,
    }


def _write_md(report: dict[str, Any], path: Path) -> None:
    s = report["summary"]
    lines = [
        "# GT Trace Recovery Over Theta0 Probe",
        "",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- tasks: {s['task_count']}; theta0/task: {report['config']['theta0_samples']}",
        f"- theta1 expression samples/theta0: {report['config']['theta1_samples']}",
        f"- exact trace probability mean: {s['recover_prob_mean']:.6e}",
        f"- trace family probability upper-bound mean: {s['trace_family_probability_upper_bound_mean']:.6e}",
        f"- empirical GT-consistent sample rate mean: {s['empirical_gt_consistent_sample_rate_mean']:.6f}",
        f"- empirical any-consistent theta0 rate: {s['theta0_rate_any_empirical_gt_consistent']:.6f}",
        f"- empirical best affine R2 mean: {s['empirical_best_r2_mean']:.6f}",
        f"- empirical best raw R2 mean: {s['empirical_best_raw_r2_mean']:.6f}",
        f"- empirical no/single/multi variable rates: {s['empirical_no_variable_rate_mean']:.6f} / {s['empirical_single_variable_rate_mean']:.6f} / {s['empirical_multi_variable_rate_mean']:.6f}",
        f"- exact trace probability median across task means: {s['recover_prob_task_median']:.6e}",
        f"- active mean prob mean: {s['active_mean_prob_mean']:.6f}",
        f"- argmax match mean: {s['argmax_match_mean']:.6f}",
        f"- task stable rate prob>=1e-3: {s['task_rate_recover_prob_ge_1e_3']:.6f}",
        f"- task stable rate prob>=1e-6: {s['task_rate_recover_prob_ge_1e_6']:.6f}",
        "",
        "## Lowest Recovery Tasks",
    ]
    for row in sorted(report["task_summaries"], key=lambda item: float(item["recover_prob"]["mean"]))[:12]:
        lines.append(
            f"- `{row['task_id']}` mean={row['recover_prob']['mean']:.3e} "
            f"emp={row['empirical_gt_consistent_sample_rate']['mean']:.3f} "
            f"best_r2={row['empirical_best_r2']['mean']:.3f} "
            f"p75={row['recover_prob']['p75']:.3e} active_mean={row['active_mean_prob']['mean']:.4f} "
            f"argmax={row['argmax_match']['mean']:.4f}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default="data/benchmark_suites/benchmark_manifest.json")
    parser.add_argument("--manifest-root", default="data/benchmark_suites")
    parser.add_argument("--suites", nargs="+", default=["nguyen", "constant", "livermore", "jin"])
    parser.add_argument("--symbolicgpt-root", default="")
    parser.add_argument("--symbolicgpt-train-limit", type=int, default=0)
    parser.add_argument("--symbolicgpt-eval-limit", type=int, default=0)
    parser.add_argument("--symbolicgpt-eval-splits", default="val,test")
    parser.add_argument("--symbolicgpt-point-train-fraction", type=float, default=0.8)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--train-task-limit", type=int, default=0)
    parser.add_argument("--eval-task-limit", type=int, default=20)
    parser.add_argument("--max-train-points", type=int, default=64)
    parser.add_argument("--max-eval-points", type=int, default=64)
    parser.add_argument("--construction-graph", choices=list(CONSTRUCTION_GRAPHS), default="graph_dag_edge_simplex")
    parser.add_argument("--num-vars", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-registers", type=int, default=0)
    parser.add_argument("--ops", default=",".join(DEFAULT_OPS))
    parser.add_argument("--op-copies", type=int, default=1)
    parser.add_argument("--output-terms", type=int, default=1)
    parser.add_argument("--gt-traces-per-task", type=int, default=4)
    parser.add_argument("--trace-copy-assignment", choices=["canonical", "random"], default="canonical")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--semantic-action-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--active-node-semantic-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--velocity-parameterization", choices=["direct_velocity", "endpoint_bridge"], default="direct_velocity")
    parser.add_argument("--global-state-mode", choices=["summary", "full"], default="summary")
    parser.add_argument("--metadata-embedding-dim", type=int, default=0)
    parser.add_argument("--task-encoder-mode", choices=["point_mlp", "stats", "hybrid_stats"], default="point_mlp")
    parser.add_argument("--task-conditioning", choices=["auto", "off", "xy", "xy_residual"], default="auto")
    parser.add_argument("--theta0-samples", type=int, default=16)
    parser.add_argument("--theta1-samples", type=int, default=8)
    parser.add_argument("--sample-eval-mode", choices=["light", "full"], default="light")
    parser.add_argument("--recover-r2-threshold", type=float, default=0.95)
    parser.add_argument("--recover-raw-r2-threshold", type=float, default=0.95)
    parser.add_argument("--theta0-noise-scale", type=float, default=1.0)
    parser.add_argument("--ode-steps", type=int, default=64)
    parser.add_argument("--rollout-velocity-gain", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    device = _resolve_device(str(args.device))
    ckpt = torch.load(str(args.checkpoint), map_location=device)
    _inherit_graph_architecture_from_checkpoint(args, ckpt)
    graph_family = canonical_construction_graph(str(getattr(args, "construction_graph", "graph_dag_edge_simplex")))
    if graph_family == "token_policy":
        raise ValueError("GT trace recovery probe supports graph_dag_edge_simplex and register_categorical_blocks checkpoints")
    task_conditioning = str(args.task_conditioning)
    if task_conditioning == "auto":
        summary = ckpt.get("summary", {}) if isinstance(ckpt, dict) else {}
        task_conditioning = str(summary.get("task_conditioning", "xy"))
        if task_conditioning == "auto":
            task_conditioning = "xy"

    template = make_construction_template(args, graph_family)
    model = FixedSymbolConditionedVelocityNet(
        template,
        hidden=int(args.hidden),
        semantic_features=bool(args.semantic_action_features),
        active_node_semantic_features=bool(args.active_node_semantic_features),
        velocity_parameterization=str(args.velocity_parameterization),
        global_state_mode=str(args.global_state_mode),
        metadata_embedding_dim=int(args.metadata_embedding_dim),
        task_encoder_mode=str(args.task_encoder_mode),
        task_conditioning=task_conditioning,
    ).to(device)
    _load_matching_state(model, ckpt["model"])
    model.eval()
    train_src, eval_src, source_counts = load_all_task_sources(args, int(args.num_vars), torch.device("cpu"))
    eval_tasks = build_task_bundles(
        eval_src,
        template,
        traces_per_task=int(args.gt_traces_per_task),
        max_train_points=int(args.max_train_points),
        max_eval_points=int(args.max_eval_points),
        device=device,
        seed=int(args.seed) + 12345,
        split="eval",
        copy_assignment=str(args.trace_copy_assignment),
    )
    eval_tasks = [task for task in eval_tasks if task.traces]
    gen = torch.Generator(device=device).manual_seed(int(args.seed) + 77_771)
    rows: list[dict[str, Any]] = []
    expression_sample_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for task in eval_tasks:
            task_rows, task_sample_rows = _task_probe_rows(model, task, args, gen, device)
            rows.extend(task_rows)
            expression_sample_rows.extend(task_sample_rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["task_id"]), []).append(row)
    task_summaries = [_task_summary(task_id, item_rows) for task_id, item_rows in grouped.items()]
    task_recover_means = [float(row["recover_prob"]["mean"]) for row in task_summaries]
    task_family_prob_means = [float(row["trace_family_probability_upper_bound"]["mean"]) for row in task_summaries]
    task_empirical_means = [float(row["empirical_gt_consistent_sample_rate"]["mean"]) for row in task_summaries]
    task_active_means = [float(row["active_mean_prob"]["mean"]) for row in task_summaries]
    task_argmax_means = [float(row["argmax_match"]["mean"]) for row in task_summaries]
    empirical_rates = [_finite(row.get("empirical_gt_consistent_sample_rate", 0.0)) for row in rows]
    empirical_raw_rates = [_finite(row.get("empirical_raw_gt_consistent_sample_rate", 0.0)) for row in rows]
    empirical_best_r2 = [_finite(row.get("empirical_best_r2", 0.0)) for row in rows]
    empirical_best_raw_r2 = [_finite(row.get("empirical_best_raw_r2", 0.0)) for row in rows]
    family_probs = [_finite(row.get("gt_trace_family_probability_upper_bound", 0.0)) for row in rows]
    summary = {
        "task_count": int(len(task_summaries)),
        "row_count": int(len(rows)),
        "expression_sample_row_count": int(len(expression_sample_rows)),
        "recover_prob_mean": float(mean([_finite(row["best_exact_trace_probability_estimate"]) for row in rows])) if rows else 0.0,
        "recover_prob_task_median": float(np.median(task_recover_means)) if task_recover_means else 0.0,
        "trace_family_probability_upper_bound_mean": float(mean(family_probs)) if family_probs else 0.0,
        "trace_family_probability_upper_bound_task_median": float(np.median(task_family_prob_means)) if task_family_prob_means else 0.0,
        "empirical_gt_consistent_sample_rate_mean": float(mean(empirical_rates)) if empirical_rates else 0.0,
        "empirical_gt_consistent_sample_rate_task_median": float(np.median(task_empirical_means)) if task_empirical_means else 0.0,
        "empirical_raw_gt_consistent_sample_rate_mean": float(mean(empirical_raw_rates)) if empirical_raw_rates else 0.0,
        "empirical_best_r2_mean": float(mean(empirical_best_r2)) if empirical_best_r2 else 0.0,
        "empirical_best_raw_r2_mean": float(mean(empirical_best_raw_r2)) if empirical_best_raw_r2 else 0.0,
        "empirical_no_variable_rate_mean": float(mean([_finite(row.get("empirical_no_variable_rate", 0.0)) for row in rows])) if rows else 0.0,
        "empirical_single_variable_rate_mean": float(mean([_finite(row.get("empirical_single_variable_rate", 0.0)) for row in rows])) if rows else 0.0,
        "empirical_multi_variable_rate_mean": float(mean([_finite(row.get("empirical_multi_variable_rate", 0.0)) for row in rows])) if rows else 0.0,
        "active_mean_prob_mean": float(mean([_finite(row["best_active_mean_prob"]) for row in rows])) if rows else 0.0,
        "active_mean_prob_task_median": float(np.median(task_active_means)) if task_active_means else 0.0,
        "argmax_match_mean": float(mean([_finite(row["best_argmax_match"]) for row in rows])) if rows else 0.0,
        "argmax_match_task_median": float(np.median(task_argmax_means)) if task_argmax_means else 0.0,
        "task_rate_recover_prob_ge_1e_3": float(mean([1.0 if row["recover_prob"]["mean"] >= 1.0e-3 else 0.0 for row in task_summaries])) if task_summaries else 0.0,
        "task_rate_recover_prob_ge_1e_6": float(mean([1.0 if row["recover_prob"]["mean"] >= 1.0e-6 else 0.0 for row in task_summaries])) if task_summaries else 0.0,
        "theta0_rate_any_empirical_gt_consistent": float(mean([1.0 if value > 0.0 else 0.0 for value in empirical_rates])) if empirical_rates else 0.0,
        "theta0_rate_empirical_gt_consistent_ge_0_01": float(mean([1.0 if value >= 0.01 else 0.0 for value in empirical_rates])) if empirical_rates else 0.0,
        "theta0_rate_empirical_gt_consistent_ge_0_10": float(mean([1.0 if value >= 0.10 else 0.0 for value in empirical_rates])) if empirical_rates else 0.0,
    }
    report = {
        "checkpoint": str(args.checkpoint),
        "config": {
            "theta0_samples": int(args.theta0_samples),
            "theta1_samples": int(args.theta1_samples),
            "sample_eval_mode": str(args.sample_eval_mode),
            "recover_r2_threshold": float(args.recover_r2_threshold),
            "recover_raw_r2_threshold": float(args.recover_raw_r2_threshold),
            "ode_steps": int(args.ode_steps),
            "num_layers": int(args.num_layers),
            "num_registers": int(getattr(template, "register_count", 0)),
            "register_count": int(getattr(template, "register_count", 0)),
            "output_terms": int(args.output_terms),
            "hidden": int(args.hidden),
            "construction_graph": str(args.construction_graph),
            "construction_family": str(graph_family),
            "semantic_action_features": bool(args.semantic_action_features),
            "active_node_semantic_features": bool(args.active_node_semantic_features),
            "task_conditioning": str(task_conditioning),
        },
        "source_counts": source_counts,
        "summary": summary,
        "task_summaries": task_summaries,
        "rows": rows,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gt_trace_recovery_over_theta0.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (out_dir / "gt_trace_recovery_over_theta0.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    with (out_dir / "theta1_expression_samples.jsonl").open("w") as f:
        for row in expression_sample_rows:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    _write_md(report, out_dir / "gt_trace_recovery_over_theta0.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
