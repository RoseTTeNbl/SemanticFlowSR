#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


VAR_RE = re.compile(r"\bx\d+\b")
FUNC_RE = re.compile(r"\b(sin|cos|exp|log|sqrt|Abs|protected_log|protected_sqrt|protected_div)\s*\(")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_finite_float(row.get(key, 0.0)) for row in rows]
    return float(mean(vals)) if vals else 0.0


def _vars(expr: Any) -> set[str]:
    return set(VAR_RE.findall(str(expr or "")))


def _prob_from_logprob(value: Any) -> float:
    logp = _finite_float(value, -1.0e9)
    if logp < -745.0:
        return 0.0
    return float(math.exp(logp))


def _structure_flags(row: dict[str, Any]) -> dict[str, Any]:
    gt = str(row.get("ground_truth", ""))
    raw = str(row.get("raw_expression", row.get("expression", "")))
    gt_vars = _vars(gt)
    raw_vars = _vars(raw)
    raw_clean = raw.strip()
    no_var = len(raw_vars) == 0
    single_var = len(raw_vars) == 1
    identity = bool(re.fullmatch(r"x\d+", raw_clean))
    raw_complexity = _finite_float(row.get("raw_complexity", row.get("complexity", 0.0)))
    raw_r2 = _finite_float(row.get("raw_test_r2_without_affine", 0.0))
    r2 = _finite_float(row.get("r2", 0.0))
    coeffs = row.get("term_fit_coefficients", [])
    coeff0 = _finite_float(coeffs[0], 0.0) if isinstance(coeffs, list) and coeffs else 0.0
    affine_zeroed = abs(coeff0) < 1.0e-8 and len(raw_vars) > 0
    function_count = len(FUNC_RE.findall(raw))
    paren_count = raw.count("(")
    nested_pathology = raw_complexity >= 45.0 or function_count >= 6 or paren_count >= 18
    multivar_gt_single_or_const = len(gt_vars) >= 2 and len(raw_vars) <= 1
    severe = bool(
        no_var
        or affine_zeroed
        or multivar_gt_single_or_const
        or (nested_pathology and raw_r2 < 0.0)
        or (r2 < 0.0 and raw_r2 < 0.0)
    )
    return {
        "gt_vars": sorted(gt_vars),
        "raw_vars": sorted(raw_vars),
        "no_variable": no_var,
        "single_variable": single_var,
        "identity_single_variable": identity,
        "variable_set_match": raw_vars == gt_vars,
        "affine_zeroed": affine_zeroed,
        "nested_pathology": nested_pathology,
        "multivar_gt_single_or_const": multivar_gt_single_or_const,
        "severe_structure_issue": severe,
    }


def analyze(run_dir: Path, *, label: str) -> dict[str, Any]:
    summary = _load_json(run_dir / "typed_op_node_flow_summary.json")
    samples = _load_jsonl(run_dir / "typed_op_node_flow_samples.jsonl")
    if not samples:
        samples = _load_jsonl(run_dir / "typed_op_node_flow_samples.partial.jsonl")
    flags = [_structure_flags(row) for row in samples]
    n = len(samples)
    severe_rows = [
        {
            "task_id": row.get("task_id", ""),
            "gt": row.get("ground_truth", ""),
            "raw": row.get("raw_expression", ""),
            "expr": row.get("expression", ""),
            "r2": _finite_float(row.get("r2", 0.0)),
            "raw_r2": _finite_float(row.get("raw_test_r2_without_affine", 0.0)),
            "endpoint_trace_family_best_active_mean_prob": _finite_float(row.get("endpoint_trace_family_best_active_mean_prob", 0.0)),
            "endpoint_trace_family_best_active_logprob_sum": _finite_float(row.get("endpoint_trace_family_best_active_logprob_sum", 0.0)),
            "gt_trace_sample_probability_estimate": _prob_from_logprob(row.get("endpoint_trace_family_best_active_logprob_sum", -1.0e9)),
            "flags": flag,
        }
        for row, flag in zip(samples, flags)
        if flag["severe_structure_issue"]
    ]
    severe_rows.sort(key=lambda row: (_finite_float(row["r2"]), _finite_float(row["raw_r2"])))
    endpoint_logprobs = [_finite_float(row.get("endpoint_trace_family_best_active_logprob_sum", -1.0e9)) for row in samples]
    endpoint_probs = [_prob_from_logprob(v) for v in endpoint_logprobs]
    out = {
        "label": label,
        "run_dir": str(run_dir),
        "sample_count": n,
        "summary_r2_mean": _finite_float(summary.get("r2_mean", 0.0)),
        "summary_raw_r2_mean": _finite_float(summary.get("raw_test_r2_without_affine_mean", 0.0)),
        "sample_r2_mean": _mean(samples, "r2"),
        "sample_raw_r2_mean": _mean(samples, "raw_test_r2_without_affine"),
        "solution_rate": _finite_float(summary.get("solution_rate", _mean(samples, "solved"))),
        "skeleton_accuracy": _finite_float(summary.get("skeleton_accuracy", 0.0)),
        "operator_dependency_accuracy": _finite_float(summary.get("operator_dependency_accuracy", 0.0)),
        "valid_expression_fraction_mean": _mean(samples, "valid_expression_fraction"),
        "unique_expression_fraction_mean": _mean(samples, "unique_expression_fraction"),
        "no_variable_rate": float(mean([1.0 if f["no_variable"] else 0.0 for f in flags])) if flags else 0.0,
        "single_variable_rate": float(mean([1.0 if f["single_variable"] else 0.0 for f in flags])) if flags else 0.0,
        "identity_single_variable_rate": float(mean([1.0 if f["identity_single_variable"] else 0.0 for f in flags])) if flags else 0.0,
        "variable_set_match_rate": float(mean([1.0 if f["variable_set_match"] else 0.0 for f in flags])) if flags else 0.0,
        "affine_zeroed_rate": float(mean([1.0 if f["affine_zeroed"] else 0.0 for f in flags])) if flags else 0.0,
        "nested_pathology_rate": float(mean([1.0 if f["nested_pathology"] else 0.0 for f in flags])) if flags else 0.0,
        "severe_structure_issue_rate": float(mean([1.0 if f["severe_structure_issue"] else 0.0 for f in flags])) if flags else 0.0,
        "endpoint_trace_family_best_active_mean_prob_mean": _mean(samples, "endpoint_trace_family_best_active_mean_prob"),
        "endpoint_trace_family_best_argmax_match_mean": _mean(samples, "endpoint_trace_family_best_argmax_match"),
        "endpoint_trace_family_best_active_logprob_sum_mean": float(mean(endpoint_logprobs)) if endpoint_logprobs else 0.0,
        "gt_trace_sample_probability_estimate_mean": float(mean(endpoint_probs)) if endpoint_probs else 0.0,
        "gt_trace_sample_probability_estimate_max": float(max(endpoint_probs)) if endpoint_probs else 0.0,
        "terminal_max_prob_mean": _finite_float(summary.get("rollout_terminal_max_prob_mean", _mean(samples, "terminal_max_prob_mean"))),
        "endpoint_masked_terminal_max_prob_mean": _finite_float(summary.get("endpoint_masked_terminal_max_prob_mean", _mean(samples, "endpoint_masked_terminal_max_prob_mean"))),
        "semantic_endpoint_projection_mode": summary.get("semantic_endpoint_projection_mode", ""),
        "semantic_endpoint_p0_per_task": _finite_float(summary.get("semantic_endpoint_p0_per_task", 0.0)),
        "semantic_endpoint_projection_fallback_rate": _finite_float(summary.get("semantic_endpoint_projection_fallback_rate", 0.0)),
        "semantic_endpoint_tilt_accept_rate": _finite_float(summary.get("semantic_endpoint_tilt_accept_rate", 0.0)),
        "semantic_endpoint_target_distance_mean_improvement_mean": _finite_float(summary.get("semantic_endpoint_target_distance_mean_improvement_mean", 0.0)),
        "semantic_endpoint_projected_target_distance_mean_improvement_mean": _finite_float(summary.get("semantic_endpoint_projected_target_distance_mean_improvement_mean", 0.0)),
        "severe_examples": severe_rows[:16],
        "sample_preview": [
            {
                "task_id": row.get("task_id", ""),
                "gt": row.get("ground_truth", ""),
                "raw": row.get("raw_expression", ""),
                "r2": _finite_float(row.get("r2", 0.0)),
                "raw_r2": _finite_float(row.get("raw_test_r2_without_affine", 0.0)),
                "flags": flag,
            }
            for row, flag in list(zip(samples, flags))[:12]
        ],
    }
    return out


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Graph flow run diagnostic: {report['label']}",
        "",
        f"- run dir: `{report['run_dir']}`",
        f"- samples: {int(report['sample_count'])}",
        f"- R2 mean: {report['summary_r2_mean']:.6f}; raw R2 mean: {report['summary_raw_r2_mean']:.6f}",
        f"- solution/skeleton/op-dep: {report['solution_rate']:.6f} / {report['skeleton_accuracy']:.6f} / {report['operator_dependency_accuracy']:.6f}",
        f"- severe structure issue rate: {report['severe_structure_issue_rate']:.6f}",
        f"- no-var/single-var/nested/affine-zeroed: {report['no_variable_rate']:.6f} / {report['single_variable_rate']:.6f} / {report['nested_pathology_rate']:.6f} / {report['affine_zeroed_rate']:.6f}",
        f"- variable-set match rate: {report['variable_set_match_rate']:.6f}",
        f"- valid/unique candidate fraction mean: {report['valid_expression_fraction_mean']:.6f} / {report['unique_expression_fraction_mean']:.6f}",
        f"- GT trace active mean prob: {report['endpoint_trace_family_best_active_mean_prob_mean']:.6f}",
        f"- GT trace sample probability estimate mean/max: {report['gt_trace_sample_probability_estimate_mean']:.6e} / {report['gt_trace_sample_probability_estimate_max']:.6e}",
        f"- terminal max prob mean: {report['terminal_max_prob_mean']:.6f}; masked endpoint max prob mean: {report['endpoint_masked_terminal_max_prob_mean']:.6f}",
        f"- Stage2 projection mode: `{report.get('semantic_endpoint_projection_mode', '')}`; p0/task: {report.get('semantic_endpoint_p0_per_task', 0.0)}",
        f"- Stage2 tilt/projected target-distance improvement: {report['semantic_endpoint_target_distance_mean_improvement_mean']:.6f} / {report['semantic_endpoint_projected_target_distance_mean_improvement_mean']:.6f}",
        "",
        "## Severe Examples",
    ]
    for row in report["severe_examples"][:12]:
        flags = row.get("flags", {})
        lines.append(
            f"- `{row.get('task_id','')}` r2={row.get('r2',0.0):.4f} raw_r2={row.get('raw_r2',0.0):.4f} "
            f"gt=`{row.get('gt','')}` raw=`{row.get('raw','')}` flags={flags}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    label = str(args.label or run_dir.name)
    report = analyze(run_dir, label=label)
    out_json = Path(args.out_json) if args.out_json else run_dir / "graph_flow_diagnostics.json"
    out_md = Path(args.out_md) if args.out_md else run_dir / "graph_flow_diagnostics.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_markdown(report, out_md)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
