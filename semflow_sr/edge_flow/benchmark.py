"""Benchmark loading and result files for Edge Flow evaluations."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import sympy as sp
import torch

from ..data.benchmark_loader import SRTask
from ..eval.baseline_runner import collect_tasks


def load_edge_flow_benchmark_tasks(
    *,
    manifest: str | Path,
    suites: list[str] | None,
    root: str | Path,
    seed: int,
    legacy_87: bool,
    feynman_root: str | Path,
    limit: int | None = None,
) -> list[SRTask]:
    return collect_tasks(
        manifest=manifest,
        suites=suites,
        root=root,
        seed=seed,
        legacy_87=legacy_87,
        feynman_root=feynman_root,
        limit=limit,
    )


def task_tensors(
    task: SRTask,
    *,
    template_num_vars: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_train = _pad_features(torch.tensor(task.X_train, dtype=torch.float32), template_num_vars)
    x_test = _pad_features(torch.tensor(task.X_test, dtype=torch.float32), template_num_vars)
    y_train = torch.tensor(task.y_train, dtype=torch.float32)
    y_test = torch.tensor(task.y_test, dtype=torch.float32)
    return x_train, y_train, x_test, y_test


def write_benchmark_result_files(records: list[dict[str, Any]], out: str | Path, tag: str) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = [with_skeleton_metrics(row) for row in records]
    summary = summarize_benchmark_records(records)
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    with (out / f"{tag}_samples.jsonl").open("w") as f:
        for row in records:
            f.write(json.dumps(_jsonable(row)) + "\n")
    (out / f"{tag}_diagnostics.json").write_text(json.dumps(_diagnostic_records(records), indent=2))
    _write_task_expressions(records, out / f"{tag}_task_expressions.csv")
    _write_task_expressions_markdown(records, out / f"{tag}_task_expressions.md")
    stats = grouped_statistics(records)
    _write_group_stats(stats, out / f"{tag}_statistics_by_group.csv")
    (out / f"{tag}_statistics_by_group.json").write_text(json.dumps(stats, indent=2))
    return summary


def summarize_benchmark_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    stats = _aggregate(records)
    return {
        "n_tasks": int(stats["n_tasks"]),
        "r2_mean": stats["r2_mean"],
        "nmse_mean": stats["nmse_mean"],
        "solution_rate": stats["solution_rate"],
        "skeleton_accuracy": stats["skeleton_accuracy"],
        "complexity_mean": stats["complexity_mean"],
        "valid_expression_fraction_mean": stats["valid_expression_fraction_mean"],
        "unique_expression_fraction_mean": stats["unique_expression_fraction_mean"],
    }


def grouped_statistics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("all", "all", records)]
    for suite in sorted({str(row.get("suite", "unknown")) for row in records}):
        groups.append(("suite", suite, [row for row in records if str(row.get("suite", "unknown")) == suite]))
    for num_vars in sorted({int(row.get("num_vars", 0)) for row in records}):
        groups.append(("num_vars", str(num_vars), [row for row in records if int(row.get("num_vars", 0)) == num_vars]))
    jin = [row for row in records if str(row.get("suite", "")) == "jin"]
    non_jin = [row for row in records if str(row.get("suite", "")) != "jin"]
    if jin:
        groups.append(("jin_vs_87", "jin", jin))
        groups.append(("jin_vs_87", "non_jin_87", non_jin))
    rows = []
    for group_type, group_value, subset in groups:
        item = _aggregate(subset)
        item.update({"group_type": group_type, "group_value": group_value})
        rows.append(item)
    return rows


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    return {
        "n_tasks": int(n),
        "r2_mean": _mean(records, "r2"),
        "nmse_mean": _mean(records, "nmse"),
        "solution_rate": _mean_bool(records, "solved"),
        "skeleton_accuracy": _mean_bool(records, "skeleton_match"),
        "complexity_mean": _mean(records, "complexity"),
        "reward_mean": _mean(records, "reward"),
        "valid_expression_fraction_mean": _mean(records, "valid_expression_fraction"),
        "unique_expression_fraction_mean": _mean(records, "unique_expression_fraction"),
    }


def _write_task_expressions(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "task_id",
        "suite",
        "num_vars",
        "ground_truth",
        "expression",
        "raw_expression",
        "gt_skeleton",
        "pred_skeleton",
        "skeleton_match",
        "r2",
        "nmse",
        "reward",
        "complexity",
        "raw_test_r2_without_affine",
        "calibration_gain",
        "train_test_r2_gap",
        "used_variable_count",
        "root_operator",
        "output_depth",
        "prior_best_r2",
        "prior_best_skeleton_match",
        "theta_star_best_r2",
        "theta_star_best_skeleton_match",
        "theta_star_projection_drop",
        "valid_expression_fraction",
        "unique_expression_fraction",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_task_expressions_markdown(records: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "| task_id | suite | vars | R2 | skeleton | GT expression | generated expression |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in records:
        lines.append(
            "| {task_id} | {suite} | {num_vars} | {r2:.6g} | {skeleton} | {gt} | {expr} |".format(
                task_id=_md(row.get("task_id", "")),
                suite=_md(row.get("suite", "")),
                num_vars=int(row.get("num_vars", 0)),
                r2=float(row.get("r2", 0.0)),
                skeleton=int(bool(row.get("skeleton_match", False))),
                gt=_md(row.get("ground_truth", "")),
                expr=_md(row.get("expression", "")),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _write_group_stats(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "group_type",
        "group_value",
        "n_tasks",
        "r2_mean",
        "nmse_mean",
        "solution_rate",
        "skeleton_accuracy",
        "complexity_mean",
        "reward_mean",
        "valid_expression_fraction_mean",
        "unique_expression_fraction_mean",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _diagnostic_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = {
        "task_id",
        "suite",
        "num_vars",
        "decoder_budget_curve",
        "prior_oracle_samples",
        "prior_best_r2",
        "prior_best_skeleton_match",
        "prior_best_expression",
        "theta_star_decode_samples",
        "theta_star_best_r2",
        "theta_star_best_skeleton_match",
        "theta_star_best_expression",
        "theta_star_projection_drop",
        "projection_per_mode_elite_count",
        "projection_per_mode_best_reward",
        "projection_target_edge_entropy_mean",
        "projection_target_ess",
        "calibration_gain",
        "train_test_r2_gap",
        "used_variable_count",
        "operator_histogram",
        "template_num_edge_groups",
        "template_candidate_count_mean",
        "template_candidate_count_max",
    }
    return [{key: _jsonable(row.get(key)) for key in sorted(keys) if key in row} for row in records]


def _pad_features(x: torch.Tensor, template_num_vars: int) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"expected 2D feature matrix, got shape {tuple(x.shape)}")
    if x.shape[1] > int(template_num_vars):
        raise ValueError(f"task has {x.shape[1]} variables but template supports {template_num_vars}")
    if x.shape[1] == int(template_num_vars):
        return x.float()
    pad = torch.zeros(x.shape[0], int(template_num_vars) - x.shape[1], dtype=x.dtype, device=x.device)
    return torch.cat([x.float(), pad.float()], dim=1)


def with_skeleton_metrics(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    gt = str(out.get("ground_truth", "") or "")
    pred = str(out.get("raw_expression", "") or out.get("expression", "") or "")
    gt_skeleton = expression_skeleton(gt)
    pred_skeleton = expression_skeleton(pred)
    out["gt_skeleton"] = gt_skeleton
    out["pred_skeleton"] = pred_skeleton
    out["skeleton_match"] = bool(gt_skeleton and pred_skeleton and gt_skeleton == pred_skeleton)
    return out


def skeleton_match(ground_truth: str, generated: str) -> bool:
    gt = expression_skeleton(ground_truth)
    pred = expression_skeleton(generated)
    return bool(gt and pred and gt == pred)


def expression_skeleton(expr_text: str) -> str:
    text = str(expr_text or "").strip()
    if not text:
        return ""
    try:
        expr = sp.sympify(text, locals={"Abs": sp.Abs})
    except Exception:
        return ""
    try:
        return _skeleton_node(expr)
    except Exception:
        return ""


def _skeleton_node(expr: sp.Expr) -> str:
    expr = sp.sympify(expr)
    if expr.is_Number:
        return "C"
    if expr.is_Symbol:
        return str(expr)
    if expr.func == sp.Abs:
        return _skeleton_node(expr.args[0])
    if expr.is_Add:
        parts = [_skeleton_node(arg) for arg in expr.args if not _is_numeric_only(arg)]
        parts = [part for part in parts if part and part != "C"]
        if not parts:
            return "C"
        if len(parts) == 1:
            return parts[0]
        return "add(" + ",".join(sorted(parts)) + ")"
    if expr.is_Mul:
        parts = [_skeleton_node(arg) for arg in expr.args if not _is_numeric_only(arg)]
        parts = [part for part in parts if part and part != "C"]
        if not parts:
            return "C"
        if len(parts) == 1:
            return parts[0]
        return "mul(" + ",".join(sorted(parts)) + ")"
    if expr.is_Pow:
        base, exponent = expr.args
        exp_key = _exponent_skeleton(exponent)
        return f"pow({_skeleton_node(base)},{exp_key})"
    name = _func_name(expr)
    parts = ",".join(_skeleton_node(arg) for arg in expr.args)
    return f"{name}({parts})"


def _is_numeric_only(expr: sp.Expr) -> bool:
    return not getattr(expr, "free_symbols", set())


def _func_name(expr: sp.Expr) -> str:
    name = getattr(expr.func, "__name__", str(expr.func))
    return {
        "sqrt": "sqrt",
        "log": "log",
        "sin": "sin",
        "cos": "cos",
        "exp": "exp",
    }.get(name, name)


def _exponent_skeleton(exponent: sp.Expr) -> str:
    if exponent.is_Integer and -8 <= int(exponent) <= 8:
        return str(int(exponent))
    if exponent.is_Rational and abs(int(exponent.p)) <= 8 and 1 < int(exponent.q) <= 8:
        return f"{int(exponent.p)}/{int(exponent.q)}"
    return "C"


def _mean(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return float(sum(float(row.get(key, 0.0) or 0.0) for row in records) / len(records))


def _mean_bool(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return float(sum(float(bool(row.get(key, False))) for row in records) / len(records))


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
