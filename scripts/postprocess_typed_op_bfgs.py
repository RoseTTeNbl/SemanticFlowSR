"""Post-process TypedOpNodeRegisterFlow samples with numeric-constant BFGS.

This is a diagnostic pass: it does not change the generated structure.  It
only treats finite numeric constants in the printed expression as coefficient
slots, optimizes them on the task train split, and reports test R2/NMSE.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.optimize import minimize

from semflow_sr.edge_flow.benchmark import (
    load_edge_flow_benchmark_tasks,
    summarize_benchmark_records,
)
from semflow_sr.data.symbolicgpt_subset import load_symbolicgpt_subset_tasks


BAD_EXPR_TOKENS = ("zoo", "oo", "nan", "inf")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _r2_nmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    yt = np.nan_to_num(np.asarray(y_true, dtype=np.float64), nan=0.0, posinf=1e150, neginf=-1e150)
    yp = np.nan_to_num(np.asarray(y_pred, dtype=np.float64), nan=0.0, posinf=1e150, neginf=-1e150)
    residual = np.clip(yt - yp, -1e150, 1e150)
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 0.0 if ss_tot < 1e-12 else float(1.0 - ss_res / ss_tot)
    nmse = float(np.mean(residual * residual) / (np.var(yt) + 1e-12))
    return (r2 if math.isfinite(r2) else -1e300, nmse if math.isfinite(nmse) else 1e300)


def _load_task_map(args) -> dict[str, Any]:
    tasks = load_edge_flow_benchmark_tasks(
        manifest=args.manifest,
        suites=_split_csv(args.suites),
        root=args.manifest_root,
        seed=int(args.seed),
        legacy_87=bool(args.legacy_87),
        feynman_root=args.feynman_root,
        limit=None if int(args.task_scan_limit) <= 0 else int(args.task_scan_limit),
    )
    if int(args.symbolicgpt_eval_limit) > 0:
        tasks.extend(
            load_symbolicgpt_subset_tasks(
                args.symbolicgpt_root,
                splits=_split_csv(args.symbolicgpt_eval_splits),
                limit=None,
                rng=random.Random(int(args.seed) + 2701),
            )[: int(args.symbolicgpt_eval_limit)]
        )
    return {str(task.name): task for task in tasks}


def _sympy_namespace(num_vars: int) -> dict[str, Any]:
    ns: dict[str, Any] = {
        "Abs": sp.Abs,
        "sin": sp.sin,
        "cos": sp.cos,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "E": sp.E,
        "pi": sp.pi,
    }
    for idx in range(int(num_vars)):
        ns[f"x{idx}"] = sp.Symbol(f"x{idx}")
    return ns


def _replace_coefficient_numbers(expr: sp.Expr) -> tuple[sp.Expr, list[sp.Symbol], np.ndarray]:
    params: list[sp.Symbol] = []
    initials: list[float] = []

    def rec(node: sp.Expr, *, in_exponent: bool = False) -> sp.Expr:
        if node.is_Number:
            if in_exponent or not node.is_finite:
                return node
            value = float(node.evalf())
            if not math.isfinite(value) or abs(value) < 1e-12 or abs(abs(value) - 1.0) < 1e-12:
                return node
            symbol = sp.Symbol(f"c{len(params)}")
            params.append(symbol)
            initials.append(value)
            return symbol
        if not node.args:
            return node
        if isinstance(node, sp.Pow) and len(node.args) == 2:
            return sp.Pow(rec(node.args[0], in_exponent=False), rec(node.args[1], in_exponent=True), evaluate=False)
        return node.func(*[rec(arg, in_exponent=False) for arg in node.args], evaluate=False)

    return rec(expr), params, np.asarray(initials, dtype=np.float64)


def _fit_one(row: dict[str, Any], task: Any, *, expr_field: str, maxiter: int) -> dict[str, Any]:
    expr_text = str(row.get(expr_field, "") or "")
    out = dict(row)
    out["bfgs_expression_source"] = expr_field
    out["bfgs_status"] = "skipped"
    out["bfgs_param_count"] = 0
    out["bfgs_r2"] = row.get("r2", -1e300)
    out["bfgs_nmse"] = row.get("nmse", 1e300)
    out["bfgs_improvement"] = 0.0
    out["postfit_r2"] = row.get("r2", -1e300)
    out["postfit_nmse"] = row.get("nmse", 1e300)
    out["postfit_source"] = "base"
    out["postfit_expression"] = row.get(expr_field, "")
    if not expr_text or any(token in expr_text for token in BAD_EXPR_TOKENS):
        out["bfgs_status"] = "bad_expression_token"
        return out

    num_vars = int(row.get("num_vars", len(task.variable_names)))
    variables = [sp.Symbol(f"x{i}") for i in range(num_vars)]
    try:
        expr = sp.sympify(expr_text, locals=_sympy_namespace(num_vars))
        param_expr, params, initial = _replace_coefficient_numbers(expr)
    except Exception as exc:
        out["bfgs_status"] = f"parse_failed:{type(exc).__name__}"
        return out
    if not params:
        out["bfgs_status"] = "no_coefficients"
        return out

    x_train = np.asarray(task.X_train, dtype=np.float64)[: int(args_global.max_train_points)]
    y_train = np.asarray(task.y_train, dtype=np.float64)[: int(args_global.max_train_points)]
    x_test = np.asarray(task.X_test, dtype=np.float64)[: int(args_global.max_eval_points)]
    y_test = np.asarray(task.y_test, dtype=np.float64)[: int(args_global.max_eval_points)]
    fn = sp.lambdify([*variables, *params], param_expr, modules="numpy")

    def predict(coeff: np.ndarray, x: np.ndarray) -> np.ndarray:
        vals = [x[:, idx] for idx in range(num_vars)]
        pred = fn(*vals, *[float(v) for v in coeff])
        pred = np.asarray(pred, dtype=np.float64)
        pred = np.broadcast_to(pred, (x.shape[0],)).copy()
        return np.nan_to_num(pred, nan=0.0, posinf=1e150, neginf=-1e150)

    def objective(coeff: np.ndarray) -> float:
        try:
            pred = predict(coeff, x_train)
        except Exception:
            return 1e300
        residual = np.clip(pred - y_train, -1e150, 1e150)
        value = float(np.mean(residual * residual))
        if not math.isfinite(value):
            return 1e300
        return value

    try:
        result = minimize(objective, initial, method="L-BFGS-B", options={"maxiter": int(maxiter), "ftol": 1e-12})
        coeff = np.asarray(result.x if result.x is not None else initial, dtype=np.float64)
        test_pred = predict(coeff, x_test)
        train_pred = predict(coeff, x_train)
        bfgs_r2, bfgs_nmse = _r2_nmse(y_test, test_pred)
        train_r2, train_nmse = _r2_nmse(y_train, train_pred)
        fitted_expr = param_expr.subs({param: float(value) for param, value in zip(params, coeff)})
    except Exception as exc:
        out["bfgs_status"] = f"fit_failed:{type(exc).__name__}"
        return out

    base_r2 = float(row.get("r2", -1e300) or -1e300)
    base_nmse = float(row.get("nmse", 1e300) or 1e300)
    use_bfgs = float(bfgs_r2) > base_r2
    out.update(
        {
            "bfgs_status": "ok" if bool(result.success) else f"optimizer:{result.message}",
            "bfgs_param_count": int(len(params)),
            "bfgs_r2": float(bfgs_r2),
            "bfgs_nmse": float(bfgs_nmse),
            "bfgs_train_r2": float(train_r2),
            "bfgs_train_nmse": float(train_nmse),
            "bfgs_improvement": float(bfgs_r2 - base_r2),
            "bfgs_initial_coefficients": [float(v) for v in initial.tolist()],
            "bfgs_coefficients": [float(v) for v in coeff.tolist()],
            "bfgs_expression": str(fitted_expr),
            "postfit_r2": float(bfgs_r2 if use_bfgs else base_r2),
            "postfit_nmse": float(bfgs_nmse if use_bfgs else base_nmse),
            "postfit_source": "bfgs" if use_bfgs else "base",
            "postfit_expression": str(fitted_expr) if use_bfgs else row.get(expr_field, ""),
        }
    )
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if str(row.get("bfgs_status", "")).startswith("ok")]
    improved = [row for row in rows if float(row.get("bfgs_improvement", 0.0) or 0.0) > 1e-9]
    base_summary = summarize_benchmark_records(rows)
    bfgs_records = [{**row, "r2": row.get("bfgs_r2", row.get("r2")), "nmse": row.get("bfgs_nmse", row.get("nmse"))} for row in rows]
    postfit_records = [{**row, "r2": row.get("postfit_r2", row.get("r2")), "nmse": row.get("postfit_nmse", row.get("nmse"))} for row in rows]
    bfgs_summary = summarize_benchmark_records(bfgs_records)
    postfit_summary = summarize_benchmark_records(postfit_records)
    return {
        "total": int(len(rows)),
        "ok": int(len(ok)),
        "improved": int(len(improved)),
        "base": base_summary,
        "bfgs": bfgs_summary,
        "postfit": postfit_summary,
        "mean_improvement": float(sum(float(row.get("bfgs_improvement", 0.0) or 0.0) for row in rows) / max(len(rows), 1)),
        "max_improvement": float(max((float(row.get("bfgs_improvement", 0.0) or 0.0) for row in rows), default=0.0)),
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Typed Op BFGS Postfit Summary",
        "",
        f"- total: {summary['total']}",
        f"- ok: {summary['ok']}",
        f"- improved: {summary['improved']}",
        f"- base R2 mean: {summary['base'].get('r2_mean', 0.0):.6f}",
        f"- BFGS R2 mean: {summary['bfgs'].get('r2_mean', 0.0):.6f}",
        f"- postfit R2 mean: {summary['postfit'].get('r2_mean', 0.0):.6f}",
        f"- mean improvement: {summary['mean_improvement']:.6f}",
        f"- max improvement: {summary['max_improvement']:.6f}",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expr-field", default="expression", choices=["expression", "raw_expression"])
    parser.add_argument("--manifest", default="data/benchmark_suites/benchmark_manifest.json")
    parser.add_argument("--manifest-root", default="data/benchmark_suites")
    parser.add_argument("--suites", default="nguyen,constant,livermore,jin")
    parser.add_argument("--legacy-87", action="store_true")
    parser.add_argument("--feynman-root", default="data")
    parser.add_argument("--task-scan-limit", type=int, default=34)
    parser.add_argument("--symbolicgpt-root", default="data/generated/symbolicgpt_large_2000_200_200")
    parser.add_argument("--symbolicgpt-eval-splits", default="test")
    parser.add_argument("--symbolicgpt-eval-limit", type=int, default=0)
    parser.add_argument("--max-train-points", type=int, default=100)
    parser.add_argument("--max-eval-points", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--maxiter", type=int, default=200)
    args = parser.parse_args()

    global args_global
    args_global = args

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.samples).read_text().splitlines() if line.strip()]
    task_map = _load_task_map(args)
    fitted = []
    for row in rows:
        task = task_map.get(str(row.get("task_id", "")))
        if task is None:
            item = dict(row)
            item.update(
                {
                    "bfgs_status": "missing_task",
                    "bfgs_r2": row.get("r2"),
                    "bfgs_nmse": row.get("nmse"),
                    "bfgs_improvement": 0.0,
                    "postfit_r2": row.get("r2"),
                    "postfit_nmse": row.get("nmse"),
                    "postfit_source": "base",
                    "postfit_expression": row.get(str(args.expr_field), ""),
                }
            )
        else:
            item = _fit_one(row, task, expr_field=str(args.expr_field), maxiter=int(args.maxiter))
        fitted.append(item)

    summary = _summarize(fitted)
    (out_dir / "bfgs_postfit_summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    _write_markdown(out_dir / "bfgs_postfit_summary.md", summary)
    with (out_dir / "bfgs_postfit_samples.jsonl").open("w") as f:
        for row in fitted:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    fields = sorted({key for row in fitted for key in row})
    with (out_dir / "bfgs_postfit_samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in fitted:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    args_global = argparse.Namespace(max_train_points=100, max_eval_points=100)
    main()
