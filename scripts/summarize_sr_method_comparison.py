#!/usr/bin/env python
"""Summarize SFSR and external-baseline SR results on a shared eval set."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from semflow_sr.edge_flow.benchmark import with_skeleton_metrics
from semflow_sr.eval.paper_metrics import (
    MethodSpec,
    PaperRecord,
    load_method_records,
    paired_comparison,
    summarize_method,
    weighted_complexity,
)


STRICT_COLUMNS = [
    "method",
    "R2 mean",
    "R2 std",
    "R2 min",
    "R2 kept count",
    "R2 excluded count",
    "R2 excluded rate",
    "NMSE mean",
    "NMSE std",
    "solution rate",
    "exact skeleton",
    "symbolic eq",
    "op/dep",
    "BLEU",
    "token acc",
    "edit dist",
    "complexity",
    "weighted complexity",
    "valid rate",
    "unique rate",
    "runtime sec mean",
]

EXTREME_NEGATIVE_R2_THRESHOLD = -1.0


def _parse_method(text: str) -> MethodSpec:
    parts = text.split("=", 1)
    if len(parts) != 2:
        raise ValueError(f"method spec must be NAME=PATH: {text}")
    name, path = parts
    name = name.strip()
    path = path.strip()
    if path.endswith(".jsonl"):
        kind = "samples_jsonl"
    elif path.endswith(".json"):
        kind = "baseline_json"
    else:
        raise ValueError(f"cannot infer result kind from {path}")
    role = "sfsr_method" if name.lower().startswith(("sfsr", "typed", "csef")) else "external_comparison"
    return MethodSpec(name=name, group="comparison", role=role, path=path, kind=kind)


def _metric_values(records: list[PaperRecord], metric: str) -> list[float]:
    vals: list[float] = []
    for rec in records:
        if metric == "r2":
            value = rec.r2
        elif metric == "nmse":
            value = rec.nmse
        elif metric == "solution":
            value = float(rec.solved) if rec.solved is not None else (float(rec.r2 >= 1.0 - 1e-12) if rec.r2 is not None else None)
        elif metric == "skeleton":
            value = float(rec.skeleton_match) if rec.skeleton_match is not None else None
        elif metric == "symbolic":
            value = float(rec.simplified_symbolic_equivalence) if rec.simplified_symbolic_equivalence is not None else None
        elif metric == "opdep":
            value = float(rec.operator_dependency_match) if rec.operator_dependency_match is not None else None
        elif metric == "bleu":
            value = rec.formula_bleu
        elif metric == "token":
            value = rec.token_similarity
        elif metric == "edit":
            value = rec.edit_distance
        elif metric == "complexity":
            value = rec.complexity
        elif metric == "weighted_complexity":
            value = float(weighted_complexity(rec.expression)) if rec.expression else None
        elif metric == "valid":
            if rec.valid_fraction is not None:
                value = rec.valid_fraction
            else:
                value = float(bool(rec.expression) and rec.status not in {"failed", "error"})
        elif metric == "runtime":
            value = rec.runtime_sec
        elif metric == "unique":
            value = rec.unique_fraction
        else:
            raise ValueError(metric)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            vals.append(value)
    return vals


def _mean(values: list[float]) -> float | str:
    return round(float(np.mean(values)), 12) if values else ""


def _std(values: list[float]) -> float | str:
    return round(float(np.std(values, ddof=1)), 12) if len(values) > 1 else (0.0 if len(values) == 1 else "")


def _median(values: list[float]) -> float | str:
    return round(float(np.median(values)), 12) if values else ""


def _ci_text(summary: dict[str, Any]) -> str:
    low = summary.get("r2_ci_low", "")
    high = summary.get("r2_ci_high", "")
    if low == "" or high == "":
        return ""
    return f"[{float(low):.6g}, {float(high):.6g}]"


def _load_and_complete_records(spec: MethodSpec) -> list[PaperRecord]:
    records = load_method_records(spec)
    completed = []
    for rec in records:
        row = {
            "task_id": rec.task_id,
            "suite": rec.suite,
            "expression": rec.expression,
            "ground_truth": rec.ground_truth,
            "formula_bleu": rec.formula_bleu,
            "formula_token_accuracy": rec.token_similarity,
            "formula_edit_distance": rec.edit_distance,
        }
        if rec.ground_truth and rec.expression:
            row = with_skeleton_metrics(row)
        completed.append(replace(
            rec,
            skeleton_match=rec.skeleton_match if rec.skeleton_match is not None else _bool_or_none(row.get("skeleton_match")),
            simplified_symbolic_equivalence=(
                rec.simplified_symbolic_equivalence
                if rec.simplified_symbolic_equivalence is not None
                else _bool_or_none(row.get("simplified_symbolic_equivalence"))
            ),
            operator_dependency_match=(
                rec.operator_dependency_match
                if rec.operator_dependency_match is not None
                else _bool_or_none(row.get("operator_dependency_match"))
            ),
            formula_bleu=rec.formula_bleu if rec.formula_bleu is not None else _float_or_none(row.get("formula_bleu")),
            token_similarity=rec.token_similarity if rec.token_similarity is not None else _float_or_none(row.get("formula_token_accuracy")),
            edit_distance=rec.edit_distance if rec.edit_distance is not None else _float_or_none(row.get("formula_edit_distance")),
            complexity=rec.complexity if rec.complexity is not None else (float(weighted_complexity(rec.expression)) if rec.expression else None),
        ))
    return completed


def _filter_records(
    records: list[PaperRecord],
    *,
    suites: list[str] | None = None,
    task_ids: set[str] | None = None,
) -> list[PaperRecord]:
    selected_suites = {str(v) for v in suites or [] if str(v)}
    out = []
    for rec in records:
        if selected_suites and str(rec.suite) not in selected_suites:
            continue
        if task_ids is not None and str(rec.task_id) not in task_ids:
            continue
        out.append(rec)
    return out


def _read_task_id_file(path: str) -> set[str]:
    task_ids: set[str] = set()
    for raw in Path(path).read_text().splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        task_ids.add(text.split(",", 1)[0].strip())
    return task_ids


def _strict_row(method: str, records: list[PaperRecord], summary: dict[str, Any]) -> dict[str, Any]:
    extreme_threshold = float(EXTREME_NEGATIVE_R2_THRESHOLD)
    r2 = _metric_values(records, "r2")
    kept_records = [
        rec for rec in records
        if rec.r2 is None or not math.isfinite(float(rec.r2)) or float(rec.r2) >= extreme_threshold
    ]
    kept_r2 = _metric_values(kept_records, "r2")
    kept_nmse = _metric_values(kept_records, "nmse")
    excluded_count = sum(1 for value in r2 if value < extreme_threshold)
    unique_values = _metric_values(records, "unique")
    if unique_values:
        unique_rate = _mean(unique_values)
    else:
        exprs = [rec.expression for rec in records if rec.expression]
        unique_rate = round(float(len(set(exprs)) / max(len(exprs), 1)), 12) if exprs else 0.0
    return {
        "method": method,
        "R2 mean": _mean(kept_r2),
        "R2 std": _std(kept_r2),
        "R2 min": round(float(min(kept_r2)), 12) if kept_r2 else "",
        "R2 kept count": int(len(kept_r2)),
        "R2 excluded count": int(excluded_count),
        "R2 excluded rate": round(float(excluded_count / max(len(r2), 1)), 12) if r2 else "",
        "NMSE mean": _mean(kept_nmse),
        "NMSE std": _std(kept_nmse),
        "solution rate": _mean(_metric_values(records, "solution")),
        "exact skeleton": _mean(_metric_values(records, "skeleton")),
        "symbolic eq": _mean(_metric_values(records, "symbolic")),
        "op/dep": _mean(_metric_values(records, "opdep")),
        "BLEU": _mean(_metric_values(records, "bleu")),
        "token acc": _mean(_metric_values(records, "token")),
        "edit dist": _mean(_metric_values(records, "edit")),
        "complexity": _mean(_metric_values(records, "complexity")),
        "weighted complexity": _mean(_metric_values(records, "weighted_complexity")),
        "valid rate": _mean(_metric_values(records, "valid")),
        "unique rate": unique_rate,
        "runtime sec mean": _mean(_metric_values(records, "runtime")),
        "record count": len(records),
        "failed count": sum(1 for rec in records if rec.status in {"failed", "error"}),
    }


def _suite_rows(records_by_method: dict[str, list[PaperRecord]]) -> list[dict[str, Any]]:
    rows = []
    for method, records in records_by_method.items():
        for suite in sorted({rec.suite for rec in records}):
            subset = [rec for rec in records if rec.suite == suite]
            summary = summarize_method(subset, bootstrap_samples=1000, seed=0, spec=MethodSpec(method, "comparison", subset[0].role if subset else "external_comparison", "", "internal"))
            row = _strict_row(method, subset, summary)
            row["suite"] = suite
            rows.append(row)
    return rows


def _per_task_rows(records_by_method: dict[str, list[PaperRecord]]) -> list[dict[str, Any]]:
    by_method = {method: {rec.task_id: rec for rec in records} for method, records in records_by_method.items()}
    task_ids = sorted(set().union(*(set(rows) for rows in by_method.values())))
    out = []
    for task_id in task_ids:
        row = {"task_id": task_id}
        suites = [rec.suite for rows in by_method.values() for tid, rec in rows.items() if tid == task_id]
        if suites:
            row["suite"] = suites[0]
        for method, records in by_method.items():
            rec = records.get(task_id)
            prefix = method
            if rec is None:
                row[f"{prefix} r2"] = ""
                row[f"{prefix} expression"] = ""
                continue
            row[f"{prefix} r2"] = rec.r2 if rec.r2 is not None else ""
            row[f"{prefix} nmse"] = rec.nmse if rec.nmse is not None else ""
            row[f"{prefix} skeleton"] = rec.skeleton_match if rec.skeleton_match is not None else ""
            row[f"{prefix} expression"] = rec.expression
            row.setdefault("ground_truth", rec.ground_truth)
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(preferred or [])
    for key in (key for row in rows for key in row):
        if key not in fields:
            fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(row.get(col, "")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def _md_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _bool_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    return bool(value)


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", action="append", required=True, help="NAME=path/to/result.json[l]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suite", nargs="*", default=None, help="optional suite filter, e.g. --suite nguyen")
    ap.add_argument("--task-id-file", default="", help="optional newline/csv first-column task-id filter")
    args = ap.parse_args()

    specs = [_parse_method(text) for text in args.method]
    task_ids = _read_task_id_file(args.task_id_file) if args.task_id_file else None
    records_by_method = {
        spec.name: _filter_records(
            _load_and_complete_records(spec),
            suites=list(args.suite or []),
            task_ids=task_ids,
        )
        for spec in specs
    }
    summary_by_method = {
        spec.name: summarize_method(records_by_method[spec.name], bootstrap_samples=1000, seed=int(args.seed), spec=spec)
        for spec in specs
    }
    method_rows = [_strict_row(spec.name, records_by_method[spec.name], summary_by_method[spec.name]) for spec in specs]
    suite_rows = _suite_rows(records_by_method)
    per_task_rows = _per_task_rows(records_by_method)
    paired_rows = []
    for i, spec_a in enumerate(specs):
        for spec_b in specs[i + 1:]:
            paired_rows.append(paired_comparison(records_by_method[spec_a.name], records_by_method[spec_b.name], metric="r2"))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "method_summary.csv", method_rows, STRICT_COLUMNS)
    _write_csv(out / "suite_summary.csv", suite_rows, ["suite", *STRICT_COLUMNS])
    _write_csv(out / "per_task_comparison.csv", per_task_rows)
    _write_csv(out / "paired_r2.csv", paired_rows)
    (out / "method_summary.json").write_text(json.dumps(method_rows, indent=2, sort_keys=True))
    (out / "suite_summary.json").write_text(json.dumps(suite_rows, indent=2, sort_keys=True))
    (out / "paired_r2.json").write_text(json.dumps(paired_rows, indent=2, sort_keys=True))
    _write_markdown_table(out / "method_summary.md", method_rows, STRICT_COLUMNS)
    _write_markdown_table(out / "suite_summary.md", suite_rows, ["suite", *STRICT_COLUMNS])
    manifest = {
        "methods": [
            {
                "name": spec.name,
                "role": spec.role,
                "kind": spec.kind,
                "path": str(spec.path),
                "exists": Path(spec.path).exists(),
                "records": len(records_by_method[spec.name]),
            }
            for spec in specs
        ],
        "suite_filter": list(args.suite or []),
        "task_id_file": str(args.task_id_file or ""),
        "task_id_count": len(task_ids) if task_ids is not None else None,
        "generated_files": [
            "method_summary.csv",
            "method_summary.md",
            "method_summary.json",
            "suite_summary.csv",
            "suite_summary.md",
            "suite_summary.json",
            "per_task_comparison.csv",
            "paired_r2.csv",
            "paired_r2.json",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
