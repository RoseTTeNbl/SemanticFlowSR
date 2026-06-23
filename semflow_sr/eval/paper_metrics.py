"""Paper-facing metric aggregation for SR benchmark outputs."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any

import numpy as np


COMPARABLE_ROLES = {"sfsr_method", "external_comparison"}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    group: str
    role: str
    path: str | Path
    kind: str


@dataclass(frozen=True)
class PaperRecord:
    method: str
    group: str
    role: str
    task_id: str
    suite: str
    r2: float | None = None
    nmse: float | None = None
    nrmse: float | None = None
    expression: str = ""
    ground_truth: str = ""
    status: str = "ok"
    solved: bool | None = None
    skeleton_match: bool | None = None
    simplified_symbolic_equivalence: bool | None = None
    operator_dependency_match: bool | None = None
    complexity: float | None = None
    formula_bleu: float | None = None
    token_similarity: float | None = None
    edit_distance: float | None = None
    valid_fraction: float | None = None
    unique_fraction: float | None = None
    runtime_sec: float | None = None
    native_r2: float | None = None
    native_loss: float | None = None

    @property
    def comparable(self) -> bool:
        return self.role in COMPARABLE_ROLES


def load_method_records(spec: MethodSpec) -> list[PaperRecord]:
    path = Path(spec.path)
    if not path.exists():
        return []
    if spec.kind == "samples_jsonl":
        return [_record_from_mapping(spec, row) for row in _read_jsonl(path)]
    if spec.kind == "baseline_json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            rows = data
        else:
            rows = []
            for task_id, item in data.items():
                row = dict(item or {})
                row.setdefault("task_id", task_id)
                rows.append(row)
        return [_record_from_mapping(spec, row) for row in rows]
    if spec.kind == "reference_json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            rows = []
            for task_id, item in data.items():
                row = dict(item or {})
                row.setdefault("task_id", task_id)
                rows.append(row)
        else:
            rows = list(data)
        return [_record_from_mapping(spec, row) for row in rows]
    raise ValueError(f"unknown paper metric input kind: {spec.kind}")


def summarize_method(
    records: list[PaperRecord],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 0,
    spec: MethodSpec | None = None,
) -> dict[str, Any]:
    method = records[0].method if records else (spec.name if spec else "")
    group = records[0].group if records else (spec.group if spec else "")
    role = records[0].role if records else (spec.role if spec else "")
    comparable = role in COMPARABLE_ROLES
    r2_values = _values(records, "r2") if comparable else []
    nmse_values = _values(records, "nmse") if comparable else []
    nrmse_values = _values(records, "nrmse") if comparable else []
    complexity_values = _values(records, "complexity")
    weighted_complexity_values = [
        float(weighted_complexity(record.expression))
        for record in records
        if record.expression
    ]
    bleu_values = _metric_values(records, "formula_bleu", _computed_bleu)
    token_similarity_values = _metric_values(records, "token_similarity", _computed_token_similarity)
    edit_distance_values = _metric_values(records, "edit_distance", _computed_edit_distance)
    solved_values = _solved_values(records) if comparable else []
    skeleton_values = [float(record.skeleton_match) for record in records if record.skeleton_match is not None]
    symbolic_values = [
        float(record.simplified_symbolic_equivalence)
        for record in records
        if record.simplified_symbolic_equivalence is not None
    ]
    operator_dependency_values = [
        float(record.operator_dependency_match)
        for record in records
        if record.operator_dependency_match is not None
    ]
    runtime_values = _values(records, "runtime_sec")
    native_r2_values = _values(records, "native_r2")
    native_loss_values = _values(records, "native_loss")
    ci_low, ci_high = _bootstrap_ci(r2_values, samples=bootstrap_samples, seed=seed)
    return {
        "group": group,
        "method": method,
        "role": role,
        "comparable": comparable,
        "coverage": len(records),
        "failure_rate": _round(_failure_rate(records)),
        "r2_mean": _mean_or_blank(r2_values),
        "r2_median": _median_or_blank(r2_values),
        "r2_ci_low": ci_low,
        "r2_ci_high": ci_high,
        "nmse_mean": _mean_or_blank(nmse_values),
        "nrmse_mean": _mean_or_blank(nrmse_values),
        "accuracy_rate": _mean_or_blank([float((record.r2 or 0.0) > 0.999) for record in records if comparable and record.r2 is not None]),
        "solution_rate": _mean_or_blank(solved_values),
        "skeleton_accuracy": _mean_or_blank(skeleton_values),
        "simplified_symbolic_equivalence_rate": _mean_or_blank(symbolic_values),
        "operator_dependency_accuracy": _mean_or_blank(operator_dependency_values),
        "complexity_mean": _mean_or_blank(complexity_values),
        "weighted_complexity_mean": _mean_or_blank(weighted_complexity_values),
        "formula_bleu_mean": _mean_or_blank(bleu_values),
        "token_similarity_mean": _mean_or_blank(token_similarity_values),
        "edit_distance_mean": _mean_or_blank(edit_distance_values),
        "valid_rate": _round(_valid_rate(records)),
        "unique_rate": _round(_unique_rate(records)),
        "runtime_sec_mean": _mean_or_blank(runtime_values),
        "runtime_sec_median": _median_or_blank(runtime_values),
        "native_r2_mean": _mean_or_blank(native_r2_values),
        "native_loss_mean": _mean_or_blank(native_loss_values),
    }


def summarize_by_suite(records_by_method: dict[str, list[PaperRecord]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, records in records_by_method.items():
        suites = sorted({record.suite for record in records})
        for suite in suites:
            subset = [record for record in records if record.suite == suite]
            row = summarize_method(subset)
            row["method"] = method
            row["suite"] = suite
            rows.append(row)
    return rows


def paired_comparison(
    records_a: list[PaperRecord],
    records_b: list[PaperRecord],
    *,
    metric: str = "r2",
) -> dict[str, Any]:
    by_a = {record.task_id: record for record in records_a if getattr(record, metric) is not None}
    by_b = {record.task_id: record for record in records_b if getattr(record, metric) is not None}
    common = sorted(set(by_a) & set(by_b))
    deltas = [float(getattr(by_a[task_id], metric)) - float(getattr(by_b[task_id], metric)) for task_id in common]
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    return {
        "method_a": records_a[0].method if records_a else "",
        "method_b": records_b[0].method if records_b else "",
        "metric": metric,
        "n_matched": len(common),
        "mean_delta": _mean_or_blank(deltas),
        "median_delta": _median_or_blank(deltas),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "sign_test_p": _round(_two_sided_sign_test(wins, losses)),
    }


def write_archive(
    specs: list[MethodSpec],
    out: str | Path,
    *,
    bootstrap_samples: int = 1000,
    seed: int = 0,
    suites: list[str] | None = None,
) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records_by_method = {
        spec.name: _filter_records(load_method_records(spec), suites)
        for spec in specs
    }
    summary_rows = [
        summarize_method(records_by_method[spec.name], bootstrap_samples=bootstrap_samples, seed=seed, spec=spec)
        for spec in specs
    ]
    suite_rows = summarize_by_suite(records_by_method)
    comparable_methods = [
        row["method"] for row in summary_rows
        if row.get("comparable") and int(row.get("coverage", 0)) > 0
    ]
    reference_methods = [
        row["method"] for row in summary_rows
        if not row.get("comparable") and int(row.get("coverage", 0)) > 0
    ]
    paired_rows = _paired_rows(records_by_method, comparable_methods)
    reference_rows = [row for row in summary_rows if row["method"] in reference_methods]
    _write_csv(out / "method_summary.csv", summary_rows)
    _write_csv(out / "suite_summary.csv", suite_rows)
    _write_csv(out / "paired_significance.csv", paired_rows)
    _write_csv(out / "reference_summary.csv", reference_rows)
    (out / "method_summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True))
    (out / "suite_summary.json").write_text(json.dumps(suite_rows, indent=2, sort_keys=True))
    (out / "paired_significance.json").write_text(json.dumps(paired_rows, indent=2, sort_keys=True))
    manifest = {
        "input_specs": [
            {
                "name": spec.name,
                "group": spec.group,
                "role": spec.role,
                "kind": spec.kind,
                "path": str(spec.path),
                "exists": Path(spec.path).exists(),
            }
            for spec in specs
        ],
        "suite_filter": list(suites or []),
        "summary_rows": summary_rows,
        "comparable_methods": comparable_methods,
        "reference_methods": reference_methods,
        "generated_files": [
            "method_summary.csv",
            "method_summary.json",
            "suite_summary.csv",
            "suite_summary.json",
            "paired_significance.csv",
            "paired_significance.json",
            "reference_summary.csv",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _filter_records(records: list[PaperRecord], suites: list[str] | None) -> list[PaperRecord]:
    if not suites:
        return records
    selected = set(suites)
    return [record for record in records if record.suite in selected]


def plot_metric_summary(summary_rows: list[dict[str, Any]], out: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [row for row in summary_rows if row.get("comparable") and _is_number(row.get("r2_mean"))]
    if not rows:
        return
    labels = [str(row["method"]) for row in rows]
    r2 = [float(row["r2_mean"]) for row in rows]
    accuracy = [_num_or_zero(row.get("accuracy_rate")) for row in rows]
    solution = [_num_or_zero(row.get("solution_rate")) for row in rows]
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(labels)), 4.2))
    ax.bar(x - width, r2, width, label="Mean R2")
    ax.bar(x, accuracy, width, label="Accuracy R2>0.999")
    ax.bar(x + width, solution, width, label="Solution rate")
    for i, row in enumerate(rows):
        low = row.get("r2_ci_low")
        high = row.get("r2_ci_high")
        if _is_number(low) and _is_number(high):
            center = float(row["r2_mean"])
            ax.errorbar(
                i - width,
                center,
                yerr=[[max(0.0, center - float(low))], [max(0.0, float(high) - center)]],
                color="black",
                capsize=3,
                linewidth=1,
            )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "metric_summary.png", dpi=160)
    fig.savefig(out / "metric_summary.pdf")
    plt.close(fig)


def plot_complexity_pareto(summary_rows: list[dict[str, Any]], out: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    rows = [
        row for row in summary_rows
        if row.get("comparable") and _is_number(row.get("r2_mean")) and _is_number(row.get("weighted_complexity_mean"))
    ]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for row in rows:
        ax.scatter(float(row["weighted_complexity_mean"]), float(row["r2_mean"]), s=65)
        ax.annotate(str(row["method"]), (float(row["weighted_complexity_mean"]), float(row["r2_mean"])), fontsize=8)
    ax.set_xlabel("Weighted complexity")
    ax.set_ylabel("Mean R2")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "pareto_r2_complexity.png", dpi=160)
    fig.savefig(out / "pareto_r2_complexity.pdf")
    plt.close(fig)


def plot_structural_metrics(summary_rows: list[dict[str, Any]], out: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    rows = [
        row for row in summary_rows
        if _is_number(row.get("formula_bleu_mean")) or _is_number(row.get("token_similarity_mean"))
    ]
    if not rows:
        return
    labels = [str(row["method"]) for row in rows]
    skeleton = [_num_or_zero(row.get("skeleton_accuracy")) for row in rows]
    symbolic = [_num_or_zero(row.get("simplified_symbolic_equivalence_rate")) for row in rows]
    op_dep = [_num_or_zero(row.get("operator_dependency_accuracy")) for row in rows]
    bleu = [_num_or_zero(row.get("formula_bleu_mean")) for row in rows]
    token = [_num_or_zero(row.get("token_similarity_mean")) for row in rows]
    edit = [_num_or_zero(row.get("edit_distance_mean")) for row in rows]
    x = np.arange(len(labels))
    width = 0.15
    fig, ax_score = plt.subplots(figsize=(max(8.0, 1.1 * len(labels)), 4.6))
    offsets = [-2 * width, -width, 0.0, width, 2 * width]
    ax_score.bar(x + offsets[0], skeleton, width, label="Exact skeleton")
    ax_score.bar(x + offsets[1], symbolic, width, label="Symbolic eq")
    ax_score.bar(x + offsets[2], op_dep, width, label="Op/dependency")
    ax_score.bar(x + offsets[3], bleu, width, label="BLEU")
    ax_score.bar(x + offsets[4], token, width, label="Token similarity")
    ax_score.set_ylim(0.0, 1.05)
    ax_score.set_ylabel("Structural score")
    ax_score.set_xticks(x)
    ax_score.set_xticklabels(labels, rotation=25, ha="right")
    ax_score.grid(axis="y", alpha=0.25)
    ax_edit = ax_score.twinx()
    ax_edit.plot(x, edit, color="black", marker="o", linewidth=1.2, label="Edit distance")
    ax_edit.set_ylabel("Edit distance")
    score_handles, score_labels = ax_score.get_legend_handles_labels()
    edit_handles, edit_labels = ax_edit.get_legend_handles_labels()
    ax_score.legend(score_handles + edit_handles, score_labels + edit_labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "structural_metrics.png", dpi=160)
    fig.savefig(out / "structural_metrics.pdf")
    plt.close(fig)


def weighted_complexity(expression: str) -> int:
    total = 0
    for token in tokenize_expression(expression):
        low = token.lower()
        if low in {"/", "div"}:
            total += 2
        elif low in {"sin", "cos"}:
            total += 3
        elif low in {"exp", "log"}:
            total += 4
        elif low in {"(", ")", ","}:
            continue
        else:
            total += 1
    return total


def tokenize_expression(expression: str) -> list[str]:
    return re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?(?:e[+-]?\d+)?|[+\-*/^(),]", str(expression))


def _record_from_mapping(spec: MethodSpec, row: dict[str, Any]) -> PaperRecord:
    meta = row.get("task_metadata") or row.get("metadata") or {}
    task_id = str(
        row.get("task_id")
        or row.get("name")
        or meta.get("task_id")
        or meta.get("name")
        or ""
    )
    suite = str(row.get("suite") or meta.get("suite") or _infer_suite(task_id))
    expression = str(row.get("expression") or row.get("raw_expression") or row.get("prediction") or "")
    ground_truth = str(row.get("ground_truth") or row.get("gt_expression") or row.get("target_expression") or "")
    return PaperRecord(
        method=spec.name,
        group=spec.group,
        role=spec.role,
        task_id=task_id,
        suite=suite,
        r2=_float(row.get("r2", row.get("r2_zero"))),
        nmse=_float(row.get("nmse")),
        nrmse=_float(row.get("nrmse")),
        expression=expression,
        ground_truth=ground_truth,
        status=str(row.get("status") or "ok"),
        solved=_bool_or_none(row.get("solved")),
        skeleton_match=_bool_or_none(row.get("skeleton_match")),
        simplified_symbolic_equivalence=_bool_or_none(row.get("simplified_symbolic_equivalence")),
        operator_dependency_match=_bool_or_none(row.get("operator_dependency_match")),
        complexity=_float(row.get("complexity")),
        formula_bleu=_float(row.get("formula_bleu", row.get("bleu"))),
        token_similarity=_float(row.get("formula_token_accuracy", row.get("token_similarity", row.get("token_accuracy")))),
        edit_distance=_float(row.get("formula_edit_distance", row.get("edit_distance", row.get("levenshtein_edit_distance")))),
        valid_fraction=_float(row.get("valid_expression_fraction", row.get("valid_rate"))),
        unique_fraction=_float(row.get("unique_expression_fraction", row.get("unique_rate"))),
        runtime_sec=_float(row.get("runtime_sec", row.get("runtime"))),
        native_r2=_float(row.get("native_r2", row.get("native_r2_mean"))),
        native_loss=_float(row.get("native_loss", row.get("loss"))),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _computed_bleu(record: PaperRecord) -> float | None:
    return bleu_score(tokenize_expression(record.ground_truth), tokenize_expression(record.expression))


def _computed_token_similarity(record: PaperRecord) -> float | None:
    ref = tokenize_expression(record.ground_truth)
    pred = tokenize_expression(record.expression)
    if not ref and not pred:
        return None
    return _round(_lcs_len(ref, pred) / max(len(ref), len(pred), 1))


def _computed_edit_distance(record: PaperRecord) -> float | None:
    ref = tokenize_expression(record.ground_truth)
    pred = tokenize_expression(record.expression)
    if not ref and not pred:
        return None
    return float(_edit_distance(ref, pred))


def bleu_score(reference: list[str], prediction: list[str], max_order: int = 4) -> float | None:
    if not reference or not prediction:
        return None
    precisions = []
    for order in range(1, max_order + 1):
        ref_counts = _ngram_counts(reference, order)
        pred_counts = _ngram_counts(prediction, order)
        if not pred_counts:
            precisions.append(1e-9)
            continue
        overlap = sum(min(count, ref_counts.get(ngram, 0)) for ngram, count in pred_counts.items())
        precisions.append((overlap + 1.0) / (sum(pred_counts.values()) + 1.0))
    brevity = 1.0 if len(prediction) > len(reference) else math.exp(1.0 - len(reference) / max(len(prediction), 1))
    return _round(brevity * math.exp(sum(math.log(p) for p in precisions) / max_order))


def _ngram_counts(tokens: list[str], order: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for idx in range(0, len(tokens) - order + 1):
        ngram = tuple(tokens[idx:idx + order])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def _lcs_len(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            old = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = old
    return dp[-1]


def _edit_distance(a: list[str], b: list[str]) -> int:
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, y in enumerate(b, start=1):
            old = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (0 if x == y else 1),
            )
            prev = old
    return dp[-1]


def _metric_values(records: list[PaperRecord], attr: str, compute_fn) -> list[float]:
    values = []
    for record in records:
        value = getattr(record, attr)
        if value is None:
            value = compute_fn(record)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _solved_values(records: list[PaperRecord]) -> list[float]:
    values = []
    for record in records:
        if record.solved is not None:
            values.append(float(record.solved))
        elif record.r2 is not None:
            values.append(float(record.r2 >= 1.0 - 1e-12))
    return values


def _values(records: list[PaperRecord], attr: str) -> list[float]:
    values = []
    for record in records:
        value = getattr(record, attr)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            values.append(value)
    return values


def _bootstrap_ci(values: list[float], *, samples: int, seed: int) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    if len(values) == 1:
        value = _round(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = np.empty(int(samples), dtype=float)
    for idx in range(int(samples)):
        means[idx] = rng.choice(arr, size=len(arr), replace=True).mean()
    return _round(np.percentile(means, 2.5)), _round(np.percentile(means, 97.5))


def _failure_rate(records: list[PaperRecord]) -> float:
    if not records:
        return 0.0
    return sum(record.status not in {"", "ok", "success"} for record in records) / len(records)


def _valid_rate(records: list[PaperRecord]) -> float:
    if not records:
        return 0.0
    explicit = [record.valid_fraction for record in records if record.valid_fraction is not None]
    if explicit:
        return float(np.mean(explicit))
    return sum(bool(record.expression) and record.status not in {"failed", "error"} for record in records) / len(records)


def _unique_rate(records: list[PaperRecord]) -> float:
    if not records:
        return 0.0
    explicit = [record.unique_fraction for record in records if record.unique_fraction is not None]
    if explicit:
        return float(np.mean(explicit))
    expressions = [record.expression for record in records if record.expression]
    if not expressions:
        return 0.0
    return len(set(expressions)) / len(expressions)


def _paired_rows(records_by_method: dict[str, list[PaperRecord]], comparable_methods: list[str]) -> list[dict[str, Any]]:
    rows = []
    for i, method_a in enumerate(comparable_methods):
        for method_b in comparable_methods[i + 1:]:
            rows.append(paired_comparison(records_by_method[method_a], records_by_method[method_b], metric="r2"))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _two_sided_sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    prob = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, prob)


def _mean_or_blank(values: list[float]) -> float | str:
    return _round(float(np.mean(values))) if values else ""


def _median_or_blank(values: list[float]) -> float | str:
    return _round(float(median(values))) if values else ""


def _round(value: float | np.floating) -> float:
    return round(float(value), 12)


def _num_or_zero(value: Any) -> float:
    return float(value) if _is_number(value) else 0.0


def _is_number(value: Any) -> bool:
    try:
        return value != "" and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _bool_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _infer_suite(task_id: str) -> str:
    if "/" in task_id:
        return task_id.split("/", 1)[0]
    if "-" in task_id:
        return task_id.split("-", 1)[0].lower()
    return "unknown"
