#!/usr/bin/env python
"""Build paper-facing benchmark tables from reusable source runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Any

from semflow_sr.eval.paper_metrics import MethodSpec, PaperRecord, load_method_records

from scripts.summarize_sr_method_comparison import (
    _load_and_complete_records,
    _per_task_rows,
    _strict_row,
    _write_csv,
    _write_markdown_table,
)


BENCHMARKS = ["nguyen", "livermore", "constant", "jin", "symbolicgpt_large"]
FINAL_COLUMNS = [
    "benchmark",
    "method",
    "r2_mean",
    "r2_std",
    "solution_rate",
    "complexity_mean",
    "valid_pct",
    "unique_pct",
    "BLEU",
    "token_similarity",
    "edit_distance",
    "skeleton_accuracy",
    "n_tasks",
    "n_valid",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/clean_benchmark_20260701/paper_complete_20260702")
    ap.add_argument("--clean-root", default="results/clean_benchmark_20260701")
    ap.add_argument("--include-sfsr", action="store_true")
    ap.add_argument("--archive-clutter", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    clean_root = Path(args.clean_root)
    source_dir = root / "source_runs"
    final_dir = root / "final_tables"
    per_task_dir = root / "per_task"
    for path in [source_dir, final_dir, per_task_dir, root / "manifests", root / "logs", root / "trained_small_models"]:
        path.mkdir(parents=True, exist_ok=True)

    sources = _discover_sources(clean_root, include_sfsr=bool(args.include_sfsr))
    copied_sources = _copy_sources(sources, source_dir)
    method_records = _load_usable_records(copied_sources)
    overview_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "root": str(root),
        "benchmarks": BENCHMARKS,
        "source_files": {method: [str(p) for p in paths] for method, paths in copied_sources.items()},
        "generated_files": [],
        "skipped_methods": [],
        "metric_notes": {
            "r2": "mean/std after excluding task records with R2 < -1",
            "valid_pct": "100 * valid expression fraction",
            "unique_pct": "100 * unique expression fraction when available, else unique generated expression ratio",
            "token_metrics": "DiffSR final_diffusion_model.ipynb tokenization and sentence-level metrics",
        },
    }

    for benchmark in BENCHMARKS:
        records_by_method: dict[str, list[PaperRecord]] = {}
        for method, records in method_records.items():
            subset = [rec for rec in records if rec.suite == benchmark]
            if subset and _has_valid_records(subset):
                records_by_method[method] = subset
        bench_dir = final_dir / f"benchmark_{benchmark}"
        bench_dir.mkdir(parents=True, exist_ok=True)
        method_rows = []
        for method, records in sorted(records_by_method.items()):
            strict = _strict_row(method, records, summary={})
            row = _final_row(benchmark, method, strict, records)
            method_rows.append(row)
            overview_rows.append(row)
        per_task_rows = _per_task_rows(records_by_method)
        _write_csv(bench_dir / "method_summary.csv", method_rows, FINAL_COLUMNS)
        _write_csv(per_task_dir / f"{benchmark}_per_task_comparison.csv", per_task_rows)
        (bench_dir / "method_summary.json").write_text(json.dumps(method_rows, indent=2, sort_keys=True))
        _write_markdown_table(bench_dir / "method_summary.md", method_rows, FINAL_COLUMNS)
        manifest["generated_files"].extend([
            str(bench_dir / "method_summary.csv"),
            str(bench_dir / "method_summary.md"),
            str(bench_dir / "method_summary.json"),
            str(per_task_dir / f"{benchmark}_per_task_comparison.csv"),
        ])

    _write_csv(final_dir / "benchmark_overview.csv", overview_rows, FINAL_COLUMNS)
    (final_dir / "benchmark_overview.json").write_text(json.dumps(overview_rows, indent=2, sort_keys=True))
    _write_markdown_table(final_dir / "benchmark_overview.md", overview_rows, FINAL_COLUMNS)
    manifest["generated_files"].extend([
        str(final_dir / "benchmark_overview.csv"),
        str(final_dir / "benchmark_overview.md"),
        str(final_dir / "benchmark_overview.json"),
    ])
    if args.archive_clutter:
        manifest["archived_files"] = _archive_clutter(clean_root, root)
    (root / "README.md").write_text(_readme(root, overview_rows, manifest))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"root": str(root), "methods": sorted(method_records), "rows": len(overview_rows)}, indent=2))


def _discover_sources(clean_root: Path, *, include_sfsr: bool) -> dict[str, list[Path]]:
    by_method: dict[str, list[Path]] = {}
    for sub in ["formula_dev", "symbolicgpt_large"]:
        for path in sorted((clean_root / "external_baselines" / sub).glob("*.json")):
            method = _method_name(path)
            if method:
                by_method.setdefault(method, []).append(path)
    if include_sfsr:
        for path in sorted((clean_root / "typed_op_node_flow").glob("*/typed_op_node_flow_samples.jsonl")):
            method = _sfsr_method_name(path)
            if method:
                by_method.setdefault(method, []).append(path)
        for path in sorted((clean_root / "ablations").glob("*/**/typed_op_node_flow_samples.jsonl")):
            method = _sfsr_method_name(path)
            if method:
                by_method.setdefault(method, []).append(path)
    return by_method


def _copy_sources(sources: dict[str, list[Path]], source_dir: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for method, paths in sources.items():
        for path in paths:
            if not path.exists():
                continue
            if path.suffix == ".json" and not _baseline_source_is_usable(path):
                continue
            method_dir = source_dir / _slug(method)
            method_dir.mkdir(parents=True, exist_ok=True)
            target = method_dir / path.name
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)
            out.setdefault(method, []).append(target)
    return out


def _baseline_source_is_usable(path: Path, *, min_valid_fraction: float = 0.5) -> bool:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    rows = list(data.values()) if isinstance(data, dict) else list(data or [])
    if not rows:
        return False
    valid = sum(1 for row in rows if row.get("status", "ok") not in {"failed", "error"} and bool(row.get("expression")))
    return float(valid) / float(max(len(rows), 1)) >= float(min_valid_fraction)


def _load_usable_records(sources: dict[str, list[Path]]) -> dict[str, list[PaperRecord]]:
    out: dict[str, list[PaperRecord]] = {}
    for method, paths in sources.items():
        rows: list[PaperRecord] = []
        for path in paths:
            kind = "samples_jsonl" if path.suffix == ".jsonl" else "baseline_json"
            spec = MethodSpec(method, "comparison", "sfsr_method" if method.startswith("SFSR") else "external_comparison", path, kind)
            try:
                rows.extend(_load_and_complete_records(spec))
            except Exception:
                continue
        usable = [rec for rec in rows if rec.status not in {"failed", "error"} and bool(rec.expression)]
        if usable:
            out[method] = rows
    return out


def _final_row(benchmark: str, method: str, strict: dict[str, Any], records: list[PaperRecord]) -> dict[str, Any]:
    valid_count = sum(1 for rec in records if rec.status not in {"failed", "error"} and bool(rec.expression))
    return {
        "benchmark": benchmark,
        "method": method,
        "r2_mean": strict.get("R2 mean", ""),
        "r2_std": strict.get("R2 std", ""),
        "solution_rate": strict.get("solution rate", ""),
        "complexity_mean": strict.get("weighted complexity", strict.get("complexity", "")),
        "valid_pct": _pct(strict.get("valid rate", "")),
        "unique_pct": _pct(strict.get("unique rate", "")),
        "BLEU": strict.get("BLEU", ""),
        "token_similarity": strict.get("token acc", ""),
        "edit_distance": strict.get("edit dist", ""),
        "skeleton_accuracy": strict.get("exact skeleton", ""),
        "n_tasks": len(records),
        "n_valid": valid_count,
    }


def _has_valid_records(records: list[PaperRecord]) -> bool:
    return any(rec.status not in {"failed", "error"} and bool(rec.expression) for rec in records)


def _method_name(path: Path) -> str:
    stem = path.stem.lower()
    if stem.endswith("_seed0"):
        stem = stem[:-6]
    if "unavailable" in stem:
        return ""
    if stem.startswith("e2e_"):
        return "E2E"
    if stem.startswith("gp_") or stem.startswith("gplearn_"):
        return "GP"
    if stem.startswith("deap_"):
        return "GP-DEAP"
    if stem.startswith("dso_"):
        return "DSO"
    if stem.startswith("tpsr") or stem.startswith("tpsr_mcts"):
        return "TPSR"
    if stem.startswith("localdiffusion") or stem.startswith("local_diffusion"):
        return "LocalDiffusionProposal"
    if stem.startswith("symgpt") or stem.startswith("symbolicgpt"):
        return "SymGPT-small"
    if stem.startswith("nesymres"):
        return "NeSymReS-small"
    if stem.startswith("hvae"):
        return "HVAE-small"
    if stem.startswith("nggp"):
        return "NGGP-small"
    return path.stem


def _sfsr_method_name(path: Path) -> str:
    text = str(path.parent.name).lower()
    if "linear_semantic" in text:
        return "SFSR-linear-semantic-clean"
    if "linear_identity" in text:
        return "SFSR-linear-identity-clean"
    return ""


def _archive_clutter(clean_root: Path, root: Path) -> list[str]:
    archive = root / "_archive" / "20260702_before_paper_complete"
    archive.mkdir(parents=True, exist_ok=True)
    archived: list[str] = []
    for path in [
        clean_root / "method_comparison" / "protected_r2_overview.csv",
        clean_root / "method_comparison" / "protected_r2_overview.md",
        clean_root / "method_comparison" / "protected_r2_overview.json",
        clean_root / "method_comparison" / "robust_r2_overview.csv",
        clean_root / "method_comparison" / "robust_r2_overview.md",
        clean_root / "method_comparison" / "robust_r2_overview.json",
    ]:
        if not path.exists():
            continue
        target = archive / path.name
        shutil.move(str(path), str(target))
        archived.append(str(target))
    return archived


def _pct(value: Any) -> Any:
    try:
        return round(float(value) * 100.0, 6)
    except (TypeError, ValueError):
        return ""


def _slug(text: str) -> str:
    return str(text).lower().replace("/", "_").replace(" ", "_").replace("-", "_")


def _readme(root: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    methods = sorted({str(row["method"]) for row in rows})
    return "\n".join([
        "# Paper Complete Benchmark Results",
        "",
        f"Root: `{root}`",
        "",
        "Final tables are in `final_tables/`; per-task comparisons are in `per_task/`; copied source runs are in `source_runs/`.",
        "",
        "R2 mean/std exclude task records with `R2 < -1`. `valid_pct` and `unique_pct` are percentages.",
        "BLEU, token similarity, and edit distance follow the DiffSR `final_diffusion_model.ipynb` token convention.",
        "",
        "Small-trained baselines (`SymGPT-small`, `NeSymReS-small`, `HVAE-small`, `NGGP-small`) are local reduced-budget adaptations trained on SymbolicGPT-large train+val. They are not official pretrained checkpoint numbers.",
        "",
        f"Methods included: {', '.join(methods) if methods else 'none'}",
        f"Generated files: {len(manifest.get('generated_files', []))}",
        "",
    ])


if __name__ == "__main__":
    main()
