#!/usr/bin/env python
"""Materialize SymbolicGPT-style JSON tasks and audit typed-flow compilability."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semflow_sr.data.benchmark_manifest import BenchmarkSuiteSpec, BenchmarkTaskSpec, write_benchmark_manifest
from semflow_sr.data.symbolicgpt_subset import load_symbolicgpt_subset_tasks
from semflow_sr.sr.parser import parse_formula

from scripts.train_sparse_register_flow import (
    TypedOpNodeTemplate,
    _compile_expr,
    _normalize_numeric_constants,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/generated/symbolicgpt_large_2000_200_200")
    ap.add_argument("--out-root", default="data/benchmark_suites/materialized/symbolicgpt_large_2000_200_200")
    ap.add_argument("--manifest-dir", default="data/benchmark_suites/symbolicgpt_large_2000_200_200")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--suite", default="symbolicgpt_large")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-vars", type=int, default=3)
    ap.add_argument("--num-registers", type=int, default=10)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--output-terms", type=int, default=3)
    ap.add_argument(
        "--op-nodes",
        nargs="+",
        default=[
            "copy",
            "add",
            "sub",
            "mul",
            "protected_div",
            "sin",
            "cos",
            "square",
            "cube",
            "exp",
            "protected_log",
            "protected_sqrt",
        ],
    )
    ap.add_argument("--max-paths", type=int, default=1)
    args = ap.parse_args()

    template = TypedOpNodeTemplate(
        num_vars=int(args.num_vars),
        num_registers=int(args.num_registers),
        num_layers=int(args.num_layers),
        op_nodes=tuple(str(v) for v in args.op_nodes),
        output_terms=int(args.output_terms),
    )
    rng = random.Random(int(args.seed))
    out_root = Path(args.out_root)
    manifest_dir = Path(args.manifest_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    all_raw_specs: dict[str, list[BenchmarkTaskSpec]] = {}
    all_compilable_specs: dict[str, list[BenchmarkTaskSpec]] = {}
    audit_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    split_counts: dict[str, dict[str, int]] = {}

    for split in args.splits:
        tasks = load_symbolicgpt_subset_tasks(
            args.root,
            splits=(str(split),),
            limit=None,
            rng=rng,
        )
        raw_specs: list[BenchmarkTaskSpec] = []
        compilable_specs: list[BenchmarkTaskSpec] = []
        for idx, task in enumerate(tasks):
            spec = _write_task(task, out_root=out_root, suite=str(args.suite), split=str(split))
            raw_specs.append(spec)
            reason = "ok"
            try:
                expr = parse_formula(str(task.expression), [f"x{i}" for i in range(int(task.X_train.shape[1]))])
                compiled = _compile_expr(
                    template,
                    expr,
                    max_paths=max(int(args.max_paths), 1),
                    seed=int(args.seed) + int(idx),
                    source="gt",
                )
                if not compiled:
                    normalized = _normalize_numeric_constants(expr)
                    compiled = _compile_expr(
                        template,
                        normalized,
                        max_paths=max(int(args.max_paths), 1),
                        seed=int(args.seed) + int(idx),
                        source="gt:constant_normalized",
                    )
                if compiled:
                    compilable_specs.append(spec)
                else:
                    reason = "not_compilable"
            except Exception as exc:  # noqa: BLE001 - audit should preserve all failure reasons.
                reason = f"{type(exc).__name__}: {str(exc)[:160]}"
            reason_counts[reason] += 1
            audit_rows.append({
                "task_id": spec.task_id,
                "split": split,
                "formula": task.expression,
                "num_vars": int(task.X_train.shape[1]),
                "status": "compilable" if reason == "ok" else "excluded",
                "reason": reason,
            })
        all_raw_specs[str(split)] = raw_specs
        all_compilable_specs[str(split)] = compilable_specs
        split_counts[str(split)] = {
            "raw": len(raw_specs),
            "compilable": len(compilable_specs),
            "excluded": len(raw_specs) - len(compilable_specs),
        }

    _write_manifest(manifest_dir / "symbolicgpt_large_raw_manifest.json", args, all_raw_specs, kind="raw")
    _write_manifest(manifest_dir / "symbolicgpt_large_compilable_manifest.json", args, all_compilable_specs, kind="compilable")
    _write_split_manifests(manifest_dir, args, all_compilable_specs)
    _write_csv(manifest_dir / "symbolicgpt_large_compilability_audit.csv", audit_rows)
    summary = {
        "root": str(args.root),
        "out_root": str(out_root),
        "manifest_dir": str(manifest_dir),
        "suite": str(args.suite),
        "split_counts": split_counts,
        "reason_counts": dict(reason_counts),
        "template": {
            "num_vars": int(args.num_vars),
            "num_registers": int(args.num_registers),
            "num_layers": int(args.num_layers),
            "output_terms": int(args.output_terms),
            "op_nodes": list(args.op_nodes),
        },
        "files": {
            "raw_manifest": str(manifest_dir / "symbolicgpt_large_raw_manifest.json"),
            "compilable_manifest": str(manifest_dir / "symbolicgpt_large_compilable_manifest.json"),
            "audit_csv": str(manifest_dir / "symbolicgpt_large_compilability_audit.csv"),
        },
    }
    (manifest_dir / "symbolicgpt_large_compilability_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


def _write_task(task, *, out_root: Path, suite: str, split: str) -> BenchmarkTaskSpec:
    task_slug = task.name.rsplit("/", 1)[-1]
    task_dir = out_root / split / task_slug
    task_dir.mkdir(parents=True, exist_ok=True)
    train_path = task_dir / "train.csv"
    test_path = task_dir / "test.csv"
    _write_points(train_path, task.X_train, task.y_train, task.variable_names)
    _write_points(test_path, task.X_test, task.y_test, task.variable_names)
    rel_train = train_path.relative_to(Path("data/benchmark_suites"))
    rel_test = test_path.relative_to(Path("data/benchmark_suites"))
    task_id = f"{suite}/{split}/{task_slug}"
    return BenchmarkTaskSpec(
        task_id=task_id,
        suite=suite,
        num_vars=int(task.X_train.shape[1]),
        variable_names=list(task.variable_names),
        train_path=str(rel_train),
        test_path=str(rel_test),
        ground_truth=str(task.expression),
        domain="synthetic_symbolicgpt",
        split=str(split),
        tags=["symbolicgpt_large", str(split)],
        source="data/generated/symbolicgpt_large_2000_200_200",
        metadata={k: v for k, v in dict(task.metadata).items() if k not in {"suite", "split"}},
    )


def _write_points(path: Path, x, y, variables: list[str]) -> None:
    rows = []
    for values, target in zip(x, y):
        row = {name: float(values[idx]) for idx, name in enumerate(variables)}
        row["target"] = float(target)
        rows.append(row)
    pd.DataFrame(rows, columns=[*variables, "target"]).to_csv(path, index=False)


def _write_manifest(path: Path, args, by_split: dict[str, list[BenchmarkTaskSpec]], *, kind: str) -> None:
    manifest = BenchmarkSuiteSpec(
        version="1.0",
        suites={str(args.suite): [spec for specs in by_split.values() for spec in specs]},
        metadata={
            "kind": kind,
            "source": str(args.root),
            "suite": str(args.suite),
            "splits": {split: len(specs) for split, specs in by_split.items()},
        },
    )
    write_benchmark_manifest(manifest, path)


def _write_split_manifests(manifest_dir: Path, args, by_split: dict[str, list[BenchmarkTaskSpec]]) -> None:
    for split, specs in by_split.items():
        write_benchmark_manifest(
            BenchmarkSuiteSpec(
                version="1.0",
                suites={str(args.suite): list(specs)},
                metadata={
                    "kind": "compilable_split",
                    "source": str(args.root),
                    "suite": str(args.suite),
                    "split": str(split),
                    "tasks": len(specs),
                },
            ),
            manifest_dir / f"symbolicgpt_large_{split}_compilable_manifest.json",
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
