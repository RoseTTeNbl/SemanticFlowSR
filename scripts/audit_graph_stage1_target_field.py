#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_complete_expression_semantic_fm import (
    DEFAULT_OPS,
    FixedSymbolTemplate,
    build_task_bundles,
    expand_ops,
    graph_block_mask,
    load_all_task_sources,
    readout_block_index,
)


def _source_kind(template: FixedSymbolTemplate, src: int) -> str:
    value = int(src)
    if value == int(template.zero_source_index):
        return "zero"
    if value == int(template.one_source_index):
        return "one"
    if 0 <= value < int(template.num_vars):
        return "variable"
    if int(template.base_count) <= value < int(template.source_count):
        return "node"
    return "other"


def _target_choice_prob(*, support_count: int, high: float, low: float) -> float:
    if int(support_count) <= 1:
        return 1.0
    numerator = math.exp(float(high))
    denominator = numerator + float(int(support_count) - 1) * math.exp(float(low))
    return float(numerator / max(denominator, 1.0e-300))


def _trace_audit(template: FixedSymbolTemplate, task: Any, trace: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    active = set(int(v) for v in trace["active_block_indices"])
    choices = list(map(int, trace["choices"]))
    block_count = len(template.blocks)
    inactive_count = max(block_count - len(active), 0)
    inactive_weight = float(args.inactive_block_loss_weight) if str(args.inactive_block_target_mode) == "zero" else 0.0
    inactive_default_weight = float(inactive_count) * max(float(inactive_weight), 0.0)
    active_readout = 0
    active_zero_readout = 0
    active_extra_readout_zero = 0
    active_nonreadout = 0
    active_zero_source_edges = 0
    active_logprob_sum = 0.0
    active_probs: list[float] = []
    for bidx in sorted(active):
        block = template.blocks[int(bidx)]
        choice = choices[int(bidx)] if int(bidx) < len(choices) else 0
        mask = graph_block_mask(template, int(bidx), device=torch.device("cpu"))
        support_count = int(mask.sum().item())
        prob = _target_choice_prob(support_count=support_count, high=float(args.target_high), low=float(args.target_low))
        active_probs.append(prob)
        active_logprob_sum += math.log(max(prob, 1.0e-300))
        if block.kind == "readout":
            active_readout += 1
            if choice == int(template.zero_source_index):
                active_zero_readout += 1
                if int(block.term) > 0:
                    active_extra_readout_zero += 1
        else:
            active_nonreadout += 1
            if choice == int(template.zero_source_index):
                active_zero_source_edges += 1
    term0_idx = readout_block_index(template, 0)
    term0_choice = choices[term0_idx] if term0_idx < len(choices) else -1
    term0_kind = _source_kind(template, int(term0_choice))
    active_structural_weight = float(active_nonreadout) + (0.0 if term0_kind == "zero" else 1.0)
    zero_default_weight = float(active_zero_readout) + float(inactive_default_weight)
    total_weight = float(len(active)) + float(inactive_default_weight)
    return {
        "task_id": str(task.task_id),
        "split": str(task.split),
        "suite": str(task.suite),
        "ground_truth": str(task.ground_truth),
        "expression_string": str(trace.get("expression_string", "")),
        "block_count": int(block_count),
        "active_count": int(len(active)),
        "inactive_count": int(inactive_count),
        "active_nonreadout_weight": float(active_nonreadout),
        "active_readout_weight": float(active_readout),
        "active_zero_readout_weight": float(active_zero_readout),
        "active_extra_readout_zero_weight": float(active_extra_readout_zero),
        "active_zero_source_edge_count": int(active_zero_source_edges),
        "inactive_default_weight": float(inactive_default_weight),
        "zero_default_weight": float(zero_default_weight),
        "active_structural_weight": float(active_structural_weight),
        "total_target_weight": float(total_weight),
        "zero_default_weight_fraction": float(zero_default_weight / max(total_weight, 1.0e-8)),
        "zero_to_structural_weight_ratio": float(zero_default_weight / max(active_structural_weight, 1.0e-8)),
        "term0_source": int(term0_choice),
        "term0_source_kind": str(term0_kind),
        "term0_is_nonzero": bool(term0_kind != "zero"),
        "active_target_prob_mean": float(mean(active_probs)) if active_probs else 0.0,
        "active_target_logprob_sum": float(active_logprob_sum),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def m(key: str) -> float:
        vals = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float))]
        return float(mean(vals)) if vals else 0.0

    return {
        "trace_count": int(len(rows)),
        "active_count_mean": m("active_count"),
        "inactive_count_mean": m("inactive_count"),
        "active_nonreadout_weight_mean": m("active_nonreadout_weight"),
        "active_readout_weight_mean": m("active_readout_weight"),
        "active_zero_readout_weight_mean": m("active_zero_readout_weight"),
        "active_extra_readout_zero_weight_mean": m("active_extra_readout_zero_weight"),
        "inactive_default_weight_mean": m("inactive_default_weight"),
        "zero_default_weight_mean": m("zero_default_weight"),
        "active_structural_weight_mean": m("active_structural_weight"),
        "total_target_weight_mean": m("total_target_weight"),
        "zero_default_weight_fraction_mean": m("zero_default_weight_fraction"),
        "zero_to_structural_weight_ratio_mean": m("zero_to_structural_weight_ratio"),
        "term0_nonzero_rate": float(mean([1.0 if row.get("term0_is_nonzero") else 0.0 for row in rows])) if rows else 0.0,
        "active_target_prob_mean": m("active_target_prob_mean"),
        "active_target_logprob_sum_mean": m("active_target_logprob_sum"),
    }


def _write_md(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Graph Stage1 Target Field Audit",
        "",
        f"- output terms: {report['config']['output_terms']}",
        f"- inactive mode/weight: `{report['config']['inactive_block_target_mode']}` / {report['config']['inactive_block_loss_weight']}",
        f"- traces: {summary['trace_count']}",
        f"- active count mean: {summary['active_count_mean']:.6f}",
        f"- inactive default weight mean: {summary['inactive_default_weight_mean']:.6f}",
        f"- active ZERO readout weight mean: {summary['active_zero_readout_weight_mean']:.6f}",
        f"- active extra-readout ZERO weight mean: {summary['active_extra_readout_zero_weight_mean']:.6f}",
        f"- active structural weight mean: {summary['active_structural_weight_mean']:.6f}",
        f"- ZERO/default weight fraction mean: {summary['zero_default_weight_fraction_mean']:.6f}",
        f"- ZERO/default to structural ratio mean: {summary['zero_to_structural_weight_ratio_mean']:.6f}",
        f"- term0 nonzero source rate: {summary['term0_nonzero_rate']:.6f}",
        "",
        "## Highest ZERO/default ratios",
    ]
    for row in sorted(report["rows"], key=lambda item: float(item["zero_to_structural_weight_ratio"]), reverse=True)[:12]:
        lines.append(
            f"- `{row['task_id']}` ratio={row['zero_to_structural_weight_ratio']:.4f} "
            f"zero_frac={row['zero_default_weight_fraction']:.4f} active={row['active_count']} "
            f"term0={row['term0_source_kind']} gt=`{row['ground_truth']}`"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--train-task-limit", type=int, default=20)
    parser.add_argument("--eval-task-limit", type=int, default=8)
    parser.add_argument("--max-train-points", type=int, default=64)
    parser.add_argument("--max-eval-points", type=int, default=64)
    parser.add_argument("--num-vars", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--ops", default=",".join(DEFAULT_OPS))
    parser.add_argument("--op-copies", type=int, default=1)
    parser.add_argument("--output-terms", type=int, default=1)
    parser.add_argument("--gt-traces-per-task", type=int, default=4)
    parser.add_argument("--trace-copy-assignment", choices=["canonical", "random"], default="canonical")
    parser.add_argument("--inactive-block-target-mode", choices=["start", "zero"], default="start")
    parser.add_argument("--inactive-block-loss-weight", type=float, default=0.0)
    parser.add_argument("--target-high", type=float, default=4.0)
    parser.add_argument("--target-low", type=float, default=-4.0)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    template = FixedSymbolTemplate(
        num_vars=int(args.num_vars),
        num_layers=int(args.num_layers),
        ops=expand_ops(str(args.ops), int(args.op_copies)),
        output_terms=int(args.output_terms),
    )
    train_src, eval_src, source_counts = load_all_task_sources(args, int(args.num_vars), device)
    train_tasks = build_task_bundles(
        train_src,
        template,
        traces_per_task=int(args.gt_traces_per_task),
        max_train_points=int(args.max_train_points),
        max_eval_points=int(args.max_eval_points),
        device=device,
        seed=int(args.seed),
        split="train",
        copy_assignment=str(args.trace_copy_assignment),
    )
    eval_tasks = build_task_bundles(
        eval_src,
        template,
        traces_per_task=int(args.gt_traces_per_task),
        max_train_points=int(args.max_train_points),
        max_eval_points=int(args.max_eval_points),
        device=device,
        seed=int(args.seed) + 99_001,
        split="eval",
        copy_assignment=str(args.trace_copy_assignment),
    )
    rows: list[dict[str, Any]] = []
    for task in train_tasks + eval_tasks:
        for trace in task.traces:
            rows.append(_trace_audit(template, task, trace, args))
    report = {
        "config": {
            "num_layers": int(args.num_layers),
            "output_terms": int(args.output_terms),
            "op_copies": int(args.op_copies),
            "source_count": int(template.source_count),
            "block_count": int(len(template.blocks)),
            "inactive_block_target_mode": str(args.inactive_block_target_mode),
            "inactive_block_loss_weight": float(args.inactive_block_loss_weight),
            "target_high": float(args.target_high),
            "target_low": float(args.target_low),
        },
        "source_counts": source_counts,
        "summary": _summarize(rows),
        "rows": rows,
    }
    (out_dir / "target_field_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (out_dir / "target_field_audit.csv").open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["task_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    _write_md(report, out_dir / "target_field_audit.md")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
