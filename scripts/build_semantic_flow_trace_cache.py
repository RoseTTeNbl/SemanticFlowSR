#!/usr/bin/env python3
"""Build the strict equivalent-GT trace cache used by v5.1 bootstrap."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semflow_sr.flow.trace_cache import trace_record, write_trace_cache
from scripts.train_complete_expression_semantic_fm import (
    build_task_bundles,
    canonical_construction_graph,
    load_all_task_sources,
    make_construction_template,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cache/semantic_flow_v5")
    parser.add_argument("--manifest", default="data/benchmark_suites/benchmark_manifest.json")
    parser.add_argument("--manifest-root", default="data/benchmark_suites")
    parser.add_argument("--suites", nargs="+", default=["nguyen", "constant", "livermore", "jin"])
    parser.add_argument("--symbolicgpt-root", default="")
    parser.add_argument("--symbolicgpt-train-limit", type=int, default=0)
    parser.add_argument("--symbolicgpt-eval-limit", type=int, default=0)
    parser.add_argument("--symbolicgpt-eval-splits", default="val,test")
    parser.add_argument("--symbolicgpt-point-train-fraction", type=float, default=0.8)
    parser.add_argument("--train-task-limit", type=int, default=0)
    parser.add_argument("--eval-task-limit", type=int, default=0)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--task-id-filter", default="")
    parser.add_argument("--allow-empty-eval", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--num-vars", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-registers", type=int, default=17)
    parser.add_argument("--ops", default="copy,add,sub,mul,protected_div,sin,cos,square,cube,protected_log,protected_sqrt,exp")
    parser.add_argument("--output-terms", type=int, default=1)
    parser.add_argument("--gt-traces-per-task", type=int, default=8)
    parser.add_argument("--max-train-points", type=int, default=64)
    parser.add_argument("--max-eval-points", type=int, default=64)
    parser.add_argument("--trace-copy-assignment", default="canonical")
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()
    args.construction_graph = "register_categorical_blocks"
    template = make_construction_template(args, canonical_construction_graph(args.construction_graph))
    train_raw, eval_raw, counts = load_all_task_sources(args, template.num_vars, torch.device("cpu"))
    bundles = []
    for split, raw, seed in (("train", train_raw, args.seed), ("eval", eval_raw, args.seed + 12_345)):
        bundles.extend(build_task_bundles(
            raw,
            template,
            traces_per_task=args.gt_traces_per_task,
            max_train_points=args.max_train_points,
            max_eval_points=args.max_eval_points,
            device=torch.device("cpu"),
            seed=seed,
            split=split,
            copy_assignment=args.trace_copy_assignment,
        ))
    failed = [task.task_id for task in bundles if not task.traces]
    if failed and not args.allow_incomplete:
        raise RuntimeError(f"cannot build strict trace cache; tasks without valid traces: {failed[:8]}")
    records = [trace_record(task, template, task.traces, task.compile_failures) for task in bundles if task.traces]
    manifest = write_trace_cache(Path(args.out), template, records)
    manifest["skipped_task_ids"] = failed
    manifest_path = Path(args.out) / "compiled_trace_families_v1.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ok", "source_counts": counts, **manifest}, indent=2))


if __name__ == "__main__":
    main()
