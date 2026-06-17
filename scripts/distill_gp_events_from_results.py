#!/usr/bin/env python
"""Extract lightweight GP operator events from baseline GP result JSON.

The output is consumed by ``scripts/run_experiment.py --gp_distill_events``. It is a
compact policy-distillation bridge: operators that appear in solved/high-R2 GP
expressions receive higher likelihood under ``GPPolicyDistillationPrior``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

OPS = ["add", "sub", "mul", "protected_div", "div", "sin", "cos", "square", "cube", "exp", "log", "sqrt"]


def expression_operator_events(name: str, item: dict, solved_threshold: float = 0.999) -> list[dict]:
    expr = str(item.get("expression", ""))
    r2 = float(item.get("r2", 0.0))
    solved = bool(r2 >= solved_threshold)
    events = []
    for op in OPS:
        count = len(re.findall(rf"\b{re.escape(op)}\b", expr))
        if count == 0:
            continue
        normalized = {
            "div": "protected_div",
            "log": "protected_log",
            "sqrt": "protected_sqrt",
        }.get(op, op)
        events.append({
            "task": name,
            "op": normalized,
            "r2": r2,
            "solved": solved,
            "weight": float(count),
        })
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/deap/deap_all_seed0.json")
    ap.add_argument("--out", default="results/gp_distill/deap_operator_events.json")
    ap.add_argument("--solved_threshold", type=float, default=0.999)
    args = ap.parse_args()
    data = json.loads(Path(args.input).read_text())
    events = []
    for name, item in data.items():
        events.extend(expression_operator_events(name, item, solved_threshold=args.solved_threshold))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": args.input, "events": events}, indent=2))
    print(f"saved {len(events)} GP distillation events to {out}")


if __name__ == "__main__":
    main()
