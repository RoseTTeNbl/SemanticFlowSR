#!/usr/bin/env python
"""Run or print SFSR benchmark matrix commands over the unified manifest."""
from __future__ import annotations

import argparse
import subprocess

from semflow_sr.eval.benchmark_matrix import (
    build_sfsr_matrix_commands,
    load_matrix_config,
    write_command_plan,
)


def _parse_ckpt_by_vars(raw: list[str] | None) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for item in raw or []:
        if ":" not in item:
            raise ValueError(f"bad checkpoint mapping {item!r}; expected N:path")
        key, value = item.split(":", 1)
        mapping[int(key)] = value
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval/sfsr_full_benchmark.yaml")
    ap.add_argument("--suite_group", nargs="+", default=None)
    ap.add_argument("--method", nargs="+", default=None)
    ap.add_argument("--ckpt_by_vars", nargs="*", default=None)
    ap.add_argument("--plan_out", default="results/benchmark_plans/sfsr_risk_flow_commands.json")
    ap.add_argument("--execute", action="store_true", help="execute commands; default only prints and writes a plan")
    args = ap.parse_args()

    config = load_matrix_config(args.config)
    commands = build_sfsr_matrix_commands(
        config,
        ckpt_by_vars=_parse_ckpt_by_vars(args.ckpt_by_vars) or None,
        suite_groups=args.suite_group,
        methods=args.method,
    )
    write_command_plan(commands, args.plan_out)
    for command in commands:
        print(command.shell())
        if args.execute:
            subprocess.run(command.argv, check=True)
    print(f"wrote command plan: {args.plan_out}")


if __name__ == "__main__":
    main()
