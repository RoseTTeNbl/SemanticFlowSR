#!/usr/bin/env python
"""Run or print external-baseline benchmark matrix commands."""
from __future__ import annotations

import argparse
import subprocess

from semflow_sr.eval.benchmark_matrix import (
    build_external_baseline_commands,
    load_matrix_config,
    write_command_plan,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval/external_baselines.yaml")
    ap.add_argument("--suite_group", nargs="+", default=None)
    ap.add_argument("--method", nargs="+", default=None)
    ap.add_argument("--conda_exe", default="conda")
    ap.add_argument("--plan_out", default="results/benchmark_plans/external_baseline_commands.json")
    ap.add_argument("--execute", action="store_true", help="execute commands; default only prints and writes a plan")
    args = ap.parse_args()

    config = load_matrix_config(args.config)
    commands = build_external_baseline_commands(
        config,
        suite_groups=args.suite_group,
        methods=args.method,
        conda_exe=args.conda_exe,
    )
    write_command_plan(commands, args.plan_out)
    for command in commands:
        print(command.shell())
        if args.execute:
            subprocess.run(command.argv, check=True)
    print(f"wrote command plan: {args.plan_out}")


if __name__ == "__main__":
    main()
