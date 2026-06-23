#!/usr/bin/env python
"""Archive paper-facing SR metrics and figures from method result files."""
from __future__ import annotations

import argparse
from pathlib import Path

from semflow_sr.eval.paper_metrics import (
    MethodSpec,
    plot_complexity_pareto,
    plot_metric_summary,
    plot_structural_metrics,
    write_archive,
)


def parse_method(values: list[str]) -> MethodSpec:
    name, group, role, kind, path = values
    return MethodSpec(name=name, group=group, role=role, kind=kind, path=Path(path))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/paper_metrics/latest")
    ap.add_argument("--bootstrap_samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suite", nargs="+", default=None, help="optional suite filter applied to every method")
    ap.add_argument(
        "--method",
        nargs=5,
        action="append",
        metavar=("NAME", "GROUP", "ROLE", "KIND", "PATH"),
        required=True,
        help="method spec; ROLE is sfsr_method, external_comparison, or native_protocol_reference",
    )
    args = ap.parse_args(argv)
    specs = [parse_method(item) for item in args.method or []]
    out = Path(args.out)
    manifest = write_archive(
        specs,
        out,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        suites=args.suite,
    )
    plot_metric_summary(manifest["summary_rows"], out)
    plot_complexity_pareto(manifest["summary_rows"], out)
    plot_structural_metrics(manifest["summary_rows"], out)
    print(f"saved paper metrics to {out}")


if __name__ == "__main__":
    main()
