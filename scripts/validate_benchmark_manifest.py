#!/usr/bin/env python
"""Validate all CSV splits referenced by a unified benchmark manifest."""
from __future__ import annotations

import argparse
import sys

from semflow_sr.data.benchmark_validate import validate_benchmark_manifest, write_validation_reports


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/benchmark_suites/benchmark_manifest.json")
    ap.add_argument("--root", default="data/benchmark_suites")
    ap.add_argument("--suite", nargs="+", default=None)
    ap.add_argument("--require-val", action="store_true")
    ap.add_argument("--out", default="results/dataset_validation")
    ap.add_argument("--fail-on-error", action="store_true")
    args = ap.parse_args()

    result = validate_benchmark_manifest(
        args.manifest,
        root=args.root,
        suites=args.suite,
        require_val=args.require_val,
    )
    write_validation_reports(result, args.out)
    print(
        f"validated {result.summary['n_valid']}/{result.summary['n_tasks']} tasks "
        f"across {result.summary['n_suites']} suites"
    )
    print(f"reports: {args.out}")
    if args.fail_on_error and not result.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
