#!/usr/bin/env python
"""Run/import the local diffusion SR repository as a native-protocol reference."""
from __future__ import annotations

import argparse
from pathlib import Path

from semflow_sr.eval.external_adapters import (
    build_local_diffusion_reference_records,
    write_json,
)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="external/Symbolic_Regression_With_Diffusion_Models")
    ap.add_argument("--out", default="results/native_references/local_diffusion_direct_reference.json")
    ap.add_argument("--status_out", default="results/native_references/local_diffusion_direct_status.json")
    args = ap.parse_args(argv)

    records, status = build_local_diffusion_reference_records(Path(args.root))
    write_json(args.out, records)
    write_json(args.status_out, status.to_json())
    print(f"wrote local diffusion reference records: {args.out}")
    print(f"wrote local diffusion run status: {args.status_out}")
    if not status.approach2_direct_runnable:
        print("Approach2 direct run skipped; missing assets:", ", ".join(status.approach2_missing))
    if not status.approach3_direct_runnable:
        print("Approach3 direct run skipped; missing assets:", ", ".join(status.approach3_missing))


if __name__ == "__main__":
    main()

