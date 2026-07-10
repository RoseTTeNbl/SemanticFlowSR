#!/usr/bin/env python
"""Generate a local SymbolicGPT-style subset for SPFF training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from semflow_sr.data.symbolicgpt_subset import SymbolicGPTSubsetConfig, generate_symbolicgpt_subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/generated/symbolicgpt_subset")
    parser.add_argument("--train_count", type=int, default=747)
    parser.add_argument("--val_count", type=int, default=160)
    parser.add_argument("--test_count", type=int, default=161)
    parser.add_argument("--num_vars", type=int, default=3)
    parser.add_argument("--num_points", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x_min", type=float, default=-5.0)
    parser.add_argument("--x_max", type=float, default=5.0)
    parser.add_argument(
        "--ops",
        nargs="+",
        default=[
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
    args = parser.parse_args()

    manifest = generate_symbolicgpt_subset(
        SymbolicGPTSubsetConfig(
            root=Path(args.root),
            train_count=int(args.train_count),
            val_count=int(args.val_count),
            test_count=int(args.test_count),
            num_vars=int(args.num_vars),
            num_points=int(args.num_points),
            max_depth=int(args.max_depth),
            seed=int(args.seed),
            x_range=(float(args.x_min), float(args.x_max)),
            ops=tuple(args.ops),
        )
    )
    print(json.dumps({
        "root": str(Path(args.root)),
        "splits": manifest["splits"],
        "manifest": str(Path(args.root) / "manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
