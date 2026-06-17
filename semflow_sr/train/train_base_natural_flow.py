"""Entry: train the base semantic natural-flow velocity model."""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from .train_velocity_gt import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    target = cfg.get("target", {}).get("name", "one_step_advantage")
    run(cfg, target=target)


if __name__ == "__main__":
    main()
