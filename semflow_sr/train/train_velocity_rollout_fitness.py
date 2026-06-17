"""Entry: train velocity model with rollout-fitness group-advantage endpoint."""
from __future__ import annotations
import argparse, yaml
from pathlib import Path
from .train_velocity_gt import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    run(yaml.safe_load(Path(a.config).read_text()), target="rollout_fitness_advantage")


if __name__ == "__main__":
    main()
