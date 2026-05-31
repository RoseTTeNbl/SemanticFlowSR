"""Entry: train velocity model with GT target endpoint.

Usage: python -m semflow_sr.train.train_velocity_gt --config configs/train/velocity_gt.yaml
"""
from __future__ import annotations
import argparse, yaml
from pathlib import Path

from .build_dataset import build_dataset
from .trainer_velocity import train_velocity, TrainConfig
from ..data.synthetic_generator import GenConfig
from ..data.collate import collate_velocity
from ..models.semantic_transformer import SemanticTransformer, SemanticTransformerConfig
from ..semantics.energy import ActionEnergyConfig
from ..utils.seed import set_seed, get_device
from ..utils.checkpoint import save_checkpoint


def run(cfg: dict, target: str):
    set_seed(cfg.get("seed", 0))
    device = get_device()
    g = cfg["gen"]
    gen = GenConfig(num_vars=g["num_vars"], max_depth=g["max_depth"], K=g["K"],
                    probe_size=g["probe_size"], ops=tuple(g["ops"]))
    ecfg = ActionEnergyConfig(**cfg.get("energy", {}))
    ds = build_dataset(gen, cfg["num_tasks"], target=target, eta=cfg.get("eta", 1.0),
                       seed=cfg.get("seed", 0), max_support=cfg.get("max_support", 256),
                       energy_cfg=ecfg)
    model = SemanticTransformer(SemanticTransformerConfig(
        d=gen.num_vars, K=gen.K, hidden=cfg["model"]["hidden"],
        row_layers=cfg["model"]["row_layers"], heads=cfg["model"]["heads"]))
    tcfg = TrainConfig(**cfg["train"])
    stats = train_velocity(model, ds, tcfg, device, collate_velocity)
    out = Path(cfg.get("out", "checkpoints")) / f"velocity_{target}.pt"
    save_checkpoint(out, model, meta={"final_loss": stats["final_loss"], "cfg": cfg})
    print(f"saved {out}  final_loss={stats['final_loss']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    run(yaml.safe_load(Path(a.config).read_text()), target="gt")


if __name__ == "__main__":
    main()
