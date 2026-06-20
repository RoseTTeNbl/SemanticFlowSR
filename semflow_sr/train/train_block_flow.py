"""Train block-only Semantic-Fisher RiskFlow models."""
from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path
from dataclasses import replace

import torch
import yaml
from torch.utils.data import DataLoader

from ..data.synthetic_generator import GenConfig
from ..models.block_flow_model import BlockFlowModel, BlockFlowModelConfig
from ..semantics.energy import ActionEnergyConfig
from ..sr.ops import N_OPS
from ..utils.checkpoint import save_checkpoint
from ..utils.seed import set_seed, get_device
from .block_flow_dataset import BlockFlowBuildConfig, build_block_flow_dataset, collate_block_flow


def run(cfg: dict):
    set_seed(cfg.get("seed", 0))
    device = get_device()
    g = cfg["gen"]
    gen = GenConfig(
        num_vars=int(g["num_vars"]),
        max_depth=int(g["max_depth"]),
        K=int(g["K"]),
        probe_size=int(g["probe_size"]),
        ops=tuple(g["ops"]),
    )
    block_size = int(cfg.get("block", {}).get("size", 3))
    action_vocab_size = N_OPS * gen.K * gen.K * gen.K
    model_cfg = BlockFlowModelConfig(
        d=gen.num_vars,
        K=gen.K,
        block_size=block_size,
        action_vocab_size=action_vocab_size,
        hidden=int(cfg.get("model", {}).get("hidden", 96)),
    )
    model = BlockFlowModel(model_cfg).to(device)
    energy_cfg = ActionEnergyConfig(**cfg.get("energy", {}))
    build_cfg = BlockFlowBuildConfig(
        block_size=block_size,
        num_trajectories=int(cfg.get("sampling", {}).get("num_trajectories", 16)),
        max_blocks=int(cfg.get("sampling", {}).get("max_blocks", 2)),
        block_pool_budget=int(cfg.get("sampling", {}).get("block_pool_budget", 64)),
        risk_alpha=float(cfg.get("risk", {}).get("alpha", 0.1)),
        risk_mode=str(cfg.get("risk", {}).get("mode", "top_alpha")),
        risk_normalize=str(cfg.get("risk", {}).get("normalize", "rank")),
        beta=float(cfg.get("flow", {}).get("beta", 1.0)),
        gamma=float(cfg.get("flow", {}).get("gamma", 0.1)),
        gram_rank=cfg.get("flow", {}).get("gram_rank", 8),
        num_time_samples=int(cfg.get("flow", {}).get("num_time_samples", 1)),
    )
    train_cfg = cfg.get("train", {})
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 5e-4)))
    total_steps = int(train_cfg.get("steps", 100))
    iterations = max(int(train_cfg.get("on_policy_iterations", 1)), 1)
    if "steps_per_iteration" in train_cfg:
        steps_per_iteration = max(int(train_cfg["steps_per_iteration"]), 1)
    else:
        steps_per_iteration = max(int(math.ceil(total_steps / iterations)), 1)
    log_every = int(train_cfg.get("log_every", 10))
    rows = []
    step = 0
    ckpt_name = cfg.get("checkpoint_name", "block_risk_flow_h3.pt")
    for iteration in range(iterations):
        behavior_model = copy.deepcopy(model).to("cpu")
        behavior_model.eval()
        iter_build_cfg = replace(
            build_cfg,
            behavior_policy_id=f"{Path(ckpt_name).stem}:iter_{iteration}",
        )
        dataset = build_block_flow_dataset(
            gen,
            num_tasks=int(cfg.get("num_tasks", 8)),
            behavior_model=behavior_model,
            seed=int(cfg.get("seed", 0)) + iteration,
            energy_cfg=energy_cfg,
            cfg=iter_build_cfg,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(train_cfg.get("batch_size", 4)),
            shuffle=True,
            collate_fn=collate_block_flow,
        )
        local_step = 0
        while local_step < steps_per_iteration:
            for batch in loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = model(
                    B=batch["B"],
                    y=batch["y"],
                    q_lambda=batch["q_lambda"],
                    lambda_value=batch["lambda"],
                    mask=batch["mask"],
                    zeta=batch["zeta"],
                )
                diff = (out.z_dot_pred - batch["zdot_target"]) * batch["mask"]
                loss = (diff * diff).sum() / batch["mask"].sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
                opt.step()
                if step % log_every == 0:
                    row = {
                        "step": step,
                        "iteration": iteration,
                        "loss": float(loss.detach().cpu().item()),
                        "support_size_mean": float(batch["mask"].sum(dim=(1, 2)).float().mean().detach().cpu().item()),
                        "num_records": len(dataset),
                    }
                    rows.append(row)
                    print(f"iter {iteration} step {step} loss {row['loss']:.6f}")
                step += 1
                local_step += 1
                if local_step >= steps_per_iteration:
                    break
    out_dir = Path(cfg.get("out", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        out_dir / ckpt_name,
        model,
        meta={
            "cfg": cfg,
            "model_cfg": model_cfg.__dict__,
            "final_loss": rows[-1]["loss"] if rows else None,
            "on_policy_iterations": iterations,
            "steps_per_iteration": steps_per_iteration,
            "gradient_steps": step,
        },
    )
    _write_curve(out_dir / f"train_curve_{Path(ckpt_name).stem}.csv", rows)
    print(f"saved {out_dir / ckpt_name}")


def _write_curve(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
