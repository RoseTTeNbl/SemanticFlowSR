"""Velocity-matching trainer."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader

from .losses import metric_weighted_velocity_loss
from ..models.semantic_transformer import SemanticTransformer
from ..utils.logging import get_logger

log = get_logger("train")


@dataclass
class TrainConfig:
    lr: float = 3e-4
    steps: int = 2000
    batch_size: int = 16
    metric_weight: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 50


def _move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_velocity(model: SemanticTransformer, dataset, cfg: TrainConfig, device,
                   collate_fn) -> dict:
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate_fn, drop_last=False)
    history = []
    step = 0
    while step < cfg.steps:
        for batch in loader:
            batch = _move(batch, device)
            out = model(
                x=batch["x"], y=batch["y"], B=batch["B"],
                p_lambda=batch["p_lambda"], lambda_value=batch["lambda"],
                action_feats=batch["action_feats"], energies=batch["energies"],
                weights=batch["weights"], action_mask=batch["action_mask"],
            )
            loss = metric_weighted_velocity_loss(
                out.v_pred, batch["dp_dlambda"], batch["p_lambda"], batch["weights"],
                mask=batch["action_mask"], metric_weight=cfg.metric_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            history.append(loss.item())
            if step % cfg.log_every == 0:
                log.info(f"step {step} loss {loss.item():.6f}")
            step += 1
            if step >= cfg.steps:
                break
    return {"loss_history": history, "final_loss": history[-1] if history else None}
