"""Data-conditioned edge-flow model."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .dataset import EdgeFlowRecord


@dataclass(frozen=True)
class EdgeFlowModelConfig:
    num_vars: int
    hidden: int = 96


@dataclass
class EdgeFlowPrediction:
    mixture_zdot: torch.Tensor
    group_zdot: dict[str, torch.Tensor]


class EdgeFlowModel(nn.Module):
    def __init__(self, cfg: EdgeFlowModelConfig):
        super().__init__()
        self.cfg = cfg
        self.task = nn.Sequential(
            nn.Linear(2 * int(cfg.num_vars) + 4, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.SiLU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(cfg.hidden + 8, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, 1),
        )

    def forward(self, record: EdgeFlowRecord) -> EdgeFlowPrediction:
        task_emb = self.task(_task_features(record.x, record.y, self.cfg.num_vars))
        mixture_zdot = self._predict_simplex_velocity(
            task_emb,
            record.theta_lambda.mixture_probs.unsqueeze(0),
            group_type_id=4.0,
            layer_id=record.template.num_layers,
            group_index=0,
        ).squeeze(0)
        group_out: dict[str, torch.Tensor] = {}
        for group_index, group in enumerate(record.template.groups):
            probs = record.theta_lambda.group_probs[group.group_id]
            group_out[group.group_id] = self._predict_simplex_velocity(
                task_emb,
                probs,
                group_type_id=_group_type_id(group.group_type),
                layer_id=group.layer_id,
                group_index=group_index,
            )
        return EdgeFlowPrediction(mixture_zdot=mixture_zdot, group_zdot=group_out)

    def _predict_simplex_velocity(
        self,
        task_emb: torch.Tensor,
        probs: torch.Tensor,
        *,
        group_type_id: float,
        layer_id: int,
        group_index: int,
    ) -> torch.Tensor:
        H, d = probs.shape
        z = probs.clamp_min(1e-12).sqrt()
        rows = []
        for h in range(H):
            k = torch.arange(d, dtype=probs.dtype, device=probs.device)
            features = torch.stack([
                torch.full_like(k, float(h)),
                torch.full_like(k, float(group_index)),
                torch.full_like(k, float(layer_id)),
                torch.full_like(k, float(group_type_id)),
                k / max(float(d - 1), 1.0),
                probs[h],
                probs[h].clamp_min(1e-12).log(),
                torch.full_like(k, _entropy(probs[h])),
            ], dim=1)
            task = task_emb.expand(d, -1)
            raw = self.edge_mlp(torch.cat([task, features], dim=1)).squeeze(1)
            raw = raw - (probs[h] * raw).sum()
            zdot = 0.5 * z[h] * raw
            zdot = zdot - (z[h] * zdot).sum() * z[h]
            rows.append(zdot)
        return torch.stack(rows, dim=0)


def edge_flow_loss(pred: EdgeFlowPrediction, record: EdgeFlowRecord) -> tuple[torch.Tensor, dict]:
    loss_mix = ((pred.mixture_zdot - record.zdot_mixture) ** 2).mean()
    losses = [((pred.group_zdot[key] - target) ** 2).mean() for key, target in record.zdot_groups.items()]
    loss_groups = torch.stack(losses).mean() if losses else torch.tensor(0.0)
    loss = loss_mix + loss_groups
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "loss_mixture": float(loss_mix.detach().cpu().item()),
        "loss_groups": float(loss_groups.detach().cpu().item()),
    }


def _task_features(x: torch.Tensor, y: torch.Tensor, num_vars: int) -> torch.Tensor:
    x = x.float()
    y = y.float()
    means = torch.zeros(num_vars, dtype=x.dtype, device=x.device)
    stds = torch.zeros(num_vars, dtype=x.dtype, device=x.device)
    used = min(int(num_vars), int(x.shape[1]))
    if used:
        means[:used] = x[:, :used].mean(dim=0)
        stds[:used] = x[:, :used].std(dim=0, unbiased=False)
    y_mean = y.mean()
    y_std = y.std(unbiased=False)
    extras = torch.tensor([y_mean, y_std, y.min(), y.max()], dtype=x.dtype, device=x.device)
    return torch.cat([means, stds, extras], dim=0).unsqueeze(0)


def _group_type_id(group_type: str) -> float:
    return {
        "ARG_SELECT": 1.0,
        "REG_UPDATE": 2.0,
        "OUTPUT_SELECT": 3.0,
    }.get(group_type, 0.0)


def _entropy(p: torch.Tensor) -> float:
    q = p.clamp_min(1e-12)
    return float((-(q * q.log()).sum()).detach().cpu().item())
