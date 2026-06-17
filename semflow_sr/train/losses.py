"""Training losses for SemanticFlowSR.

The current mainline is semantic-Fisher sphere-tangent matching for a log-rate model.
Plain Fisher path losses remain as explicit ablations.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from ..flow.natural_path import (
    natural_endpoint_from_potential,
    natural_path_from_potential,
    sphere_path_from_potential,
)
from ..flow.semantic_fisher import semantic_fisher_sphere_step


def masked_kl(p_target: torch.Tensor, p_pred: torch.Tensor, mask: torch.Tensor | None = None,
              eps: float = 1e-12) -> torch.Tensor:
    p_t = p_target.clamp(min=eps)
    p_p = p_pred.clamp(min=eps)
    kl = p_t * (p_t.log() - p_p.log())
    if mask is not None:
        kl = kl * mask
    return kl.sum(dim=-1).mean()


def _masked_rank_corr_like(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    vals = []
    for aa, bb, mm in zip(a.detach(), b.detach(), mask.detach()):
        aa = aa[mm].float()
        bb = bb[mm].float()
        if aa.numel() < 2:
            continue
        aa = aa - aa.mean()
        bb = bb - bb.mean()
        denom = aa.std(unbiased=False) * bb.std(unbiased=False)
        if float(denom.cpu()) > 1e-12:
            vals.append(float(((aa * bb).mean() / denom.clamp(min=1e-12)).cpu()))
    return float(sum(vals) / len(vals)) if vals else 0.0


@dataclass
class SpherePathLoss:
    """Square-root Fisher-sphere path matching for the old potential ablation."""

    lambda_min: float = 0.05
    num_lambda_samples: int = 1
    eps: float = 1e-12

    def __call__(
        self,
        p_start: torch.Tensor,
        advantage_target: torch.Tensor,
        score_pred: torch.Tensor,
        beta: torch.Tensor | float,
        mask: torch.Tensor | None = None,
        lambda_value: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        p_start = p_start.clamp(min=self.eps)
        if mask is not None:
            p_start = p_start * mask
        p_start = p_start / p_start.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        advantage_target = torch.nan_to_num(advantage_target.to(device=p_start.device, dtype=p_start.dtype))
        score_pred = torch.nan_to_num(score_pred.to(device=p_start.device, dtype=p_start.dtype))
        if mask is not None:
            advantage_target = advantage_target.masked_fill(~mask, 0.0)
            score_pred = score_pred.masked_fill(~mask, 0.0)

        losses = []
        tvs = []
        vmses = []
        endpoint_l2s = []
        n_samples = max(int(self.num_lambda_samples), 1)
        for _ in range(n_samples):
            if lambda_value is None:
                lam = self.lambda_min + (1.0 - self.lambda_min) * torch.rand(
                    p_start.shape[:-1], device=p_start.device, dtype=p_start.dtype
                )
            else:
                lam = torch.as_tensor(lambda_value, device=p_start.device, dtype=p_start.dtype)
            z_star, _ = sphere_path_from_potential(
                p_start, advantage_target, beta=beta, lambda_value=lam, eps=self.eps
            )
            z_pred, _ = sphere_path_from_potential(
                p_start, score_pred, beta=beta, lambda_value=lam, eps=self.eps
            )
            z_star = z_star.detach()
            zdiff = z_star - z_pred
            if mask is not None:
                zdiff = zdiff * mask
                denom = mask.sum().clamp(min=1)
                losses.append((zdiff * zdiff).sum() / denom)
            else:
                losses.append((zdiff * zdiff).mean())

            p_star, v_star = natural_path_from_potential(
                p_start, advantage_target, beta=beta, lambda_value=lam, eps=self.eps
            )
            p_pred, v_pred = natural_path_from_potential(
                p_start, score_pred, beta=beta, lambda_value=lam, eps=self.eps
            )
            p_star = p_star.detach()
            v_star = v_star.detach()
            diff = (p_star - p_pred).abs()
            if mask is not None:
                diff = diff * mask
            tvs.append(0.5 * diff.sum(dim=-1).mean())
            vdiff = v_star - v_pred
            if mask is not None:
                vdiff = vdiff * mask
                denom = mask.sum().clamp(min=1)
                vmses.append((vdiff * vdiff).sum() / denom)
            else:
                vmses.append((vdiff * vdiff).mean())

            z_end_star, _ = sphere_path_from_potential(
                p_start, advantage_target, beta=beta, lambda_value=torch.ones_like(lam), eps=self.eps
            )
            z_end_pred, _ = sphere_path_from_potential(
                p_start, score_pred, beta=beta, lambda_value=torch.ones_like(lam), eps=self.eps
            )
            z_end_star = z_end_star.detach()
            endpoint_diff = z_end_star - z_end_pred
            if mask is not None:
                endpoint_diff = endpoint_diff * mask
                endpoint_l2s.append((endpoint_diff * endpoint_diff).sum() / mask.sum().clamp(min=1))
            else:
                endpoint_l2s.append((endpoint_diff * endpoint_diff).mean())

        loss = torch.stack(losses).mean()
        endpoint_star = natural_endpoint_from_potential(p_start, advantage_target, beta=beta, eps=self.eps).detach()
        endpoint_pred = natural_endpoint_from_potential(p_start, score_pred, beta=beta, eps=self.eps)
        endpoint_kl = masked_kl(endpoint_star, endpoint_pred, mask=mask, eps=self.eps)
        if mask is None:
            mask_for_metrics = torch.ones_like(p_start, dtype=torch.bool)
        else:
            mask_for_metrics = mask
        top_target = advantage_target.masked_fill(~mask_for_metrics, -torch.inf).argmax(dim=-1)
        top_pred = score_pred.masked_fill(~mask_for_metrics, -torch.inf).argmax(dim=-1)
        metrics = {
            "sphere_path_loss": float(loss.detach().cpu()),
            "endpoint_kl": float(endpoint_kl.detach().cpu()),
            "endpoint_sphere_l2": float(torch.stack(endpoint_l2s).mean().detach().cpu()),
            "path_tv": float(torch.stack(tvs).mean().detach().cpu()),
            "plain_fisher_velocity_diag_mse": float(torch.stack(vmses).mean().detach().cpu()),
            "potential_corr": _masked_rank_corr_like(score_pred, advantage_target, mask_for_metrics),
            "potential_top1_agreement": float((top_target == top_pred).float().mean().detach().cpu()),
        }
        return loss, metrics


@dataclass
class SemanticFisherVelocityLoss:
    """Square-root sphere tangent matching for semantic-Fisher log-rate targets."""

    dt: float = 1.0

    def __call__(
        self,
        p_start: torch.Tensor,
        w_target: torch.Tensor,
        w_pred: torch.Tensor,
        zdot_target: torch.Tensor,
        z_dot_pred: torch.Tensor,
        rewards: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, dict]:
        diff = z_dot_pred - zdot_target
        if mask is not None:
            diff = diff * mask
            denom = mask.sum().clamp(min=1)
            loss = (diff * diff).sum() / denom
        else:
            loss = (diff * diff).mean()

        p_target = semantic_fisher_sphere_step(p_start, w_target, dt=self.dt, eps=eps).detach()
        p_pred = semantic_fisher_sphere_step(p_start, w_pred, dt=self.dt, eps=eps)
        endpoint_kl = masked_kl(p_target, p_pred, mask=mask, eps=eps)
        if mask is None:
            mask_for_metrics = torch.ones_like(p_start, dtype=torch.bool)
        else:
            mask_for_metrics = mask
        top_target = w_target.masked_fill(~mask_for_metrics, -torch.inf).argmax(dim=-1)
        top_pred = w_pred.masked_fill(~mask_for_metrics, -torch.inf).argmax(dim=-1)
        metrics = {
            "semantic_fisher_velocity_loss": float(loss.detach().cpu()),
            "endpoint_kl": float(endpoint_kl.detach().cpu()),
            "lograte_corr": _masked_rank_corr_like(w_pred, w_target, mask_for_metrics),
            "lograte_top1_agreement": float((top_target == top_pred).float().mean().detach().cpu()),
        }
        if rewards is not None:
            reward_ranks = _rank_desc_masked(rewards, mask_for_metrics)
            pred_reward_rank = reward_ranks.gather(1, top_pred.unsqueeze(1)).float()
            metrics["pred_top1_reward_rank_mean"] = float(pred_reward_rank.mean().detach().cpu())
        return loss, metrics


def _rank_desc_masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~mask, -torch.inf)
    order = masked.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order)
    arange = torch.arange(order.shape[-1], device=order.device).expand_as(order)
    ranks.scatter_(1, order, arange + 1)
    return ranks
