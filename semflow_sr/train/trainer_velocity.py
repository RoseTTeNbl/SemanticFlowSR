"""Trainer for local update operators.

``train_velocity`` keeps its historical name for script compatibility. The default
loss is semantic-Fisher sphere-tangent matching between the exact local target and
the model-predicted log-rate. The plain Fisher sphere path remains as an ablation.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader

from .losses import SemanticFisherVelocityLoss, SpherePathLoss
from ..flow.natural_path import natural_endpoint_from_potential
from ..flow.semantic_fisher import semantic_fisher_sphere_step
from ..models.semantic_transformer import SemanticTransformer
from ..utils.logging import get_logger

log = get_logger("train")


@dataclass
class TrainConfig:
    lr: float = 3e-4
    steps: int = 2000
    batch_size: int = 16
    loss_name: str = "semantic_fisher_velocity"
    beta: float = 1.0
    lambda_min: float = 0.05
    num_lambda_samples: int = 1
    grad_clip: float = 1.0
    log_every: int = 50
    sf_step_dt: float = 1.0


def _move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_velocity(model: SemanticTransformer, dataset, cfg: TrainConfig, device,
                   collate_fn, eval_fn=None, eval_every: int = 0) -> dict:
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate_fn, drop_last=False)
    history = []          # 每步 loss
    log_rows = []         # 结构化记录: step, epoch, loss, reward(验证r2)
    epoch = 0
    step = 0
    while step < cfg.steps:
        for batch in loader:
            batch = _move(batch, device)
            output_mode = getattr(model.cfg, "output_mode", "semantic_fisher_lograte")
            if output_mode in {"potential", "semantic_fisher_lograte"}:
                model_p = batch.get("p_lambda", batch["p_start"])
                model_lambda = batch["lambda"]
            else:
                model_p = batch["p_lambda"]
                model_lambda = batch["lambda"]
            out = model(
                x=batch["x"], y=batch["y"], B=batch["B"],
                p_lambda=model_p, lambda_value=model_lambda,
                action_feats=batch["action_feats"], energies=batch["energies"],
                weights=batch["weights"], semantic_stats=batch.get("semantic_stats"),
                gram=batch.get("gram"), beta_value=cfg.beta, action_mask=batch["action_mask"],
            )
            if cfg.loss_name == "semantic_fisher_velocity":
                loss, loss_metrics = SemanticFisherVelocityLoss(dt=cfg.sf_step_dt)(
                    p_start=model_p,
                    w_target=batch["w_target"],
                    w_pred=out.lograte_logits,
                    zdot_target=batch["zdot_target"],
                    z_dot_pred=out.z_dot_pred,
                    rewards=batch.get("rewards"),
                    mask=batch["action_mask"],
                )
            elif cfg.loss_name == "sphere_path":
                loss, loss_metrics = SpherePathLoss(
                    lambda_min=cfg.lambda_min,
                    num_lambda_samples=cfg.num_lambda_samples,
                )(
                    p_start=batch["p_start"],
                    advantage_target=batch["advantages"],
                    score_pred=out.potential_logits,
                    beta=cfg.beta,
                    mask=batch["action_mask"],
                )
            else:
                raise ValueError(f"unknown loss_name: {cfg.loss_name}")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            history.append(loss.item())
            if step % cfg.log_every == 0:
                reward = None
                if eval_fn is not None and eval_every and step > 0 and step % eval_every == 0:
                    model.eval(); reward = eval_fn(model); model.train()
                row = {"step": step, "epoch": epoch, "loss": loss.item(), "reward": reward}
                row.update(_batch_diagnostics(batch))
                row.update(loss_metrics)
                row.update(_prediction_diagnostics(batch, out, beta=cfg.beta))
                log_rows.append(row)
                msg = f"step {step} epoch {epoch} loss {loss.item():.6f}"
                log.info(msg + (f" reward(r2) {reward:.4f}" if reward is not None else ""))
            step += 1
            if step >= cfg.steps:
                break
        epoch += 1
    return {"loss_history": history, "log_rows": log_rows,
            "final_loss": history[-1] if history else None}


def _entropy(p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pp = p.clamp(min=1e-12)
    return -((pp * pp.log()) * mask).sum(dim=-1)


def _valid_values(values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return values[action_mask] if values is not None else torch.empty(0, device=action_mask.device)


def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2 or b.numel() < 2:
        return 0.0
    aa = a.detach().float()
    bb = b.detach().float()
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = aa.std(unbiased=False) * bb.std(unbiased=False)
    if float(denom.detach().cpu()) < 1e-12:
        return 0.0
    return float(((aa * bb).mean() / denom.clamp(min=1e-12)).detach().cpu())


def _masked_top1(values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~action_mask, -torch.inf)
    return masked.argmax(dim=-1)


def _batch_diagnostics(batch: dict) -> dict:
    action_mask = batch["action_mask"]
    mask = action_mask.float()
    rewards = batch.get("rewards")
    advantages = batch.get("advantages")
    p_start = batch.get("p_start", batch.get("p0"))
    p_target = batch.get("p_target", batch.get("p1"))
    out = {
        "support_size_mean": float(mask.sum(dim=-1).float().mean().detach().cpu()),
    }
    if "full_action_size" in batch:
        out["full_action_size_mean"] = float(batch["full_action_size"].float().mean().detach().cpu())
    if rewards is not None:
        valid_rewards = _valid_values(rewards, action_mask)
        out["reward_mean"] = float(valid_rewards.mean().detach().cpu()) if valid_rewards.numel() else 0.0
        out["reward_std"] = float(valid_rewards.std(unbiased=False).detach().cpu()) if valid_rewards.numel() else 0.0
    if advantages is not None:
        valid_adv = _valid_values(advantages, action_mask)
        out["advantage_min"] = float(valid_adv.min().detach().cpu()) if valid_adv.numel() else 0.0
        out["advantage_max"] = float(valid_adv.max().detach().cpu()) if valid_adv.numel() else 0.0
    if p_start is not None and p_target is not None:
        out["p_start_entropy"] = float(_entropy(p_start, mask).mean().detach().cpu())
        out["p_target_entropy"] = float(_entropy(p_target, mask).mean().detach().cpu())
        kl = (p_target.clamp(min=1e-12) * (p_target.clamp(min=1e-12) / p_start.clamp(min=1e-12)).log()) * mask
        out["kl_p_target_p_start"] = float(kl.sum(dim=-1).mean().detach().cpu())
        out["p_target_top1_mass"] = float((p_target * mask).max(dim=-1).values.mean().detach().cpu())
        # Compatibility aliases for older plots.
        out["p0_entropy"] = out["p_start_entropy"]
        out["p1_entropy"] = out["p_target_entropy"]
        out["kl_p1_p0"] = out["kl_p_target_p_start"]
        out["p1_top1_mass"] = out["p_target_top1_mass"]
    one_step_rewards = batch.get("one_step_rewards")
    rollout_rewards = batch.get("rollout_rewards")
    if one_step_rewards is not None and rollout_rewards is not None:
        valid_one = _valid_values(one_step_rewards, action_mask)
        valid_rollout = _valid_values(rollout_rewards, action_mask)
        out["one_step_reward_mean"] = float(valid_one.mean().detach().cpu()) if valid_one.numel() else 0.0
        out["one_step_reward_std"] = float(valid_one.std(unbiased=False).detach().cpu()) if valid_one.numel() else 0.0
        out["rollout_reward_mean"] = float(valid_rollout.mean().detach().cpu()) if valid_rollout.numel() else 0.0
        out["rollout_reward_std"] = float(valid_rollout.std(unbiased=False).detach().cpu()) if valid_rollout.numel() else 0.0
        out["one_step_rollout_corr"] = _safe_corr(valid_one, valid_rollout)
        out["one_step_rollout_top1_agreement"] = float(
            (_masked_top1(one_step_rewards, action_mask) == _masked_top1(rollout_rewards, action_mask))
            .float().mean().detach().cpu()
        )
    rollout_eval_mask = batch.get("rollout_eval_mask")
    if rollout_eval_mask is not None:
        valid_eval = rollout_eval_mask[action_mask].float()
        out["rollout_eval_fraction"] = float(valid_eval.mean().detach().cpu()) if valid_eval.numel() else 0.0
    rollout_rank_shift = batch.get("rollout_rank_shift")
    if rollout_rank_shift is not None:
        valid_shift = _valid_values(rollout_rank_shift, action_mask)
        out["rollout_rank_shift_mean"] = float(valid_shift.mean().detach().cpu()) if valid_shift.numel() else 0.0
        out["rollout_rank_shift_abs_mean"] = float(valid_shift.abs().mean().detach().cpu()) if valid_shift.numel() else 0.0
    rollout_best_score = batch.get("rollout_best_score")
    if rollout_best_score is not None and rollout_eval_mask is not None:
        eval_action_mask = action_mask & rollout_eval_mask.bool()
        valid_best = _valid_values(rollout_best_score, eval_action_mask)
        out["rollout_best_score_max"] = float(valid_best.max().detach().cpu()) if valid_best.numel() else 0.0
        out["rollout_best_score_mean"] = float(valid_best.mean().detach().cpu()) if valid_best.numel() else 0.0
    rollout_best_final_energy = batch.get("rollout_best_final_energy")
    if rollout_best_final_energy is not None and rollout_eval_mask is not None:
        valid_energy = _valid_values(rollout_best_final_energy, action_mask & rollout_eval_mask.bool())
        out["rollout_best_final_energy_min"] = float(valid_energy.min().detach().cpu()) if valid_energy.numel() else 0.0
    rollout_best_final_r2 = batch.get("rollout_best_final_r2")
    if rollout_best_final_r2 is not None and rollout_eval_mask is not None:
        valid_r2 = _valid_values(rollout_best_final_r2, action_mask & rollout_eval_mask.bool())
        out["rollout_best_final_r2_max"] = float(valid_r2.max().detach().cpu()) if valid_r2.numel() else 0.0
        out["rollout_best_final_r2_mean"] = float(valid_r2.mean().detach().cpu()) if valid_r2.numel() else 0.0
    return out


def _rank_desc(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~mask, -torch.inf)
    order = masked.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order)
    ar = torch.arange(order.shape[-1], device=order.device).expand_as(order)
    ranks.scatter_(dim=-1, index=order, src=ar + 1)
    return ranks


def _prediction_diagnostics(batch: dict, model_out, beta: float) -> dict:
    if getattr(model_out, "lograte_logits", None) is not None:
        mask = batch["action_mask"]
        w_pred = model_out.lograte_logits
        w_target = batch["w_target"]
        p_start = batch["p_start"]
        p_current = batch.get("p_lambda", p_start)
        rewards = batch.get("rewards", w_target)
        p_pred = semantic_fisher_sphere_step(p_current, w_pred, dt=1.0)
        p_target = semantic_fisher_sphere_step(p_current, w_target, dt=1.0)
        top_pred = w_pred.masked_fill(~mask, -torch.inf).argmax(dim=-1)
        top_target = w_target.masked_fill(~mask, -torch.inf).argmax(dim=-1)
        reward_ranks = _rank_desc(rewards, mask)
        target_ranks = _rank_desc(w_target, mask)
        pred_reward_rank = reward_ranks.gather(1, top_pred.unsqueeze(1)).float()
        pred_target_rank = target_ranks.gather(1, top_pred.unsqueeze(1)).float()
        kl_pred_start = (p_pred.clamp(min=1e-12) * (
            p_pred.clamp(min=1e-12).log() - p_current.clamp(min=1e-12).log()
        ) * mask).sum(dim=-1)
        l1_pred_start = ((p_pred - p_current).abs() * mask).sum(dim=-1)
        l1_target_start = ((p_target - p_current).abs() * mask).sum(dim=-1)
        return {
            "pred_endpoint_entropy": float(_entropy(p_pred, mask.float()).mean().detach().cpu()),
            "kl_p_pred_p_start": float(kl_pred_start.mean().detach().cpu()),
            "l1_p_pred_p_start": float(l1_pred_start.mean().detach().cpu()),
            "l1_p_target_p_start": float(l1_target_start.mean().detach().cpu()),
            "pred_top1_advantage_agreement": float((top_pred == top_target).float().mean().detach().cpu()),
            "pred_top1_reward_rank_mean": float(pred_reward_rank.mean().detach().cpu()),
            "pred_top1_advantage_rank_mean": float(pred_target_rank.mean().detach().cpu()),
        }
    if getattr(model_out, "potential_logits", None) is None:
        return {}
    mask = batch["action_mask"]
    score = model_out.potential_logits
    adv = batch["advantages"]
    p_start = batch["p_start"]
    rewards = batch.get("rewards", adv)
    p_pred = natural_endpoint_from_potential(p_start, score, beta=beta)
    p_target = natural_endpoint_from_potential(p_start, adv, beta=beta)
    top_pred = score.masked_fill(~mask, -torch.inf).argmax(dim=-1)
    top_adv = adv.masked_fill(~mask, -torch.inf).argmax(dim=-1)
    reward_ranks = _rank_desc(rewards, mask)
    adv_ranks = _rank_desc(adv, mask)
    pred_reward_rank = reward_ranks.gather(1, top_pred.unsqueeze(1)).float()
    pred_adv_rank = adv_ranks.gather(1, top_pred.unsqueeze(1)).float()
    kl_pred_start = (p_pred.clamp(min=1e-12) * (
        p_pred.clamp(min=1e-12).log() - p_start.clamp(min=1e-12).log()
    ) * mask).sum(dim=-1)
    l1_pred_start = ((p_pred - p_start).abs() * mask).sum(dim=-1)
    l1_target_start = ((p_target - p_start).abs() * mask).sum(dim=-1)
    return {
        "pred_endpoint_entropy": float(_entropy(p_pred, mask.float()).mean().detach().cpu()),
        "kl_p_pred_p_start": float(kl_pred_start.mean().detach().cpu()),
        "l1_p_pred_p_start": float(l1_pred_start.mean().detach().cpu()),
        "l1_p_target_p_start": float(l1_target_start.mean().detach().cpu()),
        "pred_top1_advantage_agreement": float((top_pred == top_adv).float().mean().detach().cpu()),
        "pred_top1_reward_rank_mean": float(pred_reward_rank.mean().detach().cpu()),
        "pred_top1_advantage_rank_mean": float(pred_adv_rank.mean().detach().cpu()),
    }
