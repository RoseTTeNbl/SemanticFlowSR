"""Collate variable-support trace samples into a padded batch with masks."""
from __future__ import annotations
import torch
from ..actions.action_features import ACTION_FEATURE_DIM, SEMANTIC_ACTION_FEATURE_DIM


def collate_velocity(batch: list[dict]) -> dict:
    Bsz = len(batch)
    m = batch[0]["B"].shape[0]
    K = batch[0]["B"].shape[1]
    d = batch[0]["x"].shape[1]
    A = max(s["action_ids"].numel() for s in batch)

    x = torch.stack([s["x"] for s in batch])           # [Bsz,m,d]
    y = torch.stack([s["y"] for s in batch])           # [Bsz,m]
    B = torch.stack([s["B"] for s in batch])           # [Bsz,m,K]
    lam = torch.stack([s["lambda"] for s in batch])    # [Bsz]

    def pad1(key, fallback_key: str | None = None, fill: float = 0.0):
        out = torch.zeros(Bsz, A)
        for i, s in enumerate(batch):
            if key in s:
                values = s[key]
            elif fallback_key is not None and fallback_key in s:
                values = s[fallback_key]
            else:
                values = torch.full_like(s["energies"], float(fill))
            n = values.numel(); out[i, :n] = values
        return out

    feats = torch.zeros(Bsz, A, ACTION_FEATURE_DIM)
    semantic_stats = torch.zeros(Bsz, A, SEMANTIC_ACTION_FEATURE_DIM)
    mask = torch.zeros(Bsz, A, dtype=torch.bool)
    action_ids = torch.zeros(Bsz, A, dtype=torch.long)
    for i, s in enumerate(batch):
        n = s["action_ids"].numel()
        feats[i, :n] = s["action_feats"]
        if "semantic_stats" in s:
            semantic_stats[i, :n] = s["semantic_stats"]
        mask[i, :n] = True
        action_ids[i, :n] = s["action_ids"]

    residual_current = torch.stack([s.get("residual_current", torch.zeros(m)) for s in batch])
    residual_next = torch.zeros(Bsz, A, m)
    xi = torch.zeros(Bsz, A, m)
    gram = torch.zeros(Bsz, A, A)
    for i, s in enumerate(batch):
        n = s["action_ids"].numel()
        if "residual_next" in s:
            residual_next[i, :n] = s["residual_next"]
        if "xi" in s:
            xi[i, :n] = s["xi"]
        if "gram" in s:
            gram[i, :n, :n] = s["gram"]

    return {
        "x": x, "y": y, "B": B, "lambda": lam,
        "action_ids": action_ids, "action_feats": feats, "semantic_stats": semantic_stats,
        "action_mask": mask,
        "energies": pad1("energies"), "rewards": pad1("rewards", fallback_key="energies"),
        "scores": pad1("scores", fallback_key="rewards"),
        "advantages": pad1("advantages"), "target_advantages": pad1("target_advantages", fallback_key="advantages"),
        "proposal_probs": pad1("proposal_probs"),
        "one_step_rewards": pad1("one_step_rewards", fallback_key="rewards"),
        "rollout_rewards": pad1("rollout_rewards", fallback_key="rewards"),
        "rollout_eval_mask": pad1("rollout_eval_mask"),
        "rollout_rank_shift": pad1("rollout_rank_shift"),
        "rollout_n_rollouts": pad1("rollout_n_rollouts"),
        "rollout_best_score": pad1("rollout_best_score"),
        "rollout_score_std": pad1("rollout_score_std"),
        "rollout_best_final_energy": pad1("rollout_best_final_energy"),
        "rollout_best_final_r2": pad1("rollout_best_final_r2"),
        "weights": pad1("weights"),
        "residual_current": residual_current,
        "residual_next": residual_next,
        "xi": xi,
        "gram": gram,
        "gamma": torch.stack([s.get("gamma", torch.tensor(0.0)) for s in batch]),
        "w_target": pad1("w_target"),
        "pdot_target": pad1("pdot_target"),
        "zdot_target": pad1("zdot_target"),
        "p_start": pad1("p_start", fallback_key="p0"),
        "p_target": pad1("p_target", fallback_key="p1"),
        "plain_p_target": pad1("plain_p_target", fallback_key="p_target"),
        "p0": pad1("p0", fallback_key="p_start"),
        "p1": pad1("p1", fallback_key="p_target"),
        "p_lambda": pad1("p_lambda"), "dp_dlambda": pad1("dp_dlambda"),
        "z_lambda": pad1("z_lambda"), "dz_dlambda": pad1("dz_dlambda"),
        "gt_action_pos": torch.stack([s["gt_action_pos"] for s in batch]),
        "full_action_size": torch.stack([s["full_action_size"] for s in batch]),
    }
