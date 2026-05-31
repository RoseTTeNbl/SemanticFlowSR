"""Collate variable-support trace samples into a padded batch with masks."""
from __future__ import annotations
import torch
from ..actions.action_features import ACTION_FEATURE_DIM


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

    def pad1(key):
        out = torch.zeros(Bsz, A)
        for i, s in enumerate(batch):
            n = s[key].numel(); out[i, :n] = s[key]
        return out

    feats = torch.zeros(Bsz, A, ACTION_FEATURE_DIM)
    mask = torch.zeros(Bsz, A, dtype=torch.bool)
    action_ids = torch.zeros(Bsz, A, dtype=torch.long)
    for i, s in enumerate(batch):
        n = s["action_ids"].numel()
        feats[i, :n] = s["action_feats"]
        mask[i, :n] = True
        action_ids[i, :n] = s["action_ids"]

    return {
        "x": x, "y": y, "B": B, "lambda": lam,
        "action_ids": action_ids, "action_feats": feats, "action_mask": mask,
        "energies": pad1("energies"), "weights": pad1("weights"),
        "p0": pad1("p0"), "p1": pad1("p1"),
        "p_lambda": pad1("p_lambda"), "dp_dlambda": pad1("dp_dlambda"),
        "gt_action_pos": torch.stack([s["gt_action_pos"] for s in batch]),
    }
