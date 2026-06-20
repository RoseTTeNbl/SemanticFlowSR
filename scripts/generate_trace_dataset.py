#!/usr/bin/env python
"""Generate local potential natural-flow samples.

Saves a single torch file containing materialized local conditions and closed-form
target records:

    B, y, action_ids, scores, advantages, p_start, p_target,
    lambda, p_lambda, dp_dlambda, gt_action

The default path is the exponential Fisher natural path. Historical p0/p1 aliases
may appear in saved samples for loader compatibility; they equal p_start/p_target.
Large-scale training can stream from the generator instead of materializing this file.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

from semflow_sr.data.synthetic_generator import GenConfig
from semflow_sr.train.build_dataset import build_dataset
from semflow_sr.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_tasks", type=int, default=2000)
    ap.add_argument("--suite", default="mixed")
    ap.add_argument("--num_vars", type=int, default=1)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--probe_size", type=int, default=128)
    ap.add_argument("--target", default="one_step_advantage",
                    choices=["one_step_advantage", "gt", "semantic_oracle", "group_advantage",
                             "semantic_advantage_flow", "rollout_fitness_advantage",
                             "rollout_fitness", "semantic_fisher_risk_flow", "risk_flow"])
    ap.add_argument("--max_support", type=int, default=128)
    ap.add_argument("--support_mode", default="mixed_topk_random",
                    choices=["full", "topk_reward", "mixed_topk_random", "proposal_importance"])
    ap.add_argument("--support_topk", type=int, default=None)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--eta_adv", type=float, default=None,
                    help="legacy alias for --beta used by endpoint compatibility shims")
    ap.add_argument("--no_adv_norm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/local_flow_traces/train")
    a = ap.parse_args()
    set_seed(a.seed)

    gen = GenConfig(num_vars=a.num_vars, max_depth=a.max_depth, K=a.K, probe_size=a.probe_size)
    beta = float(a.beta if a.eta_adv is None else a.eta_adv)
    target_kwargs = None
    if a.target in {"one_step_advantage", "group_advantage", "semantic_advantage_flow"}:
        target_kwargs = {"eta_adv": beta, "normalize": not a.no_adv_norm}
    ds = build_dataset(gen, a.num_tasks, target=a.target, beta=beta,
                       seed=a.seed, max_support=a.max_support,
                       support_mode=a.support_mode, support_topk=a.support_topk,
                       target_kwargs=target_kwargs)
    samples = [ds[i] for i in range(len(ds))]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": samples, "config": vars(a)}, out / "traces.pt")
    print(f"saved {len(samples)} step-samples to {out/'traces.pt'}")


if __name__ == "__main__":
    main()
